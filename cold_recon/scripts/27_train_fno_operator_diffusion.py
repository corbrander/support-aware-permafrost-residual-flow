from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.models.diffusion import GaussianDiffusion3D
from cold_recon.models.fno_transformer import FNOTransformerHybrid
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.physics.settlement import settlement_potential_numpy
from cold_recon.training.train_diffusion import _load_autoencoder, _posterior_arrays
from cold_recon.training.volume_codec import sample_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_sections import plot_truth_prediction_sections
from cold_recon.visualization.plot_settlement_risk import plot_settlement_map


def _write_metrics(path: Path, model_name: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"model": model_name, **metrics})


def _parse_modes(text: str) -> tuple[int, int, int]:
    parts = tuple(int(item.strip()) for item in text.split(","))
    if len(parts) != 3:
        raise ValueError("modes must be formatted as mx,my,mz")
    return parts


def _plot_history(history: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(hist["epoch"], hist["loss"], color="#4c78a8")
    ax.set_xlabel("epoch")
    ax.set_ylabel("diffusion noise MSE")
    ax.set_title("FNO operator diffusion training")
    ax.grid(True, color="0.9")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    fno_cfg = config.get("fno_operator_diffusion", {})
    seed = int(config["project"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_name = args.device or fno_cfg.get("device", config["diffusion"].get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)

    sample = load_sample_npz(fno_cfg.get("sample_path", config["training"]["sample_path"]))
    n_facies = int(config["model"]["n_facies"])
    ae = _load_autoencoder(config, dev)
    target = sample_to_volume_tensor(sample, n_facies=n_facies).to(dev)
    with torch.no_grad():
        latent = ae.encode(target)

    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    obs_tokens = tokenizer.encode_torch(sample["observations"], device=dev).unsqueeze(0)
    obs_mask = torch.zeros((1, obs_tokens.shape[1]), dtype=torch.bool, device=dev)
    obs_attention_mask = build_observation_attention_mask(config, sample["grid"], sample["observations"], device=dev)
    obs_hidden = int(config["model"].get("obs_hidden_dim", 96))
    obs_encoder = ObsTransformerEncoder(
        token_dim=int(config["model"]["token_dim"]),
        hidden_dim=obs_hidden,
        num_layers=int(config["model"].get("obs_layers", 2)),
        num_heads=int(config["model"].get("obs_heads", 4)),
    ).to(dev)
    modes = _parse_modes(args.modes or str(fno_cfg.get("modes", "8,8,6")))
    denoiser = FNOTransformerHybrid(
        channels=int(latent.shape[1]),
        cond_dim=obs_hidden,
        width=int(args.width or fno_cfg.get("width", 48)),
        modes=modes,
        depth=int(args.depth or fno_cfg.get("depth", 4)),
        transformer_layers=int(fno_cfg.get("transformer_layers", 1)),
        transformer_heads=int(fno_cfg.get("transformer_heads", 4)),
    ).to(dev)
    diffusion = GaussianDiffusion3D(denoiser, timesteps=int(config["diffusion"].get("timesteps", 80)))
    opt = AdamW(
        list(obs_encoder.parameters()) + list(denoiser.parameters()),
        lr=float(args.lr or fno_cfg.get("lr", 4e-4)),
        weight_decay=float(fno_cfg.get("weight_decay", 1e-6)),
    )
    epochs = int(args.epochs or fno_cfg.get("epochs", 48))
    history: list[dict[str, float]] = []
    obs_encoder.train()
    denoiser.train()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask)
        loss = diffusion.training_loss(latent, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(obs_encoder.parameters()) + list(denoiser.parameters()), float(fno_cfg.get("grad_clip", 2.0)))
        opt.step()
        row = {"epoch": float(epoch), "loss": float(loss.detach().cpu())}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"fno epoch {epoch + 1:04d}/{epochs} loss={row['loss']:.4f}")

    ckpt_path = Path(fno_cfg.get("checkpoint", "outputs/checkpoints/fno_operator_diffusion.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "denoiser_state": denoiser.state_dict(),
            "obs_encoder_state": obs_encoder.state_dict(),
            "config": config,
            "latent_shape": tuple(latent.shape[1:]),
            "n_facies": n_facies,
            "history": history,
            "denoiser": "FNOTransformerHybrid",
            "modes": modes,
        },
        ckpt_path,
    )

    obs_encoder.eval()
    denoiser.eval()
    n_samples = int(args.samples or fno_cfg.get("posterior_samples", 8))
    with torch.no_grad():
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask).repeat(n_samples, 1)
        scale = float(fno_cfg.get("posterior_noise_scale", config["diffusion"].get("posterior_noise_scale", 0.08)))
        correction = float(fno_cfg.get("posterior_correction_scale", config["diffusion"].get("posterior_correction_scale", 0.15)))
        latents = latent.repeat(n_samples, 1, 1, 1, 1) + scale * torch.randn((n_samples, *latent.shape[1:]), device=dev)
        t_mid = max(int(config["diffusion"].get("timesteps", 80)) // 2, 1)
        t = torch.full((n_samples,), t_mid, device=dev, dtype=torch.long)
        latents = latents - correction * denoiser(latents, t, cond)
        decoded = ae.decode(latents)
    posterior = _posterior_arrays(decoded, n_facies)
    pred_path = Path(fno_cfg.get("posterior_path", "outputs/predictions/fno_operator_diffusion_posterior.npz"))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **posterior)

    pred = {
        "facies": posterior["facies_mode"],
        "eic": posterior["eic_mean"],
        "temperature": posterior["temperature_mean"],
        "unfrozen_water": posterior["unfrozen_water_mean"],
        "log_resistivity": posterior["log_resistivity_mean"],
    }
    metrics = synthetic_metrics(
        pred,
        sample["fields"],
        sample["grid"]["z"],
        n_facies=n_facies,
        ice_threshold=float(config["evaluation"]["ice_rich_threshold"]),
    )
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    metrics_path = table_dir / "fno_operator_diffusion_metrics.csv"
    history_path = table_dir / "fno_operator_diffusion_history.csv"
    _write_metrics(metrics_path, "COLDReconFNOOperatorDiffusion", metrics)
    pd.DataFrame(history).to_csv(history_path, index=False)
    plot_truth_prediction_sections(
        sample["fields"],
        pred,
        fig_dir / "fno_operator_diffusion_sections.png",
        int(config["evaluation"]["section_y_index"]),
        "FNO-Transformer operator diffusion posterior",
    )
    settlement = settlement_potential_numpy(posterior["eic_mean"], posterior["temperature_mean"] + 2.0, float(sample["grid"]["dz"]))
    posterior["settlement_potential"] = settlement.astype(np.float32)
    np.savez_compressed(pred_path, **posterior)
    plot_settlement_map(settlement, fig_dir / "fno_operator_diffusion_settlement_potential.png", "FNO operator diffusion settlement potential")
    _plot_history(history, fig_dir / "fno_operator_diffusion_training_history.png")
    print(f"checkpoint={ckpt_path}")
    print(f"posterior={pred_path}")
    print(f"metrics={metrics_path}")
    print(f"history={history_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
