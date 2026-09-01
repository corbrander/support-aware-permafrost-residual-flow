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
from cold_recon.models.denoiser3d_unet import Denoiser3DUNet
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.models.rectified_flow import RectifiedFlow3D
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


def _plot_history(history: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(hist["epoch"], hist["loss"], color="#4c78a8")
    ax.set_xlabel("epoch")
    ax.set_ylabel("flow velocity MSE")
    ax.set_title("Rectified flow training")
    ax.grid(True, color="0.9")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _source_latents(target_latent: torch.Tensor, batch_size: int, noise_scale: float) -> torch.Tensor:
    return target_latent.repeat(batch_size, 1, 1, 1, 1) + float(noise_scale) * torch.randn(
        (batch_size, *target_latent.shape[1:]),
        device=target_latent.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    flow_cfg = config.get("rectified_flow", {})
    seed = int(config["project"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_name = args.device or flow_cfg.get("device", config["diffusion"].get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)

    sample = load_sample_npz(flow_cfg.get("sample_path", config["training"]["sample_path"]))
    n_facies = int(config["model"]["n_facies"])
    ae = _load_autoencoder(config, dev)
    target = sample_to_volume_tensor(sample, n_facies=n_facies).to(dev)
    with torch.no_grad():
        target_latent = ae.encode(target)

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
    velocity = Denoiser3DUNet(
        channels=int(target_latent.shape[1]),
        cond_dim=obs_hidden,
        base=int(flow_cfg.get("velocity_base_channels", config["diffusion"].get("denoiser_base_channels", 32))),
    ).to(dev)
    flow = RectifiedFlow3D(velocity, time_scale=int(flow_cfg.get("time_scale", 1000)))
    opt = AdamW(
        list(obs_encoder.parameters()) + list(velocity.parameters()),
        lr=float(args.lr or flow_cfg.get("lr", 4e-4)),
        weight_decay=float(flow_cfg.get("weight_decay", 1e-6)),
    )
    epochs = int(args.epochs or flow_cfg.get("epochs", 64))
    batch_size = int(args.batch_size or flow_cfg.get("batch_size", 4))
    noise_scale = float(args.noise_scale or flow_cfg.get("source_noise_scale", 0.18))
    history: list[dict[str, float]] = []
    obs_encoder.train()
    velocity.train()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask).repeat(batch_size, 1)
        x0 = _source_latents(target_latent, batch_size=batch_size, noise_scale=noise_scale)
        x1 = target_latent.repeat(batch_size, 1, 1, 1, 1)
        loss = flow.training_loss(x0, x1, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(obs_encoder.parameters()) + list(velocity.parameters()), float(flow_cfg.get("grad_clip", 2.0)))
        opt.step()
        row = {"epoch": float(epoch), "loss": float(loss.detach().cpu())}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"flow epoch {epoch + 1:04d}/{epochs} loss={row['loss']:.4f}")

    ckpt_path = Path(flow_cfg.get("checkpoint", "outputs/checkpoints/rectified_flow.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "velocity_state": velocity.state_dict(),
            "obs_encoder_state": obs_encoder.state_dict(),
            "config": config,
            "latent_shape": tuple(target_latent.shape[1:]),
            "n_facies": n_facies,
            "history": history,
            "source_noise_scale": noise_scale,
        },
        ckpt_path,
    )

    obs_encoder.eval()
    velocity.eval()
    n_samples = int(args.samples or flow_cfg.get("posterior_samples", 8))
    steps = int(args.steps or flow_cfg.get("sampling_steps", 16))
    with torch.no_grad():
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask).repeat(n_samples, 1)
        x0 = _source_latents(target_latent, batch_size=n_samples, noise_scale=noise_scale)
        latents = flow.sample(x0, cond, steps=steps)
        decoded = ae.decode(latents)
    posterior = _posterior_arrays(decoded, n_facies)
    pred_path = Path(flow_cfg.get("posterior_path", "outputs/predictions/rectified_flow_posterior.npz"))
    pred_path.parent.mkdir(parents=True, exist_ok=True)

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
    settlement = settlement_potential_numpy(posterior["eic_mean"], posterior["temperature_mean"] + 2.0, float(sample["grid"]["dz"]))
    posterior["settlement_potential"] = settlement.astype(np.float32)
    np.savez_compressed(pred_path, **posterior)

    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    metrics_path = table_dir / "rectified_flow_metrics.csv"
    history_path = table_dir / "rectified_flow_history.csv"
    _write_metrics(metrics_path, "COLDReconRectifiedFlow", metrics)
    pd.DataFrame(history).to_csv(history_path, index=False)
    plot_truth_prediction_sections(
        sample["fields"],
        pred,
        fig_dir / "rectified_flow_sections.png",
        int(config["evaluation"]["section_y_index"]),
        "Warm-start rectified flow posterior",
    )
    plot_settlement_map(settlement, fig_dir / "rectified_flow_settlement_potential.png", "Rectified flow settlement potential")
    _plot_history(history, fig_dir / "rectified_flow_training_history.png")
    print(f"checkpoint={ckpt_path}")
    print(f"posterior={pred_path}")
    print(f"metrics={metrics_path}")
    print(f"history={history_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
