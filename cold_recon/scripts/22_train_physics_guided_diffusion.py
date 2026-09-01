from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.evaluation.physics_consistency import fields_from_prediction, physics_consistency_metrics
from cold_recon.models.denoiser3d_unet import Denoiser3DUNet
from cold_recon.models.diffusion import GaussianDiffusion3D
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.physics.settlement import settlement_potential_numpy
from cold_recon.training.physics_guided_training import PhysicsGuidedTrainingConfig, physics_guided_diffusion_loss
from cold_recon.training.train_diffusion import _load_autoencoder, _posterior_arrays
from cold_recon.training.volume_codec import sample_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_sections import plot_truth_prediction_sections
from cold_recon.visualization.plot_settlement_risk import plot_settlement_map


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_metric_row(path: Path, model_name: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"model": model_name, **metrics})


def _training_config(config: dict, args: argparse.Namespace) -> PhysicsGuidedTrainingConfig:
    defaults = config.get("physics_training", {})
    return PhysicsGuidedTrainingConfig(
        epochs=int(args.epochs if args.epochs is not None else defaults.get("epochs", 16)),
        lr=float(args.lr if args.lr is not None else defaults.get("lr", 8e-5)),
        noise_weight=float(defaults.get("noise_weight", 1.0)),
        physics_weight=float(defaults.get("physics_weight", 0.08)),
        latent_anchor_weight=float(defaults.get("latent_anchor_weight", 0.05)),
        facies_anchor_weight=float(defaults.get("facies_anchor_weight", 0.20)),
        continuous_anchor_weight=float(defaults.get("continuous_anchor_weight", 0.05)),
        grad_clip=float(defaults.get("grad_clip", 1.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="COLDReconLatentDiffusionPhysicsTrained")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    cfg = _training_config(config, args)
    sample = load_sample_npz(config["training"]["sample_path"])
    n_facies = int(config["model"]["n_facies"])
    diffusion_cfg = config["diffusion"]
    training_cfg = config.get("physics_training", {})
    device_name = args.device or diffusion_cfg.get("device", config["training"].get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)
    torch.manual_seed(int(config["project"].get("seed", 42)))

    ae = _load_autoencoder(config, dev)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    target_volume = sample_to_volume_tensor(sample, n_facies=n_facies).to(dev)
    with torch.no_grad():
        latent = ae.encode(target_volume)

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
    denoiser = Denoiser3DUNet(
        channels=int(latent.shape[1]),
        cond_dim=obs_hidden,
        base=int(diffusion_cfg.get("denoiser_base_channels", 32)),
    ).to(dev)
    ckpt = torch.load(diffusion_cfg["checkpoint"], map_location=dev)
    obs_encoder.load_state_dict(ckpt["obs_encoder_state"])
    denoiser.load_state_dict(ckpt["denoiser_state"])
    for parameter in obs_encoder.parameters():
        parameter.requires_grad_(False)
    obs_encoder.eval()
    denoiser.train()
    diffusion = GaussianDiffusion3D(denoiser, timesteps=int(diffusion_cfg.get("timesteps", 80)))
    opt = AdamW(denoiser.parameters(), lr=cfg.lr, weight_decay=float(training_cfg.get("weight_decay", 1e-6)))
    spacing = tuple(float(x) for x in sample["grid"].get("spacing", (sample["grid"]["dx"], sample["grid"]["dy"], sample["grid"]["dz"])))
    history: list[dict[str, float]] = []
    with torch.no_grad():
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask)
    for epoch in range(cfg.epochs):
        opt.zero_grad(set_to_none=True)
        loss, parts = physics_guided_diffusion_loss(
            latent,
            cond,
            diffusion,
            ae,
            target_volume,
            n_facies=n_facies,
            spacing=spacing,
            cfg=cfg,
        )
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), cfg.grad_clip)
        opt.step()
        row = {"epoch": float(epoch), **{key: float(value.detach().cpu()) for key, value in parts.items()}}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 4) == 0:
            print(
                f"phys-train epoch {epoch + 1:04d}/{cfg.epochs} "
                f"loss={row['loss']:.4f} noise={row['noise']:.4f} physics={row['physics']:.4f}"
            )

    ckpt_path = Path(training_cfg.get("checkpoint", "outputs/checkpoints/latent_diffusion_physics_trained.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "denoiser_state": denoiser.state_dict(),
            "obs_encoder_state": obs_encoder.state_dict(),
            "config": config,
            "physics_training_config": cfg,
            "latent_shape": tuple(latent.shape[1:]),
            "n_facies": n_facies,
            "history": history,
        },
        ckpt_path,
    )

    denoiser.eval()
    k = int(args.samples or training_cfg.get("posterior_samples", diffusion_cfg.get("posterior_samples", 8)))
    with torch.no_grad():
        cond_samples = cond.repeat(k, 1)
        scale = float(diffusion_cfg.get("posterior_noise_scale", 0.08))
        correction = float(diffusion_cfg.get("posterior_correction_scale", 0.15))
        timesteps = int(diffusion_cfg.get("timesteps", 80))
        latents = latent.repeat(k, 1, 1, 1, 1) + scale * torch.randn((k, *latent.shape[1:]), device=dev)
        t_mid = max(timesteps // 2, 1)
        t = torch.full((k,), t_mid, device=dev, dtype=torch.long)
        latents = latents - correction * denoiser(latents, t, cond_samples)
        decoded = ae.decode(latents)
    posterior = _posterior_arrays(decoded, n_facies=n_facies)
    posterior["settlement_potential"] = settlement_potential_numpy(
        posterior["eic_mean"],
        posterior["temperature_mean"] + 2.0,
        float(sample["grid"]["dz"]),
    )
    posterior["physics_training_epochs"] = np.asarray(cfg.epochs, dtype=np.int32)
    posterior["physics_training_lr"] = np.asarray(cfg.lr, dtype=np.float32)
    posterior_path = Path(training_cfg.get("posterior_path", "outputs/predictions/diffusion_posterior_physics_trained.npz"))
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(posterior_path, **posterior)

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
    metrics_path = table_dir / "diffusion_physics_trained_metrics.csv"
    history_path = table_dir / "diffusion_physics_trained_history.csv"
    _write_metric_row(metrics_path, args.model_name, metrics)
    _write_rows(history_path, history)
    fig_dir = Path(config["paths"]["figures_dir"])
    plot_truth_prediction_sections(
        sample["fields"],
        pred,
        fig_dir / "diffusion_physics_trained_sections.png",
        int(config["evaluation"]["section_y_index"]),
        "Physics-trained latent diffusion posterior",
    )
    plot_settlement_map(
        posterior["settlement_potential"],
        fig_dir / "diffusion_physics_trained_settlement_potential.png",
        "Physics-trained diffusion settlement potential",
    )
    physics_metrics = physics_consistency_metrics(fields_from_prediction(posterior, n_facies=n_facies), spacing=spacing)
    print(f"checkpoint={ckpt_path}")
    print(f"posterior={posterior_path}")
    print(f"metrics={metrics_path}")
    print(f"history={history_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(
        "physics: "
        f"uw_mae={physics_metrics['unfrozen_water_empirical_mae']:.6f}, "
        f"rho_mae={physics_metrics['log_resistivity_empirical_mae']:.6f}, "
        f"heat_rmse={physics_metrics['heat_residual_rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
