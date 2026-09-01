from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.evaluation.physics_consistency import fields_from_prediction, physics_consistency_metrics
from cold_recon.models.denoiser3d_unet import Denoiser3DUNet
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.physics.settlement import settlement_potential_numpy
from cold_recon.training.physics_guided_sampling import LatentPhysicsGuidanceConfig, guide_latents
from cold_recon.training.train_diffusion import _load_autoencoder, _posterior_arrays
from cold_recon.training.volume_codec import sample_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_sections import plot_truth_prediction_sections
from cold_recon.visualization.plot_settlement_risk import plot_settlement_map


def _guidance_config(config: dict, args: argparse.Namespace) -> LatentPhysicsGuidanceConfig:
    defaults = config.get("physics_guidance", {})
    return LatentPhysicsGuidanceConfig(
        steps=int(args.steps if args.steps is not None else defaults.get("steps", 6)),
        learning_rate=float(args.learning_rate if args.learning_rate is not None else defaults.get("learning_rate", 0.015)),
        unfrozen_weight=float(defaults.get("unfrozen_weight", 0.50)),
        resistivity_weight=float(defaults.get("resistivity_weight", 0.10)),
        heat_weight=float(defaults.get("heat_weight", 0.0006)),
        range_weight=float(defaults.get("range_weight", 0.02)),
        anchor_weight=float(defaults.get("anchor_weight", 0.10)),
        facies_anchor_weight=float(defaults.get("facies_anchor_weight", 1.50)),
        eic_anchor_weight=float(defaults.get("eic_anchor_weight", 0.05)),
        grad_clip=float(defaults.get("grad_clip", 1.0)),
        temperature_min=float(defaults.get("temperature_min", -10.0)),
        temperature_max=float(defaults.get("temperature_max", 3.0)),
    )


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-name", default="COLDReconLatentDiffusionPhysicsGuided")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(config["training"]["sample_path"])
    n_facies = int(config["model"]["n_facies"])
    diffusion_cfg = config["diffusion"]
    guidance_cfg = _guidance_config(config, args)
    device_name = args.device or diffusion_cfg.get("device", config["training"].get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)
    torch.manual_seed(int(config["project"].get("seed", 42)))

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
    denoiser = Denoiser3DUNet(
        channels=int(latent.shape[1]),
        cond_dim=obs_hidden,
        base=int(diffusion_cfg.get("denoiser_base_channels", 32)),
    ).to(dev)
    ckpt = torch.load(diffusion_cfg["checkpoint"], map_location=dev)
    obs_encoder.load_state_dict(ckpt["obs_encoder_state"])
    denoiser.load_state_dict(ckpt["denoiser_state"])
    obs_encoder.eval()
    denoiser.eval()
    for module in (obs_encoder, denoiser, ae):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    k = int(args.samples or config.get("physics_guidance", {}).get("samples", diffusion_cfg.get("posterior_samples", 8)))
    with torch.no_grad():
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask).repeat(k, 1)
        scale = float(diffusion_cfg.get("posterior_noise_scale", 0.08))
        correction = float(diffusion_cfg.get("posterior_correction_scale", 0.15))
        timesteps = int(diffusion_cfg.get("timesteps", 80))
        latents = latent.repeat(k, 1, 1, 1, 1) + scale * torch.randn((k, *latent.shape[1:]), device=dev)
        t_mid = max(timesteps // 2, 1)
        t = torch.full((k,), t_mid, device=dev, dtype=torch.long)
        latents = latents - correction * denoiser(latents, t, cond)

    spacing = tuple(float(x) for x in sample["grid"].get("spacing", (sample["grid"]["dx"], sample["grid"]["dy"], sample["grid"]["dz"])))
    guided_latents, history = guide_latents(latents, ae, n_facies=n_facies, spacing=spacing, cfg=guidance_cfg)
    with torch.no_grad():
        decoded = ae.decode(guided_latents)
    posterior = _posterior_arrays(decoded, n_facies=n_facies)
    posterior["settlement_potential"] = settlement_potential_numpy(
        posterior["eic_mean"],
        posterior["temperature_mean"] + 2.0,
        float(sample["grid"]["dz"]),
    )
    posterior["physics_guidance_steps"] = np.asarray(guidance_cfg.steps, dtype=np.int32)
    posterior["physics_guidance_learning_rate"] = np.asarray(guidance_cfg.learning_rate, dtype=np.float32)
    out_path = Path(args.output or config.get("physics_guidance", {}).get("posterior_path", "outputs/predictions/diffusion_posterior_physics_guided.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **posterior)

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
    metrics_path = table_dir / "diffusion_physics_guided_metrics.csv"
    _write_metric_row(metrics_path, args.model_name, metrics)
    history_path = table_dir / "diffusion_physics_guided_history.csv"
    _write_rows(history_path, history)
    fig_dir = Path(config["paths"]["figures_dir"])
    plot_truth_prediction_sections(
        sample["fields"],
        pred,
        fig_dir / "diffusion_physics_guided_sections.png",
        int(config["evaluation"]["section_y_index"]),
        "Physics-guided latent diffusion posterior",
    )
    plot_settlement_map(
        posterior["settlement_potential"],
        fig_dir / "diffusion_physics_guided_settlement_potential.png",
        "Physics-guided diffusion settlement potential",
    )
    physics_metrics = physics_consistency_metrics(fields_from_prediction(posterior, n_facies=n_facies), spacing=spacing)
    print(f"posterior={out_path}")
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
