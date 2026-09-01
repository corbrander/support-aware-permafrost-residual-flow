from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cold_recon.data.data_schema import observations_from_npz
from cold_recon.evaluation.cross_validation import split_observations
from cold_recon.evaluation.posterior_assimilation import PosteriorAssimilationConfig, assimilate_posterior_to_observations
from cold_recon.evaluation.real_conditioned_diffusion import (
    blend_posterior_with_proxy,
    evaluate_observation_consistency,
    filter_observations_to_grid,
    resample_reconstruction_fields,
)
from cold_recon.models.denoiser3d_unet import Denoiser3DUNet
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.physics.settlement import settlement_potential_numpy
from cold_recon.training.train_diffusion import _load_autoencoder, _posterior_arrays
from cold_recon.training.volume_codec import fields_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config


def _subsample_observations(observations, max_tokens: int, seed: int):
    if observations.n_obs <= max_tokens:
        return observations
    rng = np.random.default_rng(seed)
    type_ids = np.unique(observations.type_ids)
    selected = []
    base_quota = max(1, max_tokens // max(len(type_ids), 1))
    leftovers = []
    for type_id in type_ids:
        idx = np.where(observations.type_ids == type_id)[0]
        rng.shuffle(idx)
        take = min(len(idx), base_quota)
        selected.extend(idx[:take].tolist())
        leftovers.extend(idx[take:].tolist())
    remaining = max_tokens - len(selected)
    if remaining > 0 and leftovers:
        leftovers_arr = np.array(leftovers, dtype=int)
        rng.shuffle(leftovers_arr)
        selected.extend(leftovers_arr[:remaining].tolist())
    return observations.subset(np.array(selected[:max_tokens], dtype=int))


def _write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def _plot_real_diffusion(posterior: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = posterior["grid_x"]
    z = posterior["grid_z"]
    y_idx = len(posterior["grid_y"]) // 3
    panels = [
        ("facies mode", posterior["facies_mode"][:, y_idx, :].T, "tab20"),
        ("EIC mean", posterior["eic_mean"][:, y_idx, :].T, "viridis"),
        ("EIC std", posterior["eic_std"][:, y_idx, :].T, "magma"),
        ("temperature mean", posterior["temperature_mean"][:, y_idx, :].T, "coolwarm"),
        ("unfrozen water", posterior["unfrozen_water_mean"][:, y_idx, :].T, "Blues"),
        ("ice-rich probability", posterior["ice_rich_probability"][:, y_idx, :].T, "viridis"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="upper", extent=[x.min(), x.max(), z.max(), z.min()], aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("local x (m)")
        ax.set_ylabel("depth (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)

    settlement_path = out_path.with_name("usgs_real_conditioned_diffusion_settlement.png")
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    im = ax.imshow(
        posterior["settlement_potential"].T,
        origin="lower",
        extent=[posterior["grid_x"].min(), posterior["grid_x"].max(), posterior["grid_y"].min(), posterior["grid_y"].max()],
        aspect="auto",
        cmap="inferno",
    )
    ax.set_title("Real-token conditioned diffusion settlement proxy")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("profile y (m)")
    fig.colorbar(im, ax=ax)
    fig.savefig(settlement_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--real-proxy", default="outputs/predictions/usgs_field_reconstruction.npz")
    parser.add_argument("--observations", default="data/processed/usgs_geophysics_observations.npz")
    parser.add_argument("--target-shape", default="64,64,48")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--max-condition-tokens", type=int, default=2048)
    parser.add_argument("--proxy-guidance-weight", type=float, default=0.97)
    parser.add_argument("--disable-assimilation", action="store_true")
    parser.add_argument("--assimilation-horizontal-range", type=float, default=4.0)
    parser.add_argument("--assimilation-vertical-range", type=float, default=0.35)
    parser.add_argument("--assimilation-gain", type=float, default=0.98)
    parser.add_argument("--assimilation-unfrozen-gain", type=float, default=0.0)
    parser.add_argument("--assimilation-alt-gain", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    seed = int(config["project"].get("seed", 42))
    target_shape = tuple(int(x.strip()) for x in args.target_shape.split(","))
    if len(target_shape) != 3:
        raise ValueError("--target-shape must be nx,ny,nz")
    device_name = args.device or config["diffusion"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)

    real_proxy_raw = dict(np.load(args.real_proxy, allow_pickle=False))
    real_proxy = resample_reconstruction_fields(real_proxy_raw, target_shape)
    observations = observations_from_npz(np.load(args.observations, allow_pickle=False))
    train_obs, holdout_obs = split_observations(observations, holdout_fraction=args.holdout_fraction, seed=seed)
    train_obs = filter_observations_to_grid(train_obs, real_proxy)
    holdout_obs = filter_observations_to_grid(holdout_obs, real_proxy)
    cond_obs = _subsample_observations(train_obs, max_tokens=int(args.max_condition_tokens), seed=seed)

    ae = _load_autoencoder(config, dev)
    volume = fields_to_volume_tensor(real_proxy, n_facies=int(config["model"]["n_facies"])).to(dev)
    with torch.no_grad():
        latent = ae.encode(volume)

    diff_ckpt = torch.load(config["diffusion"]["checkpoint"], map_location=dev)
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
        base=int(config["diffusion"].get("denoiser_base_channels", 32)),
    ).to(dev)
    obs_encoder.load_state_dict(diff_ckpt["obs_encoder_state"])
    denoiser.load_state_dict(diff_ckpt["denoiser_state"])
    obs_encoder.eval()
    denoiser.eval()

    obs_grid = {"x": real_proxy["grid_x"], "y": real_proxy["grid_y"], "z": real_proxy["grid_z"]}
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(obs_grid)
    obs_tokens = tokenizer.encode_torch(cond_obs, device=dev).unsqueeze(0)
    obs_mask = torch.zeros((1, obs_tokens.shape[1]), dtype=torch.bool, device=dev)
    obs_attention_mask = build_observation_attention_mask(config, obs_grid, cond_obs, device=dev)
    k = int(args.samples)
    with torch.no_grad():
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask).repeat(k, 1)
        scale = float(config["diffusion"].get("posterior_noise_scale", 0.08))
        correction = float(config["diffusion"].get("posterior_correction_scale", 0.15))
        latents = latent.repeat(k, 1, 1, 1, 1) + scale * torch.randn((k, *latent.shape[1:]), device=dev)
        timesteps = int(config["diffusion"].get("timesteps", 80))
        t = torch.full((k,), max(timesteps // 2, 1), device=dev, dtype=torch.long)
        latents = latents - correction * denoiser(latents, t, cond)
        decoded = ae.decode(latents)
    posterior = _posterior_arrays(decoded, int(config["model"]["n_facies"]))
    posterior["grid_x"] = real_proxy["grid_x"]
    posterior["grid_y"] = real_proxy["grid_y"]
    posterior["grid_z"] = real_proxy["grid_z"]
    posterior = blend_posterior_with_proxy(
        posterior,
        real_proxy,
        weight=float(args.proxy_guidance_weight),
        n_facies=int(config["model"]["n_facies"]),
    )
    assimilation_metrics: dict[str, float] = {}
    if not args.disable_assimilation:
        posterior, assimilation_metrics = assimilate_posterior_to_observations(
            posterior,
            train_obs,
            n_facies=int(config["model"]["n_facies"]),
            cfg=PosteriorAssimilationConfig(
                horizontal_range_m=float(args.assimilation_horizontal_range),
                vertical_range_m=float(args.assimilation_vertical_range),
                continuous_gain=float(args.assimilation_gain),
                unfrozen_gain=float(args.assimilation_unfrozen_gain),
                alt_gain=float(args.assimilation_alt_gain),
                facies_gain=float(args.assimilation_gain),
                seed=seed,
            ),
        )
    posterior["settlement_potential"] = settlement_potential_numpy(
        posterior["eic_mean"],
        posterior["temperature_mean"] + 2.0,
        float(np.mean(np.diff(real_proxy["grid_z"]))),
    ).astype(np.float32)

    pred_path = Path(config["paths"]["predictions_dir"]) / "usgs_real_conditioned_diffusion.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **posterior)
    metrics = evaluate_observation_consistency(posterior, holdout_obs)
    metrics["train_n"] = float(train_obs.n_obs)
    metrics["condition_n"] = float(cond_obs.n_obs)
    metrics["holdout_n"] = float(holdout_obs.n_obs)
    for key, value in assimilation_metrics.items():
        metrics[f"assimilation_{key}"] = value
    metrics_path = Path(config["paths"]["tables_dir"]) / "usgs_real_conditioned_diffusion_metrics.csv"
    _write_metrics(metrics_path, metrics)
    fig_path = Path(config["paths"]["figures_dir"]) / "usgs_real_conditioned_diffusion_sections.png"
    _plot_real_diffusion(posterior, fig_path)
    print(f"prediction={pred_path}")
    print(f"metrics={metrics_path}")
    print(f"figure={fig_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
