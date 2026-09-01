from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cold_recon.data.data_schema import observations_from_npz
from cold_recon.evaluation.eic_conditioned_diffusion import (
    EICProxyConfig,
    apply_calibrated_high_eic_screening,
    blend_eic_posterior_with_proxy,
    eic_conditioning_baseline_rows,
    eic_posterior_prediction_rows,
    evaluate_eic_observation_consistency,
    make_eic_proxy_fields,
    project_eic_conditioned_physics,
    split_eic_observations_by_borehole,
)
from cold_recon.evaluation.posterior_assimilation import PosteriorAssimilationConfig, assimilate_posterior_to_observations
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
    idx = np.arange(observations.n_obs)
    rng.shuffle(idx)
    return observations.subset(np.sort(idx[:max_tokens]))


def _write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def _plot_eic_diffusion(
    posterior: dict[str, np.ndarray],
    train_obs,
    holdout_obs,
    out_path: Path,
    output_prefix: str,
    source_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = posterior["grid_x"]
    z = posterior["grid_z"]
    y_idx = int(np.abs(posterior["grid_y"]).argmin())
    panels = [
        ("EIC mean", posterior["eic_mean"][:, y_idx, :].T, "viridis"),
        ("EIC std", posterior["eic_std"][:, y_idx, :].T, "magma"),
        ("ice-rich probability", posterior["ice_rich_probability"][:, y_idx, :].T, "viridis"),
        ("facies mode", posterior["facies_mode"][:, y_idx, :].T, "tab20"),
        ("temperature mean", posterior["temperature_mean"][:, y_idx, :].T, "coolwarm"),
        ("log resistivity proxy", posterior["log_resistivity_mean"][:, y_idx, :].T, "magma"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="upper", extent=[x.min(), x.max(), z.max(), z.min()], aspect="auto", cmap=cmap)
        if title == "EIC mean":
            ax.scatter(train_obs.coords[:, 0], train_obs.coords[:, 2], c=train_obs.values, s=15, cmap="viridis", edgecolors="white", linewidths=0.2)
            ax.scatter(holdout_obs.coords[:, 0], holdout_obs.coords[:, 2], c=holdout_obs.values, s=28, marker="x", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("ordered borehole x (m)")
        ax.set_ylabel("depth (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)

    proxy_path = out_path.with_name(f"{output_prefix}_ground_ice_proxy.png")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    im = ax.imshow(
        posterior["settlement_potential"].T,
        origin="lower",
        extent=[posterior["grid_x"].min(), posterior["grid_x"].max(), posterior["grid_y"].min(), posterior["grid_y"].max()],
        aspect="auto",
        cmap="inferno",
    )
    ax.set_title(f"{source_label} conditioned ground-ice proxy")
    ax.set_xlabel("ordered borehole x (m)")
    ax.set_ylabel("transverse proxy y (m)")
    fig.colorbar(im, ax=ax, label="thaw-sensitive EIC proxy")
    fig.savefig(proxy_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--observations", default="data/processed/usgs_eic_observations.npz")
    parser.add_argument("--output-prefix", default="usgs_eic_conditioned_diffusion")
    parser.add_argument("--model-name", default="COLDReconUSGSEICConditionedDiffusion")
    parser.add_argument("--source-label", default="USGS EIC")
    parser.add_argument("--target-shape", default="64,64,48")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--max-condition-tokens", type=int, default=512)
    parser.add_argument("--eic-guidance-weight", type=float, default=0.85)
    parser.add_argument("--physics-guidance-weight", type=float, default=1.00)
    parser.add_argument("--facies-guidance-weight", type=float, default=0.45)
    parser.add_argument("--physics-projection-iterations", type=int, default=32)
    parser.add_argument("--physics-projection-strength", type=float, default=0.45)
    parser.add_argument("--physics-projection-anchor", type=float, default=0.0)
    parser.add_argument("--disable-assimilation", action="store_true")
    parser.add_argument("--assimilation-horizontal-range", type=float, default=3.5)
    parser.add_argument("--assimilation-vertical-range", type=float, default=0.25)
    parser.add_argument("--assimilation-gain", type=float, default=0.98)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    target_shape = tuple(int(x.strip()) for x in args.target_shape.split(","))
    if len(target_shape) != 3:
        raise ValueError("--target-shape must be nx,ny,nz")
    seed = int(config["project"].get("seed", 42))
    device_name = args.device or config["diffusion"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)

    data = np.load(args.observations, allow_pickle=False)
    observations = observations_from_npz(data)
    if "borehole_ids" not in data.files:
        raise KeyError("EIC observations file must include borehole_ids")
    train_obs, holdout_obs, train_boreholes, holdout_boreholes = split_eic_observations_by_borehole(
        observations,
        data["borehole_ids"],
        holdout_fraction=float(args.holdout_fraction),
        seed=seed,
    )
    cond_obs = _subsample_observations(train_obs, max_tokens=int(args.max_condition_tokens), seed=seed)
    proxy = make_eic_proxy_fields(
        cond_obs,
        reference_observations=observations,
        config=EICProxyConfig(target_shape=target_shape),
    )

    ae = _load_autoencoder(config, dev)
    volume = fields_to_volume_tensor(proxy, n_facies=int(config["model"]["n_facies"])).to(dev)
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

    obs_grid = {"x": proxy["grid_x"], "y": proxy["grid_y"], "z": proxy["grid_z"]}
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(obs_grid)
    obs_tokens = tokenizer.encode_torch(cond_obs, device=dev).unsqueeze(0)
    obs_mask = torch.zeros((1, obs_tokens.shape[1]), dtype=torch.bool, device=dev)
    obs_attention_mask = build_observation_attention_mask(config, obs_grid, cond_obs, device=dev)
    n_samples = int(args.samples)
    with torch.no_grad():
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask).repeat(n_samples, 1)
        scale = float(config["diffusion"].get("posterior_noise_scale", 0.08))
        correction = float(config["diffusion"].get("posterior_correction_scale", 0.15))
        latents = latent.repeat(n_samples, 1, 1, 1, 1) + scale * torch.randn((n_samples, *latent.shape[1:]), device=dev)
        timesteps = int(config["diffusion"].get("timesteps", 80))
        t = torch.full((n_samples,), max(timesteps // 2, 1), device=dev, dtype=torch.long)
        latents = latents - correction * denoiser(latents, t, cond)
        decoded = ae.decode(latents)
    posterior = _posterior_arrays(decoded, int(config["model"]["n_facies"]))
    posterior["grid_x"] = proxy["grid_x"]
    posterior["grid_y"] = proxy["grid_y"]
    posterior["grid_z"] = proxy["grid_z"]
    posterior = blend_eic_posterior_with_proxy(
        posterior,
        proxy,
        eic_weight=float(args.eic_guidance_weight),
        physics_weight=float(args.physics_guidance_weight),
        facies_weight=float(args.facies_guidance_weight),
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
                eic_gain=float(args.assimilation_gain),
                facies_gain=float(args.assimilation_gain),
                seed=seed,
            ),
        )
    posterior = project_eic_conditioned_physics(
        posterior,
        n_facies=int(config["model"]["n_facies"]),
        heat_iterations=int(args.physics_projection_iterations),
        heat_strength=float(args.physics_projection_strength),
        heat_anchor=float(args.physics_projection_anchor),
    )
    posterior["settlement_potential"] = settlement_potential_numpy(
        posterior["eic_mean"],
        posterior["temperature_mean"] + 2.0,
        float(np.mean(np.diff(posterior["grid_z"]))),
    ).astype(np.float32)

    output_prefix = str(args.output_prefix)
    pred_path = Path(config["paths"]["predictions_dir"]) / f"{output_prefix}.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **posterior)

    metrics: dict[str, float] = {
        "train_n": float(train_obs.n_obs),
        "condition_n": float(cond_obs.n_obs),
        "holdout_n": float(holdout_obs.n_obs),
        "train_boreholes": float(len(train_boreholes)),
        "holdout_boreholes": float(len(holdout_boreholes)),
    }
    metrics.update(evaluate_eic_observation_consistency(posterior, train_obs, prefix="train_eic"))
    metrics.update(evaluate_eic_observation_consistency(posterior, holdout_obs, prefix="holdout_eic"))
    for key, value in assimilation_metrics.items():
        metrics[f"assimilation_{key}"] = value
    metrics_path = Path(config["paths"]["tables_dir"]) / f"{output_prefix}_metrics.csv"
    _write_metrics(metrics_path, metrics)
    comparison_cfg = EICProxyConfig(target_shape=target_shape)
    baseline_metrics, baseline_predictions = eic_conditioning_baseline_rows(train_obs, holdout_obs, comparison_cfg)
    diffusion_metrics, diffusion_predictions = eic_posterior_prediction_rows(
        posterior,
        holdout_obs,
        model_name=str(args.model_name),
        config=comparison_cfg,
    )
    _, train_diffusion_predictions = eic_posterior_prediction_rows(
        posterior,
        train_obs,
        model_name=str(args.model_name),
        config=comparison_cfg,
    )
    diffusion_metrics, diffusion_predictions = apply_calibrated_high_eic_screening(
        train_diffusion_predictions,
        diffusion_metrics,
        diffusion_predictions,
        model_name=str(args.model_name),
        observed_threshold=float(comparison_cfg.high_eic_threshold),
        beta=2.0,
    )
    comparison = baseline_metrics if diffusion_metrics.empty else pd.concat([baseline_metrics, diffusion_metrics], ignore_index=True)
    if not comparison.empty and "eic_rmse" in comparison:
        global_rows = comparison[comparison["model"] == "GlobalMean"]
        simple_rows = comparison[comparison["model"].isin(["GlobalMean", "DepthIDW", "SpatialDepthIDW"])]
        global_rmse = float(global_rows["eic_rmse"].iloc[0]) if not global_rows.empty else np.nan
        best_simple_rmse = float(simple_rows["eic_rmse"].min()) if not simple_rows.empty else np.nan
        comparison["eic_rmse_reduction_vs_global_mean"] = (
            1.0 - comparison["eic_rmse"].astype(float) / global_rmse if np.isfinite(global_rmse) and global_rmse > 0.0 else np.nan
        )
        comparison["eic_rmse_reduction_vs_best_simple"] = (
            1.0 - comparison["eic_rmse"].astype(float) / best_simple_rmse if np.isfinite(best_simple_rmse) and best_simple_rmse > 0.0 else np.nan
        )
    comparison_path = Path(config["paths"]["tables_dir"]) / f"{output_prefix}_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    predictions = baseline_predictions if diffusion_predictions.empty else pd.concat([baseline_predictions, diffusion_predictions], ignore_index=True)
    predictions_path = Path(config["paths"]["tables_dir"]) / f"{output_prefix}_holdout_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    fig_path = Path(config["paths"]["figures_dir"]) / f"{output_prefix}_sections.png"
    _plot_eic_diffusion(posterior, train_obs, holdout_obs, fig_path, output_prefix=output_prefix, source_label=str(args.source_label))
    print(f"prediction={pred_path}")
    print(f"metrics={metrics_path}")
    print(f"comparison={comparison_path}")
    print(f"predictions={predictions_path}")
    print(f"figure={fig_path}")
    print(f"holdout_boreholes={','.join(holdout_boreholes.tolist())}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
