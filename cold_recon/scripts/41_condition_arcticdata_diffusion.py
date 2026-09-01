from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cold_recon.data.data_schema import observations_from_npz
from cold_recon.evaluation.arcticdata_conditioned_diffusion import (
    apply_cryofacies_eic_prior,
    arcticdata_conditioning_baseline_rows,
    choose_conditioning_site,
    eic_event_calibrated_posterior,
    evaluate_adaptive_hybrid_posterior,
    facies_hybrid_calibrated_posterior,
    evaluate_conditioned_posterior,
    evaluate_hybrid_calibrated_posterior,
    evaluate_wedge_recall_posterior,
    split_observations_by_site_borehole,
    subsample_observations,
)
from cold_recon.evaluation.eic_conditioned_diffusion import (
    EICProxyConfig,
    blend_eic_posterior_with_proxy,
    make_eic_proxy_fields,
    project_eic_conditioned_physics,
)
from cold_recon.evaluation.posterior_assimilation import PosteriorAssimilationConfig, assimilate_posterior_to_observations
from cold_recon.models.denoiser3d_unet import Denoiser3DUNet
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.physics.settlement import settlement_potential_numpy
from cold_recon.training.train_diffusion import _load_autoencoder, _posterior_arrays
from cold_recon.training.volume_codec import fields_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config


def _load_split(args, config: dict):
    data = np.load(args.observations, allow_pickle=False)
    observations = observations_from_npz(data)
    required = {"site_ids", "borehole_ids"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"ArcticData observations file is missing metadata arrays: {sorted(missing)}")
    token_index = pd.read_csv(args.token_index)
    site = args.site or choose_conditioning_site(
        token_index,
        max_span_m=float(args.max_site_span_m),
        min_eic_tokens=int(args.min_eic_tokens),
        min_boreholes=int(args.min_boreholes),
    )
    split = split_observations_by_site_borehole(
        observations,
        data["site_ids"],
        data["borehole_ids"],
        site=site,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(config["project"].get("seed", 42)),
    )
    return split, token_index[token_index["site"].astype(str) == site].copy()


def _load_conditioned_models(config: dict, latent: torch.Tensor, device: torch.device):
    diff_ckpt = torch.load(config["diffusion"]["checkpoint"], map_location=device)
    obs_hidden = int(config["model"].get("obs_hidden_dim", 96))
    obs_encoder = ObsTransformerEncoder(
        token_dim=int(config["model"]["token_dim"]),
        hidden_dim=obs_hidden,
        num_layers=int(config["model"].get("obs_layers", 2)),
        num_heads=int(config["model"].get("obs_heads", 4)),
    ).to(device)
    denoiser = Denoiser3DUNet(
        channels=int(latent.shape[1]),
        cond_dim=obs_hidden,
        base=int(config["diffusion"].get("denoiser_base_channels", 32)),
    ).to(device)
    obs_encoder.load_state_dict(diff_ckpt["obs_encoder_state"])
    denoiser.load_state_dict(diff_ckpt["denoiser_state"])
    obs_encoder.eval()
    denoiser.eval()
    return obs_encoder, denoiser


def _plot_conditioned_posterior(posterior: dict[str, np.ndarray], split, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_idx = int(np.abs(posterior["grid_y"]).argmin())
    x = posterior["grid_x"]
    z = posterior["grid_z"]
    panels = [
        ("EIC mean", posterior["eic_mean"][:, y_idx, :].T, "viridis"),
        ("ice-rich probability", posterior["ice_rich_probability"][:, y_idx, :].T, "viridis"),
        ("facies mode", posterior["facies_mode"][:, y_idx, :].T, "tab20"),
        ("temperature", posterior["temperature_mean"][:, y_idx, :].T, "coolwarm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="upper", extent=[x.min(), x.max(), z.max(), z.min()], aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("local x (m)")
        ax.set_ylabel("depth (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    train_eic = split.train_observations.type_ids == 1
    hold_eic = split.holdout_observations.type_ids == 1
    axes[0, 0].scatter(split.train_observations.coords[train_eic, 0], split.train_observations.coords[train_eic, 2], c=split.train_observations.values[train_eic], s=14, cmap="viridis", edgecolors="white", linewidths=0.2)
    axes[0, 0].scatter(split.holdout_observations.coords[hold_eic, 0], split.holdout_observations.coords[hold_eic, 2], c=split.holdout_observations.values[hold_eic], s=30, marker="x", cmap="viridis")
    fig.suptitle(f"ArcticData-conditioned diffusion: {split.site}")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--observations", default="data/processed/arcticdata_cryostratigraphy_observations.npz")
    parser.add_argument("--token-index", default="outputs/tables/arcticdata_cryostratigraphy_token_index.csv")
    parser.add_argument("--site", default=None)
    parser.add_argument("--output-prefix", default="arcticdata_conditioned_diffusion")
    parser.add_argument("--max-site-span-m", type=float, default=5000.0)
    parser.add_argument("--min-eic-tokens", type=int, default=20)
    parser.add_argument("--min-boreholes", type=int, default=4)
    parser.add_argument("--target-shape", default="64,64,48")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--max-condition-tokens", type=int, default=768)
    parser.add_argument("--eic-guidance-weight", type=float, default=0.84)
    parser.add_argument("--physics-guidance-weight", type=float, default=1.0)
    parser.add_argument("--facies-guidance-weight", type=float, default=0.55)
    parser.add_argument("--cryofacies-eic-prior-weight", type=float, default=0.10)
    parser.add_argument("--hybrid-wedge-probability-threshold", type=float, default=0.80)
    parser.add_argument("--assimilation-horizontal-range", type=float, default=80.0)
    parser.add_argument("--assimilation-vertical-range", type=float, default=0.22)
    parser.add_argument("--assimilation-gain", type=float, default=0.96)
    parser.add_argument("--physics-projection-iterations", type=int, default=32)
    parser.add_argument("--physics-projection-strength", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    seed = int(config["project"].get("seed", 42))
    target_shape = tuple(int(item.strip()) for item in args.target_shape.split(","))
    if len(target_shape) != 3:
        raise ValueError("--target-shape must be nx,ny,nz")
    device_name = args.device or config["diffusion"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    split, site_tokens = _load_split(args, config)
    cond_obs = subsample_observations(split.train_observations, int(args.max_condition_tokens), seed)
    y_half_width = max(40.0, 0.5 * float(site_tokens["y"].max() - site_tokens["y"].min()) + 20.0)
    proxy = make_eic_proxy_fields(
        cond_obs,
        reference_observations=split.site_observations,
        config=EICProxyConfig(target_shape=target_shape, x_pad_m=20.0, y_half_width_m=y_half_width),
    )

    ae = _load_autoencoder(config, device)
    volume = fields_to_volume_tensor(proxy, n_facies=int(config["model"]["n_facies"])).to(device)
    with torch.no_grad():
        latent = ae.encode(volume)
    obs_encoder, denoiser = _load_conditioned_models(config, latent, device)
    grid = {"x": proxy["grid_x"], "y": proxy["grid_y"], "z": proxy["grid_z"]}
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(grid)
    obs_tokens = tokenizer.encode_torch(cond_obs, device=device).unsqueeze(0)
    obs_mask = torch.zeros((1, obs_tokens.shape[1]), dtype=torch.bool, device=device)
    obs_attention_mask = build_observation_attention_mask(config, grid, cond_obs, device=device)
    n_samples = int(args.samples)
    with torch.no_grad():
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask).repeat(n_samples, 1)
        scale = float(config["diffusion"].get("posterior_noise_scale", 0.08))
        correction = float(config["diffusion"].get("posterior_correction_scale", 0.15))
        latents = latent.repeat(n_samples, 1, 1, 1, 1) + scale * torch.randn((n_samples, *latent.shape[1:]), device=device)
        t_mid = max(int(config["diffusion"].get("timesteps", 80)) // 2, 1)
        t = torch.full((n_samples,), t_mid, device=device, dtype=torch.long)
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
    posterior, assimilation = assimilate_posterior_to_observations(
        posterior,
        split.train_observations,
        n_facies=int(config["model"]["n_facies"]),
        cfg=PosteriorAssimilationConfig(
            horizontal_range_m=float(args.assimilation_horizontal_range),
            vertical_range_m=float(args.assimilation_vertical_range),
            continuous_gain=float(args.assimilation_gain),
            eic_gain=float(args.assimilation_gain),
            facies_gain=float(args.assimilation_gain),
            max_observations_per_type=int(args.max_condition_tokens),
            seed=seed,
        ),
    )
    train_token_index = site_tokens[~site_tokens["borehole"].astype(str).isin(set(split.holdout_boreholes.tolist()))].copy()
    posterior = apply_cryofacies_eic_prior(
        posterior,
        train_token_index,
        weight=float(args.cryofacies_eic_prior_weight),
        n_facies=int(config["model"]["n_facies"]),
    )
    posterior = project_eic_conditioned_physics(
        posterior,
        n_facies=int(config["model"]["n_facies"]),
        heat_iterations=int(args.physics_projection_iterations),
        heat_strength=float(args.physics_projection_strength),
        heat_anchor=0.0,
    )
    posterior = eic_event_calibrated_posterior(posterior, split.train_observations, threshold=0.30)
    posterior = facies_hybrid_calibrated_posterior(
        posterior,
        split.train_observations,
        wedge_probability_threshold=float(args.hybrid_wedge_probability_threshold),
        knn_k=12,
    )
    posterior["settlement_potential"] = settlement_potential_numpy(
        posterior["eic_mean"],
        posterior["temperature_mean"] + 2.0,
        float(np.mean(np.diff(posterior["grid_z"]))),
    ).astype(np.float32)
    posterior["conditioning_site"] = np.asarray(split.site)
    posterior["train_boreholes_json"] = np.asarray(json.dumps(split.train_boreholes.tolist(), ensure_ascii=False))
    posterior["holdout_boreholes_json"] = np.asarray(json.dumps(split.holdout_boreholes.tolist(), ensure_ascii=False))

    baseline_metrics, baseline_predictions = arcticdata_conditioning_baseline_rows(split.train_observations, split.holdout_observations)
    diffusion_metrics, diffusion_predictions = evaluate_conditioned_posterior(posterior, split.holdout_observations)
    calibrated_metrics, calibrated_predictions = evaluate_conditioned_posterior(
        posterior,
        split.holdout_observations,
        model_name="COLDReconArcticDataEventCalibrated",
        eic_field="eic_event_calibrated_mean",
    )
    hybrid_metrics, hybrid_predictions = evaluate_hybrid_calibrated_posterior(
        posterior,
        split.train_observations,
        split.holdout_observations,
        model_name="COLDReconArcticDataHybridCalibrated",
        eic_field="eic_event_calibrated_mean",
        wedge_probability_threshold=float(args.hybrid_wedge_probability_threshold),
        knn_k=12,
    )
    adaptive_metrics, adaptive_predictions = evaluate_adaptive_hybrid_posterior(
        posterior,
        split.train_observations,
        split.holdout_observations,
        model_name="COLDReconArcticDataAdaptiveHybrid",
        eic_field="eic_mean",
        wedge_probability_threshold=float(args.hybrid_wedge_probability_threshold),
        knn_k=12,
    )
    wedge_recall_metrics, wedge_recall_predictions = evaluate_wedge_recall_posterior(
        posterior,
        split.train_observations,
        split.holdout_observations,
        model_name="COLDReconArcticDataWedgeRecallHead",
        knn_k=12,
    )
    metrics = pd.concat([baseline_metrics, diffusion_metrics, calibrated_metrics, hybrid_metrics, adaptive_metrics, wedge_recall_metrics], ignore_index=True)
    for key, value in {
        "site": split.site,
        "train_n": split.train_observations.n_obs,
        "condition_n": cond_obs.n_obs,
        "holdout_n": split.holdout_observations.n_obs,
        "train_boreholes": len(split.train_boreholes),
        "holdout_boreholes": len(split.holdout_boreholes),
        "assimilated_facies": assimilation.get("assimilated_facies", 0.0),
        "assimilated_borehole_eic": assimilation.get("assimilated_borehole_eic", 0.0),
    }.items():
        metrics[key] = value
    predictions = pd.concat([baseline_predictions, diffusion_predictions, calibrated_predictions, hybrid_predictions, adaptive_predictions, wedge_recall_predictions], ignore_index=True)

    pred_dir = Path(config["paths"]["predictions_dir"])
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    for path in (pred_dir, table_dir, fig_dir):
        path.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix).strip() or "arcticdata_conditioned_diffusion"
    pred_path = pred_dir / f"{prefix}.npz"
    metrics_path = table_dir / f"{prefix}_metrics.csv"
    predictions_path = table_dir / f"{prefix}_holdout_predictions.csv"
    fig_path = fig_dir / f"{prefix}_sections.png"
    np.savez_compressed(pred_path, **posterior)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    _plot_conditioned_posterior(posterior, split, fig_path)

    print(f"prediction={pred_path}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")
    print(f"figure={fig_path}")
    print(f"site={split.site}")
    print(f"train_boreholes={len(split.train_boreholes)}")
    print(f"holdout_boreholes={len(split.holdout_boreholes)}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
