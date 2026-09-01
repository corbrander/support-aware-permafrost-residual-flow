from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.physics_consistency import facies_to_probability
from cold_recon.physics.geotechnical import GeotechnicalParameters, geotechnical_summary
from cold_recon.utils.config import ensure_dirs, load_config


def _prediction_specs(config: dict) -> list[tuple[str, str, Path]]:
    pred_dir = Path(config["paths"]["predictions_dir"])
    return [
        ("truth", "synthetic", Path(config["training"]["sample_path"])),
        ("IDW", "synthetic", pred_dir / "baseline_idw.npz"),
        ("RandomForest", "synthetic", pred_dir / "baseline_random_forest.npz"),
        ("GradientBoosting", "synthetic", pred_dir / "baseline_gradient_boosting.npz"),
        ("KrigingGPR", "synthetic", pred_dir / "baseline_kriging.npz"),
        ("SparseUNet3D", "synthetic", pred_dir / "baseline_unet3d.npz"),
        ("COLDReconImplicit", "synthetic", pred_dir / "implicit_prediction.npz"),
        ("COLDReconLatentDiffusion", "synthetic", pred_dir / "diffusion_posterior.npz"),
        ("COLDReconFNOOperatorDiffusion", "synthetic", pred_dir / "fno_operator_diffusion_posterior.npz"),
        ("COLDReconRectifiedFlow", "synthetic", pred_dir / "rectified_flow_posterior.npz"),
        ("COLDReconLatentDiffusionPhysicsTrained", "synthetic", pred_dir / "diffusion_posterior_physics_trained.npz"),
        ("COLDReconLatentDiffusionPhysicsGuided", "synthetic", pred_dir / "diffusion_posterior_physics_guided.npz"),
        ("COLDReconLatentDiffusionPhysicsRefined", "synthetic", pred_dir / "diffusion_posterior_physics_refined.npz"),
        ("COLDReconLatentDiffusionCalibrated", "synthetic", pred_dir / "diffusion_posterior_calibrated.npz"),
        ("USGSRealConditionedDiffusion", "field", pred_dir / "usgs_real_conditioned_diffusion.npz"),
        ("USGSEICConditionedDiffusion", "field_eic", pred_dir / "usgs_eic_conditioned_diffusion.npz"),
    ]


def _truth_posterior(sample: dict) -> dict[str, np.ndarray]:
    fields = sample["fields"]
    grid = sample["grid"]
    return {
        "eic_mean": np.asarray(fields["eic"], dtype=np.float32),
        "temperature_mean": np.asarray(fields["temperature"], dtype=np.float32),
        "unfrozen_water_mean": np.asarray(fields["unfrozen_water"], dtype=np.float32),
        "facies": np.asarray(fields["facies"], dtype=np.int16),
        "grid_x": np.asarray(grid["x"], dtype=np.float32),
        "grid_y": np.asarray(grid["y"], dtype=np.float32),
        "grid_z": np.asarray(grid["z"], dtype=np.float32),
    }


def _canonical_posterior(data: dict[str, np.ndarray], sample: dict, n_facies: int) -> dict[str, np.ndarray]:
    aliases = {
        "eic_mean": ("eic_mean", "eic", "field_eic"),
        "temperature_mean": ("temperature_mean", "temperature", "field_temperature"),
        "unfrozen_water_mean": ("unfrozen_water_mean", "unfrozen_water", "field_unfrozen_water"),
        "settlement_potential": ("settlement_potential",),
    }
    posterior: dict[str, np.ndarray] = {}
    for out_key, in_keys in aliases.items():
        for key in in_keys:
            if key in data:
                posterior[out_key] = np.asarray(data[key], dtype=np.float32)
                break
    if "facies_probability" in data:
        posterior["facies_probability"] = np.asarray(data["facies_probability"], dtype=np.float32)
    elif "facies_mode" in data:
        posterior["facies_mode"] = np.asarray(data["facies_mode"], dtype=np.int16)
    elif "facies" in data:
        posterior["facies"] = np.asarray(data["facies"], dtype=np.int16)
    elif "field_facies" in data:
        posterior["facies"] = np.asarray(data["field_facies"], dtype=np.int16)
    else:
        shape = posterior["eic_mean"].shape
        posterior["facies_probability"] = facies_to_probability(np.zeros(shape, dtype=np.int16), n_facies=n_facies)
    for out_key, grid_key, sample_key in [
        ("grid_x", "grid_x", "x"),
        ("grid_y", "grid_y", "y"),
        ("grid_z", "grid_z", "z"),
    ]:
        if grid_key in data:
            posterior[out_key] = np.asarray(data[grid_key], dtype=np.float32)
        else:
            posterior[out_key] = np.asarray(sample["grid"][sample_key], dtype=np.float32)
    missing = [key for key in ("eic_mean", "temperature_mean", "unfrozen_water_mean") if key not in posterior]
    if missing:
        raise KeyError(f"Missing geotechnical posterior fields: {missing}")
    return posterior


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "domain", *sorted({key for row in rows for key in row if key not in {"model", "domain"}})]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["model"]) for row in rows]
    y = np.arange(len(labels))
    metrics = [
        ("future_shear_strength_p10_kpa", "future s_u p10 (kPa)", "#7d9f35"),
        ("future_modulus_p10_mpa", "future E_t p10 (MPa)", "#5b8fd1"),
        ("settlement_potential_p95_m", "settlement p95 (m)", "#d47a42"),
        ("engineering_risk_p95", "engineering risk p95", "#9c5fb8"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (key, title, color) in zip(axes.ravel(), metrics):
        values = [float(row.get(key, np.nan)) for row in rows]
        ax.barh(y, values, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="x", color="0.9")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _plot_maps(payloads: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]], out_path: Path) -> None:
    preferred = {
        "COLDReconLatentDiffusion",
        "COLDReconFNOOperatorDiffusion",
        "COLDReconRectifiedFlow",
        "COLDReconLatentDiffusionPhysicsRefined",
        "USGSRealConditionedDiffusion",
        "USGSEICConditionedDiffusion",
    }
    selected = [payload for payload in payloads if payload[0] in preferred]
    if not selected:
        selected = payloads[:6]
    selected = selected[:6]
    if not selected:
        return
    vmax = max(float(np.nanpercentile(fields["engineering_risk_index"], 98.0)) for _, fields, _ in selected)
    vmax = max(vmax, 1e-6)
    ncols = 3 if len(selected) > 2 else len(selected)
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, (model, fields, posterior) in zip(axes_arr, selected):
        risk = fields["engineering_risk_index"]
        x = np.asarray(posterior["grid_x"], dtype=np.float32)
        y = np.asarray(posterior["grid_y"], dtype=np.float32)
        extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
        im = ax.imshow(risk.T, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="risk index")
    for ax in axes_arr[len(selected) :]:
        ax.axis("off")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _field_artifacts(
    slug: str,
    fields: dict[str, np.ndarray],
    posterior: dict[str, np.ndarray],
    max_depth_m: float,
) -> dict[str, np.ndarray]:
    z = np.asarray(posterior["grid_z"], dtype=np.float32)
    depth_mask = z <= float(max_depth_m)
    if not np.any(depth_mask):
        depth_mask[0] = True
    return {
        f"{slug}_engineering_risk_index": fields["engineering_risk_index"].astype(np.float32),
        f"{slug}_settlement_potential": fields["settlement_potential"].astype(np.float32),
        f"{slug}_differential_settlement_gradient": fields["differential_settlement_gradient"].astype(np.float32),
        f"{slug}_strength_loss_surface": fields["strength_loss_surface"].astype(np.float32),
        f"{slug}_future_shear_strength_p10_surface_kpa": np.percentile(
            fields["future_shear_strength_kpa"][:, :, depth_mask],
            10.0,
            axis=2,
        ).astype(np.float32),
        f"{slug}_future_modulus_p10_surface_mpa": np.percentile(
            fields["future_modulus_mpa"][:, :, depth_mask],
            10.0,
            axis=2,
        ).astype(np.float32),
        f"{slug}_grid_x": np.asarray(posterior["grid_x"], dtype=np.float32),
        f"{slug}_grid_y": np.asarray(posterior["grid_y"], dtype=np.float32),
        f"{slug}_grid_z": np.asarray(posterior["grid_z"], dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--future-warming-c", type=float, default=2.0)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(config["training"]["sample_path"])
    n_facies = int(config["model"]["n_facies"])
    params = GeotechnicalParameters(future_warming_c=float(args.future_warming_c))
    rows: list[dict[str, float | str]] = []
    payloads: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    artifact_arrays: dict[str, np.ndarray] = {
        "future_warming_c": np.asarray(float(args.future_warming_c), dtype=np.float32),
        "max_depth_m": np.asarray(float(args.max_depth_m), dtype=np.float32),
    }
    model_names: list[str] = []
    for model, domain, path in _prediction_specs(config):
        if not path.exists():
            continue
        if model == "truth":
            posterior = _truth_posterior(sample)
        else:
            posterior = _canonical_posterior(dict(np.load(path, allow_pickle=False)), sample, n_facies=n_facies)
        fields, metrics = geotechnical_summary(posterior, n_facies=n_facies, params=params, max_depth_m=float(args.max_depth_m))
        row: dict[str, float | str] = {"model": model, "domain": domain}
        row.update(metrics)
        rows.append(row)
        payloads.append((model, fields, posterior))
        slug = _slug(model)
        model_names.append(model)
        artifact_arrays.update(_field_artifacts(slug, fields, posterior, max_depth_m=float(args.max_depth_m)))
    table_path = Path(config["paths"]["tables_dir"]) / "geotechnical_risk_metrics.csv"
    _write_rows(table_path, rows)
    artifact_arrays["model_names"] = np.asarray(model_names)
    fields_path = Path(config["paths"]["predictions_dir"]) / "geotechnical_risk_fields.npz"
    fields_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fields_path, **artifact_arrays)
    summary_fig = Path(config["paths"]["figures_dir"]) / "geotechnical_risk_summary.png"
    maps_fig = Path(config["paths"]["figures_dir"]) / "geotechnical_risk_maps.png"
    _plot_summary(rows, summary_fig)
    _plot_maps(payloads, maps_fig)
    print(f"metrics={table_path}")
    print(f"fields={fields_path}")
    print(f"summary_figure={summary_fig}")
    print(f"maps_figure={maps_fig}")
    for row in rows:
        print(
            f"{row['model']}: risk_p95={float(row['engineering_risk_p95']):.4f}, "
            f"future_su_p10={float(row['future_shear_strength_p10_kpa']):.3f} kPa, "
            f"future_Et_p10={float(row['future_modulus_p10_mpa']):.3f} MPa"
        )


if __name__ == "__main__":
    main()
