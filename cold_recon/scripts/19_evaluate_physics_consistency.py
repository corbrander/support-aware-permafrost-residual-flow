from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.physics_consistency import fields_from_prediction, physics_consistency_metrics, sample_truth_fields
from cold_recon.utils.config import ensure_dirs, load_config


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "domain", *sorted({key for row in rows for key in row if key not in {"model", "domain"}})]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_physics(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["model"]) for row in rows]
    metrics = [
        ("unfrozen_water_empirical_mae", "UW empirical MAE"),
        ("log_resistivity_empirical_mae", "log-rho empirical MAE"),
        ("heat_residual_rmse", "heat residual RMSE"),
        ("stratigraphic_tv_xy", "facies TV xy"),
    ]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        values = [float(row.get(key, np.nan)) for row in rows]
        ax.barh(y, values, color="#5b8fd1")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="x", color="0.9")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _prediction_specs(config: dict) -> list[tuple[str, str, str]]:
    pred_dir = Path(config["paths"]["predictions_dir"])
    return [
        ("truth", "synthetic", str(config["training"]["sample_path"])),
        ("IDW", "synthetic", str(pred_dir / "baseline_idw.npz")),
        ("RandomForest", "synthetic", str(pred_dir / "baseline_random_forest.npz")),
        ("GradientBoosting", "synthetic", str(pred_dir / "baseline_gradient_boosting.npz")),
        ("KrigingGPR", "synthetic", str(pred_dir / "baseline_kriging.npz")),
        ("SparseUNet3D", "synthetic", str(pred_dir / "baseline_unet3d.npz")),
        ("COLDReconImplicit", "synthetic", str(pred_dir / "implicit_prediction.npz")),
        ("COLDReconLatentDiffusion", "synthetic", str(pred_dir / "diffusion_posterior.npz")),
        ("COLDReconFNOOperatorDiffusion", "synthetic", str(pred_dir / "fno_operator_diffusion_posterior.npz")),
        ("COLDReconRectifiedFlow", "synthetic", str(pred_dir / "rectified_flow_posterior.npz")),
        ("COLDReconLatentDiffusionPhysicsTrained", "synthetic", str(pred_dir / "diffusion_posterior_physics_trained.npz")),
        ("COLDReconLatentDiffusionPhysicsGuided", "synthetic", str(pred_dir / "diffusion_posterior_physics_guided.npz")),
        ("COLDReconLatentDiffusionPhysicsRefined", "synthetic", str(pred_dir / "diffusion_posterior_physics_refined.npz")),
        ("COLDReconLatentDiffusionCalibrated", "synthetic", str(pred_dir / "diffusion_posterior_calibrated.npz")),
        ("USGSRealConditionedDiffusion", "field", str(pred_dir / "usgs_real_conditioned_diffusion.npz")),
        ("USGSEICConditionedDiffusion", "field_eic", str(pred_dir / "usgs_eic_conditioned_diffusion.npz")),
    ]


def _spacing_from_prediction(data: dict[str, np.ndarray], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    keys = ("grid_x", "grid_y", "grid_z")
    if not all(key in data for key in keys):
        return fallback
    spacing = []
    for key, fb in zip(keys, fallback):
        axis = np.asarray(data[key], dtype=np.float32)
        spacing.append(float(np.mean(np.diff(axis))) if len(axis) > 1 else float(fb))
    return tuple(spacing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(config["training"]["sample_path"])
    spacing = tuple(float(x) for x in sample["grid"].get("spacing", (sample["grid"]["dx"], sample["grid"]["dy"], sample["grid"]["dz"])))
    n_facies = int(config["model"]["n_facies"])
    rows: list[dict[str, float | str]] = []
    for model, domain, path_str in _prediction_specs(config):
        path = Path(path_str)
        if not path.exists():
            continue
        if model == "truth":
            fields = sample_truth_fields(sample, n_facies=n_facies)
            row_spacing = spacing
        else:
            data = dict(np.load(path, allow_pickle=False))
            fields = fields_from_prediction(data, n_facies=n_facies)
            row_spacing = _spacing_from_prediction(data, spacing)
        row: dict[str, float | str] = {"model": model, "domain": domain}
        row.update(physics_consistency_metrics(fields, spacing=row_spacing))
        rows.append(row)
    table_path = Path(config["paths"]["tables_dir"]) / "physics_consistency_metrics.csv"
    _write_rows(table_path, rows)
    fig_path = Path(config["paths"]["figures_dir"]) / "physics_consistency_summary.png"
    _plot_physics(rows, fig_path)
    print(f"metrics={table_path}")
    print(f"figure={fig_path}")
    for row in rows:
        print(
            f"{row['model']}: uw_mae={float(row['unfrozen_water_empirical_mae']):.6f}, "
            f"rho_mae={float(row['log_resistivity_empirical_mae']):.6f}, "
            f"heat_rmse={float(row['heat_residual_rmse']):.6f}"
        )


if __name__ == "__main__":
    main()
