from __future__ import annotations

import argparse
import csv
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.posterior_calibration import calibrate_posterior_spread
from cold_recon.utils.config import ensure_dirs, load_config


_uncertainty_script = import_module("cold_recon.scripts.14_calibrate_uncertainty")
_continuous_row = _uncertainty_script._continuous_row
_plot_reliability = _uncertainty_script._plot_reliability
_write_rows = _uncertainty_script._write_rows


def _write_scale_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "target",
        "calibration_method",
        "level",
        "target_coverage",
        "scale_factor",
        "bias_correction",
        "residual_half_width",
        "coverage_before",
        "coverage_after",
        "width_before",
        "width_after",
    ]
    all_fields = sorted({key for row in rows for key in row})
    fieldnames = [name for name in preferred if name in all_fields] + [name for name in all_fields if name not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_scale_factors(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["target"]) for row in rows]
    values = [float(row["scale_factor"]) for row in rows]
    methods = [str(row.get("calibration_method", "spread_scaled")) for row in rows]
    colors = ["#B64342" if method == "bias_quantile" else "#4c78a8" for method in methods]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("spread scale factor before fallback (log)")
    ax.set_title("Post-hoc posterior interval calibration")
    ax.grid(True, axis="y", color="0.9")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _calibrated_metric_rows(
    posterior: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    z: np.ndarray,
    scale_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    method_by_target = {str(row["target"]): str(row.get("calibration_method", "spread_scaled")) for row in scale_rows}
    specs = [
        ("eic", "eic_samples", "eic_mean", truth["eic"]),
        ("temperature", "temperature_samples", "temperature_mean", truth["temperature"]),
        ("unfrozen_water", "unfrozen_water_samples", "unfrozen_water_mean", truth["unfrozen_water"]),
        ("log_resistivity", "log_resistivity_samples", "log_resistivity_mean", np.log(np.maximum(truth["resistivity"], 1.0))),
    ]
    for name, sample_key, mean_key, target in specs:
        if sample_key in posterior:
            row = _continuous_row(name, posterior[sample_key], target, posterior[mean_key] if mean_key in posterior else None)
            row["calibration"] = method_by_target.get(name, "posthoc_calibrated")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--posterior", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--level", type=float, default=0.90)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    posterior_path = Path(args.posterior or config["diffusion"]["posterior_path"])
    posterior = dict(np.load(posterior_path, allow_pickle=False))
    calibrated, scale_rows = calibrate_posterior_spread(
        posterior,
        sample["fields"],
        target_coverage=float(args.target_coverage),
        level=float(args.level),
        ice_threshold=float(config["evaluation"].get("ice_rich_threshold", 0.30)),
    )
    out_path = Path(args.output or (Path(config["paths"]["predictions_dir"]) / "diffusion_posterior_calibrated.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **calibrated)

    table_dir = Path(config["paths"]["tables_dir"])
    scale_path = table_dir / "posterior_spread_calibration_factors.csv"
    _write_scale_rows(scale_path, scale_rows)
    metrics = _calibrated_metric_rows(calibrated, sample["fields"], sample["grid"]["z"], scale_rows)
    metrics_path = table_dir / "uncertainty_calibration_metrics_calibrated.csv"
    _write_rows(metrics_path, metrics)

    fig_dir = Path(config["paths"]["figures_dir"])
    _plot_reliability(metrics, fig_dir / "uncertainty_reliability_calibrated.png")
    _plot_scale_factors(scale_rows, fig_dir / "posterior_spread_scale_factors.png")

    print(f"posterior={out_path}")
    print(f"scale_factors={scale_path}")
    print(f"metrics={metrics_path}")
    for row in scale_rows:
        print(
            f"{row['target']}: method={row.get('calibration_method', 'spread_scaled')}, scale={float(row['scale_factor']):.6g}, "
            f"coverage {float(row['coverage_before']):.6g}->{float(row['coverage_after']):.6g}"
        )


if __name__ == "__main__":
    main()
