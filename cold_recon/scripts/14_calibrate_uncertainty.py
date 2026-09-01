from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import alt_from_temperature, rmse
from cold_recon.evaluation.uncertainty import (
    brier_score,
    categorical_nll,
    ensemble_crps,
    facies_entropy,
    interval_coverage,
    reliability_by_level,
    uncertainty_error_correlation,
)
from cold_recon.utils.config import ensure_dirs, load_config


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["target", "kind", "rmse", "crps", "coverage_50", "coverage_70", "coverage_90", "coverage_95", "width_90"]
    ordered = [name for name in preferred if name in fieldnames] + [name for name in fieldnames if name not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _continuous_row(name: str, samples: np.ndarray, truth: np.ndarray, mean: np.ndarray | None = None) -> dict[str, float | str]:
    mean_arr = samples.mean(axis=0) if mean is None else mean
    row: dict[str, float | str] = {
        "target": name,
        "kind": "continuous",
        "rmse": rmse(mean_arr, truth),
        "crps": ensemble_crps(samples, truth),
        "mean_std": float(np.mean(np.std(samples, axis=0))),
        "std_error_corr": uncertainty_error_correlation(np.std(samples, axis=0), np.abs(mean_arr - truth)),
    }
    for level, stats in reliability_by_level(samples, truth, levels=[0.50, 0.70, 0.90, 0.95]).items():
        suffix = int(round(level * 100))
        row[f"coverage_{suffix}"] = stats["coverage"]
        row[f"width_{suffix}"] = stats["width"]
    row["coverage_error_90"] = abs(float(row["coverage_90"]) - 0.90)
    return row


def _plot_reliability(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    levels = np.array([0.50, 0.70, 0.90, 0.95], dtype=np.float32)
    ax.plot([0.45, 1.0], [0.45, 1.0], color="0.2", linewidth=1.2, linestyle="--", label="ideal")
    for row in rows:
        if row.get("kind") != "continuous":
            continue
        observed = np.array([float(row[f"coverage_{int(round(level * 100))}"]) for level in levels], dtype=np.float32)
        ax.plot(levels, observed, marker="o", linewidth=1.8, label=str(row["target"]))
    ax.set_xlim(0.48, 0.97)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("nominal central interval")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Synthetic posterior calibration")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--posterior", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    posterior_path = Path(args.posterior or config["diffusion"]["posterior_path"])
    posterior = np.load(posterior_path, allow_pickle=False)
    truth = sample["fields"]
    rows: list[dict[str, float | str]] = []

    specs = [
        ("eic", "eic_samples", "eic_mean", truth["eic"]),
        ("temperature", "temperature_samples", "temperature_mean", truth["temperature"]),
        ("unfrozen_water", "unfrozen_water_samples", "unfrozen_water_mean", truth["unfrozen_water"]),
        ("log_resistivity", "log_resistivity_samples", "log_resistivity_mean", np.log(np.maximum(truth["resistivity"], 1.0))),
    ]
    for name, sample_key, mean_key, target in specs:
        if sample_key in posterior:
            rows.append(_continuous_row(name, posterior[sample_key], target, posterior[mean_key] if mean_key in posterior else None))

    if "temperature_samples" in posterior:
        alt_samples = np.stack([alt_from_temperature(arr, sample["grid"]["z"]) for arr in posterior["temperature_samples"]])
        alt_truth = alt_from_temperature(truth["temperature"], sample["grid"]["z"])
        rows.append(_continuous_row("active_layer_thickness", alt_samples, alt_truth))

    if "ice_rich_probability" in posterior:
        event = truth["eic"] > float(config["evaluation"].get("ice_rich_threshold", 0.30))
        rows.append(
            {
                "target": "ice_rich_probability",
                "kind": "binary_event",
                "brier": brier_score(posterior["ice_rich_probability"], event),
                "event_rate": float(np.mean(event)),
                "predicted_event_rate": float(np.mean(posterior["ice_rich_probability"])),
            }
        )

    if "facies_probability" in posterior:
        probabilities = posterior["facies_probability"]
        facies_truth = truth["facies"]
        rows.append(
            {
                "target": "facies",
                "kind": "categorical",
                "nll": categorical_nll(probabilities, facies_truth),
                "accuracy": float(np.mean(np.argmax(probabilities, axis=-1) == facies_truth)),
                "mean_entropy": float(np.mean(facies_entropy(probabilities))),
                "true_class_probability": float(np.mean(np.take_along_axis(probabilities, facies_truth[..., None], axis=-1))),
            }
        )

    table_path = Path(config["paths"]["tables_dir"]) / "uncertainty_calibration_metrics.csv"
    _write_rows(table_path, rows)
    fig_path = Path(config["paths"]["figures_dir"]) / "uncertainty_reliability.png"
    _plot_reliability(rows, fig_path)
    print(f"metrics={table_path}")
    print(f"figure={fig_path}")
    for row in rows:
        summary = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k in {"target", "kind", "crps", "coverage_90", "brier", "nll", "accuracy"})
        print(summary)


if __name__ == "__main__":
    main()
