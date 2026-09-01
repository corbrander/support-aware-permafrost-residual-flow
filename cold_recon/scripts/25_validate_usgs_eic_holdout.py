from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.data.usgs_eic_loader import read_usgs_eic_tables
from cold_recon.evaluation.usgs_eic_validation import EICValidationConfig, eic_holdout_validation_tables
from cold_recon.utils.config import ensure_dirs, load_config


def _plot_validation(predictions: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = metrics["model"].tolist()
    colors = {
        "GlobalMean": "#8c8c8c",
        "DepthIDW": "#4c78a8",
        "SpatialDepthIDW": "#f58518",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for model in models:
        group = predictions[predictions["model"] == model]
        axes[0].scatter(
            group["observed_eic"],
            group["predicted_eic"],
            s=20,
            alpha=0.68,
            label=model,
            color=colors.get(model),
            edgecolors="none",
        )
    axes[0].plot([0.0, 1.0], [0.0, 1.0], color="black", lw=1.0, ls="--")
    axes[0].set_xlim(-0.02, max(0.55, float(predictions[["observed_eic", "predicted_eic"]].max().max()) + 0.04))
    axes[0].set_ylim(-0.02, axes[0].get_xlim()[1])
    axes[0].set_xlabel("observed EIC fraction")
    axes[0].set_ylabel("predicted EIC fraction")
    axes[0].set_title("Leave-one-borehole predictions")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, color="0.9")

    x = np.arange(len(models))
    axes[1].bar(x - 0.18, metrics["mae"].astype(float), width=0.36, label="MAE", color="#4c78a8")
    axes[1].bar(x + 0.18, metrics["rmse"].astype(float), width=0.36, label="RMSE", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha="right")
    axes[1].set_ylabel("EIC fraction error")
    axes[1].set_title("Hold-out error")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", color="0.9")

    best_model = "SpatialDepthIDW" if "SpatialDepthIDW" in set(predictions["model"]) else models[-1]
    best = predictions[predictions["model"] == best_model].copy()
    sc = axes[2].scatter(best["error"], best["depth_mid_m"], c=best["observed_eic"], s=24, cmap="viridis", alpha=0.78)
    axes[2].axvline(0.0, color="black", lw=1.0, ls="--")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("prediction error")
    axes[2].set_ylabel("depth (m)")
    axes[2].set_title(f"{best_model} residuals by depth")
    axes[2].grid(True, color="0.9")
    fig.colorbar(sc, ax=axes[2], label="observed EIC")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--borehole-spacing-m", type=float, default=20.0)
    parser.add_argument("--horizontal-scale-m", type=float, default=20.0)
    parser.add_argument("--depth-scale-m", type=float, default=0.25)
    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--depth-k", type=int, default=12)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    eic, locations = read_usgs_eic_tables(Path(config["paths"]["raw_dir"]))
    validation_config = EICValidationConfig(
        borehole_spacing_m=args.borehole_spacing_m,
        horizontal_scale_m=args.horizontal_scale_m,
        depth_scale_m=args.depth_scale_m,
        idw_k=args.idw_k,
        depth_k=args.depth_k,
    )
    tables = eic_holdout_validation_tables(eic, locations, validation_config)
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = table_dir / "usgs_eic_holdout_metrics.csv"
    predictions_path = table_dir / "usgs_eic_holdout_predictions.csv"
    per_borehole_path = table_dir / "usgs_eic_holdout_per_borehole.csv"
    tables["metrics"].to_csv(metrics_path, index=False)
    tables["predictions"].to_csv(predictions_path, index=False)
    tables["per_borehole"].to_csv(per_borehole_path, index=False)
    fig_path = fig_dir / "usgs_eic_holdout_validation.png"
    _plot_validation(tables["predictions"], tables["metrics"], fig_path)
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")
    print(f"per_borehole={per_borehole_path}")
    print(f"figure={fig_path}")
    print(tables["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
