from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.data.jago_ground_ice_loader import jago_ground_ice_eic_table, write_jago_ground_ice_inventory
from cold_recon.evaluation.usgs_eic_validation import EICValidationConfig, eic_holdout_validation_tables
from cold_recon.utils.config import ensure_dirs, load_config


def _plot_validation(predictions: pd.DataFrame, metrics: pd.DataFrame, intervals: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "GlobalMean": "#8c8c8c",
        "DepthIDW": "#4c78a8",
        "SpatialDepthIDW": "#f58518",
    }
    models = metrics["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)

    for model in models:
        group = predictions[predictions["model"] == model]
        axes[0].scatter(
            group["observed_eic"],
            group["predicted_eic"],
            s=28,
            alpha=0.72,
            label=model,
            color=colors.get(model),
            edgecolors="none",
        )
    lim = max(0.65, float(predictions[["observed_eic", "predicted_eic"]].max().max()) + 0.05)
    axes[0].plot([0.0, lim], [0.0, lim], color="black", lw=0.9, ls="--")
    axes[0].set_xlim(-0.02, lim)
    axes[0].set_ylim(-0.02, lim)
    axes[0].set_xlabel("observed EIC fraction")
    axes[0].set_ylabel("predicted EIC fraction")
    axes[0].set_title("Leave-one-borehole-out")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, color="0.9")

    x = np.arange(len(models))
    axes[1].bar(x - 0.18, metrics["mae"].astype(float), width=0.36, label="MAE", color="#4c78a8")
    axes[1].bar(x + 0.18, metrics["rmse"].astype(float), width=0.36, label="RMSE", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha="right")
    axes[1].set_ylabel("EIC fraction error")
    axes[1].set_title("Held-out interval error")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, axis="y", color="0.9")

    intervals = intervals.sort_values(["BOREHOLE_ID", "depth_mid_m"])
    boreholes = {name: i for i, name in enumerate(sorted(intervals["BOREHOLE_ID"].astype(str).unique()))}
    sc = axes[2].scatter(
        intervals["BOREHOLE_ID"].map(boreholes),
        intervals["depth_mid_m"],
        c=intervals["eic_fraction"],
        s=38,
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.6, float(intervals["eic_fraction"].max())),
        edgecolors="black",
        linewidths=0.2,
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("ordered Jago borehole")
    axes[2].set_ylabel("depth (m)")
    axes[2].set_title("Measured Jago ground ice")
    axes[2].grid(True, color="0.9")
    fig.colorbar(sc, ax=axes[2], label="EIC fraction")
    fig.savefig(out_path, dpi=220, facecolor="white")
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
    paths = config["paths"]
    outputs = write_jago_ground_ice_inventory(config)
    inventory = pd.read_csv(outputs["inventory_csv"])
    eic = jago_ground_ice_eic_table(inventory)
    validation_config = EICValidationConfig(
        borehole_spacing_m=float(args.borehole_spacing_m),
        horizontal_scale_m=float(args.horizontal_scale_m),
        depth_scale_m=float(args.depth_scale_m),
        idw_k=int(args.idw_k),
        depth_k=int(args.depth_k),
    )
    tables = eic_holdout_validation_tables(eic, None, validation_config)

    table_dir = Path(paths["tables_dir"])
    fig_dir = Path(paths["figures_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    intervals_path = table_dir / "arcticdata_jago_ground_ice_validation_intervals.csv"
    metrics_path = table_dir / "arcticdata_jago_ground_ice_eic_holdout_metrics.csv"
    predictions_path = table_dir / "arcticdata_jago_ground_ice_eic_holdout_predictions.csv"
    per_borehole_path = table_dir / "arcticdata_jago_ground_ice_eic_holdout_per_borehole.csv"
    fig_path = fig_dir / "arcticdata_jago_ground_ice_holdout_validation.png"

    tables["intervals"].to_csv(intervals_path, index=False)
    tables["metrics"].to_csv(metrics_path, index=False)
    tables["predictions"].to_csv(predictions_path, index=False)
    tables["per_borehole"].to_csv(per_borehole_path, index=False)
    _plot_validation(tables["predictions"], tables["metrics"], tables["intervals"], fig_path)

    print(f"intervals={intervals_path}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")
    print(f"per_borehole={per_borehole_path}")
    print(f"figure={fig_path}")
    print(tables["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
