from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.synthetic_ablation import run_synthetic_ablation
from cold_recon.utils.config import ensure_dirs, load_config


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_ablation(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["mean_iou", "eic_rmse", "temperature_rmse", "ice_rich_recall"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        for scenario, group in df.groupby("scenario"):
            group = group.sort_values("n_boreholes")
            if metric in group:
                ax.plot(group["n_boreholes"], group[metric], marker="o", label=scenario)
        ax.set_title(metric)
        ax.set_xlabel("Number of boreholes")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--boreholes", default="2,4,8")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    boreholes = [int(x.strip()) for x in args.boreholes.split(",") if x.strip()]
    rows = run_synthetic_ablation(
        sample,
        borehole_counts=boreholes,
        seed=int(config["project"].get("seed", 42)),
        n_facies=int(config["model"]["n_facies"]),
    )
    table_path = Path(config["paths"]["tables_dir"]) / "ablation_metrics.csv"
    _write_rows(table_path, rows)
    df = pd.DataFrame(rows)
    fig_path = Path(config["paths"]["figures_dir"]) / "ablation_sparsity_curves.png"
    _plot_ablation(df, fig_path)
    print(f"metrics={table_path}")
    print(f"figure={fig_path}")


if __name__ == "__main__":
    main()
