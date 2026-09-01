from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cold_recon.evaluation.synthetic_benchmark import (
    aggregate_synthetic_benchmark,
    summarize_synthetic_paths,
    write_synthetic_benchmark_tables,
)
from cold_recon.synthetic.cryo_synth_generator import generate_synthetic_sample, save_synthetic_sample
from cold_recon.utils.config import ensure_dirs, load_config


def _ensure_samples(config: dict, n_samples: int, seed: int, force: bool = False) -> list[Path]:
    out_dir = Path(config["paths"]["synthetic_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in range(n_samples):
        path = out_dir / f"sample_{idx:04d}.npz"
        if force or not path.exists():
            sample = generate_synthetic_sample(config, seed=seed + idx, site_id=f"synthetic_{idx:04d}")
            save_synthetic_sample(path, sample)
        paths.append(path)
    return paths


def _plot_benchmark(rows: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("ice_rich_fraction", "ice-rich fraction"),
        ("active_layer_mean", "mean ALT (m)"),
        ("eic_mean", "mean EIC"),
        ("truth_heat_residual_rmse", "truth heat residual"),
    ]
    facies_cols = [col for col in rows.columns if col.startswith("facies_fraction_")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(rows["sample_id"].astype(str), rows[col].astype(float), color="#4c78a8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(True, axis="y", color="0.9")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    if facies_cols:
        facies_path = out_path.with_name("synthetic_ensemble_facies_fractions.png")
        labels = [col.replace("facies_fraction_", "") for col in facies_cols]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        bottom = None
        x = rows["sample_id"].astype(str).to_numpy()
        for col, label in zip(facies_cols, labels):
            vals = rows[col].astype(float).to_numpy()
            ax.bar(x, vals, bottom=bottom, label=label)
            bottom = vals if bottom is None else bottom + vals
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("volume fraction")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.legend(ncol=3, fontsize=8)
        ax.set_title("Synthetic ensemble facies fractions")
        fig.savefig(facies_path, dpi=180, facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    bench_cfg = config.get("synthetic_benchmark", {})
    n_samples = int(args.n_samples if args.n_samples is not None else bench_cfg.get("n_samples", 5))
    seed = int(args.seed if args.seed is not None else bench_cfg.get("seed", config.get("project", {}).get("seed", 42)))
    paths = _ensure_samples(config, n_samples=n_samples, seed=seed, force=args.force)
    rows = summarize_synthetic_paths(paths)
    summary = aggregate_synthetic_benchmark(rows)
    table_paths = write_synthetic_benchmark_tables(rows, summary, Path(config["paths"]["tables_dir"]))
    fig_path = Path(config["paths"]["figures_dir"]) / "synthetic_ensemble_benchmark.png"
    _plot_benchmark(rows, fig_path)
    print(f"detail={table_paths['detail']}")
    print(f"summary={table_paths['summary']}")
    print(f"figure={fig_path}")
    print(f"n_samples={len(paths)}")


if __name__ == "__main__":
    main()
