from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import observations_from_npz
from cold_recon.evaluation.field_reconstruction import (
    evaluate_holdout_observations,
    make_field_grid,
    reconstruct_field_from_observations,
    split_observations_by_type,
)
from cold_recon.utils.config import ensure_dirs, load_config


def _write_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def _plot_field(recon: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = recon["grid_x"]
    y = recon["grid_y"]
    z = recon["grid_z"]
    y_idx = len(y) // 3
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    panels = [
        ("log resistivity", recon["log_resistivity_mean"][:, y_idx, :].T, "magma"),
        ("unfrozen water", recon["unfrozen_water_mean"][:, y_idx, :].T, "Blues"),
        ("temperature", recon["temperature_mean"][:, y_idx, :].T, "coolwarm"),
        ("EIC mean", recon["eic_mean"][:, y_idx, :].T, "viridis"),
        ("EIC std", recon["eic_std"][:, y_idx, :].T, "magma"),
        ("ice-rich probability", recon["ice_rich_probability"][:, y_idx, :].T, "viridis"),
    ]
    for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="upper", aspect="auto", extent=[x.min(), x.max(), z.max(), z.min()], cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("distance x (m)")
        ax.set_ylabel("depth (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)

    map_path = out_path.with_name("usgs_field_settlement_potential.png")
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    im = ax.imshow(
        recon["settlement_potential"].T,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        cmap="inferno",
    )
    ax.set_title("USGS field settlement potential proxy")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("profile y (m)")
    fig.colorbar(im, ax=ax, label="potential settlement proxy (m)")
    fig.savefig(map_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--observations", default="data/processed/usgs_geophysics_observations.npz")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--posterior-samples", type=int, default=16)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--nz", type=int, default=64)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    observations = observations_from_npz(np.load(args.observations, allow_pickle=False))
    train, holdout = split_observations_by_type(
        observations,
        holdout_fraction=args.holdout_fraction,
        seed=int(config["project"].get("seed", 42)),
    )
    metrics = evaluate_holdout_observations(train, holdout)
    metrics["train_n"] = float(train.n_obs)
    metrics["holdout_n"] = float(holdout.n_obs)
    grid = make_field_grid(train, nx=args.nx, ny=args.ny, nz=args.nz, zmax=12.0)
    recon = reconstruct_field_from_observations(train, grid=grid, n_posterior=args.posterior_samples, seed=int(config["project"].get("seed", 42)))
    pred_path = Path(config["paths"]["predictions_dir"]) / "usgs_field_reconstruction.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **recon)
    metrics_path = Path(config["paths"]["tables_dir"]) / "usgs_field_holdout_metrics.csv"
    _write_metrics(metrics_path, metrics)
    fig_path = Path(config["paths"]["figures_dir"]) / "usgs_field_reconstruction_sections.png"
    _plot_field(recon, fig_path)
    print(f"prediction={pred_path}")
    print(f"metrics={metrics_path}")
    print(f"figure={fig_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

