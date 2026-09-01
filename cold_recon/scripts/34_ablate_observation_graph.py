from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.training.train_implicit import train_implicit_model
from cold_recon.utils.config import ensure_dirs, load_config


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_graph_ablation(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["mean_iou", "eic_rmse", "temperature_rmse", "unfrozen_water_rmse"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        if metric not in df.columns:
            ax.axis("off")
            continue
        ax.bar(df["scenario"], df[metric], color=["#7f8c8d", "#4c78a8"])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _prediction_dict(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        "facies": data["facies"],
        "eic": data["eic"],
        "temperature": data["temperature"],
        "unfrozen_water": data["unfrozen_water"],
        "log_resistivity": data["log_resistivity"],
    }


def _scenario_config(config: dict, scenario: str, enabled: bool, epochs: int, batch_size: int) -> dict:
    cfg = copy.deepcopy(config)
    paths = cfg["paths"]
    graph_cfg = dict(cfg.get("observation_graph", {}))
    graph_cfg["enabled"] = bool(enabled)
    cfg["observation_graph"] = graph_cfg
    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["batch_size"] = int(batch_size)
    cfg["training"]["checkpoint"] = str(Path(paths["checkpoints_dir"]) / f"observation_graph_ablation_{scenario}.pt")
    cfg["training"]["prediction_path"] = str(Path(paths["predictions_dir"]) / f"observation_graph_ablation_{scenario}.npz")
    return cfg


def run_observation_graph_ablation(
    config: dict,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> tuple[list[dict], Path, Path]:
    ablation_cfg = config.get("observation_graph_ablation", {})
    epochs = int(epochs if epochs is not None else ablation_cfg.get("epochs", 12))
    batch_size = int(batch_size if batch_size is not None else ablation_cfg.get("batch_size", 8192))
    device = device or ablation_cfg.get("device", config["training"].get("device", "cuda"))
    sample = load_sample_npz(config["training"]["sample_path"])
    rows: list[dict] = []
    scenarios = [
        ("global_attention", False),
        ("knn_graph_attention", True),
    ]
    seed = int(config.get("project", {}).get("seed", 42))
    for scenario, enabled in scenarios:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = _scenario_config(config, scenario, enabled, epochs=epochs, batch_size=batch_size)
        result = train_implicit_model(cfg, epochs=epochs, batch_size=batch_size, device=device)
        pred = _prediction_dict(Path(result["prediction_path"]))
        metrics = synthetic_metrics(
            pred,
            sample["fields"],
            sample["grid"]["z"],
            n_facies=int(config["model"]["n_facies"]),
            ice_threshold=float(config["evaluation"]["ice_rich_threshold"]),
        )
        rows.append(
            {
                "scenario": scenario,
                "graph_enabled": bool(enabled),
                "epochs": epochs,
                "batch_size": batch_size,
                "k_neighbors": int(config.get("observation_graph", {}).get("k_neighbors", 0)) if enabled else 0,
                "checkpoint": str(result["checkpoint"]),
                "prediction_path": str(result["prediction_path"]),
                **metrics,
            }
        )
    table_path = Path(config["paths"]["tables_dir"]) / "observation_graph_ablation.csv"
    figure_path = Path(config["paths"]["figures_dir"]) / "observation_graph_ablation.png"
    _write_rows(table_path, rows)
    _plot_graph_ablation(pd.DataFrame(rows), figure_path)
    return rows, table_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    _, table_path, figure_path = run_observation_graph_ablation(
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"metrics={table_path}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
