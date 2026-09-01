from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.observation_consistency import observation_consistency_table
from cold_recon.utils.config import ensure_dirs, load_config


def _prediction_specs(config: dict) -> list[tuple[str, Path]]:
    pred_dir = Path(config["paths"]["predictions_dir"])
    return [
        ("truth", Path(config["training"]["sample_path"])),
        ("IDW", pred_dir / "baseline_idw.npz"),
        ("RandomForest", pred_dir / "baseline_random_forest.npz"),
        ("GradientBoosting", pred_dir / "baseline_gradient_boosting.npz"),
        ("KrigingGPR", pred_dir / "baseline_kriging.npz"),
        ("SparseUNet3D", pred_dir / "baseline_unet3d.npz"),
        ("COLDReconImplicit", pred_dir / "implicit_prediction.npz"),
        ("COLDReconLatentDiffusion", pred_dir / "diffusion_posterior.npz"),
        ("COLDReconFNOOperatorDiffusion", pred_dir / "fno_operator_diffusion_posterior.npz"),
        ("COLDReconRectifiedFlow", pred_dir / "rectified_flow_posterior.npz"),
        ("COLDReconLatentDiffusionPhysicsTrained", pred_dir / "diffusion_posterior_physics_trained.npz"),
        ("COLDReconLatentDiffusionPhysicsGuided", pred_dir / "diffusion_posterior_physics_guided.npz"),
        ("COLDReconLatentDiffusionPhysicsRefined", pred_dir / "diffusion_posterior_physics_refined.npz"),
    ]


def _load_prediction(path: Path, sample: dict) -> dict[str, np.ndarray]:
    if path == Path("truth") or path == Path(sample.get("metadata", {}).get("path", "")):
        return sample["fields"]
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _plot_consistency(table, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    continuous = table[table["rmse"].notna()].copy() if "rmse" in table.columns else table.iloc[0:0].copy()
    facies = table[table["source"] == "borehole_facies"].copy() if "source" in table.columns else table.iloc[0:0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if not continuous.empty:
        pivot = continuous.pivot(index="model", columns="source", values="rmse")
        x = np.arange(len(pivot.index))
        width = 0.8 / max(len(pivot.columns), 1)
        for idx, source in enumerate(pivot.columns):
            axes[0].bar(x + idx * width, pivot[source].to_numpy(), width=width, label=source)
        axes[0].set_xticks(x + width * (len(pivot.columns) - 1) / 2)
        axes[0].set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=8)
        axes[0].set_ylabel("RMSE at observation locations")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, axis="y", color="0.9")
    axes[0].set_title("Continuous observation consistency")
    if not facies.empty and "accuracy" in facies.columns:
        axes[1].bar(facies["model"], facies["accuracy"], color="#4c78a8")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_ylabel("accuracy")
        axes[1].tick_params(axis="x", rotation=35, labelsize=8)
        axes[1].grid(True, axis="y", color="0.9")
    axes[1].set_title("Borehole facies consistency")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(config["training"]["sample_path"])
    predictions = []
    for model_name, path in _prediction_specs(config):
        if model_name == "truth":
            predictions.append((model_name, sample["fields"]))
        elif path.exists():
            predictions.append((model_name, _load_prediction(path, sample)))
    table = observation_consistency_table(predictions, sample)
    table_path = Path(config["paths"]["tables_dir"]) / "synthetic_observation_consistency.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    fig_path = Path(config["paths"]["figures_dir"]) / "synthetic_observation_consistency.png"
    _plot_consistency(table, fig_path)
    print(f"table={table_path}")
    print(f"figure={fig_path}")
    print(f"rows={len(table)}")


if __name__ == "__main__":
    main()
