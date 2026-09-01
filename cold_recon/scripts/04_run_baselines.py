from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cold_recon.baselines.idw import reconstruct_idw
from cold_recon.baselines.kriging import KrigingConfig, reconstruct_kriging
from cold_recon.baselines.random_forest import reconstruct_random_forest
from cold_recon.baselines.xgboost_ngb import GradientBoostingConfig, reconstruct_gradient_boosting
from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_sections import plot_truth_prediction_sections


def _write_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--rf-trees", type=int, default=30)
    parser.add_argument("--kriging-max-train", type=int, default=None)
    parser.add_argument("--gb-iterations", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    sample_path = Path(args.sample or config["training"]["sample_path"])
    sample = load_sample_npz(sample_path)
    pred_dir = Path(config["paths"]["predictions_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    table_dir = Path(config["paths"]["tables_dir"])
    rows: list[dict[str, float | str]] = []

    idw = reconstruct_idw(sample["observations"], sample["grid"], n_facies=int(config["model"]["n_facies"]))
    np.savez_compressed(pred_dir / "baseline_idw.npz", **idw)
    idw_metrics = synthetic_metrics(idw, sample["fields"], sample["grid"]["z"], n_facies=int(config["model"]["n_facies"]))
    rows.append({"model": "IDW", **idw_metrics})
    plot_truth_prediction_sections(sample["fields"], idw, fig_dir / "baseline_idw_sections.png", int(config["evaluation"]["section_y_index"]), "IDW baseline")

    rf = reconstruct_random_forest(sample, n_estimators=args.rf_trees, random_state=int(config["project"]["seed"]))
    np.savez_compressed(pred_dir / "baseline_random_forest.npz", **rf)
    rf_metrics = synthetic_metrics(rf, sample["fields"], sample["grid"]["z"], n_facies=int(config["model"]["n_facies"]))
    rows.append({"model": "RandomForest", **rf_metrics})
    plot_truth_prediction_sections(sample["fields"], rf, fig_dir / "baseline_random_forest_sections.png", int(config["evaluation"]["section_y_index"]), "Random Forest baseline")

    gb_cfg = config.get("baseline_gradient_boosting", {})
    gb = reconstruct_gradient_boosting(
        sample,
        n_facies=int(config["model"]["n_facies"]),
        config=GradientBoostingConfig(
            max_iter=int(args.gb_iterations or gb_cfg.get("max_iter", 180)),
            learning_rate=float(gb_cfg.get("learning_rate", 0.06)),
            max_leaf_nodes=int(gb_cfg.get("max_leaf_nodes", 31)),
            l2_regularization=float(gb_cfg.get("l2_regularization", 1e-3)),
            min_samples_leaf=int(gb_cfg.get("min_samples_leaf", 8)),
            random_state=int(config["project"]["seed"]),
        ),
    )
    np.savez_compressed(pred_dir / "baseline_gradient_boosting.npz", **gb)
    gb_metrics = synthetic_metrics(gb, sample["fields"], sample["grid"]["z"], n_facies=int(config["model"]["n_facies"]))
    rows.append({"model": "GradientBoosting", **gb_metrics})
    plot_truth_prediction_sections(sample["fields"], gb, fig_dir / "baseline_gradient_boosting_sections.png", int(config["evaluation"]["section_y_index"]), "Gradient boosting baseline")

    kriging_cfg = config.get("baseline_kriging", {})
    kriging = reconstruct_kriging(
        sample["observations"],
        sample["grid"],
        n_facies=int(config["model"]["n_facies"]),
        config=KrigingConfig(
            length_scale_xyz=tuple(float(x) for x in kriging_cfg.get("length_scale_xyz", (0.22, 0.22, 0.35))),
            signal_variance=float(kriging_cfg.get("signal_variance", 1.0)),
            nugget=float(kriging_cfg.get("nugget", 0.03)),
            max_train_points=int(args.kriging_max_train or kriging_cfg.get("max_train_points", 512)),
            chunk_size=int(kriging_cfg.get("chunk_size", 32768)),
            random_state=int(config["project"]["seed"]),
        ),
    )
    np.savez_compressed(pred_dir / "baseline_kriging.npz", **kriging)
    kriging_metrics = synthetic_metrics(kriging, sample["fields"], sample["grid"]["z"], n_facies=int(config["model"]["n_facies"]))
    rows.append({"model": "KrigingGPR", **kriging_metrics})
    plot_truth_prediction_sections(sample["fields"], kriging, fig_dir / "baseline_kriging_sections.png", int(config["evaluation"]["section_y_index"]), "Kriging/GPR baseline")

    out = table_dir / "baseline_metrics.csv"
    _write_metrics(out, rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
