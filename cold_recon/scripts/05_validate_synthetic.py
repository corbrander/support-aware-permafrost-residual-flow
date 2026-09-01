from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_figures_paper import make_synthetic_summary_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--prediction", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    pred_path = Path(args.prediction or config["training"]["prediction_path"])
    pred_npz = np.load(pred_path, allow_pickle=False)
    pred = {k: pred_npz[k] for k in pred_npz.files}
    metrics = synthetic_metrics(
        pred,
        sample["fields"],
        sample["grid"]["z"],
        n_facies=int(config["model"]["n_facies"]),
        ice_threshold=float(config["evaluation"]["ice_rich_threshold"]),
    )
    table_path = Path(config["paths"]["tables_dir"]) / "implicit_metrics.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"model": "COLDReconImplicit", **metrics})
    fig = make_synthetic_summary_figure(sample, pred, config["paths"]["figures_dir"], int(config["evaluation"]["section_y_index"]))
    print(f"metrics={table_path}")
    print(f"figure={fig}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

