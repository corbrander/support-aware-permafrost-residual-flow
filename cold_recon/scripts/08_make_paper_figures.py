from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cold_recon.data.data_schema import load_sample_npz
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
    pred = None
    pred_path = Path(args.prediction or config["training"]["prediction_path"])
    if pred_path.exists():
        data = np.load(pred_path, allow_pickle=False)
        pred = {k: data[k] for k in data.files}
    out = make_synthetic_summary_figure(sample, pred, config["paths"]["figures_dir"], int(config["evaluation"]["section_y_index"]))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

