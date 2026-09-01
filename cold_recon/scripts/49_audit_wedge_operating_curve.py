from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cold_recon.evaluation.wedge_operating_curve import build_wedge_operating_audit
from cold_recon.utils.config import ensure_dirs, load_config


def _parse_thresholds(value: str | None) -> np.ndarray | None:
    if value is None or not str(value).strip():
        return np.linspace(0.0, 0.95, 20, dtype=float)
    return np.asarray([float(part.strip()) for part in str(value).split(",") if part.strip()], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--predictions", default="outputs/tables/arcticdata_conditioned_diffusion_multisite_predictions.csv")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    outputs = build_wedge_operating_audit(
        predictions_path=Path(args.predictions),
        prediction_dir=Path(config["paths"]["predictions_dir"]),
        table_dir=Path(config["paths"]["tables_dir"]),
        figure_dir=Path(config["paths"]["figures_dir"]),
        thresholds=_parse_thresholds(args.thresholds),
    )
    for key, path in outputs.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
