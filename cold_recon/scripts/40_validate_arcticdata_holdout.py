from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cold_recon.data.arcticdata_cryostratigraphy_loader import write_arcticdata_cryostratigraphy_inventory
from cold_recon.evaluation.arcticdata_validation import ArcticDataValidationConfig, arcticdata_holdout_validation_tables
from cold_recon.utils.config import ensure_dirs, load_config


def _plot_validation(eic_metrics: pd.DataFrame, facies_metrics: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(eic_metrics["model"], eic_metrics["rmse"], color="#4c78a8")
    axes[0].set_ylabel("EIC RMSE")
    axes[0].set_title("ArcticData EIC holdout")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(facies_metrics["model"], facies_metrics["wedge_ice_recall"], color="#f58518")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("wedge-ice recall")
    axes[1].set_title("ArcticData cryofacies holdout")
    axes[1].tick_params(axis="x", rotation=35)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    paths = config["paths"]
    inventory_path = Path(paths["processed_dir"]) / "arcticdata_cryostratigraphy_inventory.csv"
    if not inventory_path.exists():
        write_arcticdata_cryostratigraphy_inventory(config)
    inventory = pd.read_csv(inventory_path)
    tables = arcticdata_holdout_validation_tables(inventory, ArcticDataValidationConfig())

    table_dir = Path(paths["tables_dir"])
    fig_dir = Path(paths["figures_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "intervals": table_dir / "arcticdata_validation_intervals.csv",
        "eic_predictions": table_dir / "arcticdata_eic_holdout_predictions.csv",
        "eic_metrics": table_dir / "arcticdata_eic_holdout_metrics.csv",
        "facies_predictions": table_dir / "arcticdata_facies_holdout_predictions.csv",
        "facies_metrics": table_dir / "arcticdata_facies_holdout_metrics.csv",
    }
    for key, path in outputs.items():
        tables[key].to_csv(path, index=False)
        print(f"{key}={path}")
    fig_path = fig_dir / "arcticdata_external_holdout_validation.png"
    _plot_validation(tables["eic_metrics"], tables["facies_metrics"], fig_path)
    print(f"figure={fig_path}")


if __name__ == "__main__":
    main()
