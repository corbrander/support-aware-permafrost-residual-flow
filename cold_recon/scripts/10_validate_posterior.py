from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.physics.settlement import settlement_potential_numpy
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_sections import plot_truth_prediction_sections
from cold_recon.visualization.plot_settlement_risk import plot_settlement_map
from cold_recon.visualization.plot_uncertainty import plot_uncertainty_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--posterior", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    posterior_path = Path(args.posterior or config["diffusion"]["posterior_path"])
    data = np.load(posterior_path, allow_pickle=False)
    pred = {
        "facies": data["facies_mode"],
        "eic": data["eic_mean"],
        "temperature": data["temperature_mean"],
        "unfrozen_water": data["unfrozen_water_mean"],
        "log_resistivity": data["log_resistivity_mean"],
    }
    metrics = synthetic_metrics(
        pred,
        sample["fields"],
        sample["grid"]["z"],
        n_facies=int(config["model"]["n_facies"]),
        ice_threshold=float(config["evaluation"]["ice_rich_threshold"]),
    )
    table_path = Path(config["paths"]["tables_dir"]) / "diffusion_posterior_metrics.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"model": "COLDReconLatentDiffusion", **metrics})
    y_index = int(config["evaluation"]["section_y_index"])
    fig_dir = Path(config["paths"]["figures_dir"])
    plot_truth_prediction_sections(sample["fields"], pred, fig_dir / "diffusion_posterior_sections.png", y_index, "Latent diffusion posterior mean/mode")
    plot_uncertainty_section(data["eic_std"], fig_dir / "diffusion_eic_std_section.png", y_index, "EIC posterior std")
    plot_uncertainty_section(data["facies_entropy"], fig_dir / "diffusion_facies_entropy_section.png", y_index, "Facies posterior entropy")
    settlement = settlement_potential_numpy(data["eic_mean"], data["temperature_mean"] + 2.0, float(sample["grid"]["dz"]))
    plot_settlement_map(settlement, fig_dir / "diffusion_settlement_potential.png", "Diffusion posterior settlement potential")
    print(f"metrics={table_path}")
    print(f"figures={fig_dir}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

