from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.evaluation.physics_consistency import fields_from_prediction, physics_consistency_metrics
from cold_recon.evaluation.physics_refinement import PhysicsRefinementConfig, refine_posterior_dict
from cold_recon.physics.settlement import settlement_potential_numpy
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_sections import plot_truth_prediction_sections
from cold_recon.visualization.plot_settlement_risk import plot_settlement_map


def _cfg_from_args(config: dict, args: argparse.Namespace) -> PhysicsRefinementConfig:
    defaults = config.get("physics_refinement", {})
    return PhysicsRefinementConfig(
        temperature_min=float(args.temperature_min if args.temperature_min is not None else defaults.get("temperature_min", -10.0)),
        temperature_max=float(args.temperature_max if args.temperature_max is not None else defaults.get("temperature_max", 3.0)),
        heat_iterations=int(args.heat_iterations if args.heat_iterations is not None else defaults.get("heat_iterations", 16)),
        heat_strength=float(args.heat_strength if args.heat_strength is not None else defaults.get("heat_strength", 0.35)),
        heat_anchor=float(args.heat_anchor if args.heat_anchor is not None else defaults.get("heat_anchor", 0.0)),
        unfrozen_weight=float(args.unfrozen_weight if args.unfrozen_weight is not None else defaults.get("unfrozen_weight", 0.7)),
        resistivity_weight=float(args.resistivity_weight if args.resistivity_weight is not None else defaults.get("resistivity_weight", 0.4)),
        eic_max=float(defaults.get("eic_max", 0.75)),
    )


def _write_metric_row(path: Path, model_name: str, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"model": model_name, **metrics})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--posterior", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--model-name", default="COLDReconLatentDiffusionPhysicsRefined")
    parser.add_argument("--heat-iterations", type=int, default=None)
    parser.add_argument("--heat-strength", type=float, default=None)
    parser.add_argument("--heat-anchor", type=float, default=None)
    parser.add_argument("--unfrozen-weight", type=float, default=None)
    parser.add_argument("--resistivity-weight", type=float, default=None)
    parser.add_argument("--temperature-min", type=float, default=None)
    parser.add_argument("--temperature-max", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    posterior_path = Path(args.posterior or config["diffusion"]["posterior_path"])
    output_path = Path(args.output or config.get("physics_refinement", {}).get("posterior_path", "outputs/predictions/diffusion_posterior_physics_refined.npz"))
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    dz = float(sample["grid"]["dz"])
    spacing = tuple(float(x) for x in sample["grid"].get("spacing", (sample["grid"]["dx"], sample["grid"]["dy"], sample["grid"]["dz"])))
    cfg = _cfg_from_args(config, args)

    posterior_npz = np.load(posterior_path, allow_pickle=False)
    posterior = {key: posterior_npz[key] for key in posterior_npz.files}
    refined = refine_posterior_dict(posterior, n_facies=int(config["model"]["n_facies"]), cfg=cfg, dz=dz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **refined)

    pred = {
        "facies": refined["facies_mode"],
        "eic": refined["eic_mean"],
        "temperature": refined["temperature_mean"],
        "unfrozen_water": refined["unfrozen_water_mean"],
        "log_resistivity": refined["log_resistivity_mean"],
    }
    metrics = synthetic_metrics(
        pred,
        sample["fields"],
        sample["grid"]["z"],
        n_facies=int(config["model"]["n_facies"]),
        ice_threshold=float(config["evaluation"]["ice_rich_threshold"]),
    )
    metric_path = Path(config["paths"]["tables_dir"]) / "diffusion_physics_refined_metrics.csv"
    _write_metric_row(metric_path, args.model_name, metrics)

    fig_dir = Path(config["paths"]["figures_dir"])
    y_index = int(config["evaluation"]["section_y_index"])
    plot_truth_prediction_sections(
        sample["fields"],
        pred,
        fig_dir / "diffusion_physics_refined_sections.png",
        y_index,
        "Physics-refined latent diffusion posterior",
    )
    settlement = settlement_potential_numpy(refined["eic_mean"], refined["temperature_mean"] + 2.0, dz)
    plot_settlement_map(
        settlement,
        fig_dir / "diffusion_physics_refined_settlement_potential.png",
        "Physics-refined diffusion settlement potential",
    )
    physics_metrics = physics_consistency_metrics(fields_from_prediction(refined, n_facies=int(config["model"]["n_facies"])), spacing=spacing)
    print(f"posterior={output_path}")
    print(f"metrics={metric_path}")
    print(f"figures={fig_dir}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(
        "physics: "
        f"uw_mae={physics_metrics['unfrozen_water_empirical_mae']:.6f}, "
        f"rho_mae={physics_metrics['log_resistivity_empirical_mae']:.6f}, "
        f"heat_rmse={physics_metrics['heat_residual_rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
