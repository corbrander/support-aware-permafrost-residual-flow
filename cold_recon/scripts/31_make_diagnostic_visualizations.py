from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_3d_volume import plot_3d_volume_overview
from cold_recon.visualization.plot_borehole_compare import plot_borehole_profile_comparison


def _prediction(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    aliases = {
        "facies": ("facies_mode", "facies"),
        "eic": ("eic_mean", "eic"),
        "temperature": ("temperature_mean", "temperature"),
        "unfrozen_water": ("unfrozen_water_mean", "unfrozen_water"),
        "log_resistivity": ("log_resistivity_mean", "log_resistivity"),
    }
    out: dict[str, np.ndarray] = {}
    for key, names in aliases.items():
        for name in names:
            if name in data:
                out[key] = np.asarray(data[name])
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--prediction", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    pred_path = Path(args.prediction or config.get("physics_refinement", {}).get("posterior_path", config["diffusion"]["posterior_path"]))
    data = dict(np.load(pred_path, allow_pickle=False))
    pred = _prediction(data)
    fig_dir = Path(config["paths"]["figures_dir"])
    truth_volume = fig_dir / "volume_truth_3d_overview.png"
    pred_volume = fig_dir / "volume_reconstruction_3d_overview.png"
    borehole_fig = fig_dir / "borehole_profile_comparison.png"
    plot_3d_volume_overview(sample["fields"], truth_volume, grid=sample["grid"], title="Synthetic truth 3D cryostratigraphy overview")
    plot_3d_volume_overview(pred, pred_volume, grid=sample["grid"], title=f"Posterior reconstruction 3D overview: {pred_path.name}")
    plot_borehole_profile_comparison(sample, pred, borehole_fig, title=f"Borehole truth-observation-prediction comparison: {pred_path.name}")
    print(f"truth_volume={truth_volume}")
    print(f"reconstruction_volume={pred_volume}")
    print(f"borehole_comparison={borehole_fig}")


if __name__ == "__main__":
    main()
