from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.aggregate_m1_seed_results import _hierarchical_bootstrap


COMMON_METRICS = (
    "eic_rmse",
    "temperature_rmse",
    "unfrozen_water_rmse",
    "log_resistivity_rmse",
    "eic_coverage",
    "eic_mean_width",
    "eic_crps",
    "eic_energy_score",
    "support_nrmse_borehole_eic",
)


def _read(path: Path, seed: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if seed is not None and "seed" not in frame:
        frame["seed"] = int(seed)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument(
        "--ablation-dir", default="outputs/m1_support_guided/formal_ablation"
    )
    parser.add_argument(
        "--final-dir",
        default=(
            "outputs/m1_support_guided/"
            "formal_controlled_selected_guidance"
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/m1_support_guided/tables"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    ablation_dir = Path(args.ablation_dir)
    final_dir = Path(args.final_dir)
    frames: list[pd.DataFrame] = []

    final_frames = [
        _read(final_dir / f"m1_test_id_seed{seed}_detail.csv") for seed in seeds
    ]
    a11 = pd.concat(final_frames, ignore_index=True)
    a11["ablation_id"] = "A11"
    a11["state_layout"] = "factorized"
    frames.append(a11)

    a0 = final_frames[0][
        ["scene_id", "generator_family", "anchor_eic_rmse"]
    ].copy()
    a0 = a0.rename(columns={"anchor_eic_rmse": "eic_rmse"})
    a0["seed"] = 0
    a0["ablation_id"] = "A0"
    a0["state_layout"] = "tree_prior"
    frames.append(a0)

    geostat_paths = (
        (430, Path("outputs/m1_support_guided/formal_geostat_bounded_calibrated_seed430/m1_geostatistical_test_id_detail.csv")),
        (431, Path("outputs/m1_support_guided/formal_geostat_bounded_calibrated_seed431/m1_geostatistical_test_id_detail.csv")),
        (432, Path("outputs/m1_support_guided/formal_geostat_bounded_calibrated_seed432/m1_geostatistical_test_id_detail.csv")),
    )
    geostat = pd.concat(
        [_read(path, seed=seed) for seed, path in geostat_paths],
        ignore_index=True,
    )
    geostat["ablation_id"] = "A1"
    geostat["state_layout"] = "gaussian_eic"
    frames.append(geostat)

    for ablation_id in ("A2", "A3", "A4", "A5", "A6"):
        paths = [
            ablation_dir
            / ablation_id.lower()
            / f"m1_{ablation_id.lower()}_test_id_seed{seed}_detail.csv"
            for seed in seeds
        ]
        frame = pd.concat([_read(path) for path in paths], ignore_index=True)
        frame["ablation_id"] = ablation_id
        frame["state_layout"] = "mixed_7_class"
        frames.append(frame)
    for ablation_id in ("A7", "A8", "A9", "A10"):
        paths = [
            ablation_dir
            / ablation_id.lower()
            / f"m1_test_id_seed{seed}_detail.csv"
            for seed in seeds
        ]
        frame = pd.concat([_read(path) for path in paths], ignore_index=True)
        frame["ablation_id"] = ablation_id
        frame["state_layout"] = "factorized"
        frames.append(frame)

    detail = pd.concat(frames, ignore_index=True, sort=False)
    anchor = a0[["scene_id", "eic_rmse"]].rename(
        columns={"eic_rmse": "tree_prior_eic_rmse"}
    )
    detail = detail.merge(anchor, on="scene_id", how="left")
    detail["eic_rmse_difference_vs_tree"] = (
        detail["eic_rmse"] - detail["tree_prior_eic_rmse"]
    )
    summary_rows: list[dict] = []
    for ablation_index, (ablation_id, subset) in enumerate(
        detail.groupby("ablation_id", sort=True)
    ):
        metrics = [
            metric
            for metric in (*COMMON_METRICS, "eic_rmse_difference_vs_tree")
            if metric in subset and subset[metric].notna().any()
        ]
        for metric_index, metric in enumerate(metrics):
            point, lower, upper = _hierarchical_bootstrap(
                subset,
                metric,
                seed=80_000 + 100 * ablation_index + metric_index,
                samples=int(args.bootstrap_samples),
            )
            summary_rows.append(
                {
                    "ablation_id": ablation_id,
                    "state_layout": str(subset["state_layout"].iloc[0]),
                    "metric": metric,
                    "mean": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "n_seeds": int(subset["seed"].nunique()),
                    "n_unique_scenes": int(subset["scene_id"].nunique()),
                    "n_seed_scene_pairs": int(len(subset)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "m1_a0_a11_ablation_detail.csv"
    summary_path = output_dir / "m1_a0_a11_ablation_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "seeds": seeds,
        "hierarchical_bootstrap_samples": int(args.bootstrap_samples),
        "a4_weight_source": (
            "A3 checkpoint with covariance-aware support guidance; no separate retraining"
        ),
        "a10_weight_source": (
            "A9 checkpoint with stepwise likelihood guidance; no separate retraining"
        ),
        "deterministic_a0_seed_count": 1,
        "a1_primary_baseline": (
            "support-conditioned Gaussian random field with physical-bound "
            "anomaly contraction; unconstrained Gaussian retained as sensitivity"
        ),
        "detail": str(detail_path),
        "summary": str(summary_path),
    }
    metadata_path = output_dir / "m1_a0_a11_ablation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
