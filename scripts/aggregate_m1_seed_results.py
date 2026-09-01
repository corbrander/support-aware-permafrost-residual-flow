from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTIFIERS = {"scene_id", "split", "generator_family", "seed"}


def _zero_fill_defined_object_f1(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for threshold in (20, 30, 40):
        f1_column = f"high_eic_t{threshold}_object_f1"
        truth_column = f"high_eic_t{threshold}_truth_object_count"
        if {f1_column, truth_column}.issubset(frame.columns):
            zero_match = frame[f1_column].isna() & frame[truth_column].gt(0)
            frame.loc[zero_match, f1_column] = 0.0
    return frame


def _hierarchical_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    *,
    seed: int,
    samples: int = 5000,
) -> tuple[float, float, float]:
    values = frame[["seed", "scene_id", metric]].dropna()
    if values.empty:
        return float("nan"), float("nan"), float("nan")
    seeds = np.asarray(sorted(values["seed"].unique()))
    rng = np.random.default_rng(int(seed))
    sample_count = int(samples)
    seed_count = int(len(seeds))
    grouped = {
        value: values.loc[values["seed"] == value, metric].to_numpy(
            dtype=np.float64
        )
        for value in seeds
    }
    sampled_seed_indices = rng.integers(
        0, seed_count, size=(sample_count, seed_count)
    )
    slot_means = np.empty((sample_count, seed_count), dtype=np.float64)
    # Each resampled seed occurrence receives an independent scene bootstrap,
    # matching the original two-level algorithm without pandas row sampling in
    # the inner loop.
    for slot in range(seed_count):
        slot_means[:, slot] = 0.0
        for seed_index, seed_value in enumerate(seeds):
            array = grouped[seed_value]
            draws = rng.integers(
                0, len(array), size=(sample_count, len(array))
            )
            resampled_means = array[draws].mean(axis=1)
            selected = sampled_seed_indices[:, slot] == seed_index
            slot_means[selected, slot] = resampled_means[selected]
    estimates = slot_means.mean(axis=1)
    point = float(values.groupby("seed")[metric].mean().mean())
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return point, float(lower), float(upper)


def _pair_geostatistical_replicates(
    model: pd.DataFrame,
    baselines: list[pd.DataFrame],
) -> pd.DataFrame:
    """Pair independent baseline replicates without a Cartesian seed product."""

    if not baselines:
        return pd.DataFrame()
    geostat_frames: list[pd.DataFrame] = []
    for replicate_index, baseline in enumerate(baselines):
        current = _zero_fill_defined_object_f1(baseline)
        current["_baseline_replicate"] = int(replicate_index)
        geostat_frames.append(current)
    geostat = pd.concat(geostat_frames, ignore_index=True)
    model_for_comparison = model.copy()
    model_seed_order = {
        value: index
        for index, value in enumerate(sorted(model_for_comparison["seed"].unique()))
    }
    model_for_comparison["_model_replicate"] = model_for_comparison[
        "seed"
    ].map(model_seed_order)
    if len(baselines) == 1:
        comparison = model_for_comparison.merge(
            geostat,
            on="scene_id",
            how="inner",
            suffixes=("_model", "_geostat"),
        )
    else:
        comparison = model_for_comparison.merge(
            geostat,
            left_on=["scene_id", "_model_replicate"],
            right_on=["scene_id", "_baseline_replicate"],
            how="inner",
            suffixes=("_model", "_geostat"),
        )
    if "seed_model" in comparison:
        comparison["seed"] = comparison["seed_model"]
    elif "seed" not in comparison:
        comparison["seed"] = comparison["_model_replicate"]
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--output-dir", default="outputs/m1_support_guided/tables"
    )
    parser.add_argument("--name", default="m1_test_id_three_seed")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=880)
    parser.add_argument("--noninferiority-margin", type=float, default=0.005)
    parser.add_argument(
        "--geostat-detail",
        default=(
            "outputs/m1_support_guided/tables/"
            "m1_geostatistical_test_id_detail.csv"
        ),
    )
    parser.add_argument(
        "--geostat-details",
        nargs="+",
        default=[],
        help=(
            "Independent geostatistical detail files. When multiple files are "
            "supplied, their sorted replicate order is paired with the sorted "
            "model-seed order instead of forming a Cartesian seed product."
        ),
    )
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    frame = _zero_fill_defined_object_f1(pd.concat(frames, ignore_index=True))
    if frame["seed"].nunique() < 2:
        raise ValueError("seed aggregation requires at least two independent seeds")
    if {"eic_rmse", "anchor_eic_rmse"}.issubset(frame.columns):
        frame["eic_rmse_improvement_over_anchor"] = (
            frame["anchor_eic_rmse"] - frame["eic_rmse"]
        )
        frame["eic_rmse_difference_vs_anchor"] = (
            frame["eic_rmse"] - frame["anchor_eic_rmse"]
        )
    numeric = [
        column
        for column in frame.columns
        if column not in IDENTIFIERS and pd.api.types.is_numeric_dtype(frame[column])
    ]

    def summarize(subset: pd.DataFrame, family: str) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for metric_index, metric in enumerate(numeric):
            point, lower, upper = _hierarchical_bootstrap(
                subset,
                metric,
                seed=int(args.seed) + metric_index,
                samples=int(args.bootstrap_samples),
            )
            rows.append(
                {
                    "generator_family": family,
                    "metric": metric,
                    "mean": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "n_seeds": int(subset["seed"].nunique()),
                    "n_scene_seed_pairs": int(len(subset)),
                    "n_unique_scenes": int(subset["scene_id"].nunique()),
                }
            )
        return rows

    summary_rows = summarize(frame, "all")
    for family, subset in frame.groupby("generator_family"):
        summary_rows.extend(summarize(subset, str(family)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{args.name}_detail.csv"
    summary_path = output_dir / f"{args.name}_summary.csv"
    frame.to_csv(combined_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metadata = {
        "inputs": [str(path) for path in args.inputs],
        "seeds": [int(value) for value in sorted(frame["seed"].unique())],
        "hierarchical_bootstrap_samples": int(args.bootstrap_samples),
        "detail": str(combined_path),
        "summary": str(summary_path),
    }
    if "eic_rmse_difference_vs_anchor" in frame.columns:
        difference, lower, upper = _hierarchical_bootstrap(
            frame,
            "eic_rmse_difference_vs_anchor",
            seed=int(args.seed) + 100_000,
            samples=int(args.bootstrap_samples),
        )
        margin = float(args.noninferiority_margin)
        metadata["paired_eic_noninferiority"] = {
            "estimand": "model EIC RMSE minus tree-anchor EIC RMSE",
            "mean_difference": difference,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "margin": margin,
            "pass": bool(upper <= margin),
            "model_improved_scene_seed_fraction": float(
                np.mean(frame["eic_rmse_difference_vs_anchor"] < 0.0)
            ),
        }
    requested_geostat_paths = (
        [Path(value) for value in args.geostat_details]
        if args.geostat_details
        else ([Path(args.geostat_detail)] if args.geostat_detail else [])
    )
    geostat_paths = [path for path in requested_geostat_paths if path.exists()]
    if geostat_paths:
        baseline_frames: list[pd.DataFrame] = []
        for replicate_index, path in enumerate(geostat_paths):
            current = pd.read_csv(path)
            if "seed" not in current:
                current["seed"] = int(replicate_index)
            baseline_frames.append(current)
        baseline_detail = pd.concat(baseline_frames, ignore_index=True)
        baseline_metrics = (
            "eic_rmse",
            "anchor_eic_rmse",
            "support_nrmse_borehole_eic",
            "invalid_eic_sample_fraction",
            "raw_gaussian_invalid_eic_sample_fraction",
            "eic_coverage",
            "eic_mean_width",
            "eic_crps",
            "eic_energy_score",
            "eic_pit_mean",
            "eic_pit_variance",
            "eic_calibrated_coverage",
            "eic_calibrated_mean_width",
            "high_eic_object_f1",
        )
        baseline_rows: list[dict[str, object]] = []
        for metric_index, metric in enumerate(baseline_metrics):
            if metric not in baseline_detail:
                continue
            point, lower, upper = _hierarchical_bootstrap(
                baseline_detail,
                metric,
                seed=int(args.seed) + 150_000 + metric_index,
                samples=int(args.bootstrap_samples),
            )
            baseline_rows.append(
                {
                    "metric": metric,
                    "mean": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "n_seeds": int(baseline_detail["seed"].nunique()),
                    "n_scene_seed_pairs": int(len(baseline_detail)),
                    "n_unique_scenes": int(
                        baseline_detail["scene_id"].nunique()
                    ),
                }
            )
        baseline_detail_path = (
            output_dir / "m1_geostat_bounded_calibrated_three_seed_detail.csv"
        )
        baseline_summary_path = (
            output_dir / "m1_geostat_bounded_calibrated_three_seed_summary.csv"
        )
        baseline_detail.to_csv(baseline_detail_path, index=False)
        pd.DataFrame(baseline_rows).to_csv(baseline_summary_path, index=False)
        comparison = _pair_geostatistical_replicates(
            frame, baseline_frames
        )
        comparisons: dict[str, object] = {
            "baseline_details": [str(path) for path in geostat_paths],
            "baseline_aggregate_detail": str(baseline_detail_path),
            "baseline_aggregate_summary": str(baseline_summary_path),
            "seed_pairing": (
                "one shared baseline replicate per model scene"
                if len(geostat_paths) == 1
                else "sorted one-to-one model and baseline replicate pairing"
            ),
            "matched_seed_scene_pairs": int(len(comparison)),
            "matched_scenes": int(comparison["scene_id"].nunique()),
        }
        metric_pairs = (
            ("eic_rmse", "eic_rmse", "lower_is_better"),
            ("eic_crps", "eic_crps", "lower_is_better"),
            ("eic_energy_score", "eic_energy_score", "lower_is_better"),
            (
                "support_nrmse_borehole_eic",
                "support_nrmse_borehole_eic",
                "lower_is_better",
            ),
            (
                "high_eic_t30_object_f1",
                "high_eic_object_f1",
                "higher_is_better",
            ),
        )
        for metric_index, (model_metric, baseline_metric, direction) in enumerate(
            metric_pairs
        ):
            model_column = (
                f"{model_metric}_model"
                if f"{model_metric}_model" in comparison
                else model_metric
            )
            geostat_column = (
                f"{baseline_metric}_geostat"
                if f"{baseline_metric}_geostat" in comparison
                else baseline_metric
            )
            if model_column not in comparison or geostat_column not in comparison:
                continue
            difference_column = f"difference_{model_metric}_vs_geostat"
            comparison[difference_column] = (
                comparison[model_column] - comparison[geostat_column]
            )
            point, lower, upper = _hierarchical_bootstrap(
                comparison,
                difference_column,
                seed=int(args.seed) + 200_000 + metric_index,
                samples=int(args.bootstrap_samples),
            )
            comparisons[model_metric] = {
                "difference_model_minus_geostat": point,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "direction": direction,
            }
        comparison.to_csv(
            output_dir / f"{args.name}_paired_geostat_detail.csv", index=False
        )
        metadata["paired_geostatistical_comparison"] = comparisons
    metadata_path = output_dir / f"{args.name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
