from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _hierarchical_mean_ci(
    frame: pd.DataFrame,
    metric: str,
    *,
    unit: str,
    seed_column: str = "model_seed",
    seed: int,
    samples: int = 5000,
) -> tuple[float, float, float]:
    selected = frame[[seed_column, unit, metric]].dropna()
    if selected.empty:
        return float("nan"), float("nan"), float("nan")
    model_seeds = np.asarray(sorted(selected[seed_column].unique()))
    grouped = {
        value: selected.loc[selected[seed_column] == value, metric].to_numpy(
            dtype=np.float64
        )
        for value in model_seeds
    }
    rng = np.random.default_rng(int(seed))
    n_draws = int(samples)
    n_seed_slots = len(model_seeds)
    sampled_seed_indices = rng.integers(
        0, len(model_seeds), size=(n_draws, n_seed_slots)
    )
    # A duplicated seed in one hierarchical draw receives an independent
    # within-seed resample in each slot, matching the original nested loop.
    within_seed_means = np.empty(
        (n_draws, n_seed_slots, len(model_seeds)), dtype=np.float64
    )
    for slot in range(n_seed_slots):
        for seed_index, model_seed in enumerate(model_seeds):
            values = grouped[model_seed]
            positions = rng.integers(
                0, len(values), size=(n_draws, len(values))
            )
            within_seed_means[:, slot, seed_index] = values[positions].mean(
                axis=1
            )
    draw_indices = np.arange(n_draws)[:, None]
    slot_indices = np.arange(n_seed_slots)[None, :]
    draws = within_seed_means[
        draw_indices, slot_indices, sampled_seed_indices
    ].mean(axis=1)
    point = float(selected.groupby(seed_column)[metric].mean().mean())
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return point, float(lower), float(upper)


def _summarize(
    frame: pd.DataFrame,
    *,
    groups: list[str],
    metrics: Iterable[str],
    unit: str,
    base_seed: int,
    samples: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = frame.groupby(groups, dropna=False) if groups else [((), frame)]
    for group_values, subset in grouped:
        if groups:
            group_values = (
                group_values if isinstance(group_values, tuple) else (group_values,)
            )
            metadata = dict(zip(groups, group_values, strict=True))
        else:
            metadata = {}
        for metric_index, metric in enumerate(metrics):
            if metric not in subset.columns:
                continue
            point, lower, upper = _hierarchical_mean_ci(
                subset,
                metric,
                unit=unit,
                seed=base_seed + metric_index,
                samples=samples,
            )
            rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "mean": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "n_model_seeds": int(subset["model_seed"].nunique()),
                    "n_unique_units": int(subset[unit].nunique()),
                    "n_seed_unit_pairs": int(len(subset)),
                }
            )
    return pd.DataFrame(rows)


def _read_existing(paths: Iterable[Path]) -> pd.DataFrame:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in existing], ignore_index=True)


def _derive_sequential_regret_detail(sequential: pd.DataFrame) -> pd.DataFrame:
    """Derive final and trajectory-wide policy effects from complete cycles.

    Three initial boreholes plus five additions use all eight available
    candidates. Final losses therefore tie by design. Mean post-acquisition
    loss and stepwise regret retain policy-ordering information without
    selecting a favourable intermediate step post hoc.
    """

    required = {
        "model_seed",
        "scene_id",
        "policy",
        "step",
        "engineering_loss",
        "realized_decision_loss",
    }
    missing = required - set(sequential.columns)
    if missing:
        raise KeyError(f"Sequential detail is missing columns: {sorted(missing)}")
    work = sequential.copy()
    work["step_oracle_loss"] = work.groupby(
        ["model_seed", "scene_id", "step"], sort=False
    )["engineering_loss"].transform("min")
    work["stepwise_regret"] = work["engineering_loss"] - work["step_oracle_loss"]
    rows: list[dict[str, float | int | str]] = []
    for (model_seed, scene_id, policy), subset in work.groupby(
        ["model_seed", "scene_id", "policy"], sort=False
    ):
        ordered = subset.sort_values("step")
        post = ordered.loc[ordered["step"].astype(int) > 0]
        if post.empty:
            raise RuntimeError(
                f"Sequential trajectory has no acquisition steps: {model_seed}, {scene_id}, {policy}"
            )
        initial = ordered.iloc[0]
        final = ordered.iloc[-1]
        rows.append(
            {
                "model_seed": int(model_seed),
                "scene_id": str(scene_id),
                "policy": str(policy),
                "initial_loss": float(initial["engineering_loss"]),
                "final_loss": float(final["engineering_loss"]),
                "loss_reduction": float(
                    initial["engineering_loss"] - final["engineering_loss"]
                ),
                "avoided_decision_loss": float(
                    initial["realized_decision_loss"]
                    - final["realized_decision_loss"]
                ),
                "regret": float(final["stepwise_regret"]),
                "trajectory_mean_loss": float(post["engineering_loss"].mean()),
                "trajectory_loss_reduction": float(
                    initial["engineering_loss"] - post["engineering_loss"].mean()
                ),
                "mean_stepwise_regret": float(post["stepwise_regret"].mean()),
                "cumulative_regret": float(post["stepwise_regret"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="outputs/m1_support_guided/formal_postlock"
    )
    parser.add_argument(
        "--controlled-dir",
        default=(
            "outputs/m1_support_guided/"
            "formal_controlled_selected_guidance"
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/m1_support_guided/tables"
    )
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--noninferiority-margin", type=float, default=0.005)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    input_dir = Path(args.input_dir)
    controlled_dir = Path(args.controlled_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "input_dir": str(input_dir),
        "controlled_dir": str(controlled_dir),
        "model_seeds": seeds,
        "hierarchical_bootstrap_samples": int(args.bootstrap_samples),
    }

    final_id = _read_existing(
        controlled_dir / f"m1_test_id_seed{seed}_detail.csv" for seed in seeds
    )
    if not final_id.empty:
        final_id["model_seed"] = final_id["seed"].astype(int)
        final_id["observation_mode"] = "all"

    deletion_frames: list[pd.DataFrame] = []
    for mode in (
        "no_ert",
        "no_nmr",
        "no_temperature",
        "boreholes_only",
        "half_boreholes",
        "sparse_boreholes",
    ):
        current = _read_existing(
            input_dir
            / "observation_deletion"
            / f"m1_test_id_{mode}_seed{seed}_detail.csv"
            for seed in seeds
        )
        if current.empty:
            continue
        current["model_seed"] = current["seed"].astype(int)
        current["observation_mode"] = mode
        deletion_frames.append(current)
    if not final_id.empty or deletion_frames:
        deletion = pd.concat(
            ([final_id] if not final_id.empty else []) + deletion_frames,
            ignore_index=True,
        )
        deletion_metrics = (
            "eic_rmse",
            "temperature_rmse",
            "unfrozen_water_rmse",
            "log_resistivity_rmse",
            "eic_calibrated_coverage",
            "eic_calibrated_mean_width",
            "support_nrmse_borehole_eic",
            "support_nrmse_borehole_temperature",
            "support_nrmse_ert_log_resistivity",
            "support_nrmse_nmr_unfrozen_water",
            "support_nrmse_alt",
            "high_eic_t30_auprc",
            "high_eic_t30_f1",
            "high_eic_t30_object_f1",
        )
        deletion_summary = _summarize(
            deletion,
            groups=["observation_mode"],
            metrics=deletion_metrics,
            unit="scene_id",
            base_seed=11_000,
            samples=int(args.bootstrap_samples),
        )
        deletion.to_csv(
            output_dir / "m1_observation_deletion_three_seed_detail.csv",
            index=False,
        )
        deletion_summary.to_csv(
            output_dir / "m1_observation_deletion_three_seed_summary.csv",
            index=False,
        )
        metadata["observation_deletion_rows"] = int(len(deletion))

    unguided = _read_existing(
        input_dir
        / "guidance_ablation"
        / f"m1_test_id_seed{seed}_detail.csv"
        for seed in seeds
    )
    if not final_id.empty and not unguided.empty:
        unguided["model_seed"] = unguided["seed"].astype(int)
        support_metrics = [
            "support_nrmse_borehole_eic",
            "support_nrmse_borehole_temperature",
            "support_nrmse_ert_log_resistivity",
            "support_nrmse_nmr_unfrozen_water",
            "support_nrmse_alt",
        ]
        for current in (final_id, unguided):
            current["support_fidelity_score"] = np.nanmean(
                np.log1p(current[support_metrics].to_numpy(dtype=np.float64)),
                axis=1,
            )
        guided_columns = [
            "model_seed",
            "scene_id",
            "eic_rmse",
            "support_fidelity_score",
            *support_metrics,
        ]
        paired = final_id[guided_columns].merge(
            unguided[guided_columns],
            on=["model_seed", "scene_id"],
            how="inner",
            suffixes=("_guided", "_unguided"),
        )
        for metric in guided_columns[2:]:
            paired[f"difference_{metric}_guided_minus_unguided"] = (
                paired[f"{metric}_guided"] - paired[f"{metric}_unguided"]
            )
        difference_metrics = tuple(
            column for column in paired if column.startswith("difference_")
        )
        paired_summary = _summarize(
            paired,
            groups=[],
            metrics=difference_metrics,
            unit="scene_id",
            base_seed=21_000,
            samples=int(args.bootstrap_samples),
        )
        paired.to_csv(
            output_dir / "m1_guidance_ablation_paired_detail.csv", index=False
        )
        paired_summary.to_csv(
            output_dir / "m1_guidance_ablation_paired_summary.csv", index=False
        )
        metadata["guidance_ablation_matched_seed_scene_pairs"] = int(len(paired))
        indexed = paired_summary.set_index("metric")
        eic_row = indexed.loc["difference_eic_rmse_guided_minus_unguided"]
        support_row = indexed.loc[
            "difference_support_fidelity_score_guided_minus_unguided"
        ]
        metadata["guidance_acceptance_gate"] = {
            "whole_volume_eic_rmse_difference": float(eic_row["mean"]),
            "whole_volume_eic_rmse_ci95_upper": float(eic_row["ci95_upper"]),
            "whole_volume_noninferiority_margin": 0.005,
            "whole_volume_noninferiority_pass": bool(
                float(eic_row["ci95_upper"]) <= 0.005
            ),
            "support_score_difference": float(support_row["mean"]),
            "support_score_ci95_upper": float(support_row["ci95_upper"]),
            "support_improved_at_point_estimate": bool(
                float(support_row["mean"]) < 0.0
            ),
            "support_improved_with_ci95": bool(
                float(support_row["ci95_upper"]) < 0.0
            ),
        }

    noise = _read_existing(
        input_dir / "noise" / f"noise_response_seed{seed}.csv" for seed in seeds
    )
    if not noise.empty:
        noise_summary = _summarize(
            noise,
            groups=["multiplier"],
            metrics=(
                "posterior_spread_mean",
                "support_nrmse_eic",
                "mean_shift_from_nominal",
                "distant_shift_from_nominal",
                "influence_radius_m",
                "bias_gate_mean",
            ),
            unit="scene_id",
            base_seed=31_000,
            samples=int(args.bootstrap_samples),
        )
        noise.to_csv(output_dir / "m1_noise_three_seed_detail.csv", index=False)
        noise_summary.to_csv(
            output_dir / "m1_noise_three_seed_summary.csv", index=False
        )
        metadata["noise_rows"] = int(len(noise))

    sequential = _read_existing(
        input_dir
        / "sequential"
        / f"sequential_investigation_seed{seed}.csv"
        for seed in seeds
    )
    regrets = pd.DataFrame()
    if not sequential.empty:
        sequential_summary = _summarize(
            sequential,
            groups=["policy", "step"],
            metrics=(
                "engineering_loss",
                "eic_rmse",
                "high_eic_error_rate",
                "mean_interval_width",
                "expected_decision_loss",
                "realized_decision_loss",
                "false_negative_rate",
                "false_positive_rate",
            ),
            unit="scene_id",
            base_seed=41_000,
            samples=int(args.bootstrap_samples),
        )
        sequential.to_csv(
            output_dir / "m1_sequential_three_seed_detail.csv", index=False
        )
        sequential_summary.to_csv(
            output_dir / "m1_sequential_three_seed_summary.csv", index=False
        )
        regrets = _derive_sequential_regret_detail(sequential)
    if not regrets.empty:
        regret_summary = _summarize(
            regrets,
            groups=["policy"],
            metrics=(
                "initial_loss",
                "final_loss",
                "loss_reduction",
                "avoided_decision_loss",
                "regret",
                "trajectory_mean_loss",
                "trajectory_loss_reduction",
                "mean_stepwise_regret",
                "cumulative_regret",
            ),
            unit="scene_id",
            base_seed=51_000,
            samples=int(args.bootstrap_samples),
        )
        regrets.to_csv(
            output_dir / "m1_sequential_regret_three_seed_detail.csv", index=False
        )
        regret_summary.to_csv(
            output_dir / "m1_sequential_regret_three_seed_summary.csv", index=False
        )
        random_reference = regrets.loc[
            regrets["policy"].astype(str) == "random",
            [
                "model_seed",
                "scene_id",
                "trajectory_loss_reduction",
                "mean_stepwise_regret",
            ],
        ]
        comparison_rows: list[dict[str, float | int | str]] = []
        comparison_policies = sorted(
            set(regrets["policy"].astype(str)) - {"random"}
        )
        for policy_index, policy in enumerate(comparison_policies):
            current = regrets.loc[
                regrets["policy"].astype(str) == policy,
                [
                    "model_seed",
                    "scene_id",
                    "trajectory_loss_reduction",
                    "mean_stepwise_regret",
                ],
            ]
            paired = current.merge(
                random_reference,
                on=["model_seed", "scene_id"],
                suffixes=("_policy", "_random"),
                validate="one_to_one",
            )
            for metric_index, metric in enumerate(
                ("trajectory_loss_reduction", "mean_stepwise_regret")
            ):
                difference_column = f"difference_{metric}_policy_minus_random"
                paired[difference_column] = (
                    paired[f"{metric}_policy"] - paired[f"{metric}_random"]
                )
                point, lower, upper = _hierarchical_mean_ci(
                    paired,
                    difference_column,
                    unit="scene_id",
                    seed=61_000 + 100 * policy_index + metric_index,
                    samples=int(args.bootstrap_samples),
                )
                comparison_rows.append(
                    {
                        "policy": policy,
                        "reference": "random",
                        "metric": difference_column,
                        "mean": point,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "n_model_seeds": int(paired["model_seed"].nunique()),
                        "n_unique_scenes": int(paired["scene_id"].nunique()),
                        "n_seed_scene_pairs": int(len(paired)),
                    }
                )
        pd.DataFrame(comparison_rows).to_csv(
            output_dir / "m1_sequential_policy_vs_random_summary.csv",
            index=False,
        )
        metadata["sequential_regret_rows"] = int(len(regrets))
        metadata["sequential_final_tie_reason"] = (
            "Three initial plus five additions exhaust all eight candidates; "
            "trajectory metrics compare acquisition efficiency before the common endpoint."
        )

    public_sites = (
        "usgs_eic",
        "arcticdata_jago_ground_ice",
        "arcticdata_cryostratigraphy",
    )
    public_metadata: dict[str, object] = {}
    for site_index, site in enumerate(public_sites):
        public = _read_existing(
            input_dir / "public" / f"{site}_nested_loo_seed{seed}.csv"
            for seed in seeds
        )
        if public.empty:
            continue
        public["rmse_difference_vs_anchor"] = (
            public["outer_rmse"] - public["outer_anchor_rmse"]
        )
        summary = _summarize(
            public,
            groups=[],
            metrics=(
                "outer_rmse",
                "outer_anchor_rmse",
                "rmse_difference_vs_anchor",
                "outer_mae",
                "raw_coverage_90",
                "raw_width_90",
                "calibrated_coverage_90",
                "calibrated_width_90",
                "raw_crps",
                "calibrated_crps",
                "ood_abstain",
                "allow_bias",
                "inner_noninferiority_pass",
                "exact_anchor_fallback_applied",
                "fallback_due_to_noninferiority",
                "fallback_due_to_ood",
            ),
            unit="held_group_id",
            base_seed=61_000 + 100 * site_index,
            samples=int(args.bootstrap_samples),
        )
        public.to_csv(
            output_dir / f"m1_public_{site}_three_seed_detail.csv", index=False
        )
        summary.to_csv(
            output_dir / f"m1_public_{site}_three_seed_summary.csv", index=False
        )
        difference, lower, upper = _hierarchical_mean_ci(
            public,
            "rmse_difference_vs_anchor",
            unit="held_group_id",
            seed=71_000 + site_index,
            samples=int(args.bootstrap_samples),
        )
        margin = float(args.noninferiority_margin)
        public_metadata[site] = {
            "fold_seed_pairs": int(len(public)),
            "unique_boreholes": int(public["held_group_id"].nunique()),
            "mean_rmse_difference_vs_anchor": difference,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "noninferiority_margin": margin,
            "noninferiority_pass": bool(upper <= margin),
            "exact_anchor_fallback_fraction": float(
                np.mean(public["exact_anchor_fallback_applied"].astype(bool))
            ),
            "inner_noninferiority_pass_fraction": float(
                np.mean(public["inner_noninferiority_pass"].astype(bool))
            ),
            "fallback_due_to_noninferiority_fraction": float(
                np.mean(public["fallback_due_to_noninferiority"].astype(bool))
            ),
            "fallback_due_to_ood_fraction": float(
                np.mean(public["fallback_due_to_ood"].astype(bool))
            ),
        }
    metadata["public"] = public_metadata

    metadata_path = output_dir / "m1_postlock_three_seed_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
