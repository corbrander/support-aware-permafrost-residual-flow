from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "engineering_geology_manuscript"
DIRECT = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_direct_support_ablation"
    / "m1_direct_support_ablation_summary.csv"
)
TREE_DETAIL = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_probabilistic_extra_trees"
    / "m1_probabilistic_extra_trees_test_id_detail.csv"
)
TREE_RESPONSE = TREE_DETAIL.with_name(
    "m1_probabilistic_extra_trees_test_id_response.csv"
)
FLOW_DETAIL = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_engineering_response_seed41"
    / "m1_test_id_seed41_detail.csv"
)
FLOW_RESPONSE = FLOW_DETAIL.with_name("m1_test_id_seed41_engineering_response.csv")
OUTPUT = PAPER / "M1_table8_support_and_probability_tree.md"


def _paired_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = values[
        rng.integers(0, values.size, size=(5000, values.size))
    ].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def _number(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _ci_text(mean: float, low: float, high: float, digits: int = 4) -> str:
    return f"{_number(mean, digits)} ({_number(low, digits)} to {_number(high, digits)})"


def _direct_metric(frame: pd.DataFrame, metric: str) -> pd.Series:
    rows = frame.loc[frame["metric"] == metric]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one direct-support row for {metric!r}, found {len(rows)}")
    return rows.iloc[0]


def _direct_row(
    frame: pd.DataFrame,
    outcome: str,
    metric: str,
    interpretation: str,
    *,
    digits: int = 4,
) -> list[str]:
    row = _direct_metric(frame, metric)
    return [
        "Support representation",
        outcome,
        _ci_text(
            row["support_aware_mean"],
            row["support_aware_ci95_lower"],
            row["support_aware_ci95_upper"],
            digits,
        ),
        _ci_text(
            row["nearest_voxel_mean"],
            row["nearest_voxel_ci95_lower"],
            row["nearest_voxel_ci95_upper"],
            digits,
        ),
        _ci_text(
            row["paired_support_minus_nearest"],
            row["paired_ci95_lower"],
            row["paired_ci95_upper"],
            digits,
        ),
        interpretation,
    ]


def _direct_support_bundle(
    frame: pd.DataFrame,
    outcome: str,
    nrmse_metric: str,
) -> list[str]:
    nrmse = _direct_metric(frame, nrmse_metric)
    bias = _direct_metric(frame, nrmse_metric.replace("NRMSE", "standardized bias"))
    collapsed = _direct_metric(
        frame,
        nrmse_metric.replace("support NRMSE", "collapsed-voxel NRMSE"),
    )

    def bundle(prefix: str) -> str:
        return "; ".join(
            [
                _number(nrmse[f"{prefix}_mean"], 3),
                _number(bias[f"{prefix}_mean"], 3),
                _number(collapsed[f"{prefix}_mean"], 3),
            ]
        )

    return [
        "Support representation",
        f"{outcome}: original NRMSE; standardized bias; collapsed NRMSE",
        bundle("support_aware"),
        bundle("nearest_voxel"),
        _ci_text(
            nrmse["paired_support_minus_nearest"],
            nrmse["paired_ci95_lower"],
            nrmse["paired_ci95_upper"],
            3,
        ),
        "The paired interval is for original-support NRMSE; Figure 16 gives uncertainty for all displayed support metrics.",
    ]


def _direct_two_metric_row(
    frame: pd.DataFrame,
    outcome: str,
    metric_a: str,
    metric_b: str,
    interpretation: str,
    *,
    digits: int = 4,
) -> list[str]:
    first = _direct_metric(frame, metric_a)
    second = _direct_metric(frame, metric_b)

    def method_values(prefix: str) -> str:
        return "; ".join(
            [
                _number(first[f"{prefix}_mean"], digits),
                _number(second[f"{prefix}_mean"], digits),
            ]
        )

    return [
        "Support representation",
        outcome,
        method_values("support_aware"),
        method_values("nearest_voxel"),
        "; ".join(
            [
                _ci_text(
                    first["paired_support_minus_nearest"],
                    first["paired_ci95_lower"],
                    first["paired_ci95_upper"],
                    digits,
                ),
                _ci_text(
                    second["paired_support_minus_nearest"],
                    second["paired_ci95_lower"],
                    second["paired_ci95_upper"],
                    digits,
                ),
            ]
        ),
        interpretation,
    ]


def _tree_row(
    paired: pd.DataFrame,
    outcome: str,
    flow_column: str,
    tree_column: str,
    interpretation: str,
    seed: int,
    *,
    digits: int = 4,
) -> list[str]:
    flow_values = paired[flow_column].to_numpy(dtype=float)
    tree_values = paired[tree_column].to_numpy(dtype=float)
    flow_mean, flow_low, flow_high = _paired_ci(flow_values, seed + 1)
    tree_mean, tree_low, tree_high = _paired_ci(tree_values, seed + 2)
    diff_mean, diff_low, diff_high = _paired_ci(tree_values - flow_values, seed + 3)
    return [
        "Probability tree",
        outcome,
        _ci_text(flow_mean, flow_low, flow_high, digits),
        _ci_text(tree_mean, tree_low, tree_high, digits),
        _ci_text(diff_mean, diff_low, diff_high, digits),
        interpretation,
    ]


def _tree_two_metric_row(
    paired: pd.DataFrame,
    outcome: str,
    flow_a: str,
    tree_a: str,
    flow_b: str,
    tree_b: str,
    interpretation: str,
    seed: int,
    *,
    digits: int = 4,
) -> list[str]:
    values = []
    differences = []
    for offset, (flow_column, tree_column) in enumerate(
        ((flow_a, tree_a), (flow_b, tree_b))
    ):
        flow_values = paired[flow_column].to_numpy(dtype=float)
        tree_values = paired[tree_column].to_numpy(dtype=float)
        values.append(
            (
                float(np.nanmean(flow_values)),
                float(np.nanmean(tree_values)),
            )
        )
        differences.append(_paired_ci(tree_values - flow_values, seed + offset))
    return [
        "Probability tree",
        outcome,
        "; ".join(_number(item[0], digits) for item in values),
        "; ".join(_number(item[1], digits) for item in values),
        "; ".join(_ci_text(*item, digits=digits) for item in differences),
        interpretation,
    ]


def main() -> None:
    direct = pd.read_csv(DIRECT)
    if direct.empty or set(direct["n_scenes"].astype(int)) != {100}:
        raise RuntimeError("Table 8 requires a complete 100-scene direct-support summary")
    rows: list[list[str]] = []
    rows.append(
        _direct_row(
            direct,
            "Whole-volume EIC RMSE",
            "Whole-volume EIC RMSE",
            "Negative differences favour explicit supports.",
            digits=5,
        )
    )
    support_specs = [
        ("EIC interval", "EIC interval support NRMSE"),
        ("Temperature interval", "Temperature interval support NRMSE"),
        ("ERT volume", "ERT volume support NRMSE"),
        ("NMR kernel", "NMR kernel support NRMSE"),
        ("ALT crossing", "ALT crossing support NRMSE"),
    ]
    for outcome, metric in support_specs:
        rows.append(_direct_support_bundle(direct, outcome, metric))
    rows.append(
        _direct_two_metric_row(
            direct,
            "Raw coverage; width",
            "Raw EIC coverage",
            "Raw EIC width",
            "Native calibration and sharpness, in that order.",
        )
    )
    rows.append(
        _direct_two_metric_row(
            direct,
            "Calibrated coverage; width",
            "Calibrated EIC coverage",
            "Calibrated EIC width",
            "Validation-only spatial block conformal and its width cost.",
        )
    )
    rows.append(
        _direct_two_metric_row(
            direct,
            "Voxel F1; object F1 at EIC 0.30",
            "High-EIC voxel F1 at 0.30",
            "High-EIC object F1 at 0.30",
            "Rare-structure voxel and individual-object retention.",
        )
    )

    flow = pd.read_csv(FLOW_DETAIL)
    tree = pd.read_csv(TREE_DETAIL)
    if len(flow) != 100 or len(tree) != 100:
        raise RuntimeError("Table 8 requires complete 100-scene flow and probability-tree details")
    paired = flow.merge(tree, on="scene_id", suffixes=("_flow", "_tree"), validate="one_to_one")
    if len(paired) != 100:
        raise RuntimeError(
            "Flow and probability-tree details do not contain identical scene IDs; "
            f"matched {len(paired)}"
        )
    tree_specs = [
        (
            "Whole-volume EIC RMSE",
            "eic_rmse_flow",
            "eic_rmse_tree",
            "Positive tree-minus-flow differences favour the flow mean.",
        ),
        (
            "EIC CRPS",
            "eic_crps_flow",
            "eic_crps_tree",
            "Negative tree-minus-flow differences favour the probability tree.",
        ),
        (
            "Borehole-EIC support NRMSE",
            "support_nrmse_borehole_eic_flow",
            "support_nrmse_borehole_eic_tree",
            "The paired difference is interpreted with its confidence interval.",
        ),
    ]
    for index, spec in enumerate(tree_specs):
        rows.append(_tree_row(paired, *spec, seed=4100 + 10 * index))
    rows.append(
        _tree_two_metric_row(
            paired,
            "Raw coverage; width",
            "eic_coverage_flow",
            "eic_coverage_tree",
            "eic_mean_width_flow",
            "eic_mean_width_tree",
            "The flow is natively under-dispersed; width is read with coverage.",
            seed=4200,
        )
    )
    rows.append(
        _tree_two_metric_row(
            paired,
            "Calibrated coverage; width",
            "eic_calibrated_coverage_flow",
            "eic_calibrated_coverage_tree",
            "eic_calibrated_mean_width_flow",
            "eic_calibrated_mean_width_tree",
            "Both use validation-only spatial calibration.",
            seed=4210,
        )
    )
    flow_response = pd.read_csv(FLOW_RESPONSE)
    flow_response = flow_response.loc[
        flow_response["method"] == "Conditional residual flow"
    ].copy()
    tree_response = pd.read_csv(TREE_RESPONSE)
    for index, depth in enumerate((2.0, 4.0, 6.0)):
        left = flow_response.loc[
            flow_response["thaw_depth_m"] == depth,
            ["scene_id", "response_rmse_m"],
        ]
        right = tree_response.loc[
            tree_response["thaw_depth_m"] == depth,
            ["scene_id", "response_rmse_m"],
        ]
        response = left.merge(
            right,
            on="scene_id",
            suffixes=("_flow", "_tree"),
            validate="one_to_one",
        )
        if len(response) != 100:
            raise RuntimeError(
                f"Response comparison at {depth:.0f} m matched {len(response)} rather than 100 scenes"
            )
        rows.append(
            _tree_row(
                response,
                f"Response RMSE at {depth:.0f} m",
                "response_rmse_m_flow",
                "response_rmse_m_tree",
                "No resolved response-RMSE superiority if the paired interval crosses zero.",
                seed=4300 + index * 10,
                digits=4,
            )
        )

    header = [
        "Comparison",
        "Outcome",
        "Support-aware flow",
        "Comparator",
        "Paired difference (95% CI)",
        "Interpretation",
    ]
    lines = [
        "# Table 8. Direct support-representation ablation and strong probability-tree comparison on the 100-scene ID set",
        "",
        "| " + " | ".join(header) + " |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(
        [
            "",
            "For each named support, the three semicolon-separated values are original-support NRMSE, signed standardized bias, and collapsed-voxel NRMSE; other bundled rows follow the order stated in the outcome column. For the support-representation rows, the comparator is the matched nearest-voxel branch and the difference is support-aware minus nearest-voxel; high-EIC probabilities are each branch's empirical ensemble exceedance probabilities at a 0.50 decision threshold. For the probability-tree rows, the comparator is the 64-member bootstrap Extra Trees ensemble and the difference is tree minus flow. Validation-only spatial conformal quantiles were 4.1416 for the support-aware flow, 4.1399 for the nearest-voxel flow, and 2.8067 for the probability tree. All confidence intervals use paired scene bootstrap resampling. The direct support experiment is a matched single-fit seed-41 component isolation; the probability-tree analysis is likewise a single-fit sensitivity and does not replace the primary three-seed comparison. NRMSE denotes root-mean-square residual normalized by the declared observation uncertainty.",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
