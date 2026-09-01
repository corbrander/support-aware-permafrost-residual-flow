from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m1_figure_style import (
    INK,
    apply_m1_style,
    export_m1_figure,
    panel_title,
)


ROOT = Path(__file__).resolve().parents[1]
FLOW_DETAIL = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_engineering_response_seed41"
    / "m1_test_id_seed41_detail.csv"
)
FLOW_RESPONSE = FLOW_DETAIL.with_name(
    "m1_test_id_seed41_engineering_response.csv"
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
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure17_probabilistic_tree_baseline"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_probabilistic_tree_baseline"

BLUE = "#377eb8"
ORANGE = "#d95f02"
GREY = "#65727a"
LIGHT = "#d8dee2"


def _ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(int(seed))
    draws = values[
        rng.integers(0, values.size, size=(5000, values.size))
    ].mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(values.mean()), float(lower), float(upper)


def main() -> None:
    apply_m1_style(base_font_size=9.0)
    flow = pd.read_csv(FLOW_DETAIL)
    tree = pd.read_csv(TREE_DETAIL)
    if len(flow) != 100 or len(tree) != 100:
        raise RuntimeError(
            "Figure 17 requires complete 100-scene residual-flow and "
            f"probability-tree details; found flow={len(flow)}, tree={len(tree)}"
        )
    flow_response = pd.read_csv(FLOW_RESPONSE)
    tree_response = pd.read_csv(TREE_RESPONSE)
    flow_response = flow_response[
        flow_response["method"] == "Conditional residual flow"
    ].copy()
    anchor_response = pd.read_csv(FLOW_RESPONSE)
    anchor_response = anchor_response[
        anchor_response["method"] == "Tree anchor"
    ].copy()

    paired = tree.merge(
        flow,
        on="scene_id",
        suffixes=("_tree", "_flow"),
        validate="one_to_one",
    )
    if len(paired) != 100:
        raise RuntimeError(
            "Residual-flow and probability-tree files do not contain identical "
            f"scene IDs; matched {len(paired)}"
        )
    paired["rmse_difference_tree_minus_flow"] = (
        paired["eic_rmse_tree"] - paired["eic_rmse_flow"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.7))
    ax = axes[0, 0]
    ax.scatter(
        paired["eic_rmse_flow"],
        paired["eic_rmse_tree"],
        s=18,
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
    )
    limit = float(
        max(paired["eic_rmse_flow"].max(), paired["eic_rmse_tree"].max())
    )
    ax.plot([0, limit], [0, limit], color=GREY, lw=0.9, ls="--")
    ax.set_xlim(0.045, limit * 1.03)
    ax.set_ylim(0.045, limit * 1.03)
    ax.set_xlabel("Residual flow EIC RMSE")
    ax.set_ylabel("Bootstrap Extra Trees EIC RMSE")
    ax.text(
        0.97,
        0.05,
        "Tree lower in 33% of scenes",
        transform=ax.transAxes,
        ha="right",
        color=INK,
    )
    panel_title(ax, "A", "Paired whole-volume error")

    ax = axes[0, 1]
    metric_specs = [
        ("eic_rmse", "EIC RMSE"),
        ("eic_crps", "CRPS"),
    ]
    x = np.arange(len(metric_specs), dtype=float)
    width = 0.32
    for offset, (data, label, color) in enumerate(
        [(flow, "Residual flow", BLUE), (tree, "Bootstrap Extra Trees", ORANGE)]
    ):
        values = []
        errors = []
        for index, (metric, _) in enumerate(metric_specs):
            mean, lower, upper = _ci(data[metric].to_numpy(), 1700 + index + offset)
            values.append(mean)
            errors.append([mean - lower, upper - mean])
        positions = x + (offset - 0.5) * width
        ax.bar(positions, values, width=width, color=color, alpha=0.88, label=label)
        ax.errorbar(
            positions,
            values,
            yerr=np.asarray(errors).T,
            fmt="none",
            ecolor=INK,
            elinewidth=0.75,
            capsize=2.5,
        )
    ax.set_xticks(x, [label for _, label in metric_specs])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 0.112)
    ax.legend(loc="upper right")
    panel_title(ax, "B", "Accuracy and proper score")

    ax = axes[1, 0]
    points = [
        (
            flow["eic_mean_width"].mean(),
            flow["eic_coverage"].mean(),
            "Flow raw",
            BLUE,
            "o",
        ),
        (
            flow["eic_calibrated_mean_width"].mean(),
            flow["eic_calibrated_coverage"].mean(),
            "Flow calibrated",
            BLUE,
            "s",
        ),
        (
            tree["eic_mean_width"].mean(),
            tree["eic_coverage"].mean(),
            "Tree raw",
            ORANGE,
            "o",
        ),
        (
            tree["eic_calibrated_mean_width"].mean(),
            tree["eic_calibrated_coverage"].mean(),
            "Tree calibrated",
            ORANGE,
            "s",
        ),
    ]
    for width_value, coverage, label, color, marker in points:
        ax.scatter(
            width_value,
            coverage,
            s=48,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=3,
        )
    ax.axhline(0.90, color=GREY, lw=0.9, ls="--")
    ax.text(0.218, 0.903, "nominal 0.90", ha="right", va="bottom", color=GREY)
    ax.set_xlim(0.085, 0.215)
    ax.set_ylim(0.50, 0.995)
    ax.set_xlabel("Mean 90% interval width")
    ax.set_ylabel("Coverage")
    ax.legend(loc="lower right", ncol=1)
    panel_title(ax, "C", "Coverage-width trade-off")

    ax = axes[1, 1]
    response_specs = [
        (anchor_response, "Tree anchor", GREY, "o"),
        (flow_response, "Residual flow", BLUE, "s"),
        (tree_response, "Bootstrap Extra Trees", ORANGE, "^"),
    ]
    source_rows: list[dict[str, float | str]] = []
    for method_index, (data, label, color, marker) in enumerate(response_specs):
        depths = []
        means = []
        lower = []
        upper = []
        for depth in (2.0, 4.0, 6.0):
            depth_rows = data.loc[data["thaw_depth_m"] == depth]
            if len(depth_rows) != 100 or depth_rows["scene_id"].nunique() != 100:
                raise RuntimeError(
                    f"{label} response data at {depth:.0f} m are incomplete: "
                    f"rows={len(depth_rows)}, scenes={depth_rows['scene_id'].nunique()}"
                )
            values = depth_rows["response_rmse_m"].to_numpy()
            mean, low, high = _ci(values, 1800 + method_index * 10 + int(depth))
            depths.append(depth)
            means.append(mean)
            lower.append(low)
            upper.append(high)
            source_rows.append(
                {
                    "method": label,
                    "thaw_depth_m": depth,
                    "response_rmse_m": mean,
                    "ci95_lower": low,
                    "ci95_upper": high,
                }
            )
        ax.errorbar(
            depths,
            means,
            yerr=[np.asarray(means) - np.asarray(lower), np.asarray(upper) - np.asarray(means)],
            color=color,
            marker=marker,
            markersize=4.8,
            lw=1.1,
            capsize=2.5,
            label=label,
        )
    ax.set_xticks([2, 4, 6])
    ax.set_xlabel("Prescribed thaw depth (m)")
    ax.set_ylabel("Response RMSE (m)")
    ax.set_ylim(0.20, 0.44)
    ax.legend(loc="lower right")
    panel_title(ax, "D", "Controlled response propagation")

    for axis in axes.flat:
        axis.grid(axis="y", color=LIGHT, lw=0.45, alpha=0.65)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.94, wspace=0.31, hspace=0.39)
    export_m1_figure(fig, OUTPUT)

    SOURCE.mkdir(parents=True, exist_ok=True)
    paired[
        [
            "scene_id",
            "generator_family_tree",
            "eic_rmse_flow",
            "eic_rmse_tree",
            "eic_crps_flow",
            "eic_crps_tree",
            "rmse_difference_tree_minus_flow",
        ]
    ].to_csv(SOURCE / "paired_eic_metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(SOURCE / "response_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": label,
                "interval": interval,
                "coverage": coverage,
                "mean_width": width_value,
            }
            for width_value, coverage, label, _, marker in points
            for interval in ["calibrated" if marker == "s" else "raw"]
        ]
    ).to_csv(SOURCE / "coverage_width.csv", index=False)
    tree_base = TREE_DETAIL.parent
    for name in (
        "m1_probabilistic_extra_trees_metadata.json",
        "m1_probabilistic_extra_trees_validation.csv",
        "m1_probabilistic_extra_trees_test_id_summary.csv",
    ):
        shutil.copy2(tree_base / name, SOURCE / name)
    shutil.copy2(
        FLOW_DETAIL.with_name("m1_test_id_seed41_metadata.json"),
        SOURCE / "residual_flow_seed41_metadata.json",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
