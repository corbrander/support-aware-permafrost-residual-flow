from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from m1_figure_style import INK, apply_m1_style, export_m1_figure, panel_title


ROOT = Path(__file__).resolve().parents[1]
MODEL = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_engineering_response_seed41"
    / "m1_test_id_seed41_engineering_response.csv"
)
GAUSSIAN = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_engineering_response_gaussian_seed430"
    / "m1_geostatistical_test_id_engineering_response.csv"
)
LOGIT_GAUSSIAN = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_engineering_response_logit_gaussian_seed440"
    / "m1_geostatistical_logit_test_id_engineering_response.csv"
)
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure15_engineering_decision_sensitivity"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_engineering_response_population"


METHOD_ORDER = (
    "Tree anchor",
    "Bounded Gaussian baseline",
    "Logit-Gaussian baseline",
    "Conditional residual flow",
)
METHOD_LABELS = {
    "Tree anchor": "Tree anchor",
    "Bounded Gaussian baseline": "Bounded Gaussian",
    "Logit-Gaussian baseline": "Logit-Gaussian",
    "Conditional residual flow": "Final model",
}
COLORS = {
    "Tree anchor": "#607d8b",
    "Bounded Gaussian baseline": "#d95f02",
    "Logit-Gaussian baseline": "#756bb1",
    "Conditional residual flow": "#1b9e77",
}


def _load() -> pd.DataFrame:
    required = (MODEL, GAUSSIAN, LOGIT_GAUSSIAN)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal response audits: " + "; ".join(missing))
    model = pd.read_csv(MODEL)
    gaussian = pd.read_csv(GAUSSIAN)
    logit = pd.read_csv(LOGIT_GAUSSIAN)
    model = model[model["method"].isin(("Tree anchor", "Conditional residual flow"))]
    combined = pd.concat([model, gaussian, logit], ignore_index=True)
    combined = combined[combined["method"].isin(METHOD_ORDER)].copy()
    combined["method"] = pd.Categorical(
        combined["method"], categories=METHOD_ORDER, ordered=True
    )
    combined = combined.sort_values(
        ["thaw_depth_m", "method", "scene_id"], ignore_index=True
    )
    expected = {method: 300 for method in METHOD_ORDER}
    observed = combined.groupby("method", observed=True).size().to_dict()
    if observed != expected:
        raise RuntimeError(f"Expected 300 scene-depth rows per method; found {observed}")
    return combined


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, values.size, size=(5000, values.size))].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(values)), float(lower), float(upper)


def _summary(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    metrics = (
        "response_rmse_m",
        "response_bias_m",
        "gradient_rmse_m_per_m",
        "sensitivity",
        "specificity",
        "raw_interval_mean_width_m",
        "conformal_envelope_mean_width_m",
    )
    rows: list[dict[str, float | int | str]] = []
    for (method, depth), group in data.groupby(
        ["method", "thaw_depth_m"], observed=True, sort=False
    ):
        for metric_index, metric in enumerate(metrics):
            values = group[metric].astype(float).to_numpy()
            finite_values = values[np.isfinite(values)]
            mean, lower, upper = _bootstrap_mean_ci(
                values, 1500 + metric_index + int(round(10 * float(depth)))
            )
            rows.append(
                {
                    "method": str(method),
                    "thaw_depth_m": float(depth),
                    "metric": metric,
                    "mean": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "median": (
                        float(np.median(finite_values))
                        if finite_values.size
                        else float("nan")
                    ),
                    "n_scenes": int(group["scene_id"].nunique()),
                    "n_finite": int(finite_values.size),
                }
            )
    summary = pd.DataFrame(rows)

    pivot = data.pivot_table(
        index=["scene_id", "thaw_depth_m"],
        columns="method",
        values="response_rmse_m",
        observed=True,
    )
    highlights: dict[str, float] = {}
    for depth in (2.0, 4.0, 6.0):
        depth_rows = pivot.xs(depth, level="thaw_depth_m")
        for baseline, slug in (
            ("Tree anchor", "anchor"),
            ("Bounded Gaussian baseline", "bounded_gaussian"),
            ("Logit-Gaussian baseline", "logit_gaussian"),
        ):
            difference = (
                depth_rows["Conditional residual flow"] - depth_rows[baseline]
            )
            difference_mean, difference_lower, difference_upper = _bootstrap_mean_ci(
                difference.to_numpy(), 2100 + int(depth * 10) + len(slug)
            )
            relative = 1.0 - (
                depth_rows["Conditional residual flow"] / depth_rows[baseline]
            )
            mean, lower, upper = _bootstrap_mean_ci(
                relative.to_numpy(), 2500 + int(depth * 10) + len(slug)
            )
            prefix = f"depth_{int(depth)}m_relative_rmse_reduction_vs_{slug}"
            highlights[f"{prefix}_mean"] = mean
            highlights[f"{prefix}_ci95_lower"] = lower
            highlights[f"{prefix}_ci95_upper"] = upper
            highlights[f"{prefix}_improved_fraction"] = float(np.mean(relative > 0.0))
            difference_prefix = (
                f"depth_{int(depth)}m_paired_rmse_difference_vs_{slug}"
            )
            highlights[f"{difference_prefix}_mean_m"] = difference_mean
            highlights[f"{difference_prefix}_ci95_lower_m"] = difference_lower
            highlights[f"{difference_prefix}_ci95_upper_m"] = difference_upper
    return summary, highlights


def _box_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    letter: str,
    title: str,
    ylabel: str,
    *,
    zero_line: bool = False,
    probability_axis: bool = False,
    methods: tuple[str, ...] = METHOD_ORDER,
) -> None:
    depths = (2.0, 4.0, 6.0)
    offsets = np.linspace(-0.27, 0.27, len(methods))
    width = 0.16 if len(methods) >= 4 else 0.21
    for method, offset in zip(methods, offsets, strict=True):
        values = []
        positions = []
        for depth_index, depth in enumerate(depths, start=1):
            subset = data[
                (data["method"] == method) & (data["thaw_depth_m"] == depth)
            ][metric].astype(float)
            subset = subset[np.isfinite(subset)]
            values.append(subset.to_numpy())
            positions.append(depth_index + offset)
        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": "white", "linewidth": 1.0},
            whiskerprops={"color": COLORS[method], "linewidth": 0.8},
            capprops={"color": COLORS[method], "linewidth": 0.8},
            boxprops={
                "facecolor": COLORS[method],
                "edgecolor": COLORS[method],
                "linewidth": 0.8,
                "alpha": 0.88,
            },
        )
        for median in boxes["medians"]:
            median.set_solid_capstyle("butt")
    if zero_line:
        ax.axhline(0.0, color="#455a64", linewidth=0.7, linestyle="--", zorder=0)
    if probability_axis:
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlim(0.55, 3.45)
    ax.set_xticks((1, 2, 3), ("2", "4", "6"))
    ax.set_xlabel("Prescribed thaw depth (m)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#cfd8dc", linewidth=0.45, alpha=0.8)
    panel_title(ax, letter, title, x=0.0, y=1.03, title_offset=0.14, fontsize=10.2)


def main() -> None:
    data = _load()
    summary, highlights = _summary(data)
    SOURCE.mkdir(parents=True, exist_ok=True)
    data.to_csv(SOURCE / "engineering_response_population_detail.csv", index=False)
    summary.to_csv(SOURCE / "engineering_response_population_summary.csv", index=False)
    (SOURCE / "engineering_response_population_highlights.json").write_text(
        json.dumps(highlights, indent=2), encoding="utf-8"
    )

    apply_m1_style(base_font_size=9.2)
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 142 / 25.4))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.865,
        bottom=0.095,
        wspace=0.35,
        hspace=0.52,
    )
    _box_panel(
        axes[0, 0], data, "response_rmse_m", "A", "Response error", "RMSE (m)"
    )
    _box_panel(
        axes[0, 1],
        data,
        "response_bias_m",
        "B",
        "Response bias",
        "Bias (m)",
        zero_line=True,
    )
    _box_panel(
        axes[0, 2],
        data,
        "gradient_rmse_m_per_m",
        "C",
        "Differential-response error",
        "Gradient RMSE (m/m)",
    )
    _box_panel(
        axes[1, 0],
        data,
        "sensitivity",
        "D",
        "High-response sensitivity",
        "Sensitivity",
        probability_axis=True,
    )
    _box_panel(
        axes[1, 1],
        data,
        "specificity",
        "E",
        "Low-response specificity",
        "Specificity",
        probability_axis=True,
    )
    _box_panel(
        axes[1, 2],
        data,
        "conformal_envelope_mean_width_m",
        "F",
        "Calibrated-envelope sharpness",
        "Mean width (m)",
        methods=(
            "Bounded Gaussian baseline",
            "Logit-Gaussian baseline",
            "Conditional residual flow",
        ),
    )

    handles = [
        Patch(
            facecolor=COLORS[method],
            edgecolor=COLORS[method],
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=4,
        columnspacing=1.1,
        handlelength=1.2,
        handletextpad=0.4,
    )
    fig.text(
        0.50,
        0.925,
        "100 immutable ID scenes; boxes show interquartile range, whiskers show the 5th--95th percentiles",
        ha="center",
        va="center",
        fontsize=9.0,
        color=INK,
        fontweight="normal",
    )
    export_m1_figure(fig, OUTPUT)


if __name__ == "__main__":
    main()
