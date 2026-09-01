from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m1_figure_style import apply_m1_style, export_m1_figure, panel_title


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "m1_support_guided" / "tables"
FIGURE_DIR = ROOT / "paper" / "engineering_geology_manuscript" / "figures" / "m1_final"
OUTPUT12 = FIGURE_DIR / "figure12_public_held_borehole_accuracy"
OUTPUT13 = FIGURE_DIR / "figure13_public_calibration_and_fallback"
SOURCE12 = ROOT / "outputs" / "source_data" / "m1_figure12_public_accuracy"
SOURCE13 = ROOT / "outputs" / "source_data" / "m1_figure13_public_uncertainty_safety"

INK = "#263238"
BLUE = "#377eb8"
TEAL = "#1b9e77"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
RED = "#b23a48"
GREY = "#9aa5ab"

SITES = [
    ("usgs_eic", "USGS", "38 boreholes", BLUE),
    ("arcticdata_jago_ground_ice", "Jago River", "21 boreholes", ORANGE),
    ("arcticdata_cryostratigraphy", "Arctic compilation", "50 boreholes", TEAL),
]


def metric_row(frame: pd.DataFrame, metric: str) -> pd.Series:
    selected = frame.loc[frame["metric"] == metric]
    if selected.empty:
        raise KeyError(f"Missing public-validation metric: {metric}")
    return selected.iloc[0]


def load_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    summaries: dict[str, pd.DataFrame] = {}
    details: dict[str, pd.DataFrame] = {}
    missing: list[Path] = []
    for site, _, _, _ in SITES:
        summary_path = TABLES / f"m1_public_{site}_three_seed_summary.csv"
        detail_path = TABLES / f"m1_public_{site}_three_seed_detail.csv"
        if not summary_path.exists():
            missing.append(summary_path)
        if not detail_path.exists():
            missing.append(detail_path)
        if summary_path.exists() and detail_path.exists():
            summaries[site] = pd.read_csv(summary_path)
            details[site] = pd.read_csv(detail_path)
    if missing:
        raise FileNotFoundError("Missing locked public artifact(s): " + ", ".join(str(path) for path in missing))
    expected = {"usgs_eic": 114, "arcticdata_jago_ground_ice": 63, "arcticdata_cryostratigraphy": 150}
    for site, count in expected.items():
        if len(details[site]) != count or details[site]["model_seed"].nunique() != 3:
            raise RuntimeError(f"Incomplete nested public validation for {site}: {len(details[site])} rows.")
    return summaries, details


def plot_public_rmse(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "A", "Held-borehole EIC error")
    y = np.arange(len(SITES))[::-1]
    for offset, metric, label, color, marker in (
        (0.10, "outer_rmse", "Deployed mean", BLUE, "o"),
        (-0.10, "outer_anchor_rmse", "Tree anchor", GREY, "s"),
    ):
        rows = [metric_row(summaries[site], metric) for site, _, _, _ in SITES]
        mean = np.asarray([float(row["mean"]) for row in rows])
        lower = np.asarray([float(row["ci95_lower"]) for row in rows])
        upper = np.asarray([float(row["ci95_upper"]) for row in rows])
        ax.errorbar(
            mean,
            y + offset,
            xerr=np.vstack([mean - lower, upper - mean]),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2.0,
            markersize=4.2,
            linewidth=0.9,
            label=label,
        )
    ax.set_yticks(y, [f"{label}\n{count}" for _, label, count, _ in SITES])
    ax.set_xlabel("Held-borehole EIC RMSE")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color="0.91", lw=0.45)
    ax.legend(loc="upper left", fontsize=5.8)


def plot_difference(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "B", "Paired error difference")
    y = np.arange(len(SITES))[::-1]
    rows = [metric_row(summaries[site], "rmse_difference_vs_anchor") for site, _, _, _ in SITES]
    mean = np.asarray([float(row["mean"]) for row in rows])
    lower = np.asarray([float(row["ci95_lower"]) for row in rows])
    upper = np.asarray([float(row["ci95_upper"]) for row in rows])
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - lower, upper - mean]),
        fmt="o",
        color=PURPLE,
        ecolor=PURPLE,
        capsize=2.0,
        markersize=4.2,
        linewidth=0.9,
    )
    ax.axvline(0, color=INK, lw=0.75)
    ax.axvline(0.005, color=RED, lw=0.8, ls="--", label="NI margin +0.005")
    ax.set_yticks(y, [label for _, label, _, _ in SITES])
    ax.set_xlabel("Deployed minus anchor EIC RMSE")
    ax.set_xlim(-0.0015, 0.006)
    ax.grid(axis="x", color="0.91", lw=0.45)
    ax.legend(loc="lower right", fontsize=5.8)


def plot_fold_distributions(ax: plt.Axes, details: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "C", "Outer-fold error heterogeneity")
    values = [details[site]["outer_rmse"].to_numpy(float) for site, _, _, _ in SITES]
    positions = np.arange(len(SITES))
    parts = ax.violinplot(values, positions=positions, widths=0.72, showextrema=False)
    for body, (_, _, _, color) in zip(parts["bodies"], SITES, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
    bp = ax.boxplot(values, positions=positions, widths=0.24, showfliers=False, patch_artist=True)
    for patch, (_, _, _, color) in zip(bp["boxes"], SITES, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
        patch.set_edgecolor(INK)
    for item in bp["medians"]:
        item.set_color(INK)
    ax.set_xticks(positions, [label for _, label, _, _ in SITES], rotation=15, ha="right")
    ax.set_ylabel("Outer-fold EIC RMSE")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="0.91", lw=0.45)


def plot_coverage(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "A", "Complete-borehole interval coverage")
    x = np.arange(len(SITES))
    for offset, metric, label, color in (
        (-0.10, "raw_coverage_90", "Raw", BLUE),
        (0.10, "calibrated_coverage_90", "Bounded calibrated", TEAL),
    ):
        rows = [metric_row(summaries[site], metric) for site, _, _, _ in SITES]
        mean = np.asarray([float(row["mean"]) for row in rows])
        lower = np.asarray([float(row["ci95_lower"]) for row in rows])
        upper = np.asarray([float(row["ci95_upper"]) for row in rows])
        ax.errorbar(
            x + offset,
            mean,
            yerr=np.vstack([mean - lower, upper - mean]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.0,
            markersize=4.2,
            linewidth=0.9,
            label=label,
        )
    ax.axhline(0.90, color=RED, lw=0.8, ls="--", label="Nominal 0.90")
    ax.set_xticks(x, [label for _, label, _, _ in SITES], rotation=15, ha="right")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="0.91", lw=0.45)
    ax.legend(loc="lower right", fontsize=5.5)


def plot_width(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "B", "Width cost after bounded calibration")
    x = np.arange(len(SITES))
    width = 0.30
    raw = np.asarray([float(metric_row(summaries[site], "raw_width_90")["mean"]) for site, _, _, _ in SITES])
    calibrated = np.asarray(
        [float(metric_row(summaries[site], "calibrated_width_90")["mean"]) for site, _, _, _ in SITES]
    )
    ax.bar(x - width / 2, raw, width, color=BLUE, alpha=0.72, label="Raw")
    ax.bar(x + width / 2, calibrated, width, color=TEAL, alpha=0.72, label="Bounded calibrated")
    for xpos, value in zip(x + width / 2, calibrated, strict=True):
        ax.text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(x, [label for _, label, _, _ in SITES], rotation=15, ha="right")
    ax.set_ylabel("Mean 90% EIC interval width")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="0.91", lw=0.45)
    ax.legend(loc="upper left", fontsize=5.8)


def plot_safety_gates(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "C", "Nested gate and exact-fallback outcomes")
    x = np.arange(len(SITES))
    metrics = [
        ("inner_noninferiority_pass", "Inner NI pass", ORANGE),
        ("fallback_due_to_ood", "Public OOD fallback", RED),
        ("exact_anchor_fallback_applied", "Exact mean fallback", PURPLE),
    ]
    width = 0.23
    for index, (metric, label, color) in enumerate(metrics):
        values = np.asarray([float(metric_row(summaries[site], metric)["mean"]) for site, _, _, _ in SITES])
        ax.bar(x + (index - 1) * width, values, width, color=color, alpha=0.78, label=label)
    ax.set_xticks(x, [label for _, label, _, _ in SITES], rotation=15, ha="right")
    ax.set_ylabel("Outer-fold fraction")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color="0.91", lw=0.45)


def save_sources(
    summaries: dict[str, pd.DataFrame], details: dict[str, pd.DataFrame], source: Path, figure_number: int
) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for site, _, _, _ in SITES:
        summaries[site].to_csv(source / f"figure{figure_number}_{site}_summary.csv", index=False)
        details[site].to_csv(source / f"figure{figure_number}_{site}_detail.csv", index=False)


def main() -> None:
    summaries, details = load_inputs()
    apply_m1_style()

    fig12, axes12 = plt.subplots(1, 3, figsize=(183 / 25.4, 71 / 25.4), constrained_layout=False)
    fig12.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.20, wspace=0.40)
    plot_public_rmse(axes12[0], summaries)
    plot_difference(axes12[1], summaries)
    plot_fold_distributions(axes12[2], details)
    save_sources(summaries, details, SOURCE12, 12)
    (SOURCE12 / "metadata.json").write_text(
        json.dumps(
            {
                "outer_seed_fold_pairs": 327,
                "noninferiority_margin": 0.005,
                "claim_boundary": "Held-borehole support prediction only; no dense three-dimensional field validation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    export_m1_figure(fig12, OUTPUT12)

    fig13, axes13 = plt.subplots(1, 3, figsize=(183 / 25.4, 78 / 25.4), constrained_layout=False)
    fig13.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.28, wspace=0.36)
    plot_coverage(axes13[0], summaries)
    plot_width(axes13[1], summaries)
    plot_safety_gates(axes13[2], summaries)
    handles, labels = axes13[2].get_legend_handles_labels()
    fig13.legend(handles, labels, loc="lower center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.5, 0.035))
    save_sources(summaries, details, SOURCE13, 13)
    (SOURCE13 / "metadata.json").write_text(
        json.dumps(
            {
                "calibration": "Nested complete-borehole block conformal with projection to EIC [0, 0.90]",
                "outer_seed_fold_pairs": 327,
                "exact_anchor_fallback_fraction": 1.0,
                "claim_boundary": "The deployed learned bias is disabled in all public outer folds.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    export_m1_figure(fig13, OUTPUT13)
    print(json.dumps({"figure12": str(OUTPUT12), "figure13": str(OUTPUT13)}, indent=2))


if __name__ == "__main__":
    main()
