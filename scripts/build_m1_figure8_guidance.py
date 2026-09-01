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
CALIBRATION = ROOT / "outputs" / "m1_support_guided" / "calibration"
TABLES = ROOT / "outputs" / "m1_support_guided" / "tables"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure8_likelihood_guidance"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure08_likelihood_guidance"

INK = "#263238"
BLUE = "#377eb8"
TEAL = "#1b9e77"
ORANGE = "#d95f02"
RED = "#b23a48"

SUPPORT_EFFECTS = [
    ("difference_support_fidelity_score_guided_minus_unguided", "Balanced support score"),
    ("difference_support_nrmse_borehole_eic_guided_minus_unguided", "Borehole EIC"),
    ("difference_support_nrmse_borehole_temperature_guided_minus_unguided", "Borehole temperature"),
    ("difference_support_nrmse_nmr_unfrozen_water_guided_minus_unguided", "NMR unfrozen water"),
    ("difference_support_nrmse_ert_log_resistivity_guided_minus_unguided", "ERT log-resistivity"),
    ("difference_support_nrmse_alt_guided_minus_unguided", "Active-layer crossing"),
]


def metric_row(summary: pd.DataFrame, metric: str) -> pd.Series:
    selected = summary.loc[summary["metric"] == metric]
    if selected.empty:
        raise KeyError(f"Missing guidance metric: {metric}")
    return selected.iloc[0]


def plot_selection(ax: plt.Axes, selection: pd.DataFrame) -> None:
    panel_title(ax, "A", "Validation-only guidance selection")
    x = selection["guidance_strength"].to_numpy(float)
    y = selection["support_score"].to_numpy(float)
    ax.plot(x, y, color=BLUE, marker="o", ms=4.2, lw=1.1)
    chosen = selection.loc[selection["guidance_strength"] == 2.0].iloc[0]
    ax.scatter([2.0], [chosen["support_score"]], s=58, marker="*", color=ORANGE, edgecolor=INK, linewidth=0.6, zorder=5)
    ax.annotate(
        "selected",
        (2.0, float(chosen["support_score"])),
        xytext=(-26, 12),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.6},
        fontsize=6.4,
    )
    ax.set_xlabel("Guidance strength")
    ax.set_ylabel("Type-balanced support score")
    ax.set_xticks(x)
    ax.grid(color="0.91", lw=0.45)


def plot_noninferiority(ax: plt.Axes, selection: pd.DataFrame) -> None:
    panel_title(ax, "B", "Whole-volume non-inferiority check")
    x = selection["guidance_strength"].to_numpy(float)
    mean = 1e4 * selection["eic_rmse_difference_vs_0.25"].to_numpy(float)
    lower = 1e4 * selection["eic_rmse_difference_ci95_lower"].to_numpy(float)
    upper = 1e4 * selection["eic_rmse_difference_ci95_upper"].to_numpy(float)
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([mean - lower, upper - mean]),
        fmt="o-",
        color=TEAL,
        ecolor=TEAL,
        lw=1.0,
        ms=4.0,
        capsize=2.0,
    )
    ax.axhline(0, color=INK, lw=0.7)
    ax.fill_between([x.min(), x.max()], -1.5, 0.5, color=TEAL, alpha=0.07)
    ax.text(0.02, 0.04, "All upper CIs < +0.005 margin", transform=ax.transAxes, fontsize=6.2, color=INK)
    ax.set_xlabel("Guidance strength")
    ax.set_ylabel(r"EIC RMSE difference vs 0.25 ($\times 10^{-4}$)")
    ax.set_xticks(x)
    ax.grid(color="0.91", lw=0.45)


def plot_paired_rmse(ax: plt.Axes, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    panel_title(ax, "C", "Paired ID reconstruction audit")
    ax.scatter(
        detail["eic_rmse_unguided"],
        detail["eic_rmse_guided"],
        s=12,
        color=BLUE,
        alpha=0.32,
        edgecolor="none",
    )
    limits = [
        float(min(detail["eic_rmse_unguided"].min(), detail["eic_rmse_guided"].min())),
        float(max(detail["eic_rmse_unguided"].max(), detail["eic_rmse_guided"].max())),
    ]
    ax.plot(limits, limits, color=INK, lw=0.8, ls="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    row = metric_row(summary, "difference_eic_rmse_guided_minus_unguided")
    ax.text(
        0.04,
        0.94,
        f"mean difference = {row['mean']:.6f}\n95% CI [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
    )
    ax.set_xlabel("Unguided EIC RMSE")
    ax.set_ylabel("Guided EIC RMSE")
    ax.grid(color="0.92", lw=0.4)


def plot_support_effects(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_title(ax, "D", "Paired support-residual effects")
    rows = [metric_row(summary, metric) for metric, _ in SUPPORT_EFFECTS]
    means = np.asarray([float(row["mean"]) for row in rows])
    lower = np.asarray([float(row["ci95_lower"]) for row in rows])
    upper = np.asarray([float(row["ci95_upper"]) for row in rows])
    y = np.arange(len(rows))[::-1]
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lower, upper - means]),
        fmt="o",
        color=ORANGE,
        ecolor=ORANGE,
        capsize=2.0,
        markersize=4.2,
        linewidth=0.9,
    )
    ax.axvline(0, color=INK, lw=0.75)
    ax.set_yticks(y, [label for _, label in SUPPORT_EFFECTS])
    ax.set_xlabel("Guided minus unguided normalized residual")
    ax.grid(axis="x", color="0.91", lw=0.45)


def main() -> None:
    selection_path = CALIBRATION / "m1_guidance_selection.csv"
    detail_path = TABLES / "m1_guidance_ablation_paired_detail.csv"
    summary_path = TABLES / "m1_guidance_ablation_paired_summary.csv"
    missing = [path for path in (selection_path, detail_path, summary_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing locked guidance artifact(s): " + ", ".join(str(path) for path in missing))
    selection = pd.read_csv(selection_path)
    detail = pd.read_csv(detail_path)
    summary = pd.read_csv(summary_path)
    if len(selection) != 5 or len(detail) != 300 or detail["model_seed"].nunique() != 3:
        raise RuntimeError("Figure 8 requires five validation strengths and the complete 300-pair ID audit.")

    apply_m1_style()
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 120 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.94, bottom=0.11, wspace=0.37, hspace=0.44)
    plot_selection(axes[0, 0], selection)
    plot_noninferiority(axes[0, 1], selection)
    plot_paired_rmse(axes[1, 0], detail, summary)
    plot_support_effects(axes[1, 1], summary)

    SOURCE.mkdir(parents=True, exist_ok=True)
    selection.to_csv(SOURCE / "figure8_validation_selection.csv", index=False)
    detail.to_csv(SOURCE / "figure8_paired_scene_level.csv", index=False)
    summary.to_csv(SOURCE / "figure8_paired_summary.csv", index=False)
    metadata = {
        "validation_scenes": 20,
        "candidate_strengths": selection["guidance_strength"].tolist(),
        "selected_strength": 2.0,
        "reference_strength": 0.25,
        "whole_volume_noninferiority_margin": 0.005,
        "formal_id_pairs": 300,
        "guidance_sign": "Negative paired differences favour guidance.",
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    export_m1_figure(fig, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "source": str(SOURCE)}, indent=2))


if __name__ == "__main__":
    main()
