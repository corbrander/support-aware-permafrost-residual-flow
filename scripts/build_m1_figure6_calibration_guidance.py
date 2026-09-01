from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m1_figure_style import apply_m1_style, enforce_m1_typography


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "m1_support_guided" / "tables"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure6_calibration_and_guidance"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure6"

INK = "#263238"
BLUE = "#377eb8"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
TEAL = "#1b9e77"
GREY = "#9aa5ab"
RED = "#b23a48"

SUPPORT_ROWS = [
    ("support_nrmse_borehole_eic", "Borehole EIC"),
    ("support_nrmse_borehole_temperature", "Borehole temperature"),
    ("support_nrmse_ert_log_resistivity", "ERT log-resistivity"),
    ("support_nrmse_nmr_unfrozen_water", "NMR unfrozen water"),
    ("support_nrmse_alt", "ALT crossing"),
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.035,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        ha="left",
    )
    ax.text(
        0.10,
        1.035,
        title,
        transform=ax.transAxes,
        fontsize=7.8,
        color=INK,
        ha="left",
    )


def metric_row(frame: pd.DataFrame, metric: str) -> pd.Series:
    selected = frame.loc[frame["metric"] == metric]
    if "generator_family" in selected:
        selected = selected.loc[selected["generator_family"] == "all"]
    if selected.empty:
        raise KeyError(f"Missing metric: {metric}")
    return selected.iloc[0]


def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_path = TABLES / "m1_test_id_three_seed_detail.csv"
    summary_path = TABLES / "m1_test_id_three_seed_summary.csv"
    guidance_path = TABLES / "m1_guidance_ablation_paired_summary.csv"
    missing = [path for path in (detail_path, summary_path, guidance_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Figure 6 is evidence-locked and cannot be built before these files exist: "
            + ", ".join(str(path) for path in missing)
        )
    detail = pd.read_csv(detail_path)
    if len(detail) != 300 or detail["scene_id"].nunique() != 100 or detail["seed"].nunique() != 3:
        raise RuntimeError("The final ID detail must contain 300 seed-scene rows for 100 scenes.")
    return detail, pd.read_csv(summary_path), pd.read_csv(guidance_path)


def plot_calibration(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_title(ax, "a", "Validation-only interval calibration")
    coverage = [
        float(metric_row(summary, "eic_coverage")["mean"]),
        float(metric_row(summary, "eic_calibrated_coverage")["mean"]),
    ]
    widths = [
        float(metric_row(summary, "eic_mean_width")["mean"]),
        float(metric_row(summary, "eic_calibrated_mean_width")["mean"]),
    ]
    x = np.arange(2)
    bars = ax.bar(x, coverage, width=0.52, color=[GREY, BLUE], alpha=0.88)
    ax.axhline(0.90, color=RED, lw=0.9, ls="--", label="nominal 0.90")
    ax.set_xticks(x, ["Raw", "Block conformal"])
    ax.set_ylabel("EIC interval coverage")
    ax.set_ylim(0, 1.06)
    ax.grid(axis="y", color="0.91", lw=0.55)
    for bar, value in zip(bars, coverage, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=6.2)
    twin = ax.twinx()
    twin.plot(x, widths, color=ORANGE, marker="o", lw=1.2, ms=4.0, label="mean width")
    twin.set_ylabel("Mean EIC interval width", color=ORANGE)
    twin.tick_params(axis="y", labelcolor=ORANGE)
    twin.set_ylim(0, max(widths) * 1.35)
    lines = [ax.get_lines()[0], twin.get_lines()[0]]
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right", fontsize=5.8)


def plot_pit(ax: plt.Axes, detail: pd.DataFrame) -> None:
    panel_title(ax, "b", "Raw ensemble rank diagnostic")
    for seed, color, marker in [(41, BLUE, "o"), (42, TEAL, "s"), (43, PURPLE, "^")]:
        subset = detail.loc[detail["seed"] == seed]
        ax.scatter(
            subset["eic_pit_mean"],
            subset["eic_pit_variance"],
            s=9,
            color=color,
            marker=marker,
            alpha=0.28,
            linewidths=0,
        )
        ax.scatter(
            [subset["eic_pit_mean"].mean()],
            [subset["eic_pit_variance"].mean()],
            s=31,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            label=f"seed {seed}",
        )
    ax.scatter([0.5], [1.0 / 12.0], marker="*", s=45, color=RED, label="uniform reference", zorder=5)
    ax.axvline(0.5, color=RED, lw=0.6, ls="--", alpha=0.65)
    ax.axhline(1.0 / 12.0, color=RED, lw=0.6, ls="--", alpha=0.65)
    ax.set_xlabel("Scene PIT mean")
    ax.set_ylabel("Scene PIT variance")
    ax.grid(color="0.92", lw=0.5)
    ax.legend(loc="best", fontsize=5.6)


def plot_guidance(ax: plt.Axes, guidance: pd.DataFrame) -> None:
    panel_title(ax, "c", "Guided minus unguided paired effects")
    metrics = [
        ("difference_support_fidelity_score_guided_minus_unguided", "Balanced support score"),
        ("difference_eic_rmse_guided_minus_unguided", "Whole-volume EIC RMSE"),
        *[
            (f"difference_{metric}_guided_minus_unguided", label)
            for metric, label in SUPPORT_ROWS
        ],
    ]
    rows = [metric_row(guidance, metric) for metric, _ in metrics]
    means = np.asarray([float(row["mean"]) for row in rows])
    lower = np.asarray([float(row["ci95_lower"]) for row in rows])
    upper = np.asarray([float(row["ci95_upper"]) for row in rows])
    labels = [label for _, label in metrics]
    y = np.arange(len(labels))[::-1]
    colors = [TEAL if value < 0 else RED for value in means]
    for mean, low, high, position, color in zip(
        means, lower, upper, y, colors, strict=True
    ):
        ax.errorbar(
            mean,
            position,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            ms=3.7,
            elinewidth=0.9,
            capsize=2,
        )
    ax.axvline(0, color=INK, lw=0.75)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired difference (negative favours guidance)")
    ax.grid(axis="x", color="0.92", lw=0.5)


def plot_support(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_title(ax, "d", "Supplied-support residuals")
    rows = [metric_row(summary, metric) for metric, _ in SUPPORT_ROWS]
    means = np.asarray([float(row["mean"]) for row in rows])
    lower = np.asarray([float(row["ci95_lower"]) for row in rows])
    upper = np.asarray([float(row["ci95_upper"]) for row in rows])
    labels = [label for _, label in SUPPORT_ROWS]
    y = np.arange(len(labels))[::-1]
    ax.errorbar(
        means,
        y,
        xerr=np.vstack((means - lower, upper - means)),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        ms=4.0,
        lw=0.9,
        capsize=2,
    )
    ax.axvline(1.0, color=RED, lw=0.8, ls="--", label="declared noise scale")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Normalized RMSE")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.legend(loc="lower right", fontsize=5.7)


def main() -> None:
    apply_m1_style()
    detail, summary, guidance = require_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    detail[
        [
            "scene_id",
            "seed",
            "eic_coverage",
            "eic_mean_width",
            "eic_calibrated_coverage",
            "eic_calibrated_mean_width",
            "eic_pit_mean",
            "eic_pit_variance",
            *[metric for metric, _ in SUPPORT_ROWS],
        ]
    ].to_csv(SOURCE / "figure6_id_scene_level.csv", index=False)
    summary.to_csv(SOURCE / "figure6_id_summary.csv", index=False)
    guidance.to_csv(SOURCE / "figure6_guidance_paired_summary.csv", index=False)
    metadata = {
        "posterior_members": 64,
        "sampling_steps": 10,
        "guidance_strength": 2.0,
        "calibration": "90% spatial block conformal; fitted on validation only",
        "uncertainty_boundary": "PIT is a raw ensemble diagnostic; conformal intervals change coverage and width but not the raw ranks.",
        "guidance_sign": "Negative paired differences favour the guided run.",
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 122 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.91, top=0.94, bottom=0.105, wspace=0.46, hspace=0.44)
    plot_calibration(axes[0, 0], summary)
    plot_pit(axes[0, 1], detail)
    plot_guidance(axes[1, 0], guidance)
    plot_support(axes[1, 1], summary)
    enforce_m1_typography(fig)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        OUTPUT.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(OUTPUT), "source": str(SOURCE)}, indent=2))


if __name__ == "__main__":
    main()
