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
INPUT = ROOT / "outputs" / "source_data" / "m1_figure6"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure7_uncertainty_calibration"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure07_uncertainty_calibration"

INK = "#263238"
BLUE = "#377eb8"
TEAL = "#1b9e77"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
RED = "#b23a48"

SUPPORT_ROWS = [
    ("support_nrmse_borehole_eic", "Borehole EIC"),
    ("support_nrmse_borehole_temperature", "Borehole temperature"),
    ("support_nrmse_nmr_unfrozen_water", "NMR unfrozen water"),
    ("support_nrmse_ert_log_resistivity", "ERT log-resistivity"),
    ("support_nrmse_alt", "Active-layer crossing"),
]


def metric_row(summary: pd.DataFrame, metric: str) -> pd.Series:
    selected = summary.loc[summary["metric"] == metric]
    if "generator_family" in selected.columns:
        selected = selected.loc[selected["generator_family"] == "all"]
    if selected.empty:
        raise KeyError(f"Missing locked calibration metric: {metric}")
    return selected.iloc[0]


def violin_pair(
    ax: plt.Axes,
    raw: np.ndarray,
    calibrated: np.ndarray,
    *,
    letter: str,
    title: str,
    ylabel: str,
    reference: float | None = None,
) -> None:
    parts = ax.violinplot([raw, calibrated], positions=[0, 1], widths=0.72, showextrema=False)
    for body, color in zip(parts["bodies"], [BLUE, TEAL], strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
    bp = ax.boxplot(
        [raw, calibrated],
        positions=[0, 1],
        widths=0.26,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 0.9},
        whiskerprops={"color": INK, "linewidth": 0.7},
        capprops={"color": INK, "linewidth": 0.7},
    )
    for patch, color in zip(bp["boxes"], [BLUE, TEAL], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(INK)
    for xpos, values, color in ((0, raw, BLUE), (1, calibrated, TEAL)):
        ax.scatter(xpos, np.mean(values), s=27, color=color, edgecolor="white", linewidth=0.6, zorder=5)
        ax.text(xpos, np.percentile(values, 97.5) + 0.025 * (np.nanmax([raw, calibrated]) + 1e-9), f"mean {np.mean(values):.3f}", ha="center", va="bottom", fontsize=6.2)
    if reference is not None:
        ax.axhline(reference, color=RED, ls="--", lw=0.8, label=f"Nominal {reference:.2f}")
        ax.legend(loc="lower right", fontsize=5.8)
    ax.set_xticks([0, 1], ["Raw ensemble", "Block-conformal"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.91", lw=0.45)
    panel_title(ax, letter, title)


def plot_pit(ax: plt.Axes, detail: pd.DataFrame) -> None:
    panel_title(ax, "C", "Raw ensemble rank diagnostic")
    colors = {41: BLUE, 42: ORANGE, 43: PURPLE}
    for seed, frame in detail.groupby("seed"):
        ax.scatter(
            frame["eic_pit_mean"],
            frame["eic_pit_variance"],
            s=13,
            alpha=0.38,
            color=colors.get(int(seed), "0.5"),
            edgecolor="none",
            label=f"Seed {int(seed)}",
        )
        ax.scatter(
            frame["eic_pit_mean"].mean(),
            frame["eic_pit_variance"].mean(),
            s=46,
            color=colors.get(int(seed), "0.5"),
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    ax.scatter([0.5], [1.0 / 12.0], marker="+", s=70, color=RED, linewidth=1.0, label="Uniform reference")
    ax.axvline(0.5, color="0.70", lw=0.6, ls=":")
    ax.axhline(1.0 / 12.0, color="0.70", lw=0.6, ls=":")
    ax.set_xlabel("Scene-level PIT mean")
    ax.set_ylabel("Scene-level PIT variance")
    ax.grid(color="0.92", lw=0.45)
    ax.legend(loc="upper right", fontsize=5.5)


def plot_support(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_title(ax, "D", "Residuals at supplied supports")
    rows = [metric_row(summary, metric) for metric, _ in SUPPORT_ROWS]
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
        markersize=4.2,
        capsize=2.0,
        linewidth=0.9,
    )
    ax.axvline(1.0, color=RED, lw=0.8, ls="--", label="One declared s.d.")
    ax.set_yticks(y, [label for _, label in SUPPORT_ROWS])
    ax.set_xlabel("Normalized RMSE")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color="0.91", lw=0.45)
    ax.legend(loc="lower right", fontsize=5.8)


def main() -> None:
    detail_path = INPUT / "figure6_id_scene_level.csv"
    summary_path = INPUT / "figure6_id_summary.csv"
    if not detail_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Build the locked calibration source bundle before Figure 7.")
    detail = pd.read_csv(detail_path)
    summary = pd.read_csv(summary_path)
    if len(detail) != 300 or detail["seed"].nunique() != 3 or detail["scene_id"].nunique() != 100:
        raise RuntimeError("Figure 7 requires the complete three-seed, 100-scene ID audit.")

    apply_m1_style()
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 120 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.96, top=0.94, bottom=0.11, wspace=0.35, hspace=0.44)
    violin_pair(
        axes[0, 0],
        detail["eic_coverage"].to_numpy(),
        detail["eic_calibrated_coverage"].to_numpy(),
        letter="A",
        title="Scene-level 90% interval coverage",
        ylabel="Coverage",
        reference=0.90,
    )
    violin_pair(
        axes[0, 1],
        detail["eic_mean_width"].to_numpy(),
        detail["eic_calibrated_mean_width"].to_numpy(),
        letter="B",
        title="Cost of spatial calibration",
        ylabel="Mean EIC interval width",
    )
    plot_pit(axes[1, 0], detail)
    plot_support(axes[1, 1], summary)

    SOURCE.mkdir(parents=True, exist_ok=True)
    detail.to_csv(SOURCE / "figure7_scene_level.csv", index=False)
    summary.to_csv(SOURCE / "figure7_summary.csv", index=False)
    metadata = {
        "calibration": "90% spatial block conformal fitted on validation scenes only",
        "posterior_members": 64,
        "model_seeds": [41, 42, 43],
        "n_unique_scenes": 100,
        "claim_boundary": "Conformal calibration changes interval coverage and width but does not rewrite raw ensemble ranks.",
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    export_m1_figure(fig, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "source": str(SOURCE)}, indent=2))


if __name__ == "__main__":
    main()
