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
    / "figure10_sequential_investigation"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure10_sequential"

INK = "#263238"
RED = "#b23a48"
POLICIES = [
    ("random", "Random", "#9aa5ab", "--"),
    ("grid_space_filling", "Grid", "#7f8c8d", ":"),
    ("farthest", "Farthest", "#8c6d31", "-."),
    ("variance", "Variance", "#377eb8", "-"),
    ("entropy", "Entropy", "#756bb1", "-"),
    ("high_eic_probability", "High-EIC", "#d95f02", "-"),
    ("composite", "Composite", "#1b9e77", "-"),
    ("expected_loss", "Expected loss", "#b23a48", "-"),
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
    ax.text(0.0, 1.035, f"({letter})", transform=ax.transAxes, fontsize=8.4, fontweight="normal")
    ax.text(0.10, 1.035, title, transform=ax.transAxes, fontsize=7.8, color=INK)


def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        TABLES / "m1_sequential_three_seed_detail.csv",
        TABLES / "m1_sequential_three_seed_summary.csv",
        TABLES / "m1_sequential_regret_three_seed_summary.csv",
        TABLES / "m1_sequential_policy_vs_random_summary.csv",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Figure 9 cannot be built before the complete sequential aggregates exist: "
            + ", ".join(str(path) for path in missing)
        )
    detail, summary, regret, paired = (pd.read_csv(path) for path in paths)
    expected = {policy for policy, _, _, _ in POLICIES}
    if set(detail["policy"].unique()) != expected or detail["model_seed"].nunique() != 3:
        raise RuntimeError("Sequential detail does not contain all eight policies and three seeds.")
    if detail["scene_id"].nunique() != 10 or int(detail["step"].max()) < 5:
        raise RuntimeError("Sequential detail must contain 10 scenes and five added-borehole steps.")
    return detail, summary, regret, paired


def summary_rows(summary: pd.DataFrame, policy: str, metric: str) -> pd.DataFrame:
    selected = summary.loc[(summary["policy"] == policy) & (summary["metric"] == metric)].sort_values("step")
    if selected.empty:
        raise KeyError(f"Missing sequential summary: {policy}, {metric}")
    return selected


def curve_panel(ax: plt.Axes, summary: pd.DataFrame, metric: str, letter: str, title: str, ylabel: str) -> None:
    panel_title(ax, letter, title)
    for policy, label, color, linestyle in POLICIES:
        rows = summary_rows(summary, policy, metric)
        linewidth = 1.55 if policy == "expected_loss" else 0.9
        markersize = 3.8 if policy == "expected_loss" else 2.8
        ax.plot(
            rows["step"],
            rows["mean"],
            color=color,
            ls=linestyle,
            marker="o",
            ms=markersize,
            lw=linewidth,
            label=label,
            zorder=4 if policy == "expected_loss" else 2,
        )
        if policy in {"expected_loss", "random"}:
            ax.fill_between(
                rows["step"].to_numpy(float),
                rows["ci95_lower"].to_numpy(float),
                rows["ci95_upper"].to_numpy(float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    ax.set_xlabel("Added boreholes")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["step"].unique()))
    ax.grid(color="0.92", lw=0.5)


def regret_row(regret: pd.DataFrame, policy: str, metric: str) -> pd.Series:
    selected = regret.loc[(regret["policy"] == policy) & (regret["metric"] == metric)]
    if selected.empty:
        raise KeyError(f"Missing regret summary: {policy}, {metric}")
    return selected.iloc[0]


def plot_regret(ax: plt.Axes, regret: pd.DataFrame) -> None:
    panel_title(ax, "d", "Trajectory efficiency and regret")
    for policy, label, color, _ in POLICIES:
        reduction = regret_row(regret, policy, "trajectory_loss_reduction")
        regret_value = regret_row(regret, policy, "mean_stepwise_regret")
        x = float(reduction["mean"])
        y = float(regret_value["mean"])
        ax.errorbar(
            x,
            y,
            xerr=np.asarray(
                [[x - float(reduction["ci95_lower"])], [float(reduction["ci95_upper"]) - x]]
            ),
            yerr=np.asarray(
                [[y - float(regret_value["ci95_lower"])], [float(regret_value["ci95_upper"]) - y]]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            ms=5.2 if policy == "expected_loss" else 3.8,
            lw=0.75,
            capsize=1.8,
            zorder=4 if policy == "expected_loss" else 2,
        )
    ax.axhline(0, color=INK, lw=0.7)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_xlabel("Mean engineering-loss reduction after additions")
    ax.set_ylabel("Mean stepwise regret")
    ax.grid(color="0.92", lw=0.5)
    ax.text(
        0.02,
        0.97,
        "Final loss ties after all candidates are used",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=INK,
    )


def main() -> None:
    apply_m1_style()
    detail, summary, regret, paired = require_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    detail.to_csv(SOURCE / "figure10_sequential_scene_level.csv", index=False)
    summary.to_csv(SOURCE / "figure10_sequential_summary.csv", index=False)
    regret.to_csv(SOURCE / "figure10_regret_summary.csv", index=False)
    paired.to_csv(SOURCE / "figure10_policy_vs_random_summary.csv", index=False)
    (SOURCE / "metadata.json").write_text(
        json.dumps(
            {
                "cycles": "Every step is reconstruct-select-add-observation-reconstruct; no retrospective fixed ranking is used.",
                "initial_design": "The evaluator starts from the predeclared sparse borehole set and adds five boreholes.",
                "scene_eligibility": "First ten immutable-manifest ID scenes with at least eight unique candidate boreholes (three initial plus five additions).",
                "posterior_members": 8,
                "sampling_steps": 3,
                "model_seeds": [41, 42, 43],
                "scenes_per_seed": 10,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 134 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.94, bottom=0.18, wspace=0.34, hspace=0.44)
    curve_panel(axes[0, 0], summary, "engineering_loss", "a", "Engineering decision loss", "Engineering loss")
    curve_panel(axes[0, 1], summary, "eic_rmse", "b", "Whole-volume reconstruction error", "EIC RMSE")
    curve_panel(axes[1, 0], summary, "mean_interval_width", "c", "Posterior contraction", "Mean EIC interval width")
    plot_regret(axes[1, 1], regret)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=5.8, bbox_to_anchor=(0.5, 0.035))
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
