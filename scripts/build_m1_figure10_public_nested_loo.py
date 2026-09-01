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
    / "figure10_public_nested_borehole_validation"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure10"

INK = "#263238"
BLUE = "#377eb8"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
TEAL = "#1b9e77"
GREY = "#9aa5ab"
RED = "#b23a48"

SITES = [
    ("usgs_eic", "USGS\n38 boreholes"),
    ("arcticdata_jago_ground_ice", "Jago River\n21 boreholes"),
    ("arcticdata_cryostratigraphy", "Arctic compilation\n50 evaluated folds"),
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
    ax.text(0.0, 1.035, f"({letter})", transform=ax.transAxes, fontsize=8.4, fontweight="bold")
    ax.text(0.10, 1.035, title, transform=ax.transAxes, fontsize=7.8, color=INK)


def metric_row(frame: pd.DataFrame, metric: str) -> pd.Series:
    selected = frame.loc[frame["metric"] == metric]
    if selected.empty:
        raise KeyError(f"Missing public metric: {metric}")
    return selected.iloc[0]


def require_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    summaries: dict[str, pd.DataFrame] = {}
    details: dict[str, pd.DataFrame] = {}
    missing: list[Path] = []
    for site, _ in SITES:
        summary_path = TABLES / f"m1_public_{site}_three_seed_summary.csv"
        detail_path = TABLES / f"m1_public_{site}_three_seed_detail.csv"
        for path in (summary_path, detail_path):
            if not path.exists():
                missing.append(path)
        if summary_path.exists() and detail_path.exists():
            summaries[site] = pd.read_csv(summary_path)
            details[site] = pd.read_csv(detail_path)
    metadata_path = TABLES / "m1_postlock_three_seed_metadata.json"
    if not metadata_path.exists():
        missing.append(metadata_path)
    if missing:
        raise FileNotFoundError(
            "Figure 10 cannot be built before nested complete-borehole LOO closes: "
            + ", ".join(str(path) for path in missing)
        )
    for site, _ in SITES:
        detail = details[site]
        if detail["model_seed"].nunique() != 3:
            raise RuntimeError(f"Public detail lacks three model seeds: {site}")
        if detail.duplicated(["model_seed", "held_group_id"]).any():
            raise RuntimeError(f"Duplicate outer borehole folds: {site}")
    return summaries, details, json.loads(metadata_path.read_text(encoding="utf-8"))


def plot_rmse(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "a", "Held-borehole support RMSE")
    x = np.arange(len(SITES), dtype=float)
    width = 0.19
    for offset, metric, color, marker, label in [
        (-width / 2, "outer_rmse", BLUE, "o", "deployment mean"),
        (width / 2, "outer_anchor_rmse", GREY, "s", "tree anchor"),
    ]:
        rows = [metric_row(summaries[site], metric) for site, _ in SITES]
        mean = np.asarray([float(row["mean"]) for row in rows])
        low = np.asarray([float(row["ci95_lower"]) for row in rows])
        high = np.asarray([float(row["ci95_upper"]) for row in rows])
        ax.errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            fmt=marker,
            color=color,
            ecolor=color,
            ms=4.2,
            lw=0.9,
            capsize=2,
            label=label,
        )
    ax.set_xticks(x, [label for _, label in SITES])
    ax.set_ylabel("Outer-fold EIC RMSE")
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(loc="best", fontsize=5.8)


def plot_difference(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "b", "Paired non-inferiority to tree anchor")
    rows = [metric_row(summaries[site], "rmse_difference_vs_anchor") for site, _ in SITES]
    mean = np.asarray([float(row["mean"]) for row in rows])
    low = np.asarray([float(row["ci95_lower"]) for row in rows])
    high = np.asarray([float(row["ci95_upper"]) for row in rows])
    y = np.arange(len(SITES))[::-1]
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack((mean - low, high - mean)),
        fmt="o",
        color=TEAL,
        ecolor=TEAL,
        ms=4.2,
        lw=0.9,
        capsize=2,
    )
    ax.axvline(0, color=INK, lw=0.7)
    ax.axvline(0.005, color=RED, lw=0.9, ls="--", label="NI margin +0.005")
    ax.set_yticks(y, [label.replace("\n", " ") for _, label in SITES])
    ax.set_xlabel("Model minus anchor EIC RMSE")
    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.legend(loc="best", fontsize=5.8)


def plot_calibration(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "c", "Outer-fold calibration and width")
    x = np.arange(len(SITES), dtype=float)
    width = 0.27
    raw_cov = np.asarray([float(metric_row(summaries[site], "raw_coverage_90")["mean"]) for site, _ in SITES])
    cal_cov = np.asarray([float(metric_row(summaries[site], "calibrated_coverage_90")["mean"]) for site, _ in SITES])
    raw_width = np.asarray([float(metric_row(summaries[site], "raw_width_90")["mean"]) for site, _ in SITES])
    cal_width = np.asarray([float(metric_row(summaries[site], "calibrated_width_90")["mean"]) for site, _ in SITES])
    ax.bar(x - width / 2, raw_cov, width, color=GREY, alpha=0.82, label="raw coverage")
    ax.bar(x + width / 2, cal_cov, width, color=BLUE, alpha=0.86, label="calibrated coverage")
    ax.axhline(0.90, color=RED, lw=0.8, ls="--")
    ax.set_xticks(x, [label for _, label in SITES])
    ax.set_ylabel("Coverage of nominal 90% interval")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color="0.92", lw=0.5)
    twin = ax.twinx()
    twin.plot(x, raw_width, color=ORANGE, marker="o", ls=":", lw=1.0, label="raw width")
    twin.plot(x, cal_width, color=PURPLE, marker="s", lw=1.1, label="calibrated width")
    twin.set_ylabel("Mean interval width", color=PURPLE)
    twin.tick_params(axis="y", labelcolor=PURPLE)
    lines = [*ax.containers[:2], *twin.get_lines()]
    labels = ["raw coverage", "calibrated coverage", "raw width", "calibrated width"]
    ax.legend(lines, labels, loc="lower right", fontsize=5.4)


def plot_fallback(ax: plt.Axes, summaries: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, "d", "Predeclared fallback gates")
    x = np.arange(len(SITES), dtype=float)
    metrics = [
        ("exact_anchor_fallback_applied", "exact fallback", INK),
        ("fallback_due_to_noninferiority", "inner NI failure", ORANGE),
        ("fallback_due_to_ood", "OOD screen", RED),
    ]
    width = 0.22
    for index, (metric, label, color) in enumerate(metrics):
        values = np.asarray([float(metric_row(summaries[site], metric)["mean"]) for site, _ in SITES])
        ax.bar(x + (index - 1) * width, values, width, color=color, alpha=0.82, label=label)
    ax.set_xticks(x, [label for _, label in SITES])
    ax.set_ylabel("Outer-fold fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.legend(loc="best", fontsize=5.6)


def main() -> None:
    apply_m1_style()
    summaries, details, metadata = require_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    for site, _ in SITES:
        summaries[site].to_csv(SOURCE / f"figure10_{site}_summary.csv", index=False)
        details[site].to_csv(SOURCE / f"figure10_{site}_detail.csv", index=False)
    (SOURCE / "metadata.json").write_text(
        json.dumps(
            {
                "postlock_metadata": metadata,
                "claim_boundary": "All metrics are held-borehole support predictions, not dense three-dimensional field validation.",
                "outer_fold_rule": "The held borehole is excluded from the tree anchor, adapter, calibration and OOD feature vector.",
                "noninferiority_margin": 0.005,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 122 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.91, top=0.94, bottom=0.11, wspace=0.44, hspace=0.44)
    plot_rmse(axes[0, 0], summaries)
    plot_difference(axes[0, 1], summaries)
    plot_calibration(axes[1, 0], summaries)
    plot_fallback(axes[1, 1], summaries)
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
