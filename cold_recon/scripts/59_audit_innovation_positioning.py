from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.evaluation.innovation_positioning import (
    EVIDENCE_COLUMNS,
    build_innovation_positioning_audit,
    write_innovation_positioning_outputs,
)
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "teal": "#42949E",
    "green": "#2E9E44",
    "amber": "#C9A227",
    "red": "#B64342",
    "neutral": "#767676",
    "neutral_light": "#E4E4E4",
    "black": "#272727",
}


def _style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False


def _panel(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _short_dimension(value: str) -> str:
    return {
        "Multi-source observation tokens": "multi-source\nobservation\ntokens",
        "Conditional posterior neural operator": "posterior\nneural\noperator",
        "Physics and calibration gates": "physics and\ncalibration\ngates",
        "Rare cryostructure operating points": "rare\ncryostructure\noperating points",
        "Public transfer applicability": "public transfer\napplicability",
        "Posterior observation design": "posterior\nobservation\ndesign",
    }.get(str(value), str(value))


def _coverage_label(value: float) -> str:
    if value >= 0.99:
        return "full"
    if value >= 0.49:
        return "partial"
    return "open"


def _save(fig: plt.Figure, fig_dir: Path, stem: str) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("svg", "pdf", "png", "tiff"):
        path = fig_dir / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def save_innovation_positioning_figure(audit: pd.DataFrame, summary: dict, fig_dir: Path) -> list[Path]:
    if audit.empty:
        raise ValueError("innovation positioning audit is empty")
    _style()
    view = audit.copy().reset_index(drop=True)
    labels = view["innovation_dimension"].map(_short_dimension)

    fig = plt.figure(figsize=(7.4, 5.1))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.42, 1.0],
        height_ratios=[1.05, 0.95],
        wspace=0.42,
        hspace=0.46,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    matrix = view[list(EVIDENCE_COLUMNS)].astype(float).to_numpy()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "coverage", [PALETTE["neutral_light"], PALETTE["amber"], PALETTE["green"]]
    )
    im = ax_a.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    col_labels = ["method", "controlled", "baseline", "public", "boundary", "repro."]
    ax_a.set_xticks(np.arange(len(col_labels)))
    ax_a.set_xticklabels(col_labels, rotation=32, ha="right", fontsize=6)
    ax_a.set_yticks(np.arange(len(view)))
    ax_a.set_yticklabels(labels, fontsize=6)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_a.text(j, i, _coverage_label(float(matrix[i, j])), ha="center", va="center", fontsize=4.8)
    ax_a.set_title("evidence coverage behind each innovation claim")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.032, pad=0.012)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["open", "partial", "full"])
    cbar.ax.tick_params(labelsize=5.5)
    _panel(ax_a, "a", x=-0.08)

    y = np.arange(len(view))
    current = view["current_maturity"].astype(float).to_numpy()
    gap = view["eg_target_maturity"].astype(float).to_numpy() - current
    ax_b.barh(y, current, color=PALETTE["cold"], label="current")
    ax_b.barh(y, gap, left=current, color=PALETTE["neutral_light"], label="gap to prospective EG")
    ax_b.axvline(5.0, color=PALETTE["black"], lw=0.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=5.5)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 5.2)
    ax_b.set_xlabel("evidence maturity")
    ax_b.set_title("CG evidence versus prospective EG target")
    ax_b.legend(fontsize=5.4, loc="lower right")
    ax_b.grid(axis="x", color="0.9", lw=0.55)
    _panel(ax_b, "b", x=-0.18)

    score = 100.0 * view["evidence_coverage_score"].astype(float).to_numpy()
    colors = [
        PALETTE["green"] if value >= 90 else PALETTE["cold"] if value >= 75 else PALETTE["amber"]
        for value in score
    ]
    ax_c.barh(y, score, color=colors, edgecolor=PALETTE["black"], linewidth=0.25)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels, fontsize=5.5)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 105)
    ax_c.set_xlabel("coverage score (%)")
    ax_c.set_title("novelty claims are bounded by audited evidence")
    for yi, value in zip(y, score):
        ax_c.text(value + 1.5, yi, f"{value:.0f}", va="center", fontsize=5.5)
    ax_c.grid(axis="x", color="0.9", lw=0.55)
    _panel(ax_c, "c", x=-0.18)

    fig.suptitle("COLD-Recon innovation positioning is evidence-mapped rather than rhetorical", fontsize=9, y=0.995)
    return _save(fig, fig_dir, "innovation_positioning_audit")


def _write_source_data(path: Path, audit: pd.DataFrame, summary: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_rows = audit.copy()
    audit_rows.insert(0, "record_type", "innovation_dimension")
    summary_rows = pd.DataFrame([summary])
    summary_rows.insert(0, "record_type", "summary")
    pd.concat([audit_rows, summary_rows], ignore_index=True, sort=False).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    source_dir = Path("outputs/source_data")

    audit, summary = build_innovation_positioning_audit(table_dir)
    audit_path, summary_path = write_innovation_positioning_outputs(audit, summary, table_dir)
    source_path = _write_source_data(source_dir / "innovation_positioning_audit_source_data.csv", audit, summary)
    figures = save_innovation_positioning_figure(audit, summary, fig_dir)

    print(f"audit={audit_path}")
    print(f"summary={summary_path}")
    print(f"source_data={source_path}")
    for path in figures:
        print(f"figure={path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
