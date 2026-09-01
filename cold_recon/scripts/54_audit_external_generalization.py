from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.evaluation.external_generalization_audit import (
    build_external_generalization_audit,
    build_external_generalization_site_deltas,
)
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "cold_light": "#3775BA",
    "green": "#2E9E44",
    "red": "#B64342",
    "gold": "#C9A227",
    "teal": "#42949E",
    "neutral": "#767676",
    "neutral_light": "#D8D8D8",
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


def _panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _short_site(site: str) -> str:
    return {
        "Anaktuvuk Fire August": "Anaktuvuk\nAug.",
        "Anaktuvuk River Fire June": "Anaktuvuk\nJune",
        "Prudhoe Bay August September": "Prudhoe\nBay",
        "Itkillik June July": "Itkillik",
        "Tuktoyaktuk September": "Tuktoyaktuk",
    }.get(str(site), str(site).replace(" ", "\n", 1))


def _fmt(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


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


def save_external_generalization_figure(
    site_deltas: pd.DataFrame,
    audit: pd.DataFrame,
    fig_dir: Path,
    stem: str = "external_generalization_audit",
) -> list[Path]:
    if site_deltas.empty or audit.empty:
        raise ValueError("external generalization audit inputs are empty")
    _style()
    site_view = site_deltas.copy()
    site_view["site_label"] = site_view["site"].map(_short_site)
    site_view = site_view.sort_values("site")

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    y = np.arange(len(site_view))
    facies_delta = pd.to_numeric(site_view["facies_delta"], errors="coerce").to_numpy()
    colors = [PALETTE["green"] if np.isfinite(v) and v >= 0 else PALETTE["red"] for v in facies_delta]
    ax_a.barh(y, facies_delta, color=colors, edgecolor="black", linewidth=0.25)
    ax_a.axvline(0, color=PALETTE["black"], lw=0.65)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(site_view["site_label"], fontsize=6)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("facies accuracy delta vs spatial KNN")
    ax_a.grid(axis="x", color="0.9", lw=0.55)
    for yi, value in zip(y, facies_delta):
        if np.isfinite(value):
            ax_a.text(value + (0.008 if value >= 0 else -0.008), yi, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=5.5)
    _panel(ax_a, "a")

    eic_reduction = pd.to_numeric(site_view["eic_rmse_reduction_vs_best_simple"], errors="coerce").to_numpy()
    colors = [PALETTE["cold"] if np.isfinite(v) and v >= 0 else PALETTE["red"] for v in eic_reduction]
    ax_b.barh(y, 100.0 * eic_reduction, color=colors, edgecolor="black", linewidth=0.25)
    ax_b.axvline(0, color=PALETTE["black"], lw=0.65)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(site_view["site_label"], fontsize=6)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("EIC RMSE reduction vs best simple (%)")
    ax_b.grid(axis="x", color="0.9", lw=0.55)
    finite_reduction = 100.0 * eic_reduction[np.isfinite(eic_reduction)]
    if finite_reduction.size:
        ax_b.set_xlim(float(np.nanmin(finite_reduction)) - 1.4, float(np.nanmax(finite_reduction)) + 1.4)
    for yi, value in zip(y, eic_reduction):
        if np.isfinite(value):
            if value >= 0:
                ax_b.text(100.0 * value + 0.4, yi, f"{100.0 * value:+.1f}", va="center", ha="left", fontsize=5.5)
            else:
                ax_b.text(100.0 * value + 0.4, yi, f"{100.0 * value:+.1f}", va="center", ha="left", fontsize=5.5, color="white")
    _panel(ax_b, "b")

    wedge = audit[audit["task"].astype(str).eq("wedge-ice recall")].iloc[0]
    recall_vals = [float(wedge["baseline_value"]), float(wedge["model_value"])]
    precision_vals = [float(wedge["secondary_baseline_value"]), float(wedge["secondary_model_value"])]
    x = np.arange(2)
    width = 0.35
    ax_c.bar(x - width / 2, recall_vals, width=width, color=PALETTE["cold"], label="recall")
    ax_c.bar(x + width / 2, precision_vals, width=width, color=PALETTE["gold"], label="precision")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["Spatial\nKNN", "Recall\nhead"], fontsize=6)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel("pooled metric")
    ax_c.grid(axis="y", color="0.9", lw=0.55)
    ax_c.legend(fontsize=5.5, loc="upper right")
    for xi, rec, prec in zip(x, recall_vals, precision_vals):
        ax_c.text(xi - width / 2, rec + 0.025, _fmt(rec), ha="center", va="bottom", fontsize=5.5)
        ax_c.text(xi + width / 2, prec + 0.025, _fmt(prec), ha="center", va="bottom", fontsize=5.5)
    _panel(ax_c, "c", x=-0.16)

    heat_tasks = ["cryofacies", "EIC regression", "wedge-ice recall", "high-EIC event"]
    heat_cols = ["site_win_rate", "site_noninferior_rate"]
    matrix = []
    for task in heat_tasks:
        row = audit[audit["task"].astype(str).eq(task)].iloc[0]
        matrix.append([float(row[col]) if pd.notna(row[col]) else np.nan for col in heat_cols])
    mat = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad(color="#E6E6E6")
    im = ax_d.imshow(masked, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax_d.set_xticks(np.arange(len(heat_cols)))
    ax_d.set_xticklabels(["win\nrate", "noninferior\nrate"], fontsize=6)
    ax_d.set_yticks(np.arange(len(heat_tasks)))
    ax_d.set_yticklabels(["facies", "EIC", "wedge", "high-EIC"], fontsize=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            text_color = "white" if np.isfinite(value) and value >= 0.75 else "black"
            ax_d.text(j, i, "NA" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=5.5, color=text_color)
    ax_d.set_title("site-level robustness", fontsize=7, pad=4)
    fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    _panel(ax_d, "d", x=-0.16)

    fig.suptitle("External ArcticData holdouts show aggregate gains with site-level boundaries", fontsize=9, y=0.995)
    return _save(fig, fig_dir, stem)


def _write_source_data(path: Path, site_deltas: pd.DataFrame, audit: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    site = site_deltas.copy()
    site.insert(0, "record_type", "site_delta")
    summary = audit.copy()
    summary.insert(0, "record_type", "task_summary")
    pd.concat([site, summary], ignore_index=True, sort=False).to_csv(path, index=False)
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
    metrics_path = table_dir / "arcticdata_conditioned_diffusion_multisite_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing multi-site metrics table: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    site_deltas = build_external_generalization_site_deltas(metrics)
    audit = build_external_generalization_audit(site_deltas)
    site_path = table_dir / "external_generalization_site_deltas.csv"
    audit_path = table_dir / "external_generalization_audit.csv"
    source_path = source_dir / "external_generalization_audit_source_data.csv"
    site_deltas.to_csv(site_path, index=False)
    audit.to_csv(audit_path, index=False)
    _write_source_data(source_path, site_deltas, audit)
    figure_paths = save_external_generalization_figure(site_deltas, audit, fig_dir)

    print(f"site_deltas={site_path}")
    print(f"audit={audit_path}")
    print(f"source_data={source_path}")
    for path in figure_paths:
        print(f"figure={path}")
    print(audit[["task", "model_value", "baseline_value", "site_win_rate", "site_noninferior_rate", "failure_sites"]].to_string(index=False))


if __name__ == "__main__":
    main()
