from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.evaluation.transfer_failure_attribution import (
    build_transfer_failure_attribution_summary,
    build_transfer_failure_site_diagnostics,
)
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "green": "#2E9E44",
    "red": "#B64342",
    "gold": "#C9A227",
    "amber": "#D69D2A",
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


def _panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _short_site(site: str) -> str:
    return {
        "Anaktuvuk Fire August": "Anaktuvuk Aug.",
        "Anaktuvuk River Fire June": "Anaktuvuk June",
        "Prudhoe Bay August September": "Prudhoe Bay",
        "Itkillik June July": "Itkillik",
        "Tuktoyaktuk September": "Tuktoy.",
    }.get(str(site), str(site))


def _outcome_color(outcome: str) -> str:
    if outcome == "failure":
        return PALETTE["red"]
    if outcome == "noninferior":
        return PALETTE["amber"]
    return PALETTE["green"]


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


def save_transfer_failure_attribution_figure(
    site_diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    fig_dir: Path,
    stem: str = "transfer_failure_attribution",
) -> list[Path]:
    if site_diagnostics.empty or summary.empty:
        raise ValueError("transfer failure attribution inputs are empty")
    _style()
    site = site_diagnostics.copy().sort_values("site")
    site["site_label"] = site["site"].map(_short_site)

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.16, 1.0], height_ratios=[1.0, 1.0], wspace=0.58, hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    y = np.arange(len(site))
    model = pd.to_numeric(site["eic_model_rmse"], errors="coerce").to_numpy()
    best = pd.to_numeric(site["eic_best_simple_rmse"], errors="coerce").to_numpy()
    y_best = y - 0.055
    y_model = y + 0.055
    ax_a.plot(best, y_best, "o", color=PALETTE["neutral"], label="best simple", markersize=4.3)
    ax_a.plot(model, y_model, "o", color=PALETTE["cold"], label="COLD-Recon", markersize=4.3)
    for yi, b, m, outcome in zip(y, best, model, site["eic_transfer_outcome"]):
        ax_a.plot([b, m], [yi - 0.055, yi + 0.055], color=_outcome_color(str(outcome)), lw=1.0)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(site["site_label"], fontsize=6)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("EIC RMSE")
    ax_a.grid(axis="x", color="0.9", lw=0.55)
    ax_a.legend(fontsize=5.6, loc="lower right")
    _panel(ax_a, "a")

    x = pd.to_numeric(site["spatial_idw_advantage_vs_global"], errors="coerce")
    yy = pd.to_numeric(site["eic_model_gap_vs_best_simple"], errors="coerce")
    colors = [_outcome_color(str(outcome)) for outcome in site["eic_transfer_outcome"]]
    sizes = 30 + 1.0 * pd.to_numeric(site["holdout_n"], errors="coerce").fillna(0).to_numpy()
    ax_b.scatter(x, yy, s=sizes, c=colors, edgecolors="black", linewidths=0.35)
    ax_b.axhline(0, color=PALETTE["black"], lw=0.65)
    ax_b.axvline(0, color=PALETTE["black"], lw=0.65)
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(yy, errors="coerce").to_numpy(dtype=float)
    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values)]
    if finite_x.size:
        x_min = float(finite_x.min())
        x_max = float(finite_x.max())
        x_pad = max((x_max - x_min) * 0.16, 0.008)
        ax_b.set_xlim(x_min - x_pad, x_max + x_pad)
    else:
        x_min, x_max, x_pad = -0.01, 0.01, 0.004
    if finite_y.size:
        y_min = float(finite_y.min())
        y_max = float(finite_y.max())
        y_pad = max((y_max - y_min) * 0.18, 0.0009)
        ax_b.set_ylim(y_min - y_pad, y_max + y_pad)
    else:
        y_min, y_max, y_pad = -0.001, 0.001, 0.0003
    for xi, yi, label in zip(x, yy, site["site_label"]):
        if np.isfinite(xi) and np.isfinite(yi):
            x_offset = -0.0025 if float(xi) > x_max - x_pad * 0.8 else 0.0025
            y_offset = -0.00035 if float(yi) > y_max - y_pad * 0.8 else 0.00028
            if str(label) == "Itkillik":
                y_offset = 0.00036
            elif str(label) == "Tuktoy.":
                y_offset = -0.00042
            ha = "right" if x_offset < 0 else "left"
            va = "top" if y_offset < 0 else "bottom"
            ax_b.text(
                float(xi) + x_offset,
                float(yi) + y_offset,
                str(label).replace(" ", "\n", 1),
                fontsize=5.1,
                ha=ha,
                va=va,
                clip_on=True,
            )
    ax_b.set_xlabel("SpatialDepthIDW advantage vs global mean")
    ax_b.set_ylabel("COLD-Recon RMSE gap vs best simple")
    ax_b.grid(color="0.9", lw=0.55)
    _panel(ax_b, "b")

    assoc = summary[summary["spearman"].notna()].copy()
    assoc = assoc[assoc["signal"].ne("transfer readiness score")]
    assoc = assoc.sort_values("spearman")
    assoc["plot_signal"] = assoc["signal"].replace(
        {
            "training observations": "training obs.",
            "holdout observations": "holdout obs.",
            "holdout boreholes": "holdout boreholes",
            "spatial IDW advantage over global mean": "IDW advantage",
            "facies accuracy delta": "facies delta",
        }
    )
    y2 = np.arange(len(assoc))
    vals = pd.to_numeric(assoc["spearman"], errors="coerce").to_numpy()
    assoc_colors = [PALETTE["red"] if value < 0 else PALETTE["green"] for value in vals]
    ax_c.barh(y2, vals, color=assoc_colors, edgecolor="black", linewidth=0.25)
    ax_c.axvline(0, color=PALETTE["black"], lw=0.65)
    ax_c.set_yticks(y2)
    ax_c.set_yticklabels(assoc["plot_signal"], fontsize=5.6)
    ax_c.set_xlim(-1.05, 1.05)
    ax_c.set_xlabel("Spearman with EIC RMSE reduction")
    ax_c.grid(axis="x", color="0.9", lw=0.55)
    _panel(ax_c, "c", x=-0.18)

    component_cols = [
        "facies_noninferior",
        "eic_noninferior_vs_best_simple",
        "high_eic_noninferior_vs_spatial_idw",
        "wedge_recall_noninferior",
    ]
    component_labels = ["facies", "EIC", "high-EIC", "wedge"]
    matrix = []
    for _, row in site.iterrows():
        matrix.append([float(bool(row[col])) if col in row.index and pd.notna(row[col]) else np.nan for col in component_cols])
    mat = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad("#E6E6E6")
    im = ax_d.imshow(masked, vmin=0, vmax=1, aspect="auto", cmap=cmap)
    ax_d.set_xticks(np.arange(len(component_cols)))
    ax_d.set_xticklabels(component_labels, rotation=28, ha="right", fontsize=5.7)
    ax_d.set_yticks(np.arange(len(site)))
    ax_d.set_yticklabels(site["site_label"], fontsize=5.7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            ax_d.text(j, i, "NA" if not np.isfinite(value) else str(int(value)), ha="center", va="center", fontsize=5.2)
    ax_d.set_title("transfer readiness components", fontsize=7, pad=4)
    fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    _panel(ax_d, "d", x=-0.18)

    fig.suptitle("Compact-site spatial guard controls EIC transfer failures", fontsize=9, y=0.995)
    return _save(fig, fig_dir, stem)


def _write_source(path: Path, site_diagnostics: pd.DataFrame, summary: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    site = site_diagnostics.copy()
    site.insert(0, "record_type", "site_diagnostic")
    summary_out = summary.copy()
    summary_out.insert(0, "record_type", "summary")
    pd.concat([site, summary_out], ignore_index=True, sort=False).to_csv(path, index=False)
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

    site_path = table_dir / "external_generalization_site_deltas.csv"
    if not site_path.exists():
        raise FileNotFoundError(f"Missing external generalization site deltas: {site_path}")
    site_deltas = pd.read_csv(site_path)
    diagnostics = build_transfer_failure_site_diagnostics(site_deltas)
    summary = build_transfer_failure_attribution_summary(diagnostics)

    diagnostics_path = table_dir / "transfer_failure_site_diagnostics.csv"
    summary_path = table_dir / "transfer_failure_attribution_summary.csv"
    source_path = source_dir / "transfer_failure_attribution_source_data.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_source(source_path, diagnostics, summary)
    figure_paths = save_transfer_failure_attribution_figure(diagnostics, summary, fig_dir)

    print(f"site_diagnostics={diagnostics_path}")
    print(f"summary={summary_path}")
    print(f"source_data={source_path}")
    for path in figure_paths:
        print(f"figure={path}")
    print_cols = [
        col
        for col in [
            "site",
            "eic_transfer_outcome",
            "eic_model_gap_vs_best_simple",
            "spatial_idw_advantage_vs_global",
            "adaptive_eic_method",
            "adaptive_eic_transfer_guard_reason",
            "failure_attribution",
        ]
        if col in diagnostics.columns
    ]
    print(diagnostics[print_cols].to_string(index=False))


if __name__ == "__main__":
    main()
