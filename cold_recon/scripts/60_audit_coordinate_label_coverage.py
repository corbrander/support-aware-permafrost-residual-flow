from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.evaluation.coordinate_label_coverage import (
    build_coordinate_label_coverage_audit,
    write_coordinate_label_coverage_outputs,
)
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "green": "#2E9E44",
    "amber": "#C9A227",
    "red": "#B64342",
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


def _panel(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _short_site(site: str) -> str:
    words = str(site).split()
    if len(words) <= 2:
        return str(site).replace(" ", "\n")
    return "\n".join([" ".join(words[:2]), " ".join(words[2:])])


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


def save_coordinate_label_coverage_figure(site_audit: pd.DataFrame, summary: dict, fig_dir: Path) -> list[Path]:
    if site_audit.empty:
        raise ValueError("coordinate-label coverage audit is empty")
    _style()
    view = site_audit.sort_values("n_georeferenced_units", ascending=False).head(12).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(view))

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.38, 1.0], height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.45)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    georef = view["n_georeferenced_units"].astype(float).to_numpy()
    non_georef = (view["n_units"].astype(float) - view["n_georeferenced_units"].astype(float)).clip(lower=0).to_numpy()
    ax_a.barh(y, georef, color=PALETTE["cold"], edgecolor=PALETTE["black"], linewidth=0.25, label="georeferenced")
    ax_a.barh(y, non_georef, left=georef, color=PALETTE["neutral_light"], edgecolor=PALETTE["black"], linewidth=0.25, label="not georeferenced")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([_short_site(site) for site in view["site"]], fontsize=5.8)
    ax_a.set_xlabel("vertical cryostratigraphic units")
    ax_a.set_title("public coordinate coverage by site")
    ax_a.grid(axis="x", color="0.9", lw=0.55)
    ax_a.legend(fontsize=5.5, loc="lower right")
    _panel(ax_a, "a")

    x = site_audit["georeferenced_unit_fraction"].astype(float).to_numpy()
    eic = site_audit["n_eic_measurements"].astype(float).to_numpy()
    boreholes = site_audit["n_georeferenced_boreholes"].astype(float).clip(lower=1.0).to_numpy()
    sizes = 12 + 3.2 * np.sqrt(boreholes)
    colors = np.where(eic > 0, PALETTE["teal"], PALETTE["neutral_light"])
    ax_b.scatter(x, eic, s=sizes, color=colors, edgecolor=PALETTE["black"], linewidth=0.3, alpha=0.88)
    ax_b.axvline(float(summary.get("georeferenced_unit_fraction", 0.0)), color=PALETTE["amber"], lw=0.8, ls="--")
    ax_b.set_xlim(0, 1.02)
    ax_b.set_xlabel("georeferenced unit fraction")
    ax_b.set_ylabel("EIC measurements")
    ax_b.set_title("coordinate coverage versus EIC labels")
    ax_b.grid(color="0.9", lw=0.55)
    _panel(ax_b, "b", x=-0.16)

    labels = ["facies\nunits", "EIC\nmeasurements", "high-EIC\nunits", "wedge-ice\nunits", "georef.\nunits"]
    values = [
        float(summary.get("n_model_facies_units", 0)),
        float(summary.get("n_eic_measurements", 0)),
        float(summary.get("n_high_eic_units", 0)),
        float(summary.get("n_wedge_ice_units", 0)),
        float(summary.get("n_georeferenced_units", 0)),
    ]
    colors_c = [PALETTE["cold"], PALETTE["teal"], PALETTE["amber"], PALETTE["green"], PALETTE["neutral"]]
    bars = ax_c.bar(np.arange(len(labels)), values, color=colors_c, edgecolor=PALETTE["black"], linewidth=0.25)
    ax_c.set_xticks(np.arange(len(labels)))
    ax_c.set_xticklabels(labels, fontsize=5.8)
    ax_c.set_ylabel("records")
    ax_c.set_title("label classes supporting EG-readiness")
    ax_c.grid(axis="y", color="0.9", lw=0.55)
    for bar, val in zip(bars, values):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.02, f"{val:,.0f}", ha="center", va="bottom", fontsize=5.8)
    _panel(ax_c, "c", x=-0.16)

    frac = 100.0 * float(summary.get("georeferenced_unit_fraction", 0.0))
    fig.suptitle(
        f"Public cryostratigraphy includes {frac:.1f}% georeferenced vertical labels, but not dense 3D ground truth",
        fontsize=9,
        y=0.995,
    )
    return _save(fig, fig_dir, "coordinate_label_coverage_audit")


def _write_source_data(path: Path, site_audit: pd.DataFrame, summary: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    site = site_audit.copy()
    site.insert(0, "record_type", "site_coordinate_label_coverage")
    summary_rows = pd.DataFrame([summary])
    summary_rows.insert(0, "record_type", "summary")
    pd.concat([site, summary_rows], ignore_index=True, sort=False).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    processed_dir = Path(config["paths"].get("processed_dir", "data/processed"))
    source_dir = Path("outputs/source_data")

    inventory_path = processed_dir / "arcticdata_cryostratigraphy_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing ArcticData inventory: {inventory_path}")
    result = build_coordinate_label_coverage_audit(pd.read_csv(inventory_path))
    audit_path, summary_path = write_coordinate_label_coverage_outputs(result, table_dir)
    source_path = _write_source_data(source_dir / "coordinate_label_coverage_audit_source_data.csv", result.site_audit, result.summary)
    figures = save_coordinate_label_coverage_figure(result.site_audit, result.summary, fig_dir)

    print(f"audit={audit_path}")
    print(f"summary={summary_path}")
    print(f"source_data={source_path}")
    for path in figures:
        print(f"figure={path}")
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
