from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.voi_backtest import build_voi_backtest, write_voi_backtest_outputs
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "cold_light": "#3775BA",
    "teal": "#42949E",
    "gold": "#D4AE24",
    "red": "#B64342",
    "neutral": "#767676",
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


def _panel(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def _target_label(name: str) -> str:
    return {
        "composite_error": "composite",
        "eic_abs_error": "EIC abs. error",
        "facies_error": "facies error",
        "high_eic_mismatch": "high-EIC mismatch",
        "wedge_miss": "wedge miss",
    }.get(str(name), str(name).replace("_", " "))


def _predictor_label(name: str) -> str:
    return {
        "voi_score": "VOI score",
        "uncertainty": "posterior uncertainty",
        "ice_rich_ambiguity": "ice-rich ambiguity",
        "thaw_sensitive_eic_proxy": "thaw-sensitive EIC",
        "eic_gradient_proxy": "EIC gradient",
        "novelty": "novelty",
    }.get(str(name), str(name).replace("_", " "))


def _write_source_data(path: Path, audit: pd.DataFrame, summary: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = audit.copy()
    rows.insert(1, "summary_readiness_status", summary.get("readiness_status", "not_available"))
    rows.insert(2, "summary_boundary", summary.get("readiness_boundary", ""))
    rows.to_csv(path, index=False)
    return path


def save_voi_backtest_figure(audit: pd.DataFrame, summary: dict, fig_dir: Path, stem: str = "voi_backtest_audit") -> list[Path]:
    if audit.empty:
        raise ValueError("VOI backtest audit is empty")
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    metric = audit[audit["record_type"].eq("target_metric")].copy()
    corr = audit[audit["record_type"].eq("component_correlation")].copy()
    boreholes = audit[audit["record_type"].eq("selected_borehole")].copy()
    target_order = ["composite_error", "high_eic_mismatch", "facies_error", "wedge_miss", "eic_abs_error"]
    metric["order"] = metric["target"].map({name: i for i, name in enumerate(target_order)})
    metric = metric.sort_values("order")
    corr["spearman_predictor_error"] = pd.to_numeric(corr["spearman_predictor_error"], errors="coerce")
    corr = corr.sort_values("spearman_predictor_error", ascending=True)

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.82], wspace=0.42, hspace=0.56)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    y = np.arange(len(metric))
    enrichment = pd.to_numeric(metric["top_voi_error_enrichment"], errors="coerce").to_numpy()
    colors = [PALETTE["cold"] if str(t) == "composite_error" else PALETTE["teal"] for t in metric["target"]]
    ax_a.barh(y, enrichment, color=colors, height=0.66)
    ax_a.axvline(1.0, color="0.45", lw=0.8, ls="--")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([_target_label(t) for t in metric["target"]])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("top VOI decile error / global error")
    ax_a.set_xlim(0.0, max(1.8, float(np.nanmax(enrichment)) + 0.16))
    ax_a.grid(axis="x", color="0.9", lw=0.6)
    for yi, value, capture in zip(y, enrichment, pd.to_numeric(metric["top_voi_captures_top_error_rate"], errors="coerce")):
        if np.isfinite(value):
            ax_a.text(value + 0.03, yi, f"{value:.2f}x; cap {capture:.2f}", va="center", fontsize=5.6)
    _panel(ax_a, "a")

    cx = pd.to_numeric(corr["spearman_predictor_error"], errors="coerce").to_numpy()
    cy = np.arange(len(corr))
    ccols = [PALETTE["cold"] if str(p) == "voi_score" else PALETTE["neutral"] for p in corr["predictor"]]
    ax_b.barh(cy, cx, color=ccols, height=0.62)
    ax_b.axvline(0.0, color="0.35", lw=0.8)
    ax_b.set_yticks(cy)
    ax_b.set_yticklabels([_predictor_label(p) for p in corr["predictor"]], fontsize=5.7)
    ax_b.set_xlabel("Spearman rho with composite error")
    ax_b.set_xlim(min(-0.15, float(np.nanmin(cx)) - 0.04), max(0.35, float(np.nanmax(cx)) + 0.04))
    ax_b.grid(axis="x", color="0.92", lw=0.55)
    ax_b.set_title("fixed components remain auditable", fontsize=7)
    _panel(ax_b, "b", x=-0.18)

    if not boreholes.empty:
        boreholes = boreholes.sort_values("rank").head(8)
        bx = np.arange(len(boreholes))
        bscore = pd.to_numeric(boreholes["voi_score"], errors="coerce").to_numpy()
        berror = pd.to_numeric(boreholes["composite_error"], errors="coerce").to_numpy()
        ax_c.bar(bx, berror, color=PALETTE["gold"], width=0.66, label="realized composite error")
        ax_c2 = ax_c.twinx()
        ax_c2.plot(bx, bscore, color=PALETTE["black"], marker="o", ms=3, lw=1.0, label="VOI score")
        ax_c.set_xticks(bx)
        ax_c.set_xticklabels([str(int(v)) for v in boreholes["rank"]])
        ax_c.set_ylabel("composite error")
        ax_c2.set_ylabel("VOI score")
        ax_c.set_xlabel("ranked borehole target")
        ax_c.grid(axis="y", color="0.92", lw=0.55)
        ax_c.set_title("recommended columns are traceable", fontsize=7)
    else:
        ax_c.text(0.5, 0.5, "No selected boreholes", ha="center", va="center", transform=ax_c.transAxes)
        ax_c.set_axis_off()
    _panel(ax_c, "c", x=-0.18)

    fig.suptitle(
        "Retrospective full-field VOI backtest supports observation-design readiness, not prospective validation",
        fontsize=8.8,
        y=0.995,
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--posterior", default="outputs/predictions/diffusion_posterior_physics_trained.npz")
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--quantile", type=float, default=0.90)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    posterior = dict(np.load(args.posterior, allow_pickle=False))
    posterior.setdefault("grid_x", sample["grid"]["x"].astype(np.float32))
    posterior.setdefault("grid_y", sample["grid"]["y"].astype(np.float32))
    posterior.setdefault("grid_z", sample["grid"]["z"].astype(np.float32))

    result = build_voi_backtest(
        posterior,
        sample["fields"],
        observations=sample["observations"],
        max_depth=float(args.max_depth),
        quantile=float(args.quantile),
    )
    table_dir = Path(config["paths"]["tables_dir"])
    source_dir = Path(config["project"]["output_root"]) / "source_data"
    fig_dir = Path(config["paths"]["figures_dir"])
    audit_path, summary_path = write_voi_backtest_outputs(result, table_dir)
    source_path = _write_source_data(source_dir / "voi_backtest_audit_source_data.csv", result.audit, result.summary)
    figure_paths = save_voi_backtest_figure(result.audit, result.summary, fig_dir)

    print(f"audit={audit_path}")
    print(f"summary={summary_path}")
    print(f"source_data={source_path}")
    for path in figure_paths:
        print(f"figure={path}")
    print(
        "readiness_status="
        f"{result.summary['readiness_status']} composite_enrichment={result.summary['composite_top_voi_error_enrichment']:.6f} "
        f"high_eic_enrichment={result.summary['high_eic_top_voi_error_enrichment']:.6f}"
    )


if __name__ == "__main__":
    main()
