from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.posterior_uncertainty_alignment import build_posterior_uncertainty_alignment
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "cold_light": "#3775BA",
    "teal": "#42949E",
    "red": "#B64342",
    "gold": "#D4AE24",
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


def _short_model(name: str) -> str:
    return {
        "COLDReconLatentDiffusion": "Latent diff.",
        "COLDReconFNOOperatorDiffusion": "FNO diff.",
        "COLDReconRectifiedFlow": "Rectified flow",
        "COLDReconLatentDiffusionPhysicsTrained": "Physics-trained",
        "COLDReconLatentDiffusionPhysicsGuided": "Physics-guided",
        "COLDReconLatentDiffusionPhysicsRefined": "Physics-refined",
        "COLDReconLatentDiffusionCalibrated": "Interval-calib.",
    }.get(name, name)


def _panel(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def _ordered_models(audit: pd.DataFrame) -> list[str]:
    order = [
        "COLDReconLatentDiffusion",
        "COLDReconFNOOperatorDiffusion",
        "COLDReconRectifiedFlow",
        "COLDReconLatentDiffusionPhysicsTrained",
        "COLDReconLatentDiffusionPhysicsGuided",
        "COLDReconLatentDiffusionPhysicsRefined",
        "COLDReconLatentDiffusionCalibrated",
    ]
    present = set(audit["model"].astype(str)) if "model" in audit.columns else set()
    return [model for model in order if model in present]


def save_alignment_figure(audit: pd.DataFrame, fig_dir: Path, stem: str = "posterior_uncertainty_alignment") -> list[Path]:
    if audit.empty:
        raise ValueError("posterior uncertainty alignment audit is empty")
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = _ordered_models(audit)

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.82], wspace=0.42, hspace=0.55)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    eic = audit[audit["target"].eq("eic")].copy()
    eic["order"] = eic["model"].map({model: i for i, model in enumerate(models)})
    eic = eic.sort_values("order")
    y = np.arange(len(eic))
    enrichment = pd.to_numeric(eic["top_uncertainty_error_enrichment"], errors="coerce").to_numpy()
    capture = pd.to_numeric(eic["top_uncertainty_captures_top_error_rate"], errors="coerce").to_numpy()
    ax_a.barh(y, enrichment, color=PALETTE["cold"], height=0.68, label="top-uncertainty error enrichment")
    ax_a.axvline(1.0, color="0.45", lw=0.8, ls="--")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([_short_model(str(model)) for model in eic["model"]])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("EIC error in top uncertainty decile / global EIC error")
    ax_a.set_xlim(0.0, max(2.25, float(np.nanmax(enrichment)) + 0.15))
    ax_a.grid(axis="x", color="0.9", lw=0.6)
    for yi, value, cap in zip(y, enrichment, capture):
        if np.isfinite(value):
            ax_a.text(value + 0.03, yi, f"{value:.2f}x; cap {cap:.2f}", va="center", fontsize=5.5)
    _panel(ax_a, "a")

    targets = ["eic", "temperature", "unfrozen_water", "log_resistivity"]
    heat = np.full((len(models), len(targets)), np.nan, dtype=float)
    for i, model in enumerate(models):
        for j, target in enumerate(targets):
            row = audit[audit["model"].eq(model) & audit["target"].eq(target)]
            if not row.empty:
                heat[i, j] = float(pd.to_numeric(row["spearman_uncertainty_error"], errors="coerce").iloc[0])
    im = ax_b.imshow(heat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax_b.set_xticks(np.arange(len(targets)))
    ax_b.set_xticklabels(["EIC", "T", "water", "log rho"], rotation=25, ha="right")
    ax_b.set_yticks(np.arange(len(models)))
    ax_b.set_yticklabels([_short_model(model) for model in models], fontsize=5.4)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                ax_b.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=4.8, color=PALETTE["black"])
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02)
    cbar.set_label("Spearman rho", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.4)
    ax_b.set_title("uncertainty-error rank alignment", fontsize=7)
    _panel(ax_b, "b", x=-0.18)

    facies = audit[audit["target"].eq("facies")].copy()
    facies["order"] = facies["model"].map({model: i for i, model in enumerate(models)})
    facies = facies.sort_values("order")
    x = np.arange(len(facies))
    top_rate = pd.to_numeric(facies["top_uncertainty_error_mean"], errors="coerce").to_numpy()
    global_rate = pd.to_numeric(facies["global_error_mean"], errors="coerce").to_numpy()
    width = 0.36
    ax_c.bar(x - width / 2, global_rate, width=width, color=PALETTE["neutral"], label="global")
    ax_c.bar(x + width / 2, top_rate, width=width, color=PALETTE["teal"], label="top entropy decile")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([_short_model(str(model)) for model in facies["model"]], rotation=28, ha="right", fontsize=5.4)
    ax_c.set_ylabel("facies error rate")
    ax_c.set_ylim(0, max(0.16, float(np.nanmax([top_rate, global_rate])) + 0.025))
    ax_c.grid(axis="y", color="0.92", lw=0.55)
    ax_c.legend(fontsize=5.4, loc="upper right")
    ax_c.set_title("facies entropy boundary", fontsize=7)
    _panel(ax_c, "c", x=-0.18)

    fig.suptitle("Posterior uncertainty localizes EIC error but remains target-specific", fontsize=9, y=0.995)
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
    parser.add_argument("--quantile", type=float, default=0.90)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    pred_dir = Path(config["paths"]["predictions_dir"])
    audit = build_posterior_uncertainty_alignment(
        pred_dir,
        sample["fields"],
        quantile=float(args.quantile),
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "posterior_uncertainty_alignment.csv"
    audit.to_csv(table_path, index=False)
    figure_paths = save_alignment_figure(audit, fig_dir)
    print(f"audit={table_path}")
    for path in figure_paths:
        print(f"figure={path}")
    cols = [
        "model",
        "target",
        "spearman_uncertainty_error",
        "top_uncertainty_error_enrichment",
        "top_uncertainty_captures_top_error_rate",
    ]
    print(audit[[col for col in cols if col in audit.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
