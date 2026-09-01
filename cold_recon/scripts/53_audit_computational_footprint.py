from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.evaluation.computational_footprint import build_computational_footprint
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


def _panel(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def _short_model(name: str) -> str:
    return {
        "IDW": "IDW",
        "GradientBoosting": "GB",
        "SparseUNet3D": "3D U-Net",
        "COLDReconImplicit": "Implicit",
        "COLDReconLatentDiffusion": "Latent diff.",
        "COLDReconFNOOperatorDiffusion": "FNO diff.",
        "COLDReconRectifiedFlow": "Rectified flow",
        "COLDReconLatentDiffusionPhysicsTrained": "Physics-trained",
        "COLDReconLatentDiffusionRareFaciesHybrid": "Rare hybrid",
        "COLDReconLatentDiffusionPhysicsGuided": "Physics-guided",
        "COLDReconLatentDiffusionPhysicsRefined": "Physics-refined",
    }.get(name, name)


def _model_color(name: str) -> str:
    if name == "COLDReconFNOOperatorDiffusion":
        return PALETTE["red"]
    if name == "COLDReconLatentDiffusionRareFaciesHybrid":
        return PALETTE["gold"]
    if str(name).startswith("COLDRecon"):
        return PALETTE["cold"]
    return PALETTE["neutral"]


def _selected(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    view = df[df["model"].isin(models)].copy()
    view["order"] = view["model"].map({model: i for i, model in enumerate(models)})
    return view.sort_values("order")


def save_footprint_figure(
    footprint: pd.DataFrame,
    fig_dir: Path,
    stem: str = "computational_footprint_summary",
) -> list[Path]:
    if footprint.empty:
        raise ValueError("computational footprint table is empty")
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        "SparseUNet3D",
        "COLDReconImplicit",
        "COLDReconLatentDiffusion",
        "COLDReconFNOOperatorDiffusion",
        "COLDReconRectifiedFlow",
        "COLDReconLatentDiffusionPhysicsTrained",
        "COLDReconLatentDiffusionRareFaciesHybrid",
        "COLDReconLatentDiffusionPhysicsRefined",
    ]
    view = _selected(footprint, selected)

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 0.9], wspace=0.42, hspace=0.52)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    scatter = view.dropna(subset=["total_params_m", "mean_iou"]).copy()
    x = pd.to_numeric(scatter["total_params_m"], errors="coerce").to_numpy()
    y = pd.to_numeric(scatter["mean_iou"], errors="coerce").to_numpy()
    sizes = np.clip(pd.to_numeric(scatter["prediction_mb"], errors="coerce").fillna(1.0).to_numpy(), 1.0, 80.0) * 4.0
    colors = [_model_color(str(model)) for model in scatter["model"]]
    ax_a.scatter(x, y, s=sizes, c=colors, edgecolors="black", linewidths=0.35, alpha=0.95)
    label_offsets = {
        "SparseUNet3D": (1.10, 0.006),
        "COLDReconImplicit": (1.06, 0.006),
        "COLDReconLatentDiffusion": (1.05, -0.010),
        "COLDReconFNOOperatorDiffusion": (1.06, 0.004),
        "COLDReconRectifiedFlow": (1.05, -0.001),
        "COLDReconLatentDiffusionPhysicsTrained": (1.05, 0.012),
        "COLDReconLatentDiffusionRareFaciesHybrid": (1.05, -0.019),
        "COLDReconLatentDiffusionPhysicsRefined": (1.05, -0.028),
    }
    for xi, yi, model in zip(x, y, scatter["model"]):
        x_mul, y_add = label_offsets.get(str(model), (1.05, 0.004))
        ax_a.text(float(xi) * x_mul if xi > 0 else 0.02, float(yi) + y_add, _short_model(str(model)), fontsize=5.3)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("trainable parameters (million, log scale)")
    ax_a.set_ylabel("mean facies IoU")
    ax_a.set_ylim(0.08, max(0.60, float(np.nanmax(y)) + 0.03))
    ax_a.grid(color="0.9", lw=0.55)
    _panel(ax_a, "a")

    artifact_models = [
        "COLDReconImplicit",
        "COLDReconLatentDiffusion",
        "COLDReconFNOOperatorDiffusion",
        "COLDReconRectifiedFlow",
        "COLDReconLatentDiffusionPhysicsTrained",
        "COLDReconLatentDiffusionRareFaciesHybrid",
        "COLDReconLatentDiffusionPhysicsRefined",
    ]
    art = _selected(footprint, artifact_models)
    y_pos = np.arange(len(art))
    checkpoint = pd.to_numeric(art["checkpoint_mb"], errors="coerce").fillna(0.0).to_numpy()
    prediction = pd.to_numeric(art["prediction_mb"], errors="coerce").fillna(0.0).to_numpy()
    ax_b.barh(y_pos, checkpoint, color=PALETTE["neutral"], label="checkpoint")
    ax_b.barh(y_pos, prediction, left=checkpoint, color=PALETTE["cold"], label="prediction")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([_short_model(str(model)) for model in art["model"]], fontsize=5.4)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("artifact size (MB)")
    ax_b.grid(axis="x", color="0.9", lw=0.55)
    ax_b.legend(fontsize=5.5, loc="lower right")
    _panel(ax_b, "b", x=-0.18)

    gen = art[art["model"].astype(str).str.startswith("COLDRecon")].copy()
    x_pos = np.arange(len(gen))
    samples = pd.to_numeric(gen["posterior_samples"], errors="coerce").to_numpy()
    epochs = pd.to_numeric(gen["training_epochs"], errors="coerce").to_numpy()
    width = 0.36
    ax_c.bar(x_pos - width / 2, np.nan_to_num(samples, nan=0.0), width=width, color=PALETTE["teal"], label="posterior samples")
    ax_c2 = ax_c.twinx()
    ax_c2.bar(x_pos + width / 2, np.nan_to_num(epochs, nan=0.0), width=width, color=PALETTE["gold"], label="training epochs")
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([_short_model(str(model)) for model in gen["model"]], rotation=27, ha="right", fontsize=5.2)
    ax_c.set_ylabel("samples")
    ax_c2.set_ylabel("epochs")
    ax_c.grid(axis="y", color="0.9", lw=0.55)
    lines = [
        plt.Line2D([0], [0], color=PALETTE["teal"], lw=6, label="posterior samples"),
        plt.Line2D([0], [0], color=PALETTE["gold"], lw=6, label="training epochs"),
    ]
    ax_c.legend(handles=lines, fontsize=5.5, loc="upper right")
    _panel(ax_c, "c", x=-0.18)

    fig.suptitle("COLD-Recon performance gains have an auditable computational footprint", fontsize=9, y=0.995)
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
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    root = Path(".").resolve()
    table_dir = Path(config["paths"]["tables_dir"])
    pred_dir = Path(config["paths"]["predictions_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])

    footprint = build_computational_footprint(table_dir, pred_dir, root)
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "computational_footprint.csv"
    footprint.to_csv(table_path, index=False)
    figure_paths = save_footprint_figure(footprint, fig_dir)

    print(f"footprint={table_path}")
    for path in figure_paths:
        print(f"figure={path}")
    cols = [
        "model",
        "total_params_m",
        "checkpoint_mb",
        "prediction_mb",
        "posterior_samples",
        "training_epochs",
        "mean_iou",
    ]
    print(footprint[[col for col in cols if col in footprint.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
