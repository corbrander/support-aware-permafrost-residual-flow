from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m1_figure_style import apply_m1_style, enforce_m1_typography, export_m1_figure
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "source_data" / "m1_figure5" / "test_id_combined_00008_seed41.npz"
ARCHIVE_META = ARCHIVE.with_suffix(".json")
DETAIL = ROOT / "outputs" / "m1_support_guided" / "formal_fixed_scene" / "m1_test_id_seed41_detail.csv"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure5_fixed_scene_reconstruction"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure5"

INK = "#263238"
BLUE = "#377eb8"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.3,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def show(
    fig: plt.Figure,
    ax: plt.Axes,
    field: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    letter: str,
    title: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    label: str,
) -> None:
    image = ax.imshow(
        field.T,
        origin="upper",
        extent=(float(x.min()), float(x.max()), float(z.max()), float(z.min())),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.text(0.0, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8.2, fontweight="normal", ha="left")
    ax.text(0.11, 1.04, title, transform=ax.transAxes, fontsize=7.5, color=INK, ha="left")
    ax.set_ylabel("depth (m)")
    ax.set_xticks([0, 64, 126])
    ax.set_yticks([0, 6, 11.75])
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.075, pad=0.17)
    bar.set_label(label, fontsize=5.8, labelpad=1)
    bar.ax.tick_params(labelsize=5.2, length=1.5, pad=1)


def main() -> None:
    apply_m1_style()
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    values = np.load(ARCHIVE, allow_pickle=False)
    metadata = json.loads(ARCHIVE_META.read_text(encoding="utf-8"))
    if metadata.get("selection_status") != "locked before reconstruction inspection":
        raise RuntimeError("The example scene was not prospectively locked.")
    scene_id = str(values["scene_id"].item())
    if scene_id != "test_id_combined_00008":
        raise RuntimeError(f"Unexpected scene: {scene_id}")
    y_index = int(len(values["y"]) // 2)
    threshold_index = int(np.argmin(np.abs(values["event_thresholds"].astype(float) - 0.30)))
    truth = values["truth_eic"][:, y_index, :].astype(float)
    anchor = values["anchor_eic"][:, y_index, :].astype(float)
    posterior = values["posterior_eic_mean"][:, y_index, :].astype(float)
    absolute_error = np.abs(posterior - truth)
    posterior_std = values["posterior_eic_std"][:, y_index, :].astype(float)
    event = values["event_probability"][threshold_index, :, y_index, :].astype(float)
    x = values["x"].astype(float)
    z = values["z"].astype(float)

    detail = pd.read_csv(DETAIL).iloc[0]
    source = pd.DataFrame(
        {
            "x_m": np.repeat(x, len(z)),
            "z_m": np.tile(z, len(x)),
            "truth_eic": truth.reshape(-1),
            "tree_anchor_eic": anchor.reshape(-1),
            "posterior_eic_mean": posterior.reshape(-1),
            "posterior_absolute_error": absolute_error.reshape(-1),
            "posterior_eic_std": posterior_std.reshape(-1),
            "event_probability_eic_gt_0p30": event.reshape(-1),
        }
    )
    SOURCE.mkdir(parents=True, exist_ok=True)
    source.to_csv(SOURCE / "figure5_midplane_source.csv", index=False)
    figure_metadata = {
        **metadata,
        "section_rule": "fixed midpoint y index; not selected using reconstruction error",
        "section_y_index": y_index,
        "section_y_m": float(values["y"][y_index]),
        "event_threshold": float(values["event_thresholds"][threshold_index]),
        "full_volume_eic_rmse": float(detail["eic_rmse"]),
        "full_volume_anchor_eic_rmse": float(detail["anchor_eic_rmse"]),
        "ood_score": float(values["ood_score"].item()),
        "exact_anchor_fallback_applied": bool(values["exact_anchor_fallback_applied"].item()),
        "display_limits": {
            "truth_anchor_posterior_eic": [0.0, 0.9],
            "absolute_error": [0.0, 0.45],
            "posterior_std": [0.0, 0.20],
            "event_probability": [0.0, 1.0],
        },
    }
    (SOURCE / "figure5_metadata.json").write_text(json.dumps(figure_metadata, indent=2), encoding="utf-8")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 119 / 25.4))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.915, bottom=0.105, wspace=0.26, hspace=0.55)
    show(fig, axes[0, 0], truth, x, z, "a", "Controlled truth", cmap="cividis", vmin=0, vmax=0.9, label="EIC")
    show(fig, axes[0, 1], anchor, x, z, "b", "Tree anchor", cmap="cividis", vmin=0, vmax=0.9, label="EIC")
    show(fig, axes[0, 2], posterior, x, z, "c", "Posterior mean", cmap="cividis", vmin=0, vmax=0.9, label="EIC")
    show(fig, axes[1, 0], absolute_error, x, z, "d", "Absolute posterior error", cmap="magma", vmin=0, vmax=0.45, label="absolute EIC error")
    show(fig, axes[1, 1], posterior_std, x, z, "e", "Posterior standard deviation", cmap="viridis", vmin=0, vmax=0.20, label="EIC standard deviation")
    show(fig, axes[1, 2], event, x, z, "f", r"Event probability, EIC $>0.30$", cmap="inferno", vmin=0, vmax=1.0, label="probability")
    fig.text(
        0.5,
        0.972,
        f"Prospectively locked scene: {scene_id} | full-volume RMSE {float(detail['eic_rmse']):.3f} (anchor {float(detail['anchor_eic_rmse']):.3f}) | 64 members",
        ha="center",
        va="top",
        fontsize=7.2,
        color=INK,
    )
    enforce_m1_typography(fig)
    export_m1_figure(fig, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "source_rows": int(len(source)), **figure_metadata}, indent=2))


if __name__ == "__main__":
    main()
