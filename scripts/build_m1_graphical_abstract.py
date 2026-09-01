"""Build the optional Engineering Geology graphical abstract."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from m1_figure_style import INK, apply_m1_style, enforce_m1_typography


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "source_data" / "m1_figure5" / "test_id_combined_00008_seed41.npz"
OUTPUT = ROOT / "paper" / "engineering_geology_manuscript" / "figures" / "m1_submission_set" / "Graphical_Abstract"

BLUE = "#3B83BD"
TEAL = "#1B9E77"
ORANGE = "#D95F02"
RED = "#B23A48"
PALE = "#F3F6F7"


def box(ax, xy, width, height, text, edge, face="white", fontsize=8.0):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.015",
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.05,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="#6B7478",
        )
    )


def label(ax, letter, title):
    ax.text(0.0, 1.04, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.0)
    ax.text(0.16, 1.04, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.6)


def draw_evidence(ax):
    ax.set_xlim(0, 128)
    ax.set_ylim(12, 0)
    ax.set_facecolor("#EDF5F7")
    ax.fill_between([0, 128], [0, 0], [3.1, 3.8], color="#E5DDCD", zorder=0)
    for x, depth in [(20, 9.5), (69, 7.2), (108, 10.5)]:
        ax.plot([x, x], [0, depth], color=INK, linewidth=1.1)
        ax.scatter([x], [depth], s=22, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.add_patch(Rectangle((47, 1.8), 22, 4.4, facecolor="#E69F55", edgecolor=ORANGE, alpha=0.75))
    ax.plot([8, 120], [4.4, 4.4], color=TEAL, linewidth=2.0)
    ax.text(5, 11.3, "boreholes + temperature + NMR + ERT", fontsize=8.0, color=INK)
    ax.set_xticks([])
    ax.set_yticks([0, 6, 12])
    ax.set_ylabel("depth (m)")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)


def draw_model(ax):
    ax.set_axis_off()
    box(ax, (0.02, 0.61), 0.35, 0.22, "tree anchor\nEIC baseline", BLUE, "#EAF2F8")
    box(ax, (0.02, 0.20), 0.35, 0.22, "support-aware\nconditioning", TEAL, "#E8F6F2")
    box(ax, (0.53, 0.45), 0.44, 0.25, "conditional residual\nflow", ORANGE, "#FBEFE6")
    arrow(ax, (0.38, 0.72), (0.52, 0.61))
    arrow(ax, (0.38, 0.31), (0.52, 0.52))
    ax.text(0.75, 0.31, "64 posterior members", transform=ax.transAxes, ha="center", fontsize=8.0, color=INK)
    ax.text(0.75, 0.18, "OOD → anchor fallback", transform=ax.transAxes, ha="center", fontsize=8.0, color=RED)


def draw_outputs(ax, posterior, response, x, y, z):
    ax.set_axis_off()
    left = ax.inset_axes([0.00, 0.18, 0.47, 0.67])
    right = ax.inset_axes([0.53, 0.18, 0.47, 0.67])
    section_extent = [float(x.min()), float(x.max()), float(z.max()), float(z.min())]
    response_extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    left.imshow(posterior.T, extent=section_extent, aspect="auto", cmap="cividis", vmin=0.0, vmax=0.9, interpolation="nearest")
    right.imshow(response.T, extent=response_extent, origin="lower", aspect="auto", cmap="cividis", vmin=0.0,
                 vmax=float(np.quantile(response, 0.99)), interpolation="nearest")
    for inset, title in [(left, "posterior EIC"), (right, "excess-ice response")]:
        inset.set_title(title, fontsize=8.0, pad=2)
        inset.set_xticks([0, 64, 126])
        inset.set_yticks([0, 6, 12] if inset is left else [0, 64, 126])
        inset.tick_params(labelsize=8.0, length=2)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)
    right.set_yticklabels([])
    ax.text(0.50, 0.02, "calibrated EIC → screen for confirmation, not design", transform=ax.transAxes, ha="center", fontsize=8.0, color=INK)


def main():
    apply_m1_style(base_font_size=8.2)
    archive = np.load(ARCHIVE)
    y_index = archive["posterior_eic_mean"].shape[1] // 2
    posterior = archive["posterior_eic_mean"][:, y_index, :]
    z = archive["z"].astype(float)
    dz = float(np.mean(np.diff(z)))
    response = np.sum(archive["posterior_eic_mean"][:, :, z < 6.0], axis=-1) * dz

    fig = plt.figure(figsize=(7.5, 3.0), facecolor="white")
    grid = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.84, bottom=0.19, width_ratios=[0.92, 1.05, 1.30], wspace=0.28)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    label(axes[0], "A", "Heterogeneous evidence")
    label(axes[1], "B", "Auditable residual model")
    label(axes[2], "C", "Engineering screening outputs")
    draw_evidence(axes[0])
    draw_model(axes[1])
    draw_outputs(axes[2], posterior, response, archive["x"], archive["y"], z)
    fig.text(
        0.5,
        0.055,
        "Sparse mixed-support observations → anchored probabilistic EIC → response screening with explicit fallback",
        ha="center",
        va="center",
        fontsize=8.2,
        color=INK,
    )

    enforce_m1_typography(fig)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
