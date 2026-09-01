from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from m1_figure_style import INK, apply_m1_style, export_m1_figure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure3_support_aware_residual_flow"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure03_architecture"

BLUE = "#377eb8"
SKY = "#9ecae1"
TEAL = "#1b9e77"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
RED = "#b23a48"
GOLD = "#d6a632"
GREY = "#607d8b"
LIGHT = "#f7f9fa"


def box(ax, xy, wh, title, lines, *, color, label, fontsize=7.4):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.010",
        linewidth=0.9,
        edgecolor=color,
        facecolor=LIGHT,
    )
    ax.add_patch(patch)
    ax.text(x + 0.014, y + h - 0.028, f"({label})", color=INK, fontsize=9.0, va="top")
    ax.text(x + 0.057, y + h - 0.028, title, color=color, fontsize=8.2, va="top")
    ax.text(
        x + 0.016, y + h - 0.078, "\n".join(lines),
        color=INK, fontsize=fontsize, va="top", linespacing=1.26,
    )
    return patch


def arrow(ax, start, end, *, color=GREY, rad=0.0, width=1.0):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9,
            linewidth=width, color=color, connectionstyle=f"arc3,rad={rad}",
        )
    )


def support_icons(ax, x, y):
    xs = [x + i * 0.033 for i in range(5)]
    ax.add_patch(Circle((xs[0], y), 0.006, facecolor=BLUE, edgecolor=INK, linewidth=0.5))
    ax.plot([xs[1], xs[1]], [y - 0.014, y + 0.014], color=TEAL, lw=3.0, solid_capstyle="round")
    ax.add_patch(Rectangle((xs[2] - 0.010, y - 0.011), 0.020, 0.022, facecolor=SKY, edgecolor=INK, lw=0.5))
    for radius, alpha in ((0.015, 0.12), (0.010, 0.22), (0.005, 0.48)):
        ax.add_patch(Circle((xs[3], y), radius, facecolor=PURPLE, edgecolor="none", alpha=alpha))
    ax.plot([xs[4] - 0.014, xs[4] + 0.014], [y + 0.011, y - 0.011], color=ORANGE, lw=1.1)
    ax.plot([xs[4] - 0.014, xs[4] + 0.014], [y - 0.011, y + 0.011], color=ORANGE, lw=1.1)


def factorized_stack(ax, x, y):
    for idx, color in enumerate((BLUE, TEAL, ORANGE, PURPLE)):
        offset = idx * 0.008
        ax.add_patch(
            Polygon(
                [[x + offset, y + offset], [x + 0.040 + offset, y + offset],
                 [x + 0.050 + offset, y + 0.010 + offset], [x + 0.010 + offset, y + 0.010 + offset]],
                closed=True, facecolor=color, edgecolor="white", linewidth=0.4, alpha=0.88,
            )
        )


def main() -> None:
    apply_m1_style(base_font_size=8.2)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 132 / 25.4))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.988, bottom=0.012)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_y, top_h = 0.700, 0.270
    top_w, gap = 0.225, 0.020
    top_x = [0.015 + i * (top_w + gap) for i in range(4)]
    box(ax, (top_x[0], top_y), (top_w, top_h), "Heterogeneous supports",
        ["Point | interval | volume", "Gaussian kernel | 0 °C crossing", "Value, support, uncertainty, quality"],
        color=BLUE, label="A")
    support_icons(ax, top_x[0] + 0.026, top_y + 0.046)
    box(ax, (top_x[1], top_y), (top_w, top_h), "Conventional anchor",
        ["24-tree observation-derived volume", "Factorized 14-channel state", "Rebuilt after deletion; exact fallback"],
        color=TEAL, label="B")
    factorized_stack(ax, top_x[1] + 0.080, top_y + 0.038)
    box(ax, (top_x[2], top_y), (top_w, top_h), "Hybrid conditioning",
        ["37-channel raster pathway", "52-dimensional irregular tokens", "Support-aware cross-attention"],
        color=PURPLE, label="C")
    ax.text(top_x[2] + 0.052, top_y + 0.043, "raster", color=BLUE, fontsize=7.1)
    arrow(ax, (top_x[2] + 0.105, top_y + 0.050), (top_x[2] + 0.137, top_y + 0.050))
    ax.text(top_x[2] + 0.145, top_y + 0.043, "tokens", color=ORANGE, fontsize=7.1)
    box(ax, (top_x[3], top_y), (top_w, top_h), "Frozen factorized codec",
        ["64 × 64 × 48 physical grid", "14 physical to 16 latent channels", "Bounded physical decoder"],
        color=GOLD, label="D")
    for idx in range(3):
        arrow(ax, (top_x[idx] + top_w + 0.002, top_y + top_h / 2),
              (top_x[idx + 1] - 0.002, top_y + top_h / 2))

    mid_y, mid_h = 0.365, 0.270
    mid_w, mid_gap = 0.300, 0.035
    mid_x = [0.015 + i * (mid_w + mid_gap) for i in range(3)]
    box(ax, (mid_x[0], mid_y), (mid_w, mid_h), "Safe posterior decomposition",
        ["z = z0 + g(c)b(c) + s(c)a", "Tree mean | gated bias | local scale", "E[a | c] = 0 preserves the mean"],
        color=TEAL, label="E", fontsize=7.6)
    for px, color, label in ((mid_x[0] + 0.065, GREY, "z0"),
                             (mid_x[0] + 0.150, BLUE, "g b"),
                             (mid_x[0] + 0.235, ORANGE, "s a")):
        ax.add_patch(Circle((px, mid_y + 0.045), 0.022, facecolor=color, edgecolor="white", lw=0.5, alpha=0.88))
        ax.text(px, mid_y + 0.045, label, ha="center", va="center", color="white", fontsize=7.0)
    box(ax, (mid_x[1], mid_y), (mid_w, mid_h), "Conditional residual flow",
        ["Three local-spectral velocity blocks", "10 Heun predictor-corrector steps", "Support likelihood at every step"],
        color=ORANGE, label="F", fontsize=7.6)
    nodes = [(mid_x[1] + 0.065 + 0.050 * i, mid_y + 0.045) for i in range(5)]
    for idx, (nx, ny) in enumerate(nodes):
        ax.add_patch(Circle((nx, ny), 0.014, facecolor=GREY if idx == 0 else ORANGE, edgecolor="white", lw=0.5))
        if idx < len(nodes) - 1:
            arrow(ax, (nx + 0.016, ny), (nodes[idx + 1][0] - 0.016, ny), color=ORANGE, width=0.8)
    box(ax, (mid_x[2], mid_y), (mid_w, mid_h), "Decoded posterior products",
        ["64 factorized three-dimensional members", "Continuous EIC means and spreads", "Raw posterior retained for audit"],
        color=BLUE, label="G", fontsize=7.6)
    for idx in range(4):
        ax.add_patch(Rectangle((mid_x[2] + 0.105 + idx * 0.020, mid_y + 0.025 + idx * 0.006),
                               0.080, 0.050, facecolor=SKY, edgecolor=BLUE, lw=0.5,
                               alpha=0.25 + idx * 0.12))
    arrow(ax, (top_x[3] + top_w / 2, top_y), (mid_x[2] + mid_w / 2, mid_y + mid_h))
    arrow(ax, (mid_x[0] + mid_w, mid_y + mid_h / 2), (mid_x[1], mid_y + mid_h / 2))
    arrow(ax, (mid_x[1] + mid_w, mid_y + mid_h / 2), (mid_x[2], mid_y + mid_h / 2))

    bot_y, bot_h = 0.035, 0.265
    bot_w = 0.465
    box(ax, (0.015, bot_y), (bot_w, bot_h), "Validation-only deployment control",
        ["Dual-max OOD score: pattern + tree context", "0.95: attenuate bias and inflate intervals", "0.99: restore exact tree-anchor mean"],
        color=RED, label="H", fontsize=7.6)
    ax.plot([0.070, 0.405], [bot_y + 0.042, bot_y + 0.042], color=GREY, lw=1.0)
    ax.scatter([0.310, 0.385], [bot_y + 0.042, bot_y + 0.042], s=28, color=[ORANGE, RED], zorder=3)
    ax.text(0.310, bot_y + 0.012, "0.95", ha="center", fontsize=7.0)
    ax.text(0.385, bot_y + 0.012, "0.99 fallback", ha="center", fontsize=7.0, color=RED)
    box(ax, (0.515, bot_y), (bot_w, bot_h), "Engineering outputs and evidence boundary",
        ["Ensemble EIC model and calibrated intervals", "Screening and response propagation", "Confirmation required before design use"],
        color=GOLD, label="I", fontsize=7.6)
    for px, color, label in ((0.625, BLUE, "EIC"), (0.750, ORANGE, "screen"), (0.875, TEAL, "confirm")):
        ax.add_patch(Circle((px, bot_y + 0.042), 0.027, facecolor=color, edgecolor="white", lw=0.5, alpha=0.88))
        ax.text(px, bot_y + 0.042, label, ha="center", va="center", fontsize=6.6, color="white")
    arrow(ax, (mid_x[2] + mid_w / 2, mid_y), (0.750, bot_y + bot_h), color=BLUE)
    arrow(ax, (0.515, bot_y + 0.130), (0.480, bot_y + 0.130), color=RED)

    SOURCE.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "FINAL_MODEL_LOCK_FOR_FIGURE3.md",
        "diagram_type": "deterministic vector schematic",
        "panel_labels": [f"({letter})" for letter in "ABCDEFGHI"],
        "typography": "Times New Roman, normal weight",
        "locked_settings": {
            "grid": [64, 64, 48], "physical_channels": 14, "latent_channels": 16,
            "raster_channels": 37, "token_dimensions": 52, "posterior_members": 64,
            "heun_steps": 10, "ood_control_onset": 0.95, "exact_fallback_threshold": 0.99,
        },
        "claim_boundary": "Architecture schematic only; no performance values are inferred from the artwork.",
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    export_m1_figure(fig, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
