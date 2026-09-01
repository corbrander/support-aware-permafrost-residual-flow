from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle
import numpy as np

from m1_figure_style import apply_m1_style, enforce_m1_typography, export_m1_figure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure1_support_aware_observations"
)

INK = "#263238"
BLUE = "#377eb8"
TEAL = "#1b9e77"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
RED = "#b23a48"
LIGHT = "#f5f7f8"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.02,
        1.04,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="normal",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.085,
        1.04,
        title,
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="bottom",
        color=INK,
    )


def draw_supports(ax: plt.Axes) -> None:
    x = np.linspace(0.0, 128.0, 500)
    b1 = 1.7 + 0.35 * np.sin(x / 14.0)
    b2 = 5.2 + 0.45 * np.sin(x / 18.0 + 0.8)
    ax.fill_between(x, 0, b1, color="#ddd6c9")
    ax.fill_between(x, b1, b2, color="#b6c6cc")
    ax.fill_between(x, b2, 12, color="#d7e8ef")
    ax.plot(x, b1, color="white", lw=0.8)
    ax.plot(x, b2, color="white", lw=0.8)

    # Point support.
    ax.scatter([12], [6.7], s=32, color=BLUE, edgecolor="white", lw=0.6, zorder=5)
    ax.annotate("point", (12, 6.7), (4, 8.4), color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))

    # Borehole interval.
    ax.plot([34, 34], [0, 11.2], color=INK, lw=1.0)
    ax.add_patch(Rectangle((32.2, 3.3), 3.6, 3.2, facecolor=TEAL, edgecolor="white", lw=0.7))
    ax.annotate("finite borehole\ninterval", (34, 4.9), (22, 8.8), color=TEAL, ha="center", arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7))

    # ERT finite volume.
    cell = Polygon([[50, 2.2], [74, 2.8], [70, 6.2], [54, 5.7]], closed=True, facecolor=ORANGE, alpha=0.72, edgecolor=ORANGE, lw=1.0)
    ax.add_patch(cell)
    ax.plot([45, 80], [0.25, 0.25], color=ORANGE, lw=1.6)
    ax.text(62, 7.2, "ERT finite cell / volume", color=ORANGE, ha="center")

    # Normalized NMR Gaussian kernel.
    for width, alpha in [(15, 0.10), (10, 0.18), (5, 0.32)]:
        ax.add_patch(Ellipse((94, 7.1), width, width * 0.28, facecolor=PURPLE, edgecolor=PURPLE, lw=0.5, alpha=alpha))
    ax.scatter([94], [7.1], s=12, color=PURPLE, zorder=5)
    ax.text(94, 9.1, "normalized NMR\nGaussian kernel", color=PURPLE, ha="center")

    # Active-layer zero crossing.
    z = np.linspace(0.1, 5.0, 100)
    temp = 1.5 - 0.9 * z
    x_profile = 117 + 2.0 * temp
    ax.plot(x_profile, z, color=RED, lw=1.2)
    crossing_z = 1.5 / 0.9
    ax.scatter([117], [crossing_z], marker="D", s=24, color=RED, edgecolor="white", lw=0.5, zorder=5)
    ax.text(112, 5.1, "0 deg C crossing", color=RED, ha="center")

    ax.set_xlim(0, 128)
    ax.set_ylim(12, 0)
    ax.set_xlabel("horizontal distance (m)")
    ax.set_ylabel("depth (m)")
    ax.set_xticks([0, 32, 64, 96, 128])
    ax.set_yticks([0, 3, 6, 9, 12])
    panel_title(ax, "a", "Five implemented measurement supports")


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        0.16,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor=mcolors.to_rgba(color, 0.14),
        edgecolor=color,
        lw=0.8,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + 0.08, text, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=INK)


def draw_record(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_title(ax, "b", "One typed observation retains provenance")
    ax.text(0.5, 0.88, "o_i = {y_i, tau_i, r_i, S_i, sigma_i, q_i, s_i, d_i, g_i}", transform=ax.transAxes, ha="center", va="center", fontsize=11, color=INK)
    fields = [
        ("y_i\nvalue", BLUE),
        ("tau_i\nvariable/type", TEAL),
        ("r_i\nlocation", ORANGE),
        ("S_i\nsupport", PURPLE),
        ("sigma_i\nuncertainty", RED),
        ("q_i\nquality", "#5c6b73"),
        ("s_i\nsite", BLUE),
        ("d_i\nsource", TEAL),
        ("g_i\nborehole/profile", ORANGE),
    ]
    for index, (text, color) in enumerate(fields):
        row, col = divmod(index, 3)
        rounded_box(ax, (0.03 + 0.325 * col, 0.58 - 0.23 * row), 0.29, text, color)
    ax.text(0.5, 0.04, "No target truth or dense withheld voxel enters the record", transform=ax.transAxes, ha="center", color="0.35", fontsize=6.5)


def draw_covariance(ax: plt.Axes) -> None:
    panel_title(ax, "c", "Observation equation and error covariance")
    ax.text(0.5, 0.93, "y_i = H_i[x] + b_source(i) + epsilon_i", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=INK)
    ax.text(0.5, 0.82, "epsilon ~ N(0, Sigma)", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=INK)
    n = 14
    cov = np.eye(n) * 0.15
    for i in range(6, n):
        for j in range(6, n):
            cov[i, j] = 0.62 ** abs(i - j)
    image = ax.imshow(cov, extent=(0.08, 0.72, 0.05, 0.68), origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.add_patch(Rectangle((0.08, 0.05), 0.275, 0.27, fill=False, edgecolor=BLUE, lw=1.0))
    ax.add_patch(Rectangle((0.355, 0.32), 0.365, 0.36, fill=False, edgecolor=ORANGE, lw=1.0))
    ax.text(0.76, 0.22, "diagonal\nerrors", transform=ax.transAxes, color=BLUE, va="center")
    ax.text(0.76, 0.51, "profile-correlated\nERT block", transform=ax.transAxes, color=ORANGE, va="center")
    ax.text(0.76, 0.08, "b_source(i): source bias", transform=ax.transAxes, color=RED, va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    image.set_rasterized(False)


def draw_evaluation(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_title(ax, "d", "Support-weighted likelihood, not nearest-voxel labels")
    names = ["point", "interval", "volume", "kernel", "crossing"]
    colors = [BLUE, TEAL, ORANGE, PURPLE, RED]
    for index, (name, color) in enumerate(zip(names, colors, strict=True)):
        y = 0.82 - index * 0.145
        ax.add_patch(FancyBboxPatch((0.03, y - 0.045), 0.20, 0.09, boxstyle="round,pad=0.01", transform=ax.transAxes, facecolor=mcolors.to_rgba(color, 0.16), edgecolor=color, lw=0.8))
        ax.text(0.13, y, name, transform=ax.transAxes, ha="center", va="center", color=INK)
        ax.annotate("", xy=(0.58, y), xytext=(0.25, y), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
        ax.text(0.60, y, "H_i[x]", transform=ax.transAxes, ha="left", va="center", fontsize=8.5, color=INK)
    ax.text(0.84, 0.53, "Sigma^-1", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=ORANGE)
    ax.text(0.84, 0.40, "support\nlikelihood", transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color=INK)
    ax.plot([0.71, 0.97], [0.18, 0.18], transform=ax.transAxes, color="0.55", lw=0.8)
    ax.scatter([0.78, 0.84, 0.90], [0.18, 0.18, 0.18], transform=ax.transAxes, s=22, color="0.65")
    ax.plot([0.73, 0.95], [0.09, 0.27], transform=ax.transAxes, color=RED, lw=2.0)
    ax.plot([0.73, 0.95], [0.27, 0.09], transform=ax.transAxes, color=RED, lw=2.0)
    ax.text(0.84, 0.03, "nearest-voxel assignment", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4, color=RED)


def main() -> None:
    apply_m1_style()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(183 / 25.4, 116 / 25.4), constrained_layout=True, facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 0.88), height_ratios=(1.0, 0.92))
    draw_supports(fig.add_subplot(grid[0, 0]))
    draw_record(fig.add_subplot(grid[0, 1]))
    draw_covariance(fig.add_subplot(grid[1, 0]))
    draw_evaluation(fig.add_subplot(grid[1, 1]))
    enforce_m1_typography(fig)
    export_m1_figure(fig, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
