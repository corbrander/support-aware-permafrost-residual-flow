from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import numpy as np

from m1_figure_style import apply_m1_style, enforce_m1_typography, export_m1_figure


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "m1_support_guided_benchmark" / "m1_scene_manifest.json"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure2_controlled_benchmark"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure2"

INK = "#263238"
BLUE = "#377eb8"
TEAL = "#1b9e77"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
RED = "#b23a48"
GREY = "#d8dee2"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.8,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str, x: float = 0.0) -> None:
    ax.text(x, 1.025, f"({letter})", transform=ax.transAxes, fontsize=10.0, fontweight="normal", ha="left", va="bottom")
    ax.text(x + 0.13, 1.025, f"\u2002{title}", transform=ax.transAxes, fontsize=9.8, fontweight="normal", ha="left", va="bottom", color=INK)


def load_scene() -> tuple[dict, dict[str, np.ndarray], int]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = next(
        row
        for row in manifest["records"]
        if row["split"] == "test_id" and row["generator_family"] == "combined"
    )
    archive = np.load(MANIFEST.parent / record["relative_path"], allow_pickle=False)
    values = {name: archive[name] for name in archive.files}
    scores = []
    for y_index in range(values["field_eic"].shape[1]):
        score = (
            0.8 * np.std(values["field_eic"][:, y_index, :])
            + 0.2 * len(np.unique(values["field_lithology"][:, y_index, :]))
            + 0.2 * len(np.unique(values["field_thermal_state"][:, y_index, :]))
            + 0.2 * len(np.unique(values["field_ice_structure"][:, y_index, :]))
        )
        scores.append(float(score))
    return {**manifest, "selected_record": record}, values, int(np.argmax(scores))


def show_section(
    fig: plt.Figure,
    ax: plt.Axes,
    field: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    title: str,
    *,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    colorbar: bool = False,
    colorbar_label: str = "",
) -> None:
    image = ax.imshow(
        field.T,
        origin="upper",
        extent=(float(x.min()), float(x.max()), float(z.max()), float(z.min())),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, loc="left", pad=1.5, color=INK, fontweight="normal")
    ax.set_xticks([0, 64, 126])
    ax.set_yticks([0, 6, 11.75])
    if colorbar:
        bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.085, pad=0.16)
        bar.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
        bar.set_label(colorbar_label, fontsize=5.4, labelpad=1)


def draw_factorized_panel(fig: plt.Figure, spec, data: dict[str, np.ndarray], y_index: int) -> list[plt.Axes]:
    sub = spec.subgridspec(
        3,
        4,
        height_ratios=(1.0, 1.0, 0.28),
        wspace=0.32,
        hspace=0.62,
    )
    positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    axes = [fig.add_subplot(sub[i, j]) for i, j in positions]
    x = data["grid_x"].astype(float)
    z = data["grid_z"].astype(float)
    lith_cmap = mcolors.ListedColormap(["#7b5d44", "#b4a078", "#d9bd72", "#9ca7ad"])
    thermal_cmap = mcolors.ListedColormap(["#d95f5f", "#66a7c5", "#e7a95b"])
    ice_cmap = mcolors.ListedColormap(["#d9e6eb", "#73b8cf", "#285f8f"])
    show_section(fig, axes[0], data["field_lithology"][:, y_index, :], x, z, "L | lithology", cmap=lith_cmap, norm=mcolors.BoundaryNorm(np.arange(-0.5, 4.5), 4))
    show_section(fig, axes[1], data["field_thermal_state"][:, y_index, :], x, z, "S | thermal state", cmap=thermal_cmap, norm=mcolors.BoundaryNorm(np.arange(-0.5, 3.5), 3))
    show_section(fig, axes[2], data["field_ice_structure"][:, y_index, :], x, z, "I | ice structure", cmap=ice_cmap, norm=mcolors.BoundaryNorm(np.arange(-0.5, 3.5), 3))
    show_section(fig, axes[3], data["field_eic"][:, y_index, :], x, z, "E | excess ice", cmap="cividis", vmin=0, vmax=0.9, colorbar=True, colorbar_label="EIC")
    show_section(fig, axes[4], data["field_temperature"][:, y_index, :], x, z, "T | temperature", cmap="coolwarm", norm=mcolors.TwoSlopeNorm(vmin=-12, vcenter=0, vmax=4), colorbar=True, colorbar_label="deg C")
    show_section(fig, axes[5], data["field_unfrozen_water"][:, y_index, :], x, z, "W | unfrozen-water proxy", cmap="Blues", vmin=0, vmax=0.85, colorbar=True, colorbar_label="W")
    show_section(fig, axes[6], np.log(np.maximum(data["field_resistivity"][:, y_index, :], 1.0)), x, z, "log R | resistivity", cmap="magma", vmin=0, vmax=15, colorbar=True, colorbar_label="natural log")
    legend = fig.add_subplot(sub[2, :])
    legend.set_axis_off()
    labels = [
        "L peat",
        "L silt",
        "L sand/gravel",
        "L other",
        "S thawed",
        "S frozen",
        "S near-thaw",
        "I matrix",
        "I lens-rich",
        "I massive/wedge",
    ]
    colors = [*lith_cmap.colors, *thermal_cmap.colors, *ice_cmap.colors]
    handles = [Patch(facecolor=color, edgecolor="white", linewidth=0.3) for color in colors]
    legend.legend(
        handles,
        labels,
        ncol=5,
        loc="center",
        frameon=False,
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.35,
        title="Categorical keys",
    )
    axes[0].set_ylabel("depth (m)")
    axes[4].set_ylabel("depth (m)")
    return axes


def draw_acquisition(ax: plt.Axes) -> None:
    panel_title(ax, "b", "Support extraction and perturbation")
    x = np.linspace(0, 128, 300)
    boundary = 3.2 + 0.5 * np.sin(x / 18)
    ax.fill_between(x, 0, boundary, color="#e4ded2")
    ax.fill_between(x, boundary, 12, color="#d5e7ee")
    ax.plot([18, 18], [0, 11], color=INK, lw=1.0)
    ax.add_patch(Rectangle((16.7, 4.0), 2.6, 3.0, facecolor=TEAL, edgecolor="white", lw=0.5))
    ax.add_patch(Rectangle((42, 2.3), 24, 3.3, facecolor=ORANGE, edgecolor=ORANGE, alpha=0.65))
    for width, alpha in [(14, 0.10), (8, 0.20), (3.5, 0.35)]:
        ax.add_patch(plt.matplotlib.patches.Ellipse((83, 7.1), width, width * 0.25, color=PURPLE, alpha=alpha))
    ax.scatter([105], [1.8], marker="D", s=24, color=RED, edgecolor="white", lw=0.4)
    ax.scatter([119], [7.8], s=24, color=BLUE, edgecolor="white", lw=0.4)
    ax.set_xlim(0, 128)
    ax.set_ylim(12, 0)
    ax.set_xticks([0, 64, 128])
    ax.set_yticks([0, 6, 12])
    ax.set_xlabel("horizontal distance (m)")
    ax.set_ylabel("depth (m)")
    ax.text(0.50, 0.03, "Only sparse supports and surface context\nenter the reconstruction model", transform=ax.transAxes, fontsize=5.3, color=INK, ha="center", va="bottom", bbox=dict(facecolor="white", edgecolor="0.8", pad=1.2))
    noise = "noise audit\n$\\sigma$ x {0.5, 1, 2, 4}\n1-5% outliers\nsource bias\ncorrelated ERT"
    ax.text(0.98, 0.95, noise, transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=INK, linespacing=1.15, bbox=dict(facecolor="white", edgecolor=ORANGE, boxstyle="round,pad=0.25"))


def draw_splits(ax: plt.Axes, manifest: dict) -> None:
    ax.set_axis_off()
    panel_title(ax, "c", "Immutable scene-level split")
    rows = [
        ("training", 500, BLUE),
        ("validation", 100, TEAL),
        ("ID test", 100, "#4e79a7"),
        ("geometry OOD", 50, ORANGE),
        ("coupling OOD", 50, PURPLE),
        ("saline OOD", 50, RED),
    ]
    total = 850.0
    x0 = 0.05
    width = 0.90
    cursor = x0
    for name, count, color in rows:
        current = width * count / total
        ax.add_patch(Rectangle((cursor, 0.62), current, 0.18, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=0.8))
        if count >= 100:
            ax.text(cursor + current / 2, 0.71, str(count), transform=ax.transAxes, color="white", ha="center", va="center", fontweight="normal")
        cursor += current
    y = 0.50
    for index, (name, count, color) in enumerate(rows):
        col = index % 2
        row = index // 2
        yy = y - row * 0.12
        xx = 0.05 + col * 0.56
        ax.add_patch(Rectangle((xx, yy - 0.025), 0.035, 0.05, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(xx + 0.05, yy, f"{name}: {count}", transform=ax.transAxes, va="center", fontsize=5.9)
    ax.text(0.5, 0.12, "850 independently seeded scenes\nno shared scene identifiers or seeds", transform=ax.transAxes, ha="center", va="center", color=INK, fontweight="normal")


def flow_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012", transform=ax.transAxes, facecolor=mcolors.to_rgba(color, 0.14), edgecolor=color, lw=0.8))
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color=INK, linespacing=1.15)


def draw_boundary(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_title(ax, "d", "Boundary of validation evidence")
    ax.plot([0.5, 0.5], [0.08, 0.88], transform=ax.transAxes, color="0.78", lw=0.8)
    ax.text(0.25, 0.88, "CONTROLLED", transform=ax.transAxes, ha="center", fontweight="normal", color=BLUE)
    ax.text(0.75, 0.88, "PUBLIC", transform=ax.transAxes, ha="center", fontweight="normal", color=TEAL)
    flow_box(ax, (0.04, 0.60), 0.42, 0.17, "complete 3-D truth\nknown by construction", BLUE)
    flow_box(ax, (0.04, 0.32), 0.42, 0.17, "voxel, support and\nrare-object metrics", BLUE)
    flow_box(ax, (0.54, 0.60), 0.42, 0.17, "nested complete-\nborehole LOO", TEAL)
    flow_box(ax, (0.54, 0.32), 0.42, 0.17, "held-support EIC +\ncalibrated intervals", TEAL)
    ax.annotate("", xy=(0.25, 0.52), xytext=(0.25, 0.60), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.45"))
    ax.annotate("", xy=(0.75, 0.52), xytext=(0.75, 0.60), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.45"))
    ax.text(0.75, 0.14, "sparse observational validation\nNO dense 3-D field truth", transform=ax.transAxes, ha="center", va="center", color=RED, fontweight="normal", fontsize=6.1)


def write_source_data(manifest: dict, data: dict[str, np.ndarray], y_index: int) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    x = data["grid_x"].astype(float)
    z = data["grid_z"].astype(float)
    with (SOURCE / "factorized_section.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_m", "z_m", "lithology", "thermal_state", "ice_structure", "eic", "temperature_c", "unfrozen_water", "log_resistivity"])
        log_r = np.log(np.maximum(data["field_resistivity"][:, y_index, :], 1.0))
        for ix, x_value in enumerate(x):
            for iz, z_value in enumerate(z):
                writer.writerow([x_value, z_value, int(data["field_lithology"][ix, y_index, iz]), int(data["field_thermal_state"][ix, y_index, iz]), int(data["field_ice_structure"][ix, y_index, iz]), float(data["field_eic"][ix, y_index, iz]), float(data["field_temperature"][ix, y_index, iz]), float(data["field_unfrozen_water"][ix, y_index, iz]), float(log_r[ix, iz])])
    metadata = {
        "scene_id": manifest["selected_record"]["scene_id"],
        "generator_family": manifest["selected_record"]["generator_family"],
        "manifest_sha256": manifest["manifest_sha256"],
        "section_y_index": int(y_index),
        "section_y_m": float(data["grid_y"][y_index]),
        "split_counts": manifest["counts"],
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    apply_m1_style(base_font_size=6.9)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest, data, y_index = load_scene()
    write_source_data(manifest, data, y_index)
    fig = plt.figure(figsize=(183 / 25.4, 145 / 25.4), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.18, 0.82),
        width_ratios=(1.00, 1.08, 1.02),
        left=0.055,
        right=0.985,
        top=0.90,
        bottom=0.085,
        hspace=0.34,
        wspace=0.30,
    )
    draw_factorized_panel(fig, grid[0, :], data, y_index)
    fig.text(0.055, 0.965, "(A)", fontsize=8.3, fontweight="normal", ha="left", va="top")
    fig.text(0.095, 0.965, "Registered factorized controlled truth", fontsize=7.8, ha="left", va="top", color=INK)
    draw_acquisition(fig.add_subplot(grid[1, 0]))
    draw_splits(fig.add_subplot(grid[1, 1]), manifest)
    draw_boundary(fig.add_subplot(grid[1, 2]))
    enforce_m1_typography(fig)
    export_m1_figure(fig, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
