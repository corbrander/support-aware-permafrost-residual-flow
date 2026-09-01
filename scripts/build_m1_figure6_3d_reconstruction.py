from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

from m1_figure_style import apply_m1_style, export_m1_figure, panel_title


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "source_data" / "m1_figure5" / "test_id_combined_00008_seed41.npz"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure6_three_dimensional_reconstruction"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure06_three_dimensional"

INK = "#263238"
TRUTH = "#d6a632"
ANCHOR = "#607d8b"
POSTERIOR = "#1b9e77"
UNCERTAINTY = "#d95f02"
EVENT = "#b23a48"


def add_surface(
    ax: plt.Axes,
    volume: np.ndarray,
    level: float,
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    color: str,
    alpha: float = 0.72,
    step_size: int = 2,
    linewidth: float = 0.08,
) -> dict[str, int | float]:
    if not float(np.nanmin(volume)) < level < float(np.nanmax(volume)):
        raise ValueError(f"Surface level {level} is outside [{np.nanmin(volume)}, {np.nanmax(volume)}].")
    x, y, z = coordinates
    spacing = (
        float(np.mean(np.diff(x))),
        float(np.mean(np.diff(y))),
        float(np.mean(np.diff(z))),
    )
    vertices, faces, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=level,
        spacing=spacing,
        step_size=step_size,
        allow_degenerate=False,
    )
    vertices += np.asarray([x[0], y[0], z[0]])
    mesh = Poly3DCollection(
        vertices[faces],
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        antialiased=True,
    )
    ax.add_collection3d(mesh)
    return {"level": float(level), "vertices": int(len(vertices)), "faces": int(len(faces))}


def format_3d(ax: plt.Axes, coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    x, y, z = coordinates
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_zlim(float(z.max()), float(z.min()))
    ax.set_xticks([0, 64, 128])
    ax.set_yticks([0, 64, 128])
    ax.set_zticks([0, 6, 12])
    ax.set_xlabel("Distance, x (m)", labelpad=-1)
    ax.set_ylabel("Distance, y (m)", labelpad=-1)
    ax.set_zlabel("Depth, z (m)", labelpad=-2)
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((1.0, 1.0, 0.48))
    ax.tick_params(pad=-1, length=2.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))
        axis._axinfo["grid"].update(color=(0.90, 0.90, 0.90, 1.0), linewidth=0.35)


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Locked volumetric archive not found: {ARCHIVE}")
    apply_m1_style(base_font_size=6.8)
    archive = np.load(ARCHIVE, allow_pickle=False)
    required = {
        "x",
        "y",
        "z",
        "truth_eic",
        "anchor_eic",
        "posterior_eic_mean",
        "posterior_eic_std",
        "event_probability",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise KeyError(f"Locked archive is missing arrays: {missing}")

    coordinates = tuple(np.asarray(archive[name], dtype=float) for name in ("x", "y", "z"))
    truth = np.asarray(archive["truth_eic"], dtype=float)
    anchor = np.asarray(archive["anchor_eic"], dtype=float)
    posterior = np.asarray(archive["posterior_eic_mean"], dtype=float)
    spread = np.asarray(archive["posterior_eic_std"], dtype=float)
    event_probability = np.asarray(archive["event_probability"], dtype=float)
    if truth.shape != anchor.shape or truth.shape != posterior.shape or truth.shape != spread.shape:
        raise ValueError("The locked truth, anchor, posterior and spread volumes do not share one grid.")
    if event_probability.shape[1:] != truth.shape or event_probability.shape[0] < 2:
        raise ValueError("The locked event-probability array does not contain the EIC > 0.30 channel.")

    display_level = 0.15
    spread_level = float(np.quantile(spread, 0.90))
    event_level = 0.50
    high_event = event_probability[1]

    fig = plt.figure(figsize=(183 / 25.4, 119 / 25.4), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.03, right=0.985, top=0.95, bottom=0.085, wspace=0.02, hspace=0.17)
    axes = [fig.add_subplot(grid[index // 3, index % 3], projection="3d") for index in range(6)]
    titles = [
        "Controlled EIC truth",
        "Observation-derived anchor",
        "Posterior mean",
        "Truth–posterior overlap",
        "Posterior uncertainty envelope",
        "High-EIC event probability",
    ]
    for index, (ax, title) in enumerate(zip(axes, titles, strict=True)):
        format_3d(ax, coordinates)
        panel_title(ax, chr(ord("A") + index), title, x=0.0, y=0.99, title_offset=0.13, fontsize=7.5)

    meshes: dict[str, dict[str, int | float] | list[dict[str, int | float]]] = {}
    meshes["truth"] = add_surface(axes[0], truth, display_level, coordinates, color=TRUTH)
    meshes["anchor"] = add_surface(axes[1], anchor, display_level, coordinates, color=ANCHOR)
    meshes["posterior"] = add_surface(axes[2], posterior, display_level, coordinates, color=POSTERIOR)
    meshes["overlap"] = [
        add_surface(axes[3], truth, display_level, coordinates, color=TRUTH, alpha=0.25, linewidth=0.04),
        add_surface(axes[3], posterior, display_level, coordinates, color=POSTERIOR, alpha=0.62, linewidth=0.04),
    ]
    meshes["uncertainty"] = add_surface(
        axes[4], spread, spread_level, coordinates, color=UNCERTAINTY, alpha=0.62, linewidth=0.04
    )
    meshes["event_probability"] = add_surface(
        axes[5], high_event, event_level, coordinates, color=EVENT, alpha=0.68, linewidth=0.04
    )

    fig.text(
        0.5,
        0.018,
        f"(A–D) EIC = 0.15 isosurfaces; (E) posterior s.d. = {spread_level:.3f} (90th percentile); "
        "(F) P(EIC > 0.30) = 0.50",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=INK,
    )
    axes[3].legend(
        handles=[
            Line2D([0], [0], color=TRUTH, lw=3, label="Truth"),
            Line2D([0], [0], color=POSTERIOR, lw=3, label="Posterior"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.90),
        fontsize=5.8,
    )

    SOURCE.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_archive": str(ARCHIVE.relative_to(ROOT)),
        "scene_id": "test_id_combined_00008",
        "model_seed": 41,
        "posterior_members": 64,
        "grid_shape": list(truth.shape),
        "display_eic_isosurface": display_level,
        "uncertainty_isosurface_quantile": 0.90,
        "uncertainty_isosurface_value": spread_level,
        "event_definition": "EIC > 0.30",
        "event_probability_isosurface": event_level,
        "surface_extraction": "marching cubes on the locked full-resolution arrays; step_size=2; no smoothing",
        "mesh_statistics": meshes,
        "claim_boundary": "The figure visualizes the locked controlled volume and does not constitute dense field validation.",
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    export_m1_figure(fig, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "source": str(SOURCE), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
