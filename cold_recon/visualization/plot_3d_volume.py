from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _axes_from_grid(grid: dict | None, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid is None:
        return tuple(np.arange(n, dtype=np.float32) for n in shape)  # type: ignore[return-value]
    x = np.asarray(grid.get("grid_x", grid.get("x", np.arange(shape[0]))), dtype=np.float32)
    y = np.asarray(grid.get("grid_y", grid.get("y", np.arange(shape[1]))), dtype=np.float32)
    z = np.asarray(grid.get("grid_z", grid.get("z", np.arange(shape[2]))), dtype=np.float32)
    return x, y, z


def _field(fields: dict[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in fields:
            return np.asarray(fields[name])
    return None


def _sample_mask(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        coords = np.argwhere(np.ones_like(mask, dtype=bool))
    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        coords = coords[rng.choice(len(coords), size=max_points, replace=False)]
    return coords


def _scatter_panel(
    ax,
    coords: np.ndarray,
    values: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    cmap: str,
    colorbar_label: str,
    fig,
) -> None:
    x, y, z = axes
    xx = x[coords[:, 0]]
    yy = y[coords[:, 1]]
    zz = z[coords[:, 2]]
    colors = values[coords[:, 0], coords[:, 1], coords[:, 2]]
    sc = ax.scatter(xx, yy, zz, c=colors, s=5, cmap=cmap, alpha=0.72, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("depth (m)")
    ax.set_zlim(float(np.max(z)), float(np.min(z)))
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def plot_3d_volume_overview(
    fields: dict[str, np.ndarray],
    out_path: str | Path,
    grid: dict | None = None,
    title: str = "3D permafrost reconstruction overview",
    max_points: int = 7000,
    ice_threshold: float = 0.25,
) -> None:
    """Create a compact paper-style 3D overview of ice-rich, facies, and thermal structure."""
    eic = _field(fields, "eic", "eic_mean")
    temp = _field(fields, "temperature", "temperature_mean")
    facies = _field(fields, "facies", "facies_mode")
    if eic is None and temp is None and facies is None:
        raise ValueError("fields must contain at least eic, temperature, or facies")
    ref = next(arr for arr in (eic, temp, facies) if arr is not None)
    shape = tuple(int(v) for v in ref.shape[:3])
    axes = _axes_from_grid(grid, shape)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    if eic is not None:
        mask = np.asarray(eic, dtype=np.float32) >= float(ice_threshold)
        coords = _sample_mask(mask, max_points, seed=11)
        _scatter_panel(ax1, coords, np.asarray(eic, dtype=np.float32), axes, "Ice-rich voxels", "viridis", "EIC", fig)
    else:
        ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    if facies is not None:
        facies_arr = np.asarray(facies, dtype=np.float32)
        mask = np.isin(facies_arr.astype(np.int16), [1, 3, 5, 6])
        coords = _sample_mask(mask, max_points, seed=23)
        _scatter_panel(ax2, coords, facies_arr, axes, "Engineering facies subset", "tab20", "facies id", fig)
    else:
        ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    if temp is not None:
        temp_arr = np.asarray(temp, dtype=np.float32)
        mask = np.abs(temp_arr) <= 1.5
        coords = _sample_mask(mask, max_points, seed=37)
        _scatter_panel(ax3, coords, temp_arr, axes, "Near-thaw thermal corridor", "coolwarm", "temperature (C)", fig)
    else:
        ax3.axis("off")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
