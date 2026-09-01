from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable


def _canonical(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    aliases = {
        "facies": ("facies", "facies_mode"),
        "eic": ("eic", "eic_mean"),
        "temperature": ("temperature", "temperature_mean"),
        "unfrozen_water": ("unfrozen_water", "unfrozen_water_mean"),
        "log_resistivity": ("log_resistivity", "log_resistivity_mean"),
    }
    out: dict[str, np.ndarray] = {}
    for key, names in aliases.items():
        for name in names:
            if name in fields:
                out[key] = np.asarray(fields[name])
                break
    return out


def _nearest_xy(grid: dict, xy: tuple[float, float]) -> tuple[int, int]:
    x = np.asarray(grid.get("grid_x", grid.get("x")), dtype=np.float32)
    y = np.asarray(grid.get("grid_y", grid.get("y")), dtype=np.float32)
    ix = int(np.abs(x - float(xy[0])).argmin())
    iy = int(np.abs(y - float(xy[1])).argmin())
    return ix, iy


def _selected_boreholes(observations: ObservationTable, max_boreholes: int) -> list[tuple[float, float]]:
    mask = observations.type_ids == OBS_TYPES["borehole_facies"]
    coords = observations.coords[mask]
    if len(coords) == 0:
        return []
    xy = np.round(coords[:, :2], decimals=4)
    unique, counts = np.unique(xy, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [(float(unique[i, 0]), float(unique[i, 1])) for i in order[:max_boreholes]]


def _obs_profile(observations: ObservationTable, xy: tuple[float, float], type_name: str, tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    type_id = OBS_TYPES[type_name]
    mask = (
        (observations.type_ids == type_id)
        & (np.abs(observations.coords[:, 0] - float(xy[0])) <= tol)
        & (np.abs(observations.coords[:, 1] - float(xy[1])) <= tol)
    )
    coords = observations.coords[mask]
    values = observations.values[mask]
    order = np.argsort(coords[:, 2]) if len(coords) else np.array([], dtype=np.int64)
    return coords[order, 2] if len(coords) else np.array([], dtype=np.float32), values[order]


def plot_borehole_profile_comparison(
    sample: dict,
    prediction: dict[str, np.ndarray],
    out_path: str | Path,
    max_boreholes: int = 4,
    title: str = "Borehole profile comparison",
) -> None:
    """Plot truth, prediction, and sparse observations along selected boreholes."""
    truth = _canonical(sample["fields"])
    pred = _canonical(prediction)
    grid = sample["grid"]
    z = np.asarray(grid.get("grid_z", grid.get("z")), dtype=np.float32)
    observations: ObservationTable = sample["observations"]
    boreholes = _selected_boreholes(observations, max_boreholes=max_boreholes)
    if not boreholes:
        raise ValueError("No borehole facies observations found for profile comparison.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(boreholes), 3, figsize=(11.5, 2.65 * len(boreholes)), squeeze=False, constrained_layout=True)
    fig.suptitle(title, fontsize=13)
    for row, xy in enumerate(boreholes):
        ix, iy = _nearest_xy(grid, xy)
        label = f"x={xy[0]:.1f}, y={xy[1]:.1f}"
        ax = axes[row, 0]
        if "facies" in truth:
            ax.step(truth["facies"][ix, iy, :], z, where="mid", color="black", lw=1.3, label="truth")
        if "facies" in pred:
            ax.step(pred["facies"][ix, iy, :], z, where="mid", color="#d95f02", lw=1.2, label="pred")
        obs_z, obs_v = _obs_profile(observations, xy, "borehole_facies")
        if len(obs_z):
            ax.scatter(obs_v, obs_z, s=10, color="#1b9e77", label="obs", zorder=3)
        ax.set_title(f"{label} facies", fontsize=9)
        ax.set_xlabel("facies id")
        ax.set_ylabel("depth (m)")
        ax.set_ylim(float(np.max(z)), float(np.min(z)))
        ax.grid(True, color="0.9")
        if row == 0:
            ax.legend(fontsize=7)

        ax = axes[row, 1]
        if "eic" in truth:
            ax.plot(truth["eic"][ix, iy, :], z, color="black", lw=1.3, label="truth")
        if "eic" in pred:
            ax.plot(pred["eic"][ix, iy, :], z, color="#d95f02", lw=1.2, label="pred")
        obs_z, obs_v = _obs_profile(observations, xy, "borehole_eic")
        if len(obs_z):
            ax.scatter(obs_v, obs_z, s=10, color="#1b9e77", label="obs", zorder=3)
        ax.set_title("EIC", fontsize=9)
        ax.set_xlabel("EIC")
        ax.set_ylim(float(np.max(z)), float(np.min(z)))
        ax.grid(True, color="0.9")

        ax = axes[row, 2]
        if "temperature" in truth:
            ax.plot(truth["temperature"][ix, iy, :], z, color="black", lw=1.3, label="truth")
        if "temperature" in pred:
            ax.plot(pred["temperature"][ix, iy, :], z, color="#d95f02", lw=1.2, label="pred")
        obs_z, obs_v = _obs_profile(observations, xy, "borehole_temperature")
        if len(obs_z):
            ax.scatter(obs_v, obs_z, s=10, color="#1b9e77", label="obs", zorder=3)
        ax.axvline(0.0, color="0.5", lw=0.8, ls="--")
        ax.set_title("temperature", fontsize=9)
        ax.set_xlabel("C")
        ax.set_ylim(float(np.max(z)), float(np.min(z)))
        ax.grid(True, color="0.9")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
