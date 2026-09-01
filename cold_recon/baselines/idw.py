from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable


def _grid_points(grid: dict) -> np.ndarray:
    xx, yy, zz = np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def idw_interpolate(
    obs_coords: np.ndarray,
    obs_values: np.ndarray,
    query_coords: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    tree = cKDTree(obs_coords)
    k = min(k, len(obs_coords))
    dist, idx = tree.query(query_coords, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.power(dist + eps, power)
    return np.sum(weights * obs_values[idx], axis=1) / np.sum(weights, axis=1)


def reconstruct_idw(observations: ObservationTable, grid: dict, n_facies: int = 7) -> dict[str, np.ndarray]:
    query = _grid_points(grid)
    shape = (len(grid["x"]), len(grid["y"]), len(grid["z"]))
    out: dict[str, np.ndarray] = {}

    for type_name, field_name in [
        ("borehole_eic", "eic"),
        ("borehole_temperature", "temperature"),
        ("nmr_unfrozen_water", "unfrozen_water"),
        ("ert_log_resistivity", "log_resistivity"),
    ]:
        mask = observations.mask & (observations.type_ids == OBS_TYPES[type_name])
        if np.any(mask):
            pred = idw_interpolate(observations.coords[mask], observations.values[mask], query)
            out[field_name] = pred.reshape(shape).astype(np.float32)

    facies_mask = observations.mask & (observations.type_ids == OBS_TYPES["borehole_facies"])
    if np.any(facies_mask):
        facies_coords = observations.coords[facies_mask]
        facies_values = observations.values[facies_mask].astype(np.int64)
        tree = cKDTree(facies_coords)
        dist, idx = tree.query(query, k=min(7, len(facies_coords)))
        if idx.ndim == 1:
            idx = idx[:, None]
            dist = dist[:, None]
        weights = 1.0 / np.power(dist + 1e-6, 2.0)
        logits = np.zeros((query.shape[0], n_facies), dtype=np.float32)
        for col in range(idx.shape[1]):
            logits[np.arange(query.shape[0]), np.clip(facies_values[idx[:, col]], 0, n_facies - 1)] += weights[:, col]
        out["facies"] = np.argmax(logits, axis=1).reshape(shape).astype(np.int16)
        out["facies_logits"] = logits.reshape(*shape, n_facies)
    return out
