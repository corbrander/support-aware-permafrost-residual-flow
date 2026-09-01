from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.physics.settlement import settlement_potential_numpy


@dataclass
class FieldGrid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.x), len(self.y), len(self.z))

    @property
    def dz(self) -> float:
        return float(np.mean(np.diff(self.z))) if len(self.z) > 1 else 1.0


def make_field_grid(
    observations: ObservationTable,
    nx: int = 128,
    ny: int = 48,
    nz: int = 64,
    zmax: float = 12.0,
    pad_xy: float = 5.0,
) -> FieldGrid:
    valid = np.isfinite(observations.coords).all(axis=1)
    coords = observations.coords[valid]
    x_min, y_min = np.nanmin(coords[:, :2], axis=0) - pad_xy
    x_max, y_max = np.nanmax(coords[:, :2], axis=0) + pad_xy
    x_min = min(float(x_min), 0.0)
    y_min = min(float(y_min), 0.0)
    z_upper = min(float(np.nanmax(coords[:, 2])), zmax)
    z_upper = max(z_upper, 2.0)
    return FieldGrid(
        x=np.linspace(x_min, float(x_max), nx, dtype=np.float32),
        y=np.linspace(y_min, float(y_max), ny, dtype=np.float32),
        z=np.linspace(0.0, z_upper, nz, dtype=np.float32),
    )


def split_observations_by_type(
    observations: ObservationTable,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    min_holdout: int = 5,
) -> tuple[ObservationTable, ObservationTable]:
    rng = np.random.default_rng(seed)
    train_idx = []
    holdout_idx = []
    for type_id in np.unique(observations.type_ids):
        idx = np.where(observations.type_ids == type_id)[0]
        rng.shuffle(idx)
        n_hold = min(len(idx), max(min_holdout if len(idx) >= min_holdout else 1, int(round(len(idx) * holdout_fraction))))
        if len(idx) <= n_hold:
            n_hold = max(1, len(idx) // 3)
        holdout_idx.extend(idx[:n_hold].tolist())
        train_idx.extend(idx[n_hold:].tolist())
    return observations.subset(np.array(train_idx, dtype=int)), observations.subset(np.array(holdout_idx, dtype=int))


def idw_predict(
    obs_coords: np.ndarray,
    obs_values: np.ndarray,
    query_coords: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    max_points: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    if len(obs_coords) == 0:
        raise ValueError("Cannot interpolate with no observations")
    tree = cKDTree(obs_coords)
    k = min(k, len(obs_coords))
    preds = np.empty(query_coords.shape[0], dtype=np.float32)
    nn_dist = np.empty(query_coords.shape[0], dtype=np.float32)
    for start in range(0, query_coords.shape[0], max_points):
        sl = slice(start, min(start + max_points, query_coords.shape[0]))
        dist, idx = tree.query(query_coords[sl], k=k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        weights = 1.0 / np.power(dist + 1e-6, power)
        preds[sl] = (weights * obs_values[idx]).sum(axis=1) / weights.sum(axis=1)
        nn_dist[sl] = dist[:, 0]
    return preds, nn_dist


def _grid_points(grid: FieldGrid) -> np.ndarray:
    xx, yy, zz = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def _surface_points(grid: FieldGrid) -> np.ndarray:
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float32)]).astype(np.float32)


def reconstruct_field_from_observations(
    observations: ObservationTable,
    grid: FieldGrid | None = None,
    n_posterior: int = 16,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    if grid is None:
        grid = make_field_grid(observations)
    rng = np.random.default_rng(seed)
    shape = grid.shape
    points = _grid_points(grid)
    surface = _surface_points(grid)
    xx, yy, zz = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")

    alt_mask = observations.type_ids == OBS_TYPES["alt"]
    nmr_mask = observations.type_ids == OBS_TYPES["nmr_unfrozen_water"]
    ert_mask = observations.type_ids == OBS_TYPES["ert_log_resistivity"]
    alt_surface, alt_dist = idw_predict(observations.coords[alt_mask], observations.values[alt_mask], surface, k=8)
    alt = alt_surface.reshape(len(grid.x), len(grid.y))
    alt_unc = (0.05 + 0.01 * alt_dist.reshape(len(grid.x), len(grid.y))).astype(np.float32)

    log_rho, rho_dist = idw_predict(observations.coords[ert_mask], observations.values[ert_mask], points, k=10)
    log_rho = log_rho.reshape(shape)
    rho_unc = (0.12 + 0.006 * rho_dist.reshape(shape)).astype(np.float32)

    theta_u, theta_dist = idw_predict(observations.coords[nmr_mask], observations.values[nmr_mask], points, k=6)
    theta_u = np.clip(theta_u.reshape(shape), 0.0, 0.9)
    theta_unc = (0.04 + 0.015 * theta_dist.reshape(shape)).astype(np.float32)

    alt3 = alt[:, :, None]
    temperature = np.where(
        zz <= alt3,
        0.8 * (1.0 - zz / np.maximum(alt3, 0.2)),
        -0.25 - 0.35 * (zz - alt3),
    ).astype(np.float32)
    thawed_or_wet = (theta_u > 0.45) & (zz > alt3)
    temperature = np.where(thawed_or_wet, np.maximum(temperature, -0.05), temperature)

    rho_norm = (log_rho - np.nanpercentile(log_rho, 35)) / max(np.nanpercentile(log_rho, 92) - np.nanpercentile(log_rho, 35), 1e-6)
    rho_norm = np.clip(rho_norm, 0.0, 1.0)
    frozen = (zz > alt3).astype(np.float32)
    eic = np.clip(0.03 + 0.55 * rho_norm * frozen * (1.0 - np.clip(theta_u, 0.0, 0.85)), 0.0, 0.75).astype(np.float32)

    facies = np.full(shape, 2, dtype=np.int16)
    facies[zz <= alt3] = 0
    facies[(zz < 0.45) & (theta_u > 0.35)] = 1
    facies[eic > 0.25] = 3
    facies[(log_rho > np.nanpercentile(log_rho, 88)) & (eic <= 0.25)] = 4
    facies[(theta_u > 0.50) & (zz > alt3)] = 5

    eic_samples = []
    temp_samples = []
    theta_samples = []
    log_rho_samples = []
    facies_samples = []
    for _ in range(n_posterior):
        alt_s = np.clip(alt + rng.normal(0.0, alt_unc), 0.2, grid.z[-1])
        rho_s = log_rho + rng.normal(0.0, rho_unc)
        theta_s = np.clip(theta_u + rng.normal(0.0, theta_unc), 0.0, 0.9)
        alt3_s = alt_s[:, :, None]
        temp_s = np.where(
            zz <= alt3_s,
            0.8 * (1.0 - zz / np.maximum(alt3_s, 0.2)),
            -0.25 - 0.35 * (zz - alt3_s),
        )
        temp_s = np.where((theta_s > 0.45) & (zz > alt3_s), np.maximum(temp_s, -0.05), temp_s)
        rho_norm_s = (rho_s - np.nanpercentile(rho_s, 35)) / max(np.nanpercentile(rho_s, 92) - np.nanpercentile(rho_s, 35), 1e-6)
        rho_norm_s = np.clip(rho_norm_s, 0.0, 1.0)
        eic_s = np.clip(0.03 + 0.55 * rho_norm_s * (zz > alt3_s) * (1.0 - np.clip(theta_s, 0.0, 0.85)), 0.0, 0.75)
        fac_s = np.full(shape, 2, dtype=np.int16)
        fac_s[zz <= alt3_s] = 0
        fac_s[(zz < 0.45) & (theta_s > 0.35)] = 1
        fac_s[eic_s > 0.25] = 3
        fac_s[(rho_s > np.nanpercentile(rho_s, 88)) & (eic_s <= 0.25)] = 4
        fac_s[(theta_s > 0.50) & (zz > alt3_s)] = 5
        eic_samples.append(eic_s.astype(np.float32))
        temp_samples.append(temp_s.astype(np.float32))
        theta_samples.append(theta_s.astype(np.float32))
        log_rho_samples.append(rho_s.astype(np.float32))
        facies_samples.append(fac_s)

    eic_samples_arr = np.stack(eic_samples)
    temp_samples_arr = np.stack(temp_samples)
    theta_samples_arr = np.stack(theta_samples)
    log_rho_samples_arr = np.stack(log_rho_samples)
    facies_samples_arr = np.stack(facies_samples)
    n_facies = 7
    probs = np.zeros((*shape, n_facies), dtype=np.float32)
    for cls in range(n_facies):
        probs[..., cls] = np.mean(facies_samples_arr == cls, axis=0)
    settlement = settlement_potential_numpy(eic_samples_arr.mean(axis=0), temp_samples_arr.mean(axis=0) + 2.0, grid.dz)
    return {
        "grid_x": grid.x,
        "grid_y": grid.y,
        "grid_z": grid.z,
        "alt_mean": alt.astype(np.float32),
        "alt_std": alt_unc,
        "log_resistivity_mean": log_rho.astype(np.float32),
        "log_resistivity_std": log_rho_samples_arr.std(axis=0).astype(np.float32),
        "unfrozen_water_mean": theta_u.astype(np.float32),
        "unfrozen_water_std": theta_samples_arr.std(axis=0).astype(np.float32),
        "temperature_mean": temperature,
        "temperature_std": temp_samples_arr.std(axis=0).astype(np.float32),
        "eic_mean": eic,
        "eic_std": eic_samples_arr.std(axis=0).astype(np.float32),
        "facies_mode": np.argmax(probs, axis=-1).astype(np.int16),
        "facies_probability": probs,
        "ice_rich_probability": np.mean(eic_samples_arr > 0.30, axis=0).astype(np.float32),
        "settlement_potential": settlement.astype(np.float32),
        "eic_samples": eic_samples_arr.astype(np.float32),
    }


def evaluate_holdout_observations(train: ObservationTable, holdout: ObservationTable) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for type_name, type_id in [
        ("ert_log_resistivity", OBS_TYPES["ert_log_resistivity"]),
        ("nmr_unfrozen_water", OBS_TYPES["nmr_unfrozen_water"]),
        ("alt", OBS_TYPES["alt"]),
    ]:
        tr = train.type_ids == type_id
        ho = holdout.type_ids == type_id
        if not np.any(tr) or not np.any(ho):
            continue
        pred, _ = idw_predict(train.coords[tr], train.values[tr], holdout.coords[ho], k=8)
        obs = holdout.values[ho]
        err = pred - obs
        metrics[f"{type_name}_holdout_n"] = float(np.sum(ho))
        metrics[f"{type_name}_holdout_mae"] = float(np.mean(np.abs(err)))
        metrics[f"{type_name}_holdout_rmse"] = float(np.sqrt(np.mean(err**2)))
    return metrics

