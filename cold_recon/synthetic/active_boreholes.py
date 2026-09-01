from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cold_recon.baselines.idw import reconstruct_idw
from cold_recon.data.data_schema import OBS_TYPES, ObservationTable


@dataclass
class ActiveBoreholeConfig:
    enabled: bool = False
    n_boreholes: int = 0
    depth_step: int = 2
    max_depth_m: float = 4.0
    min_spacing_cells: int = 6
    log_resistivity_weight: float = 1.0
    eic_weight: float = 1.2
    cold_weight: float = 0.25
    dry_weight: float = 0.20
    facies_event_weight: float = 0.35
    eic_noise: float = 0.03
    temperature_noise: float = 0.20
    facies_sigma: float = 0.0
    seed: int = 42


def active_borehole_config_from_dict(config: dict[str, Any]) -> ActiveBoreholeConfig:
    cfg = config.get("multisample_diffusion", {}).get("active_borehole_sampling", {})
    noise = config.get("synthetic", {}).get("noise", {})
    return ActiveBoreholeConfig(
        enabled=bool(cfg.get("enabled", False)),
        n_boreholes=int(cfg.get("n_boreholes", 0)),
        depth_step=int(cfg.get("depth_step", cfg.get("borehole_depth_step", 2))),
        max_depth_m=float(cfg.get("max_depth_m", 4.0)),
        min_spacing_cells=int(cfg.get("min_spacing_cells", 6)),
        log_resistivity_weight=float(cfg.get("log_resistivity_weight", 1.0)),
        eic_weight=float(cfg.get("eic_weight", 1.2)),
        cold_weight=float(cfg.get("cold_weight", 0.25)),
        dry_weight=float(cfg.get("dry_weight", 0.20)),
        facies_event_weight=float(cfg.get("facies_event_weight", 0.35)),
        eic_noise=float(cfg.get("eic_noise", noise.get("eic", 0.03))),
        temperature_noise=float(cfg.get("temperature_noise", noise.get("temperature", 0.20))),
        facies_sigma=float(cfg.get("facies_sigma", 0.0)),
        seed=int(cfg.get("seed", config.get("project", {}).get("seed", 42))),
    )


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [5.0, 95.0])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - float(lo)) / float(hi - lo), 0.0, 1.0).astype(np.float32)


def _nearest_xy_indices(coords: np.ndarray, grid: dict[str, Any]) -> np.ndarray:
    x = np.asarray(grid["x"], dtype=np.float32)
    y = np.asarray(grid["y"], dtype=np.float32)
    ix = np.abs(x[None, :] - coords[:, 0:1]).argmin(axis=1)
    iy = np.abs(y[None, :] - coords[:, 1:2]).argmin(axis=1)
    return np.column_stack([ix, iy]).astype(np.int64)


def active_borehole_score(sample: dict[str, Any], cfg: ActiveBoreholeConfig, n_facies: int = 7) -> np.ndarray:
    proxy = reconstruct_idw(sample["observations"], sample["grid"], n_facies=n_facies)
    shape = sample["fields"]["eic"].shape
    zeros = np.zeros(shape, dtype=np.float32)
    eic = np.asarray(proxy.get("eic", zeros), dtype=np.float32)
    temperature = np.asarray(proxy.get("temperature", zeros), dtype=np.float32)
    unfrozen = np.asarray(proxy.get("unfrozen_water", zeros), dtype=np.float32)
    log_resistivity = np.asarray(proxy.get("log_resistivity", zeros), dtype=np.float32)
    facies = np.asarray(proxy.get("facies", np.zeros(shape, dtype=np.int16)), dtype=np.int16)

    score_3d = (
        cfg.log_resistivity_weight * _robust_unit_interval(log_resistivity)
        + cfg.eic_weight * _robust_unit_interval(eic)
        + cfg.cold_weight * _robust_unit_interval(-temperature)
        + cfg.dry_weight * _robust_unit_interval(-unfrozen)
        + cfg.facies_event_weight * np.isin(facies, [3, 6]).astype(np.float32)
    )
    z = np.asarray(sample["grid"]["z"], dtype=np.float32)
    shallow = z <= float(cfg.max_depth_m)
    if not np.any(shallow):
        shallow = np.ones_like(z, dtype=bool)
    return np.max(score_3d[:, :, shallow], axis=2).astype(np.float32)


def select_active_boreholes(
    sample: dict[str, Any],
    cfg: ActiveBoreholeConfig,
    n_facies: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.enabled or cfg.n_boreholes <= 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    score = active_borehole_score(sample, cfg, n_facies=n_facies)
    order = np.argsort(score.reshape(-1))[::-1]
    nx, ny = score.shape
    selected: list[tuple[int, int]] = []
    selected_scores: list[float] = []

    observations = sample["observations"]
    facies_mask = observations.type_ids == OBS_TYPES["borehole_facies"]
    occupied = _nearest_xy_indices(observations.coords[facies_mask], sample["grid"]) if np.any(facies_mask) else np.zeros((0, 2), dtype=np.int64)
    blocked = [tuple(map(int, row)) for row in occupied]
    min_spacing = max(int(cfg.min_spacing_cells), 0)

    def far_enough(ix: int, iy: int, points: list[tuple[int, int]]) -> bool:
        if min_spacing <= 0:
            return True
        for px, py in points:
            if (ix - px) ** 2 + (iy - py) ** 2 < min_spacing**2:
                return False
        return True

    for flat_idx in order:
        ix = int(flat_idx // ny)
        iy = int(flat_idx % ny)
        if not far_enough(ix, iy, blocked + selected):
            continue
        selected.append((ix, iy))
        selected_scores.append(float(score[ix, iy]))
        if len(selected) >= int(cfg.n_boreholes):
            break
    return np.asarray(selected, dtype=np.int64), np.asarray(selected_scores, dtype=np.float32)


def _append_observations(base: ObservationTable, coords: list[tuple[float, float, float]], type_ids: list[int], values: list[float], sigma: list[float]) -> ObservationTable:
    if not coords:
        return base
    new_n = len(coords)
    times = base.times if base.times is not None else np.full((base.n_obs,), np.nan, dtype=np.float32)
    return ObservationTable(
        coords=np.concatenate([base.coords, np.asarray(coords, dtype=np.float32)], axis=0),
        type_ids=np.concatenate([base.type_ids, np.asarray(type_ids, dtype=np.int64)], axis=0),
        values=np.concatenate([base.values, np.asarray(values, dtype=np.float32)], axis=0),
        sigma=np.concatenate([base.sigma, np.asarray(sigma, dtype=np.float32)], axis=0),
        mask=np.concatenate([base.mask, np.ones((new_n,), dtype=bool)], axis=0),
        times=np.concatenate([times, np.full((new_n,), np.nan, dtype=np.float32)], axis=0),
    )


def augment_sample_with_active_boreholes(
    sample: dict[str, Any],
    cfg: ActiveBoreholeConfig,
    n_facies: int = 7,
    seed: int | None = None,
) -> dict[str, Any]:
    if not cfg.enabled or cfg.n_boreholes <= 0:
        return sample
    selected, scores = select_active_boreholes(sample, cfg, n_facies=n_facies)
    if selected.size == 0:
        return sample
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    grid = sample["grid"]
    fields = sample["fields"]
    z = np.asarray(grid["z"], dtype=np.float32)
    max_depth_idx = int(np.searchsorted(z, float(cfg.max_depth_m), side="right"))
    max_depth_idx = max(1, min(max_depth_idx, len(z)))
    depth_indices = range(0, max_depth_idx, max(1, int(cfg.depth_step)))
    coords: list[tuple[float, float, float]] = []
    type_ids: list[int] = []
    values: list[float] = []
    sigma: list[float] = []
    wedge_obs = 0
    ice_rich_obs = 0

    for ix, iy in selected:
        for iz in depth_indices:
            facies_value = int(fields["facies"][ix, iy, iz])
            eic_value = float(fields["eic"][ix, iy, iz]) + float(rng.normal(0.0, cfg.eic_noise))
            temp_value = float(fields["temperature"][ix, iy, iz]) + float(rng.normal(0.0, cfg.temperature_noise))
            coord = (float(grid["x"][ix]), float(grid["y"][iy]), float(grid["z"][iz]))
            coords.extend([coord, coord, coord])
            type_ids.extend([OBS_TYPES["borehole_facies"], OBS_TYPES["borehole_eic"], OBS_TYPES["borehole_temperature"]])
            values.extend([float(facies_value), eic_value, temp_value])
            sigma.extend([float(cfg.facies_sigma), float(cfg.eic_noise), float(cfg.temperature_noise)])
            if facies_value == 6:
                wedge_obs += 1
            if facies_value in (3, 6):
                ice_rich_obs += 1

    out = {
        **sample,
        "base_observations": sample["observations"],
        "observations": _append_observations(sample["observations"], coords, type_ids, values, sigma),
        "metadata": dict(sample.get("metadata", {})),
    }
    out["metadata"]["active_borehole_sampling"] = {
        "enabled": True,
        "n_boreholes_requested": int(cfg.n_boreholes),
        "n_boreholes_added": int(len(selected)),
        "n_observations_added": int(len(coords)),
        "depth_step": int(cfg.depth_step),
        "max_depth_m": float(cfg.max_depth_m),
        "selected_xy_indices": selected.astype(int).tolist(),
        "selected_scores": scores.astype(float).tolist(),
        "active_wedge_ice_observations_n": int(wedge_obs),
        "active_ice_rich_observations_n": int(ice_rich_obs),
    }
    return out
