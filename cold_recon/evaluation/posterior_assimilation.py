from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.evaluation.metrics import alt_from_temperature
from cold_recon.evaluation.uncertainty import facies_entropy


CONTINUOUS_OBS_FIELDS: dict[int, str] = {
    OBS_TYPES["borehole_eic"]: "eic",
    OBS_TYPES["borehole_temperature"]: "temperature",
    OBS_TYPES["ert_log_resistivity"]: "log_resistivity",
    OBS_TYPES["nmr_unfrozen_water"]: "unfrozen_water",
    OBS_TYPES["gtnp_temperature"]: "temperature",
}

FIELD_PRIOR_STD: dict[str, float] = {
    "eic": 0.08,
    "temperature": 0.75,
    "unfrozen_water": 0.08,
    "log_resistivity": 0.60,
}


@dataclass(frozen=True)
class PosteriorAssimilationConfig:
    horizontal_range_m: float = 4.0
    vertical_range_m: float = 0.35
    radius_factor: float = 3.0
    kernel_floor: float = 1e-4
    continuous_gain: float = 0.98
    eic_gain: float | None = None
    temperature_gain: float | None = None
    unfrozen_gain: float | None = None
    log_resistivity_gain: float | None = None
    facies_gain: float = 0.98
    alt_gain: float = 0.85
    min_sigma: float = 0.03
    max_observations_per_type: int = 2048
    seed: int = 42
    eic_min: float = 0.0
    eic_max: float = 1.0
    temperature_min: float = -10.0
    temperature_max: float = 3.0
    unfrozen_min: float = 0.0
    unfrozen_max: float = 0.8
    log_resistivity_min: float = 0.0
    log_resistivity_max: float = 12.0
    alt_transition_width_m: float = 0.35
    alt_warm_margin_c: float = 0.08
    alt_cold_margin_c: float = -0.08


def posterior_grid(posterior: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "grid_x": np.asarray(posterior["grid_x"], dtype=np.float32),
        "grid_y": np.asarray(posterior["grid_y"], dtype=np.float32),
        "grid_z": np.asarray(posterior["grid_z"], dtype=np.float32),
    }


def filter_observations_to_posterior(
    observations: ObservationTable,
    posterior: dict[str, np.ndarray],
) -> ObservationTable:
    grid = posterior_grid(posterior)
    coords = observations.coords
    keep_xyz = (
        (coords[:, 0] >= float(grid["grid_x"].min()))
        & (coords[:, 0] <= float(grid["grid_x"].max()))
        & (coords[:, 1] >= float(grid["grid_y"].min()))
        & (coords[:, 1] <= float(grid["grid_y"].max()))
        & (coords[:, 2] >= float(grid["grid_z"].min()))
        & (coords[:, 2] <= float(grid["grid_z"].max()))
    )
    alt = observations.type_ids == OBS_TYPES["alt"]
    keep_alt = (
        alt
        & (coords[:, 0] >= float(grid["grid_x"].min()))
        & (coords[:, 0] <= float(grid["grid_x"].max()))
        & (coords[:, 1] >= float(grid["grid_y"].min()))
        & (coords[:, 1] <= float(grid["grid_y"].max()))
    )
    return observations.subset(np.where(keep_xyz | keep_alt)[0])


def nearest_indices(coords: np.ndarray, posterior: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = posterior_grid(posterior)
    ix = np.abs(grid["grid_x"][None, :] - coords[:, 0:1]).argmin(axis=1)
    iy = np.abs(grid["grid_y"][None, :] - coords[:, 1:2]).argmin(axis=1)
    iz = np.abs(grid["grid_z"][None, :] - coords[:, 2:3]).argmin(axis=1)
    return ix, iy, iz


def _axis_window(axis: np.ndarray, center: float, radius: float) -> slice:
    idx = np.where(np.abs(axis - center) <= radius)[0]
    if len(idx) == 0:
        nearest = int(np.abs(axis - center).argmin())
        return slice(nearest, nearest + 1)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _local_kernel(
    posterior: dict[str, np.ndarray],
    coord: np.ndarray,
    cfg: PosteriorAssimilationConfig,
    include_z: bool = True,
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    grid = posterior_grid(posterior)
    h_range = max(float(cfg.horizontal_range_m), 1e-6)
    z_range = max(float(cfg.vertical_range_m), 1e-6)
    x_slice = _axis_window(grid["grid_x"], float(coord[0]), h_range * float(cfg.radius_factor))
    y_slice = _axis_window(grid["grid_y"], float(coord[1]), h_range * float(cfg.radius_factor))
    if include_z:
        z_slice = _axis_window(grid["grid_z"], float(coord[2]), z_range * float(cfg.radius_factor))
    else:
        z_slice = slice(0, len(grid["grid_z"]))
    x = grid["grid_x"][x_slice]
    y = grid["grid_y"][y_slice]
    z = grid["grid_z"][z_slice]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    dist2 = ((xx - float(coord[0])) / h_range) ** 2 + ((yy - float(coord[1])) / h_range) ** 2
    if include_z:
        dist2 = dist2 + ((zz - float(coord[2])) / z_range) ** 2
    kernel = np.exp(-0.5 * dist2).astype(np.float32)
    kernel[kernel < float(cfg.kernel_floor)] = 0.0
    return (x_slice, y_slice, z_slice), kernel


def _field_samples(out: dict[str, np.ndarray], field: str) -> tuple[np.ndarray, bool]:
    sample_key = f"{field}_samples"
    mean_key = f"{field}_mean"
    if sample_key in out:
        return np.asarray(out[sample_key], dtype=np.float32).copy(), True
    if mean_key in out:
        return np.asarray(out[mean_key], dtype=np.float32)[None, ...].copy(), False
    if field in out:
        return np.asarray(out[field], dtype=np.float32)[None, ...].copy(), False
    raise KeyError(f"Missing posterior field for assimilation: {sample_key}, {mean_key}, or {field}")


def _write_field_samples(out: dict[str, np.ndarray], field: str, samples: np.ndarray, had_samples: bool) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    if had_samples:
        out[f"{field}_samples"] = samples
    out[f"{field}_mean"] = samples.mean(axis=0).astype(np.float32)
    out[f"{field}_std"] = samples.std(axis=0).astype(np.float32)


def _clip_field(field: str, samples: np.ndarray, cfg: PosteriorAssimilationConfig) -> np.ndarray:
    if field == "eic":
        return np.clip(samples, cfg.eic_min, cfg.eic_max).astype(np.float32)
    if field == "temperature":
        return np.clip(samples, cfg.temperature_min, cfg.temperature_max).astype(np.float32)
    if field == "unfrozen_water":
        return np.clip(samples, cfg.unfrozen_min, cfg.unfrozen_max).astype(np.float32)
    if field == "log_resistivity":
        return np.clip(samples, cfg.log_resistivity_min, cfg.log_resistivity_max).astype(np.float32)
    return samples.astype(np.float32)


def _selected_indices(observations: ObservationTable, type_id: int, cfg: PosteriorAssimilationConfig) -> np.ndarray:
    idx = np.where((observations.type_ids == type_id) & observations.mask & np.isfinite(observations.values))[0]
    max_n = int(cfg.max_observations_per_type)
    if max_n <= 0 or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(int(cfg.seed) + int(type_id))
    selected = idx.copy()
    rng.shuffle(selected)
    return np.sort(selected[:max_n])


def _continuous_gain_for_field(field: str, cfg: PosteriorAssimilationConfig) -> float:
    if field == "eic" and cfg.eic_gain is not None:
        return float(cfg.eic_gain)
    if field == "temperature" and cfg.temperature_gain is not None:
        return float(cfg.temperature_gain)
    if field == "unfrozen_water" and cfg.unfrozen_gain is not None:
        return float(cfg.unfrozen_gain)
    if field == "log_resistivity" and cfg.log_resistivity_gain is not None:
        return float(cfg.log_resistivity_gain)
    return float(cfg.continuous_gain)


def _assimilate_continuous_type(
    out: dict[str, np.ndarray],
    observations: ObservationTable,
    type_id: int,
    field: str,
    cfg: PosteriorAssimilationConfig,
) -> int:
    idx = _selected_indices(observations, type_id, cfg)
    base_gain = _continuous_gain_for_field(field, cfg)
    if len(idx) == 0 or base_gain <= 0.0:
        return 0
    samples, had_samples = _field_samples(out, field)
    updated = 0
    for obs_idx in idx:
        coord = observations.coords[obs_idx]
        ix, iy, iz = nearest_indices(coord[None, :], out)
        pred_at_obs = samples[:, ix[0], iy[0], iz[0]]
        sigma = max(float(observations.sigma[obs_idx]), float(cfg.min_sigma))
        prior_std = max(float(np.std(pred_at_obs)), FIELD_PRIOR_STD.get(field, sigma))
        gain = base_gain * (prior_std * prior_std) / (prior_std * prior_std + sigma * sigma)
        if gain <= 0.0:
            continue
        slices, kernel = _local_kernel(out, coord, cfg, include_z=True)
        residual = float(observations.values[obs_idx]) - pred_at_obs
        xs, ys, zs = slices
        samples[:, xs, ys, zs] += gain * residual[:, None, None, None] * kernel[None, ...]
        updated += 1
    samples = _clip_field(field, samples, cfg)
    _write_field_samples(out, field, samples, had_samples)
    return int(updated)


def _facies_probability(out: dict[str, np.ndarray], n_facies: int) -> np.ndarray:
    if "facies_probability" in out:
        probs = np.asarray(out["facies_probability"], dtype=np.float32).copy()
    elif "facies_samples" in out:
        samples = np.asarray(out["facies_samples"], dtype=np.int64)
        probs = np.zeros((*samples.shape[1:], n_facies), dtype=np.float32)
        for cls in range(n_facies):
            probs[..., cls] = np.mean(samples == cls, axis=0)
    elif "facies_mode" in out:
        mode = np.asarray(out["facies_mode"], dtype=np.int64)
        probs = np.zeros((*mode.shape, n_facies), dtype=np.float32)
        for cls in range(n_facies):
            probs[..., cls] = mode == cls
    else:
        shape = next(np.asarray(v).shape for k, v in out.items() if k.endswith("_mean"))
        probs = np.zeros((*shape, n_facies), dtype=np.float32)
        probs[..., 0] = 1.0
    return probs


def _assimilate_facies(
    out: dict[str, np.ndarray],
    observations: ObservationTable,
    n_facies: int,
    cfg: PosteriorAssimilationConfig,
) -> int:
    idx = _selected_indices(observations, OBS_TYPES["borehole_facies"], cfg)
    if len(idx) == 0 or float(cfg.facies_gain) <= 0.0:
        return 0
    probs = _facies_probability(out, n_facies=n_facies)
    for obs_idx in idx:
        cls = int(np.clip(round(float(observations.values[obs_idx])), 0, n_facies - 1))
        target = np.zeros((n_facies,), dtype=np.float32)
        target[cls] = 1.0
        slices, kernel = _local_kernel(out, observations.coords[obs_idx], cfg, include_z=True)
        alpha = np.clip(float(cfg.facies_gain) * kernel[..., None], 0.0, 1.0)
        xs, ys, zs = slices
        probs[xs, ys, zs, :] = (1.0 - alpha) * probs[xs, ys, zs, :] + alpha * target
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-6)
    out["facies_probability"] = probs.astype(np.float32)
    out["facies_entropy"] = facies_entropy(probs).astype(np.float32)
    out["facies_mode"] = np.argmax(probs, axis=-1).astype(np.int16)
    if "facies_samples" in out:
        samples = np.asarray(out["facies_samples"], dtype=np.int16).copy()
        ix, iy, iz = nearest_indices(observations.coords[idx], out)
        for obs_i, x_i, y_i, z_i in zip(idx, ix, iy, iz):
            cls = int(np.clip(round(float(observations.values[obs_i])), 0, n_facies - 1))
            samples[:, x_i, y_i, z_i] = cls
        out["facies_samples"] = samples
    return int(len(idx))


def _assimilate_alt(
    out: dict[str, np.ndarray],
    observations: ObservationTable,
    cfg: PosteriorAssimilationConfig,
) -> int:
    idx = _selected_indices(observations, OBS_TYPES["alt"], cfg)
    if len(idx) == 0 or float(cfg.alt_gain) <= 0.0 or ("temperature_samples" not in out and "temperature_mean" not in out):
        return 0
    samples, had_samples = _field_samples(out, "temperature")
    z = posterior_grid(out)["grid_z"]
    for obs_idx in idx:
        coord = observations.coords[obs_idx]
        alt_depth = float(np.clip(observations.values[obs_idx], z.min(), z.max()))
        target_profile = np.where(z <= alt_depth, cfg.alt_warm_margin_c, cfg.alt_cold_margin_c).astype(np.float32)
        transition = np.exp(-0.5 * ((z - alt_depth) / max(float(cfg.alt_transition_width_m), 1e-6)) ** 2).astype(np.float32)
        coord_for_kernel = np.asarray([coord[0], coord[1], alt_depth], dtype=np.float32)
        slices, kernel = _local_kernel(out, coord_for_kernel, cfg, include_z=False)
        xs, ys, _ = slices
        xy_kernel = kernel[:, :, 0]
        current = samples[:, xs, ys, :]
        correction = target_profile[None, None, None, :] - current
        weight = float(cfg.alt_gain) * xy_kernel[None, :, :, None] * transition[None, None, None, :]
        samples[:, xs, ys, :] += weight * correction
    samples = _clip_field("temperature", samples, cfg)
    _write_field_samples(out, "temperature", samples, had_samples)
    return int(len(idx))


def _refresh_derived_fields(out: dict[str, np.ndarray]) -> None:
    if "log_resistivity_samples" in out:
        res = np.exp(np.clip(out["log_resistivity_samples"], 0.0, 12.0)).astype(np.float32)
        out["resistivity_samples"] = res
        out["resistivity_mean"] = res.mean(axis=0).astype(np.float32)
        out["resistivity_std"] = res.std(axis=0).astype(np.float32)
    elif "log_resistivity_mean" in out:
        out["resistivity_mean"] = np.exp(np.clip(out["log_resistivity_mean"], 0.0, 12.0)).astype(np.float32)
    if "eic_samples" in out:
        out["ice_rich_probability"] = np.mean(out["eic_samples"] > 0.30, axis=0).astype(np.float32)
    elif "eic_mean" in out:
        out["ice_rich_probability"] = (out["eic_mean"] > 0.30).astype(np.float32)


def observation_residual_metrics(
    posterior: dict[str, np.ndarray],
    observations: ObservationTable,
) -> dict[str, float]:
    obs = filter_observations_to_posterior(observations, posterior)
    metrics: dict[str, float] = {"n": float(obs.n_obs)}
    if obs.n_obs == 0:
        return metrics
    ix, iy, iz = nearest_indices(obs.coords, posterior)
    for type_id, field in CONTINUOUS_OBS_FIELDS.items():
        mask = obs.type_ids == type_id
        mean_key = f"{field}_mean"
        if not np.any(mask) or mean_key not in posterior:
            continue
        pred = posterior[mean_key][ix[mask], iy[mask], iz[mask]]
        err = pred - obs.values[mask]
        name = next(name for name, value in OBS_TYPES.items() if value == type_id)
        metrics[f"{name}_n"] = float(np.sum(mask))
        metrics[f"{name}_mae"] = float(np.mean(np.abs(err)))
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(err**2)))
    alt_mask = obs.type_ids == OBS_TYPES["alt"]
    if np.any(alt_mask) and "temperature_mean" in posterior:
        grid = posterior_grid(posterior)
        pred_alt = alt_from_temperature(posterior["temperature_mean"], grid["grid_z"])
        pred = pred_alt[ix[alt_mask], iy[alt_mask]]
        err = pred - obs.values[alt_mask]
        metrics["alt_n"] = float(np.sum(alt_mask))
        metrics["alt_mae"] = float(np.mean(np.abs(err)))
        metrics["alt_rmse"] = float(np.sqrt(np.mean(err**2)))
    return metrics


def assimilate_posterior_to_observations(
    posterior: dict[str, np.ndarray],
    observations: ObservationTable,
    n_facies: int = 7,
    cfg: PosteriorAssimilationConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    cfg = cfg or PosteriorAssimilationConfig()
    obs = filter_observations_to_posterior(observations, posterior)
    out = {key: np.asarray(value).copy() for key, value in posterior.items()}
    before = observation_residual_metrics(out, obs)
    counts: dict[str, float] = {"assimilation_n": float(obs.n_obs)}
    counts["assimilated_facies"] = float(_assimilate_facies(out, obs, n_facies=n_facies, cfg=cfg))
    for type_id, field in CONTINUOUS_OBS_FIELDS.items():
        name = next(name for name, value in OBS_TYPES.items() if value == type_id)
        counts[f"assimilated_{name}"] = float(_assimilate_continuous_type(out, obs, type_id, field, cfg))
    counts["assimilated_alt"] = float(_assimilate_alt(out, obs, cfg))
    _refresh_derived_fields(out)
    after = observation_residual_metrics(out, obs)
    metrics = dict(counts)
    for key, value in before.items():
        metrics[f"before_{key}"] = value
    for key, value in after.items():
        metrics[f"after_{key}"] = value
    out["posterior_assimilation_gain"] = np.asarray(cfg.continuous_gain, dtype=np.float32)
    out["posterior_assimilation_horizontal_range_m"] = np.asarray(cfg.horizontal_range_m, dtype=np.float32)
    out["posterior_assimilation_vertical_range_m"] = np.asarray(cfg.vertical_range_m, dtype=np.float32)
    return out, metrics
