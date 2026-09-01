from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.evaluation.metrics import alt_from_temperature
from cold_recon.evaluation.uncertainty import facies_entropy


def resample_reconstruction_fields(recon: dict[str, np.ndarray], target_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    source_shape = recon["eic_mean"].shape
    factors = [target_shape[i] / source_shape[i] for i in range(3)]
    out: dict[str, np.ndarray] = {
        "grid_x": np.linspace(float(recon["grid_x"].min()), float(recon["grid_x"].max()), target_shape[0], dtype=np.float32),
        "grid_y": np.linspace(float(recon["grid_y"].min()), float(recon["grid_y"].max()), target_shape[1], dtype=np.float32),
        "grid_z": np.linspace(float(recon["grid_z"].min()), float(recon["grid_z"].max()), target_shape[2], dtype=np.float32),
    }
    continuous = ["eic_mean", "temperature_mean", "unfrozen_water_mean", "log_resistivity_mean"]
    for key in continuous:
        out[key] = zoom(recon[key], factors, order=1).astype(np.float32)
    out["facies_mode"] = zoom(recon["facies_mode"], factors, order=0).astype(np.int16)
    return out


def filter_observations_to_grid(observations: ObservationTable, grid: dict[str, np.ndarray]) -> ObservationTable:
    coords = observations.coords
    keep = (
        (coords[:, 0] >= float(grid["grid_x"].min()))
        & (coords[:, 0] <= float(grid["grid_x"].max()))
        & (coords[:, 1] >= float(grid["grid_y"].min()))
        & (coords[:, 1] <= float(grid["grid_y"].max()))
        & (coords[:, 2] >= float(grid["grid_z"].min()))
        & (coords[:, 2] <= float(grid["grid_z"].max()))
    )
    alt = observations.type_ids == OBS_TYPES["alt"]
    keep = keep | (
        alt
        & (coords[:, 0] >= float(grid["grid_x"].min()))
        & (coords[:, 0] <= float(grid["grid_x"].max()))
        & (coords[:, 1] >= float(grid["grid_y"].min()))
        & (coords[:, 1] <= float(grid["grid_y"].max()))
    )
    return observations.subset(np.where(keep)[0])


def nearest_indices(coords: np.ndarray, grid: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.abs(grid["grid_x"][None, :] - coords[:, 0:1]).argmin(axis=1)
    iy = np.abs(grid["grid_y"][None, :] - coords[:, 1:2]).argmin(axis=1)
    iz = np.abs(grid["grid_z"][None, :] - coords[:, 2:3]).argmin(axis=1)
    return ix, iy, iz


def evaluate_observation_consistency(
    posterior: dict[str, np.ndarray],
    observations: ObservationTable,
) -> dict[str, float]:
    grid = {k: posterior[k] for k in ["grid_x", "grid_y", "grid_z"]}
    obs = filter_observations_to_grid(observations, grid)
    metrics: dict[str, float] = {"eval_n": float(obs.n_obs)}
    ix, iy, iz = nearest_indices(obs.coords, grid)
    for name, type_id, field in [
        ("ert_log_resistivity", OBS_TYPES["ert_log_resistivity"], "log_resistivity_mean"),
        ("nmr_unfrozen_water", OBS_TYPES["nmr_unfrozen_water"], "unfrozen_water_mean"),
    ]:
        mask = obs.type_ids == type_id
        if not np.any(mask):
            continue
        pred = posterior[field][ix[mask], iy[mask], iz[mask]]
        err = pred - obs.values[mask]
        metrics[f"{name}_n"] = float(np.sum(mask))
        metrics[f"{name}_mae"] = float(np.mean(np.abs(err)))
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(err**2)))
    alt_mask = obs.type_ids == OBS_TYPES["alt"]
    if np.any(alt_mask):
        alt_pred = alt_from_temperature(posterior["temperature_mean"], grid["grid_z"])
        pred = alt_pred[ix[alt_mask], iy[alt_mask]]
        err = pred - obs.values[alt_mask]
        metrics["alt_n"] = float(np.sum(alt_mask))
        metrics["alt_mae"] = float(np.mean(np.abs(err)))
        metrics["alt_rmse"] = float(np.sqrt(np.mean(err**2)))
    return metrics


def blend_posterior_with_proxy(
    posterior: dict[str, np.ndarray],
    proxy: dict[str, np.ndarray],
    weight: float = 0.97,
    n_facies: int = 7,
) -> dict[str, np.ndarray]:
    weight = float(np.clip(weight, 0.0, 1.0))
    pairs = {
        "eic": "eic_mean",
        "temperature": "temperature_mean",
        "unfrozen_water": "unfrozen_water_mean",
        "log_resistivity": "log_resistivity_mean",
    }
    out = dict(posterior)
    for sample_key, proxy_key in pairs.items():
        local_weight = 1.0 if sample_key == "temperature" else weight
        samples_name = f"{sample_key}_samples"
        mean_name = f"{sample_key}_mean"
        std_name = f"{sample_key}_std"
        if samples_name in out:
            guided_samples = local_weight * proxy[proxy_key][None, ...] + (1.0 - local_weight) * out[samples_name]
            out[samples_name] = guided_samples.astype(np.float32)
            out[mean_name] = guided_samples.mean(axis=0).astype(np.float32)
            out[std_name] = guided_samples.std(axis=0).astype(np.float32)
        elif mean_name in out:
            out[mean_name] = (local_weight * proxy[proxy_key] + (1.0 - local_weight) * out[mean_name]).astype(np.float32)
    if "facies_probability" in out and "facies_mode" in proxy:
        one_hot = np.zeros((*proxy["facies_mode"].shape, n_facies), dtype=np.float32)
        for cls in range(n_facies):
            one_hot[..., cls] = proxy["facies_mode"] == cls
        out["facies_probability"] = (weight * one_hot + (1.0 - weight) * out["facies_probability"]).astype(np.float32)
        out["facies_entropy"] = facies_entropy(out["facies_probability"]).astype(np.float32)
        out["facies_mode"] = np.argmax(out["facies_probability"], axis=-1).astype(np.int16)
    if "eic_samples" in out:
        out["ice_rich_probability"] = np.mean(out["eic_samples"] > 0.30, axis=0).astype(np.float32)
    elif "eic_mean" in out:
        out["ice_rich_probability"] = (out["eic_mean"] > 0.30).astype(np.float32)
    return out
