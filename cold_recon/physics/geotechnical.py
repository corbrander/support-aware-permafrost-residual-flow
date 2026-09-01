from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cold_recon.evaluation.physics_consistency import facies_to_probability
from cold_recon.physics.settlement import settlement_potential_numpy


@dataclass(frozen=True)
class GeotechnicalParameters:
    base_shear_strength_kpa: tuple[float, ...] = (45.0, 18.0, 80.0, 65.0, 150.0, 22.0, 8.0)
    base_modulus_mpa: tuple[float, ...] = (25.0, 6.0, 45.0, 35.0, 120.0, 10.0, 2.0)
    thaw_shear_reduction: float = 0.45
    thaw_modulus_reduction: float = 0.55
    eic_shear_penalty: float = 0.55
    eic_modulus_penalty: float = 0.65
    unfrozen_shear_penalty: float = 0.35
    unfrozen_modulus_penalty: float = 0.45
    min_shear_strength_kpa: float = 3.0
    min_modulus_mpa: float = 0.5
    future_warming_c: float = 2.0


def _as_facies_probability(fields: dict[str, np.ndarray], n_facies: int) -> np.ndarray:
    if "facies_probability" in fields:
        probs = np.asarray(fields["facies_probability"], dtype=np.float32)
    elif "facies_mode" in fields:
        probs = facies_to_probability(fields["facies_mode"], n_facies=n_facies)
    elif "facies" in fields:
        probs = facies_to_probability(fields["facies"], n_facies=n_facies)
    else:
        raise KeyError("fields must contain facies_probability, facies_mode, or facies")
    if probs.shape[-1] < n_facies:
        pad = np.zeros((*probs.shape[:-1], n_facies - probs.shape[-1]), dtype=np.float32)
        probs = np.concatenate([probs, pad], axis=-1)
    return probs[..., :n_facies].astype(np.float32)


def _expected_by_facies(facies_probability: np.ndarray, values: tuple[float, ...], n_facies: int) -> np.ndarray:
    value_arr = np.asarray(values[:n_facies], dtype=np.float32)
    if len(value_arr) < n_facies:
        value_arr = np.pad(value_arr, (0, n_facies - len(value_arr)), constant_values=float(value_arr[-1]))
    return np.sum(facies_probability * value_arr[None, None, None, :], axis=-1).astype(np.float32)


def _degradation_factor(
    eic: np.ndarray,
    temperature: np.ndarray,
    unfrozen_water: np.ndarray,
    thaw_reduction: float,
    eic_penalty: float,
    unfrozen_penalty: float,
) -> np.ndarray:
    thaw_arg = np.clip(-3.0 * np.asarray(temperature, dtype=np.float32), -60.0, 60.0)
    thawed = 1.0 / (1.0 + np.exp(thaw_arg))
    eic_term = np.clip(np.asarray(eic, dtype=np.float32), 0.0, 1.0)
    uw_term = np.clip(np.asarray(unfrozen_water, dtype=np.float32) / 0.8, 0.0, 1.0)
    factor = (1.0 - thaw_reduction * thawed) * (1.0 - eic_penalty * eic_term * thawed) * (1.0 - unfrozen_penalty * uw_term)
    return np.clip(factor, 0.02, 1.5).astype(np.float32)


def geotechnical_property_fields(
    fields: dict[str, np.ndarray],
    n_facies: int = 7,
    params: GeotechnicalParameters | None = None,
) -> dict[str, np.ndarray]:
    params = params or GeotechnicalParameters()
    required = ["eic_mean", "temperature_mean", "unfrozen_water_mean"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise KeyError(f"Missing geotechnical fields: {missing}")
    facies_probability = _as_facies_probability(fields, n_facies=n_facies)
    eic = np.clip(np.asarray(fields["eic_mean"], dtype=np.float32), 0.0, 1.0)
    temperature = np.asarray(fields["temperature_mean"], dtype=np.float32)
    unfrozen = np.clip(np.asarray(fields["unfrozen_water_mean"], dtype=np.float32), 0.0, 0.8)
    base_su = _expected_by_facies(facies_probability, params.base_shear_strength_kpa, n_facies)
    base_modulus = _expected_by_facies(facies_probability, params.base_modulus_mpa, n_facies)
    current_su = base_su * _degradation_factor(
        eic,
        temperature,
        unfrozen,
        params.thaw_shear_reduction,
        params.eic_shear_penalty,
        params.unfrozen_shear_penalty,
    )
    current_modulus = base_modulus * _degradation_factor(
        eic,
        temperature,
        unfrozen,
        params.thaw_modulus_reduction,
        params.eic_modulus_penalty,
        params.unfrozen_modulus_penalty,
    )
    future_temp = temperature + float(params.future_warming_c)
    future_su = base_su * _degradation_factor(
        eic,
        future_temp,
        unfrozen,
        params.thaw_shear_reduction,
        params.eic_shear_penalty,
        params.unfrozen_shear_penalty,
    )
    future_modulus = base_modulus * _degradation_factor(
        eic,
        future_temp,
        unfrozen,
        params.thaw_modulus_reduction,
        params.eic_modulus_penalty,
        params.unfrozen_modulus_penalty,
    )
    current_su = np.clip(current_su, params.min_shear_strength_kpa, None).astype(np.float32)
    future_su = np.clip(future_su, params.min_shear_strength_kpa, None).astype(np.float32)
    current_modulus = np.clip(current_modulus, params.min_modulus_mpa, None).astype(np.float32)
    future_modulus = np.clip(future_modulus, params.min_modulus_mpa, None).astype(np.float32)
    strength_loss_ratio = np.clip((current_su - future_su) / np.maximum(current_su, 1e-6), 0.0, 1.0).astype(np.float32)
    modulus_loss_ratio = np.clip((current_modulus - future_modulus) / np.maximum(current_modulus, 1e-6), 0.0, 1.0).astype(np.float32)
    return {
        "current_shear_strength_kpa": current_su,
        "future_shear_strength_kpa": future_su,
        "current_modulus_mpa": current_modulus,
        "future_modulus_mpa": future_modulus,
        "strength_loss_ratio": strength_loss_ratio,
        "modulus_loss_ratio": modulus_loss_ratio,
    }


def _spacing(axis: np.ndarray) -> float:
    arr = np.asarray(axis, dtype=np.float32)
    return float(np.mean(np.diff(arr))) if len(arr) > 1 else 1.0


def differential_settlement_gradient(settlement: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if settlement.shape[0] < 2 or settlement.shape[1] < 2:
        return np.zeros_like(settlement, dtype=np.float32)
    gx, gy = np.gradient(settlement.astype(np.float32), x.astype(np.float32), y.astype(np.float32), edge_order=1)
    return np.hypot(gx, gy).astype(np.float32)


def engineering_risk_index(
    settlement: np.ndarray,
    differential_gradient: np.ndarray,
    strength_loss_surface: np.ndarray,
    low_modulus_surface: np.ndarray,
) -> np.ndarray:
    def norm(arr: np.ndarray, q: float = 95.0) -> np.ndarray:
        vals = np.asarray(arr, dtype=np.float32)
        finite = np.isfinite(vals)
        if not np.any(finite):
            return np.zeros_like(vals, dtype=np.float32)
        scale = float(np.nanpercentile(vals[finite], q))
        if scale <= 0.0:
            return np.zeros_like(vals, dtype=np.float32)
        return np.clip(vals / scale, 0.0, 1.0).astype(np.float32)

    risk = (
        0.35 * norm(settlement)
        + 0.25 * norm(differential_gradient)
        + 0.25 * norm(strength_loss_surface)
        + 0.15 * norm(low_modulus_surface)
    )
    return np.clip(risk, 0.0, 1.0).astype(np.float32)


def geotechnical_summary(
    posterior: dict[str, np.ndarray],
    n_facies: int = 7,
    params: GeotechnicalParameters | None = None,
    max_depth_m: float = 3.0,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    params = params or GeotechnicalParameters()
    fields = geotechnical_property_fields(posterior, n_facies=n_facies, params=params)
    x = np.asarray(posterior.get("grid_x", np.arange(posterior["eic_mean"].shape[0])), dtype=np.float32)
    y = np.asarray(posterior.get("grid_y", np.arange(posterior["eic_mean"].shape[1])), dtype=np.float32)
    z = np.asarray(posterior.get("grid_z", np.arange(posterior["eic_mean"].shape[2])), dtype=np.float32)
    dz = _spacing(z)
    if "settlement_potential" in posterior:
        settlement = np.asarray(posterior["settlement_potential"], dtype=np.float32)
    else:
        settlement = settlement_potential_numpy(
            posterior["eic_mean"],
            posterior["temperature_mean"] + params.future_warming_c,
            dz,
        ).astype(np.float32)
    diff_grad = differential_settlement_gradient(settlement, x, y)
    depth_mask = z <= float(max_depth_m)
    if not np.any(depth_mask):
        depth_mask[0] = True
    strength_loss_surface = np.nanmean(fields["strength_loss_ratio"][:, :, depth_mask], axis=2).astype(np.float32)
    low_modulus_surface = (1.0 / np.maximum(np.nanmean(fields["future_modulus_mpa"][:, :, depth_mask], axis=2), 1e-6)).astype(np.float32)
    risk_index = engineering_risk_index(settlement, diff_grad, strength_loss_surface, low_modulus_surface)
    fields.update(
        {
            "settlement_potential": settlement.astype(np.float32),
            "differential_settlement_gradient": diff_grad.astype(np.float32),
            "strength_loss_surface": strength_loss_surface,
            "low_modulus_surface": low_modulus_surface,
            "engineering_risk_index": risk_index,
        }
    )
    metrics = {
        "current_shear_strength_p10_kpa": float(np.percentile(fields["current_shear_strength_kpa"], 10.0)),
        "future_shear_strength_p10_kpa": float(np.percentile(fields["future_shear_strength_kpa"], 10.0)),
        "current_modulus_p10_mpa": float(np.percentile(fields["current_modulus_mpa"], 10.0)),
        "future_modulus_p10_mpa": float(np.percentile(fields["future_modulus_mpa"], 10.0)),
        "strength_loss_mean": float(np.mean(fields["strength_loss_ratio"])),
        "modulus_loss_mean": float(np.mean(fields["modulus_loss_ratio"])),
        "settlement_potential_mean_m": float(np.mean(settlement)),
        "settlement_potential_p95_m": float(np.percentile(settlement, 95.0)),
        "differential_settlement_gradient_p95": float(np.percentile(diff_grad, 95.0)),
        "engineering_risk_mean": float(np.mean(risk_index)),
        "engineering_risk_p95": float(np.percentile(risk_index, 95.0)),
    }
    return fields, metrics
