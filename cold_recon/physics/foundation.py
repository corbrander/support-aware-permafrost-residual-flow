from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.ndimage import uniform_filter

from cold_recon.physics.geotechnical import GeotechnicalParameters, geotechnical_property_fields
from cold_recon.physics.settlement import settlement_potential_numpy


@dataclass(frozen=True)
class FoundationDesign:
    width_m: float = 2.0
    length_m: float = 4.0
    embedment_m: float = 0.5
    net_pressure_kpa: float = 75.0
    unit_weight_kpa_per_m: float = 18.0
    poisson_ratio: float = 0.35
    settlement_influence_factor: float = 1.10
    bearing_nc: float = 5.14
    target_factor_of_safety: float = 2.5
    allowable_total_settlement_m: float = 0.075
    influence_depth_factor: float = 2.0


def _axis_spacing(axis: np.ndarray) -> float:
    arr = np.asarray(axis, dtype=np.float32)
    return float(np.mean(np.diff(arr))) if len(arr) > 1 else 1.0


def _depth_weights(z: np.ndarray, influence_depth_m: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    if len(z) == 0:
        raise ValueError("z axis cannot be empty")
    weights = np.clip(1.0 - z / max(float(influence_depth_m), 1e-6), 0.0, 1.0)
    if not np.any(weights > 0.0):
        weights[0] = 1.0
    return weights.astype(np.float32)


def depth_weighted_harmonic_mean(field: np.ndarray, z: np.ndarray, influence_depth_m: float) -> np.ndarray:
    values = np.maximum(np.asarray(field, dtype=np.float32), 1e-6)
    weights = _depth_weights(z, influence_depth_m)
    weights = weights / np.maximum(float(np.sum(weights)), 1e-6)
    denom = np.sum(weights[None, None, :] / values, axis=2)
    return (1.0 / np.maximum(denom, 1e-6)).astype(np.float32)


def _footprint_average(surface: np.ndarray, x: np.ndarray, y: np.ndarray, design: FoundationDesign) -> np.ndarray:
    dx = max(_axis_spacing(x), 1e-6)
    dy = max(_axis_spacing(y), 1e-6)
    wx = max(1, int(round(float(design.width_m) / dx)))
    wy = max(1, int(round(float(design.length_m) / dy)))
    return uniform_filter(np.asarray(surface, dtype=np.float32), size=(wx, wy), mode="nearest").astype(np.float32)


def _risk_normalize_positive(arr: np.ndarray, q: float = 95.0) -> np.ndarray:
    vals = np.maximum(np.asarray(arr, dtype=np.float32), 0.0)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.zeros_like(vals, dtype=np.float32)
    scale = float(np.nanpercentile(vals[finite], q))
    if scale <= 0.0:
        return np.zeros_like(vals, dtype=np.float32)
    return np.clip(vals / scale, 0.0, 1.0).astype(np.float32)


def foundation_response_fields(
    geotech_fields: dict[str, np.ndarray],
    posterior: dict[str, np.ndarray],
    design: FoundationDesign | None = None,
    settlement: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute mapped shallow-foundation response from reconstructed geotechnical fields."""
    design = design or FoundationDesign()
    x = np.asarray(posterior.get("grid_x", np.arange(geotech_fields["future_shear_strength_kpa"].shape[0])), dtype=np.float32)
    y = np.asarray(posterior.get("grid_y", np.arange(geotech_fields["future_shear_strength_kpa"].shape[1])), dtype=np.float32)
    z = np.asarray(posterior.get("grid_z", np.arange(geotech_fields["future_shear_strength_kpa"].shape[2])), dtype=np.float32)
    influence_depth = float(design.influence_depth_factor) * float(design.width_m)
    su = depth_weighted_harmonic_mean(geotech_fields["future_shear_strength_kpa"], z, influence_depth)
    modulus_mpa = depth_weighted_harmonic_mean(geotech_fields["future_modulus_mpa"], z, influence_depth)
    su = _footprint_average(su, x, y, design)
    modulus_mpa = _footprint_average(modulus_mpa, x, y, design)
    shape_factor = 1.0 + 0.2 * min(float(design.width_m), float(design.length_m)) / max(float(design.width_m), float(design.length_m), 1e-6)
    ultimate_bearing = (
        float(design.bearing_nc) * shape_factor * su
        + float(design.unit_weight_kpa_per_m) * float(design.embedment_m)
    ).astype(np.float32)
    factor_of_safety = (ultimate_bearing / max(float(design.net_pressure_kpa), 1e-6)).astype(np.float32)
    modulus_kpa = np.maximum(modulus_mpa * 1000.0, 1e-6)
    elastic_settlement = (
        float(design.net_pressure_kpa)
        * float(design.width_m)
        * (1.0 - float(design.poisson_ratio) ** 2)
        * float(design.settlement_influence_factor)
        / modulus_kpa
    ).astype(np.float32)
    if settlement is None:
        dz = _axis_spacing(z)
        settlement = settlement_potential_numpy(
            posterior["eic_mean"],
            posterior["temperature_mean"] + 2.0,
            dz,
        )
    thaw_settlement = _footprint_average(np.asarray(settlement, dtype=np.float32), x, y, design)
    total_settlement = (elastic_settlement + thaw_settlement).astype(np.float32)
    fs_deficit = np.maximum(float(design.target_factor_of_safety) - factor_of_safety, 0.0) / max(float(design.target_factor_of_safety), 1e-6)
    serviceability_ratio = total_settlement / max(float(design.allowable_total_settlement_m), 1e-6)
    if total_settlement.shape[0] > 1 and total_settlement.shape[1] > 1:
        gx, gy = np.gradient(total_settlement, x, y, edge_order=1)
        differential_settlement = np.hypot(gx, gy).astype(np.float32)
    else:
        differential_settlement = np.zeros_like(total_settlement, dtype=np.float32)
    foundation_risk = (
        0.35 * _risk_normalize_positive(fs_deficit)
        + 0.45 * _risk_normalize_positive(serviceability_ratio - 1.0)
        + 0.20 * _risk_normalize_positive(differential_settlement)
    ).astype(np.float32)
    return {
        "foundation_shear_strength_kpa": su.astype(np.float32),
        "foundation_modulus_mpa": modulus_mpa.astype(np.float32),
        "ultimate_bearing_capacity_kpa": ultimate_bearing.astype(np.float32),
        "bearing_factor_of_safety": factor_of_safety.astype(np.float32),
        "elastic_settlement_m": elastic_settlement.astype(np.float32),
        "thaw_settlement_m": thaw_settlement.astype(np.float32),
        "total_service_settlement_m": total_settlement.astype(np.float32),
        "serviceability_ratio": serviceability_ratio.astype(np.float32),
        "differential_total_settlement_gradient": differential_settlement.astype(np.float32),
        "foundation_risk_index": np.clip(foundation_risk, 0.0, 1.0).astype(np.float32),
    }


def _single_sample_response(
    posterior: dict[str, np.ndarray],
    design: FoundationDesign,
    geotech_params: GeotechnicalParameters,
    n_facies: int,
    sample_index: int | None = None,
) -> dict[str, np.ndarray]:
    if sample_index is None:
        geotech_fields = geotechnical_property_fields(posterior, n_facies=n_facies, params=geotech_params)
        settlement = np.asarray(posterior["settlement_potential"], dtype=np.float32) if "settlement_potential" in posterior else None
        return foundation_response_fields(geotech_fields, posterior, design=design, settlement=settlement)
    fields = {
        "eic_mean": np.asarray(posterior["eic_samples"][sample_index], dtype=np.float32),
        "temperature_mean": np.asarray(posterior["temperature_samples"][sample_index], dtype=np.float32),
        "unfrozen_water_mean": np.asarray(posterior["unfrozen_water_samples"][sample_index], dtype=np.float32),
        "grid_x": np.asarray(posterior.get("grid_x", np.arange(posterior["eic_samples"].shape[1])), dtype=np.float32),
        "grid_y": np.asarray(posterior.get("grid_y", np.arange(posterior["eic_samples"].shape[2])), dtype=np.float32),
        "grid_z": np.asarray(posterior.get("grid_z", np.arange(posterior["eic_samples"].shape[3])), dtype=np.float32),
    }
    if "facies_samples" in posterior:
        fields["facies"] = np.asarray(posterior["facies_samples"][sample_index], dtype=np.int16)
    elif "facies_mode" in posterior:
        fields["facies_mode"] = np.asarray(posterior["facies_mode"], dtype=np.int16)
    elif "facies" in posterior:
        fields["facies"] = np.asarray(posterior["facies"], dtype=np.int16)
    else:
        fields["facies_probability"] = np.asarray(posterior["facies_probability"], dtype=np.float32)
    geotech_fields = geotechnical_property_fields(fields, n_facies=n_facies, params=geotech_params)
    dz = _axis_spacing(fields["grid_z"])
    settlement = settlement_potential_numpy(fields["eic_mean"], fields["temperature_mean"] + geotech_params.future_warming_c, dz)
    return foundation_response_fields(geotech_fields, fields, design=design, settlement=settlement)


def foundation_reliability_summary(
    posterior: dict[str, np.ndarray],
    design: FoundationDesign | None = None,
    geotech_params: GeotechnicalParameters | None = None,
    n_facies: int = 7,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return deterministic response maps plus sample-based reliability maps when posterior samples exist."""
    design = design or FoundationDesign()
    geotech_params = geotech_params or GeotechnicalParameters()
    response = _single_sample_response(posterior, design, geotech_params, n_facies=n_facies)
    if all(key in posterior for key in ("eic_samples", "temperature_samples", "unfrozen_water_samples")):
        n_samples = int(np.asarray(posterior["eic_samples"]).shape[0])
        sample_responses = [_single_sample_response(posterior, design, geotech_params, n_facies=n_facies, sample_index=i) for i in range(n_samples)]
        fs_stack = np.stack([item["bearing_factor_of_safety"] for item in sample_responses], axis=0)
        settlement_stack = np.stack([item["total_service_settlement_m"] for item in sample_responses], axis=0)
    else:
        n_samples = 1
        fs_stack = response["bearing_factor_of_safety"][None, ...]
        settlement_stack = response["total_service_settlement_m"][None, ...]
    bearing_failure_probability = np.mean(fs_stack < float(design.target_factor_of_safety), axis=0).astype(np.float32)
    serviceability_exceedance_probability = np.mean(
        settlement_stack > float(design.allowable_total_settlement_m),
        axis=0,
    ).astype(np.float32)
    response.update(
        {
            "bearing_failure_probability": bearing_failure_probability,
            "serviceability_exceedance_probability": serviceability_exceedance_probability,
            "bearing_factor_of_safety_p05": np.percentile(fs_stack, 5.0, axis=0).astype(np.float32),
            "total_service_settlement_p95_m": np.percentile(settlement_stack, 95.0, axis=0).astype(np.float32),
        }
    )
    pf = float(np.nanmean(np.maximum(bearing_failure_probability, serviceability_exceedance_probability)))
    pf_clipped = min(max(pf, 1e-6), 1.0 - 1e-6)
    beta = -NormalDist().inv_cdf(pf_clipped)
    metrics = {
        "n_foundation_samples": float(n_samples),
        "design_pressure_kpa": float(design.net_pressure_kpa),
        "foundation_width_m": float(design.width_m),
        "foundation_length_m": float(design.length_m),
        "bearing_fs_p05": float(np.nanpercentile(response["bearing_factor_of_safety"], 5.0)),
        "bearing_fs_p50": float(np.nanpercentile(response["bearing_factor_of_safety"], 50.0)),
        "bearing_failure_area_fraction": float(np.nanmean(response["bearing_factor_of_safety"] < float(design.target_factor_of_safety))),
        "bearing_failure_probability_mean": float(np.nanmean(bearing_failure_probability)),
        "elastic_settlement_p95_m": float(np.nanpercentile(response["elastic_settlement_m"], 95.0)),
        "thaw_settlement_p95_m": float(np.nanpercentile(response["thaw_settlement_m"], 95.0)),
        "total_service_settlement_p95_m": float(np.nanpercentile(response["total_service_settlement_m"], 95.0)),
        "serviceability_exceedance_area_fraction": float(
            np.nanmean(response["total_service_settlement_m"] > float(design.allowable_total_settlement_m))
        ),
        "serviceability_exceedance_probability_mean": float(np.nanmean(serviceability_exceedance_probability)),
        "foundation_risk_p95": float(np.nanpercentile(response["foundation_risk_index"], 95.0)),
        "foundation_system_failure_probability_mean": pf,
        "foundation_reliability_index_beta": float(beta),
    }
    return response, metrics
