from __future__ import annotations

import numpy as np

from cold_recon.evaluation.uncertainty import facies_entropy
from cold_recon.physics.settlement import settlement_potential_numpy


def facies_to_probability(facies: np.ndarray, n_facies: int = 7) -> np.ndarray:
    labels = np.asarray(facies, dtype=np.int64)
    probs = np.zeros((*labels.shape, n_facies), dtype=np.float32)
    for cls in range(n_facies):
        probs[..., cls] = labels == cls
    return probs


def empirical_unfrozen_water_np(
    temperature: np.ndarray,
    facies_probs: np.ndarray | None = None,
    theta_sat: float = 0.42,
    theta_res: float = 0.06,
    a: float = 0.09,
    b: float = 0.45,
) -> np.ndarray:
    temp = np.asarray(temperature, dtype=np.float32)
    frozen = theta_res + a / np.power(np.clip(np.abs(temp), 0.08, None), b)
    theta = np.where(temp >= 0.0, theta_sat, frozen)
    if facies_probs is not None:
        ice_prob = facies_probs[..., 6] if facies_probs.shape[-1] > 6 else 0.0
        peat_prob = facies_probs[..., 1] if facies_probs.shape[-1] > 1 else 0.0
        theta = theta * (1.0 - 0.65 * ice_prob) + 0.22 * peat_prob
    return np.clip(theta, 0.0, 0.8).astype(np.float32)


def empirical_log_resistivity_np(
    eic: np.ndarray,
    temperature: np.ndarray,
    unfrozen_water: np.ndarray,
    facies_probs: np.ndarray | None = None,
) -> np.ndarray:
    log_rho = 5.0 + 2.4 * np.asarray(eic, dtype=np.float32) - 2.8 * np.asarray(unfrozen_water, dtype=np.float32) - 0.10 * np.asarray(temperature, dtype=np.float32)
    if facies_probs is not None:
        ice_prob = facies_probs[..., 6] if facies_probs.shape[-1] > 6 else 0.0
        talik_prob = facies_probs[..., 5] if facies_probs.shape[-1] > 5 else 0.0
        sand_prob = facies_probs[..., 4] if facies_probs.shape[-1] > 4 else 0.0
        log_rho = log_rho + 1.0 * ice_prob - 0.9 * talik_prob + 0.35 * sand_prob
    return log_rho.astype(np.float32)


def estimate_thermal_conductivity_np(eic: np.ndarray, facies_probs: np.ndarray | None = None) -> np.ndarray:
    k = 1.15 + 1.65 * np.asarray(eic, dtype=np.float32)
    if facies_probs is not None:
        peat_prob = facies_probs[..., 1] if facies_probs.shape[-1] > 1 else 0.0
        sand_prob = facies_probs[..., 4] if facies_probs.shape[-1] > 4 else 0.0
        ice_prob = facies_probs[..., 6] if facies_probs.shape[-1] > 6 else 0.0
        k = k - 0.35 * peat_prob + 0.45 * sand_prob + 0.70 * ice_prob
    return np.clip(k, 0.35, 4.5).astype(np.float32)


def heat_residual_np(temperature: np.ndarray, conductivity: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    temp = np.asarray(temperature, dtype=np.float32)
    k = np.asarray(conductivity, dtype=np.float32)
    dx, dy, dz = spacing
    tx, ty, tz = np.gradient(temp, float(dx), float(dy), float(dz), edge_order=1)
    qx = k * tx
    qy = k * ty
    qz = k * tz
    div_x = np.gradient(qx, float(dx), axis=0, edge_order=1)
    div_y = np.gradient(qy, float(dy), axis=1, edge_order=1)
    div_z = np.gradient(qz, float(dz), axis=2, edge_order=1)
    return (div_x + div_y + div_z).astype(np.float32)


def stratigraphic_total_variation(facies_probs: np.ndarray) -> tuple[float, float]:
    probs = np.asarray(facies_probs, dtype=np.float32)
    if probs.ndim != 4:
        raise ValueError("facies_probs must have shape [x, y, z, n_facies]")
    tv_xy = 0.0
    n_xy = 0
    if probs.shape[0] > 1:
        tv_xy += float(np.mean(np.abs(np.diff(probs, axis=0))))
        n_xy += 1
    if probs.shape[1] > 1:
        tv_xy += float(np.mean(np.abs(np.diff(probs, axis=1))))
        n_xy += 1
    tv_z = float(np.mean(np.abs(np.diff(probs, axis=2)))) if probs.shape[2] > 1 else 0.0
    return (tv_xy / max(n_xy, 1), tv_z)


def fields_from_prediction(data: dict[str, np.ndarray], n_facies: int = 7) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    aliases = {
        "eic": ("eic_mean", "eic"),
        "temperature": ("temperature_mean", "temperature"),
        "unfrozen_water": ("unfrozen_water_mean", "unfrozen_water"),
        "log_resistivity": ("log_resistivity_mean", "log_resistivity"),
    }
    for name, keys in aliases.items():
        for key in keys:
            if key in data:
                out[name] = np.asarray(data[key], dtype=np.float32)
                break
    if "facies_probability" in data:
        out["facies_probability"] = np.asarray(data["facies_probability"], dtype=np.float32)
    elif "facies_mode" in data:
        out["facies_probability"] = facies_to_probability(data["facies_mode"], n_facies=n_facies)
    elif "facies" in data:
        out["facies_probability"] = facies_to_probability(data["facies"], n_facies=n_facies)
    if "facies_mode" in data:
        out["facies"] = np.asarray(data["facies_mode"], dtype=np.int16)
    elif "facies" in data:
        out["facies"] = np.asarray(data["facies"], dtype=np.int16)
    if "thermal_conductivity" in data:
        out["thermal_conductivity"] = np.asarray(data["thermal_conductivity"], dtype=np.float32)
    elif "field_thermal_conductivity" in data:
        out["thermal_conductivity"] = np.asarray(data["field_thermal_conductivity"], dtype=np.float32)
    if "settlement_potential" in data:
        out["settlement_potential"] = np.asarray(data["settlement_potential"], dtype=np.float32)
    return out


def sample_truth_fields(sample: dict, n_facies: int = 7) -> dict[str, np.ndarray]:
    fields = sample["fields"]
    out = {
        "eic": np.asarray(fields["eic"], dtype=np.float32),
        "temperature": np.asarray(fields["temperature"], dtype=np.float32),
        "unfrozen_water": np.asarray(fields["unfrozen_water"], dtype=np.float32),
        "log_resistivity": np.log(np.maximum(np.asarray(fields["resistivity"], dtype=np.float32), 1.0)),
        "facies": np.asarray(fields["facies"], dtype=np.int16),
        "facies_probability": facies_to_probability(fields["facies"], n_facies=n_facies),
    }
    if "thermal_conductivity" in fields:
        out["thermal_conductivity"] = np.asarray(fields["thermal_conductivity"], dtype=np.float32)
    return out


def physics_consistency_metrics(
    fields: dict[str, np.ndarray],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    future_warming_c: float = 2.0,
) -> dict[str, float]:
    required = ["eic", "temperature", "unfrozen_water", "log_resistivity", "facies_probability"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"Missing fields for physics consistency: {missing}")
    eic = fields["eic"]
    temperature = fields["temperature"]
    unfrozen = fields["unfrozen_water"]
    log_rho = fields["log_resistivity"]
    facies_probs = fields["facies_probability"]
    theta_emp = empirical_unfrozen_water_np(temperature, facies_probs)
    rho_emp = empirical_log_resistivity_np(eic, temperature, unfrozen, facies_probs)
    conductivity = fields.get("thermal_conductivity", estimate_thermal_conductivity_np(eic, facies_probs))
    residual = heat_residual_np(temperature, conductivity, spacing)
    tv_xy, tv_z = stratigraphic_total_variation(facies_probs)
    settlement = fields.get(
        "settlement_potential",
        settlement_potential_numpy(eic, temperature + float(future_warming_c), float(spacing[2])),
    )
    return {
        "unfrozen_water_empirical_mae": float(np.mean(np.abs(unfrozen - theta_emp))),
        "unfrozen_water_empirical_rmse": float(np.sqrt(np.mean((unfrozen - theta_emp) ** 2))),
        "log_resistivity_empirical_mae": float(np.mean(np.abs(log_rho - rho_emp))),
        "log_resistivity_empirical_rmse": float(np.sqrt(np.mean((log_rho - rho_emp) ** 2))),
        "heat_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "heat_residual_abs_mean": float(np.mean(np.abs(residual))),
        "stratigraphic_tv_xy": float(tv_xy),
        "stratigraphic_tv_z": float(tv_z),
        "facies_entropy_mean": float(np.mean(facies_entropy(facies_probs))),
        "eic_out_of_range_fraction": float(np.mean((eic < 0.0) | (eic > 1.0))),
        "unfrozen_water_out_of_range_fraction": float(np.mean((unfrozen < 0.0) | (unfrozen > 0.8))),
        "settlement_potential_mean": float(np.mean(settlement)),
        "settlement_potential_p95": float(np.percentile(settlement, 95.0)),
    }
