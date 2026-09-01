from __future__ import annotations

import numpy as np


def compute_temperature(
    facies: np.ndarray,
    grid: dict,
    surface_features: dict[str, np.ndarray],
    interfaces: dict[str, np.ndarray],
) -> np.ndarray:
    z = grid["z"]
    zz = np.broadcast_to(z[None, None, :], facies.shape)
    air = surface_features["air_temp_ma"]
    snow = surface_features["snow_proxy"]
    lake = interfaces["lake_proxy"]
    active = interfaces["active_depth"]
    geothermal_gradient = 0.035
    surface_temp = air + 2.0 * snow + 1.5 * lake
    temp = surface_temp[:, :, None] + geothermal_gradient * zz
    seasonal_active = 2.7 * np.maximum(0.0, 1.0 - zz / np.maximum(active[:, :, None], 0.1))
    temp += seasonal_active
    talik_warm = 3.2 * lake[:, :, None] * np.exp(-zz / 2.8)
    temp += talik_warm
    temp = np.where(facies == 5, np.maximum(temp, 0.4 + 0.18 * zz), temp)
    temp = np.where(facies == 6, temp - 0.8, temp)
    return temp.astype(np.float32)


def compute_thermal_properties(
    facies: np.ndarray,
    ice_content: np.ndarray,
    unfrozen_water: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k_base = np.array([1.2, 0.45, 1.55, 1.85, 2.1, 0.8, 2.2], dtype=np.float32)
    c_base = np.array([2.2, 2.8, 2.1, 2.0, 1.9, 3.5, 1.8], dtype=np.float32)
    k = k_base[np.clip(facies, 0, len(k_base) - 1)] + 0.65 * ice_content - 0.25 * unfrozen_water
    heat_capacity = c_base[np.clip(facies, 0, len(c_base) - 1)] + 1.2 * unfrozen_water + 0.35 * ice_content
    return k.astype(np.float32), heat_capacity.astype(np.float32)

