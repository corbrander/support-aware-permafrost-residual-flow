from __future__ import annotations

import numpy as np


def compute_unfrozen_water(temperature: np.ndarray, facies: np.ndarray) -> np.ndarray:
    theta_sat = np.array([0.38, 0.72, 0.42, 0.40, 0.25, 0.48, 0.08], dtype=np.float32)
    theta_res = np.array([0.06, 0.12, 0.07, 0.06, 0.03, 0.25, 0.01], dtype=np.float32)
    a = np.array([0.08, 0.18, 0.09, 0.08, 0.05, 0.20, 0.01], dtype=np.float32)
    b = np.array([0.42, 0.36, 0.45, 0.46, 0.50, 0.25, 0.50], dtype=np.float32)
    f = np.clip(facies, 0, len(theta_sat) - 1)
    frozen_curve = theta_res[f] + a[f] / np.power(np.maximum(np.abs(temperature), 0.08), b[f])
    theta = np.where(temperature >= 0.0, theta_sat[f], frozen_curve)
    return np.clip(theta, 0.0, theta_sat[f]).astype(np.float32)


def compute_resistivity(
    facies: np.ndarray,
    temperature: np.ndarray,
    unfrozen_water: np.ndarray,
    ice_content: np.ndarray,
    porosity: np.ndarray | None = None,
) -> np.ndarray:
    rho0 = np.array([120, 95, 180, 260, 420, 55, 900], dtype=np.float32)
    f = np.clip(facies, 0, len(rho0) - 1)
    if porosity is None:
        porosity = np.clip(0.35 + 0.25 * unfrozen_water - 0.12 * ice_content, 0.08, 0.85)
    rho = rho0[f] * np.exp(2.4 * ice_content) * np.exp(-2.8 * unfrozen_water) * np.exp(-0.10 * temperature)
    rho *= np.power(np.maximum(0.08, porosity), -0.65)
    return np.clip(rho, 8.0, 20000.0).astype(np.float32)

