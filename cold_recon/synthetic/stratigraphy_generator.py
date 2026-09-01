from __future__ import annotations

import numpy as np

from cold_recon.data.data_schema import SURFACE_FEATURE_NAMES


def generate_surface_features(grid: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    x = grid["x"]
    y = grid["y"]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    lx = max(float(x[-1] - x[0]), 1.0)
    ly = max(float(y[-1] - y[0]), 1.0)
    dem = (
        2.0 * np.sin(2 * np.pi * xx / lx)
        + 1.3 * np.cos(2 * np.pi * yy / ly)
        + 0.8 * np.sin(2 * np.pi * (xx + 0.35 * yy) / (0.75 * lx))
    )
    dem += rng.normal(0.0, 0.05, size=dem.shape)
    gx, gy = np.gradient(dem, float(grid["dx"]), float(grid["dy"]), edge_order=1)
    slope = np.hypot(gx, gy)
    curvature = np.gradient(gx, float(grid["dx"]), axis=0) + np.gradient(gy, float(grid["dy"]), axis=1)
    lake = np.exp(-(((xx - 0.62 * lx) / (0.18 * lx)) ** 2 + ((yy - 0.42 * ly) / (0.14 * ly)) ** 2))
    ndvi = np.clip(0.55 + 0.18 * np.sin(2 * np.pi * yy / ly) - 0.35 * lake + rng.normal(0, 0.02, dem.shape), 0, 1)
    soil_clay = np.clip(0.28 + 0.15 * np.cos(2 * np.pi * xx / lx) + 0.18 * lake, 0.05, 0.8)
    soil_sand = np.clip(0.55 - 0.35 * soil_clay + 0.08 * np.sin(2 * np.pi * yy / ly), 0.05, 0.9)
    air_temp_ma = -7.2 + 0.25 * dem + 2.2 * lake
    snow_proxy = np.clip(0.35 + 0.45 * lake + 0.12 * (1.0 - ndvi), 0.0, 1.0)
    features = {
        "dem": dem,
        "slope": slope,
        "curvature": curvature,
        "ndvi": ndvi,
        "soil_sand": soil_sand,
        "soil_clay": soil_clay,
        "air_temp_ma": air_temp_ma,
        "snow_proxy": snow_proxy,
    }
    return {k: features[k].astype(np.float32) for k in SURFACE_FEATURE_NAMES}


def generate_stratigraphy(
    grid: dict,
    surface_features: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = grid["x"]
    y = grid["y"]
    z = grid["z"]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    lx = max(float(x[-1] - x[0]), 1.0)
    ly = max(float(y[-1] - y[0]), 1.0)
    dem = surface_features["dem"]
    lake = np.clip((surface_features["snow_proxy"] - 0.45) / 0.55, 0, 1)

    active_depth = 0.55 + 0.18 * np.sin(2 * np.pi * x[:, None] / lx) + 0.18 * lake
    peat_bottom = 0.18 + 0.15 * surface_features["ndvi"] + 0.05 * np.cos(2 * np.pi * y[None, :] / ly)
    silt_bottom = 2.2 + 0.35 * np.sin(2 * np.pi * x[:, None] / lx + 0.8) + 0.22 * np.cos(2 * np.pi * y[None, :] / ly)
    ice_top = 2.6 + 0.35 * np.sin(2 * np.pi * (x[:, None] + 0.3 * y[None, :]) / lx)
    ice_bottom = ice_top + 0.85 + 0.2 * np.cos(2 * np.pi * y[None, :] / ly)
    sand_top = 4.5 + 0.45 * np.sin(2 * np.pi * (x[:, None] - y[None, :]) / (1.25 * lx))

    facies = np.full((len(x), len(y), len(z)), 4, dtype=np.int16)
    facies[zz < silt_bottom[:, :, None]] = 2
    facies[zz < active_depth[:, :, None]] = 0
    facies[zz < peat_bottom[:, :, None]] = 1
    facies[(zz >= ice_top[:, :, None]) & (zz < ice_bottom[:, :, None])] = 3
    facies[zz > sand_top[:, :, None]] = 4

    lake_mask_2d = lake > 0.62
    talik_depth = 3.8 + 2.3 * lake
    talik_mask = lake_mask_2d[:, :, None] & (zz < talik_depth[:, :, None]) & (zz > 0.4)
    facies[talik_mask] = 5

    n_wedges = 5
    for _ in range(n_wedges):
        x0 = rng.uniform(0.1 * lx, 0.9 * lx)
        angle = rng.uniform(-0.7, 0.7)
        line_distance = np.abs((xx[:, :, 0] - x0) - angle * (yy[:, :, 0] - 0.5 * ly))
        max_depth = rng.uniform(2.0, 4.0)
        width = np.maximum(0.12, 0.75 * (1.0 - zz / max_depth))
        wedge_mask = (zz < max_depth) & (line_distance[:, :, None] < width)
        wedge_mask &= ~talik_mask
        facies[wedge_mask] = 6

    interfaces = {
        "active_depth": active_depth.astype(np.float32),
        "peat_bottom": peat_bottom.astype(np.float32),
        "silt_bottom": silt_bottom.astype(np.float32),
        "ice_top": ice_top.astype(np.float32),
        "ice_bottom": ice_bottom.astype(np.float32),
        "sand_top": sand_top.astype(np.float32),
        "lake_proxy": lake.astype(np.float32),
        "dem": dem.astype(np.float32),
    }
    return facies, interfaces

