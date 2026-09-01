from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable, SURFACE_FEATURE_NAMES


def _grid_points_and_surface(sample: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    grid = sample["grid"]
    xx, yy, zz = np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    nx, ny, nz = len(grid["x"]), len(grid["y"]), len(grid["z"])
    surf = np.stack([sample["surface_features"][name] for name in SURFACE_FEATURE_NAMES], axis=-1)
    surf3 = np.repeat(surf[:, :, None, :], nz, axis=2).reshape(-1, len(SURFACE_FEATURE_NAMES))
    xmax = max(float(grid["x"][-1]), 1.0)
    ymax = max(float(grid["y"][-1]), 1.0)
    zmax = max(float(grid["z"][-1]), 1.0)
    features = np.concatenate([coords / np.array([xmax, ymax, zmax], dtype=np.float32), surf3], axis=1)
    return features.astype(np.float32), coords, (nx, ny, nz)


def _surface_at_obs(sample: dict, obs_coords: np.ndarray) -> np.ndarray:
    grid = sample["grid"]
    ix = np.clip(np.round(obs_coords[:, 0] / float(grid["dx"])).astype(int), 0, len(grid["x"]) - 1)
    iy = np.clip(np.round(obs_coords[:, 1] / float(grid["dy"])).astype(int), 0, len(grid["y"]) - 1)
    surf = np.stack([sample["surface_features"][name][ix, iy] for name in SURFACE_FEATURE_NAMES], axis=1)
    xyz_max = np.array([max(grid["x"][-1], 1.0), max(grid["y"][-1], 1.0), max(grid["z"][-1], 1.0)], dtype=np.float32)
    return np.concatenate([obs_coords / xyz_max[None, :], surf], axis=1).astype(np.float32)


def reconstruct_random_forest(
    sample: dict,
    n_estimators: int = 80,
    random_state: int = 0,
    n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    obs: ObservationTable = sample["observations"]
    query_features, _, shape = _grid_points_and_surface(sample)
    out: dict[str, np.ndarray] = {}

    facies_mask = obs.mask & (obs.type_ids == OBS_TYPES["borehole_facies"])
    if np.sum(facies_mask) >= 2:
        x_train = _surface_at_obs(sample, obs.coords[facies_mask])
        y_train = obs.values[facies_mask].astype(np.int64)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            n_jobs=int(n_jobs),
            random_state=random_state,
        )
        clf.fit(x_train, y_train)
        out["facies"] = clf.predict(query_features).reshape(shape).astype(np.int16)

    for type_name, field_name in [
        ("borehole_eic", "eic"),
        ("borehole_temperature", "temperature"),
        ("nmr_unfrozen_water", "unfrozen_water"),
        ("ert_log_resistivity", "log_resistivity"),
    ]:
        mask = obs.mask & (obs.type_ids == OBS_TYPES[type_name])
        if np.sum(mask) >= 4:
            x_train = _surface_at_obs(sample, obs.coords[mask])
            y_train = obs.values[mask].astype(np.float32)
            reg = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                n_jobs=int(n_jobs),
                random_state=random_state,
            )
            reg.fit(x_train, y_train)
            out[field_name] = reg.predict(query_features).reshape(shape).astype(np.float32)
    return out
