from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.baselines.idw import _grid_points


@dataclass(frozen=True)
class KrigingConfig:
    length_scale_xyz: tuple[float, float, float] = (0.22, 0.22, 0.35)
    signal_variance: float = 1.0
    nugget: float = 0.03
    max_train_points: int = 512
    chunk_size: int = 32768
    random_state: int = 0


def _normalize_coords(coords: np.ndarray, grid: dict) -> np.ndarray:
    scale = np.asarray(
        [
            max(float(grid["x"][-1]), 1.0),
            max(float(grid["y"][-1]), 1.0),
            max(float(grid["z"][-1]), 1.0),
        ],
        dtype=np.float32,
    )
    return np.asarray(coords, dtype=np.float32) / scale[None, :]


def _subsample(
    coords: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    max_points: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) <= max_points:
        return coords, values, sigma
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(values), size=max_points, replace=False)
    return coords[indices], values[indices], sigma[indices]


def _fit_gpr(coords: np.ndarray, values: np.ndarray, sigma: np.ndarray, cfg: KrigingConfig) -> GaussianProcessRegressor:
    kernel = ConstantKernel(float(cfg.signal_variance), constant_value_bounds="fixed") * RBF(
        length_scale=np.asarray(cfg.length_scale_xyz, dtype=np.float32),
        length_scale_bounds="fixed",
    )
    alpha = np.maximum(np.asarray(sigma, dtype=np.float64) ** 2, float(cfg.nugget) ** 2)
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=None, normalize_y=True)
    model.fit(coords, values.astype(np.float64))
    return model


def _predict_chunks(model: GaussianProcessRegressor, query: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for start in range(0, len(query), chunk_size):
        mean, std = model.predict(query[start : start + chunk_size], return_std=True)
        means.append(mean.astype(np.float32))
        stds.append(std.astype(np.float32))
    return np.concatenate(means, axis=0), np.concatenate(stds, axis=0)


def _continuous_kriging(
    obs: ObservationTable,
    grid: dict,
    type_name: str,
    cfg: KrigingConfig,
    query_norm: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    mask = obs.mask & (obs.type_ids == OBS_TYPES[type_name])
    if int(np.sum(mask)) < 3:
        return None
    coords = _normalize_coords(obs.coords[mask], grid)
    values = obs.values[mask].astype(np.float32)
    sigma = obs.sigma[mask].astype(np.float32)
    coords, values, sigma = _subsample(coords, values, sigma, cfg.max_train_points, cfg.random_state)
    model = _fit_gpr(coords, values, sigma, cfg)
    mean, std = _predict_chunks(model, query_norm, cfg.chunk_size)
    return mean.reshape(shape).astype(np.float32), std.reshape(shape).astype(np.float32)


def _indicator_kriging(
    obs: ObservationTable,
    grid: dict,
    n_facies: int,
    cfg: KrigingConfig,
    query_norm: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    mask = obs.mask & (obs.type_ids == OBS_TYPES["borehole_facies"])
    if int(np.sum(mask)) < 3:
        return None
    coords = _normalize_coords(obs.coords[mask], grid)
    labels = np.clip(obs.values[mask].astype(np.int64), 0, n_facies - 1)
    sigma = np.full_like(labels, float(cfg.nugget), dtype=np.float32)
    coords, labels, sigma = _subsample(coords, labels.astype(np.float32), sigma, cfg.max_train_points, cfg.random_state)
    probs = np.zeros((len(query_norm), n_facies), dtype=np.float32)
    for cls in range(n_facies):
        indicator = (labels.astype(np.int64) == cls).astype(np.float32)
        if float(np.max(indicator)) == 0.0:
            continue
        model = _fit_gpr(coords, indicator, sigma, cfg)
        mean, _ = _predict_chunks(model, query_norm, cfg.chunk_size)
        probs[:, cls] = np.clip(mean, 0.0, 1.0)
    denom = np.sum(probs, axis=1, keepdims=True)
    missing = denom[:, 0] <= 1e-6
    if np.any(missing):
        counts = np.bincount(labels.astype(np.int64), minlength=n_facies).astype(np.float32)
        prior = counts / max(float(np.sum(counts)), 1.0)
        probs[missing] = prior[None, :]
        denom = np.sum(probs, axis=1, keepdims=True)
    probs = probs / np.maximum(denom, 1e-6)
    facies = np.argmax(probs, axis=1).reshape(shape).astype(np.int16)
    return facies, probs.reshape(*shape, n_facies).astype(np.float32)


def reconstruct_kriging(
    observations: ObservationTable,
    grid: dict,
    n_facies: int = 7,
    config: KrigingConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or KrigingConfig()
    query = _grid_points(grid)
    query_norm = _normalize_coords(query, grid)
    shape = (len(grid["x"]), len(grid["y"]), len(grid["z"]))
    out: dict[str, np.ndarray] = {}
    for type_name, field_name in [
        ("borehole_eic", "eic"),
        ("borehole_temperature", "temperature"),
        ("nmr_unfrozen_water", "unfrozen_water"),
        ("ert_log_resistivity", "log_resistivity"),
    ]:
        result = _continuous_kriging(observations, grid, type_name, cfg, query_norm, shape)
        if result is not None:
            mean, std = result
            out[field_name] = mean
            out[f"{field_name}_std"] = std
    facies_result = _indicator_kriging(observations, grid, n_facies, cfg, query_norm, shape)
    if facies_result is not None:
        facies, probs = facies_result
        out["facies"] = facies
        out["facies_probability"] = probs
        out["facies_logits"] = probs
    return out


class KrigingBaseline:
    """Fixed-kernel Gaussian-process baseline equivalent to simple ordinary and indicator kriging."""

    def __init__(self, config: KrigingConfig | None = None, n_facies: int = 7) -> None:
        self.config = config or KrigingConfig()
        self.n_facies = int(n_facies)

    def reconstruct(self, observations: ObservationTable, grid: dict) -> dict[str, np.ndarray]:
        return reconstruct_kriging(observations, grid, n_facies=self.n_facies, config=self.config)
