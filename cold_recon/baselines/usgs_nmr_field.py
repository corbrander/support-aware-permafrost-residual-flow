from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable


FEATURE_VERSION = "usgs-btl-ert-alt-doy-ridge-v1"
FEATURE_NAMES = (
    "depth_m",
    "depth_squared_m2",
    "ert_ln_resistivity_idw",
    "alt_depth_m_idw",
    "depth_below_alt_m",
    "depth_above_alt_m",
    "day_of_year_sin",
    "day_of_year_cos",
    "depth_x_day_of_year_sin",
    "depth_x_day_of_year_cos",
    "log1p_nearest_ert_horizontal_m",
    "log1p_nearest_alt_horizontal_m",
)
SECONDS_PER_DAY = 86_400.0
UNIX_EPOCH_ORDINAL = 719163


def _day_of_year(days_since_1970: np.ndarray) -> np.ndarray:
    """Convert finite UTC days since 1970-01-01 to Gregorian day of year."""

    days = np.asarray(days_since_1970, dtype=np.float64)
    if days.ndim != 1 or not np.all(np.isfinite(days)):
        raise ValueError(
            "the seasonal field baseline requires one finite query time per support; "
            "time is geometry/provenance, not a target value"
        )
    # NumPy datetime arithmetic is deterministic and avoids locale parsing.
    dates = np.datetime64("1970-01-01", "D") + np.floor(days).astype("timedelta64[D]")
    years = dates.astype("datetime64[Y]")
    return (dates - years).astype("timedelta64[D]").astype(np.float64) + 1.0


def _active_type(observations: ObservationTable, type_id: int) -> np.ndarray:
    return (
        np.asarray(observations.mask, dtype=bool)
        & (np.asarray(observations.type_ids, dtype=np.int64) == int(type_id))
        & np.isfinite(observations.values)
        & np.all(np.isfinite(observations.coords), axis=1)
    )


def _idw_neighbours(
    source_coords: np.ndarray,
    source_values: np.ndarray,
    query_coords: np.ndarray,
    *,
    scale: tuple[float, float, float],
    neighbours: int,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_coords, dtype=np.float64)
    query = np.asarray(query_coords, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    if len(source) == 0:
        raise ValueError("cannot interpolate from an empty conditioning modality")
    k = min(max(1, int(neighbours)), len(source))
    scale_array = np.asarray(scale, dtype=np.float64)
    tree = cKDTree(source / scale_array[None, :])
    distance, indices = tree.query(query / scale_array[None, :], k=k)
    if k == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.square(np.asarray(distance, dtype=np.float64) + float(softening))
    interpolated = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
    selected = source[indices]
    horizontal_distance = np.sqrt(
        np.sum(np.square(query[:, None, :2] - selected[:, :, :2]), axis=2)
    ).min(axis=1)
    return interpolated, horizontal_distance


def field_features(
    conditioning: ObservationTable,
    query_coords: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Build target-independent ERT, ALT, depth, and seasonal features.

    ``conditioning`` must contain ERT and ALT only in strict prediction use.
    The function never inspects an NMR target value.  Development code may
    pass an ERT+ALT subset of a public development table.
    """

    query = np.asarray(query_coords, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3 or not np.all(np.isfinite(query)):
        raise ValueError("query_coords must be a finite [n, 3] array")
    doy = _day_of_year(np.asarray(query_times, dtype=np.float64))
    if len(doy) != len(query):
        raise ValueError("query_times must align one-to-one with query_coords")

    ert_selected = _active_type(conditioning, OBS_TYPES["ert_log_resistivity"])
    alt_selected = _active_type(conditioning, OBS_TYPES["alt"])
    if not np.any(ert_selected) or not np.any(alt_selected):
        raise ValueError("field baseline conditioning requires active ERT and ALT rows")

    ert, ert_horizontal_distance = _idw_neighbours(
        conditioning.coords[ert_selected],
        conditioning.values[ert_selected],
        query,
        scale=(5.0, 5.0, 0.25),
        neighbours=32,
        softening=0.25,
    )
    # ALT has no meaningful vertical coordinate for the interpolation itself.
    alt_source_coords = np.asarray(conditioning.coords[alt_selected], dtype=np.float64).copy()
    alt_source_coords[:, 2] = 0.0
    alt_query_coords = query.copy()
    alt_query_coords[:, 2] = 0.0
    alt, alt_horizontal_distance = _idw_neighbours(
        alt_source_coords,
        conditioning.values[alt_selected],
        alt_query_coords,
        scale=(5.0, 5.0, 1.0),
        neighbours=8,
        softening=0.25,
    )

    depth = query[:, 2]
    phase = 2.0 * np.pi * (doy - 172.0) / 365.2425
    seasonal_sin = np.sin(phase)
    seasonal_cos = np.cos(phase)
    features = np.column_stack(
        [
            depth,
            np.square(depth),
            ert,
            alt,
            np.maximum(depth - alt, 0.0),
            np.maximum(alt - depth, 0.0),
            seasonal_sin,
            seasonal_cos,
            depth * seasonal_sin,
            depth * seasonal_cos,
            np.log1p(ert_horizontal_distance),
            np.log1p(alt_horizontal_distance),
        ]
    )
    if features.shape != (len(query), len(FEATURE_NAMES)) or not np.all(
        np.isfinite(features)
    ):
        raise ValueError("field baseline produced non-finite or malformed features")
    return features.astype(np.float64)


def _logit(values: np.ndarray, epsilon: float) -> np.ndarray:
    bounded = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(bounded / (1.0 - bounded))


def _expit(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    # Stable form avoids overflow on deliberately extreme out-of-domain input.
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


@dataclass(frozen=True)
class FrozenUSGSNMRBaseline:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float
    logit_epsilon: float
    metadata: dict[str, Any]

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape [n, {len(FEATURE_NAMES)}] for {FEATURE_VERSION}"
            )
        standardized = (matrix - self.feature_mean[None, :]) / self.feature_scale[None, :]
        return _expit(standardized @ self.coefficients + float(self.intercept)).astype(
            np.float32
        )

    def predict(
        self,
        conditioning: ObservationTable,
        query_coords: np.ndarray,
        query_times: np.ndarray,
    ) -> np.ndarray:
        return self.predict_features(field_features(conditioning, query_coords, query_times))


def fit_ridge_baseline(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 10.0,
    logit_epsilon: float = 0.005,
    metadata: dict[str, Any] | None = None,
) -> FrozenUSGSNMRBaseline:
    matrix = np.asarray(features, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("development features have the wrong feature contract")
    if values.shape != (len(matrix),) or not np.all(np.isfinite(values)):
        raise ValueError("development targets must be a finite aligned vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("development NMR water fractions must lie in [0, 1]")
    if len(values) <= len(FEATURE_NAMES):
        raise ValueError("insufficient development rows for the fixed ridge feature set")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("ridge alpha must be finite and positive")

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1.0e-10, scale, 1.0)
    standardized = (matrix - mean[None, :]) / scale[None, :]
    transformed = _logit(values, float(logit_epsilon))
    intercept = float(transformed.mean())
    centered = transformed - intercept
    gram = standardized.T @ standardized + float(alpha) * np.eye(standardized.shape[1])
    coefficients = np.linalg.solve(gram, standardized.T @ centered)
    return FrozenUSGSNMRBaseline(
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        intercept=intercept,
        alpha=float(alpha),
        logit_epsilon=float(logit_epsilon),
        metadata=dict(metadata or {}),
    )


def save_frozen_baseline(path: str | Path, model: FrozenUSGSNMRBaseline) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        feature_version=np.asarray(FEATURE_VERSION),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U"),
        feature_mean=np.asarray(model.feature_mean, dtype=np.float64),
        feature_scale=np.asarray(model.feature_scale, dtype=np.float64),
        coefficients=np.asarray(model.coefficients, dtype=np.float64),
        intercept=np.asarray(model.intercept, dtype=np.float64),
        alpha=np.asarray(model.alpha, dtype=np.float64),
        logit_epsilon=np.asarray(model.logit_epsilon, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(model.metadata, ensure_ascii=False, sort_keys=True)),
    )


def load_frozen_baseline(path: str | Path) -> FrozenUSGSNMRBaseline:
    allowed = {
        "feature_version",
        "feature_names",
        "feature_mean",
        "feature_scale",
        "coefficients",
        "intercept",
        "alpha",
        "logit_epsilon",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as saved:
        unexpected = sorted(set(saved.files).difference(allowed))
        missing = sorted(allowed.difference(saved.files))
        if unexpected or missing:
            raise ValueError(
                f"invalid frozen baseline keys; missing={missing}, unexpected={unexpected}"
            )
        if str(saved["feature_version"].item()) != FEATURE_VERSION:
            raise ValueError("frozen field-baseline feature version mismatch")
        names = tuple(np.asarray(saved["feature_names"]).astype(str).tolist())
        if names != FEATURE_NAMES:
            raise ValueError("frozen field-baseline feature names mismatch")
        mean = np.asarray(saved["feature_mean"], dtype=np.float64)
        scale = np.asarray(saved["feature_scale"], dtype=np.float64)
        coefficients = np.asarray(saved["coefficients"], dtype=np.float64)
        metadata = json.loads(str(saved["metadata_json"].item()))
        intercept = float(saved["intercept"].item())
        alpha = float(saved["alpha"].item())
        epsilon = float(saved["logit_epsilon"].item())
    expected_shape = (len(FEATURE_NAMES),)
    if mean.shape != expected_shape or scale.shape != expected_shape or coefficients.shape != expected_shape:
        raise ValueError("frozen field-baseline coefficient arrays have invalid shapes")
    if not np.all(np.isfinite(np.r_[mean, scale, coefficients, intercept, alpha, epsilon])):
        raise ValueError("frozen field-baseline model contains non-finite parameters")
    if np.any(scale <= 0.0) or alpha <= 0.0 or not (0.0 < epsilon < 0.5):
        raise ValueError("frozen field-baseline parameter bounds are invalid")
    return FrozenUSGSNMRBaseline(
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        intercept=intercept,
        alpha=alpha,
        logit_epsilon=epsilon,
        metadata=metadata,
    )
