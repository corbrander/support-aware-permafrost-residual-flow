from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable


def observation_ood_features(observations: ObservationTable, grid: dict) -> np.ndarray:
    valid = np.asarray(observations.mask, dtype=bool)
    features: list[float] = []
    for type_id in range(max(OBS_TYPES.values()) + 1):
        selected = valid & (observations.type_ids == type_id)
        features.append(float(np.log1p(selected.sum())))
        features.append(float(np.nanmedian(observations.sigma[selected])) if np.any(selected) else 0.0)
        if np.any(selected):
            values = np.asarray(observations.values[selected], dtype=np.float64)
            lower, upper = np.nanpercentile(values, [25.0, 75.0])
            features.extend(
                [float(np.nanmedian(values)), float(upper - lower)]
            )
        else:
            features.extend([0.0, 0.0])
    if np.any(valid):
        extent = observations.support_extent[valid]
        features.extend(np.nanmean(extent, axis=0).astype(float).tolist())
        quality = observations.quality[valid]
        features.extend([float(np.mean(quality)), float(np.std(quality))])
        coords = observations.coords[valid]
        domain = np.asarray(
            [
                max(float(grid["x"][-1]) - float(grid["x"][0]), 1.0e-6),
                max(float(grid["y"][-1]) - float(grid["y"][0]), 1.0e-6),
                max(float(grid["z"][-1]) - float(grid["z"][0]), 1.0e-6),
            ]
        )
        features.extend((np.ptp(coords, axis=0) / domain).astype(float).tolist())
    else:
        features.extend([0.0] * 8)
    return np.asarray(features, dtype=np.float64)


def prior_context_ood_features(prior: dict[str, np.ndarray]) -> np.ndarray:
    """Summarize deployable prior geometry without access to volume truth."""

    features: list[float] = []
    for name in ("eic", "temperature", "unfrozen_water", "log_resistivity"):
        values = np.asarray(prior[name], dtype=np.float64)
        quantiles = np.nanpercentile(values, [5.0, 25.0, 50.0, 75.0, 95.0])
        features.extend(quantiles.astype(float).tolist())
        gradients = np.gradient(values)
        horizontal = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
        vertical = np.abs(gradients[2])
        features.extend(
            [
                float(np.nanmean(horizontal)),
                float(np.nanpercentile(horizontal, 95.0)),
                float(np.nanmean(vertical)),
                float(np.nanpercentile(vertical, 95.0)),
            ]
        )
    eic = np.asarray(prior["eic"], dtype=np.float64)
    features.extend([float(np.mean(eic >= threshold)) for threshold in (0.20, 0.30, 0.40)])
    facies = np.asarray(prior["facies"], dtype=np.int64)
    for axis in range(3):
        features.append(float(np.mean(np.diff(facies, axis=axis) != 0)))
    counts = np.bincount(facies.reshape(-1), minlength=7).astype(np.float64)
    features.extend((counts / max(float(counts.sum()), 1.0)).tolist())
    return np.asarray(features, dtype=np.float64)


def scene_ood_features(
    observations: ObservationTable,
    grid: dict,
    prior: dict[str, np.ndarray],
) -> np.ndarray:
    """Combine acquisition-pattern and conventional-prior context features."""

    return np.concatenate(
        [
            observation_ood_features(observations, grid),
            prior_context_ood_features(prior),
        ]
    )


@dataclass
class MahalanobisOODController:
    mean: np.ndarray | None = None
    precision: np.ndarray | None = None
    reference_distances: np.ndarray | None = None
    abstention_quantile: float = 0.99
    shrinkage: float = 0.10

    def fit(self, features: np.ndarray) -> "MahalanobisOODController":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 3:
            raise ValueError("features must contain at least three reference scenes")
        self.mean = values.mean(axis=0)
        covariance = np.cov(values, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.asarray([[float(covariance)]])
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - float(self.shrinkage)) * covariance + float(self.shrinkage) * diagonal
        covariance += np.eye(covariance.shape[0]) * 1.0e-6
        self.precision = np.linalg.pinv(covariance)
        self.reference_distances = self.distance(values)
        return self

    def distance(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.precision is None:
            raise RuntimeError("OOD controller has not been fitted")
        values = np.atleast_2d(np.asarray(features, dtype=np.float64))
        centered = values - self.mean[None, :]
        distance2 = np.einsum("bi,ij,bj->b", centered, self.precision, centered)
        return np.sqrt(np.maximum(distance2, 0.0))

    def calibrate_reference_distances(
        self, features: np.ndarray
    ) -> "MahalanobisOODController":
        """Freeze the score ECDF on an independent ID calibration split."""

        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 3:
            raise ValueError("calibration features must contain at least three scenes")
        self.reference_distances = self.distance(values)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.reference_distances is None:
            raise RuntimeError("OOD controller has not been fitted")
        distances = self.distance(features)
        reference = np.sort(self.reference_distances)
        ranks = np.searchsorted(reference, distances, side="right") / float(len(reference) + 1)
        return np.clip(ranks, 0.0, 1.0)

    def control(
        self,
        features: np.ndarray,
        max_interval_inflation: float = 3.0,
        control_start_quantile: float = 0.95,
    ) -> dict[str, np.ndarray]:
        score = self.score(features)
        start = float(
            np.clip(
                control_start_quantile,
                0.0,
                max(float(self.abstention_quantile) - 1.0e-4, 0.0),
            )
        )
        risk = np.clip(
            (score - start)
            / max(float(self.abstention_quantile) - start, 1.0e-6),
            0.0,
            1.0,
        )
        return {
            "ood_score": score,
            "ood_risk": risk,
            "bias_gate_multiplier": 1.0 - risk,
            "interval_inflation": 1.0 + float(max_interval_inflation) * risk**2,
            "abstain": score >= float(self.abstention_quantile),
        }


@dataclass
class MaxScoreOODController:
    """Conservative union of independently calibrated OOD screens."""

    controllers: Sequence[MahalanobisOODController]
    abstention_quantile: float = 0.99

    def score(self, feature_sets: Sequence[np.ndarray]) -> np.ndarray:
        if len(feature_sets) != len(self.controllers):
            raise ValueError("one feature matrix is required for each OOD controller")
        scores = [
            controller.score(features)
            for controller, features in zip(
                self.controllers, feature_sets, strict=True
            )
        ]
        return np.max(np.stack(scores, axis=0), axis=0)

    def control(
        self,
        feature_sets: Sequence[np.ndarray],
        max_interval_inflation: float = 3.0,
        control_start_quantile: float = 0.95,
    ) -> dict[str, np.ndarray]:
        score = self.score(feature_sets)
        start = float(
            np.clip(
                control_start_quantile,
                0.0,
                max(float(self.abstention_quantile) - 1.0e-4, 0.0),
            )
        )
        risk = np.clip(
            (score - start)
            / max(float(self.abstention_quantile) - start, 1.0e-6),
            0.0,
            1.0,
        )
        return {
            "ood_score": score,
            "ood_risk": risk,
            "bias_gate_multiplier": 1.0 - risk,
            "interval_inflation": 1.0 + float(max_interval_inflation) * risk**2,
            "abstain": score >= float(self.abstention_quantile),
        }
