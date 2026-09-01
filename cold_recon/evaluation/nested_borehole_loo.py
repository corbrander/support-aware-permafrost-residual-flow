from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cold_recon.data.data_schema import ObservationTable
from cold_recon.evaluation.block_conformal import BlockConformalCalibrator


@dataclass(frozen=True)
class NestedLOOFoldResult:
    held_group_id: int
    n_development_observations: int
    n_held_observations: int
    rmse: float
    mae: float
    raw_coverage: float
    calibrated_coverage: float
    raw_width: float
    calibrated_width: float
    conformal_quantile: float


def run_nested_borehole_loo(
    observations: ObservationTable,
    *,
    fit_adapter: Callable[[ObservationTable], Any],
    reconstruct: Callable[[Any, ObservationTable], Any],
    predict_observations: Callable[[Any, ObservationTable], tuple[np.ndarray, np.ndarray]],
    level: float = 0.90,
    min_inner_groups: int = 3,
) -> list[NestedLOOFoldResult]:
    """Nested complete-borehole LOO with leakage-safe adapter and calibration fits."""

    active = np.asarray(observations.mask, dtype=bool) & (observations.group_ids >= 0)
    groups = np.unique(observations.group_ids[active])
    if len(groups) < max(4, int(min_inner_groups) + 1):
        raise ValueError("nested LOO requires enough complete borehole groups")
    alpha = 0.5 * (1.0 - float(level))
    normal_quantile = 1.6448536269514722 if abs(float(level) - 0.90) < 1.0e-8 else 1.96
    rows: list[NestedLOOFoldResult] = []

    for outer_group in groups:
        development_indices = np.flatnonzero(active & (observations.group_ids != outer_group))
        held_indices = np.flatnonzero(active & (observations.group_ids == outer_group))
        development = observations.subset(development_indices)
        held = observations.subset(held_indices)
        inner_groups = np.unique(development.group_ids)
        calibration_truth: list[np.ndarray] = []
        calibration_mean: list[np.ndarray] = []
        calibration_std: list[np.ndarray] = []
        calibration_blocks: list[np.ndarray] = []
        for inner_group in inner_groups:
            inner_train = development.subset(np.flatnonzero(development.group_ids != inner_group))
            inner_held = development.subset(np.flatnonzero(development.group_ids == inner_group))
            adapter = fit_adapter(inner_train)
            posterior = reconstruct(adapter, inner_train)
            mean, std = predict_observations(posterior, inner_held)
            calibration_truth.append(inner_held.values.astype(np.float64))
            calibration_mean.append(np.asarray(mean, dtype=np.float64))
            calibration_std.append(np.asarray(std, dtype=np.float64))
            calibration_blocks.append(np.full(inner_held.n_obs, int(inner_group), dtype=np.int64))

        calibrator = BlockConformalCalibrator(
            level=float(level),
            min_blocks_per_stratum=int(min_inner_groups),
        ).fit(
            np.concatenate(calibration_truth),
            np.concatenate(calibration_mean),
            np.concatenate(calibration_std),
            np.concatenate(calibration_blocks),
        )
        final_adapter = fit_adapter(development)
        final_posterior = reconstruct(final_adapter, development)
        mean, std = predict_observations(final_posterior, held)
        mean = np.asarray(mean, dtype=np.float64)
        std = np.maximum(np.asarray(std, dtype=np.float64), calibrator.std_floor)
        truth = held.values.astype(np.float64)
        raw_lower = mean - normal_quantile * std
        raw_upper = mean + normal_quantile * std
        calibrated_lower, calibrated_upper = calibrator.interval(mean, std)
        rows.append(
            NestedLOOFoldResult(
                held_group_id=int(outer_group),
                n_development_observations=development.n_obs,
                n_held_observations=held.n_obs,
                rmse=float(np.sqrt(np.mean((mean - truth) ** 2))),
                mae=float(np.mean(np.abs(mean - truth))),
                raw_coverage=float(np.mean((truth >= raw_lower) & (truth <= raw_upper))),
                calibrated_coverage=float(
                    np.mean((truth >= calibrated_lower) & (truth <= calibrated_upper))
                ),
                raw_width=float(np.mean(raw_upper - raw_lower)),
                calibrated_width=float(np.mean(calibrated_upper - calibrated_lower)),
                conformal_quantile=float(calibrator.global_quantile),
            )
        )
    return rows
