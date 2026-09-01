from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_THAW_DEPTHS_M = (2.0, 4.0, 6.0)
DEFAULT_SCREENING_THRESHOLD_M = 0.30
DEFAULT_DECISION_PROBABILITY = 0.50


def potential_settlement(
    eic: np.ndarray,
    depth_mask: np.ndarray,
    dz: float,
) -> np.ndarray:
    """Excess-ice-only potential settlement after complete thaw and drainage."""

    return np.sum(np.asarray(eic)[..., depth_mask], axis=-1) * float(dz)


def horizontal_gradient_magnitude(
    field: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    gx, gy = np.gradient(
        np.asarray(field, dtype=np.float64),
        float(dx),
        float(dy),
        edge_order=1,
    )
    return np.hypot(gx, gy)


def _rmse(candidate: np.ndarray, truth: np.ndarray) -> float:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64
    )
    return float(np.sqrt(np.mean(delta**2)))


def _safe_rate(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return float("nan")
    return float(numerator / denominator)


def engineering_response_metrics(
    *,
    scene_id: str,
    method: str,
    seed: int,
    truth_eic: np.ndarray,
    candidate_eic_mean: np.ndarray,
    grid: dict[str, Any],
    thaw_depth_m: float,
    candidate_eic_samples: np.ndarray | None = None,
    candidate_eic_std: np.ndarray | None = None,
    conformal_quantile: float | None = None,
    screening_threshold_m: float = DEFAULT_SCREENING_THRESHOLD_M,
    decision_probability: float = DEFAULT_DECISION_PROBABILITY,
) -> dict[str, float | int | str]:
    """Compute scene-level metrics for the controlled response stress test.

    The response is deliberately restricted to the integral of excess ice over a
    prescribed newly thawed thickness. It excludes consolidation, drainage rate,
    creep, and structural load transfer, and therefore is not a settlement model.
    """

    z = np.asarray(grid["z"], dtype=np.float64)
    dx = float(grid.get("dx", np.mean(np.diff(np.asarray(grid["x"])))))
    dy = float(grid.get("dy", np.mean(np.diff(np.asarray(grid["y"])))))
    dz = float(grid.get("dz", np.mean(np.diff(z))))
    depth_mask = z < float(thaw_depth_m)
    represented_depth = float(np.sum(depth_mask) * dz)
    if not np.isclose(represented_depth, float(thaw_depth_m), atol=1.0e-6):
        raise ValueError(
            f"thaw depth {thaw_depth_m} m is not represented exactly; "
            f"grid represents {represented_depth} m"
        )

    truth_response = potential_settlement(truth_eic, depth_mask, dz)
    candidate_response = potential_settlement(candidate_eic_mean, depth_mask, dz)
    truth_gradient = horizontal_gradient_magnitude(truth_response, dx, dy)
    candidate_gradient = horizontal_gradient_magnitude(candidate_response, dx, dy)

    truth_flag = truth_response > float(screening_threshold_m)
    if candidate_eic_samples is not None:
        response_samples = potential_settlement(
            np.asarray(candidate_eic_samples), depth_mask, dz
        )
        exceedance_probability = np.mean(
            response_samples > float(screening_threshold_m), axis=0
        )
        candidate_flag = exceedance_probability >= float(decision_probability)
        raw_lower = np.quantile(response_samples, 0.05, axis=0)
        raw_upper = np.quantile(response_samples, 0.95, axis=0)
        raw_coverage = float(
            np.mean((truth_response >= raw_lower) & (truth_response <= raw_upper))
        )
        raw_width = float(np.mean(raw_upper - raw_lower))
    else:
        candidate_flag = candidate_response > float(screening_threshold_m)
        raw_coverage = float("nan")
        raw_width = float("nan")

    positives = int(np.sum(truth_flag))
    negatives = int(np.sum(~truth_flag))
    tp = int(np.sum(candidate_flag & truth_flag))
    tn = int(np.sum(~candidate_flag & ~truth_flag))

    conformal_coverage = float("nan")
    conformal_width = float("nan")
    if candidate_eic_std is not None and conformal_quantile is not None:
        eic_std = np.asarray(candidate_eic_std, dtype=np.float64)
        half_width = float(conformal_quantile) * np.maximum(eic_std, 0.001)
        lower_eic = np.clip(
            np.asarray(candidate_eic_mean, dtype=np.float64) - half_width,
            0.0,
            0.90,
        )
        upper_eic = np.clip(
            np.asarray(candidate_eic_mean, dtype=np.float64) + half_width,
            0.0,
            0.90,
        )
        lower_response = potential_settlement(lower_eic, depth_mask, dz)
        upper_response = potential_settlement(upper_eic, depth_mask, dz)
        conformal_coverage = float(
            np.mean(
                (truth_response >= lower_response)
                & (truth_response <= upper_response)
            )
        )
        conformal_width = float(np.mean(upper_response - lower_response))

    return {
        "scene_id": str(scene_id),
        "method": str(method),
        "seed": int(seed),
        "thaw_depth_m": float(thaw_depth_m),
        "screening_threshold_m": float(screening_threshold_m),
        "decision_probability": float(decision_probability),
        "truth_mean_response_m": float(np.mean(truth_response)),
        "predicted_mean_response_m": float(np.mean(candidate_response)),
        "response_bias_m": float(np.mean(candidate_response - truth_response)),
        "response_rmse_m": _rmse(candidate_response, truth_response),
        "gradient_rmse_m_per_m": _rmse(candidate_gradient, truth_gradient),
        "truth_gradient_p95_m_per_m": float(np.quantile(truth_gradient, 0.95)),
        "predicted_gradient_p95_m_per_m": float(
            np.quantile(candidate_gradient, 0.95)
        ),
        "truth_flagged_fraction": float(np.mean(truth_flag)),
        "predicted_flagged_fraction": float(np.mean(candidate_flag)),
        "sensitivity": _safe_rate(tp, positives),
        "specificity": _safe_rate(tn, negatives),
        "positive_cells": positives,
        "negative_cells": negatives,
        "raw_interval_coverage": raw_coverage,
        "raw_interval_mean_width_m": raw_width,
        "conformal_envelope_coverage": conformal_coverage,
        "conformal_envelope_mean_width_m": conformal_width,
    }

