from __future__ import annotations

import numpy as np

from cold_recon.evaluation.engineering_response import (
    engineering_response_metrics,
    potential_settlement,
)


def _grid() -> dict[str, np.ndarray | float]:
    return {
        "x": np.arange(3, dtype=np.float32) * 2.0,
        "y": np.arange(3, dtype=np.float32) * 2.0,
        "z": np.arange(8, dtype=np.float32) * 0.25,
        "dx": 2.0,
        "dy": 2.0,
        "dz": 0.25,
    }


def test_potential_settlement_integrates_over_last_axis() -> None:
    eic = np.full((2, 3, 8), 0.20, dtype=np.float32)
    response = potential_settlement(eic, _grid()["z"] < 1.0, 0.25)
    assert response.shape == (2, 3)
    np.testing.assert_allclose(response, 0.20)


def test_response_metrics_distinguish_bias_and_interval_width() -> None:
    truth = np.zeros((3, 3, 8), dtype=np.float32)
    truth[1:, 1:, :4] = 0.40
    candidate = truth + 0.05
    samples = np.stack([candidate - 0.02, candidate + 0.02], axis=0)
    metrics = engineering_response_metrics(
        scene_id="scene",
        method="candidate",
        seed=1,
        truth_eic=truth,
        candidate_eic_mean=candidate,
        candidate_eic_samples=samples,
        candidate_eic_std=samples.std(axis=0),
        conformal_quantile=2.0,
        grid=_grid(),
        thaw_depth_m=1.0,
        screening_threshold_m=0.30,
    )
    assert np.isclose(metrics["response_bias_m"], 0.05)
    assert np.isclose(metrics["response_rmse_m"], 0.05)
    assert metrics["raw_interval_mean_width_m"] > 0.0
    assert metrics["conformal_envelope_mean_width_m"] > 0.0
    assert 0.0 <= metrics["sensitivity"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0


def test_response_metrics_returns_nan_when_class_is_absent() -> None:
    truth = np.zeros((3, 3, 8), dtype=np.float32)
    metrics = engineering_response_metrics(
        scene_id="scene",
        method="candidate",
        seed=1,
        truth_eic=truth,
        candidate_eic_mean=truth,
        grid=_grid(),
        thaw_depth_m=1.0,
        screening_threshold_m=0.30,
    )
    assert np.isnan(metrics["sensitivity"])
    assert np.isclose(metrics["specificity"], 1.0)

