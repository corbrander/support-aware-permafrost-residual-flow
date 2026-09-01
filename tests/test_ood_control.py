from __future__ import annotations

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.evaluation.ood_control import (
    MahalanobisOODController,
    MaxScoreOODController,
    observation_ood_features,
    prior_context_ood_features,
    scene_ood_features,
)


def test_ood_controller_reduces_gate_inflates_interval_and_abstains() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(0.0, 1.0, size=(100, 5))
    controller = MahalanobisOODController(abstention_quantile=0.95).fit(reference)
    control = controller.control(np.array([[12.0, 12.0, 12.0, 12.0, 12.0]]))
    assert control["ood_score"][0] > 0.95
    assert control["bias_gate_multiplier"][0] < 0.05
    assert control["interval_inflation"][0] > 1.0
    assert control["abstain"][0]

    ordinary = controller.control(reference[[0]])
    if ordinary["ood_score"][0] <= 0.95:
        assert ordinary["bias_gate_multiplier"][0] == 1.0
        assert ordinary["interval_inflation"][0] == 1.0

    calibration = rng.normal(1.0, 1.0, size=(40, 5))
    controller.calibrate_reference_distances(calibration)
    assert len(controller.reference_distances) == 40


def test_observation_ood_features_include_robust_value_distribution() -> None:
    observations = ObservationTable(
        coords=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        type_ids=np.full(2, OBS_TYPES["borehole_eic"], dtype=np.int64),
        values=np.asarray([0.1, 0.3], dtype=np.float32),
        sigma=np.full(2, 0.02, dtype=np.float32),
        mask=np.ones(2, dtype=bool),
    )
    grid = {
        "x": np.arange(3, dtype=np.float32),
        "y": np.arange(2, dtype=np.float32),
        "z": np.arange(3, dtype=np.float32),
    }
    shifted = observations.subset(np.arange(2))
    shifted.values += 1.0

    baseline_features = observation_ood_features(observations, grid)
    shifted_features = observation_ood_features(shifted, grid)

    assert baseline_features.shape == shifted_features.shape
    assert not np.allclose(baseline_features, shifted_features)


def test_scene_ood_features_detect_prior_geometry_without_truth() -> None:
    observations = ObservationTable(
        coords=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        type_ids=np.asarray([OBS_TYPES["borehole_eic"]]),
        values=np.asarray([0.2], dtype=np.float32),
        sigma=np.asarray([0.03], dtype=np.float32),
        mask=np.ones(1, dtype=bool),
    )
    grid = {
        "x": np.arange(4, dtype=np.float32),
        "y": np.arange(4, dtype=np.float32),
        "z": np.arange(3, dtype=np.float32),
    }
    shape = (4, 4, 3)
    smooth = np.zeros(shape, dtype=np.float32)
    abrupt = smooth.copy()
    abrupt[2:] = 0.6

    def prior(eic: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "eic": eic,
            "temperature": -2.0 + eic,
            "unfrozen_water": 0.1 + 0.2 * eic,
            "log_resistivity": 6.0 + eic,
            "facies": (eic > 0.3).astype(np.int16),
        }

    smooth_features = prior_context_ood_features(prior(smooth))
    abrupt_features = prior_context_ood_features(prior(abrupt))
    assert smooth_features.shape == abrupt_features.shape
    assert np.isfinite(abrupt_features).all()
    assert not np.allclose(smooth_features, abrupt_features)
    combined = scene_ood_features(observations, grid, prior(abrupt))
    assert len(combined) > len(abrupt_features)


def test_max_score_ood_controller_uses_conservative_union() -> None:
    rng = np.random.default_rng(19)
    reference = rng.normal(size=(80, 2))
    first = MahalanobisOODController().fit(reference)
    second = MahalanobisOODController().fit(reference)
    union = MaxScoreOODController((first, second), abstention_quantile=0.98)
    ordinary = reference[[0]]
    shifted = np.asarray([[20.0, 20.0]])
    control = union.control((ordinary, shifted))
    assert control["ood_score"][0] >= second.score(shifted)[0]
    assert control["abstain"][0]
