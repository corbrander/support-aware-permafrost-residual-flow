from __future__ import annotations

import numpy as np

from cold_recon.data.state_factorization import (
    factorize_legacy_state,
    partial_label_cross_entropy_numpy,
)


def test_factorization_separates_material_thermal_and_ice_semantics() -> None:
    facies = np.array([1, 2, 4, 5, 6, 0], dtype=np.int16)
    eic = np.array([0.1, 0.25, 0.05, 0.0, 0.7, 0.0], dtype=np.float32)
    temperature = np.array([-2.0, -1.0, -1.0, 0.2, -3.0, 1.0], dtype=np.float32)
    state = factorize_legacy_state(facies, eic, temperature)
    np.testing.assert_array_equal(state.lithology[:3], [0, 1, 2])
    assert not state.label_mask_lithology[3]
    assert state.thermal_state[3] == 2
    assert state.ice_structure[4] == 2
    assert state.thermal_state[5] == 0


def test_partial_label_loss_ignores_unobserved_targets() -> None:
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    labels = np.array([0, 0])
    masked = partial_label_cross_entropy_numpy(probabilities, labels, np.array([True, False]))
    reference = float(-np.log(0.9))
    np.testing.assert_allclose(masked, reference, rtol=1e-6)
