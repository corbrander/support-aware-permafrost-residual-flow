from __future__ import annotations

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, SUPPORT_TYPES, ObservationTable
from cold_recon.operators.support import (
    apply_surface_crossing,
    build_error_covariance,
    build_observation_operator,
    normalized_misfit,
    point_trilinear_operator,
)


def _grid() -> dict:
    return {
        "x": np.arange(4, dtype=np.float32),
        "y": np.arange(3, dtype=np.float32),
        "z": np.arange(5, dtype=np.float32) * 0.5,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 0.5,
    }


def test_point_trilinear_operator_reproduces_linear_field() -> None:
    grid = _grid()
    coord = np.array([[1.25, 0.5, 0.75]], dtype=np.float32)
    xx, yy, zz = np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij")
    field = 2.0 * xx - yy + 3.0 * zz
    operator = point_trilinear_operator(coord, grid)
    assert np.allclose(np.asarray(operator.sum(axis=1)).ravel(), 1.0)
    predicted = np.asarray(operator @ field.reshape(-1)).ravel()
    np.testing.assert_allclose(predicted, [2.0 * 1.25 - 0.5 + 3.0 * 0.75], atol=1e-6)


def test_interval_volume_and_kernel_rows_are_normalized() -> None:
    grid = _grid()
    observations = ObservationTable(
        coords=np.array([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32),
        type_ids=np.array(
            [OBS_TYPES["borehole_eic"], OBS_TYPES["ert_log_resistivity"], OBS_TYPES["nmr_unfrozen_water"]]
        ),
        values=np.zeros(3, dtype=np.float32),
        sigma=np.full(3, 0.1, dtype=np.float32),
        mask=np.ones(3, dtype=bool),
        support_type_ids=np.array(
            [SUPPORT_TYPES["borehole_interval"], SUPPORT_TYPES["ert_volume"], SUPPORT_TYPES["nmr_kernel"]]
        ),
        support_extent=np.array([[0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [2.0, 2.0, 1.0]], dtype=np.float32),
    )
    operator = build_observation_operator(observations, grid)
    np.testing.assert_allclose(np.asarray(operator.matrix.sum(axis=1)).ravel(), 1.0)
    assert np.all(operator.matrix.getnnz(axis=1) > 1)


def test_categorical_interval_operator_averages_probabilities_not_class_ids() -> None:
    grid = _grid()
    observations = ObservationTable(
        coords=np.array([[1.0, 1.0, 0.75]], dtype=np.float32),
        type_ids=np.array([OBS_TYPES["borehole_facies"]]),
        values=np.array([0.0], dtype=np.float32),
        sigma=np.array([0.05], dtype=np.float32),
        mask=np.ones(1, dtype=bool),
        support_type_ids=np.array([SUPPORT_TYPES["borehole_interval"]]),
        support_extent=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    probabilities = np.zeros((4, 3, 5, 2), dtype=np.float32)
    probabilities[..., 0] = 0.25
    probabilities[..., 1] = 0.75
    result = build_observation_operator(observations, grid).apply_probabilities(probabilities)
    np.testing.assert_allclose(result, [[0.25, 0.75]], atol=1e-7)


def test_profile_covariance_is_spd_and_changes_normalized_misfit() -> None:
    observations = ObservationTable(
        coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float32),
        type_ids=np.full(3, OBS_TYPES["ert_log_resistivity"]),
        values=np.zeros(3, dtype=np.float32),
        sigma=np.full(3, 0.2, dtype=np.float32),
        mask=np.ones(3, dtype=bool),
        group_ids=np.array([7, 7, 7]),
    )
    diagonal = build_error_covariance(observations, correlated=False)
    correlated = build_error_covariance(observations, correlated=True, length_scale=2.0)
    assert np.all(np.linalg.eigvalsh(correlated) > 0.0)
    assert correlated[0, 1] > correlated[0, 2] > 0.0
    residual = np.array([0.1, 0.1, 0.1])
    assert normalized_misfit(residual, np.zeros(3), correlated) != normalized_misfit(
        residual, np.zeros(3), diagonal
    )


def test_surface_crossing_interpolates_zero_degree_depth() -> None:
    temperature = np.array([[[1.0, 0.5, -0.5, -1.0]]], dtype=np.float32)
    z = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(apply_surface_crossing(temperature, z), [[0.75]])
