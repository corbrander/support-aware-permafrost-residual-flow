from __future__ import annotations

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, SUPPORT_TYPES, ObservationTable
from cold_recon.data.support_raster import (
    PreparedSupportRaster,
    build_nearest_voxel_raster,
    build_support_raster,
    collapse_to_nearest_voxel_observations,
)


def test_support_raster_spreads_interval_information_and_reports_distance() -> None:
    grid = {
        "x": np.arange(3, dtype=np.float32),
        "y": np.arange(3, dtype=np.float32),
        "z": np.arange(6, dtype=np.float32) * 0.5,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 0.5,
    }
    observations = ObservationTable(
        coords=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        type_ids=np.array([OBS_TYPES["borehole_eic"]]),
        values=np.array([0.4], dtype=np.float32),
        sigma=np.array([0.1], dtype=np.float32),
        mask=np.ones(1, dtype=bool),
        support_type_ids=np.array([SUPPORT_TYPES["borehole_interval"]]),
        support_extent=np.array([[0.0, 0.0, 1.5]], dtype=np.float32),
    )
    raster, diagnostics = build_support_raster(observations, grid)
    eic_value = raster[2]
    eic_density = raster[3]
    assert np.count_nonzero(eic_density) >= 3
    np.testing.assert_allclose(eic_value[eic_density > 0], 0.4)
    assert diagnostics["distance_to_support"][0, 0, -1] > 0.0
    cached_raster, cached_diagnostics = PreparedSupportRaster.prepare(
        observations, grid
    ).apply(observations)
    np.testing.assert_allclose(cached_raster, raster)
    np.testing.assert_allclose(
        cached_diagnostics["distance_to_support"], diagnostics["distance_to_support"]
    )
    nearest, _ = build_nearest_voxel_raster(observations, grid)
    assert np.count_nonzero(nearest[3]) == 1
    assert np.count_nonzero(nearest[3]) < np.count_nonzero(eic_density)


def test_nearest_voxel_ablation_removes_finite_support_but_preserves_data() -> None:
    grid = {
        "x": np.arange(3, dtype=np.float32),
        "y": np.arange(3, dtype=np.float32),
        "z": np.arange(4, dtype=np.float32) * 0.5,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 0.5,
    }
    observations = ObservationTable(
        coords=np.array([[0.6, 1.4, 0.74]], dtype=np.float32),
        type_ids=np.array([OBS_TYPES["ert_log_resistivity"]]),
        values=np.array([7.2], dtype=np.float32),
        sigma=np.array([0.3], dtype=np.float32),
        mask=np.ones(1, dtype=bool),
        support_type_ids=np.array([SUPPORT_TYPES["ert_volume"]]),
        support_extent=np.array([[2.0, 2.0, 1.0]], dtype=np.float32),
        orientation=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        group_ids=np.array([9], dtype=np.int64),
    )
    collapsed = collapse_to_nearest_voxel_observations(observations, grid)
    np.testing.assert_allclose(collapsed.coords, [[1.0, 1.0, 0.5]])
    np.testing.assert_allclose(collapsed.values, observations.values)
    np.testing.assert_allclose(collapsed.sigma, observations.sigma)
    assert collapsed.type_ids[0] == observations.type_ids[0]
    assert collapsed.support_type_ids[0] == SUPPORT_TYPES["point"]
    assert collapsed.group_ids[0] == -1
    assert not np.any(collapsed.support_extent)
    assert not np.any(collapsed.orientation)
