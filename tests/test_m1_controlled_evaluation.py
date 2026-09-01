from __future__ import annotations

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from scripts.evaluate_m1_controlled import (
    _fit_cached_spatial_conformal,
    apply_observation_mode,
    enforce_exact_anchor_fallback,
    inflate_continuous_posterior,
    spatial_block_ids,
)


def test_inflate_continuous_posterior_refreshes_dependent_products() -> None:
    samples = np.asarray([[[[0.1]]], [[[0.5]]]], dtype=np.float32)
    posterior: dict[str, np.ndarray] = {}
    for name in ("eic", "temperature", "unfrozen_water", "log_resistivity"):
        posterior[f"{name}_samples"] = samples.copy()
        posterior[f"{name}_mean"] = samples.mean(axis=0)
        posterior[f"{name}_std"] = samples.std(axis=0)
    posterior["resistivity_samples"] = np.ones_like(samples)
    posterior["resistivity_mean"] = np.ones_like(samples[0])
    posterior["ice_rich_probability"] = np.zeros_like(samples[0])

    result = inflate_continuous_posterior(posterior, 2.0)

    np.testing.assert_allclose(result["eic_samples"].reshape(-1), [0.0, 0.6], atol=1e-7)
    np.testing.assert_allclose(result["eic_mean"], 0.3, atol=1e-7)
    np.testing.assert_allclose(result["eic_std"], 0.3, atol=1e-7)
    np.testing.assert_allclose(result["ice_rich_probability"], 0.5)
    expected_resistivity = np.exp(np.clip(result["log_resistivity_samples"], 0.0, 15.0))
    np.testing.assert_allclose(result["resistivity_samples"], expected_resistivity)
    np.testing.assert_allclose(
        result["resistivity_mean"], expected_resistivity.mean(axis=0)
    )


def test_exact_fallback_sets_all_means_and_modes_to_tree_anchor() -> None:
    samples = np.asarray([[[[0.1]]], [[[0.5]]], [[[0.3]]]], dtype=np.float32)
    posterior: dict[str, np.ndarray] = {}
    targets = {
        "eic": np.asarray([[[0.22]]], dtype=np.float32),
        "temperature": np.asarray([[[-2.5]]], dtype=np.float32),
        "unfrozen_water": np.asarray([[[0.11]]], dtype=np.float32),
        "log_resistivity": np.asarray([[[6.2]]], dtype=np.float32),
    }
    for index, name in enumerate(targets):
        current = samples + float(index)
        posterior[f"{name}_samples"] = current
        posterior[f"{name}_mean"] = current.mean(axis=0)
        posterior[f"{name}_std"] = current.std(axis=0)
    for name, classes in (("lithology", 4), ("thermal_state", 3), ("ice_structure", 3)):
        posterior[f"{name}_mode"] = np.zeros((1, 1, 1), dtype=np.int16)
        posterior[f"{name}_probability"] = np.full((1, 1, 1, classes), 1 / classes)
        posterior[f"{name}_entropy"] = np.ones((1, 1, 1), dtype=np.float32)
    posterior["resistivity_samples"] = np.ones_like(samples)
    posterior["resistivity_mean"] = np.ones((1, 1, 1), dtype=np.float32)
    posterior["ice_rich_probability"] = np.zeros((1, 1, 1), dtype=np.float32)
    anchor = {
        **targets,
        "lithology": np.asarray([[[2]]], dtype=np.int16),
        "thermal_state": np.asarray([[[1]]], dtype=np.int16),
        "ice_structure": np.asarray([[[2]]], dtype=np.int16),
    }

    result = enforce_exact_anchor_fallback(posterior, anchor)

    for name, target in targets.items():
        np.testing.assert_allclose(result[f"{name}_mean"], target, atol=1e-6)
    for name in ("lithology", "thermal_state", "ice_structure"):
        np.testing.assert_array_equal(result[f"{name}_mode"], anchor[name])
        assert np.all(result[f"{name}_entropy"] == 0.0)


def test_observation_deletion_modes_mask_complete_boreholes_without_leakage() -> None:
    group_ids = np.repeat(np.arange(4), 3)
    type_ids = np.tile(
        [
            OBS_TYPES["borehole_facies"],
            OBS_TYPES["borehole_eic"],
            OBS_TYPES["borehole_temperature"],
        ],
        4,
    )
    observations = ObservationTable(
        coords=np.column_stack(
            [group_ids, np.zeros(12), np.tile(np.arange(3), 4)]
        ).astype(np.float32),
        type_ids=type_ids,
        values=np.zeros(12, dtype=np.float32),
        sigma=np.ones(12, dtype=np.float32),
        mask=np.ones(12, dtype=bool),
        group_ids=group_ids,
    )
    sample = {"observations": observations}

    sparse = apply_observation_mode(sample, "sparse_boreholes", seed=3)
    retained = np.unique(
        sparse["observations"].group_ids[sparse["observations"].mask]
    )
    assert len(retained) == 2
    assert observations.mask.all()


def test_spatial_block_ids_are_contiguous_and_respect_all_three_axes() -> None:
    labels = spatial_block_ids((5, 6, 7), (2, 3, 4))

    assert labels.shape == (5, 6, 7)
    assert np.array_equal(np.unique(labels), np.arange(12))
    assert labels[0, 0, 0] == labels[1, 2, 3]
    assert labels[2, 0, 0] != labels[1, 0, 0]
    assert labels[0, 3, 0] != labels[0, 2, 0]
    assert labels[0, 0, 4] != labels[0, 0, 3]


def test_cached_spatial_conformal_keeps_scene_blocks_independent(tmp_path) -> None:
    records = [{"scene_id": "a"}, {"scene_id": "b"}]
    for index, record in enumerate(records):
        score = np.full((4, 4, 2), 1.0 + index, dtype=np.float32)
        np.savez_compressed(
            tmp_path / f"{record['scene_id']}_eic_score.npz",
            score=score,
            block_shape=np.asarray((2, 2, 1), dtype=np.int32),
        )

    calibrator, scenes, blocks = _fit_cached_spatial_conformal(
        tmp_path,
        records,
        level=0.90,
        within_block_quantile=0.90,
        std_floor=0.001,
    )

    assert scenes == 2
    assert blocks == 16
    assert calibrator.global_quantile == 2.0
