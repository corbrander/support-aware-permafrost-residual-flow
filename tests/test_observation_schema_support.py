from __future__ import annotations

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, SUPPORT_TYPES, ObservationTable, observations_from_npz
from cold_recon.models.observation_tokenizer import ObservationTokenizer


def test_legacy_observation_archive_receives_safe_support_defaults() -> None:
    legacy = {
        "obs_coords": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "obs_type_ids": np.array([OBS_TYPES["borehole_eic"]]),
        "obs_values": np.array([0.3], dtype=np.float32),
        "obs_sigma": np.array([0.1], dtype=np.float32),
        "obs_mask": np.array([1], dtype=np.uint8),
        "obs_times": np.array([np.nan], dtype=np.float32),
    }
    table = observations_from_npz(legacy)
    assert table.support_type_ids[0] == SUPPORT_TYPES["point"]
    np.testing.assert_allclose(table.support_extent, 0.0)
    assert table.source_ids[0] == OBS_TYPES["borehole_eic"]


def test_support_aware_tokens_include_log_sigma_support_and_provenance() -> None:
    table = ObservationTable(
        coords=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        type_ids=np.array([OBS_TYPES["ert_log_resistivity"]]),
        values=np.array([5.0], dtype=np.float32),
        sigma=np.array([0.2], dtype=np.float32),
        mask=np.ones(1, dtype=bool),
        support_type_ids=np.array([SUPPORT_TYPES["ert_volume"]]),
        support_extent=np.array([[4.0, 2.0, 1.0]], dtype=np.float32),
        orientation=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        quality=np.array([0.8], dtype=np.float32),
        site_ids=np.array([2]),
        source_ids=np.array([3]),
    )
    tokenizer = ObservationTokenizer(n_types=9, support_aware=True, n_sites=4, n_sources=10)
    tokens = tokenizer.encode_numpy(table)
    assert tokens.shape == (1, tokenizer.token_dim)
    assert np.isclose(tokens[0, 4 + 9 + 1], np.log(0.2))
    assert np.isfinite(tokens).all()
