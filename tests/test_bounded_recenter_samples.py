from __future__ import annotations

import numpy as np

from cold_recon.training.factorized_volume_codec import bounded_recenter_samples


def test_bounded_recenter_preserves_mean_and_bounds() -> None:
    samples = np.asarray(
        [
            [[[0.0, 0.8]]],
            [[[0.5, 0.2]]],
            [[[1.0, -0.4]]],
        ],
        dtype=np.float32,
    )
    target = np.asarray([[[0.05, 0.85]]], dtype=np.float32)
    result = bounded_recenter_samples(samples, target, 0.0, 0.90)

    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 0.90 + 1.0e-6
    np.testing.assert_allclose(result.mean(axis=0), target, atol=1.0e-6)
