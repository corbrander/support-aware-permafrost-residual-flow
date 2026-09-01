from __future__ import annotations

import numpy as np

from cold_recon.evaluation.posterior_calibration import (
    bias_quantile_calibrated_samples,
    calibrate_posterior_spread,
    find_spread_scale,
    spread_scale_samples,
)
from cold_recon.evaluation.uncertainty import interval_coverage


def test_spread_scale_preserves_mean() -> None:
    samples = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    scaled = spread_scale_samples(samples, scale=3.0)
    np.testing.assert_allclose(scaled.mean(axis=0), samples.mean(axis=0))
    assert scaled.std() > samples.std()


def test_find_spread_scale_improves_coverage() -> None:
    samples = np.array([[0.9, 1.9, 2.9], [1.1, 2.1, 3.1]], dtype=np.float32)
    truth = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    before, _ = interval_coverage(samples, truth, level=0.9)
    scale, after = find_spread_scale(samples, truth, target_coverage=0.9, level=0.9, max_scale=100.0)
    assert scale > 1.0
    assert after >= before


def test_calibrate_posterior_spread_updates_std_and_ice_probability() -> None:
    shape = (2, 2, 2)
    base = np.ones(shape, dtype=np.float32)
    posterior = {
        "eic_samples": np.stack([base * 0.1, base * 0.2]),
        "eic_mean": base * 0.15,
        "eic_std": base * 0.05,
    }
    truth = {"eic": base * 0.5}
    calibrated, rows = calibrate_posterior_spread(posterior, truth, target_coverage=0.9, level=0.9)
    assert rows[0]["target"] == "eic"
    assert calibrated["eic_std"].mean() > posterior["eic_std"].mean()
    assert "ice_rich_probability" in calibrated


def test_bias_quantile_fallback_handles_biased_zero_spread_samples() -> None:
    truth = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    samples = np.stack([truth * 0.0 + 0.2, truth * 0.0 + 0.2]).astype(np.float32)
    before, _ = interval_coverage(samples, truth, level=0.9)
    calibrated, info = bias_quantile_calibrated_samples(samples, truth, target_coverage=0.9, level=0.9)
    after, _ = interval_coverage(calibrated, truth, level=0.9)
    assert before < 0.9
    assert after >= 0.9
    assert np.isfinite(info["bias_correction"])
    assert info["residual_half_width"] > 0.0


def test_calibrate_posterior_spread_uses_bias_quantile_when_spread_scaling_cannot_cover() -> None:
    truth = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    samples = np.stack([truth * 0.0 + 0.2, truth * 0.0 + 0.2]).astype(np.float32)
    posterior = {
        "unfrozen_water_samples": samples,
        "unfrozen_water_mean": samples.mean(axis=0),
        "unfrozen_water_std": samples.std(axis=0),
    }
    calibrated, rows = calibrate_posterior_spread(
        posterior,
        {"unfrozen_water": truth},
        target_coverage=0.9,
        level=0.9,
    )
    assert rows[0]["target"] == "unfrozen_water"
    assert rows[0]["calibration_method"] == "bias_quantile"
    assert float(rows[0]["coverage_after"]) >= 0.9
    assert calibrated["unfrozen_water_std"].mean() > 0.0
