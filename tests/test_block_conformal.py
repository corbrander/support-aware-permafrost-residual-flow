from __future__ import annotations

import numpy as np

from cold_recon.evaluation.block_conformal import (
    BlockConformalCalibrator,
    energy_score,
    interval_diagnostics,
    posterior_diagnostics,
)


def test_block_conformal_uses_equal_block_scores_and_improves_coverage() -> None:
    truth = np.linspace(0.0, 1.0, 40)
    mean = truth + 0.2
    std = np.full(40, 0.05)
    blocks = np.repeat(np.arange(8), 5)
    strata = np.repeat(["shallow", "deep"], 20)
    calibrator = BlockConformalCalibrator(level=0.90, min_blocks_per_stratum=3).fit(
        truth, mean, std, blocks, strata
    )
    raw = interval_diagnostics(truth, mean - 1.645 * std, mean + 1.645 * std)
    lower, upper = calibrator.interval(mean, std, strata)
    calibrated = interval_diagnostics(truth, lower, upper)
    assert calibrated["coverage"] > raw["coverage"]
    assert calibrator.global_quantile is not None


def test_block_conformal_matches_interleaved_blockwise_quantiles() -> None:
    scores = np.asarray([1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0])
    blocks = np.asarray([4, 9, 4, 9, 4, 9, 4, 9])
    calibrator = BlockConformalCalibrator(
        level=0.50,
        within_block_quantile=0.50,
    ).fit(
        truth=scores,
        mean=np.zeros_like(scores),
        std=np.ones_like(scores),
        block_ids=blocks,
    )
    block_scores = np.asarray(
        [np.quantile(scores[blocks == block], 0.50) for block in np.unique(blocks)]
    )
    adjusted = min(
        1.0,
        np.ceil((len(block_scores) + 1) * 0.50) / len(block_scores),
    )
    expected = np.quantile(block_scores, adjusted, method="higher")
    assert np.isclose(calibrator.global_quantile, expected)


def test_posterior_diagnostics_reports_crps_energy_and_pit() -> None:
    rng = np.random.default_rng(2)
    truth = np.zeros((3, 2), dtype=np.float32)
    samples = rng.normal(0.0, 0.2, size=(16, 3, 2)).astype(np.float32)
    diagnostics = posterior_diagnostics(samples, truth)
    assert {"coverage", "mean_width", "crps", "energy_score", "pit_mean"}.issubset(diagnostics)
    assert np.isfinite(list(diagnostics.values())).all()


def test_energy_score_matches_direct_pairwise_definition() -> None:
    samples = np.asarray(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 1.5], [2.5, 4.0]],
            [[-0.5, 0.5], [1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    truth = np.asarray([[0.2, 1.1], [2.2, 3.4]], dtype=np.float32)
    flat = samples.reshape(len(samples), -1).astype(np.float64)
    target = truth.reshape(-1).astype(np.float64)
    direct = np.mean(np.linalg.norm(flat - target[None, :], axis=1))
    direct -= 0.5 * np.mean(
        np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    )

    assert np.isclose(energy_score(samples, truth), direct)
