from __future__ import annotations

import numpy as np


def ensemble_mean_std(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(samples, axis=0), np.std(samples, axis=0)


def facies_entropy(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(probabilities, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def exceedance_probability(samples: np.ndarray, threshold: float) -> np.ndarray:
    return np.mean(samples > threshold, axis=0)


def central_interval(samples: np.ndarray, level: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    alpha = 1.0 - float(level)
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower.astype(np.float32), upper.astype(np.float32)


def interval_coverage(samples: np.ndarray, truth: np.ndarray, level: float = 0.90) -> tuple[float, float]:
    lower, upper = central_interval(samples, level=level)
    valid = np.isfinite(truth) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(valid):
        return float("nan"), float("nan")
    covered = (truth[valid] >= lower[valid]) & (truth[valid] <= upper[valid])
    width = upper[valid] - lower[valid]
    return float(np.mean(covered)), float(np.mean(width))


def ensemble_crps(samples: np.ndarray, truth: np.ndarray, max_points: int | None = 250_000, seed: int = 0) -> float:
    samples = np.asarray(samples, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    if samples.ndim < 2:
        raise ValueError("samples must have shape [ensemble, ...]")
    flat_samples = samples.reshape(samples.shape[0], -1)
    flat_truth = truth.reshape(-1)
    valid = np.isfinite(flat_truth) & np.all(np.isfinite(flat_samples), axis=0)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return float("nan")
    if max_points is not None and len(idx) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=int(max_points), replace=False)
    ens = flat_samples[:, idx]
    obs = flat_truth[idx]
    abs_error = np.mean(np.abs(ens - obs[None, :]), axis=0)
    # For sorted ensemble members x_(i),
    #   0.5 / M^2 * sum_{i,j} |x_i - x_j|
    # = 1 / M^2 * sum_i (2i - M - 1) x_(i).
    # This is exactly the empirical CRPS pairwise term without materializing
    # an [M, M, N] array (several GB for a 64-member 3-D field ensemble).
    member_count = int(ens.shape[0])
    sorted_ensemble = np.sort(ens, axis=0)
    weights = (
        2.0 * np.arange(1, member_count + 1, dtype=np.float64)
        - member_count
        - 1.0
    )
    pairwise_half = np.sum(
        sorted_ensemble.astype(np.float64, copy=False) * weights[:, None],
        axis=0,
    ) / float(member_count**2)
    return float(np.mean(abs_error.astype(np.float64) - pairwise_half))


def reliability_by_level(samples: np.ndarray, truth: np.ndarray, levels: list[float] | tuple[float, ...]) -> dict[float, dict[str, float]]:
    return {
        float(level): {
            "coverage": coverage,
            "width": width,
        }
        for level in levels
        for coverage, width in [interval_coverage(samples, truth, level=float(level))]
    }


def brier_score(probability: np.ndarray, event: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float32)
    event = np.asarray(event, dtype=bool)
    valid = np.isfinite(probability)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((probability[valid] - event[valid].astype(np.float32)) ** 2))


def categorical_nll(probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = (flat_labels >= 0) & (flat_labels < flat_probs.shape[1]) & np.all(np.isfinite(flat_probs), axis=1)
    if not np.any(valid):
        return float("nan")
    selected = flat_probs[np.where(valid)[0], flat_labels[valid]]
    return float(-np.mean(np.log(np.clip(selected, eps, 1.0))))


def uncertainty_error_correlation(std: np.ndarray, abs_error: np.ndarray) -> float:
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    abs_error = np.asarray(abs_error, dtype=np.float32).reshape(-1)
    valid = np.isfinite(std) & np.isfinite(abs_error)
    if np.sum(valid) < 2:
        return float("nan")
    s = std[valid]
    e = abs_error[valid]
    if float(np.std(s)) == 0.0 or float(np.std(e)) == 0.0:
        return float("nan")
    return float(np.corrcoef(s, e)[0, 1])
