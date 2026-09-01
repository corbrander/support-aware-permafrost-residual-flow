from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cold_recon.evaluation.uncertainty import ensemble_crps


def _finite_scores(
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    std_floor: float,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return np.abs(truth - mean) / np.maximum(std, float(std_floor))


@dataclass
class BlockConformalCalibrator:
    level: float = 0.90
    std_floor: float = 1.0e-3
    within_block_quantile: float = 0.90
    min_blocks_per_stratum: int = 4
    global_quantile: float | None = None
    stratum_quantiles: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        truth: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        block_ids: np.ndarray,
        strata: np.ndarray | None = None,
    ) -> "BlockConformalCalibrator":
        scores = _finite_scores(truth, mean, std, self.std_floor).reshape(-1)
        blocks = np.asarray(block_ids).reshape(-1)
        if len(scores) != len(blocks):
            raise ValueError("block_ids length does not match calibration values")
        finite = np.isfinite(scores)

        def block_scores(mask: np.ndarray) -> np.ndarray:
            """Reduce voxel scores to one score per block in O(n log n).

            The former implementation scanned the complete voxel vector once
            for every block.  A 100-scene validation set contains about 32,000
            blocks and 12 million voxels, making that quadratic-style scan
            impractical.  Sorting the selected block labels once preserves the
            exact blockwise quantiles while keeping memory and run time bounded.
            """

            selected_blocks = np.asarray(blocks[mask])
            selected_scores = np.asarray(scores[mask], dtype=np.float64)
            if selected_scores.size == 0:
                return np.empty(0, dtype=np.float64)
            order = np.argsort(selected_blocks, kind="stable")
            selected_blocks = selected_blocks[order]
            selected_scores = selected_scores[order]
            starts = np.r_[0, np.flatnonzero(np.diff(selected_blocks)) + 1]
            stops = np.r_[starts[1:], selected_scores.size]
            return np.fromiter(
                (
                    np.quantile(
                        selected_scores[start:stop],
                        self.within_block_quantile,
                    )
                    for start, stop in zip(starts, stops, strict=True)
                ),
                dtype=np.float64,
                count=len(starts),
            )

        global_blocks = block_scores(finite)
        if len(global_blocks) == 0:
            raise ValueError("no finite conformal calibration scores")
        adjusted = min(1.0, np.ceil((len(global_blocks) + 1) * self.level) / len(global_blocks))
        self.global_quantile = float(np.quantile(global_blocks, adjusted, method="higher"))
        self.stratum_quantiles = {}
        if strata is None:
            return self
        labels = np.asarray(strata, dtype=object).reshape(-1)
        if len(labels) != len(scores):
            raise ValueError("strata length does not match calibration values")
        for label in np.unique(labels[finite]):
            values = block_scores(finite & (labels == label))
            if len(values) < int(self.min_blocks_per_stratum):
                continue
            adjusted_local = min(1.0, np.ceil((len(values) + 1) * self.level) / len(values))
            self.stratum_quantiles[str(label)] = float(
                np.quantile(values, adjusted_local, method="higher")
            )
        return self

    def quantile_for(self, strata: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
        if self.global_quantile is None:
            raise RuntimeError("calibrator has not been fitted")
        quantile = np.full(shape, float(self.global_quantile), dtype=np.float64)
        if strata is not None:
            labels = np.asarray(strata, dtype=object)
            for label, value in self.stratum_quantiles.items():
                quantile[labels == label] = float(value)
        return quantile

    def interval(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        strata: np.ndarray | None = None,
        inflation: np.ndarray | float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        quantile = self.quantile_for(strata, mean.shape)
        half_width = quantile * np.maximum(std, self.std_floor) * np.asarray(inflation)
        return (mean - half_width).astype(np.float32), (mean + half_width).astype(np.float32)


def interval_diagnostics(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    finite = np.isfinite(truth) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(finite):
        return {"coverage": float("nan"), "mean_width": float("nan")}
    return {
        "coverage": float(np.mean((truth[finite] >= lower[finite]) & (truth[finite] <= upper[finite]))),
        "mean_width": float(np.mean(upper[finite] - lower[finite])),
    }


def pit_values(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    truth = np.asarray(truth)
    if samples.shape[1:] != truth.shape:
        raise ValueError("sample and truth shapes differ")
    return ((np.sum(samples < truth[None, ...], axis=0) + 0.5) / (samples.shape[0] + 1.0)).astype(np.float32)


def energy_score(samples: np.ndarray, truth: np.ndarray, max_members: int = 64) -> float:
    ensemble = np.asarray(samples, dtype=np.float64).reshape(samples.shape[0], -1)
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    ensemble = ensemble[: int(max_members)]
    first = np.mean(np.linalg.norm(ensemble - target[None, :], axis=1))
    squared_norm = np.einsum("ij,ij->i", ensemble, ensemble)
    squared_distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * (ensemble @ ensemble.T)
    )
    pairwise_distance = np.sqrt(np.maximum(squared_distance, 0.0))
    second = 0.5 * float(np.mean(pairwise_distance))
    return float(first - second)


def posterior_diagnostics(samples: np.ndarray, truth: np.ndarray, level: float = 0.90) -> dict[str, float]:
    samples = np.asarray(samples, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    alpha = 0.5 * (1.0 - float(level))
    lower = np.quantile(samples, alpha, axis=0)
    upper = np.quantile(samples, 1.0 - alpha, axis=0)
    interval = interval_diagnostics(truth, lower, upper)
    return {
        **interval,
        "crps": float(ensemble_crps(samples, truth)),
        "energy_score": energy_score(samples, truth),
        "pit_mean": float(np.mean(pit_values(samples, truth))),
        "pit_variance": float(np.var(pit_values(samples, truth))),
    }
