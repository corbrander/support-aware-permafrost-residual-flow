from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cold_recon.evaluation.uncertainty import facies_entropy


@dataclass(frozen=True)
class PosteriorAlignmentModel:
    model: str
    prediction_path: str


DEFAULT_POSTERIORS: tuple[PosteriorAlignmentModel, ...] = (
    PosteriorAlignmentModel("COLDReconLatentDiffusion", "diffusion_posterior.npz"),
    PosteriorAlignmentModel("COLDReconFNOOperatorDiffusion", "fno_operator_diffusion_posterior.npz"),
    PosteriorAlignmentModel("COLDReconRectifiedFlow", "rectified_flow_posterior.npz"),
    PosteriorAlignmentModel("COLDReconLatentDiffusionPhysicsTrained", "diffusion_posterior_physics_trained.npz"),
    PosteriorAlignmentModel("COLDReconLatentDiffusionPhysicsGuided", "diffusion_posterior_physics_guided.npz"),
    PosteriorAlignmentModel("COLDReconLatentDiffusionPhysicsRefined", "diffusion_posterior_physics_refined.npz"),
    PosteriorAlignmentModel("COLDReconLatentDiffusionCalibrated", "diffusion_posterior_calibrated.npz"),
)

CONTINUOUS_TARGETS: tuple[dict[str, str], ...] = (
    {"target": "eic", "mean_key": "eic_mean", "std_key": "eic_std", "truth_key": "eic"},
    {"target": "temperature", "mean_key": "temperature_mean", "std_key": "temperature_std", "truth_key": "temperature"},
    {"target": "unfrozen_water", "mean_key": "unfrozen_water_mean", "std_key": "unfrozen_water_std", "truth_key": "unfrozen_water"},
    {"target": "log_resistivity", "mean_key": "log_resistivity_mean", "std_key": "log_resistivity_std", "truth_key": "resistivity"},
)


def _as_log_resistivity(resistivity: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(resistivity, dtype=np.float32), 1.0)).astype(np.float32)


def rank_correlation(uncertainty: np.ndarray, error: np.ndarray) -> float:
    unc, err = _valid_flatten(uncertainty, error)
    if len(unc) < 2 or float(np.nanstd(unc)) == 0.0 or float(np.nanstd(err)) == 0.0:
        return float("nan")
    return float(pd.Series(unc).corr(pd.Series(err), method="spearman"))


def _valid_flatten(uncertainty: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unc = np.asarray(uncertainty, dtype=np.float32).reshape(-1)
    err = np.asarray(error, dtype=np.float32).reshape(-1)
    valid = np.isfinite(unc) & np.isfinite(err)
    return unc[valid], err[valid]


def top_uncertainty_metrics(uncertainty: np.ndarray, error: np.ndarray, quantile: float = 0.90) -> dict[str, float]:
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0")
    unc, err = _valid_flatten(uncertainty, error)
    if len(unc) == 0:
        return {
            "n_voxels": 0.0,
            "global_error_mean": float("nan"),
            "bottom_uncertainty_error_mean": float("nan"),
            "top_uncertainty_error_mean": float("nan"),
            "top_uncertainty_error_enrichment": float("nan"),
            "bottom_uncertainty_error_ratio": float("nan"),
            "top_uncertainty_captures_top_error_rate": float("nan"),
        }
    n_tail = max(1, int(np.ceil((1.0 - float(quantile)) * len(unc))))
    order = np.argsort(unc, kind="mergesort")
    top_unc = np.zeros(len(unc), dtype=bool)
    bottom_unc = np.zeros(len(unc), dtype=bool)
    top_unc[order[-n_tail:]] = True
    bottom_unc[order[:n_tail]] = True
    unique_error = np.unique(err[np.isfinite(err)])
    if len(unique_error) <= 2 and set(np.round(unique_error, 6).tolist()).issubset({0.0, 1.0}):
        top_err = err > 0.0
    else:
        q_err = float(np.quantile(err, quantile))
        top_err = err >= q_err
    global_error = float(np.mean(err))
    top_error_mean = float(np.mean(err[top_unc])) if np.any(top_unc) else float("nan")
    bottom_error_mean = float(np.mean(err[bottom_unc])) if np.any(bottom_unc) else float("nan")
    return {
        "n_voxels": float(len(unc)),
        "top_uncertainty_fraction": float(np.mean(top_unc)),
        "bottom_uncertainty_fraction": float(np.mean(bottom_unc)),
        "global_error_mean": global_error,
        "bottom_uncertainty_error_mean": bottom_error_mean,
        "top_uncertainty_error_mean": top_error_mean,
        "top_uncertainty_error_enrichment": float(top_error_mean / global_error) if global_error > 0 else float("nan"),
        "bottom_uncertainty_error_ratio": float(bottom_error_mean / global_error) if global_error > 0 else float("nan"),
        "top_uncertainty_captures_top_error_rate": float(np.mean(top_unc[top_err])) if np.any(top_err) else float("nan"),
    }


def continuous_alignment_rows(
    model: str,
    prediction_path: Path,
    posterior: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    quantile: float = 0.90,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for spec in CONTINUOUS_TARGETS:
        if spec["mean_key"] not in posterior or spec["std_key"] not in posterior or spec["truth_key"] not in truth:
            continue
        pred = np.asarray(posterior[spec["mean_key"]], dtype=np.float32)
        uncertainty = np.asarray(posterior[spec["std_key"]], dtype=np.float32)
        target_truth = _as_log_resistivity(truth[spec["truth_key"]]) if spec["target"] == "log_resistivity" else np.asarray(truth[spec["truth_key"]], dtype=np.float32)
        abs_error = np.abs(pred - target_truth).astype(np.float32)
        metrics = top_uncertainty_metrics(uncertainty, abs_error, quantile=quantile)
        rows.append(
            {
                "model": model,
                "prediction_path": prediction_path.as_posix(),
                "target": spec["target"],
                "kind": "continuous",
                "uncertainty_measure": spec["std_key"],
                "error_measure": "absolute_error",
                "quantile": float(quantile),
                "spearman_uncertainty_error": rank_correlation(uncertainty, abs_error),
                **metrics,
            }
        )
    return rows


def facies_alignment_row(
    model: str,
    prediction_path: Path,
    posterior: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    quantile: float = 0.90,
) -> dict[str, float | str] | None:
    if "facies_probability" not in posterior or "facies" not in truth:
        return None
    probability = np.asarray(posterior["facies_probability"], dtype=np.float32)
    mode = np.asarray(posterior.get("facies_mode", np.argmax(probability, axis=-1)), dtype=np.int16)
    uncertainty = facies_entropy(probability).astype(np.float32)
    error = (mode != np.asarray(truth["facies"], dtype=np.int16)).astype(np.float32)
    metrics = top_uncertainty_metrics(uncertainty, error, quantile=quantile)
    return {
        "model": model,
        "prediction_path": prediction_path.as_posix(),
        "target": "facies",
        "kind": "categorical",
        "uncertainty_measure": "facies_entropy",
        "error_measure": "misclassification",
        "quantile": float(quantile),
        "spearman_uncertainty_error": rank_correlation(uncertainty, error),
        **metrics,
    }


def build_posterior_uncertainty_alignment(
    prediction_dir: Path,
    truth: dict[str, np.ndarray],
    models: Iterable[PosteriorAlignmentModel] = DEFAULT_POSTERIORS,
    quantile: float = 0.90,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for item in models:
        prediction_path = prediction_dir / item.prediction_path
        if not prediction_path.exists():
            continue
        with np.load(prediction_path, allow_pickle=False) as data:
            posterior = {key: data[key] for key in data.files}
            rows.extend(continuous_alignment_rows(item.model, prediction_path, posterior, truth, quantile=quantile))
            facies_row = facies_alignment_row(item.model, prediction_path, posterior, truth, quantile=quantile)
            if facies_row is not None:
                rows.append(facies_row)
    return pd.DataFrame(rows)
