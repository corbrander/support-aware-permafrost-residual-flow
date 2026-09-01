from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cold_recon.data.data_schema import FACIES_NAMES, OBS_TYPES, ObservationTable


DEFAULT_PREDICTIONS: tuple[tuple[str, str], ...] = (
    ("IDW", "baseline_idw.npz"),
    ("RandomForest", "baseline_random_forest.npz"),
    ("GradientBoosting", "baseline_gradient_boosting.npz"),
    ("KrigingGPR", "baseline_kriging.npz"),
    ("SparseUNet3D", "baseline_unet3d.npz"),
    ("COLDReconImplicit", "implicit_prediction.npz"),
    ("COLDReconLatentDiffusion", "diffusion_posterior.npz"),
    ("COLDReconFNOOperatorDiffusion", "fno_operator_diffusion_posterior.npz"),
    ("COLDReconRectifiedFlow", "rectified_flow_posterior.npz"),
    ("COLDReconLatentDiffusionPhysicsTrained", "diffusion_posterior_physics_trained.npz"),
    ("COLDReconLatentDiffusionRareFaciesHybrid", "diffusion_posterior_rare_facies_hybrid.npz"),
    ("COLDReconLatentDiffusionPhysicsGuided", "diffusion_posterior_physics_guided.npz"),
    ("COLDReconLatentDiffusionPhysicsRefined", "diffusion_posterior_physics_refined.npz"),
)


def binary_event_metrics(observed: np.ndarray, predicted: np.ndarray, beta: float = 2.0) -> dict[str, float]:
    obs = np.asarray(observed, dtype=bool)
    pred = np.asarray(predicted, dtype=bool)
    if obs.shape != pred.shape:
        raise ValueError("observed and predicted event masks must have the same shape")
    tp = int(np.sum(obs & pred))
    fp = int(np.sum(~obs & pred))
    fn = int(np.sum(obs & ~pred))
    tn = int(np.sum(~obs & ~pred))
    n = int(obs.size)
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0.0 else np.nan
    fbeta = (
        (1.0 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and beta * beta * precision + recall > 0.0
        else np.nan
    )
    union = int(np.sum(obs | pred))
    iou = tp / union if union else np.nan
    return {
        "n_voxels": float(n),
        "truth_rate": float(np.mean(obs)) if n else np.nan,
        "predicted_rate": float(np.mean(pred)) if n else np.nan,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision) if np.isfinite(precision) else np.nan,
        "recall": float(recall) if np.isfinite(recall) else np.nan,
        "specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        "false_positive_rate": float(fpr) if np.isfinite(fpr) else np.nan,
        "f1": float(f1) if np.isfinite(f1) else np.nan,
        "f2": float(fbeta) if np.isfinite(fbeta) else np.nan,
        "iou": float(iou) if np.isfinite(iou) else np.nan,
    }


def _posterior_field(data: Any, name: str) -> np.ndarray | None:
    if name in data.files:
        return np.asarray(data[name])
    mean_name = f"{name}_mean"
    if mean_name in data.files:
        return np.asarray(data[mean_name])
    return None


def _facies_mode(data: Any) -> np.ndarray | None:
    if "facies_mode" in data.files:
        return np.asarray(data["facies_mode"])
    if "facies" in data.files:
        return np.asarray(data["facies"])
    if "facies_probability" in data.files:
        return np.argmax(np.asarray(data["facies_probability"]), axis=-1)
    if "facies_samples" in data.files:
        samples = np.asarray(data["facies_samples"]).astype(np.int16)
        n_classes = int(np.nanmax(samples)) + 1
        counts = np.stack([(samples == cls).sum(axis=0) for cls in range(n_classes)], axis=-1)
        return np.argmax(counts, axis=-1)
    return None


def observation_eic_event_rate(observations: ObservationTable, threshold: float) -> float:
    mask = observations.type_ids == OBS_TYPES["borehole_eic"]
    if not np.any(mask):
        return np.nan
    values = np.asarray(observations.values[mask], dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.mean(values[finite] > float(threshold)))


def rate_constrained_threshold(score: np.ndarray, target_rate: float) -> float:
    values = np.asarray(score, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    rate = float(np.clip(target_rate, 0.0, 1.0))
    if rate <= 0.0:
        return float(np.nanmax(values[finite]) + 1.0)
    if rate >= 1.0:
        return float(np.nanmin(values[finite]))
    return float(np.nanquantile(values[finite], 1.0 - rate))


def audit_prediction(
    model: str,
    prediction_path: Path,
    truth: dict[str, np.ndarray],
    observations: ObservationTable,
    eic_threshold: float = 0.30,
    observation_rate_multiplier: float = 2.0,
) -> dict[str, float | str]:
    data = np.load(prediction_path, allow_pickle=False)
    truth_eic_event = np.asarray(truth["eic"], dtype=float) > float(eic_threshold)
    observed_eic_rate = observation_eic_event_rate(observations, float(eic_threshold))
    target_rate = float(np.clip(observed_eic_rate * float(observation_rate_multiplier), 0.0, 1.0)) if np.isfinite(observed_eic_rate) else np.nan

    row: dict[str, float | str] = {
        "model": model,
        "prediction_path": prediction_path.as_posix(),
        "observed_eic_event_rate": observed_eic_rate,
        "observation_rate_multiplier": float(observation_rate_multiplier),
        "target_eic_event_rate": target_rate,
    }
    eic_score = _posterior_field(data, "eic")
    if eic_score is not None:
        raw_eic_event = np.asarray(eic_score, dtype=float) > float(eic_threshold)
        raw = binary_event_metrics(truth_eic_event, raw_eic_event)
        row.update({f"raw_eic_{key}": value for key, value in raw.items()})
        threshold = rate_constrained_threshold(eic_score, target_rate) if np.isfinite(target_rate) else np.nan
        calibrated_eic_event = np.asarray(eic_score, dtype=float) >= threshold if np.isfinite(threshold) else np.zeros_like(truth_eic_event, dtype=bool)
        calibrated = binary_event_metrics(truth_eic_event, calibrated_eic_event)
        row["rate_constrained_eic_threshold"] = threshold
        row.update({f"rate_constrained_eic_{key}": value for key, value in calibrated.items()})

    facies = _facies_mode(data)
    if facies is not None:
        truth_facies = np.asarray(truth["facies"]).astype(np.int16)
        pred_facies = np.asarray(facies).astype(np.int16)
        rare_truth = np.isin(truth_facies, [3, 6])
        rare_pred = np.isin(pred_facies, [3, 6])
        rare = binary_event_metrics(rare_truth, rare_pred)
        row.update({f"rare_facies_{key}": value for key, value in rare.items()})
        for cls in (3, 6):
            metrics = binary_event_metrics(truth_facies == cls, pred_facies == cls)
            prefix = f"facies_{cls}_{FACIES_NAMES.get(cls, str(cls))}"
            row[f"{prefix}_support_rate"] = metrics["truth_rate"]
            row[f"{prefix}_predicted_rate"] = metrics["predicted_rate"]
            row[f"{prefix}_recall"] = metrics["recall"]
            row[f"{prefix}_precision"] = metrics["precision"]
            row[f"{prefix}_iou"] = metrics["iou"]
    return row


def build_rare_cryostructure_audit(
    prediction_dir: Path,
    truth: dict[str, np.ndarray],
    observations: ObservationTable,
    eic_threshold: float = 0.30,
    observation_rate_multiplier: float = 2.0,
    model_paths: tuple[tuple[str, str], ...] = DEFAULT_PREDICTIONS,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model, filename in model_paths:
        path = prediction_dir / filename
        if not path.exists():
            continue
        rows.append(
            audit_prediction(
                model,
                path,
                truth,
                observations,
                eic_threshold=eic_threshold,
                observation_rate_multiplier=observation_rate_multiplier,
            )
        )
    return pd.DataFrame(rows)
