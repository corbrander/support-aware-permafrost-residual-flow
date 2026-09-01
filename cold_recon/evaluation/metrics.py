from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))


def facies_iou(pred: np.ndarray, target: np.ndarray, n_classes: int) -> dict[str, float]:
    scores = {}
    vals = []
    for cls in range(n_classes):
        p = pred == cls
        t = target == cls
        union = np.logical_or(p, t).sum()
        inter = np.logical_and(p, t).sum()
        score = float(inter / union) if union else float("nan")
        scores[f"iou_{cls}"] = score
        if not np.isnan(score):
            vals.append(score)
    scores["mean_iou"] = float(np.mean(vals)) if vals else float("nan")
    return scores


def ice_rich_recall(pred_eic: np.ndarray, true_eic: np.ndarray, threshold: float = 0.30) -> float:
    true = true_eic > threshold
    pred = pred_eic > threshold
    denom = true.sum()
    return float(np.logical_and(pred, true).sum() / denom) if denom else float("nan")


def alt_from_temperature(temperature: np.ndarray, z: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    thawed = temperature > threshold
    alt = np.zeros(temperature.shape[:2], dtype=np.float32)
    for i in range(temperature.shape[0]):
        for j in range(temperature.shape[1]):
            idx = np.where(thawed[i, j])[0]
            alt[i, j] = float(z[idx[-1]]) if len(idx) else 0.0
    return alt


def synthetic_metrics(pred: dict[str, np.ndarray], truth: dict[str, np.ndarray], z: np.ndarray, n_facies: int = 7, ice_threshold: float = 0.30) -> dict[str, float]:
    out: dict[str, float] = {}
    if "facies" in pred:
        out.update(facies_iou(pred["facies"], truth["facies"], n_facies))
    if "eic" in pred:
        out["eic_rmse"] = rmse(pred["eic"], truth["eic"])
        out["ice_rich_recall"] = ice_rich_recall(pred["eic"], truth["eic"], ice_threshold)
    if "temperature" in pred:
        out["temperature_rmse"] = rmse(pred["temperature"], truth["temperature"])
        out["alt_mae"] = mae(alt_from_temperature(pred["temperature"], z), alt_from_temperature(truth["temperature"], z))
    if "unfrozen_water" in pred:
        out["unfrozen_water_rmse"] = rmse(pred["unfrozen_water"], truth["unfrozen_water"])
    if "log_resistivity" in pred:
        out["log_resistivity_rmse"] = rmse(pred["log_resistivity"], np.log(np.maximum(truth["resistivity"], 1.0)))
    return out

