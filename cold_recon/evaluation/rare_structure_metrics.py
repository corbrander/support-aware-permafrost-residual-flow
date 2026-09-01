from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score


def binary_event_metrics(
    predicted_probability: np.ndarray,
    truth_event: np.ndarray,
    *,
    probability_threshold: float = 0.50,
) -> dict[str, float]:
    probability = np.asarray(predicted_probability, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth_event, dtype=bool).reshape(-1)
    finite = np.isfinite(probability)
    probability = np.clip(probability[finite], 0.0, 1.0)
    truth = truth[finite]
    if len(probability) == 0:
        return {
            "brier": float("nan"),
            "auprc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    predicted = probability >= float(probability_threshold)
    true_positive = int(np.sum(predicted & truth))
    false_positive = int(np.sum(predicted & ~truth))
    false_negative = int(np.sum(~predicted & truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    if np.any(truth) and np.any(~truth):
        auprc = float(average_precision_score(truth.astype(np.int8), probability))
    else:
        auprc = float("nan")
    return {
        "brier": float(np.mean((probability - truth.astype(np.float64)) ** 2)),
        "auprc": auprc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _components(mask: np.ndarray, minimum_voxels: int) -> tuple[np.ndarray, list[np.ndarray]]:
    labels, count = label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    components: list[np.ndarray] = []
    relabeled = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for component_id in range(1, count + 1):
        component = labels == component_id
        if int(component.sum()) < int(minimum_voxels):
            continue
        relabeled[component] = next_id
        components.append(component)
        next_id += 1
    return relabeled, components


def high_eic_object_metrics(
    predicted_probability: np.ndarray,
    truth_eic: np.ndarray,
    *,
    eic_threshold: float = 0.30,
    probability_threshold: float = 0.50,
    minimum_voxels: int = 8,
    match_iou: float = 0.10,
    dz: float = 1.0,
) -> dict[str, float]:
    predicted_mask = np.asarray(predicted_probability) >= float(probability_threshold)
    truth_mask = np.asarray(truth_eic) >= float(eic_threshold)
    _, predicted = _components(predicted_mask, minimum_voxels)
    _, truth = _components(truth_mask, minimum_voxels)
    iou = np.zeros((len(truth), len(predicted)), dtype=np.float64)
    for truth_index, truth_component in enumerate(truth):
        for predicted_index, predicted_component in enumerate(predicted):
            intersection = np.sum(truth_component & predicted_component)
            union = np.sum(truth_component | predicted_component)
            iou[truth_index, predicted_index] = intersection / union if union else 0.0
    matches: list[float] = []
    if iou.size:
        truth_indices, predicted_indices = linear_sum_assignment(1.0 - iou)
        matches = [
            float(iou[t, p])
            for t, p in zip(truth_indices, predicted_indices, strict=True)
            if iou[t, p] >= float(match_iou)
        ]
    true_positive = len(matches)
    precision = (
        true_positive / len(predicted)
        if predicted
        else (0.0 if truth else float("nan"))
    )
    recall = true_positive / len(truth) if truth else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0.0
        else (0.0 if np.isfinite(precision) and np.isfinite(recall) else float("nan"))
    )

    def mean_thickness(components: list[np.ndarray]) -> float:
        values: list[float] = []
        for component in components:
            columns = np.sum(component, axis=2)
            occupied = columns > 0
            if np.any(occupied):
                values.append(float(np.mean(columns[occupied]) * float(dz)))
        return float(np.mean(values)) if values else float("nan")

    truth_volume = float(np.sum(truth_mask))
    predicted_volume = float(np.sum(predicted_mask))
    return {
        "truth_object_count": float(len(truth)),
        "predicted_object_count": float(len(predicted)),
        "matched_object_count": float(true_positive),
        "object_precision": float(precision),
        "object_recall": float(recall),
        "object_f1": float(f1),
        "mean_matched_iou": float(np.mean(matches)) if matches else float("nan"),
        "object_count_absolute_error": float(abs(len(predicted) - len(truth))),
        "predicted_to_truth_volume_ratio": predicted_volume / truth_volume if truth_volume > 0 else float("nan"),
        "truth_mean_thickness_m": mean_thickness(truth),
        "predicted_mean_thickness_m": mean_thickness(predicted),
    }


def altered_coupling_retention(id_metric: float, altered_metric: float) -> float:
    if not np.isfinite(id_metric) or float(id_metric) <= 0.0:
        return float("nan")
    return float(altered_metric) / float(id_metric)
