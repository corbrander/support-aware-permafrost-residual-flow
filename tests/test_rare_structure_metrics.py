from __future__ import annotations

import numpy as np

from cold_recon.evaluation.rare_structure_metrics import (
    altered_coupling_retention,
    binary_event_metrics,
    high_eic_object_metrics,
)


def test_object_metrics_detect_matching_high_eic_body() -> None:
    truth = np.zeros((8, 8, 6), dtype=np.float32)
    truth[2:6, 2:6, 1:4] = 0.5
    probability = np.zeros_like(truth)
    probability[2:6, 2:6, 1:4] = 0.9
    metrics = high_eic_object_metrics(probability, truth, minimum_voxels=4, dz=0.25)
    assert metrics["object_f1"] == 1.0
    assert metrics["mean_matched_iou"] == 1.0
    assert altered_coupling_retention(0.8, 0.4) == 0.5


def test_object_metrics_count_zero_matches_as_zero_f1() -> None:
    truth = np.zeros((8, 8, 6), dtype=np.float32)
    truth[1:4, 1:4, 1:4] = 0.5
    probability = np.zeros_like(truth)
    probability[5:8, 5:8, 1:4] = 0.9
    metrics = high_eic_object_metrics(
        probability,
        truth,
        minimum_voxels=4,
        dz=0.25,
    )
    assert metrics["truth_object_count"] == 1.0
    assert metrics["predicted_object_count"] == 1.0
    assert metrics["matched_object_count"] == 0.0
    assert metrics["object_precision"] == 0.0
    assert metrics["object_recall"] == 0.0
    assert metrics["object_f1"] == 0.0


def test_binary_event_metrics_reports_calibration_and_detection() -> None:
    probability = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    truth = np.asarray([1, 1, 0, 0], dtype=bool)
    metrics = binary_event_metrics(probability, truth)
    assert metrics["f1"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["brier"] < 0.05
