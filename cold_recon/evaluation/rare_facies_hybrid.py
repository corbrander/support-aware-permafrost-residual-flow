from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cold_recon.evaluation.metrics import facies_iou, synthetic_metrics
from cold_recon.evaluation.physics_consistency import facies_to_probability
from cold_recon.evaluation.rare_cryostructure import binary_event_metrics


def _copy_posterior(posterior: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in posterior.items()}


def rare_facies_gate_mask(
    rare_source_facies: np.ndarray,
    eic_mean: np.ndarray,
    rare_class: int = 6,
    eic_floor: float = 0.10,
) -> np.ndarray:
    source = np.asarray(rare_source_facies, dtype=np.int16)
    eic = np.asarray(eic_mean, dtype=np.float32)
    if source.shape != eic.shape:
        raise ValueError("rare source facies and eic_mean must have the same shape")
    return (source == int(rare_class)) & np.isfinite(eic) & (eic >= float(eic_floor))


def apply_rare_facies_hybrid(
    posterior: dict[str, np.ndarray],
    rare_source_facies: np.ndarray,
    n_facies: int = 7,
    rare_class: int = 6,
    eic_floor: float = 0.10,
    gate_probability: float = 0.95,
) -> dict[str, np.ndarray]:
    if "eic_mean" not in posterior:
        raise KeyError("posterior must contain eic_mean")
    out = _copy_posterior(posterior)
    mask = rare_facies_gate_mask(rare_source_facies, out["eic_mean"], rare_class=rare_class, eic_floor=eic_floor)
    if "facies_probability" in out:
        probs = np.asarray(out["facies_probability"], dtype=np.float32).copy()
    elif "facies_mode" in out:
        probs = facies_to_probability(out["facies_mode"], n_facies=n_facies)
    elif "facies" in out:
        probs = facies_to_probability(out["facies"], n_facies=n_facies)
    else:
        raise KeyError("posterior must contain facies_probability, facies_mode, or facies")

    if np.any(mask):
        other = probs[mask] * max(0.0, 1.0 - float(gate_probability))
        other[:, int(rare_class)] = float(gate_probability)
        other_sum = np.sum(other, axis=-1, keepdims=True)
        probs[mask] = other / np.clip(other_sum, 1e-8, None)
        out["facies_probability"] = probs.astype(np.float32)
        out["facies_mode"] = np.argmax(probs, axis=-1).astype(np.int16)
        out["facies"] = out["facies_mode"].astype(np.int16)
        if "facies_samples" in out:
            samples = np.asarray(out["facies_samples"], dtype=np.int16).copy()
            samples[:, mask] = int(rare_class)
            out["facies_samples"] = samples
    else:
        out["facies_probability"] = probs.astype(np.float32)
        out["facies_mode"] = np.argmax(probs, axis=-1).astype(np.int16)
        out["facies"] = out["facies_mode"].astype(np.int16)

    out["rare_facies_hybrid_mask"] = mask.astype(np.uint8)
    out["rare_facies_hybrid_eic_floor"] = np.asarray(float(eic_floor), dtype=np.float32)
    out["rare_facies_hybrid_gate_probability"] = np.asarray(float(gate_probability), dtype=np.float32)
    out["rare_facies_hybrid_rare_class"] = np.asarray(int(rare_class), dtype=np.int16)
    return out


def rare_facies_hybrid_metrics(
    hybrid: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    z: np.ndarray,
    model_name: str,
    n_facies: int = 7,
    rare_class: int = 6,
    ice_threshold: float = 0.30,
) -> dict[str, float | str]:
    pred = {
        "facies": np.asarray(hybrid["facies_mode"] if "facies_mode" in hybrid else hybrid["facies"], dtype=np.int16),
        "eic": np.asarray(hybrid["eic_mean"], dtype=np.float32),
        "temperature": np.asarray(hybrid["temperature_mean"], dtype=np.float32),
        "unfrozen_water": np.asarray(hybrid["unfrozen_water_mean"], dtype=np.float32),
        "log_resistivity": np.asarray(hybrid["log_resistivity_mean"], dtype=np.float32),
    }
    metrics: dict[str, float | str] = {"model": model_name}
    metrics.update(synthetic_metrics(pred, truth, z, n_facies=n_facies, ice_threshold=ice_threshold))
    truth_facies = np.asarray(truth["facies"], dtype=np.int16)
    wedge = binary_event_metrics(truth_facies == int(rare_class), pred["facies"] == int(rare_class))
    metrics["wedge_ice_recall"] = wedge["recall"]
    metrics["wedge_ice_precision"] = wedge["precision"]
    metrics["wedge_ice_f1"] = wedge["f1"]
    metrics["wedge_ice_predicted_rate"] = wedge["predicted_rate"]
    return metrics


def rare_facies_hybrid_operating_curve(
    posterior: dict[str, np.ndarray],
    rare_source_facies: np.ndarray,
    truth: dict[str, np.ndarray],
    z: np.ndarray,
    eic_floors: list[float] | tuple[float, ...],
    n_facies: int = 7,
    rare_class: int = 6,
    gate_probability: float = 0.95,
    ice_threshold: float = 0.30,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if "facies_mode" in posterior:
        base_mode = np.asarray(posterior["facies_mode"], dtype=np.int16)
    elif "facies" in posterior:
        base_mode = np.asarray(posterior["facies"], dtype=np.int16)
    elif "facies_probability" in posterior:
        base_mode = np.argmax(np.asarray(posterior["facies_probability"], dtype=np.float32), axis=-1).astype(np.int16)
    else:
        raise KeyError("posterior must contain facies_probability, facies_mode, or facies")
    base_iou = facies_iou(base_mode, truth["facies"], n_classes=n_facies)
    for floor in eic_floors:
        hybrid = apply_rare_facies_hybrid(
            posterior,
            rare_source_facies,
            n_facies=n_facies,
            rare_class=rare_class,
            eic_floor=float(floor),
            gate_probability=gate_probability,
        )
        metrics = rare_facies_hybrid_metrics(
            hybrid,
            truth,
            z,
            model_name="COLDReconLatentDiffusionRareFaciesHybrid",
            n_facies=n_facies,
            rare_class=rare_class,
            ice_threshold=ice_threshold,
        )
        mask = np.asarray(hybrid["rare_facies_hybrid_mask"], dtype=bool)
        rows.append(
            {
                "eic_floor": float(floor),
                "gate_probability": float(gate_probability),
                "rare_class": float(rare_class),
                "gate_fraction": float(np.mean(mask)),
                "mean_iou": float(metrics["mean_iou"]),
                "mean_iou_delta_vs_base": float(metrics["mean_iou"]) - float(base_iou["mean_iou"]),
                "wedge_ice_iou": float(metrics.get("iou_6", np.nan)),
                "wedge_ice_recall": float(metrics["wedge_ice_recall"]),
                "wedge_ice_precision": float(metrics["wedge_ice_precision"]) if np.isfinite(metrics["wedge_ice_precision"]) else np.nan,
                "wedge_ice_f1": float(metrics["wedge_ice_f1"]) if np.isfinite(metrics["wedge_ice_f1"]) else np.nan,
                "wedge_ice_predicted_rate": float(metrics["wedge_ice_predicted_rate"]),
            }
        )
    return pd.DataFrame(rows)


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
