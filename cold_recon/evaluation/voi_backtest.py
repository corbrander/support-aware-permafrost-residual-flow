from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PANDAS_FUTURE_INFER_STRING", "0")
import pandas as pd

pd.set_option("future.infer_string", False)

from cold_recon.data.data_schema import ObservationTable
from cold_recon.evaluation.site_investigation import (
    VOIWeights,
    build_voi_score,
    posterior_score_components,
    recommend_boreholes,
)


@dataclass(frozen=True)
class VOIBacktestResult:
    audit: pd.DataFrame
    summary: dict[str, Any]


def _robust_normalize(values: np.ndarray, lower_q: float = 5.0, upper_q: float = 95.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.nanpercentile(arr[finite], lower_q))
    high = float(np.nanpercentile(arr[finite], upper_q))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _upper_mask(z: np.ndarray, max_depth: float) -> np.ndarray:
    mask = np.asarray(z, dtype=np.float32) <= float(max_depth)
    if not np.any(mask):
        mask[0] = True
    return mask


def _surface_mean(volume: np.ndarray, z: np.ndarray, max_depth: float) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError("Expected a 2D surface or 3D volume")
    return np.nanmean(arr[:, :, _upper_mask(z, max_depth)], axis=2).astype(np.float32)


def _surface_max(volume: np.ndarray, z: np.ndarray, max_depth: float) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError("Expected a 2D surface or 3D volume")
    return np.nanmax(arr[:, :, _upper_mask(z, max_depth)], axis=2).astype(np.float32)


def _valid_flatten(score: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = np.isfinite(s) & np.isfinite(v)
    return s[valid], v[valid]


def _spearman(score: np.ndarray, values: np.ndarray) -> float:
    s, v = _valid_flatten(score, values)
    if len(s) < 2 or float(np.nanstd(s)) == 0.0 or float(np.nanstd(v)) == 0.0:
        return float("nan")
    return float(pd.Series(s).corr(pd.Series(v), method="spearman"))


def _tail_metrics(score: np.ndarray, error: np.ndarray, quantile: float) -> dict[str, float]:
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0")
    s, e = _valid_flatten(score, error)
    if len(s) == 0:
        return {
            "n_cells": 0.0,
            "global_error_mean": float("nan"),
            "top_voi_error_mean": float("nan"),
            "bottom_voi_error_mean": float("nan"),
            "top_voi_error_enrichment": float("nan"),
            "bottom_voi_error_ratio": float("nan"),
            "top_voi_captures_top_error_rate": float("nan"),
        }
    n_tail = max(1, int(np.ceil((1.0 - float(quantile)) * len(s))))
    order = np.argsort(s, kind="mergesort")
    top = np.zeros(len(s), dtype=bool)
    bottom = np.zeros(len(s), dtype=bool)
    top[order[-n_tail:]] = True
    bottom[order[:n_tail]] = True

    unique = np.unique(e[np.isfinite(e)])
    if len(unique) <= 2 and set(np.round(unique, 6).tolist()).issubset({0.0, 1.0}):
        high_error = e > 0.0
    else:
        high_error = e >= float(np.quantile(e, quantile))
    global_mean = float(np.mean(e))
    top_mean = float(np.mean(e[top])) if np.any(top) else float("nan")
    bottom_mean = float(np.mean(e[bottom])) if np.any(bottom) else float("nan")
    return {
        "n_cells": float(len(s)),
        "tail_fraction": float(n_tail / len(s)),
        "global_error_mean": global_mean,
        "top_voi_error_mean": top_mean,
        "bottom_voi_error_mean": bottom_mean,
        "top_voi_error_enrichment": float(top_mean / global_mean) if global_mean > 0 else float("nan"),
        "bottom_voi_error_ratio": float(bottom_mean / global_mean) if global_mean > 0 else float("nan"),
        "top_voi_captures_top_error_rate": float(np.mean(top[high_error])) if np.any(high_error) else float("nan"),
    }


def _target_error_surfaces(
    posterior: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    max_depth: float,
    high_eic_threshold: float,
    wedge_class_id: int,
) -> dict[str, np.ndarray]:
    z = np.asarray(posterior["grid_z"], dtype=np.float32)
    if "eic_mean" not in posterior or "eic" not in truth:
        raise KeyError("VOI backtest requires posterior['eic_mean'] and truth['eic']")
    if "facies_mode" in posterior:
        facies_pred = np.asarray(posterior["facies_mode"], dtype=np.int16)
    elif "facies_probability" in posterior:
        facies_pred = np.argmax(np.asarray(posterior["facies_probability"], dtype=np.float32), axis=-1).astype(np.int16)
    else:
        raise KeyError("VOI backtest requires facies_mode or facies_probability")
    if "facies" not in truth:
        raise KeyError("VOI backtest requires truth['facies']")

    eic_pred = np.asarray(posterior["eic_mean"], dtype=np.float32)
    eic_truth = np.asarray(truth["eic"], dtype=np.float32)
    facies_truth = np.asarray(truth["facies"], dtype=np.int16)
    mask = _upper_mask(z, max_depth)
    eic_abs_error = np.nanmean(np.abs(eic_pred[:, :, mask] - eic_truth[:, :, mask]), axis=2).astype(np.float32)
    facies_error = np.nanmean((facies_pred[:, :, mask] != facies_truth[:, :, mask]).astype(np.float32), axis=2).astype(np.float32)
    high_eic_mismatch = np.nanmean(
        ((eic_pred[:, :, mask] > float(high_eic_threshold)) != (eic_truth[:, :, mask] > float(high_eic_threshold))).astype(np.float32),
        axis=2,
    ).astype(np.float32)
    wedge_miss = np.nanmax(
        ((facies_truth[:, :, mask] == int(wedge_class_id)) & (facies_pred[:, :, mask] != int(wedge_class_id))).astype(np.float32),
        axis=2,
    ).astype(np.float32)
    composite_error = (
        0.45 * _robust_normalize(eic_abs_error)
        + 0.25 * _robust_normalize(facies_error)
        + 0.20 * _robust_normalize(high_eic_mismatch)
        + 0.10 * wedge_miss
    ).astype(np.float32)
    return {
        "eic_abs_error": eic_abs_error,
        "facies_error": facies_error,
        "high_eic_mismatch": high_eic_mismatch,
        "wedge_miss": wedge_miss,
        "composite_error": composite_error,
    }


def build_voi_backtest(
    posterior: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    observations: ObservationTable | None = None,
    max_depth: float = 3.0,
    exclusion_radius: float = 3.0,
    min_spacing: float = 8.0,
    top_k: int = 8,
    quantile: float = 0.90,
    high_eic_threshold: float = 0.30,
    wedge_class_id: int = 6,
    weights: VOIWeights | None = None,
    model: str = "COLDReconLatentDiffusionPhysicsTrained",
) -> VOIBacktestResult:
    components = posterior_score_components(
        posterior,
        observations=observations,
        max_depth=float(max_depth),
        exclusion_radius=float(exclusion_radius),
    )
    score = build_voi_score(components, weights=weights)
    errors = _target_error_surfaces(
        posterior,
        truth,
        max_depth=float(max_depth),
        high_eic_threshold=float(high_eic_threshold),
        wedge_class_id=int(wedge_class_id),
    )

    rows: list[dict[str, object]] = []
    for target, error in errors.items():
        metrics = _tail_metrics(score, error, quantile=float(quantile))
        rows.append(
            {
                "record_type": "target_metric",
                "model": model,
                "target": target,
                "score_name": "fixed_weight_voi",
                "quantile": float(quantile),
                "spearman_voi_error": _spearman(score, error),
                **metrics,
            }
        )

    composite = errors["composite_error"]
    predictor_map = {
        "voi_score": score,
        "uncertainty": components["uncertainty"],
        "ice_rich_ambiguity": components["ice_rich_ambiguity"],
        "thaw_sensitive_eic_proxy": components["settlement_risk"],
        "eic_gradient_proxy": components["differential_settlement"],
        "novelty": components["novelty"],
    }
    for predictor, values in predictor_map.items():
        rows.append(
            {
                "record_type": "component_correlation",
                "model": model,
                "predictor": predictor,
                "target": "composite_error",
                "spearman_predictor_error": _spearman(values, composite),
            }
        )

    boreholes = recommend_boreholes(
        score,
        posterior,
        components,
        top_k=int(top_k),
        min_spacing=float(min_spacing),
        max_depth=float(max_depth),
    )
    x = np.asarray(posterior["grid_x"], dtype=np.float32)
    y = np.asarray(posterior["grid_y"], dtype=np.float32)
    for row in boreholes:
        ix = int(np.argmin(np.abs(x - float(row["x"]))))
        iy = int(np.argmin(np.abs(y - float(row["y"]))))
        rows.append(
            {
                "record_type": "selected_borehole",
                "model": model,
                "rank": int(row["rank"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "recommended_depth_m": float(row["recommended_depth_m"]),
                "voi_score": float(row["voi_score"]),
                "composite_error": float(errors["composite_error"][ix, iy]),
                "eic_abs_error": float(errors["eic_abs_error"][ix, iy]),
                "facies_error": float(errors["facies_error"][ix, iy]),
                "high_eic_mismatch": float(errors["high_eic_mismatch"][ix, iy]),
                "wedge_miss": float(errors["wedge_miss"][ix, iy]),
            }
        )

    audit = pd.DataFrame.from_records(rows)
    for column in ("record_type", "model", "target", "score_name", "predictor"):
        if column in audit.columns:
            audit[column] = audit[column].astype(object)
    metric_rows = audit[audit["record_type"].eq("target_metric")]
    composite_row = metric_rows[metric_rows["target"].eq("composite_error")]
    high_eic_row = metric_rows[metric_rows["target"].eq("high_eic_mismatch")]
    composite_enrichment = float(composite_row["top_voi_error_enrichment"].iloc[0]) if not composite_row.empty else float("nan")
    high_eic_enrichment = float(high_eic_row["top_voi_error_enrichment"].iloc[0]) if not high_eic_row.empty else float("nan")
    composite_rho = float(composite_row["spearman_voi_error"].iloc[0]) if not composite_row.empty else float("nan")
    readiness = (
        "conditional"
        if np.isfinite(composite_enrichment)
        and composite_enrichment >= 1.25
        and np.isfinite(high_eic_enrichment)
        and high_eic_enrichment >= 1.20
        and np.isfinite(composite_rho)
        and composite_rho > 0.0
        else "not_yet"
    )
    summary = {
        "model": model,
        "audit_scope": "synthetic_retrospective_full_field_truth",
        "max_depth_m": float(max_depth),
        "quantile": float(quantile),
        "top_k_boreholes": int(len(boreholes)),
        "composite_top_voi_error_enrichment": composite_enrichment,
        "high_eic_top_voi_error_enrichment": high_eic_enrichment,
        "composite_spearman_voi_error": composite_rho,
        "readiness_status": readiness,
        "readiness_score": 0.5 if readiness == "conditional" else 0.0,
        "readiness_boundary": (
            "The VOI score is retrospectively supported under synthetic full-field truth, "
            "but this is not a prospective field acquisition validation."
        ),
    }
    return VOIBacktestResult(audit=audit, summary=summary)


def write_voi_backtest_outputs(
    result: VOIBacktestResult,
    table_dir: Path,
    summary_path: Path | None = None,
) -> tuple[Path, Path]:
    import json

    table_dir.mkdir(parents=True, exist_ok=True)
    audit_path = table_dir / "voi_backtest_audit.csv"
    result.audit.to_csv(audit_path, index=False)
    out_summary = summary_path or table_dir / "voi_backtest_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path, out_summary
