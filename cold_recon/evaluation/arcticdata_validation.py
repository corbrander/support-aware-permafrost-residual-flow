from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from cold_recon.data.arcticdata_cryostratigraphy_loader import _local_xy_from_inventory


@dataclass(frozen=True)
class ArcticDataValidationConfig:
    horizontal_scale_m: float = 50.0
    depth_scale_m: float = 0.25
    idw_k: int = 8
    knn_k: int = 12
    high_eic_threshold: float = 0.30
    rare_class_weight_power: float = 0.5
    cryofacies_blend_weight: float = 0.35


def prepare_arcticdata_validation_intervals(
    inventory: pd.DataFrame,
    config: ArcticDataValidationConfig | None = None,
) -> pd.DataFrame:
    _ = config or ArcticDataValidationConfig()
    df = _local_xy_from_inventory(inventory).copy()
    df["borehole_group"] = df["site"].astype(str) + "::" + df["borehole"].astype(str)
    df["depth_mid_m"] = (df["sample_depth_cm_mid"].where(df["sample_depth_cm_mid"].notna(), df["unit_depth_cm_mid"]) / 100.0).astype(float)
    df["unit_depth_mid_m"] = (df["unit_depth_cm_mid"] / 100.0).astype(float)
    return df


def _feature_matrix(df: pd.DataFrame, depth_col: str, cfg: ArcticDataValidationConfig) -> np.ndarray:
    return np.column_stack(
        [
            df["local_x_m"].to_numpy(dtype=np.float32) / float(cfg.horizontal_scale_m),
            df["local_y_m"].to_numpy(dtype=np.float32) / float(cfg.horizontal_scale_m),
            df[depth_col].to_numpy(dtype=np.float32) / float(cfg.depth_scale_m),
        ]
    ).astype(np.float32)


def _idw_predict(train_features: np.ndarray, train_values: np.ndarray, query_features: np.ndarray, k: int) -> np.ndarray:
    k = min(max(1, int(k)), len(train_values))
    dist, idx = cKDTree(train_features).query(query_features, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.square(dist + 1e-6)
    return ((weights * train_values[idx]).sum(axis=1) / weights.sum(axis=1)).astype(np.float32)


def leave_one_borehole_eic_predictions(
    inventory: pd.DataFrame,
    config: ArcticDataValidationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ArcticDataValidationConfig()
    intervals = prepare_arcticdata_validation_intervals(inventory, cfg)
    intervals = intervals[intervals["eic_fraction"].notna() & intervals["depth_mid_m"].notna()].copy()
    if intervals["borehole_group"].nunique() < 2:
        raise ValueError("ArcticData EIC holdout validation requires at least two boreholes")
    rows: list[dict[str, Any]] = []
    for borehole_group in sorted(intervals["borehole_group"].unique()):
        holdout = intervals[intervals["borehole_group"] == borehole_group]
        train = intervals[intervals["borehole_group"] != borehole_group]
        if train.empty or holdout.empty:
            continue
        train_values = train["eic_fraction"].to_numpy(dtype=np.float32)
        global_pred = np.full(len(holdout), float(train_values.mean()), dtype=np.float32)
        site_means = train.groupby("site")["eic_fraction"].mean()
        site_pred = holdout["site"].map(site_means).fillna(float(train_values.mean())).to_numpy(dtype=np.float32)
        spatial_pred = _idw_predict(_feature_matrix(train, "depth_mid_m", cfg), train_values, _feature_matrix(holdout, "depth_mid_m", cfg), cfg.idw_k)
        cryo_means = train.groupby("cryofacies_class")["eic_fraction"].mean()
        cryo_pred = holdout["cryofacies_class"].map(cryo_means).fillna(float(train_values.mean())).to_numpy(dtype=np.float32)
        blend_pred = ((1.0 - cfg.cryofacies_blend_weight) * spatial_pred + cfg.cryofacies_blend_weight * cryo_pred).astype(np.float32)
        for model, pred in [
            ("GlobalMean", global_pred),
            ("SiteMean", site_pred),
            ("SpatialDepthIDW", spatial_pred),
            ("CryofaciesPrior", cryo_pred),
            ("CryoSpatialBlend", blend_pred),
        ]:
            for idx, (_, row) in enumerate(holdout.iterrows()):
                obs = float(row["eic_fraction"])
                value = float(np.clip(pred[idx], 0.0, 1.0))
                rows.append(
                    {
                        "model": model,
                        "site": str(row["site"]),
                        "borehole": str(row["borehole"]),
                        "borehole_group": str(row["borehole_group"]),
                        "source_file": str(row["source_file"]),
                        "depth_mid_m": float(row["depth_mid_m"]),
                        "cryofacies_class": str(row["cryofacies_class"]),
                        "observed_eic": obs,
                        "predicted_eic": value,
                        "error": value - obs,
                        "abs_error": abs(value - obs),
                        "squared_error": (value - obs) ** 2,
                        "train_n": int(len(train)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_arcticdata_eic_predictions(
    predictions: pd.DataFrame,
    config: ArcticDataValidationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ArcticDataValidationConfig()
    rows: list[dict[str, Any]] = []
    for model, group in predictions.groupby("model", sort=False):
        obs = group["observed_eic"].to_numpy(dtype=float)
        pred = group["predicted_eic"].to_numpy(dtype=float)
        err = pred - obs
        obs_event = obs >= cfg.high_eic_threshold
        pred_event = pred >= cfg.high_eic_threshold
        tp = float(np.sum(obs_event & pred_event))
        fp = float(np.sum(~obs_event & pred_event))
        fn = float(np.sum(obs_event & ~pred_event))
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0.0 else np.nan
        rows.append(
            {
                "model": str(model),
                "n": int(len(group)),
                "n_sites": int(group["site"].nunique()),
                "n_boreholes": int(group["borehole_group"].nunique()),
                "observed_mean_eic": float(obs.mean()),
                "predicted_mean_eic": float(pred.mean()),
                "bias": float(err.mean()),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "high_eic_threshold": float(cfg.high_eic_threshold),
                "high_eic_prevalence": float(obs_event.mean()),
                "high_eic_accuracy": float((obs_event == pred_event).mean()),
                "high_eic_precision": float(precision) if np.isfinite(precision) else np.nan,
                "high_eic_recall": float(recall) if np.isfinite(recall) else np.nan,
                "high_eic_f1": float(f1) if np.isfinite(f1) else np.nan,
            }
        )
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["model"] == "GlobalMean", "rmse"]
    metrics["rmse_reduction_vs_global_mean"] = 1.0 - metrics["rmse"] / float(baseline.iloc[0]) if not baseline.empty and float(baseline.iloc[0]) > 0.0 else np.nan
    return metrics


def _weighted_vote(
    train_features: np.ndarray,
    train_classes: np.ndarray,
    query_features: np.ndarray,
    k: int,
    class_weights: dict[int, float] | None = None,
) -> np.ndarray:
    k = min(max(1, int(k)), len(train_classes))
    dist, idx = cKDTree(train_features).query(query_features, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.square(dist + 1e-6)
    pred = np.zeros(query_features.shape[0], dtype=np.int64)
    for row_idx in range(query_features.shape[0]):
        scores: dict[int, float] = {}
        for nbr_idx, weight in zip(idx[row_idx], weights[row_idx]):
            cls = int(train_classes[nbr_idx])
            class_weight = 1.0 if class_weights is None else float(class_weights.get(cls, 1.0))
            scores[cls] = scores.get(cls, 0.0) + float(weight) * class_weight
        pred[row_idx] = max(scores.items(), key=lambda item: (item[1], -item[0]))[0]
    return pred


def leave_one_borehole_facies_predictions(
    inventory: pd.DataFrame,
    config: ArcticDataValidationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ArcticDataValidationConfig()
    intervals = prepare_arcticdata_validation_intervals(inventory, cfg)
    intervals = intervals[intervals["model_facies_id"].notna() & intervals["unit_depth_mid_m"].notna()].copy()
    intervals["model_facies_id"] = intervals["model_facies_id"].astype(int)
    if intervals["borehole_group"].nunique() < 2:
        raise ValueError("ArcticData facies holdout validation requires at least two boreholes")
    rows: list[dict[str, Any]] = []
    for borehole_group in sorted(intervals["borehole_group"].unique()):
        holdout = intervals[intervals["borehole_group"] == borehole_group]
        train = intervals[intervals["borehole_group"] != borehole_group]
        if train.empty or holdout.empty:
            continue
        train_classes = train["model_facies_id"].to_numpy(dtype=np.int64)
        global_majority = int(pd.Series(train_classes).mode().iloc[0])
        site_majority = train.groupby("site")["model_facies_id"].agg(lambda item: int(pd.Series(item).mode().iloc[0]))
        site_pred = holdout["site"].map(site_majority).fillna(global_majority).to_numpy(dtype=np.int64)
        train_features = _feature_matrix(train, "unit_depth_mid_m", cfg)
        hold_features = _feature_matrix(holdout, "unit_depth_mid_m", cfg)
        spatial_pred = _weighted_vote(train_features, train_classes, hold_features, cfg.knn_k)
        unique, counts = np.unique(train_classes, return_counts=True)
        freq = {int(cls): int(count) for cls, count in zip(unique, counts)}
        class_weights = {cls: (len(train_classes) / max(count, 1)) ** float(cfg.rare_class_weight_power) for cls, count in freq.items()}
        rare_pred = _weighted_vote(train_features, train_classes, hold_features, cfg.knn_k, class_weights=class_weights)
        for model, pred in [
            ("GlobalMajority", np.full(len(holdout), global_majority, dtype=np.int64)),
            ("SiteMajority", site_pred),
            ("SpatialDepthKNN", spatial_pred),
            ("RareAwareSpatialKNN", rare_pred),
        ]:
            for idx, (_, row) in enumerate(holdout.iterrows()):
                obs = int(row["model_facies_id"])
                value = int(pred[idx])
                rows.append(
                    {
                        "model": model,
                        "site": str(row["site"]),
                        "borehole": str(row["borehole"]),
                        "borehole_group": str(row["borehole_group"]),
                        "source_file": str(row["source_file"]),
                        "depth_mid_m": float(row["unit_depth_mid_m"]),
                        "observed_facies": obs,
                        "predicted_facies": value,
                        "correct": bool(obs == value),
                        "train_n": int(len(train)),
                    }
                )
    return pd.DataFrame(rows)


def _class_recall(obs: np.ndarray, pred: np.ndarray, cls: int) -> float:
    mask = obs == cls
    return float(np.mean(pred[mask] == cls)) if np.any(mask) else np.nan


def _class_precision(obs: np.ndarray, pred: np.ndarray, cls: int) -> float:
    mask = pred == cls
    return float(np.mean(obs[mask] == cls)) if np.any(mask) else np.nan


def _macro_f1(obs: np.ndarray, pred: np.ndarray) -> float:
    scores: list[float] = []
    for cls in sorted(set(obs.tolist()) | set(pred.tolist())):
        precision = _class_precision(obs, pred, int(cls))
        recall = _class_recall(obs, pred, int(cls))
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0.0:
            scores.append(2.0 * precision * recall / (precision + recall))
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else np.nan


def summarize_arcticdata_facies_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in predictions.groupby("model", sort=False):
        obs = group["observed_facies"].to_numpy(dtype=np.int64)
        pred = group["predicted_facies"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "model": str(model),
                "n": int(len(group)),
                "n_sites": int(group["site"].nunique()),
                "n_boreholes": int(group["borehole_group"].nunique()),
                "accuracy": float(np.mean(obs == pred)),
                "macro_f1": _macro_f1(obs, pred),
                "ice_rich_recall": _class_recall(obs, pred, 3),
                "ice_rich_precision": _class_precision(obs, pred, 3),
                "wedge_ice_recall": _class_recall(obs, pred, 6),
                "wedge_ice_precision": _class_precision(obs, pred, 6),
            }
        )
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["model"] == "GlobalMajority", "accuracy"]
    metrics["accuracy_gain_vs_global_majority"] = metrics["accuracy"] - float(baseline.iloc[0]) if not baseline.empty else np.nan
    return metrics


def arcticdata_holdout_validation_tables(
    inventory: pd.DataFrame,
    config: ArcticDataValidationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or ArcticDataValidationConfig()
    intervals = prepare_arcticdata_validation_intervals(inventory, cfg)
    eic_predictions = leave_one_borehole_eic_predictions(inventory, cfg)
    facies_predictions = leave_one_borehole_facies_predictions(inventory, cfg)
    return {
        "intervals": intervals,
        "eic_predictions": eic_predictions,
        "eic_metrics": summarize_arcticdata_eic_predictions(eic_predictions, cfg),
        "facies_predictions": facies_predictions,
        "facies_metrics": summarize_arcticdata_facies_predictions(facies_predictions),
    }
