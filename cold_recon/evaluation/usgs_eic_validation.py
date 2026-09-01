from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class EICValidationConfig:
    borehole_spacing_m: float = 20.0
    horizontal_scale_m: float = 20.0
    depth_scale_m: float = 0.25
    idw_k: int = 8
    depth_k: int = 12
    high_eic_threshold: float = 0.30
    sigma_fraction: float = 0.05


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _ordered_borehole_coords(borehole_ids: Iterable[str], spacing_m: float) -> pd.DataFrame:
    ids = sorted({str(item) for item in borehole_ids})
    return pd.DataFrame(
        {
            "BOREHOLE_ID": ids,
            "local_x_m": np.arange(len(ids), dtype=np.float32) * float(spacing_m),
            "local_y_m": np.zeros(len(ids), dtype=np.float32),
            "coordinate_source": "ordered_borehole_index",
        }
    )


def _location_coords(locations: pd.DataFrame | None, borehole_ids: Iterable[str], spacing_m: float) -> pd.DataFrame:
    if locations is None or locations.empty:
        return _ordered_borehole_coords(borehole_ids, spacing_m)
    loc = locations.copy()
    if "BOREHOLE_ID" not in loc.columns:
        return _ordered_borehole_coords(borehole_ids, spacing_m)
    upper_to_col = {str(col).upper(): col for col in loc.columns}
    lat_col = upper_to_col.get("LATITUDE") or upper_to_col.get("LAT") or upper_to_col.get("Y")
    lon_col = upper_to_col.get("LONGITUDE") or upper_to_col.get("LON") or upper_to_col.get("X")
    if lat_col is None or lon_col is None:
        return _ordered_borehole_coords(borehole_ids, spacing_m)
    lat = _coerce_numeric(loc[lat_col])
    lon = _coerce_numeric(loc[lon_col])
    valid = lat.notna() & lon.notna()
    if int(valid.sum()) < 2:
        return _ordered_borehole_coords(borehole_ids, spacing_m)
    loc = loc.loc[valid, ["BOREHOLE_ID"]].copy()
    lat = lat.loc[valid].astype(float)
    lon = lon.loc[valid].astype(float)
    lat0 = float(lat.mean())
    lon0 = float(lon.mean())
    x = (lon.to_numpy() - lon0) * 111_320.0 * np.cos(np.deg2rad(lat0))
    y = (lat.to_numpy() - lat0) * 110_540.0
    loc["local_x_m"] = x.astype(np.float32)
    loc["local_y_m"] = y.astype(np.float32)
    loc["coordinate_source"] = "public_lat_lon"
    loc = loc.drop_duplicates("BOREHOLE_ID", keep="first")
    fallback = _ordered_borehole_coords(borehole_ids, spacing_m)
    return fallback.drop(columns=["local_x_m", "local_y_m", "coordinate_source"]).merge(
        loc,
        on="BOREHOLE_ID",
        how="left",
    ).assign(
        local_x_m=lambda df: df["local_x_m"].fillna(fallback["local_x_m"]),
        local_y_m=lambda df: df["local_y_m"].fillna(fallback["local_y_m"]),
        coordinate_source=lambda df: df["coordinate_source"].fillna("ordered_borehole_index"),
    )


def prepare_eic_intervals(
    eic: pd.DataFrame,
    locations: pd.DataFrame | None = None,
    config: EICValidationConfig | None = None,
) -> pd.DataFrame:
    """Return cleaned EIC intervals with reproducible local coordinates."""
    cfg = config or EICValidationConfig()
    required = {"BOREHOLE_ID", "DEPTH_TOP", "DEPTH_BOTTOM", "EXCESS_ICE_CONTENT"}
    missing = required.difference(eic.columns)
    if missing:
        raise ValueError(f"Missing EIC columns: {sorted(missing)}")
    out = eic.copy()
    out["BOREHOLE_ID"] = out["BOREHOLE_ID"].astype(str)
    out["DEPTH_TOP"] = _coerce_numeric(out["DEPTH_TOP"])
    out["DEPTH_BOTTOM"] = _coerce_numeric(out["DEPTH_BOTTOM"])
    out["EXCESS_ICE_CONTENT"] = _coerce_numeric(out["EXCESS_ICE_CONTENT"])
    out = out.dropna(subset=["BOREHOLE_ID", "DEPTH_TOP", "DEPTH_BOTTOM", "EXCESS_ICE_CONTENT"]).copy()
    out = out[out["DEPTH_BOTTOM"] >= out["DEPTH_TOP"]].copy()
    out["depth_mid_m"] = 0.5 * (out["DEPTH_TOP"].astype(float) + out["DEPTH_BOTTOM"].astype(float))
    out["eic_fraction"] = (out["EXCESS_ICE_CONTENT"].astype(float) / 100.0).clip(0.0, 1.0)
    coords = _location_coords(locations, out["BOREHOLE_ID"], cfg.borehole_spacing_m)
    out = out.merge(coords, on="BOREHOLE_ID", how="left")
    out["local_x_m"] = out["local_x_m"].astype(np.float32)
    out["local_y_m"] = out["local_y_m"].astype(np.float32)
    out["coordinate_source"] = out["coordinate_source"].astype(str)
    out = out.sort_values(["BOREHOLE_ID", "DEPTH_TOP", "DEPTH_BOTTOM"]).reset_index(drop=True)
    out["interval_id"] = np.arange(len(out), dtype=np.int64)
    return out


def _feature_matrix(df: pd.DataFrame, cfg: EICValidationConfig) -> np.ndarray:
    return np.column_stack(
        [
            df["local_x_m"].to_numpy(dtype=np.float32) / float(cfg.horizontal_scale_m),
            df["local_y_m"].to_numpy(dtype=np.float32) / float(cfg.horizontal_scale_m),
            df["depth_mid_m"].to_numpy(dtype=np.float32) / float(cfg.depth_scale_m),
        ]
    ).astype(np.float32)


def _idw_query(train_features: np.ndarray, train_values: np.ndarray, query_features: np.ndarray, k: int) -> np.ndarray:
    if len(train_values) == 0:
        raise ValueError("Cannot predict EIC with no training intervals")
    k = min(max(1, int(k)), len(train_values))
    tree = cKDTree(train_features)
    dist, idx = tree.query(query_features, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.square(dist + 1e-6)
    return ((weights * train_values[idx]).sum(axis=1) / weights.sum(axis=1)).astype(np.float32)


def _depth_idw(train: pd.DataFrame, query: pd.DataFrame, cfg: EICValidationConfig) -> np.ndarray:
    train_depth = train["depth_mid_m"].to_numpy(dtype=np.float32)[:, None] / float(cfg.depth_scale_m)
    query_depth = query["depth_mid_m"].to_numpy(dtype=np.float32)[:, None] / float(cfg.depth_scale_m)
    return _idw_query(train_depth, train["eic_fraction"].to_numpy(dtype=np.float32), query_depth, cfg.depth_k)


def _prediction_rows_for_holdout(train: pd.DataFrame, holdout: pd.DataFrame, cfg: EICValidationConfig) -> list[dict[str, object]]:
    train_values = train["eic_fraction"].to_numpy(dtype=np.float32)
    global_pred = np.full(len(holdout), float(np.mean(train_values)), dtype=np.float32)
    depth_pred = _depth_idw(train, holdout, cfg)
    spatial_pred = _idw_query(_feature_matrix(train, cfg), train_values, _feature_matrix(holdout, cfg), cfg.idw_k)
    models = [
        ("GlobalMean", global_pred),
        ("DepthIDW", depth_pred),
        ("SpatialDepthIDW", spatial_pred),
    ]
    rows: list[dict[str, object]] = []
    for model, pred in models:
        for idx, (_, row) in enumerate(holdout.iterrows()):
            observed = float(row["eic_fraction"])
            prediction = float(np.clip(pred[idx], 0.0, 1.0))
            error = prediction - observed
            rows.append(
                {
                    "model": model,
                    "borehole_id": str(row["BOREHOLE_ID"]),
                    "interval_id": int(row["interval_id"]),
                    "depth_top_m": float(row["DEPTH_TOP"]),
                    "depth_bottom_m": float(row["DEPTH_BOTTOM"]),
                    "depth_mid_m": float(row["depth_mid_m"]),
                    "local_x_m": float(row["local_x_m"]),
                    "local_y_m": float(row["local_y_m"]),
                    "coordinate_source": str(row["coordinate_source"]),
                    "observed_eic": observed,
                    "predicted_eic": prediction,
                    "error": error,
                    "abs_error": abs(error),
                    "squared_error": error * error,
                    "train_n": int(len(train)),
                }
            )
    return rows


def leave_one_borehole_out_predictions(
    eic: pd.DataFrame,
    locations: pd.DataFrame | None = None,
    config: EICValidationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or EICValidationConfig()
    intervals = prepare_eic_intervals(eic, locations, cfg)
    boreholes = sorted(intervals["BOREHOLE_ID"].unique())
    if len(boreholes) < 2:
        raise ValueError("Leave-one-borehole-out validation requires at least two boreholes")
    rows: list[dict[str, object]] = []
    for borehole_id in boreholes:
        holdout = intervals[intervals["BOREHOLE_ID"] == borehole_id]
        train = intervals[intervals["BOREHOLE_ID"] != borehole_id]
        if holdout.empty or train.empty:
            continue
        rows.extend(_prediction_rows_for_holdout(train, holdout, cfg))
    return pd.DataFrame(rows)


def summarize_eic_holdout_predictions(
    predictions: pd.DataFrame,
    config: EICValidationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or EICValidationConfig()
    if predictions.empty:
        raise ValueError("Cannot summarize empty EIC prediction table")
    rows: list[dict[str, float | str]] = []
    for model, group in predictions.groupby("model", sort=False):
        obs = group["observed_eic"].to_numpy(dtype=float)
        pred = group["predicted_eic"].to_numpy(dtype=float)
        err = pred - obs
        obs_event = obs >= cfg.high_eic_threshold
        pred_event = pred >= cfg.high_eic_threshold
        tp = float(np.sum(obs_event & pred_event))
        fp = float(np.sum(~obs_event & pred_event))
        fn = float(np.sum(obs_event & ~pred_event))
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan
        corr = np.corrcoef(obs, pred)[0, 1] if np.std(obs) > 0.0 and np.std(pred) > 0.0 else np.nan
        rows.append(
            {
                "model": str(model),
                "n": float(len(group)),
                "n_boreholes": float(group["borehole_id"].nunique()),
                "observed_mean_eic": float(np.mean(obs)),
                "predicted_mean_eic": float(np.mean(pred)),
                "bias": float(np.mean(err)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
                "normalized_rmse": float(np.sqrt(np.mean(np.square(err / cfg.sigma_fraction)))),
                "pearson_r": float(corr) if np.isfinite(corr) else np.nan,
                "high_eic_threshold": float(cfg.high_eic_threshold),
                "high_eic_prevalence": float(np.mean(obs_event)),
                "high_eic_accuracy": float(np.mean(obs_event == pred_event)),
                "high_eic_precision": float(precision) if np.isfinite(precision) else np.nan,
                "high_eic_recall": float(recall) if np.isfinite(recall) else np.nan,
                "high_eic_f1": float(f1) if np.isfinite(f1) else np.nan,
            }
        )
    metrics = pd.DataFrame(rows)
    global_rmse = metrics.loc[metrics["model"] == "GlobalMean", "rmse"]
    if not global_rmse.empty and float(global_rmse.iloc[0]) > 0.0:
        baseline = float(global_rmse.iloc[0])
        metrics["rmse_reduction_vs_global_mean"] = 1.0 - metrics["rmse"].astype(float) / baseline
    else:
        metrics["rmse_reduction_vs_global_mean"] = np.nan

    per_borehole_rows: list[dict[str, float | str]] = []
    for (model, borehole_id), group in predictions.groupby(["model", "borehole_id"], sort=False):
        obs = group["observed_eic"].to_numpy(dtype=float)
        pred = group["predicted_eic"].to_numpy(dtype=float)
        err = pred - obs
        per_borehole_rows.append(
            {
                "model": str(model),
                "borehole_id": str(borehole_id),
                "n": float(len(group)),
                "observed_mean_eic": float(np.mean(obs)),
                "predicted_mean_eic": float(np.mean(pred)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
            }
        )
    return metrics, pd.DataFrame(per_borehole_rows)


def eic_holdout_validation_tables(
    eic: pd.DataFrame,
    locations: pd.DataFrame | None = None,
    config: EICValidationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or EICValidationConfig()
    predictions = leave_one_borehole_out_predictions(eic, locations, cfg)
    metrics, per_borehole = summarize_eic_holdout_predictions(predictions, cfg)
    intervals = prepare_eic_intervals(eic, locations, cfg)
    return {"intervals": intervals, "predictions": predictions, "metrics": metrics, "per_borehole": per_borehole}
