from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from cold_recon.data.arcticdata_cryostratigraphy_loader import CRYOFACIES_TO_MODEL_FACIES
from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.evaluation.posterior_assimilation import filter_observations_to_posterior, nearest_indices


@dataclass(frozen=True)
class ArcticDataConditioningSplit:
    site: str
    site_observations: ObservationTable
    train_observations: ObservationTable
    holdout_observations: ObservationTable
    train_boreholes: np.ndarray
    holdout_boreholes: np.ndarray


def choose_conditioning_site(
    token_index: pd.DataFrame,
    max_span_m: float = 5000.0,
    min_eic_tokens: int = 20,
    min_boreholes: int = 4,
) -> str:
    rows: list[dict[str, Any]] = []
    for site, group in token_index.groupby("site", sort=True):
        eic_n = int(np.sum(group["type_id"].to_numpy(dtype=int) == OBS_TYPES["borehole_eic"]))
        boreholes = int(group["borehole"].nunique())
        x_span = float(group["x"].max() - group["x"].min())
        y_span = float(group["y"].max() - group["y"].min())
        rows.append(
            {
                "site": str(site),
                "n": int(len(group)),
                "eic_n": eic_n,
                "boreholes": boreholes,
                "x_span_m": x_span,
                "y_span_m": y_span,
                "eligible": eic_n >= min_eic_tokens and boreholes >= min_boreholes and max(x_span, y_span) <= max_span_m,
            }
        )
    stats = pd.DataFrame(rows)
    eligible = stats[stats["eligible"]].copy()
    if eligible.empty:
        eligible = stats[(stats["eic_n"] >= min_eic_tokens) & (stats["boreholes"] >= min_boreholes)].copy()
    if eligible.empty:
        raise ValueError("No ArcticData site has enough EIC tokens and boreholes for conditioned diffusion")
    eligible["score"] = eligible["eic_n"] * 3 + eligible["n"] - 0.01 * np.maximum(eligible["x_span_m"], eligible["y_span_m"])
    return str(eligible.sort_values("score", ascending=False).iloc[0]["site"])


def split_observations_by_site_borehole(
    observations: ObservationTable,
    site_ids: np.ndarray,
    borehole_ids: np.ndarray,
    site: str,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> ArcticDataConditioningSplit:
    site_ids = np.asarray(site_ids).astype(str)
    borehole_ids = np.asarray(borehole_ids).astype(str)
    site_mask = site_ids == str(site)
    site_idx = np.where(site_mask)[0]
    if site_idx.size == 0:
        raise ValueError(f"No observations found for ArcticData site {site!r}")
    site_obs = observations.subset(site_idx)
    site_boreholes = borehole_ids[site_idx]
    groups = np.unique(site_boreholes)
    if groups.size < 2:
        raise ValueError("At least two boreholes are required for ArcticData holdout conditioning")
    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    n_hold = max(1, int(round(groups.size * float(holdout_fraction))))
    n_hold = min(n_hold, max(1, groups.size - 1))
    holdout_groups = set(shuffled[:n_hold].tolist())

    def has_type(group_set: set[str], type_id: int) -> bool:
        mask = np.isin(site_boreholes, list(group_set))
        return bool(np.any(site_obs.type_ids[mask] == type_id))

    for required_type in (OBS_TYPES["borehole_eic"], OBS_TYPES["borehole_facies"]):
        if has_type(holdout_groups, required_type):
            continue
        for group in shuffled[n_hold:]:
            candidate = str(group)
            if len(holdout_groups) >= groups.size - 1:
                break
            holdout_groups.add(candidate)
            if has_type(holdout_groups, required_type):
                break
    train_groups = set(groups.tolist()) - holdout_groups
    if not train_groups or not has_type(train_groups, OBS_TYPES["borehole_eic"]):
        raise ValueError("ArcticData split left no EIC observations for conditioning")

    hold_mask_site = np.isin(site_boreholes, list(holdout_groups))
    train_idx = site_idx[~hold_mask_site]
    holdout_idx = site_idx[hold_mask_site]
    return ArcticDataConditioningSplit(
        site=str(site),
        site_observations=site_obs,
        train_observations=observations.subset(train_idx),
        holdout_observations=observations.subset(holdout_idx),
        train_boreholes=np.asarray(sorted(train_groups), dtype=str),
        holdout_boreholes=np.asarray(sorted(holdout_groups), dtype=str),
    )


def subsample_observations(observations: ObservationTable, max_tokens: int, seed: int) -> ObservationTable:
    if observations.n_obs <= int(max_tokens):
        return observations
    rng = np.random.default_rng(seed)
    idx = np.arange(observations.n_obs)
    rare = (observations.type_ids == OBS_TYPES["borehole_facies"]) & np.isin(np.rint(observations.values).astype(int), [3, 6])
    eic = observations.type_ids == OBS_TYPES["borehole_eic"]
    keep = np.where(rare | eic)[0]
    if keep.size >= max_tokens:
        selected = keep.copy()
        rng.shuffle(selected)
        return observations.subset(np.sort(selected[:max_tokens]))
    remaining = np.setdiff1d(idx, keep, assume_unique=False)
    rng.shuffle(remaining)
    selected = np.concatenate([keep, remaining[: max_tokens - keep.size]])
    return observations.subset(np.sort(selected))


def _scaled_features(coords: np.ndarray, horizontal_scale_m: float = 50.0, depth_scale_m: float = 0.25) -> np.ndarray:
    return np.column_stack(
        [
            coords[:, 0] / float(horizontal_scale_m),
            coords[:, 1] / float(horizontal_scale_m),
            coords[:, 2] / float(depth_scale_m),
        ]
    ).astype(np.float32)


def _idw(train_coords: np.ndarray, train_values: np.ndarray, query_coords: np.ndarray, k: int = 8) -> np.ndarray:
    k = min(max(1, int(k)), len(train_values))
    dist, idx = cKDTree(_scaled_features(train_coords)).query(_scaled_features(query_coords), k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.square(dist + 1e-6)
    return ((weights * train_values[idx]).sum(axis=1) / weights.sum(axis=1)).astype(np.float32)


def _knn_class(train_coords: np.ndarray, train_classes: np.ndarray, query_coords: np.ndarray, k: int = 12) -> np.ndarray:
    k = min(max(1, int(k)), len(train_classes))
    dist, idx = cKDTree(_scaled_features(train_coords)).query(_scaled_features(query_coords), k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.square(dist + 1e-6)
    pred = np.zeros(query_coords.shape[0], dtype=np.int64)
    for row in range(query_coords.shape[0]):
        scores: dict[int, float] = {}
        for nbr, weight in zip(idx[row], weights[row]):
            cls = int(train_classes[nbr])
            scores[cls] = scores.get(cls, 0.0) + float(weight)
        pred[row] = max(scores.items(), key=lambda item: (item[1], -item[0]))[0]
    return pred


def _classification_scores(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {"facies_accuracy": float(np.mean(obs == pred)) if obs.size else float("nan")}
    for name, cls in {"ice_rich": 3, "wedge_ice": 6}.items():
        cls_obs = obs == cls
        cls_pred = pred == cls
        out[f"{name}_recall"] = float(np.mean(pred[cls_obs] == cls)) if np.any(cls_obs) else float("nan")
        out[f"{name}_precision"] = float(np.mean(obs[cls_pred] == cls)) if np.any(cls_pred) else float("nan")
    return out


def _eic_scores(obs: np.ndarray, pred: np.ndarray, high_threshold: float = 0.30) -> dict[str, float]:
    if obs.size == 0:
        return {"eic_n": 0.0}
    err = pred - obs
    obs_event = obs >= float(high_threshold)
    pred_event = pred >= float(high_threshold)
    tp = float(np.sum(obs_event & pred_event))
    fp = float(np.sum(~obs_event & pred_event))
    fn = float(np.sum(obs_event & ~pred_event))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0.0 else np.nan
    return {
        "eic_n": float(obs.size),
        "eic_mae": float(np.mean(np.abs(err))),
        "eic_rmse": float(np.sqrt(np.mean(err**2))),
        "high_eic_accuracy": float(np.mean(obs_event == pred_event)),
        "high_eic_precision": float(precision) if np.isfinite(precision) else np.nan,
        "high_eic_recall": float(recall) if np.isfinite(recall) else np.nan,
        "high_eic_f1": float(f1) if np.isfinite(f1) else np.nan,
    }


def _posterior_values_at_coords(posterior: dict[str, np.ndarray], coords: np.ndarray, field: str) -> np.ndarray:
    ix, iy, iz = nearest_indices(coords, posterior)
    return np.asarray(posterior[field][ix, iy, iz], dtype=np.float32)


def _horizontal_group_ids(coords: np.ndarray, decimals: int = 3) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    _, inverse = np.unique(np.round(coords[:, :2].astype(float), int(decimals)), axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def _eic_candidate_predictions(
    posterior: dict[str, np.ndarray],
    train_coords: np.ndarray,
    train_values: np.ndarray,
    query_coords: np.ndarray,
    eic_field: str,
    threshold: float,
) -> dict[str, np.ndarray]:
    global_mean = np.full(query_coords.shape[0], float(train_values.mean()), dtype=np.float32)
    idw = np.clip(_idw(train_coords, train_values, query_coords, k=8), 0.0, 1.0)
    raw = np.clip(_posterior_values_at_coords(posterior, query_coords, eic_field), 0.0, 1.0)
    low = train_values[train_values < float(threshold)]
    high = train_values[train_values >= float(threshold)]
    low_mean = float(low.mean()) if low.size else float(train_values.mean())
    high_mean = float(high.mean()) if high.size else float(train_values.mean())
    event = np.where(raw >= float(threshold), high_mean, low_mean).astype(np.float32)
    return {
        "global_mean": global_mean,
        "spatial_depth_idw": idw,
        "diffusion_raw": raw,
        "diffusion_event_calibrated": event,
        "transfer_idw_adapter": idw.astype(np.float32),
        "spatial_event_guarded_ensemble": np.clip(0.50 * idw + 0.25 * event + 0.25 * global_mean, 0.0, 1.0).astype(np.float32),
        "spatial_raw_guarded_ensemble": np.clip(0.50 * idw + 0.25 * raw + 0.25 * global_mean, 0.0, 1.0).astype(np.float32),
        "spatial_mean_guarded_ensemble": np.clip(0.65 * idw + 0.35 * global_mean, 0.0, 1.0).astype(np.float32),
    }


def _facies_candidate_predictions(
    posterior: dict[str, np.ndarray],
    train_coords: np.ndarray,
    train_classes: np.ndarray,
    query_coords: np.ndarray,
    wedge_probability_threshold: float,
    knn_k: int,
    confidence_threshold: float,
) -> dict[str, np.ndarray]:
    knn = _knn_class(train_coords, train_classes, query_coords, k=knn_k)
    if "facies_probability" not in posterior:
        return {"knn": knn}
    ix, iy, iz = nearest_indices(query_coords, posterior)
    probs = np.asarray(posterior["facies_probability"][ix, iy, iz], dtype=np.float32)
    diffusion_mode = np.argmax(probs, axis=1).astype(np.int64)
    if "facies_mode" in posterior:
        diffusion_mode = np.asarray(posterior["facies_mode"][ix, iy, iz], dtype=np.int64)
    confidence = probs.max(axis=1)
    wedge_probability = probs[:, 6] if probs.shape[1] > 6 else np.zeros(query_coords.shape[0], dtype=np.float32)
    wedge_hybrid = np.where(wedge_probability >= float(wedge_probability_threshold), 6, knn).astype(np.int64)
    confidence_guarded = np.where(confidence >= float(confidence_threshold), diffusion_mode, knn).astype(np.int64)
    confidence_wedge_guarded = np.where(wedge_probability >= float(wedge_probability_threshold), 6, confidence_guarded).astype(np.int64)
    return {
        "knn": knn,
        "diffusion_mode": diffusion_mode,
        "wedge_hybrid": wedge_hybrid,
        "confidence_guarded": confidence_guarded,
        "confidence_wedge_guarded": confidence_wedge_guarded,
    }


def _wedge_scores(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    wedge_obs = obs == 6
    wedge_pred = pred == 6
    recall = float(np.mean(wedge_pred[wedge_obs])) if np.any(wedge_obs) else np.nan
    precision = float(np.mean(wedge_obs[wedge_pred])) if np.any(wedge_pred) else np.nan
    beta = 2.0
    f_beta = (
        (1.0 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and beta * beta * precision + recall > 0.0
        else np.nan
    )
    return {
        "accuracy": float(np.mean(obs == pred)) if obs.size else np.nan,
        "wedge_recall": recall,
        "wedge_precision": precision,
        "wedge_f2": float(f_beta) if np.isfinite(f_beta) else np.nan,
    }


def adaptive_wedge_recall_predictions(
    posterior: dict[str, np.ndarray],
    train: ObservationTable,
    query: ObservationTable,
    thresholds: np.ndarray | None = None,
    min_cv_precision: float = 0.20,
    knn_k: int = 12,
) -> tuple[np.ndarray, dict[str, float | str]]:
    train_mask = train.type_ids == OBS_TYPES["borehole_facies"]
    query_mask = query.type_ids == OBS_TYPES["borehole_facies"]
    train_classes = np.rint(train.values[train_mask]).astype(np.int64)
    train_coords = train.coords[train_mask]
    query_coords = query.coords[query_mask]
    if train_classes.size == 0 or query_coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), {"adaptive_wedge_method": "none"}
    if "facies_probability" not in posterior or not np.any(train_classes == 6):
        pred = _knn_class(train_coords, train_classes, query_coords, k=knn_k)
        return pred.astype(np.int64), {"adaptive_wedge_method": "knn_no_train_wedge", "adaptive_wedge_threshold": np.nan}

    thresholds = np.asarray(thresholds if thresholds is not None else np.linspace(0.2, 0.9, 8), dtype=np.float32)
    group_ids = _horizontal_group_ids(train_coords)
    groups = np.unique(group_ids)
    cv_rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred_cv = np.zeros(train_classes.shape[0], dtype=np.int64)
        for group in groups:
            query_group = group_ids == group
            keep = ~query_group
            if not np.any(keep):
                pred_cv[query_group] = int(pd.Series(train_classes).mode().iloc[0])
                continue
            pred_cv[query_group] = _facies_candidate_predictions(
                posterior,
                train_coords[keep],
                train_classes[keep],
                train_coords[query_group],
                wedge_probability_threshold=float(threshold),
                knn_k=knn_k,
                confidence_threshold=1.1,
            )["wedge_hybrid"]
        scores = _wedge_scores(train_classes, pred_cv)
        cv_rows.append({"threshold": float(threshold), **scores})
    finite_recall = [row for row in cv_rows if np.isfinite(row["wedge_recall"])]
    precision_feasible = [row for row in finite_recall if np.nan_to_num(row["wedge_precision"], nan=0.0) >= float(min_cv_precision)]
    pool = precision_feasible or finite_recall or cv_rows
    selected = max(
        pool,
        key=lambda row: (
            np.nan_to_num(row["wedge_recall"], nan=-1.0),
            -row["threshold"],
            np.nan_to_num(row["wedge_precision"], nan=-1.0),
            np.nan_to_num(row["accuracy"], nan=-1.0),
        ),
    )
    pred = _facies_candidate_predictions(
        posterior,
        train_coords,
        train_classes,
        query_coords,
        wedge_probability_threshold=float(selected["threshold"]),
        knn_k=knn_k,
        confidence_threshold=1.1,
    )["wedge_hybrid"]
    info: dict[str, float | str] = {
        "adaptive_wedge_method": "recall_first_wedge_probability",
        "adaptive_wedge_threshold": float(selected["threshold"]),
        "adaptive_wedge_cv_accuracy": float(selected["accuracy"]),
        "adaptive_wedge_cv_recall": float(selected["wedge_recall"]) if np.isfinite(selected["wedge_recall"]) else np.nan,
        "adaptive_wedge_cv_precision": float(selected["wedge_precision"]) if np.isfinite(selected["wedge_precision"]) else np.nan,
        "adaptive_wedge_cv_f2": float(selected["wedge_f2"]) if np.isfinite(selected["wedge_f2"]) else np.nan,
    }
    return pred.astype(np.int64), info


def adaptive_facies_predictions(
    posterior: dict[str, np.ndarray],
    train: ObservationTable,
    query: ObservationTable,
    wedge_probability_threshold: float = 0.80,
    knn_k: int = 12,
    confidence_threshold: float = 0.75,
    diffusion_cv_margin: float = 0.07,
    wedge_cv_tolerance: float = 0.02,
) -> tuple[np.ndarray, dict[str, float | str]]:
    train_mask = train.type_ids == OBS_TYPES["borehole_facies"]
    query_mask = query.type_ids == OBS_TYPES["borehole_facies"]
    train_classes = np.rint(train.values[train_mask]).astype(np.int64)
    train_coords = train.coords[train_mask]
    query_coords = query.coords[query_mask]
    if train_classes.size == 0 or query_coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), {"adaptive_facies_method": "none"}
    if train_classes.size == 1:
        pred = np.full(query_coords.shape[0], int(train_classes[0]), dtype=np.int64)
        return pred, {"adaptive_facies_method": "single_train_class", "adaptive_facies_cv_accuracy": 1.0}

    probe = _facies_candidate_predictions(
        posterior,
        train_coords,
        train_classes,
        train_coords[:1],
        wedge_probability_threshold=wedge_probability_threshold,
        knn_k=knn_k,
        confidence_threshold=confidence_threshold,
    )
    cv = {name: np.zeros(train_classes.shape[0], dtype=np.int64) for name in probe}
    group_ids = _horizontal_group_ids(train_coords)
    groups = np.unique(group_ids)
    if groups.size <= 1:
        groups = np.arange(train_classes.size)
        group_ids = groups.copy()
    for group in groups:
        query_group = group_ids == group
        keep = ~query_group
        if not np.any(keep):
            for name in cv:
                cv[name][query_group] = int(pd.Series(train_classes).mode().iloc[0])
            continue
        group_pred = _facies_candidate_predictions(
            posterior,
            train_coords[keep],
            train_classes[keep],
            train_coords[query_group],
            wedge_probability_threshold=wedge_probability_threshold,
            knn_k=knn_k,
            confidence_threshold=confidence_threshold,
        )
        for name, pred in group_pred.items():
            cv[name][query_group] = pred
    accuracies = {name: float(np.mean(pred == train_classes)) for name, pred in cv.items()}
    selected = "knn"
    if accuracies.get("wedge_hybrid", -np.inf) >= accuracies.get("knn", -np.inf) - float(wedge_cv_tolerance):
        selected = "wedge_hybrid"
    if accuracies.get("diffusion_mode", -np.inf) >= accuracies.get("knn", -np.inf) + float(diffusion_cv_margin):
        selected = "confidence_wedge_guarded" if "confidence_wedge_guarded" in accuracies else "diffusion_mode"

    candidates = _facies_candidate_predictions(
        posterior,
        train_coords,
        train_classes,
        query_coords,
        wedge_probability_threshold=wedge_probability_threshold,
        knn_k=knn_k,
        confidence_threshold=confidence_threshold,
    )
    pred = candidates[selected]
    info: dict[str, float | str] = {"adaptive_facies_method": selected, "adaptive_facies_cv_accuracy": accuracies[selected]}
    for name, value in accuracies.items():
        info[f"adaptive_facies_cv_accuracy_{name}"] = value
    return pred.astype(np.int64), info


def adaptive_eic_predictions(
    posterior: dict[str, np.ndarray],
    train: ObservationTable,
    query: ObservationTable,
    eic_field: str = "eic_mean",
    threshold: float = 0.30,
    compact_transfer_max_groups: int = 15,
    compact_transfer_max_observations: int = 180,
    compact_transfer_idw_tolerance: float = 0.20,
) -> tuple[np.ndarray, dict[str, float | str]]:
    train_mask = train.type_ids == OBS_TYPES["borehole_eic"]
    query_mask = query.type_ids == OBS_TYPES["borehole_eic"]
    train_values = np.clip(train.values[train_mask], 0.0, 1.0).astype(np.float32)
    train_coords = train.coords[train_mask]
    query_coords = query.coords[query_mask]
    if train_values.size == 0 or query_coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), {"adaptive_eic_method": "none"}
    if train_values.size == 1:
        pred = np.full(query_coords.shape[0], float(train_values[0]), dtype=np.float32)
        return pred, {"adaptive_eic_method": "single_train_mean", "adaptive_eic_cv_rmse": 0.0}

    cv = {name: np.zeros_like(train_values) for name in _eic_candidate_predictions(posterior, train_coords, train_values, train_coords[:1], eic_field, threshold)}
    group_ids = _horizontal_group_ids(train_coords)
    groups = np.unique(group_ids)
    if groups.size <= 1:
        groups = np.arange(train_values.size)
        group_ids = groups.copy()
    for group in groups:
        query_group = group_ids == group
        keep = ~query_group
        if not np.any(keep):
            for name in cv:
                cv[name][query_group] = float(train_values.mean())
            continue
        group_pred = _eic_candidate_predictions(posterior, train_coords[keep], train_values[keep], train_coords[query_group], eic_field, threshold)
        for name, pred in group_pred.items():
            cv[name][query_group] = pred
    rmses = {name: float(np.sqrt(np.mean((pred - train_values) ** 2))) for name, pred in cv.items()}
    guarded = [
        "spatial_event_guarded_ensemble",
        "spatial_raw_guarded_ensemble",
        "spatial_mean_guarded_ensemble",
        "transfer_idw_adapter",
    ]
    selected = min(guarded, key=lambda name: (rmses.get(name, float("inf")), name))
    transfer_guard_reason = "cv_selected"
    idw_cv = rmses.get("transfer_idw_adapter", float("inf"))
    global_cv = rmses.get("global_mean", float("inf"))
    if (
        groups.size <= int(compact_transfer_max_groups)
        and train.n_obs <= int(compact_transfer_max_observations)
        and np.isfinite(idw_cv)
        and np.isfinite(global_cv)
        and idw_cv <= global_cv * (1.0 + float(compact_transfer_idw_tolerance))
    ):
        selected = "transfer_idw_adapter"
        transfer_guard_reason = "compact_site_spatial_guard"

    candidates = _eic_candidate_predictions(posterior, train_coords, train_values, query_coords, eic_field, threshold)
    query_pred = candidates[selected]
    info: dict[str, float | str] = {
        "adaptive_eic_method": selected,
        "adaptive_eic_cv_rmse": rmses[selected],
        "adaptive_eic_train_observations": int(train.n_obs),
        "adaptive_eic_train_groups": int(groups.size),
        "adaptive_eic_transfer_guard_reason": transfer_guard_reason,
    }
    for name, value in rmses.items():
        info[f"adaptive_eic_cv_rmse_{name}"] = value
    return query_pred.astype(np.float32), info


def eic_event_calibrated_posterior(
    posterior: dict[str, np.ndarray],
    train_observations: ObservationTable,
    threshold: float = 0.30,
    field_name: str = "eic_event_calibrated_mean",
) -> dict[str, np.ndarray]:
    eic_mask = train_observations.type_ids == OBS_TYPES["borehole_eic"]
    values = np.clip(train_observations.values[eic_mask], 0.0, 1.0)
    out = {key: np.asarray(value).copy() for key, value in posterior.items()}
    if values.size == 0 or "eic_mean" not in out:
        return out
    low = values[values < float(threshold)]
    high = values[values >= float(threshold)]
    global_mean = float(np.mean(values))
    low_mean = float(np.mean(low)) if low.size else min(global_mean, float(threshold) * 0.5)
    high_mean = float(np.mean(high)) if high.size else max(global_mean, float(threshold) + 0.15)
    if low_mean >= float(threshold):
        low_mean = max(0.0, float(threshold) * 0.5)
    if high_mean < float(threshold):
        high_mean = min(1.0, float(threshold) + 0.15)
    calibrated = np.where(out["eic_mean"] >= float(threshold), high_mean, low_mean).astype(np.float32)
    out[field_name] = calibrated
    out["eic_event_calibration_threshold"] = np.asarray(threshold, dtype=np.float32)
    out["eic_event_calibration_low_mean"] = np.asarray(low_mean, dtype=np.float32)
    out["eic_event_calibration_high_mean"] = np.asarray(high_mean, dtype=np.float32)
    return out


def facies_hybrid_calibrated_posterior(
    posterior: dict[str, np.ndarray],
    train_observations: ObservationTable,
    wedge_probability_threshold: float = 0.80,
    knn_k: int = 12,
    field_name: str = "facies_hybrid_mode",
) -> dict[str, np.ndarray]:
    facies_mask = train_observations.type_ids == OBS_TYPES["borehole_facies"]
    if not np.any(facies_mask) or "facies_probability" not in posterior:
        return posterior
    out = {key: np.asarray(value).copy() for key, value in posterior.items()}
    grid_x = np.asarray(out["grid_x"], dtype=np.float32)
    grid_y = np.asarray(out["grid_y"], dtype=np.float32)
    grid_z = np.asarray(out["grid_z"], dtype=np.float32)
    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    query = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    train_classes = np.rint(train_observations.values[facies_mask]).astype(np.int64)
    knn = _knn_class(train_observations.coords[facies_mask], train_classes, query, k=knn_k).reshape(out["facies_probability"].shape[:3])
    wedge_prob = out["facies_probability"][..., 6]
    hybrid = np.where(wedge_prob >= float(wedge_probability_threshold), 6, knn).astype(np.int16)
    out[field_name] = hybrid
    out["facies_hybrid_knn_mode"] = knn.astype(np.int16)
    out["facies_hybrid_wedge_probability_threshold"] = np.asarray(wedge_probability_threshold, dtype=np.float32)
    out["facies_hybrid_knn_k"] = np.asarray(knn_k, dtype=np.int32)
    return out


def arcticdata_conditioning_baseline_rows(
    train: ObservationTable,
    holdout: ObservationTable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    eic_train = train.type_ids == OBS_TYPES["borehole_eic"]
    eic_hold = holdout.type_ids == OBS_TYPES["borehole_eic"]
    fac_train = train.type_ids == OBS_TYPES["borehole_facies"]
    fac_hold = holdout.type_ids == OBS_TYPES["borehole_facies"]
    eic_models: list[tuple[str, np.ndarray]] = []
    if np.any(eic_train) and np.any(eic_hold):
        train_values = np.clip(train.values[eic_train], 0.0, 1.0)
        eic_models = [
            ("GlobalMean", np.full(np.sum(eic_hold), float(np.mean(train_values)), dtype=np.float32)),
            ("SpatialDepthIDW", np.clip(_idw(train.coords[eic_train], train_values, holdout.coords[eic_hold], k=8), 0.0, 1.0)),
        ]
    fac_models: list[tuple[str, np.ndarray]] = []
    if np.any(fac_train) and np.any(fac_hold):
        train_classes = np.rint(train.values[fac_train]).astype(np.int64)
        majority = int(pd.Series(train_classes).mode().iloc[0])
        fac_models = [
            ("GlobalMajority", np.full(np.sum(fac_hold), majority, dtype=np.int64)),
            ("SpatialDepthKNN", _knn_class(train.coords[fac_train], train_classes, holdout.coords[fac_hold], k=12)),
        ]
    for model_name in sorted({name for name, _ in eic_models + fac_models}):
        row: dict[str, Any] = {"model": model_name}
        for eic_name, pred in eic_models:
            if eic_name == model_name:
                row.update(_eic_scores(np.clip(holdout.values[eic_hold], 0.0, 1.0), pred))
                for coord, obs, value in zip(holdout.coords[eic_hold], holdout.values[eic_hold], pred):
                    predictions.append({"model": model_name, "type": "borehole_eic", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(obs), "predicted": float(value)})
        for fac_name, pred in fac_models:
            if fac_name == model_name:
                row.update(_classification_scores(np.rint(holdout.values[fac_hold]).astype(np.int64), pred))
                for coord, obs, value in zip(holdout.coords[fac_hold], holdout.values[fac_hold], pred):
                    predictions.append({"model": model_name, "type": "borehole_facies", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(obs), "predicted": float(value)})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def evaluate_conditioned_posterior(
    posterior: dict[str, np.ndarray],
    holdout: ObservationTable,
    model_name: str = "COLDReconArcticDataDiffusion",
    eic_field: str = "eic_mean",
    facies_field: str = "facies_mode",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = filter_observations_to_posterior(holdout, posterior)
    row: dict[str, Any] = {"model": model_name}
    predictions: list[dict[str, Any]] = []
    if obs.n_obs == 0:
        return pd.DataFrame([row]), pd.DataFrame()
    ix, iy, iz = nearest_indices(obs.coords, posterior)
    eic = obs.type_ids == OBS_TYPES["borehole_eic"]
    if np.any(eic):
        pred = np.clip(posterior[eic_field][ix[eic], iy[eic], iz[eic]], 0.0, 1.0)
        truth = np.clip(obs.values[eic], 0.0, 1.0)
        row.update(_eic_scores(truth, pred))
        for coord, observed, value in zip(obs.coords[eic], truth, pred):
            predictions.append({"model": model_name, "type": "borehole_eic", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(observed), "predicted": float(value)})
    facies = obs.type_ids == OBS_TYPES["borehole_facies"]
    if np.any(facies) and facies_field in posterior:
        pred_cls = posterior[facies_field][ix[facies], iy[facies], iz[facies]].astype(np.int64)
        truth_cls = np.rint(obs.values[facies]).astype(np.int64)
        row.update(_classification_scores(truth_cls, pred_cls))
        for coord, observed, value in zip(obs.coords[facies], truth_cls, pred_cls):
            predictions.append({"model": model_name, "type": "borehole_facies", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(observed), "predicted": float(value)})
    return pd.DataFrame([row]), pd.DataFrame(predictions)


def evaluate_hybrid_calibrated_posterior(
    posterior: dict[str, np.ndarray],
    train: ObservationTable,
    holdout: ObservationTable,
    model_name: str = "COLDReconArcticDataHybridCalibrated",
    eic_field: str = "eic_event_calibrated_mean",
    wedge_probability_threshold: float = 0.80,
    knn_k: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = filter_observations_to_posterior(holdout, posterior)
    row: dict[str, Any] = {"model": model_name}
    predictions: list[dict[str, Any]] = []
    if obs.n_obs == 0:
        return pd.DataFrame([row]), pd.DataFrame()
    ix, iy, iz = nearest_indices(obs.coords, posterior)
    eic = obs.type_ids == OBS_TYPES["borehole_eic"]
    if np.any(eic) and eic_field in posterior:
        pred = np.clip(posterior[eic_field][ix[eic], iy[eic], iz[eic]], 0.0, 1.0)
        truth = np.clip(obs.values[eic], 0.0, 1.0)
        row.update(_eic_scores(truth, pred))
        for coord, observed, value in zip(obs.coords[eic], truth, pred):
            predictions.append({"model": model_name, "type": "borehole_eic", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(observed), "predicted": float(value)})
    facies = obs.type_ids == OBS_TYPES["borehole_facies"]
    train_facies = train.type_ids == OBS_TYPES["borehole_facies"]
    if np.any(facies) and np.any(train_facies) and "facies_probability" in posterior:
        knn = _knn_class(
            train.coords[train_facies],
            np.rint(train.values[train_facies]).astype(np.int64),
            obs.coords[facies],
            k=knn_k,
        )
        wedge_prob = posterior["facies_probability"][ix[facies], iy[facies], iz[facies], 6]
        pred_cls = np.where(wedge_prob >= float(wedge_probability_threshold), 6, knn).astype(np.int64)
        truth_cls = np.rint(obs.values[facies]).astype(np.int64)
        row.update(_classification_scores(truth_cls, pred_cls))
        for coord, observed, value in zip(obs.coords[facies], truth_cls, pred_cls):
            predictions.append({"model": model_name, "type": "borehole_facies", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(observed), "predicted": float(value)})
    return pd.DataFrame([row]), pd.DataFrame(predictions)


def evaluate_adaptive_hybrid_posterior(
    posterior: dict[str, np.ndarray],
    train: ObservationTable,
    holdout: ObservationTable,
    model_name: str = "COLDReconArcticDataAdaptiveHybrid",
    eic_field: str = "eic_mean",
    wedge_probability_threshold: float = 0.80,
    knn_k: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = filter_observations_to_posterior(holdout, posterior)
    row: dict[str, Any] = {"model": model_name}
    predictions: list[dict[str, Any]] = []
    if obs.n_obs == 0:
        return pd.DataFrame([row]), pd.DataFrame()
    eic = obs.type_ids == OBS_TYPES["borehole_eic"]
    if np.any(eic):
        pred, info = adaptive_eic_predictions(posterior, train, obs, eic_field=eic_field)
        truth = np.clip(obs.values[eic], 0.0, 1.0)
        row.update(_eic_scores(truth, pred))
        row.update(info)
        for coord, observed, value in zip(obs.coords[eic], truth, pred):
            predictions.append({"model": model_name, "type": "borehole_eic", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(observed), "predicted": float(value)})
    facies = obs.type_ids == OBS_TYPES["borehole_facies"]
    train_facies = train.type_ids == OBS_TYPES["borehole_facies"]
    if np.any(facies) and np.any(train_facies):
        pred_cls, info = adaptive_facies_predictions(
            posterior,
            train,
            obs,
            wedge_probability_threshold=wedge_probability_threshold,
            knn_k=knn_k,
        )
        truth_cls = np.rint(obs.values[facies]).astype(np.int64)
        row.update(_classification_scores(truth_cls, pred_cls))
        row.update(info)
        for coord, observed, value in zip(obs.coords[facies], truth_cls, pred_cls):
            predictions.append({"model": model_name, "type": "borehole_facies", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(observed), "predicted": float(value)})
    return pd.DataFrame([row]), pd.DataFrame(predictions)


def evaluate_wedge_recall_posterior(
    posterior: dict[str, np.ndarray],
    train: ObservationTable,
    holdout: ObservationTable,
    model_name: str = "COLDReconArcticDataWedgeRecallHead",
    knn_k: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = filter_observations_to_posterior(holdout, posterior)
    row: dict[str, Any] = {"model": model_name}
    predictions: list[dict[str, Any]] = []
    if obs.n_obs == 0:
        return pd.DataFrame([row]), pd.DataFrame()
    facies = obs.type_ids == OBS_TYPES["borehole_facies"]
    train_facies = train.type_ids == OBS_TYPES["borehole_facies"]
    if np.any(facies) and np.any(train_facies):
        pred_cls, info = adaptive_wedge_recall_predictions(posterior, train, obs, knn_k=knn_k)
        truth_cls = np.rint(obs.values[facies]).astype(np.int64)
        row.update(_classification_scores(truth_cls, pred_cls))
        row.update(info)
        for coord, observed, value in zip(obs.coords[facies], truth_cls, pred_cls):
            predictions.append({"model": model_name, "type": "borehole_facies", "x": coord[0], "y": coord[1], "z": coord[2], "observed": float(observed), "predicted": float(value)})
    return pd.DataFrame([row]), pd.DataFrame(predictions)


def apply_cryofacies_eic_prior(
    posterior: dict[str, np.ndarray],
    token_index: pd.DataFrame,
    weight: float = 0.30,
    n_facies: int = 7,
) -> dict[str, np.ndarray]:
    """Blend EIC with an empirical ArcticData cryofacies-conditioned prior.

    The prior is learned only from conditioning tokens, using EIC samples grouped by
    cryofacies class, and then projected through the posterior facies probabilities.
    """
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return posterior
    eic_rows = token_index[token_index["type_id"].astype(int) == OBS_TYPES["borehole_eic"]].copy()
    eic_rows["facies_id"] = eic_rows["cryofacies_class"].map(CRYOFACIES_TO_MODEL_FACIES)
    eic_rows = eic_rows[eic_rows["facies_id"].notna()].copy()
    eic_rows["value"] = pd.to_numeric(eic_rows["value"], errors="coerce")
    eic_rows = eic_rows[eic_rows["value"].notna()]
    if eic_rows.empty:
        return posterior
    global_mean = float(np.clip(eic_rows["value"].mean(), 0.0, 1.0))
    class_prior = np.full((n_facies,), global_mean, dtype=np.float32)
    for facies_id, group in eic_rows.groupby("facies_id"):
        class_prior[int(facies_id)] = float(np.clip(group["value"].mean(), 0.0, 1.0))
    class_prior[6] = max(float(class_prior[6]), 0.70)
    out = {key: np.asarray(value).copy() for key, value in posterior.items()}
    if "facies_probability" in out:
        prior_grid = np.tensordot(out["facies_probability"].astype(np.float32), class_prior, axes=([-1], [0])).astype(np.float32)
    elif "facies_mode" in out:
        prior_grid = class_prior[np.clip(out["facies_mode"].astype(int), 0, n_facies - 1)].astype(np.float32)
    else:
        return posterior
    if "eic_samples" in out:
        out["eic_samples"] = ((1.0 - weight) * out["eic_samples"] + weight * prior_grid[None, ...]).astype(np.float32)
        out["eic_mean"] = out["eic_samples"].mean(axis=0).astype(np.float32)
        out["eic_std"] = out["eic_samples"].std(axis=0).astype(np.float32)
    elif "eic_mean" in out:
        out["eic_mean"] = ((1.0 - weight) * out["eic_mean"] + weight * prior_grid).astype(np.float32)
    out["arcticdata_cryofacies_eic_prior"] = prior_grid.astype(np.float32)
    out["arcticdata_cryofacies_eic_prior_weight"] = np.asarray(weight, dtype=np.float32)
    out["arcticdata_cryofacies_eic_class_prior"] = class_prior.astype(np.float32)
    if "eic_samples" in out:
        out["ice_rich_probability"] = np.mean(out["eic_samples"] > 0.30, axis=0).astype(np.float32)
    elif "eic_mean" in out:
        out["ice_rich_probability"] = (out["eic_mean"] > 0.30).astype(np.float32)
    return out
