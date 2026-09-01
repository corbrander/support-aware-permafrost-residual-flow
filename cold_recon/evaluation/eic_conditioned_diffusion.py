from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.spatial import cKDTree

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.evaluation.physics_consistency import empirical_log_resistivity_np, empirical_unfrozen_water_np, facies_to_probability
from cold_recon.evaluation.physics_refinement import PhysicsRefinementConfig, smooth_temperature_field
from cold_recon.evaluation.real_conditioned_diffusion import nearest_indices
from cold_recon.evaluation.uncertainty import facies_entropy


@dataclass(frozen=True)
class EICProxyConfig:
    target_shape: tuple[int, int, int] = (64, 64, 48)
    x_pad_m: float = 10.0
    y_half_width_m: float = 40.0
    z_pad_m: float = 0.25
    min_zmax_m: float = 3.0
    active_layer_m: float = 0.60
    horizontal_scale_m: float = 20.0
    depth_scale_m: float = 0.25
    idw_k: int = 8
    depth_k: int = 12
    eic_sigma_fraction: float = 0.05
    high_eic_threshold: float = 0.30


def split_eic_observations_by_borehole(
    observations: ObservationTable,
    borehole_ids: np.ndarray,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[ObservationTable, ObservationTable, np.ndarray, np.ndarray]:
    borehole_ids = np.asarray(borehole_ids).astype(str)
    if len(borehole_ids) != observations.n_obs:
        raise ValueError("borehole_ids length must match observations")
    unique = np.unique(borehole_ids)
    if len(unique) < 2:
        raise ValueError("At least two boreholes are required for hold-out splitting")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_hold = max(1, int(round(len(shuffled) * float(holdout_fraction))))
    if n_hold >= len(shuffled):
        n_hold = max(1, len(shuffled) // 3)
    holdout_boreholes = np.sort(shuffled[:n_hold])
    train_boreholes = np.sort(shuffled[n_hold:])
    hold_mask = np.isin(borehole_ids, holdout_boreholes)
    train_mask = ~hold_mask
    return (
        observations.subset(np.where(train_mask)[0]),
        observations.subset(np.where(hold_mask)[0]),
        train_boreholes,
        holdout_boreholes,
    )


def _idw_values_and_distance(
    train_coords: np.ndarray,
    train_values: np.ndarray,
    query_coords: np.ndarray,
    cfg: EICProxyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if len(train_values) == 0:
        raise ValueError("Cannot build EIC proxy without training observations")
    train_features = np.column_stack(
        [
            train_coords[:, 0] / float(cfg.horizontal_scale_m),
            train_coords[:, 1] / float(cfg.horizontal_scale_m),
            train_coords[:, 2] / float(cfg.depth_scale_m),
        ]
    ).astype(np.float32)
    query_features = np.column_stack(
        [
            query_coords[:, 0] / float(cfg.horizontal_scale_m),
            query_coords[:, 1] / float(cfg.horizontal_scale_m),
            query_coords[:, 2] / float(cfg.depth_scale_m),
        ]
    ).astype(np.float32)
    k = min(max(1, int(cfg.idw_k)), len(train_values))
    dist, idx = cKDTree(train_features).query(query_features, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.square(dist + 1e-6)
    values = (weights * train_values[idx]).sum(axis=1) / weights.sum(axis=1)
    return values.astype(np.float32), dist[:, 0].astype(np.float32)


def _idw_values(train_coords: np.ndarray, train_values: np.ndarray, query_coords: np.ndarray, cfg: EICProxyConfig) -> np.ndarray:
    values, _ = _idw_values_and_distance(train_coords, train_values, query_coords, cfg)
    return values


def _idw_query_features(train_features: np.ndarray, train_values: np.ndarray, query_features: np.ndarray, k: int) -> np.ndarray:
    if len(train_values) == 0:
        raise ValueError("Cannot predict EIC with no training observations")
    k = min(max(1, int(k)), len(train_values))
    dist, idx = cKDTree(train_features).query(query_features, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.square(dist + 1e-6)
    return ((weights * train_values[idx]).sum(axis=1) / weights.sum(axis=1)).astype(np.float32)


def _eic_metric_row(obs: np.ndarray, pred: np.ndarray, model_name: str, cfg: EICProxyConfig) -> dict[str, float | str]:
    obs = np.clip(np.asarray(obs, dtype=np.float32), 0.0, 1.0)
    pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)
    err = pred - obs
    obs_event = obs >= float(cfg.high_eic_threshold)
    pred_event = pred >= float(cfg.high_eic_threshold)
    tp = float(np.sum(obs_event & pred_event))
    fp = float(np.sum(~obs_event & pred_event))
    fn = float(np.sum(obs_event & ~pred_event))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0.0 else np.nan
    corr = np.corrcoef(obs, pred)[0, 1] if obs.size > 1 and np.std(obs) > 0.0 and np.std(pred) > 0.0 else np.nan
    return {
        "model": model_name,
        "eic_n": float(obs.size),
        "observed_mean_eic": float(np.mean(obs)) if obs.size else np.nan,
        "predicted_mean_eic": float(np.mean(pred)) if pred.size else np.nan,
        "eic_bias": float(np.mean(err)) if err.size else np.nan,
        "eic_mae": float(np.mean(np.abs(err))) if err.size else np.nan,
        "eic_rmse": float(np.sqrt(np.mean(err**2))) if err.size else np.nan,
        "eic_normalized_rmse": float(np.sqrt(np.mean((err / float(cfg.eic_sigma_fraction)) ** 2))) if err.size else np.nan,
        "eic_pearson_r": float(corr) if np.isfinite(corr) else np.nan,
        "high_eic_threshold": float(cfg.high_eic_threshold),
        "high_eic_accuracy": float(np.mean(obs_event == pred_event)) if obs.size else np.nan,
        "high_eic_precision": float(precision) if np.isfinite(precision) else np.nan,
        "high_eic_recall": float(recall) if np.isfinite(recall) else np.nan,
        "high_eic_f1": float(f1) if np.isfinite(f1) else np.nan,
    }


def high_eic_event_scores(
    obs: np.ndarray,
    pred: np.ndarray,
    observed_threshold: float = 0.30,
    prediction_threshold: float | None = None,
) -> dict[str, float]:
    """Binary high-EIC event scores with separate observed and predicted thresholds."""

    obs = np.clip(np.asarray(obs, dtype=np.float32), 0.0, 1.0)
    pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)
    pred_threshold = float(observed_threshold if prediction_threshold is None else prediction_threshold)
    obs_event = obs >= float(observed_threshold)
    pred_event = pred >= pred_threshold
    tp = float(np.sum(obs_event & pred_event))
    fp = float(np.sum(~obs_event & pred_event))
    fn = float(np.sum(obs_event & ~pred_event))
    tn = float(np.sum(~obs_event & ~pred_event))
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan
    f2 = 5.0 * precision * recall / (4.0 * precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (4.0 * precision + recall) > 0 else np.nan
    return {
        "high_eic_threshold": float(observed_threshold),
        "high_eic_prediction_threshold": float(pred_threshold),
        "high_eic_accuracy": float((tp + tn) / max(float(obs.size), 1.0)) if obs.size else np.nan,
        "high_eic_precision": float(precision) if np.isfinite(precision) else np.nan,
        "high_eic_recall": float(recall) if np.isfinite(recall) else np.nan,
        "high_eic_f1": float(f1) if np.isfinite(f1) else np.nan,
        "high_eic_f2": float(f2) if np.isfinite(f2) else np.nan,
        "high_eic_tp": float(tp),
        "high_eic_fp": float(fp),
        "high_eic_fn": float(fn),
        "high_eic_tn": float(tn),
    }


def calibrate_high_eic_prediction_threshold(
    obs: np.ndarray,
    pred: np.ndarray,
    observed_threshold: float = 0.30,
    beta: float = 2.0,
    score_tolerance: float = 0.06,
) -> dict[str, float | str]:
    """Choose a training-split event threshold for recall-oriented high-EIC screening.

    The observed high-EIC definition remains fixed at ``observed_threshold``. Only the
    model prediction threshold is tuned on the training split. Ties are resolved toward
    the lowest threshold, making the calibrated head conservative for screening.
    """

    obs = np.clip(np.asarray(obs, dtype=np.float32), 0.0, 1.0)
    pred = np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)
    if obs.size == 0 or np.sum(obs >= observed_threshold) == 0 or np.unique(pred).size < 2:
        scores = high_eic_event_scores(obs, pred, observed_threshold, observed_threshold)
        return {**scores, "high_eic_threshold_source": "fixed_no_train_event"}
    values = np.unique(pred[np.isfinite(pred)])
    values = np.sort(values)
    mids = (values[:-1] + values[1:]) / 2.0
    candidates = np.unique(np.clip(np.concatenate([[0.0, float(observed_threshold)], values, mids]), 0.0, 1.0))
    scored: list[tuple[float, float, dict[str, float]]] = []
    best: dict[str, float] | None = None
    best_score = -np.inf
    beta2 = float(beta) ** 2
    for threshold in candidates:
        scores = high_eic_event_scores(obs, pred, observed_threshold, float(threshold))
        precision = scores["high_eic_precision"]
        recall = scores["high_eic_recall"]
        if np.isfinite(precision) and np.isfinite(recall) and (beta2 * precision + recall) > 0:
            score = (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
        else:
            score = -np.inf
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and best is not None and threshold < best["high_eic_prediction_threshold"]):
            best_score = float(score)
            best = scores
        if np.isfinite(score):
            scored.append((float(score), float(scores["high_eic_recall"]) if np.isfinite(scores["high_eic_recall"]) else -np.inf, scores))
    if best is None:
        best = high_eic_event_scores(obs, pred, observed_threshold, observed_threshold)
        return {**best, "high_eic_threshold_source": "fixed_unscored"}
    if scored:
        max_recall = max(item[1] for item in scored)
        tolerant = [
            item[2]
            for item in scored
            if item[0] >= best_score - float(score_tolerance) and item[1] >= max_recall - 1e-12
        ]
        if tolerant:
            best = min(tolerant, key=lambda item: item["high_eic_prediction_threshold"])
            return {
                **best,
                "high_eic_threshold_source": f"train_split_f{float(beta):.1f}_recall_tolerant_low",
                "high_eic_train_score_tolerance": float(score_tolerance),
            }
    return {
        **best,
        "high_eic_threshold_source": f"train_split_f{float(beta):.1f}_recall_tie_low",
        "high_eic_train_score_tolerance": float(score_tolerance),
    }


def apply_calibrated_high_eic_screening(
    train_predictions: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    model_name: str,
    observed_threshold: float = 0.30,
    beta: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update one model row with a training-calibrated high-EIC event head."""

    if train_predictions.empty or holdout_metrics.empty or holdout_predictions.empty:
        return holdout_metrics, holdout_predictions
    train_rows = train_predictions[train_predictions["model"].astype(str) == str(model_name)]
    holdout_rows = holdout_predictions[holdout_predictions["model"].astype(str) == str(model_name)]
    if train_rows.empty or holdout_rows.empty:
        return holdout_metrics, holdout_predictions
    calibration = calibrate_high_eic_prediction_threshold(
        train_rows["observed_eic"].to_numpy(dtype=np.float32),
        train_rows["predicted_eic"].to_numpy(dtype=np.float32),
        observed_threshold=observed_threshold,
        beta=beta,
    )
    fixed_scores = high_eic_event_scores(
        holdout_rows["observed_eic"].to_numpy(dtype=np.float32),
        holdout_rows["predicted_eic"].to_numpy(dtype=np.float32),
        observed_threshold=observed_threshold,
        prediction_threshold=observed_threshold,
    )
    holdout_scores = high_eic_event_scores(
        holdout_rows["observed_eic"].to_numpy(dtype=np.float32),
        holdout_rows["predicted_eic"].to_numpy(dtype=np.float32),
        observed_threshold=observed_threshold,
        prediction_threshold=float(calibration["high_eic_prediction_threshold"]),
    )
    holdout_metrics = holdout_metrics.copy()
    metric_rows = holdout_metrics["model"].astype(str) == str(model_name)
    for key in ["high_eic_accuracy", "high_eic_precision", "high_eic_recall", "high_eic_f1", "high_eic_f2"]:
        holdout_metrics.loc[metric_rows, f"{key}_fixed_0p30"] = fixed_scores[key]
        holdout_metrics.loc[metric_rows, key] = holdout_scores[key]
    for key in ["high_eic_tp", "high_eic_fp", "high_eic_fn", "high_eic_tn"]:
        holdout_metrics.loc[metric_rows, key] = holdout_scores[key]
    holdout_metrics.loc[metric_rows, "high_eic_threshold"] = float(observed_threshold)
    holdout_metrics.loc[metric_rows, "high_eic_prediction_threshold"] = float(calibration["high_eic_prediction_threshold"])
    holdout_metrics.loc[metric_rows, "high_eic_threshold_source"] = str(calibration["high_eic_threshold_source"])
    holdout_metrics.loc[metric_rows, "train_high_eic_f2"] = float(calibration["high_eic_f2"])
    if "high_eic_train_score_tolerance" in calibration:
        holdout_metrics.loc[metric_rows, "high_eic_train_score_tolerance"] = float(calibration["high_eic_train_score_tolerance"])
    holdout_predictions = holdout_predictions.copy()
    mask = holdout_predictions["model"].astype(str) == str(model_name)
    holdout_predictions.loc[mask, "high_eic_observed"] = holdout_predictions.loc[mask, "observed_eic"].astype(float) >= float(observed_threshold)
    holdout_predictions.loc[mask, "high_eic_predicted_fixed_0p30"] = holdout_predictions.loc[mask, "predicted_eic"].astype(float) >= float(observed_threshold)
    holdout_predictions.loc[mask, "high_eic_predicted_calibrated"] = holdout_predictions.loc[mask, "predicted_eic"].astype(float) >= float(calibration["high_eic_prediction_threshold"])
    holdout_predictions.loc[mask, "high_eic_prediction_threshold"] = float(calibration["high_eic_prediction_threshold"])
    return holdout_metrics, holdout_predictions


def eic_conditioning_baseline_rows(
    train: ObservationTable,
    holdout: ObservationTable,
    config: EICProxyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or EICProxyConfig()
    train_mask = train.type_ids == OBS_TYPES["borehole_eic"]
    holdout_mask = holdout.type_ids == OBS_TYPES["borehole_eic"]
    train_values = np.clip(train.values[train_mask], 0.0, 1.0).astype(np.float32)
    holdout_values = np.clip(holdout.values[holdout_mask], 0.0, 1.0).astype(np.float32)
    train_coords = train.coords[train_mask]
    holdout_coords = holdout.coords[holdout_mask]
    if train_values.size == 0 or holdout_values.size == 0:
        return pd.DataFrame(), pd.DataFrame()
    train_features = np.column_stack(
        [
            train_coords[:, 0] / float(cfg.horizontal_scale_m),
            train_coords[:, 1] / float(cfg.horizontal_scale_m),
            train_coords[:, 2] / float(cfg.depth_scale_m),
        ]
    ).astype(np.float32)
    holdout_features = np.column_stack(
        [
            holdout_coords[:, 0] / float(cfg.horizontal_scale_m),
            holdout_coords[:, 1] / float(cfg.horizontal_scale_m),
            holdout_coords[:, 2] / float(cfg.depth_scale_m),
        ]
    ).astype(np.float32)
    train_depth = train_coords[:, 2:3] / float(cfg.depth_scale_m)
    holdout_depth = holdout_coords[:, 2:3] / float(cfg.depth_scale_m)
    models = {
        "GlobalMean": np.full(holdout_values.shape[0], float(train_values.mean()), dtype=np.float32),
        "DepthIDW": np.clip(_idw_query_features(train_depth, train_values, holdout_depth, cfg.depth_k), 0.0, 1.0),
        "SpatialDepthIDW": np.clip(_idw_query_features(train_features, train_values, holdout_features, cfg.idw_k), 0.0, 1.0),
    }
    metric_rows: list[dict[str, float | str]] = []
    prediction_rows: list[dict[str, float | str]] = []
    for model_name, pred in models.items():
        metric_rows.append(_eic_metric_row(holdout_values, pred, model_name, cfg))
        for idx, (coord, observed, value) in enumerate(zip(holdout_coords, holdout_values, pred)):
            prediction_rows.append(
                {
                    "model": model_name,
                    "holdout_index": float(idx),
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "z": float(coord[2]),
                    "observed_eic": float(observed),
                    "predicted_eic": float(value),
                    "error": float(value - observed),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def make_eic_proxy_fields(
    train_observations: ObservationTable,
    reference_observations: ObservationTable | None = None,
    config: EICProxyConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or EICProxyConfig()
    reference = reference_observations or train_observations
    eic_mask = train_observations.type_ids == OBS_TYPES["borehole_eic"]
    if not np.any(eic_mask):
        raise ValueError("EIC proxy requires borehole_eic observations")
    ref_coords = reference.coords
    train_coords = train_observations.coords[eic_mask]
    train_values = np.clip(train_observations.values[eic_mask], 0.0, 1.0).astype(np.float32)
    x_min = float(np.nanmin(ref_coords[:, 0])) - float(cfg.x_pad_m)
    x_max = float(np.nanmax(ref_coords[:, 0])) + float(cfg.x_pad_m)
    z_max = max(float(np.nanmax(ref_coords[:, 2])) + float(cfg.z_pad_m), float(cfg.min_zmax_m))
    nx, ny, nz = cfg.target_shape
    grid_x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    grid_y = np.linspace(-float(cfg.y_half_width_m), float(cfg.y_half_width_m), ny, dtype=np.float32)
    grid_z = np.linspace(0.0, z_max, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    query = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    eic_values, nearest_scaled_distance = _idw_values_and_distance(train_coords, train_values, query, cfg)
    eic = np.clip(eic_values.reshape(cfg.target_shape), 0.0, 1.0).astype(np.float32)
    eic_proxy_std = np.clip(0.04 + 0.035 * nearest_scaled_distance.reshape(cfg.target_shape), 0.04, 0.28).astype(np.float32)

    active_layer = float(cfg.active_layer_m)
    temperature = np.where(
        zz <= active_layer,
        0.6 * (1.0 - zz / max(active_layer, 1e-6)),
        -0.25 - 0.45 * (zz - active_layer),
    ).astype(np.float32)
    temperature = (temperature - 1.1 * eic * (zz > active_layer)).astype(np.float32)
    frozen = np.maximum(np.abs(temperature), 0.08)
    unfrozen = np.where(temperature >= 0.0, 0.42, 0.06 + 0.09 / np.power(frozen, 0.45))
    unfrozen = np.clip(unfrozen * (1.0 - 0.45 * eic), 0.0, 0.8).astype(np.float32)
    log_resistivity = np.clip(4.8 + 2.2 * eic + 0.25 * np.maximum(-temperature, 0.0) - 1.2 * unfrozen, 2.0, 10.0).astype(np.float32)

    facies = np.full(cfg.target_shape, 2, dtype=np.int16)
    facies[zz <= active_layer] = 0
    facies[(zz <= 0.35) & (eic < 0.08)] = 1
    facies[(zz > active_layer) & (eic >= cfg.high_eic_threshold)] = 3
    facies[(zz > active_layer) & (eic >= 0.75)] = 6
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "facies_mode": facies,
        "eic_mean": eic,
        "eic_std": eic_proxy_std,
        "temperature_mean": temperature,
        "unfrozen_water_mean": unfrozen,
        "log_resistivity_mean": log_resistivity,
        "ice_rich_probability": (1.0 - ndtr((cfg.high_eic_threshold - eic) / np.maximum(eic_proxy_std, 1e-6))).astype(np.float32),
    }


def blend_eic_posterior_with_proxy(
    posterior: dict[str, np.ndarray],
    proxy: dict[str, np.ndarray],
    eic_weight: float = 0.85,
    physics_weight: float = 0.50,
    facies_weight: float = 0.45,
    n_facies: int = 7,
) -> dict[str, np.ndarray]:
    out = dict(posterior)
    weights = {
        "eic": float(np.clip(eic_weight, 0.0, 1.0)),
        "temperature": float(np.clip(physics_weight, 0.0, 1.0)),
        "unfrozen_water": float(np.clip(physics_weight, 0.0, 1.0)),
        "log_resistivity": float(np.clip(physics_weight, 0.0, 1.0)),
    }
    proxy_keys = {
        "eic": "eic_mean",
        "temperature": "temperature_mean",
        "unfrozen_water": "unfrozen_water_mean",
        "log_resistivity": "log_resistivity_mean",
    }
    for sample_key, proxy_key in proxy_keys.items():
        samples_name = f"{sample_key}_samples"
        mean_name = f"{sample_key}_mean"
        std_name = f"{sample_key}_std"
        weight = weights[sample_key]
        if samples_name in out:
            blended = weight * proxy[proxy_key][None, ...] + (1.0 - weight) * out[samples_name]
            out[samples_name] = blended.astype(np.float32)
            out[mean_name] = blended.mean(axis=0).astype(np.float32)
            out[std_name] = blended.std(axis=0).astype(np.float32)
        elif mean_name in out:
            out[mean_name] = (weight * proxy[proxy_key] + (1.0 - weight) * out[mean_name]).astype(np.float32)
    if "eic_std" in out and "eic_std" in proxy:
        out["eic_std"] = np.maximum(out["eic_std"], proxy["eic_std"]).astype(np.float32)
    if "facies_probability" in out:
        one_hot = np.zeros((*proxy["facies_mode"].shape, n_facies), dtype=np.float32)
        for cls in range(n_facies):
            one_hot[..., cls] = proxy["facies_mode"] == cls
        weight = float(np.clip(facies_weight, 0.0, 1.0))
        out["facies_probability"] = (weight * one_hot + (1.0 - weight) * out["facies_probability"]).astype(np.float32)
        out["facies_entropy"] = facies_entropy(out["facies_probability"]).astype(np.float32)
        out["facies_mode"] = np.argmax(out["facies_probability"], axis=-1).astype(np.int16)
    if "eic_mean" in out and "eic_std" in out:
        std = np.maximum(out["eic_std"], 1e-6)
        out["ice_rich_probability"] = (1.0 - ndtr((0.30 - out["eic_mean"]) / std)).astype(np.float32)
    elif "eic_samples" in out:
        out["ice_rich_probability"] = np.mean(out["eic_samples"] >= 0.30, axis=0).astype(np.float32)
    elif "eic_mean" in out:
        out["ice_rich_probability"] = (out["eic_mean"] >= 0.30).astype(np.float32)
    return out


def project_eic_conditioned_physics(
    posterior: dict[str, np.ndarray],
    n_facies: int = 7,
    heat_iterations: int = 32,
    heat_strength: float = 0.45,
    heat_anchor: float = 0.0,
) -> dict[str, np.ndarray]:
    out = dict(posterior)
    cfg = PhysicsRefinementConfig(
        heat_iterations=int(heat_iterations),
        heat_strength=float(heat_strength),
        heat_anchor=float(heat_anchor),
        unfrozen_weight=1.0,
        resistivity_weight=1.0,
    )
    if "temperature_samples" in out and "facies_samples" in out:
        temp_samples = []
        uw_samples = []
        rho_samples = []
        res_samples = []
        facies_samples = out["facies_samples"].astype(np.int16)
        eic_samples = out["eic_samples"] if "eic_samples" in out else np.repeat(out["eic_mean"][None, ...], facies_samples.shape[0], axis=0)
        for idx in range(facies_samples.shape[0]):
            facies_probs = facies_to_probability(facies_samples[idx], n_facies=n_facies)
            temp = smooth_temperature_field(out["temperature_samples"][idx], cfg)
            uw = empirical_unfrozen_water_np(temp, facies_probs)
            rho = empirical_log_resistivity_np(eic_samples[idx], temp, uw, facies_probs)
            temp_samples.append(temp.astype(np.float32))
            uw_samples.append(uw.astype(np.float32))
            rho_samples.append(rho.astype(np.float32))
            res_samples.append(np.exp(np.clip(rho, 0.0, 12.0)).astype(np.float32))
        for key, values in {
            "temperature": temp_samples,
            "unfrozen_water": uw_samples,
            "log_resistivity": rho_samples,
            "resistivity": res_samples,
        }.items():
            arr = np.stack(values, axis=0).astype(np.float32)
            out[f"{key}_samples"] = arr
            out[f"{key}_mean"] = arr.mean(axis=0).astype(np.float32)
            out[f"{key}_std"] = arr.std(axis=0).astype(np.float32)
    else:
        if "facies_probability" in out:
            facies_probs = out["facies_probability"].astype(np.float32)
        elif "facies_mode" in out:
            facies_probs = facies_to_probability(out["facies_mode"], n_facies=n_facies)
        else:
            raise KeyError("Posterior must contain facies_probability or facies_mode for EIC physics projection")
        temp = smooth_temperature_field(out["temperature_mean"], cfg)
        uw = empirical_unfrozen_water_np(temp, facies_probs)
        rho = empirical_log_resistivity_np(out["eic_mean"], temp, uw, facies_probs)
        out["temperature_mean"] = temp.astype(np.float32)
        out["unfrozen_water_mean"] = uw.astype(np.float32)
        out["log_resistivity_mean"] = rho.astype(np.float32)
        out["resistivity_mean"] = np.exp(np.clip(rho, 0.0, 12.0)).astype(np.float32)
    out["eic_physics_projection_heat_iterations"] = np.asarray(heat_iterations, dtype=np.int32)
    out["eic_physics_projection_heat_strength"] = np.asarray(heat_strength, dtype=np.float32)
    out["eic_physics_projection_heat_anchor"] = np.asarray(heat_anchor, dtype=np.float32)
    return out


def evaluate_eic_observation_consistency(
    posterior: dict[str, np.ndarray],
    observations: ObservationTable,
    prefix: str = "eic",
    sigma_fraction: float = 0.05,
    high_eic_threshold: float = 0.30,
) -> dict[str, float]:
    grid = {key: posterior[key] for key in ["grid_x", "grid_y", "grid_z"]}
    coords = observations.coords
    keep = (
        (coords[:, 0] >= float(grid["grid_x"].min()))
        & (coords[:, 0] <= float(grid["grid_x"].max()))
        & (coords[:, 1] >= float(grid["grid_y"].min()))
        & (coords[:, 1] <= float(grid["grid_y"].max()))
        & (coords[:, 2] >= float(grid["grid_z"].min()))
        & (coords[:, 2] <= float(grid["grid_z"].max()))
        & (observations.type_ids == OBS_TYPES["borehole_eic"])
    )
    idx = np.where(keep)[0]
    metrics: dict[str, float] = {f"{prefix}_n": float(len(idx))}
    if len(idx) == 0:
        return metrics
    ix, iy, iz = nearest_indices(observations.coords[idx], grid)
    pred = posterior["eic_mean"][ix, iy, iz]
    obs = np.clip(observations.values[idx], 0.0, 1.0)
    err = pred - obs
    obs_event = obs >= high_eic_threshold
    pred_event = pred >= high_eic_threshold
    tp = float(np.sum(obs_event & pred_event))
    fn = float(np.sum(obs_event & ~pred_event))
    fp = float(np.sum(~obs_event & pred_event))
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    metrics.update(
        {
            f"{prefix}_bias": float(np.mean(err)),
            f"{prefix}_mae": float(np.mean(np.abs(err))),
            f"{prefix}_rmse": float(np.sqrt(np.mean(err**2))),
            f"{prefix}_normalized_rmse": float(np.sqrt(np.mean((err / float(sigma_fraction)) ** 2))),
            f"{prefix}_high_eic_accuracy": float(np.mean(obs_event == pred_event)),
            f"{prefix}_high_eic_precision": float(precision) if np.isfinite(precision) else np.nan,
            f"{prefix}_high_eic_recall": float(recall) if np.isfinite(recall) else np.nan,
        }
    )
    return metrics


def eic_posterior_prediction_rows(
    posterior: dict[str, np.ndarray],
    observations: ObservationTable,
    model_name: str = "COLDReconUSGSEICConditionedDiffusion",
    config: EICProxyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or EICProxyConfig()
    grid = {key: posterior[key] for key in ["grid_x", "grid_y", "grid_z"]}
    coords = observations.coords
    keep = (
        (coords[:, 0] >= float(grid["grid_x"].min()))
        & (coords[:, 0] <= float(grid["grid_x"].max()))
        & (coords[:, 1] >= float(grid["grid_y"].min()))
        & (coords[:, 1] <= float(grid["grid_y"].max()))
        & (coords[:, 2] >= float(grid["grid_z"].min()))
        & (coords[:, 2] <= float(grid["grid_z"].max()))
        & (observations.type_ids == OBS_TYPES["borehole_eic"])
    )
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return pd.DataFrame([{"model": model_name, "eic_n": 0.0}]), pd.DataFrame()
    ix, iy, iz = nearest_indices(observations.coords[idx], grid)
    pred = np.clip(posterior["eic_mean"][ix, iy, iz], 0.0, 1.0).astype(np.float32)
    obs = np.clip(observations.values[idx], 0.0, 1.0).astype(np.float32)
    metrics = pd.DataFrame([_eic_metric_row(obs, pred, model_name, cfg)])
    predictions = pd.DataFrame(
        [
            {
                "model": model_name,
                "holdout_index": float(obs_idx),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
                "observed_eic": float(observed),
                "predicted_eic": float(value),
                "error": float(value - observed),
            }
            for obs_idx, coord, observed, value in zip(idx, observations.coords[idx], obs, pred)
        ]
    )
    return metrics, predictions
