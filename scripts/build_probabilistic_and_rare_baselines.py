from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from cold_recon.baselines.idw import idw_interpolate, reconstruct_idw
from cold_recon.baselines.random_forest import _grid_points_and_surface, _surface_at_obs
from cold_recon.data.data_schema import OBS_TYPES, ObservationTable, load_sample_npz
from cold_recon.evaluation.metrics import facies_iou, rmse
from cold_recon.models.diffusion import GaussianDiffusion3D
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.training.train_diffusion import _load_autoencoder, _posterior_arrays
from cold_recon.training.train_multisample_diffusion import _proxy_from_observations, subsample_observations_by_type
from cold_recon.training.volume_codec import fields_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config

from scripts.build_nonleaking_20scene_benchmark import (
    NeuralModel,
    build_fno_denoiser,
    build_obs_encoder,
    ensure_synthetic_samples,
)


CONTINUOUS_TARGET = "eic"


@dataclass
class InferenceCase:
    sample_id: str
    path: Path
    sample: dict[str, Any]
    tokens: torch.Tensor
    token_mask: torch.Tensor
    attention_mask: torch.Tensor | None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 123) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def prepare_inference_case(
    config: dict,
    path: Path,
    device: torch.device,
    max_condition_tokens: int,
    seed: int,
) -> InferenceCase:
    sample = load_sample_npz(path)
    observations = subsample_observations_by_type(sample["observations"], max_condition_tokens, seed=seed)
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    tokens = tokenizer.encode_torch(observations, device=device).unsqueeze(0)
    token_mask = torch.zeros((1, tokens.shape[1]), dtype=torch.bool, device=device)
    attention_mask = build_observation_attention_mask(config, sample["grid"], observations, device=device)
    return InferenceCase(
        sample_id=path.stem,
        path=path,
        sample=sample,
        tokens=tokens,
        token_mask=token_mask,
        attention_mask=attention_mask,
    )


def load_fno_posterior_model(config: dict, device: torch.device, checkpoint: Path) -> NeuralModel:
    ckpt = torch.load(checkpoint, map_location=device)
    latent_channels = int(config["autoencoder"].get("latent_channels", 16))
    obs_encoder = build_obs_encoder(config, device)
    denoiser = build_fno_denoiser(config, latent_channels, device)
    obs_encoder.load_state_dict(ckpt["obs_encoder_state"])
    denoiser.load_state_dict(ckpt["denoiser_state"])
    obs_encoder.eval()
    denoiser.eval()
    return NeuralModel(
        name="COLDReconFNOPosterior",
        warm_start_input=str(ckpt.get("warm_start_input", "IDW observation-proxy latent")),
        obs_encoder=obs_encoder,
        denoiser=denoiser,
        diffusion=GaussianDiffusion3D(denoiser, timesteps=int(config["diffusion"].get("timesteps", 80))),
    )


def cold_posterior(
    config: dict,
    model: NeuralModel,
    ae,
    case: InferenceCase,
    device: torch.device,
    n_facies: int,
    posterior_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    proxy_fields = _proxy_from_observations(case.sample, n_facies=n_facies, use_base_observations=False)
    with torch.no_grad():
        proxy_volume = fields_to_volume_tensor(proxy_fields, n_facies=n_facies).to(device)
        anchor = ae.encode(proxy_volume).detach()
        cond = model.obs_encoder(case.tokens, case.token_mask, case.attention_mask).repeat(int(posterior_samples), 1)
        scale = float(config["diffusion"].get("posterior_noise_scale", 0.08))
        correction = float(config["diffusion"].get("posterior_correction_scale", 0.15))
        timesteps = int(config["diffusion"].get("timesteps", 80))
        latents = anchor.repeat(int(posterior_samples), 1, 1, 1, 1) + scale * torch.randn(
            (int(posterior_samples), *anchor.shape[1:]), device=device
        )
        t_mid = max(timesteps // 2, 1)
        t = torch.full((int(posterior_samples),), t_mid, device=device, dtype=torch.long)
        latents = latents - correction * model.denoiser(latents, t, cond)
        decoded = ae.decode(latents)
    return _posterior_arrays(decoded, n_facies=n_facies)


def entropy(prob: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float32), 1e-8, 1.0)
    return -np.sum(p * np.log(p), axis=axis).astype(np.float32)


def random_forest_probabilistic(sample: dict, n_facies: int, n_estimators: int, seed: int) -> dict[str, np.ndarray]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    obs: ObservationTable = sample["observations"]
    query_features, _, shape = _grid_points_and_surface(sample)
    out: dict[str, np.ndarray] = {}
    facies_mask = obs.type_ids == OBS_TYPES["borehole_facies"]
    if np.sum(facies_mask) >= 2:
        x_train = _surface_at_obs(sample, obs.coords[facies_mask])
        y_train = np.clip(obs.values[facies_mask].astype(np.int64), 0, n_facies - 1)
        clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        clf.fit(x_train, y_train)
        raw = clf.predict_proba(query_features)
        probs = np.zeros((query_features.shape[0], n_facies), dtype=np.float32)
        for col, cls in enumerate(clf.classes_.astype(np.int64)):
            if 0 <= cls < n_facies:
                probs[:, cls] = raw[:, col]
        denom = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(denom, 1e-6)
        out["facies_probability"] = probs.reshape(*shape, n_facies).astype(np.float32)
        out["facies"] = np.argmax(probs, axis=1).reshape(shape).astype(np.int16)
        out["facies_entropy"] = entropy(out["facies_probability"])

    eic_mask = obs.type_ids == OBS_TYPES["borehole_eic"]
    if np.sum(eic_mask) >= 4:
        x_train = _surface_at_obs(sample, obs.coords[eic_mask])
        y_train = obs.values[eic_mask].astype(np.float32)
        reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        reg.fit(x_train, y_train)
        samples = np.stack(
            [tree.predict(query_features).reshape(shape).astype(np.float32) for tree in reg.estimators_],
            axis=0,
        )
        out["eic_samples"] = np.clip(samples, 0.0, 1.0).astype(np.float32)
        out["eic_mean"] = out["eic_samples"].mean(axis=0).astype(np.float32)
        out["eic_std"] = out["eic_samples"].std(axis=0).astype(np.float32)
        out["ice_rich_probability"] = (out["eic_samples"] > 0.30).mean(axis=0).astype(np.float32)
    return out


def quantile_gradient_boosting(sample: dict, n_facies: int, seed: int) -> dict[str, np.ndarray]:
    obs: ObservationTable = sample["observations"]
    query_features, _, shape = _grid_points_and_surface(sample)
    out: dict[str, np.ndarray] = {}
    facies_mask = obs.type_ids == OBS_TYPES["borehole_facies"]
    if np.sum(facies_mask) >= 3:
        x_train = _surface_at_obs(sample, obs.coords[facies_mask])
        y_train = np.clip(obs.values[facies_mask].astype(np.int64), 0, n_facies - 1)
        clf = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            min_samples_leaf=8,
            random_state=seed,
        )
        clf.fit(x_train, y_train)
        raw = clf.predict_proba(query_features)
        probs = np.zeros((query_features.shape[0], n_facies), dtype=np.float32)
        for col, cls in enumerate(clf.classes_.astype(np.int64)):
            if 0 <= cls < n_facies:
                probs[:, cls] = raw[:, col]
        denom = probs.sum(axis=1, keepdims=True)
        probs = probs / np.maximum(denom, 1e-6)
        out["facies_probability"] = probs.reshape(*shape, n_facies).astype(np.float32)
        out["facies"] = np.argmax(probs, axis=1).reshape(shape).astype(np.int16)
        out["facies_entropy"] = entropy(out["facies_probability"])

    eic_mask = obs.type_ids == OBS_TYPES["borehole_eic"]
    if np.sum(eic_mask) >= 6:
        x_train = _surface_at_obs(sample, obs.coords[eic_mask])
        y_train = obs.values[eic_mask].astype(np.float32)
        preds: dict[float, np.ndarray] = {}
        for q in (0.05, 0.50, 0.95):
            reg = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=180,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=1e-3,
                min_samples_leaf=8,
                random_state=seed + int(q * 1000),
            )
            reg.fit(x_train, y_train)
            preds[q] = reg.predict(query_features).reshape(shape).astype(np.float32)
        q05 = np.minimum(preds[0.05], preds[0.95])
        q95 = np.maximum(preds[0.05], preds[0.95])
        mean = np.clip(preds[0.50], 0.0, 1.0)
        sigma = np.maximum((q95 - q05) / 3.289707253902945, 0.005).astype(np.float32)
        rng = np.random.default_rng(seed)
        samples = rng.normal(mean[None, ...], sigma[None, ...], size=(32, *shape)).astype(np.float32)
        out["eic_samples"] = np.clip(samples, 0.0, 1.0).astype(np.float32)
        out["eic_mean"] = mean.astype(np.float32)
        out["eic_std"] = sigma.astype(np.float32)
        out["eic_q05"] = np.clip(q05, 0.0, 1.0).astype(np.float32)
        out["eic_q95"] = np.clip(q95, 0.0, 1.0).astype(np.float32)
        out["ice_rich_probability"] = (out["eic_samples"] > 0.30).mean(axis=0).astype(np.float32)
    return out


def loo_idw_sigma(coords: np.ndarray, values: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.size < 5:
        return 0.05
    preds = []
    for i in range(values.size):
        mask = np.ones(values.size, dtype=bool)
        mask[i] = False
        preds.append(idw_interpolate(coords[mask], values[mask], coords[i : i + 1], k=min(8, values.size - 1))[0])
    resid = values - np.asarray(preds, dtype=np.float32)
    sigma = float(np.sqrt(np.mean(resid**2)))
    return max(sigma, 0.025)


def conditional_gaussian_idw(sample: dict, n_facies: int, seed: int) -> dict[str, np.ndarray]:
    obs: ObservationTable = sample["observations"]
    query_features, query_coords, shape = _grid_points_and_surface(sample)
    idw = reconstruct_idw(obs, sample["grid"], n_facies=n_facies)
    out: dict[str, np.ndarray] = {}
    if "facies_logits" in idw:
        logits = np.asarray(idw["facies_logits"], dtype=np.float32)
        probs = logits / np.maximum(logits.sum(axis=-1, keepdims=True), 1e-6)
        out["facies_probability"] = probs.astype(np.float32)
        out["facies_entropy"] = entropy(probs)
        out["facies"] = np.asarray(idw["facies"], dtype=np.int16)
    eic_mask = obs.type_ids == OBS_TYPES["borehole_eic"]
    if np.any(eic_mask) and "eic" in idw:
        coords = obs.coords[eic_mask]
        values = obs.values[eic_mask].astype(np.float32)
        sigma0 = loo_idw_sigma(coords, values)
        tree = cKDTree(coords)
        dist, _ = tree.query(query_coords, k=1)
        scale = 0.75 + np.clip(dist / max(np.percentile(dist, 90), 1e-6), 0.0, 2.5)
        sigma = (sigma0 * scale.reshape(shape)).astype(np.float32)
        mean = np.clip(np.asarray(idw["eic"], dtype=np.float32), 0.0, 1.0)
        rng = np.random.default_rng(seed)
        samples = rng.normal(mean[None, ...], sigma[None, ...], size=(32, *shape)).astype(np.float32)
        out["eic_samples"] = np.clip(samples, 0.0, 1.0).astype(np.float32)
        out["eic_mean"] = mean
        out["eic_std"] = sigma
        out["ice_rich_probability"] = (out["eic_samples"] > 0.30).mean(axis=0).astype(np.float32)
    return out


def eic_interval(posterior: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "eic_q05" in posterior and "eic_q95" in posterior:
        return posterior["eic_q05"], posterior["eic_q95"]
    samples = np.asarray(posterior["eic_samples"], dtype=np.float32)
    return np.percentile(samples, 5, axis=0).astype(np.float32), np.percentile(samples, 95, axis=0).astype(np.float32)


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    if samples.shape[0] > 32:
        idx = np.linspace(0, samples.shape[0] - 1, 32).round().astype(int)
        samples = samples[idx]
    term1 = np.mean(np.abs(samples - truth[None, ...]), axis=0)
    diffs = np.abs(samples[:, None, ...] - samples[None, :, ...]).mean(axis=(0, 1))
    return float(np.mean(term1 - 0.5 * diffs))


def scene_probability_metrics(
    model: str,
    scene: str,
    posterior: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    n_facies: int,
    rng: np.random.Generator,
    max_voxels: int,
) -> dict[str, Any]:
    eic_truth = np.asarray(truth["eic"], dtype=np.float32)
    eic_mean = np.asarray(posterior["eic_mean"], dtype=np.float32)
    eic_samples = np.asarray(posterior["eic_samples"], dtype=np.float32)
    eic_std = np.asarray(posterior["eic_std"], dtype=np.float32)
    q05, q95 = eic_interval(posterior)
    flat_n = eic_truth.size
    idx = rng.choice(flat_n, size=min(int(max_voxels), flat_n), replace=False)
    y = eic_truth.ravel()[idx]
    mu = eic_mean.ravel()[idx]
    std = eic_std.ravel()[idx]
    lo = q05.ravel()[idx]
    hi = q95.ravel()[idx]
    samples = eic_samples.reshape(eic_samples.shape[0], -1)[:, idx]
    abs_err = np.abs(mu - y)
    corr = spearmanr(std, abs_err, nan_policy="omit").correlation
    if not np.isfinite(corr):
        corr = float("nan")
    top = std >= np.quantile(std, 0.90)
    enrichment = float(abs_err[top].mean() / max(abs_err.mean(), 1e-8)) if np.any(top) else float("nan")
    facies_pred = posterior.get("facies")
    if facies_pred is None:
        facies_pred = posterior.get("facies_mode")
    out = {
        "model": model,
        "scene": scene,
        "facies_iou": facies_iou(np.asarray(facies_pred), truth["facies"], n_facies)["mean_iou"],
        "eic_rmse": rmse(eic_mean, eic_truth),
        "uncertainty_error_corr": float(corr),
        "coverage_90": float(np.mean((y >= lo) & (y <= hi))),
        "crps": crps_ensemble(samples, y),
        "voi_enrichment": enrichment,
        "mean_interval_width_90": float(np.mean(hi - lo)),
    }
    return out


def summarize_probability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["facies_iou", "eic_rmse", "uncertainty_error_corr", "coverage_90", "crps", "voi_enrichment", "mean_interval_width_90"]
    out: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        item: dict[str, Any] = {"model": model, "scenes": len(model_rows)}
        for metric in metrics:
            vals = np.asarray([float(row.get(metric, np.nan)) for row in model_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(vals, seed=5000 + len(out) * 37 + len(metric))
            item[f"{metric}_mean"] = float(vals.mean())
            item[f"{metric}_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            item[f"{metric}_ci95_low"] = lo
            item[f"{metric}_ci95_high"] = hi
            item[f"{metric}_mean_sd_ci"] = f"{vals.mean():.4f} +/- {vals.std(ddof=1) if vals.size > 1 else 0.0:.4f} [{lo:.4f}, {hi:.4f}]"
        out.append(item)
    return out


def train_rates(paths: list[Path], ice_threshold: float) -> dict[str, float]:
    rare = []
    high = []
    for path in paths:
        sample = load_sample_npz(path)
        truth = sample["fields"]
        high_mask = np.asarray(truth["eic"]) > float(ice_threshold)
        rare_mask = high_mask | (np.asarray(truth["facies"]) == 6)
        rare.append(float(np.mean(rare_mask)))
        high.append(float(np.mean(high_mask)))
    return {
        "rare_rate": float(np.mean(rare)),
        "high_eic_rate": float(np.mean(high)),
    }


def top_fraction_mask(score: np.ndarray, rate: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    rate = float(np.clip(rate, 1e-5, 0.50))
    flat = score.ravel()
    k = max(1, int(round(rate * flat.size)))
    k = min(k, flat.size)
    order = np.lexsort((np.arange(flat.size), -flat))
    mask = np.zeros(flat.size, dtype=bool)
    mask[order[:k]] = True
    return mask.reshape(score.shape)


def object_counts(
    truth_mask: np.ndarray,
    pred_mask: np.ndarray,
    min_size: int = 8,
    min_overlap_fraction: float = 0.10,
) -> tuple[int, int, int, int]:
    structure = np.ones((3, 3, 3), dtype=np.int8)
    truth_lab, truth_n = cc_label(truth_mask.astype(bool), structure=structure)
    pred_lab, pred_n = cc_label(pred_mask.astype(bool), structure=structure)
    true_total = 0
    true_hit = 0
    for comp in range(1, truth_n + 1):
        mask = truth_lab == comp
        if int(mask.sum()) < int(min_size):
            continue
        true_total += 1
        if float(np.mean(pred_mask[mask])) >= float(min_overlap_fraction):
            true_hit += 1
    pred_total = 0
    pred_hit = 0
    for comp in range(1, pred_n + 1):
        mask = pred_lab == comp
        if int(mask.sum()) < int(min_size):
            continue
        pred_total += 1
        if float(np.mean(truth_mask[mask])) >= float(min_overlap_fraction):
            pred_hit += 1
    return true_hit, true_total, pred_hit, pred_total


def rare_score_from_posterior(posterior: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    high_score = np.asarray(posterior.get("ice_rich_probability", posterior["eic_mean"] > 0.30), dtype=np.float32)
    facies_prob = posterior.get("facies_probability")
    if facies_prob is not None and facies_prob.shape[-1] > 6:
        wedge_score = np.asarray(facies_prob[..., 6], dtype=np.float32)
        rare_score = np.maximum(high_score, wedge_score).astype(np.float32)
    else:
        rare_score = high_score
    return rare_score, high_score


def rare_detail_row(
    model: str,
    scene: str,
    posterior: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    rates: dict[str, float],
    ice_threshold: float,
) -> dict[str, Any]:
    rare_score, high_score = rare_score_from_posterior(posterior)
    high_truth = np.asarray(truth["eic"]) > float(ice_threshold)
    rare_truth = high_truth | (np.asarray(truth["facies"]) == 6)
    rare_pred = top_fraction_mask(rare_score, rates["rare_rate"])
    high_pred = top_fraction_mask(high_score, rates["high_eic_rate"])
    tp = int(np.logical_and(high_pred, high_truth).sum())
    fp = int(np.logical_and(high_pred, ~high_truth).sum())
    fn = int(np.logical_and(~high_pred, high_truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    high_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    true_hit, true_total, pred_hit, pred_total = object_counts(rare_truth, rare_pred)
    return {
        "model": model,
        "scene": scene,
        "rare_auprc": float(average_precision_score(rare_truth.ravel().astype(np.uint8), rare_score.ravel())),
        "object_true_hit": true_hit,
        "object_true_total": true_total,
        "object_pred_hit": pred_hit,
        "object_pred_total": pred_total,
        "object_recall": true_hit / max(true_total, 1),
        "object_precision": pred_hit / max(pred_total, 1),
        "high_eic_tp": tp,
        "high_eic_fp": fp,
        "high_eic_fn": fn,
        "high_eic_f1": float(high_f1),
        "rare_operating_rate": rates["rare_rate"],
        "high_eic_operating_rate": rates["high_eic_rate"],
    }


def summarize_rare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        auprc_vals = np.asarray([float(row["rare_auprc"]) for row in model_rows], dtype=np.float64)
        f1_vals = np.asarray([float(row["high_eic_f1"]) for row in model_rows], dtype=np.float64)
        true_hit = sum(int(row["object_true_hit"]) for row in model_rows)
        true_total = sum(int(row["object_true_total"]) for row in model_rows)
        pred_hit = sum(int(row["object_pred_hit"]) for row in model_rows)
        pred_total = sum(int(row["object_pred_total"]) for row in model_rows)
        tp = sum(int(row["high_eic_tp"]) for row in model_rows)
        fp = sum(int(row["high_eic_fp"]) for row in model_rows)
        fn = sum(int(row["high_eic_fn"]) for row in model_rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        high_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        lo, hi = bootstrap_ci(auprc_vals, seed=7000 + len(out))
        out.append(
            {
                "model": model,
                "scenes": len(model_rows),
                "rare_auprc_mean": float(auprc_vals.mean()),
                "rare_auprc_sd": float(auprc_vals.std(ddof=1)) if auprc_vals.size > 1 else 0.0,
                "rare_auprc_ci95_low": lo,
                "rare_auprc_ci95_high": hi,
                "object_recall": true_hit / max(true_total, 1),
                "object_precision": pred_hit / max(pred_total, 1),
                "high_eic_f1": float(high_f1),
                "high_eic_f1_scene_mean": float(f1_vals.mean()),
                "true_objects": true_total,
                "predicted_objects": pred_total,
            }
        )
    return out


def prediction_set(
    case: InferenceCase,
    config: dict,
    n_facies: int,
    rf_trees: int,
    seed: int,
    cold_model: NeuralModel,
    ae,
    device: torch.device,
    posterior_samples: int,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "RFEnsemble": random_forest_probabilistic(case.sample, n_facies=n_facies, n_estimators=rf_trees, seed=seed),
        "QuantileGB": quantile_gradient_boosting(case.sample, n_facies=n_facies, seed=seed + 11),
        "ConditionalGaussianIDW": conditional_gaussian_idw(case.sample, n_facies=n_facies, seed=seed + 23),
        "COLDReconFNOPosterior": cold_posterior(
            config,
            cold_model,
            ae,
            case,
            device,
            n_facies,
            posterior_samples=posterior_samples,
            seed=seed + 37,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=20)
    parser.add_argument("--start-test-index", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-condition-tokens", type=int, default=512)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--rf-trees", type=int, default=80)
    parser.add_argument("--max-probability-voxels", type=int, default=30000)
    parser.add_argument("--cold-checkpoint", default="outputs/checkpoints/nonleaking20_FNOOperator.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    total = int(args.start_test_index) + int(args.test_scenes)
    paths = ensure_synthetic_samples(config, total_samples=total, seed=seed, regenerate=False)
    train_paths = paths[: int(args.train_scenes)]
    test_paths = paths[int(args.start_test_index) : int(args.start_test_index) + int(args.test_scenes)]
    device_name = args.device or config.get("diffusion", {}).get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    n_facies = int(config["model"]["n_facies"])
    ice_threshold = float(config["evaluation"].get("ice_rich_threshold", 0.30))
    rates = train_rates(train_paths, ice_threshold=ice_threshold)
    print(f"rare operating rates from training scenes: {rates}")

    ae = _load_autoencoder(config, device)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    ae.eval()
    cold_model = load_fno_posterior_model(config, device, Path(args.cold_checkpoint))

    probability_rows: list[dict[str, Any]] = []
    rare_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for scene_idx, path in enumerate(test_paths):
        case = prepare_inference_case(config, path, device, int(args.max_condition_tokens), seed + 10_000 + scene_idx)
        print(f"probabilistic/rare audit scene {scene_idx + 1:02d}/{len(test_paths)} {case.sample_id}")
        preds = prediction_set(
            case,
            config,
            n_facies=n_facies,
            rf_trees=int(args.rf_trees),
            seed=seed + scene_idx,
            cold_model=cold_model,
            ae=ae,
            device=device,
            posterior_samples=int(args.posterior_samples),
        )
        for model_name, posterior in preds.items():
            probability_rows.append(
                scene_probability_metrics(
                    model_name,
                    case.sample_id,
                    posterior,
                    case.sample["fields"],
                    n_facies=n_facies,
                    rng=rng,
                    max_voxels=int(args.max_probability_voxels),
                )
            )
            if model_name in {"RFEnsemble", "QuantileGB", "COLDReconFNOPosterior"}:
                rare_rows.append(
                    rare_detail_row(
                        model_name,
                        case.sample_id,
                        posterior,
                        case.sample["fields"],
                        rates,
                        ice_threshold=ice_threshold,
                    )
                )

    probability_summary = summarize_probability(probability_rows)
    rare_summary = summarize_rare(rare_rows)
    table_dirs = [Path(config["paths"]["tables_dir"]), Path("tables")]
    for table_dir in table_dirs:
        write_csv(table_dir / "probabilistic_baseline_detail.csv", probability_rows)
        write_csv(table_dir / "probabilistic_baseline_summary.csv", probability_summary)
        write_csv(table_dir / "rare_structure_baseline_detail.csv", rare_rows)
        write_csv(table_dir / "rare_structure_baseline_summary.csv", rare_summary)
    print("wrote probabilistic_baseline_detail/summary and rare_structure_baseline_detail/summary")


if __name__ == "__main__":
    main()
