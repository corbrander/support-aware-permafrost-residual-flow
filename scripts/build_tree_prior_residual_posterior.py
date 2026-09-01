from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from cold_recon.baselines.random_forest import reconstruct_random_forest
from cold_recon.baselines.xgboost_ngb import GradientBoostingConfig, reconstruct_gradient_boosting
from cold_recon.data.data_schema import OBS_TYPES, SURFACE_FEATURE_NAMES, load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.evaluation.observation_consistency import nearest_grid_indices
from cold_recon.training.train_multisample_diffusion import _proxy_from_observations
from cold_recon.training.volume_codec import fields_to_volume_tensor, sample_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config

from scripts.build_nonleaking_20scene_benchmark import ensure_synthetic_samples
from scripts.build_probabilistic_and_rare_baselines import (
    bootstrap_ci,
    rare_detail_row,
    scene_probability_metrics,
    train_rates,
)


SUMMARY_METRICS = (
    "facies_iou",
    "mean_iou",
    "eic_rmse",
    "temperature_rmse",
    "unfrozen_water_rmse",
    "log_resistivity_rmse",
    "uncertainty_error_corr",
    "coverage_90",
    "crps",
    "rare_auprc",
    "high_eic_f1",
    "voi_enrichment",
)

HIGHER_IS_BETTER = {
    "facies_iou",
    "mean_iou",
    "uncertainty_error_corr",
    "coverage_90",
    "rare_auprc",
    "high_eic_f1",
    "voi_enrichment",
}


@dataclass
class ResidualCase:
    sample_id: str
    path: Path
    sample: dict[str, Any]
    x: torch.Tensor
    prior_volume: torch.Tensor
    prior_logits: torch.Tensor
    target: torch.Tensor
    target_facies: torch.Tensor
    prior_fields: dict[str, np.ndarray]


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


def fill_proxy_defaults(proxy: dict[str, np.ndarray], defaults: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in ("facies", "eic", "temperature", "unfrozen_water", "log_resistivity"):
        if key in proxy:
            out[key] = np.asarray(proxy[key]).copy()
        elif key in defaults:
            out[key] = np.asarray(defaults[key]).copy()
        else:
            raise KeyError(f"Missing proxy field {key}")
    out["facies"] = np.asarray(out["facies"], dtype=np.int16)
    out["eic"] = np.clip(np.asarray(out["eic"], dtype=np.float32), 0.0, 1.0)
    out["temperature"] = np.asarray(out["temperature"], dtype=np.float32)
    out["unfrozen_water"] = np.clip(np.asarray(out["unfrozen_water"], dtype=np.float32), 0.0, 1.0)
    out["log_resistivity"] = np.asarray(out["log_resistivity"], dtype=np.float32)
    if "facies_probability" in proxy:
        out["facies_probability"] = np.asarray(proxy["facies_probability"], dtype=np.float32)
    return out


def tree_prior_fields(
    sample: dict[str, Any],
    n_facies: int,
    seed: int,
    rf_trees: int,
    rf_n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    defaults = fill_proxy_defaults(_proxy_from_observations(sample, n_facies=n_facies, use_base_observations=False), {})
    rf = fill_proxy_defaults(
        reconstruct_random_forest(
            sample,
            n_estimators=int(rf_trees),
            random_state=int(seed),
            n_jobs=int(rf_n_jobs),
        ),
        defaults,
    )
    gb = fill_proxy_defaults(
        reconstruct_gradient_boosting(
            sample,
            n_facies=n_facies,
            config=GradientBoostingConfig(random_state=int(seed) + 11),
        ),
        defaults,
    )
    out: dict[str, np.ndarray] = {
        "facies": gb["facies"].astype(np.int16),
        "eic": rf["eic"].astype(np.float32),
        "temperature": rf["temperature"].astype(np.float32),
        "unfrozen_water": rf["unfrozen_water"].astype(np.float32),
        "log_resistivity": rf["log_resistivity"].astype(np.float32),
    }
    if "facies_probability" in gb:
        out["facies_probability"] = np.asarray(gb["facies_probability"], dtype=np.float32)
    return fill_proxy_defaults(out, defaults)


def prior_logits_from_fields(fields: dict[str, np.ndarray], n_facies: int) -> torch.Tensor:
    if "facies_probability" in fields:
        prob = np.asarray(fields["facies_probability"], dtype=np.float32)
        prob = prob / np.maximum(prob.sum(axis=-1, keepdims=True), 1e-6)
        logits = np.log(np.clip(prob, 1e-5, 1.0)).transpose(3, 0, 1, 2)
        return torch.as_tensor(logits, dtype=torch.float32).unsqueeze(0)
    facies = np.asarray(fields["facies"], dtype=np.int64)
    onehot = np.eye(n_facies, dtype=np.float32)[np.clip(facies, 0, n_facies - 1)]
    smooth = onehot * 0.94 + (1.0 - onehot) * (0.06 / max(n_facies - 1, 1))
    return torch.as_tensor(np.log(np.clip(smooth, 1e-5, 1.0)).transpose(3, 0, 1, 2), dtype=torch.float32).unsqueeze(0)


def normalize_surface_feature(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(float(hi - lo)) < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / float(hi - lo), -2.0, 3.0).astype(np.float32)


def coordinate_channels(sample: dict[str, Any]) -> np.ndarray:
    grid = sample["grid"]
    x = np.asarray(grid["x"], dtype=np.float32)
    y = np.asarray(grid["y"], dtype=np.float32)
    z = np.asarray(grid["z"], dtype=np.float32)
    xx, yy, zz = np.meshgrid(
        x / max(float(x[-1]), 1.0),
        y / max(float(y[-1]), 1.0),
        z / max(float(z[-1]), 1.0),
        indexing="ij",
    )
    return np.stack([xx, yy, zz], axis=0).astype(np.float32)


def surface_channels(sample: dict[str, Any], nz: int) -> np.ndarray:
    channels: list[np.ndarray] = []
    for name in SURFACE_FEATURE_NAMES:
        surf = normalize_surface_feature(sample["surface_features"][name])
        channels.append(np.repeat(surf[:, :, None], nz, axis=2))
    return np.stack(channels, axis=0).astype(np.float32)


def observation_channels(sample: dict[str, Any], n_facies: int) -> np.ndarray:
    grid = sample["grid"]
    shape = (len(grid["x"]), len(grid["y"]), len(grid["z"]))
    obs = sample["observations"]
    specs = [
        ("borehole_facies", max(n_facies - 1, 1), 0.0),
        ("borehole_eic", 1.0, 0.0),
        ("borehole_temperature", 10.0, 0.0),
        ("nmr_unfrozen_water", 1.0, 0.0),
        ("ert_log_resistivity", 10.0, 0.0),
    ]
    channels: list[np.ndarray] = []
    valid_obs = np.asarray(obs.mask, dtype=bool)
    for type_name, scale, offset in specs:
        values = np.full(shape, float(offset), dtype=np.float32)
        mask_grid = np.zeros(shape, dtype=np.float32)
        mask = (obs.type_ids == OBS_TYPES[type_name]) & valid_obs
        if np.any(mask):
            coords = obs.coords[mask]
            ix, iy, iz = nearest_grid_indices(coords, grid)
            scaled = obs.values[mask].astype(np.float32) / float(scale)
            values[ix, iy, iz] = scaled
            mask_grid[ix, iy, iz] = 1.0
        channels.extend([values, mask_grid])
    return np.stack(channels, axis=0).astype(np.float32)


def make_case(path: Path, n_facies: int, seed: int, rf_trees: int) -> ResidualCase:
    sample = load_sample_npz(path)
    prior = tree_prior_fields(sample, n_facies=n_facies, seed=seed, rf_trees=rf_trees)
    prior_volume = fields_to_volume_tensor(prior, n_facies=n_facies).float()
    _, _, nx, ny, nz = prior_volume.shape
    input_channels = np.concatenate(
        [
            prior_volume[0].numpy(),
            prior_logits_from_fields(prior, n_facies=n_facies)[0].numpy(),
            coordinate_channels(sample),
            surface_channels(sample, nz=nz),
            observation_channels(sample, n_facies=n_facies),
        ],
        axis=0,
    )
    target = sample_to_volume_tensor(sample, n_facies=n_facies).float()
    target_facies = torch.as_tensor(sample["fields"]["facies"].astype(np.int64)).unsqueeze(0)
    return ResidualCase(
        sample_id=path.stem,
        path=path,
        sample=sample,
        x=torch.as_tensor(input_channels, dtype=torch.float32).unsqueeze(0),
        prior_volume=prior_volume,
        prior_logits=prior_logits_from_fields(prior, n_facies=n_facies).float(),
        target=target,
        target_facies=target_facies,
        prior_fields=prior,
    )


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = norm_groups(channels)
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TreePriorResidualCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int = 32, depth: int = 4, dropout: float = 0.08) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, int(width), kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups(int(width)), int(width)),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock3D(int(width), dropout=float(dropout)) for _ in range(int(depth))))
        self.head = nn.Conv3d(int(width), out_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def norm_groups(channels: int) -> int:
    channels = int(channels)
    for group in (8, 6, 4, 3, 2):
        if channels % group == 0:
            return group
    return 1


def dice_loss(logits: torch.Tensor, target_facies: torch.Tensor, n_facies: int) -> torch.Tensor:
    prob = F.softmax(logits, dim=1)
    target = F.one_hot(target_facies.squeeze(1), num_classes=n_facies).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob + target, dim=dims)
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def unpack_prediction(
    residual: torch.Tensor,
    prior_volume: torch.Tensor,
    prior_logits: torch.Tensor,
    n_facies: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    facies_logits = prior_logits + residual[:, :n_facies]
    prior_cont = prior_volume[:, n_facies : n_facies + 4]
    cont = prior_cont + residual[:, n_facies : n_facies + 4]
    return facies_logits, cont


def train_model(
    model: nn.Module,
    cases: list[ResidualCase],
    n_facies: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> list[dict[str, Any]]:
    opt = AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(20260701)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        order = rng.permutation(len(cases))
        for idx in order:
            case = cases[int(idx)]
            x = case.x.to(device)
            prior_volume = case.prior_volume.to(device)
            prior_logits = case.prior_logits.to(device)
            target = case.target.to(device)
            target_facies = case.target_facies.to(device)
            residual = model(x)
            facies_logits, cont = unpack_prediction(residual, prior_volume, prior_logits, n_facies=n_facies)
            ce = F.cross_entropy(facies_logits, target_facies.squeeze(1))
            dsc = dice_loss(facies_logits, target_facies, n_facies=n_facies)
            eic = F.mse_loss(cont[:, 0].clamp(0.0, 1.0), target[:, n_facies + 0])
            temp = F.mse_loss(cont[:, 1], target[:, n_facies + 1])
            uw = F.mse_loss(cont[:, 2].clamp(0.0, 1.0), target[:, n_facies + 2])
            rho = F.mse_loss(cont[:, 3], target[:, n_facies + 3])
            resid_l2 = torch.mean(residual**2)
            loss = ce + 0.60 * dsc + 4.0 * eic + 0.25 * temp + 1.5 * uw + 0.15 * rho + 1.0e-4 * resid_l2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch {epoch:03d}/{epochs} loss={mean_loss:.6f}")
    return history


def fields_from_tensors(
    facies_logits: torch.Tensor,
    cont: torch.Tensor,
    n_facies: int,
) -> dict[str, np.ndarray]:
    prob = F.softmax(facies_logits, dim=1).detach().cpu().numpy()[0]
    facies = np.argmax(prob, axis=0).astype(np.int16)
    cont_np = cont.detach().cpu().numpy()[0]
    eic = np.clip(cont_np[0], 0.0, 1.0).astype(np.float32)
    temp = (cont_np[1] * 10.0).astype(np.float32)
    uw = np.clip(cont_np[2], 0.0, 1.0).astype(np.float32)
    log_res = (cont_np[3] * 10.0).astype(np.float32)
    return {
        "facies": facies,
        "facies_probability": np.moveaxis(prob.astype(np.float32), 0, -1),
        "eic": eic,
        "temperature": temp,
        "unfrozen_water": uw,
        "log_resistivity": log_res,
        "resistivity": np.exp(np.clip(log_res, 0.0, 12.0)).astype(np.float32),
        "n_facies": np.asarray(n_facies),
    }


def posterior_from_model(
    model: nn.Module,
    case: ResidualCase,
    n_facies: int,
    device: torch.device,
    posterior_samples: int,
) -> dict[str, np.ndarray]:
    facies_samples: list[np.ndarray] = []
    prob_samples: list[np.ndarray] = []
    eic_samples: list[np.ndarray] = []
    temp_samples: list[np.ndarray] = []
    uw_samples: list[np.ndarray] = []
    lr_samples: list[np.ndarray] = []
    model.train()
    with torch.no_grad():
        for _ in range(int(posterior_samples)):
            residual = model(case.x.to(device))
            logits, cont = unpack_prediction(
                residual,
                case.prior_volume.to(device),
                case.prior_logits.to(device),
                n_facies=n_facies,
            )
            fields = fields_from_tensors(logits, cont, n_facies=n_facies)
            facies_samples.append(fields["facies"])
            prob_samples.append(fields["facies_probability"])
            eic_samples.append(fields["eic"])
            temp_samples.append(fields["temperature"])
            uw_samples.append(fields["unfrozen_water"])
            lr_samples.append(fields["log_resistivity"])
    eic_stack = np.stack(eic_samples, axis=0).astype(np.float32)
    temp_stack = np.stack(temp_samples, axis=0).astype(np.float32)
    uw_stack = np.stack(uw_samples, axis=0).astype(np.float32)
    lr_stack = np.stack(lr_samples, axis=0).astype(np.float32)
    prob_mean = np.mean(np.stack(prob_samples, axis=0), axis=0).astype(np.float32)
    return {
        "facies_samples": np.stack(facies_samples, axis=0).astype(np.int16),
        "facies": np.argmax(prob_mean, axis=-1).astype(np.int16),
        "facies_mode": np.argmax(prob_mean, axis=-1).astype(np.int16),
        "facies_probability": prob_mean,
        "eic_samples": eic_stack,
        "eic_mean": eic_stack.mean(axis=0).astype(np.float32),
        "eic_std": eic_stack.std(axis=0).astype(np.float32),
        "eic_q05": np.percentile(eic_stack, 5, axis=0).astype(np.float32),
        "eic_q95": np.percentile(eic_stack, 95, axis=0).astype(np.float32),
        "ice_rich_probability": (eic_stack > 0.30).mean(axis=0).astype(np.float32),
        "temperature_samples": temp_stack,
        "temperature_mean": temp_stack.mean(axis=0).astype(np.float32),
        "temperature_std": temp_stack.std(axis=0).astype(np.float32),
        "unfrozen_water_samples": uw_stack,
        "unfrozen_water_mean": uw_stack.mean(axis=0).astype(np.float32),
        "unfrozen_water_std": uw_stack.std(axis=0).astype(np.float32),
        "log_resistivity_samples": lr_stack,
        "log_resistivity_mean": lr_stack.mean(axis=0).astype(np.float32),
        "log_resistivity_std": lr_stack.std(axis=0).astype(np.float32),
    }


def deterministic_posterior_from_fields(fields: dict[str, np.ndarray], n_samples: int = 8) -> dict[str, np.ndarray]:
    eic = np.asarray(fields["eic"], dtype=np.float32)
    temp = np.asarray(fields["temperature"], dtype=np.float32)
    uw = np.asarray(fields["unfrozen_water"], dtype=np.float32)
    lr = np.asarray(fields["log_resistivity"], dtype=np.float32)
    if "facies_probability" in fields:
        prob = np.asarray(fields["facies_probability"], dtype=np.float32)
    else:
        n_facies = int(np.max(fields["facies"]) + 1)
        prob = np.eye(max(n_facies, 7), dtype=np.float32)[np.asarray(fields["facies"], dtype=np.int64)]
    return {
        "facies": np.asarray(fields["facies"], dtype=np.int16),
        "facies_mode": np.asarray(fields["facies"], dtype=np.int16),
        "facies_probability": prob.astype(np.float32),
        "eic_samples": np.repeat(eic[None, ...], int(n_samples), axis=0),
        "eic_mean": eic,
        "eic_std": np.full_like(eic, 0.005, dtype=np.float32),
        "eic_q05": eic,
        "eic_q95": eic,
        "ice_rich_probability": (eic > 0.30).astype(np.float32),
        "temperature_mean": temp,
        "unfrozen_water_mean": uw,
        "log_resistivity_mean": lr,
    }


def metric_row(
    model_name: str,
    scene: str,
    posterior: dict[str, np.ndarray],
    sample: dict[str, Any],
    n_facies: int,
    rates: dict[str, float],
    ice_threshold: float,
    rng: np.random.Generator,
    max_voxels: int,
    model_role: str,
) -> dict[str, Any]:
    row = scene_probability_metrics(
        model_name,
        scene,
        posterior,
        sample["fields"],
        n_facies=n_facies,
        rng=rng,
        max_voxels=int(max_voxels),
    )
    pred = {
        "facies": np.asarray(posterior["facies_mode"], dtype=np.int16),
        "eic": np.asarray(posterior["eic_mean"], dtype=np.float32),
        "temperature": np.asarray(posterior["temperature_mean"], dtype=np.float32),
        "unfrozen_water": np.asarray(posterior["unfrozen_water_mean"], dtype=np.float32),
        "log_resistivity": np.asarray(posterior["log_resistivity_mean"], dtype=np.float32),
    }
    row.update(synthetic_metrics(pred, sample["fields"], z=sample["grid"]["z"], n_facies=n_facies))
    rare = rare_detail_row(model_name, scene, posterior, sample["fields"], rates, ice_threshold=ice_threshold)
    row.update(
        {
            "rare_auprc": rare["rare_auprc"],
            "object_recall": rare["object_recall"],
            "object_precision": rare["object_precision"],
            "high_eic_f1": rare["high_eic_f1"],
            "model": model_name,
            "scene": scene,
            "model_role": model_role,
            "uses_dense_target_latent_at_inference": "false",
            "warm_start_input": "GB facies prior plus RF continuous prior; observation-only and covariate inputs",
        }
    )
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name]
        item: dict[str, Any] = {
            "model": model_name,
            "scenes": len(model_rows),
            "model_role": model_rows[0].get("model_role", ""),
            "uses_dense_target_latent_at_inference": "false",
        }
        for metric in SUMMARY_METRICS:
            vals = np.asarray([float(row.get(metric, np.nan)) for row in model_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(vals, seed=12_000 + len(out) * 101 + len(metric))
            item[f"{metric}_mean"] = float(vals.mean())
            item[f"{metric}_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            item[f"{metric}_ci95_low"] = lo
            item[f"{metric}_ci95_high"] = hi
            item[f"{metric}_mean_sd_ci"] = f"{vals.mean():.4f} +/- {vals.std(ddof=1) if vals.size > 1 else 0.0:.4f} [{lo:.4f}, {hi:.4f}]"
        out.append(item)
    return out


def sign_flip_pvalue(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas) & (np.abs(deltas) > 1e-12)]
    n = int(deltas.size)
    if n == 0:
        return float("nan")
    k = int(np.sum(deltas > 0))
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / float(2**n)
    return float(min(1.0, 2.0 * tail))


def paired_deltas(rows: list[dict[str, Any]], candidate: str, baseline: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    scenes = sorted({row["scene"] for row in rows if row["model"] == candidate} & {row["scene"] for row in rows if row["model"] == baseline})
    by_key = {(row["model"], row["scene"]): row for row in rows}
    for metric in SUMMARY_METRICS:
        deltas: list[float] = []
        for scene in scenes:
            cand = float(by_key[(candidate, scene)].get(metric, np.nan))
            base = float(by_key[(baseline, scene)].get(metric, np.nan))
            if not np.isfinite(cand) or not np.isfinite(base):
                continue
            direction = 1.0 if metric in HIGHER_IS_BETTER else -1.0
            deltas.append(direction * (cand - base))
        vals = np.asarray(deltas, dtype=np.float64)
        if vals.size == 0:
            continue
        lo, hi = bootstrap_ci(vals, seed=14_000 + len(out) * 71 + len(metric))
        out.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "directional_delta_mean": float(vals.mean()),
                "directional_delta_sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "directional_delta_ci95_low": lo,
                "directional_delta_ci95_high": hi,
                "sign_flip_p": sign_flip_pvalue(vals),
                "scenes": int(vals.size),
                "positive_scene_count": int(np.sum(vals > 0)),
                "interpretation": "positive means candidate is better after metric direction is applied",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=20)
    parser.add_argument("--start-test-index", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--rf-trees", type=int, default=32)
    parser.add_argument("--max-probability-voxels", type=int, default=15000)
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
    if device.type == "cuda":
        print(f"using device=cuda name={torch.cuda.get_device_name(0)}")
    else:
        print("using device=cpu")
    n_facies = int(config["model"]["n_facies"])
    ice_threshold = float(config["evaluation"].get("ice_rich_threshold", 0.30))

    print(f"preparing {len(train_paths)} train scenes and {len(test_paths)} held-out scenes")
    train_cases = [
        make_case(path, n_facies=n_facies, seed=seed + i, rf_trees=int(args.rf_trees))
        for i, path in enumerate(train_paths)
    ]
    test_cases = [
        make_case(path, n_facies=n_facies, seed=seed + 10_000 + i, rf_trees=int(args.rf_trees))
        for i, path in enumerate(test_paths)
    ]
    in_channels = int(train_cases[0].x.shape[1])
    model = TreePriorResidualCNN(
        in_channels=in_channels,
        out_channels=n_facies + 4,
        width=int(args.width),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device)
    print(f"TreePriorResidualCNN in_channels={in_channels} width={args.width} depth={args.depth}")
    history = train_model(
        model,
        train_cases,
        n_facies=n_facies,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
    )

    checkpoint_dir = Path(config["paths"]["checkpoints_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "in_channels": in_channels,
            "n_facies": n_facies,
            "model_name": "COLDReconTreePriorResidual",
            "formula": "X_pred = X_tree + Delta_theta(O,G,X_tree)",
        },
        checkpoint_dir / "tree_prior_residual_posterior.pt",
    )

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 707)
    rates = train_rates(train_paths, ice_threshold=ice_threshold)
    for idx, case in enumerate(test_cases, start=1):
        print(f"evaluating tree-prior residual scene {idx:02d}/{len(test_cases)} {case.sample_id}")
        base_posterior = deterministic_posterior_from_fields(case.prior_fields, n_samples=int(args.posterior_samples))
        rows.append(
            metric_row(
                "TreePriorBase_GBFacies_RFContinuous",
                case.sample_id,
                base_posterior,
                case.sample,
                n_facies=n_facies,
                rates=rates,
                ice_threshold=ice_threshold,
                rng=rng,
                max_voxels=int(args.max_probability_voxels),
                model_role="deterministic tree prior used by the residual model",
            )
        )
        posterior = posterior_from_model(
            model,
            case,
            n_facies=n_facies,
            device=device,
            posterior_samples=int(args.posterior_samples),
        )
        rows.append(
            metric_row(
                "COLDReconTreePriorResidual",
                case.sample_id,
                posterior,
                case.sample,
                n_facies=n_facies,
                rates=rates,
                ice_threshold=ice_threshold,
                rng=rng,
                max_voxels=int(args.max_probability_voxels),
                model_role="trained residual posterior branch: X_tree plus learned residual",
            )
        )

    tables = Path("tables")
    outputs = Path(config["paths"]["tables_dir"])
    detail_path = tables / "tree_prior_residual_detail.csv"
    summary_path = tables / "tree_prior_residual_summary.csv"
    delta_path = tables / "tree_prior_residual_paired_deltas.csv"
    history_path = tables / "tree_prior_residual_training_history.csv"
    write_csv(detail_path, rows)
    summary_rows = summarize(rows)
    write_csv(summary_path, summary_rows)
    delta_rows = paired_deltas(rows, "COLDReconTreePriorResidual", "TreePriorBase_GBFacies_RFContinuous")
    write_csv(delta_path, delta_rows)
    write_csv(history_path, history)
    write_csv(outputs / detail_path.name, rows)
    write_csv(outputs / summary_path.name, summary_rows)
    write_csv(outputs / delta_path.name, delta_rows)
    write_csv(outputs / history_path.name, history)
    print(f"wrote {detail_path}, {summary_path}, {delta_path}")


if __name__ == "__main__":
    main()
