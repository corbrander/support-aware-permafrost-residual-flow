from __future__ import annotations

import argparse
import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from cold_recon.baselines.idw import reconstruct_idw
from cold_recon.baselines.random_forest import reconstruct_random_forest
from cold_recon.baselines.xgboost_ngb import GradientBoostingConfig, reconstruct_gradient_boosting
from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.models.denoiser3d_unet import Denoiser3DUNet
from cold_recon.models.diffusion import GaussianDiffusion3D
from cold_recon.models.fno_transformer import FNOTransformerHybrid
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.synthetic.cryo_synth_generator import generate_synthetic_sample, save_synthetic_sample
from cold_recon.training.physics_guided_training import PhysicsGuidedTrainingConfig, physics_guided_diffusion_loss
from cold_recon.training.train_diffusion import _load_autoencoder, _posterior_arrays
from cold_recon.training.train_multisample_diffusion import _proxy_from_observations, subsample_observations_by_type
from cold_recon.training.volume_codec import fields_to_volume_tensor, sample_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config


METRICS = ["mean_iou", "eic_rmse", "temperature_rmse", "unfrozen_water_rmse"]
LOWER_IS_BETTER = {"eic_rmse", "temperature_rmse", "unfrozen_water_rmse", "log_resistivity_rmse"}


@dataclass
class BenchmarkCase:
    sample_id: str
    path: Path
    sample: dict[str, Any]
    target_volume: torch.Tensor
    latent: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor
    attention_mask: torch.Tensor | None


@dataclass
class NeuralModel:
    name: str
    warm_start_input: str
    obs_encoder: ObsTransformerEncoder
    denoiser: torch.nn.Module
    diffusion: GaussianDiffusion3D


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


def ensure_synthetic_samples(config: dict, total_samples: int, seed: int, regenerate: bool = False) -> list[Path]:
    out_dir = Path(config["paths"]["synthetic_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(total_samples):
        path = out_dir / f"sample_{idx:04d}.npz"
        if regenerate or not path.exists():
            sample = generate_synthetic_sample(config, seed=seed + idx, site_id=f"synthetic_{idx:04d}")
            save_synthetic_sample(path, sample)
            print(f"generated {path}")
        paths.append(path)
    return paths


def prepare_case(
    config: dict,
    path: Path,
    ae,
    device: torch.device,
    n_facies: int,
    max_condition_tokens: int,
    seed: int,
) -> BenchmarkCase:
    sample = load_sample_npz(path)
    target_volume = sample_to_volume_tensor(sample, n_facies=n_facies).to(device)
    with torch.no_grad():
        latent = ae.encode(target_volume).detach()
    observations = subsample_observations_by_type(sample["observations"], max_condition_tokens, seed=seed)
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    tokens = tokenizer.encode_torch(observations, device=device).unsqueeze(0)
    token_mask = torch.zeros((1, tokens.shape[1]), dtype=torch.bool, device=device)
    attention_mask = build_observation_attention_mask(config, sample["grid"], observations, device=device)
    return BenchmarkCase(
        sample_id=path.stem,
        path=path,
        sample=sample,
        target_volume=target_volume,
        latent=latent,
        tokens=tokens,
        token_mask=token_mask,
        attention_mask=attention_mask,
    )


def build_obs_encoder(config: dict, device: torch.device) -> ObsTransformerEncoder:
    return ObsTransformerEncoder(
        token_dim=int(config["model"]["token_dim"]),
        hidden_dim=int(config["model"].get("obs_hidden_dim", 96)),
        num_layers=int(config["model"].get("obs_layers", 2)),
        num_heads=int(config["model"].get("obs_heads", 4)),
    ).to(device)


def build_unet_denoiser(config: dict, channels: int, device: torch.device) -> Denoiser3DUNet:
    return Denoiser3DUNet(
        channels=int(channels),
        cond_dim=int(config["model"].get("obs_hidden_dim", 96)),
        base=int(config["diffusion"].get("denoiser_base_channels", 32)),
    ).to(device)


def parse_modes(text: str) -> tuple[int, int, int]:
    parts = tuple(int(item.strip()) for item in text.split(","))
    if len(parts) != 3:
        raise ValueError("FNO modes must be formatted as mx,my,mz")
    return parts


def build_fno_denoiser(config: dict, channels: int, device: torch.device) -> FNOTransformerHybrid:
    cfg = config.get("fno_operator_diffusion", {})
    return FNOTransformerHybrid(
        channels=int(channels),
        cond_dim=int(config["model"].get("obs_hidden_dim", 96)),
        width=int(cfg.get("width", 48)),
        modes=parse_modes(str(cfg.get("modes", "8,8,6"))),
        depth=int(cfg.get("depth", 4)),
        transformer_layers=int(cfg.get("transformer_layers", 1)),
        transformer_heads=int(cfg.get("transformer_heads", 4)),
    ).to(device)


def train_noise_model(
    name: str,
    config: dict,
    train_cases: list[BenchmarkCase],
    denoiser: torch.nn.Module,
    steps: int,
    lr: float,
    device: torch.device,
) -> NeuralModel:
    obs_encoder = build_obs_encoder(config, device)
    diffusion = GaussianDiffusion3D(denoiser, timesteps=int(config["diffusion"].get("timesteps", 80)))
    opt = AdamW(
        list(obs_encoder.parameters()) + list(denoiser.parameters()),
        lr=float(lr),
        weight_decay=float(config.get("multisample_diffusion", {}).get("weight_decay", 1e-6)),
    )
    obs_encoder.train()
    denoiser.train()
    for step in range(int(steps)):
        case = train_cases[step % len(train_cases)]
        opt.zero_grad(set_to_none=True)
        cond = obs_encoder(case.tokens, case.token_mask, case.attention_mask)
        loss = diffusion.training_loss(case.latent, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(obs_encoder.parameters()) + list(denoiser.parameters()),
            float(config.get("multisample_diffusion", {}).get("grad_clip", 2.0)),
        )
        opt.step()
        if step == 0 or (step + 1) % max(1, int(steps) // 4) == 0:
            print(f"{name} step {step + 1:04d}/{steps} loss={float(loss.detach().cpu()):.4f}")
    obs_encoder.eval()
    denoiser.eval()
    return NeuralModel(
        name=name,
        warm_start_input="IDW observation-proxy latent; no dense target latent at inference",
        obs_encoder=obs_encoder,
        denoiser=denoiser,
        diffusion=diffusion,
    )


def train_physics_model(
    config: dict,
    ae,
    train_cases: list[BenchmarkCase],
    base_model: NeuralModel,
    steps: int,
    lr: float,
    device: torch.device,
    n_facies: int,
) -> NeuralModel:
    obs_encoder = build_obs_encoder(config, device)
    denoiser = build_unet_denoiser(config, train_cases[0].latent.shape[1], device)
    obs_encoder.load_state_dict(copy.deepcopy(base_model.obs_encoder.state_dict()))
    denoiser.load_state_dict(copy.deepcopy(base_model.denoiser.state_dict()))
    for parameter in obs_encoder.parameters():
        parameter.requires_grad_(False)
    obs_encoder.eval()
    denoiser.train()
    diffusion = GaussianDiffusion3D(denoiser, timesteps=int(config["diffusion"].get("timesteps", 80)))
    train_cfg = config.get("physics_training", {})
    loss_cfg = PhysicsGuidedTrainingConfig(
        epochs=int(steps),
        lr=float(lr),
        noise_weight=float(train_cfg.get("noise_weight", 1.0)),
        physics_weight=float(train_cfg.get("physics_weight", 0.08)),
        latent_anchor_weight=float(train_cfg.get("latent_anchor_weight", 0.05)),
        facies_anchor_weight=float(train_cfg.get("facies_anchor_weight", 0.20)),
        continuous_anchor_weight=float(train_cfg.get("continuous_anchor_weight", 0.05)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
    )
    opt = AdamW(denoiser.parameters(), lr=float(lr), weight_decay=float(train_cfg.get("weight_decay", 1e-6)))
    for step in range(int(steps)):
        case = train_cases[step % len(train_cases)]
        spacing = tuple(float(x) for x in case.sample["grid"].get("spacing", (case.sample["grid"]["dx"], case.sample["grid"]["dy"], case.sample["grid"]["dz"])))
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            cond = obs_encoder(case.tokens, case.token_mask, case.attention_mask)
        loss, parts = physics_guided_diffusion_loss(
            case.latent,
            cond,
            diffusion,
            ae,
            case.target_volume,
            n_facies=n_facies,
            spacing=spacing,
            cfg=loss_cfg,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), float(loss_cfg.grad_clip))
        opt.step()
        if step == 0 or (step + 1) % max(1, int(steps) // 4) == 0:
            print(
                f"PhysicsTrained step {step + 1:04d}/{steps} "
                f"loss={float(parts['loss'].detach().cpu()):.4f} "
                f"physics={float(parts['physics'].detach().cpu()):.4f}"
            )
    denoiser.eval()
    return NeuralModel(
        name="PhysicsTrained",
        warm_start_input="IDW observation-proxy latent; no dense target latent at inference",
        obs_encoder=obs_encoder,
        denoiser=denoiser,
        diffusion=diffusion,
    )


def predict_neural(
    config: dict,
    model: NeuralModel,
    ae,
    case: BenchmarkCase,
    device: torch.device,
    n_facies: int,
    posterior_samples: int,
) -> dict[str, np.ndarray]:
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
    posterior = _posterior_arrays(decoded, n_facies=n_facies)
    return {
        "facies": posterior["facies_mode"],
        "eic": posterior["eic_mean"],
        "temperature": posterior["temperature_mean"],
        "unfrozen_water": posterior["unfrozen_water_mean"],
        "log_resistivity": posterior["log_resistivity_mean"],
    }


def metric_row(model: str, warm_start_input: str, case: BenchmarkCase, pred: dict[str, np.ndarray], n_facies: int, ice_threshold: float) -> dict[str, Any]:
    metrics = synthetic_metrics(
        pred,
        case.sample["fields"],
        case.sample["grid"]["z"],
        n_facies=n_facies,
        ice_threshold=ice_threshold,
    )
    return {
        "model": model,
        "warm_start_input": warm_start_input,
        "scene": case.sample_id,
        "uses_dense_target_at_inference": "False",
        **metrics,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 123) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_p(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    deltas = deltas[np.abs(deltas) > 1e-12]
    n = int(deltas.size)
    if n == 0:
        return float("nan")
    positives = int(np.sum(deltas > 0.0))
    k = min(positives, n - positives)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def summarize_detail(rows: list[dict[str, Any]], test_scenes: int) -> list[dict[str, Any]]:
    models = sorted({str(row["model"]) for row in rows})
    summary: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        out: dict[str, Any] = {
            "model": model,
            "warm_start_input": model_rows[0].get("warm_start_input", ""),
            "scenes": int(test_scenes),
        }
        for metric in METRICS:
            vals = np.asarray([float(row.get(metric, np.nan)) for row in model_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(vals, seed=1000 + len(summary) * 17 + len(metric))
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            out[f"{metric}_ci95_low"] = lo
            out[f"{metric}_ci95_high"] = hi
            out[f"{metric}_mean_sd_ci"] = f"{vals.mean():.4f} +/- {vals.std(ddof=1) if vals.size > 1 else 0.0:.4f} [{lo:.4f}, {hi:.4f}]"
        summary.append(out)
    return summary


def paired_delta_rows(rows: list[dict[str, Any]], baselines: tuple[str, ...] = ("IDW", "RandomForest", "GradientBoosting")) -> list[dict[str, Any]]:
    by_scene_model = {(str(row["scene"]), str(row["model"])): row for row in rows}
    scenes = sorted({str(row["scene"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows if str(row["model"]) not in baselines})
    out: list[dict[str, Any]] = []
    for candidate in models:
        for baseline in baselines:
            for metric in METRICS:
                deltas = []
                for scene in scenes:
                    cand = by_scene_model.get((scene, candidate))
                    base = by_scene_model.get((scene, baseline))
                    if cand is None or base is None:
                        continue
                    c = float(cand.get(metric, np.nan))
                    b = float(base.get(metric, np.nan))
                    if not np.isfinite(c) or not np.isfinite(b):
                        continue
                    delta = (b - c) if metric in LOWER_IS_BETTER else (c - b)
                    deltas.append(delta)
                arr = np.asarray(deltas, dtype=np.float64)
                if arr.size == 0:
                    continue
                lo, hi = bootstrap_ci(arr, seed=2000 + len(out) * 19)
                out.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "improvement_direction": "baseline_minus_candidate" if metric in LOWER_IS_BETTER else "candidate_minus_baseline",
                        "paired_delta_mean": float(arr.mean()),
                        "paired_delta_sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                        "paired_delta_ci95_low": lo,
                        "paired_delta_ci95_high": hi,
                        "sign_flip_p": sign_flip_p(arr),
                        "n": int(arr.size),
                        "wins": int(np.sum(arr > 0.0)),
                        "losses": int(np.sum(arr < 0.0)),
                        "ties": int(np.sum(np.abs(arr) <= 1e-12)),
                    }
                )
    return out


def save_checkpoints(out_dir: Path, models: list[NeuralModel]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        torch.save(
            {
                "model": model.name,
                "warm_start_input": model.warm_start_input,
                "obs_encoder_state": model.obs_encoder.state_dict(),
                "denoiser_state": model.denoiser.state_dict(),
            },
            out_dir / f"nonleaking20_{model.name}.pt",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--train-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=20)
    parser.add_argument("--start-test-index", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-condition-tokens", type=int, default=512)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--latent-steps", type=int, default=240)
    parser.add_argument("--physics-steps", type=int, default=160)
    parser.add_argument("--fno-steps", type=int, default=160)
    parser.add_argument("--rf-trees", type=int, default=80)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    total = int(args.start_test_index) + int(args.test_scenes)
    paths = ensure_synthetic_samples(config, total_samples=total, seed=seed, regenerate=bool(args.regenerate))
    train_paths = paths[: int(args.train_scenes)]
    test_paths = paths[int(args.start_test_index) : int(args.start_test_index) + int(args.test_scenes)]
    if len(test_paths) != int(args.test_scenes):
        raise ValueError("Not enough test scenes after generation")

    device_name = args.device or config.get("diffusion", {}).get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_facies = int(config["model"]["n_facies"])
    ice_threshold = float(config["evaluation"].get("ice_rich_threshold", 0.30))
    ae = _load_autoencoder(config, device)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    ae.eval()

    print(f"preparing {len(train_paths)} train scenes and {len(test_paths)} held-out test scenes on {device}")
    train_cases = [
        prepare_case(config, path, ae, device, n_facies, int(args.max_condition_tokens), seed + idx)
        for idx, path in enumerate(train_paths)
    ]
    test_cases = [
        prepare_case(config, path, ae, device, n_facies, int(args.max_condition_tokens), seed + 10_000 + idx)
        for idx, path in enumerate(test_paths)
    ]

    channels = int(train_cases[0].latent.shape[1])
    latent_model = train_noise_model(
        "LatentDiffusion",
        config,
        train_cases,
        build_unet_denoiser(config, channels, device),
        steps=int(args.latent_steps),
        lr=float(config.get("multisample_diffusion", {}).get("lr", config["diffusion"].get("lr", 5e-4))),
        device=device,
    )
    physics_model = train_physics_model(
        config,
        ae,
        train_cases,
        latent_model,
        steps=int(args.physics_steps),
        lr=float(config.get("physics_training", {}).get("lr", 8e-5)),
        device=device,
        n_facies=n_facies,
    )
    fno_model = train_noise_model(
        "FNOOperator",
        config,
        train_cases,
        build_fno_denoiser(config, channels, device),
        steps=int(args.fno_steps),
        lr=float(config.get("fno_operator_diffusion", {}).get("lr", 4e-4)),
        device=device,
    )
    neural_models = [latent_model, physics_model, fno_model]
    save_checkpoints(Path(config["paths"]["checkpoints_dir"]), neural_models)

    gb_cfg_raw = config.get("baseline_gradient_boosting", {})
    gb_cfg = GradientBoostingConfig(
        max_iter=int(gb_cfg_raw.get("max_iter", 180)),
        learning_rate=float(gb_cfg_raw.get("learning_rate", 0.06)),
        max_leaf_nodes=int(gb_cfg_raw.get("max_leaf_nodes", 31)),
        l2_regularization=float(gb_cfg_raw.get("l2_regularization", 1e-3)),
        min_samples_leaf=int(gb_cfg_raw.get("min_samples_leaf", 8)),
        random_state=seed,
    )
    detail_rows: list[dict[str, Any]] = []
    for scene_idx, case in enumerate(test_cases):
        print(f"evaluating held-out scene {scene_idx + 1:02d}/{len(test_cases)} {case.sample_id}")
        idw = reconstruct_idw(case.sample["observations"], case.sample["grid"], n_facies=n_facies)
        detail_rows.append(metric_row("IDW", "observations only", case, idw, n_facies, ice_threshold))
        rf = reconstruct_random_forest(case.sample, n_estimators=int(args.rf_trees), random_state=seed + scene_idx)
        detail_rows.append(metric_row("RandomForest", "observations + covariates", case, rf, n_facies, ice_threshold))
        gb = reconstruct_gradient_boosting(case.sample, n_facies=n_facies, config=gb_cfg)
        detail_rows.append(metric_row("GradientBoosting", "observations + covariates", case, gb, n_facies, ice_threshold))
        for model in neural_models:
            pred = predict_neural(config, model, ae, case, device, n_facies, int(args.posterior_samples))
            detail_rows.append(metric_row(model.name, model.warm_start_input, case, pred, n_facies, ice_threshold))

    table_dir = Path(config["paths"]["tables_dir"])
    submission_table_dir = Path("tables")
    detail_path = table_dir / "nonleaking_20scene_benchmark_detail.csv"
    summary_path = table_dir / "nonleaking_20scene_benchmark_summary.csv"
    delta_path = table_dir / "nonleaking_20scene_paired_deltas.csv"
    summary_rows = summarize_detail(detail_rows, int(args.test_scenes))
    delta_rows = paired_delta_rows(detail_rows)
    for out_dir in (table_dir, submission_table_dir):
        write_csv(out_dir / "nonleaking_20scene_benchmark_detail.csv", detail_rows)
        write_csv(out_dir / "nonleaking_20scene_benchmark_summary.csv", summary_rows)
        write_csv(out_dir / "nonleaking_20scene_paired_deltas.csv", delta_rows)
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")
    print(f"paired_deltas={delta_path}")
    print(f"submission_tables={submission_table_dir}")


if __name__ == "__main__":
    main()
