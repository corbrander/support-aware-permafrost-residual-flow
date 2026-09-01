from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable, load_sample_npz
from cold_recon.data.observation_augmentation import (
    ObservationAugmentationConfig,
    augment_observations,
)
from cold_recon.data.state_factorization import factorize_legacy_state
from cold_recon.data.support_raster import (
    PreparedNearestVoxelRaster,
    PreparedSupportRaster,
    build_nearest_voxel_raster,
    build_support_raster,
    collapse_to_nearest_voxel_observations,
)
from cold_recon.models.autoencoder3d import Autoencoder3D
from cold_recon.models.high_eic_head import HighEICEventHead3D, focal_tversky_loss
from cold_recon.models.observation_tokenizer import ObservationTokenizer
from cold_recon.models.posterior_decomposition import LocalBiasScaleHead3D
from cold_recon.models.support_aware_residual_flow import SupportAwareResidualFlow3D
from cold_recon.training.factorized_volume_codec import (
    FACTORIZED_CHANNELS,
    N_ICE_STRUCTURE,
    N_LITHOLOGY,
    N_THERMAL_STATE,
    factorized_label_masks,
    factorized_reconstruction_loss,
    sample_to_factorized_tensor,
)
from cold_recon.training.probabilistic_constitutive import probabilistic_constitutive_loss
from cold_recon.utils.config import load_config

from scripts.build_tree_prior_residual_posterior import (
    coordinate_channels,
    surface_channels,
    tree_prior_fields,
)


FACTORIZED_ABLATIONS: dict[str, dict[str, bool]] = {
    "A7": {
        "bias_anomaly_decomposition": False,
        "probabilistic_constitutive": False,
        "high_eic_event_head": False,
    },
    "A8": {
        "bias_anomaly_decomposition": True,
        "probabilistic_constitutive": False,
        "high_eic_event_head": False,
    },
    "A9": {
        "bias_anomaly_decomposition": True,
        "probabilistic_constitutive": True,
        "high_eic_event_head": False,
    },
}


def _load_autoencoder(config: dict, device: torch.device) -> Autoencoder3D:
    cfg = config["autoencoder"]
    saved = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = Autoencoder3D(
        in_channels=int(saved.get("in_channels", FACTORIZED_CHANNELS)),
        latent_channels=int(saved.get("latent_channels", cfg.get("latent_channels", 16))),
        base=int(saved.get("base_channels", cfg.get("base_channels", 24))),
    ).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _prior_tensor(prior: dict[str, np.ndarray]) -> torch.Tensor:
    state = factorize_legacy_state(prior["facies"], prior["eic"], prior["temperature"])
    pseudo = {
        "fields": {
            **state.as_fields(),
            "eic": np.asarray(prior["eic"], dtype=np.float32),
            "temperature": np.asarray(prior["temperature"], dtype=np.float32),
            "unfrozen_water": np.asarray(prior["unfrozen_water"], dtype=np.float32),
            "resistivity": np.exp(np.asarray(prior["log_resistivity"], dtype=np.float32)),
        }
    }
    return sample_to_factorized_tensor(pseudo)


def _load_or_build_prior(
    sample: dict[str, Any],
    record: dict,
    cache_dir: Path,
    n_facies: int,
    rf_trees: int,
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{record['scene_id']}_rf{int(rf_trees)}.npz"
    if cache_path.exists():
        # Independent formal seed runners can request the same deterministic
        # tree prior concurrently. Retry if an older direct writer has exposed
        # a partially written zip archive.
        for attempt in range(61):
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    return {
                        name: np.asarray(data[name]) for name in data.files
                    }
            except (OSError, EOFError, ValueError, zipfile.BadZipFile):
                if attempt == 60:
                    break
                time.sleep(0.5)
    prior = tree_prior_fields(
        sample,
        n_facies=int(n_facies),
        seed=int(record["seed"]) + 91,
        rf_trees=int(rf_trees),
    )
    arrays = {name: np.asarray(value) for name, value in prior.items()}
    temporary = cache_path.with_name(
        f"{cache_path.stem}.{os.getpid()}.tmp.npz"
    )
    np.savez_compressed(temporary, **arrays)
    for attempt in range(61):
        try:
            os.replace(temporary, cache_path)
            break
        except PermissionError:
            # A legacy direct writer may still own the destination on Windows.
            # Prefer its completed deterministic cache once it becomes valid;
            # otherwise retry the atomic replacement for at most 30 seconds.
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    existing = {
                        name: np.asarray(data[name]) for name in data.files
                    }
                temporary.unlink(missing_ok=True)
                return existing
            except (
                OSError,
                EOFError,
                ValueError,
                zipfile.BadZipFile,
            ):
                if attempt == 60:
                    temporary.unlink(missing_ok=True)
                    raise
                time.sleep(0.5)
    return arrays


def _context_raster(
    sample: dict[str, Any],
    prior_tensor: torch.Tensor,
    observations: ObservationTable,
    prepared_support: PreparedSupportRaster | PreparedNearestVoxelRaster | None = None,
    support_mode: str = "support-aware",
) -> torch.Tensor:
    if support_mode == "nearest-voxel":
        support, diagnostics = (
            prepared_support.apply(observations)
            if prepared_support is not None
            else build_nearest_voxel_raster(observations, sample["grid"])
        )
    elif support_mode == "support-aware":
        support, diagnostics = (
            prepared_support.apply(observations)
            if prepared_support is not None
            else build_support_raster(observations, sample["grid"])
        )
    else:
        raise ValueError(f"unknown support mode: {support_mode}")
    nz = prior_tensor.shape[-1]
    raster = np.concatenate(
        [
            prior_tensor[0].numpy(),
            coordinate_channels(sample),
            surface_channels(sample, nz=nz),
            support,
            diagnostics["support_density"][None, ...],
            diagnostics["distance_to_support"][None, ...],
        ],
        axis=0,
    )
    return torch.as_tensor(raster, dtype=torch.float32).unsqueeze(0)


def _subsample_tokens(
    observations: ObservationTable,
    maximum: int,
    rng: np.random.Generator,
) -> ObservationTable:
    valid = np.flatnonzero(observations.mask)
    if len(valid) <= int(maximum):
        return observations.subset(valid)
    retained: list[int] = []
    type_ids = np.unique(observations.type_ids[valid])
    per_type = max(1, int(maximum) // max(len(type_ids), 1))
    for type_id in type_ids:
        candidates = valid[observations.type_ids[valid] == type_id]
        weights = np.maximum(observations.quality[candidates], 0.05)
        weights = weights / weights.sum()
        take = min(len(candidates), per_type)
        retained.extend(rng.choice(candidates, size=take, replace=False, p=weights).tolist())
    if len(retained) < int(maximum):
        remaining = np.setdiff1d(valid, np.asarray(retained, dtype=np.int64), assume_unique=False)
        take = min(len(remaining), int(maximum) - len(retained))
        retained.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return observations.subset(np.asarray(sorted(retained[: int(maximum)]), dtype=np.int64))


def _token_tensor(
    observations: ObservationTable,
    sample: dict[str, Any],
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    model_cfg = config["model"]
    tokenizer = ObservationTokenizer(
        n_types=int(model_cfg["n_types"]),
        support_aware=True,
        n_support_types=int(model_cfg["n_support_types"]),
        n_sites=int(model_cfg["n_sites"]),
        n_sources=int(model_cfg["n_sources"]),
    ).fit_from_grid(sample["grid"])
    return tokenizer.encode_torch(observations, device=device).unsqueeze(0)


def _manifest_records(path: Path, split: str) -> tuple[Path, list[dict], dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == split]
    return path.parent, records, manifest


def _build_models(config: dict, latent_channels: int, device: torch.device):
    cfg = config["model"]
    model = SupportAwareResidualFlow3D(
        latent_channels=int(latent_channels),
        raster_in_channels=int(cfg["raster_channels"]),
        token_dim=int(cfg["token_dim"]),
        context_channels=int(cfg["context_channels"]),
        width=int(cfg["width"]),
        modes=tuple(int(value) for value in cfg["modes"]),
        depth=int(cfg["depth"]),
        attention_hidden=int(cfg["attention_hidden"]),
        attention_chunk=int(cfg["attention_chunk"]),
        support_extent_offset=int(cfg["support_extent_offset"]),
    ).to(device)
    bias_head = LocalBiasScaleHead3D(
        int(cfg["context_channels"]), int(latent_channels), width=int(cfg["context_channels"])
    ).to(device)
    event_head = HighEICEventHead3D(
        int(cfg["context_channels"]),
        thresholds=tuple(config["evaluation"]["high_eic_thresholds"]),
        raster_channels=int(cfg["raster_channels"]),
    ).to(device)
    return model, bias_head, event_head


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.autoencoder_checkpoint:
        config["autoencoder"]["checkpoint"] = args.autoencoder_checkpoint
    train_cfg = config["m1_training"]
    ablation_id = str(getattr(args, "ablation_id", "") or "").upper()
    variant = FACTORIZED_ABLATIONS.get(
        ablation_id,
        {
            "bias_anomaly_decomposition": True,
            "probabilistic_constitutive": True,
            "high_eic_event_head": True,
        },
    )
    manifest_path = Path(args.manifest or train_cfg["manifest"])
    root, records, manifest = _manifest_records(manifest_path, "train")
    if not records:
        raise ValueError("no training records")
    missing = [record["relative_path"] for record in records if not (root / record["relative_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"benchmark not materialized; first missing scene: {missing[0]}")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    seed = int(args.seed)
    support_mode = str(getattr(args, "support_mode", "support-aware"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)

    autoencoder = _load_autoencoder(config, device)
    latent_channels = int(config["model"]["latent_channels"])
    model, bias_head, event_head = _build_models(config, latent_channels, device)
    parameters = list(model.parameters())
    if bool(variant["bias_anomaly_decomposition"]):
        parameters += list(bias_head.parameters())
    if bool(variant["high_eic_event_head"]):
        parameters += list(event_head.parameters())
    optimizer = AdamW(
        parameters,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    if args.resume_from:
        saved = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        bias_head.load_state_dict(saved["bias_head_state"])
        event_head.load_state_dict(saved["event_head_state"], strict=False)
        if "optimizer_state" in saved:
            optimizer.load_state_dict(saved["optimizer_state"])

    rng = np.random.default_rng(seed + 701)
    steps = int(args.steps or train_cfg["steps"])
    source_ensemble = int(train_cfg["source_ensemble"])
    n_facies = int(config["model"]["n_facies"])
    rf_trees = int(args.rf_trees or train_cfg["rf_trees"])
    cache_dir = root / "prior_cache" / "train"
    history: list[dict[str, Any]] = []
    support_cache: dict[
        str, PreparedSupportRaster | PreparedNearestVoxelRaster
    ] = {}
    # Borehole/profile counts already vary across the 500 generated scenes.
    # Keep each scene's anchor and active mask aligned; within-scene augmentation
    # changes uncertainty, bias, outliers, and values but does not secretly give
    # the cached tree anchor observations that were deleted from the encoder.
    augmentation_config = ObservationAugmentationConfig(
        min_boreholes=10_000,
        min_ert_profiles=10_000,
        source_dropout_probability=0.0,
    )
    start_time = time.time()
    for step in range(1, steps + 1):
        record = records[int(rng.integers(0, len(records)))]
        sample = load_sample_npz(root / record["relative_path"])
        prepared_support = support_cache.get(record["scene_id"])
        if prepared_support is None:
            prepared_support = (
                PreparedNearestVoxelRaster.prepare(
                    sample["observations"], sample["grid"]
                )
                if support_mode == "nearest-voxel"
                else PreparedSupportRaster.prepare(
                    sample["observations"], sample["grid"]
                )
            )
            if len(support_cache) >= 512:
                support_cache.pop(next(iter(support_cache)))
            support_cache[record["scene_id"]] = prepared_support
        prior = _load_or_build_prior(sample, record, cache_dir, n_facies, rf_trees)
        prior_tensor = _prior_tensor(prior)
        target = sample_to_factorized_tensor(sample)
        augmented, augmentation = augment_observations(
            sample["observations"], rng, augmentation_config
        )
        conditioning_observations = (
            collapse_to_nearest_voxel_observations(augmented, sample["grid"])
            if support_mode == "nearest-voxel"
            else augmented
        )
        token_observations = _subsample_tokens(
            conditioning_observations,
            int(config["model"]["max_condition_tokens"]),
            rng,
        )
        raster = _context_raster(
            sample,
            prior_tensor,
            conditioning_observations,
            prepared_support=prepared_support,
            support_mode=support_mode,
        ).to(device)
        tokens = _token_tensor(token_observations, sample, config, device)
        target = target.to(device)
        prior_tensor = prior_tensor.to(device)
        masks = {name: value.to(device) for name, value in factorized_label_masks(sample).items()}
        with torch.no_grad():
            anchor = autoencoder.encode(prior_tensor)
            target_latent = autoencoder.encode(target)
        optimizer.zero_grad(set_to_none=True)
        amp = device.type == "cuda" and str(train_cfg.get("amp", "bfloat16")) == "bfloat16"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            encoded = model.encode_context(raster, tokens, target_shape=anchor.shape[-3:])
            target_correction = target_latent - anchor
            if bool(variant["bias_anomaly_decomposition"]):
                bias, gate, local_scale = bias_head(encoded)
                gated_bias = gate * bias
            else:
                gated_bias = torch.zeros_like(target_correction)
                gate = torch.zeros_like(target_correction[:, :1])
                local_scale = torch.ones_like(target_correction)
            target_anomaly = (target_correction - gated_bias) / local_scale.clamp_min(1.0e-4)
            source = torch.randn(
                (source_ensemble, *target_anomaly.shape[1:]), device=device, dtype=target_anomaly.dtype
            )
            target_anomaly_b = target_anomaly.expand(source_ensemble, -1, -1, -1, -1)
            tau = torch.rand((source_ensemble,), device=device, dtype=target_anomaly.dtype)
            tau_view = tau[:, None, None, None, None]
            state = (1.0 - tau_view) * source + tau_view * target_anomaly_b
            target_velocity = target_anomaly_b - source
            predicted_velocity = model.velocity_from_encoded(
                state,
                tau * 79.0,
                anchor.expand(source_ensemble, -1, -1, -1, -1),
                encoded.expand(source_ensemble, -1, -1, -1, -1),
            )
            flow_loss = F.mse_loss(predicted_velocity, target_velocity)
            predicted_anomaly = state + (1.0 - tau_view) * predicted_velocity
            bias_loss = F.smooth_l1_loss(gated_bias, target_correction)
            scale_nll = torch.mean(target_anomaly.square() + 2.0 * torch.log(local_scale.clamp_min(1.0e-4)))
            predicted_latent = anchor + gated_bias + local_scale * predicted_anomaly[:1]
            decoded = autoencoder.decode(predicted_latent)
            reconstruction, _ = factorized_reconstruction_loss(decoded, target, masks)
            if bool(variant["probabilistic_constitutive"]):
                constitutive, _ = probabilistic_constitutive_loss(decoded)
            else:
                constitutive = reconstruction.new_zeros(())
            if bool(variant["high_eic_event_head"]):
                event_logits = event_head(encoded)
                eic = target[:, N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE :][:, :1]
                thresholds = torch.as_tensor(
                    config["evaluation"]["high_eic_thresholds"], device=device, dtype=eic.dtype
                )
                event_target = (eic > thresholds[None, :, None, None, None]).float()
                event_target = F.adaptive_max_pool3d(
                    event_target, output_size=event_logits.shape[-3:]
                )
                event_loss = focal_tversky_loss(event_logits, event_target)
            else:
                event_loss = reconstruction.new_zeros(())
            total = (
                float(train_cfg["flow_weight"]) * flow_loss
                + float(train_cfg["bias_weight"])
                * (bias_loss + 0.02 * scale_nll)
                * float(bool(variant["bias_anomaly_decomposition"]))
                + float(train_cfg["reconstruction_weight"]) * reconstruction
                + float(train_cfg["constitutive_weight"]) * constitutive
                + float(train_cfg["event_weight"]) * event_loss
            )
        total.backward()
        nn.utils.clip_grad_norm_(parameters, float(train_cfg["grad_clip"]))
        optimizer.step()
        row = {
            "step": step,
            "scene_id": record["scene_id"],
            "family": record["generator_family"],
            "loss": float(total.detach().cpu()),
            "flow": float(flow_loss.detach().cpu()),
            "bias": float(bias_loss.detach().cpu()),
            "scale_nll": float(scale_nll.detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "constitutive": float(constitutive.detach().cpu()),
            "event": float(event_loss.detach().cpu()),
            "gate_mean": float(gate.mean().detach().cpu()),
            "local_scale_mean": float(local_scale.mean().detach().cpu()),
            "sigma_multiplier": float(augmentation["sigma_multiplier"]),
        }
        history.append(row)
        if step == 1 or step % 50 == 0:
            print(
                f"step {step:05d}/{steps} loss={row['loss']:.5f} flow={row['flow']:.5f} "
                f"bias={row['bias']:.5f} event={row['event']:.5f} family={row['family']}",
                flush=True,
            )

    checkpoint_pattern = str(train_cfg["checkpoint_pattern"])
    default_checkpoint = (
        f"outputs/m1_support_guided/checkpoints/ablation_{ablation_id.lower()}_seed{seed}.pt"
        if ablation_id
        else checkpoint_pattern.format(seed=seed)
    )
    checkpoint = Path(args.checkpoint or default_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "model_state": model.state_dict(),
        "bias_head_state": bias_head.state_dict(),
        "event_head_state": event_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "seed": seed,
        "steps": steps,
        "manifest_sha256": manifest["manifest_sha256"],
        "history": history,
        "training_seconds": time.time() - start_time,
        "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "model_name": (
            "nearest-voxel noise-conditioned residual flow"
            if support_mode == "nearest-voxel"
            else "support-aware noise-conditioned residual flow"
        ),
        "support_mode": support_mode,
        "state_layout": "L+S+I+E+T+W+logR",
        "ablation_id": ablation_id or "A11_training",
        "variant": variant,
    }
    torch.save(saved, checkpoint)
    history_path = Path(
        args.history
        or Path(config["paths"]["tables_dir"])
        / (
            f"ablation_{ablation_id.lower()}_training_seed{seed}.json"
            if ablation_id
            else f"m1_training_history_seed{seed}.json"
        )
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"checkpoint": str(checkpoint), "history": str(history_path), "training_seconds": saved["training_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_support_guided.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--autoencoder-checkpoint", default="")
    parser.add_argument("--rf-trees", type=int, default=None)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--support-mode",
        choices=("support-aware", "nearest-voxel"),
        default="support-aware",
    )
    parser.add_argument("--history", default="")
    parser.add_argument("--ablation-id", default="", choices=["", *sorted(FACTORIZED_ABLATIONS)])
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
