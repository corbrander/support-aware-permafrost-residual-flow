from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.data.observation_augmentation import (
    ObservationAugmentationConfig,
    augment_observations,
)
from cold_recon.data.support_raster import (
    PreparedSupportRaster,
    build_nearest_voxel_raster,
)
from cold_recon.models.autoencoder3d import Autoencoder3D
from cold_recon.models.support_aware_residual_flow import SupportAwareResidualFlow3D
from cold_recon.training.mixed_volume_codec import (
    MIXED_CHANNELS,
    mixed_reconstruction_loss,
    prior_to_mixed_tensor,
    sample_to_mixed_tensor,
)
from cold_recon.utils.config import load_config
from scripts.build_tree_prior_residual_posterior import (
    coordinate_channels,
    surface_channels,
)
from scripts.train_m1_support_guided_flow import (
    _load_or_build_prior,
    _manifest_records,
    _subsample_tokens,
    _token_tensor,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "A2": {
        "name": "legacy_nearest_raster_flow",
        "support_raster": "nearest_voxel",
        "token_conditioning": False,
        "noise_conditioning": False,
    },
    "A3": {
        "name": "explicit_support_raster_flow",
        "support_raster": "explicit",
        "token_conditioning": False,
        "noise_conditioning": False,
    },
    "A5": {
        "name": "hybrid_support_encoder",
        "support_raster": "explicit",
        "token_conditioning": True,
        "noise_conditioning": False,
    },
    "A6": {
        "name": "noise_conditioned_hybrid",
        "support_raster": "explicit",
        "token_conditioning": True,
        "noise_conditioning": True,
    },
}


def _load_autoencoder(path: Path, device: torch.device) -> Autoencoder3D:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if int(saved["in_channels"]) != MIXED_CHANNELS:
        raise ValueError("mixed-state autoencoder must have 11 input channels")
    model = Autoencoder3D(
        in_channels=MIXED_CHANNELS,
        latent_channels=int(saved["latent_channels"]),
        base=int(saved["base_channels"]),
    ).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _mixed_context_raster(
    sample: dict,
    prior_tensor: torch.Tensor,
    observations,
    *,
    support_raster: str,
    prepared_support: PreparedSupportRaster | None = None,
) -> torch.Tensor:
    if support_raster == "explicit":
        if prepared_support is None:
            prepared_support = PreparedSupportRaster.prepare(
                sample["observations"], sample["grid"]
            )
        support, diagnostics = prepared_support.apply(observations)
    elif support_raster == "nearest_voxel":
        support, diagnostics = build_nearest_voxel_raster(
            observations, sample["grid"]
        )
    else:
        raise ValueError(f"unknown support raster: {support_raster}")
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


def train(args: argparse.Namespace) -> dict[str, Any]:
    ablation_id = str(args.ablation_id).upper()
    if ablation_id not in VARIANTS:
        raise ValueError(f"mixed-state flow supports {sorted(VARIANTS)}")
    variant = VARIANTS[ablation_id]
    config = load_config(args.config)
    manifest_path = Path(args.manifest or config["m1_training"]["manifest"])
    root, records, manifest = _manifest_records(manifest_path, "train")
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)

    autoencoder_path = Path(args.autoencoder_checkpoint)
    autoencoder = _load_autoencoder(autoencoder_path, device)
    latent_channels = int(config["model"]["latent_channels"])
    raster_channels = int(config["model"]["raster_channels"]) - (
        int(config["model"]["factorized_channels"]) - MIXED_CHANNELS
    )
    model_cfg = config["model"]
    model = SupportAwareResidualFlow3D(
        latent_channels=latent_channels,
        raster_in_channels=raster_channels,
        token_dim=int(model_cfg["token_dim"]),
        context_channels=int(model_cfg["context_channels"]),
        width=int(model_cfg["width"]),
        modes=tuple(int(value) for value in model_cfg["modes"]),
        depth=int(model_cfg["depth"]),
        attention_hidden=int(model_cfg["attention_hidden"]),
        attention_chunk=int(model_cfg["attention_chunk"]),
        support_extent_offset=int(model_cfg["support_extent_offset"]),
        use_token_conditioning=bool(variant["token_conditioning"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["m1_training"]["learning_rate"]),
        weight_decay=float(config["m1_training"]["weight_decay"]),
    )
    rng = np.random.default_rng(seed + 701)
    augmentation_config = ObservationAugmentationConfig(
        min_boreholes=10_000,
        min_ert_profiles=10_000,
        source_dropout_probability=0.0,
    )
    support_cache: dict[str, PreparedSupportRaster] = {}
    history: list[dict[str, Any]] = []
    started = time.time()
    for step in range(1, int(args.steps) + 1):
        record = records[int(rng.integers(0, len(records)))]
        sample = load_sample_npz(root / record["relative_path"])
        prior = _load_or_build_prior(
            sample,
            record,
            root / "prior_cache" / "train",
            int(config["model"]["n_facies"]),
            int(args.rf_trees),
        )
        prior_tensor = prior_to_mixed_tensor(prior)
        target = sample_to_mixed_tensor(sample)
        observations = sample["observations"]
        if bool(variant["noise_conditioning"]):
            observations, augmentation = augment_observations(
                observations, rng, augmentation_config
            )
        else:
            augmentation = {"sigma_multiplier": 1.0}
        prepared = support_cache.get(record["scene_id"])
        if prepared is None and variant["support_raster"] == "explicit":
            prepared = PreparedSupportRaster.prepare(
                sample["observations"], sample["grid"]
            )
            support_cache[record["scene_id"]] = prepared
        raster = _mixed_context_raster(
            sample,
            prior_tensor,
            observations,
            support_raster=str(variant["support_raster"]),
            prepared_support=prepared,
        ).to(device)
        token_observations = _subsample_tokens(
            observations, int(model_cfg["max_condition_tokens"]), rng
        )
        tokens = _token_tensor(token_observations, sample, config, device)
        prior_tensor = prior_tensor.to(device)
        target = target.to(device)
        with torch.no_grad():
            anchor = autoencoder.encode(prior_tensor)
            target_latent = autoencoder.encode(target)
        optimizer.zero_grad(set_to_none=True)
        amp = device.type == "cuda"
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp
        ):
            encoded = model.encode_context(
                raster, tokens, target_shape=anchor.shape[-3:]
            )
            target_residual = target_latent - anchor
            source = torch.randn(
                (int(args.source_ensemble), *target_residual.shape[1:]),
                device=device,
                dtype=target_residual.dtype,
            )
            target_residual_b = target_residual.expand(
                int(args.source_ensemble), -1, -1, -1, -1
            )
            tau = torch.rand(
                (int(args.source_ensemble),),
                device=device,
                dtype=target_residual.dtype,
            )
            tau_view = tau[:, None, None, None, None]
            state = (1.0 - tau_view) * source + tau_view * target_residual_b
            target_velocity = target_residual_b - source
            predicted_velocity = model.velocity_from_encoded(
                state,
                tau * 79.0,
                anchor.expand(int(args.source_ensemble), -1, -1, -1, -1),
                encoded.expand(int(args.source_ensemble), -1, -1, -1, -1),
            )
            flow_loss = F.mse_loss(predicted_velocity, target_velocity)
            predicted_residual = state + (1.0 - tau_view) * predicted_velocity
            decoded = autoencoder.decode(anchor + predicted_residual[:1])
            reconstruction, _ = mixed_reconstruction_loss(decoded, target)
            total = flow_loss + float(args.reconstruction_weight) * reconstruction
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        row = {
            "step": step,
            "scene_id": record["scene_id"],
            "family": record["generator_family"],
            "loss": float(total.detach().cpu()),
            "flow": float(flow_loss.detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "sigma_multiplier": float(augmentation["sigma_multiplier"]),
        }
        history.append(row)
        if step == 1 or step % 50 == 0:
            print(
                f"{ablation_id} step {step:05d}/{int(args.steps)} "
                f"loss={row['loss']:.5f} flow={row['flow']:.5f} "
                f"family={row['family']}",
                flush=True,
            )

    checkpoint = Path(
        args.checkpoint
        or (
            "outputs/m1_support_guided/checkpoints/"
            f"ablation_{ablation_id.lower()}_seed{seed}.pt"
        )
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "seed": seed,
        "steps": int(args.steps),
        "ablation_id": ablation_id,
        "variant": variant,
        "state_layout": "mixed_7_class+E+T+W+logR",
        "autoencoder_checkpoint": str(autoencoder_path),
        "raster_channels": raster_channels,
        "manifest_sha256": manifest["manifest_sha256"],
        "history": history,
        "training_seconds": time.time() - started,
        "max_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    torch.save(payload, checkpoint)
    history_path = Path(
        args.history
        or (
            "outputs/m1_support_guided/tables/"
            f"ablation_{ablation_id.lower()}_training_seed{seed}.json"
        )
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "checkpoint": str(checkpoint),
        "history": str(history_path),
        "training_seconds": payload["training_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_support_guided.yaml")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--ablation-id", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--source-ensemble", type=int, default=2)
    parser.add_argument("--reconstruction-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rf-trees", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--autoencoder-checkpoint",
        default=(
            "outputs/m1_support_guided/checkpoints/"
            "mixed_state_autoencoder3d.pt"
        ),
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--history", default="")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
