from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from cold_recon.data.data_schema import SURFACE_FEATURE_NAMES, load_sample_npz
from cold_recon.models.cold_recon_model import build_model_from_config
from cold_recon.models.implicit_field import prediction_loss
from cold_recon.models.observation_tokenizer import ObservationTokenizer, TokenizerStats, build_observation_attention_mask
from cold_recon.physics.constraints import physics_regularization


@dataclass
class TrainingArrays:
    coords: np.ndarray
    surface: np.ndarray
    targets: dict[str, np.ndarray]
    shape: tuple[int, int, int]
    surface_mean: np.ndarray
    surface_std: np.ndarray


def prepare_training_arrays(sample: dict[str, Any]) -> TrainingArrays:
    grid = sample["grid"]
    nx, ny, nz = len(grid["x"]), len(grid["y"]), len(grid["z"])
    xx, yy, zz = np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij")
    xyz_max = np.array([max(grid["x"][-1], 1.0), max(grid["y"][-1], 1.0), max(grid["z"][-1], 1.0)], dtype=np.float32)
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32) / xyz_max[None, :]
    surf2 = np.stack([sample["surface_features"][name] for name in SURFACE_FEATURE_NAMES], axis=-1).astype(np.float32)
    surf3 = np.repeat(surf2[:, :, None, :], nz, axis=2).reshape(-1, len(SURFACE_FEATURE_NAMES))
    mean = surf3.mean(axis=0, keepdims=True)
    std = surf3.std(axis=0, keepdims=True) + 1e-6
    surf3 = (surf3 - mean) / std
    fields = sample["fields"]
    targets = {
        "facies": fields["facies"].reshape(-1).astype(np.int64),
        "eic": fields["eic"].reshape(-1).astype(np.float32),
        "temperature": fields["temperature"].reshape(-1).astype(np.float32),
        "unfrozen_water": fields["unfrozen_water"].reshape(-1).astype(np.float32),
        "log_resistivity": np.log(np.maximum(fields["resistivity"].reshape(-1).astype(np.float32), 1.0)),
    }
    return TrainingArrays(coords=coords, surface=surf3, targets=targets, shape=(nx, ny, nz), surface_mean=mean, surface_std=std)


def _batch_targets(arrays: TrainingArrays, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(values[idx], device=device)
        for name, values in arrays.targets.items()
    }


def train_implicit_model(
    config: dict[str, Any],
    sample_path: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    train_cfg = config["training"]
    sample_path = Path(sample_path or train_cfg["sample_path"])
    sample = load_sample_npz(sample_path)
    arrays = prepare_training_arrays(sample)
    device_str = device or train_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    dev = torch.device(device_str)
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    obs_tokens = tokenizer.encode_torch(sample["observations"], device=dev).unsqueeze(0)
    obs_mask = torch.zeros((1, obs_tokens.shape[1]), dtype=torch.bool, device=dev)
    obs_attention_mask = build_observation_attention_mask(config, sample["grid"], sample["observations"], device=dev)
    model = build_model_from_config(config).to(dev)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 8e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
    )
    n = arrays.coords.shape[0]
    epochs = int(epochs or train_cfg.get("epochs", 100))
    batch_size = int(batch_size or train_cfg.get("batch_size", 8192))
    rng = np.random.default_rng(int(config.get("project", {}).get("seed", 42)))
    weights = train_cfg.get("loss_weights", {})
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        coords = torch.as_tensor(arrays.coords[idx], dtype=torch.float32, device=dev).unsqueeze(0)
        surface = torch.as_tensor(arrays.surface[idx], dtype=torch.float32, device=dev).unsqueeze(0)
        target = {k: v.unsqueeze(0) for k, v in _batch_targets(arrays, idx, dev).items()}
        optimizer.zero_grad(set_to_none=True)
        pred = model(coords, surface, obs_tokens, obs_mask, obs_attention_mask)
        loss, parts = prediction_loss(pred, target, weights)
        physics = physics_regularization(pred)
        loss = loss + float(weights.get("unfrozen_physics", 0.0)) * physics["unfrozen_water"]
        loss = loss + float(weights.get("resistivity_physics", 0.0)) * physics["resistivity"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        row = {"epoch": float(epoch), "loss": float(loss.detach().cpu())}
        row.update({k: float(v.detach().cpu()) for k, v in parts.items()})
        row.update({f"phys_{k}": float(v.detach().cpu()) for k, v in physics.items()})
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(
                f"epoch {epoch + 1:04d}/{epochs} loss={row['loss']:.4f} "
                f"facies={row.get('facies', 0):.4f} eic={row.get('eic', 0):.4f} temp={row.get('temperature', 0):.4f}"
            )

    ckpt_path = Path(train_cfg.get("checkpoint", "outputs/checkpoints/implicit_mlp.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "surface_mean": arrays.surface_mean,
            "surface_std": arrays.surface_std,
            "tokenizer_stats": tokenizer.stats.__dict__ if tokenizer.stats else None,
            "history": history,
        },
        ckpt_path,
    )
    pred_path = Path(train_cfg.get("prediction_path", "outputs/predictions/implicit_prediction.npz"))
    pred = reconstruct_full_grid(
        model,
        arrays,
        obs_tokens,
        obs_mask,
        dev,
        obs_attention_mask=obs_attention_mask,
        chunk_size=max(batch_size, 8192),
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **pred)
    return {"checkpoint": ckpt_path, "prediction_path": pred_path, "history": history}


@torch.no_grad()
def reconstruct_full_grid(
    model: torch.nn.Module,
    arrays: TrainingArrays,
    obs_tokens: torch.Tensor,
    obs_mask: torch.Tensor,
    device: torch.device,
    obs_attention_mask: torch.Tensor | None = None,
    chunk_size: int = 32768,
) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {
        "facies": [],
        "eic": [],
        "temperature": [],
        "unfrozen_water": [],
        "log_resistivity": [],
    }
    n = arrays.coords.shape[0]
    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        coords = torch.as_tensor(arrays.coords[sl], dtype=torch.float32, device=device).unsqueeze(0)
        surface = torch.as_tensor(arrays.surface[sl], dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(coords, surface, obs_tokens, obs_mask, obs_attention_mask)
        outs["facies"].append(torch.argmax(pred["facies_logits"], dim=-1).squeeze(0).cpu().numpy().astype(np.int16))
        for key in ["eic", "temperature", "unfrozen_water", "log_resistivity"]:
            outs[key].append(pred[key].squeeze(0).cpu().numpy().astype(np.float32))
    shape = arrays.shape
    result = {key: np.concatenate(parts, axis=0).reshape(shape) for key, parts in outs.items()}
    result["resistivity"] = np.exp(np.clip(result["log_resistivity"], 0.0, 12.0)).astype(np.float32)
    return result
