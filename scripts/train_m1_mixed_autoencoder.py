from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.models.autoencoder3d import Autoencoder3D
from cold_recon.training.mixed_volume_codec import (
    MIXED_CHANNELS,
    mixed_reconstruction_loss,
    sample_to_mixed_tensor,
)
from cold_recon.utils.config import load_config


def _records(manifest_path: Path, split: str) -> tuple[Path, list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == split]
    if not records:
        raise ValueError(f"manifest contains no {split} scenes")
    return manifest_path.parent, records


@torch.no_grad()
def _validate(
    model: Autoencoder3D,
    root: Path,
    records: list[dict],
    device: torch.device,
    limit: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for record in records[: int(limit)]:
        sample = load_sample_npz(root / record["relative_path"])
        target = sample_to_mixed_tensor(sample).to(device)
        loss, _ = mixed_reconstruction_loss(model(target), target)
        losses.append(float(loss.cpu()))
    return {
        "validation_loss": float(np.mean(losses)),
        "validation_scenes": float(len(losses)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_support_guided.yaml")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/m1_support_guided/checkpoints/"
            "mixed_state_autoencoder3d.pt"
        ),
    )
    parser.add_argument(
        "--history",
        default=(
            "outputs/m1_support_guided/tables/"
            "mixed_state_autoencoder_history.json"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = Path(args.manifest or config["m1_training"]["manifest"])
    root, train_records = _records(manifest_path, "train")
    _, validation_records = _records(manifest_path, "validation")
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    auto_cfg = config["autoencoder"]
    model = Autoencoder3D(
        in_channels=MIXED_CHANNELS,
        latent_channels=int(auto_cfg.get("latent_channels", 16)),
        base=int(auto_cfg.get("base_channels", 24)),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(auto_cfg.get("learning_rate", 5.0e-4)),
        weight_decay=float(auto_cfg.get("weight_decay", 1.0e-6)),
    )
    if args.resume_from:
        saved = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        if "optimizer_state" in saved:
            optimizer.load_state_dict(saved["optimizer_state"])

    rng = np.random.default_rng(int(args.seed) + 31)
    history: list[dict[str, float | str]] = []
    started = time.time()
    model.train()
    for step in range(1, int(args.steps) + 1):
        record = train_records[int(rng.integers(0, len(train_records)))]
        sample = load_sample_npz(root / record["relative_path"])
        target = sample_to_mixed_tensor(sample).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(target)
            loss, parts = mixed_reconstruction_loss(prediction, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row: dict[str, float | str] = {
            "step": float(step),
            "scene_id": str(record["scene_id"]),
            "loss": float(loss.detach().cpu()),
            **{
                name: float(value.detach().cpu())
                for name, value in parts.items()
                if name != "total"
            },
        }
        history.append(row)
        if step == 1 or step % 100 == 0:
            print(
                f"step {step:05d}/{int(args.steps)} loss={row['loss']:.5f} "
                f"scene={record['scene_id']}",
                flush=True,
            )

    validation = _validate(
        model,
        root,
        validation_records,
        device,
        int(args.validation_limit),
    )
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "in_channels": MIXED_CHANNELS,
        "latent_channels": int(auto_cfg.get("latent_channels", 16)),
        "base_channels": int(auto_cfg.get("base_channels", 24)),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "state_layout": "mixed_7_class+E+T+W+logR",
        "history": history,
        "validation": validation,
        "manifest_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))[
            "manifest_sha256"
        ],
        "training_seconds": time.time() - started,
    }
    torch.save(payload, checkpoint)
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps({"history": history, "validation": validation}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": str(checkpoint), **validation}, indent=2))


if __name__ == "__main__":
    main()
