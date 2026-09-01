from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.models.autoencoder3d import Autoencoder3D
from cold_recon.training.volume_codec import sample_to_volume_tensor, volume_tensor_to_fields
from cold_recon.utils.config import ensure_dirs, load_config


def _autoencoder_loss(recon: torch.Tensor, target: torch.Tensor, n_facies: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    facies_loss = F.cross_entropy(recon[:, :n_facies], target[:, :n_facies].argmax(dim=1))
    continuous_loss = F.mse_loss(recon[:, n_facies:], target[:, n_facies:])
    eic_weight = 1.0 + 8.0 * (target[:, n_facies : n_facies + 1] > 0.25).float()
    eic_loss = torch.mean(eic_weight * (recon[:, n_facies : n_facies + 1] - target[:, n_facies : n_facies + 1]).square())
    total = facies_loss + continuous_loss + 2.0 * eic_loss
    return total, {"facies": facies_loss, "continuous": continuous_loss, "eic": eic_loss}


def train_autoencoder(config: dict, epochs: int | None = None, device: str | None = None) -> dict:
    cfg = config["autoencoder"]
    sample = load_sample_npz(cfg.get("sample_path", config["training"]["sample_path"]))
    n_facies = int(config["model"]["n_facies"])
    device_name = device or cfg.get("device", config["training"].get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)
    target = sample_to_volume_tensor(sample, n_facies=n_facies).to(dev)
    model = Autoencoder3D(
        in_channels=target.shape[1],
        latent_channels=int(cfg.get("latent_channels", 16)),
        base=int(cfg.get("base_channels", 24)),
    ).to(dev)
    opt = AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=1e-6)
    epochs = int(epochs or cfg.get("epochs", 80))
    history = []
    model.train()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        recon = model(target)
        loss, parts = _autoencoder_loss(recon, target, n_facies)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        row = {"epoch": epoch, "loss": float(loss.detach().cpu())}
        row.update({k: float(v.detach().cpu()) for k, v in parts.items()})
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(
                f"ae epoch {epoch + 1:04d}/{epochs} loss={row['loss']:.4f} "
                f"facies={row['facies']:.4f} cont={row['continuous']:.4f} eic={row['eic']:.4f}"
            )
    ckpt = Path(cfg.get("checkpoint", "outputs/checkpoints/autoencoder3d.pt"))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "in_channels": int(target.shape[1]),
            "n_facies": n_facies,
            "history": history,
        },
        ckpt,
    )
    model.eval()
    with torch.no_grad():
        recon = model(target)
    fields = volume_tensor_to_fields(recon, n_facies=n_facies)
    out_path = Path(cfg.get("reconstruction_path", "outputs/predictions/autoencoder_reconstruction.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **fields)
    return {"checkpoint": ckpt, "reconstruction_path": out_path, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = train_autoencoder(config, epochs=args.epochs, device=args.device)
    print(f"checkpoint={result['checkpoint']}")
    print(f"reconstruction={result['reconstruction_path']}")


if __name__ == "__main__":
    main()
