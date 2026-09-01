from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.uncertainty import facies_entropy
from cold_recon.models.autoencoder3d import Autoencoder3D
from cold_recon.models.denoiser3d_unet import Denoiser3DUNet
from cold_recon.models.diffusion import GaussianDiffusion3D
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.training.volume_codec import batch_volume_to_field_ensemble, sample_to_volume_tensor
from cold_recon.utils.config import ensure_dirs, load_config


def _load_autoencoder(config: dict, device: torch.device) -> Autoencoder3D:
    cfg = config["diffusion"]
    ckpt_path = Path(cfg.get("autoencoder_checkpoint", config["autoencoder"]["checkpoint"]))
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Autoencoder3D(
        in_channels=int(ckpt.get("in_channels", 11)),
        latent_channels=int(config["autoencoder"].get("latent_channels", 16)),
        base=int(config["autoencoder"].get("base_channels", 24)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _posterior_arrays(decoded: torch.Tensor, n_facies: int) -> dict[str, np.ndarray]:
    ensemble = batch_volume_to_field_ensemble(decoded, n_facies=n_facies)
    out: dict[str, np.ndarray] = {}
    for key, arr in ensemble.items():
        out[f"{key}_samples"] = arr
    for key in ["eic", "temperature", "unfrozen_water", "log_resistivity", "resistivity"]:
        arr = ensemble[key]
        out[f"{key}_mean"] = arr.mean(axis=0).astype(np.float32)
        out[f"{key}_std"] = arr.std(axis=0).astype(np.float32)
    facies_samples = ensemble["facies"].astype(np.int64)
    probs = np.zeros((*facies_samples.shape[1:], n_facies), dtype=np.float32)
    for cls in range(n_facies):
        probs[..., cls] = np.mean(facies_samples == cls, axis=0)
    out["facies_probability"] = probs
    out["facies_entropy"] = facies_entropy(probs).astype(np.float32)
    out["facies_mode"] = np.argmax(probs, axis=-1).astype(np.int16)
    out["ice_rich_probability"] = np.mean(ensemble["eic"] > 0.30, axis=0).astype(np.float32)
    return out


def train_diffusion(config: dict, epochs: int | None = None, samples: int | None = None, device: str | None = None) -> dict:
    cfg = config["diffusion"]
    sample = load_sample_npz(cfg.get("sample_path", config["training"]["sample_path"]))
    n_facies = int(config["model"]["n_facies"])
    device_name = device or cfg.get("device", config["training"].get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)
    ae = _load_autoencoder(config, dev)
    target = sample_to_volume_tensor(sample, n_facies=n_facies).to(dev)
    with torch.no_grad():
        latent = ae.encode(target)

    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    obs_tokens = tokenizer.encode_torch(sample["observations"], device=dev).unsqueeze(0)
    obs_mask = torch.zeros((1, obs_tokens.shape[1]), dtype=torch.bool, device=dev)
    obs_attention_mask = build_observation_attention_mask(config, sample["grid"], sample["observations"], device=dev)
    obs_hidden = int(config["model"].get("obs_hidden_dim", 96))
    obs_encoder = ObsTransformerEncoder(
        token_dim=int(config["model"]["token_dim"]),
        hidden_dim=obs_hidden,
        num_layers=int(config["model"].get("obs_layers", 2)),
        num_heads=int(config["model"].get("obs_heads", 4)),
    ).to(dev)
    denoiser = Denoiser3DUNet(
        channels=int(latent.shape[1]),
        cond_dim=obs_hidden,
        base=int(cfg.get("denoiser_base_channels", 32)),
    ).to(dev)
    diffusion = GaussianDiffusion3D(
        denoiser,
        timesteps=int(cfg.get("timesteps", 80)),
    )
    opt = AdamW(list(obs_encoder.parameters()) + list(denoiser.parameters()), lr=float(cfg.get("lr", 5e-4)), weight_decay=1e-6)
    epochs = int(epochs or cfg.get("epochs", 200))
    history = []
    obs_encoder.train()
    denoiser.train()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask)
        loss = diffusion.training_loss(latent, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(obs_encoder.parameters()) + list(denoiser.parameters()), 2.0)
        opt.step()
        row = {"epoch": epoch, "loss": float(loss.detach().cpu())}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"diff epoch {epoch + 1:04d}/{epochs} loss={row['loss']:.4f}")

    ckpt_path = Path(cfg.get("checkpoint", "outputs/checkpoints/latent_diffusion.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "denoiser_state": denoiser.state_dict(),
            "obs_encoder_state": obs_encoder.state_dict(),
            "config": config,
            "latent_shape": tuple(latent.shape[1:]),
            "n_facies": n_facies,
            "history": history,
        },
        ckpt_path,
    )

    obs_encoder.eval()
    denoiser.eval()
    k = int(samples or cfg.get("posterior_samples", 8))
    with torch.no_grad():
        cond = obs_encoder(obs_tokens, obs_mask, obs_attention_mask).repeat(k, 1)
        strategy = str(cfg.get("sampling_strategy", "warm_latent"))
        if strategy == "ancestral":
            latents = diffusion.sample((k, *latent.shape[1:]), cond, dev)
        elif strategy == "warm_latent":
            scale = float(cfg.get("posterior_noise_scale", 0.08))
            correction = float(cfg.get("posterior_correction_scale", 0.15))
            latents = latent.repeat(k, 1, 1, 1, 1) + scale * torch.randn((k, *latent.shape[1:]), device=dev)
            t_mid = max(int(cfg.get("timesteps", 80)) // 2, 1)
            t = torch.full((k,), t_mid, device=dev, dtype=torch.long)
            latents = latents - correction * denoiser(latents, t, cond)
        else:
            raise ValueError(f"Unknown diffusion sampling_strategy={strategy}")
        decoded = ae.decode(latents)
    posterior = _posterior_arrays(decoded, n_facies)
    out_path = Path(cfg.get("posterior_path", "outputs/predictions/diffusion_posterior.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **posterior)
    return {"checkpoint": ckpt_path, "posterior_path": out_path, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = train_diffusion(config, epochs=args.epochs, samples=args.samples, device=args.device)
    print(f"checkpoint={result['checkpoint']}")
    print(f"posterior={result['posterior_path']}")


if __name__ == "__main__":
    main()
