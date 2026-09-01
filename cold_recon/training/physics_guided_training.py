from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from cold_recon.models.diffusion import GaussianDiffusion3D
from cold_recon.training.physics_guided_sampling import LatentPhysicsGuidanceConfig, latent_physics_loss


@dataclass(frozen=True)
class PhysicsGuidedTrainingConfig:
    epochs: int = 16
    lr: float = 8e-5
    noise_weight: float = 1.0
    physics_weight: float = 0.08
    latent_anchor_weight: float = 0.05
    facies_anchor_weight: float = 0.20
    continuous_anchor_weight: float = 0.05
    grad_clip: float = 1.0
    physics: LatentPhysicsGuidanceConfig = LatentPhysicsGuidanceConfig(
        unfrozen_weight=0.50,
        resistivity_weight=0.10,
        heat_weight=0.0006,
        range_weight=0.02,
    )


def predict_x0_from_noise(
    xt: torch.Tensor,
    predicted_noise: torch.Tensor,
    t: torch.Tensor,
    diffusion: GaussianDiffusion3D,
) -> torch.Tensor:
    vals = diffusion._to(xt.device)
    alpha_bar = vals["alpha_bar"][t].view(-1, 1, 1, 1, 1)
    return (xt - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(torch.clamp(alpha_bar, min=1e-6))


def physics_guided_diffusion_loss(
    latent: torch.Tensor,
    cond: torch.Tensor,
    diffusion: GaussianDiffusion3D,
    autoencoder,
    target_volume: torch.Tensor,
    n_facies: int,
    spacing: tuple[float, float, float],
    cfg: PhysicsGuidedTrainingConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = cfg or PhysicsGuidedTrainingConfig()
    t = torch.randint(0, diffusion.timesteps, (latent.shape[0],), device=latent.device)
    noise = torch.randn_like(latent)
    xt = diffusion.q_sample(latent, t, noise)
    predicted_noise = diffusion.denoiser(xt, t, cond)
    noise_loss = F.mse_loss(predicted_noise, noise)
    x0_hat = predict_x0_from_noise(xt, predicted_noise, t, diffusion)
    decoded_hat = autoencoder.decode(x0_hat)
    physics_total, physics_parts = latent_physics_loss(decoded_hat, n_facies=n_facies, spacing=spacing, cfg=cfg.physics)
    latent_anchor = F.mse_loss(x0_hat, latent)
    facies_anchor = F.mse_loss(decoded_hat[:, :n_facies], target_volume[:, :n_facies])
    continuous_anchor = F.mse_loss(decoded_hat[:, n_facies:], target_volume[:, n_facies:])
    total = (
        cfg.noise_weight * noise_loss
        + cfg.physics_weight * physics_total
        + cfg.latent_anchor_weight * latent_anchor
        + cfg.facies_anchor_weight * facies_anchor
        + cfg.continuous_anchor_weight * continuous_anchor
    )
    parts = {
        "loss": total,
        "noise": noise_loss,
        "physics": physics_total,
        "latent_anchor": latent_anchor,
        "facies_anchor": facies_anchor,
        "continuous_anchor": continuous_anchor,
    }
    for key, value in physics_parts.items():
        parts[f"phys_{key}"] = value
    return total, parts
