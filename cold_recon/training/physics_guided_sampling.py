from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from cold_recon.physics.heat_equation import steady_heat_loss_3d
from cold_recon.physics.resistivity import empirical_log_resistivity
from cold_recon.physics.unfrozen_water import empirical_unfrozen_water


@dataclass(frozen=True)
class LatentPhysicsGuidanceConfig:
    steps: int = 6
    learning_rate: float = 0.015
    unfrozen_weight: float = 0.50
    resistivity_weight: float = 0.10
    heat_weight: float = 0.0006
    range_weight: float = 0.02
    anchor_weight: float = 0.10
    facies_anchor_weight: float = 1.50
    eic_anchor_weight: float = 0.05
    grad_clip: float = 1.0
    temperature_min: float = -10.0
    temperature_max: float = 3.0
    log_resistivity_min: float = 0.0
    log_resistivity_max: float = 12.0


def decoded_volume_fields(decoded: torch.Tensor, n_facies: int = 7) -> dict[str, torch.Tensor]:
    if decoded.dim() != 5:
        raise ValueError("decoded volume must have shape [batch, channels, x, y, z]")
    facies_logits = decoded[:, :n_facies].permute(0, 2, 3, 4, 1)
    facies_probs = torch.softmax(facies_logits, dim=-1)
    eic_raw = decoded[:, n_facies]
    temp_raw = decoded[:, n_facies + 1] * 10.0
    unfrozen_raw = decoded[:, n_facies + 2]
    log_rho_raw = decoded[:, n_facies + 3] * 10.0
    return {
        "facies_logits": facies_logits,
        "facies_probability": facies_probs,
        "eic_raw": eic_raw,
        "temperature_raw": temp_raw,
        "unfrozen_water_raw": unfrozen_raw,
        "log_resistivity_raw": log_rho_raw,
        "eic": torch.clamp(eic_raw, 0.0, 1.0),
        "temperature": torch.clamp(temp_raw, -20.0, 20.0),
        "unfrozen_water": torch.clamp(unfrozen_raw, 0.0, 1.0),
        "log_resistivity": log_rho_raw,
    }


def estimate_thermal_conductivity_torch(eic: torch.Tensor, facies_probs: torch.Tensor) -> torch.Tensor:
    conductivity = 1.15 + 1.65 * eic
    if facies_probs.shape[-1] > 1:
        conductivity = conductivity - 0.35 * facies_probs[..., 1]
    if facies_probs.shape[-1] > 4:
        conductivity = conductivity + 0.45 * facies_probs[..., 4]
    if facies_probs.shape[-1] > 6:
        conductivity = conductivity + 0.70 * facies_probs[..., 6]
    return torch.clamp(conductivity, 0.35, 4.5)


def range_barrier_loss(fields: dict[str, torch.Tensor], cfg: LatentPhysicsGuidanceConfig) -> torch.Tensor:
    eic_raw = fields["eic_raw"]
    temp_raw = fields["temperature_raw"]
    uw_raw = fields["unfrozen_water_raw"]
    rho_raw = fields["log_resistivity_raw"]
    losses = [
        F.relu(-eic_raw).square().mean(),
        F.relu(eic_raw - 0.8).square().mean(),
        F.relu(-uw_raw).square().mean(),
        F.relu(uw_raw - 0.8).square().mean(),
        F.relu(cfg.temperature_min - temp_raw).square().mean() / 100.0,
        F.relu(temp_raw - cfg.temperature_max).square().mean() / 100.0,
        F.relu(cfg.log_resistivity_min - rho_raw).square().mean() / 100.0,
        F.relu(rho_raw - cfg.log_resistivity_max).square().mean() / 100.0,
    ]
    return torch.stack(losses).sum()


def latent_physics_loss(
    decoded: torch.Tensor,
    n_facies: int,
    spacing: tuple[float, float, float],
    cfg: LatentPhysicsGuidanceConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = cfg or LatentPhysicsGuidanceConfig()
    fields = decoded_volume_fields(decoded, n_facies=n_facies)
    theta_empirical = empirical_unfrozen_water(fields["temperature"], fields["facies_probability"])
    rho_empirical = empirical_log_resistivity(
        fields["eic"],
        fields["temperature"],
        fields["unfrozen_water"],
        fields["facies_probability"],
    )
    conductivity = estimate_thermal_conductivity_torch(fields["eic"], fields["facies_probability"])
    losses = {
        "unfrozen": torch.mean(torch.abs(fields["unfrozen_water"] - theta_empirical)),
        "resistivity": torch.mean(torch.abs(fields["log_resistivity"] - rho_empirical)),
        "heat": steady_heat_loss_3d(fields["temperature"], conductivity, spacing),
        "range": range_barrier_loss(fields, cfg),
    }
    total = (
        cfg.unfrozen_weight * losses["unfrozen"]
        + cfg.resistivity_weight * losses["resistivity"]
        + cfg.heat_weight * losses["heat"]
        + cfg.range_weight * losses["range"]
    )
    losses["total_physics"] = total
    return total, losses


def guide_latents(
    latents: torch.Tensor,
    decoder,
    n_facies: int,
    spacing: tuple[float, float, float],
    cfg: LatentPhysicsGuidanceConfig | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    cfg = cfg or LatentPhysicsGuidanceConfig()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    anchor = latents.detach()
    with torch.no_grad():
        decoded_anchor = decoder.decode(anchor) if hasattr(decoder, "decode") else decoder(anchor)
    guided = anchor.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([guided], lr=float(cfg.learning_rate))
    history: list[dict[str, float]] = []
    for step in range(int(cfg.steps)):
        opt.zero_grad(set_to_none=True)
        decoded = decoder.decode(guided) if hasattr(decoder, "decode") else decoder(guided)
        physics_total, parts = latent_physics_loss(decoded, n_facies=n_facies, spacing=spacing, cfg=cfg)
        anchor_loss = F.mse_loss(guided, anchor)
        facies_anchor_loss = F.mse_loss(decoded[:, :n_facies], decoded_anchor[:, :n_facies])
        eic_anchor_loss = F.mse_loss(decoded[:, n_facies], decoded_anchor[:, n_facies])
        loss = (
            physics_total
            + cfg.anchor_weight * anchor_loss
            + cfg.facies_anchor_weight * facies_anchor_loss
            + cfg.eic_anchor_weight * eic_anchor_loss
        )
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([guided], float(cfg.grad_clip))
        opt.step()
        row = {key: float(value.detach().cpu()) for key, value in parts.items()}
        row["anchor"] = float(anchor_loss.detach().cpu())
        row["facies_anchor"] = float(facies_anchor_loss.detach().cpu())
        row["eic_anchor"] = float(eic_anchor_loss.detach().cpu())
        row["objective"] = float(loss.detach().cpu())
        row["step"] = float(step)
        history.append(row)
    return guided.detach(), history
