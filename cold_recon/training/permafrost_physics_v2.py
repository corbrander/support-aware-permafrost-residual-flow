from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.nn import functional as F

from cold_recon.physics.heat_equation import steady_heat_residual_3d
from cold_recon.physics.resistivity import empirical_log_resistivity
from cold_recon.physics.unfrozen_water import empirical_unfrozen_water
from cold_recon.training.physics_guided_sampling import (
    decoded_volume_fields,
    estimate_thermal_conductivity_torch,
)


@dataclass(frozen=True)
class PermafrostPhysicsV2Config:
    """Dimensionless frozen-ground constraints and sampling guidance.

    The residual scales are fixed before test evaluation.  They convert unlike
    units to order-one losses so that a temperature curvature term cannot
    numerically swamp an unfrozen-water or resistivity term.
    """

    unfrozen_weight: float = 1.0
    resistivity_weight: float = 0.75
    heat_weight: float = 0.25
    phase_weight: float = 0.35
    range_weight: float = 0.20
    unfrozen_scale: float = 0.08
    resistivity_scale: float = 1.50
    heat_scale: float = 4.0
    phase_temperature_scale: float = 0.50
    eic_range: tuple[float, float] = (0.0, 0.8)
    unfrozen_range: tuple[float, float] = (0.0, 0.8)
    temperature_range: tuple[float, float] = (-10.0, 3.0)
    log_resistivity_range: tuple[float, float] = (0.0, 12.0)
    guidance_steps: int = 8
    guidance_lr: float = 0.006
    guidance_grad_clip: float = 0.35
    latent_anchor_weight: float = 0.20
    facies_anchor_weight: float = 1.50
    continuous_anchor_weight: float = 0.20
    temperature_guidance_steps: int = 0
    temperature_guidance_lr: float = 0.001
    temperature_anchor_weight: float = 2.0


def _rms(value: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.square()) + float(eps))


def _normalized_range_barrier(
    raw: torch.Tensor,
    lower: float,
    upper: float,
    scale: float,
) -> torch.Tensor:
    violation = F.relu(float(lower) - raw) + F.relu(raw - float(upper))
    return _rms(violation) / max(float(scale), 1.0e-6)


def normalized_permafrost_physics_loss(
    decoded: torch.Tensor,
    n_facies: int,
    spacing: tuple[float, float, float],
    cfg: PermafrostPhysicsV2Config | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate differentiable, dimensionless constraints on the full state."""

    cfg = cfg or PermafrostPhysicsV2Config()
    fields = decoded_volume_fields(decoded.float(), n_facies=n_facies)
    facies = fields["facies_probability"]
    eic = fields["eic"]
    temperature = fields["temperature"]
    unfrozen = fields["unfrozen_water"]
    log_rho = fields["log_resistivity"]

    theta_eq = empirical_unfrozen_water(temperature, facies)
    rho_eq = empirical_log_resistivity(eic, temperature, unfrozen, facies)
    conductivity = estimate_thermal_conductivity_torch(eic, facies)
    heat_residual = steady_heat_residual_3d(temperature, conductivity, spacing)

    ice_rich_prob = facies[..., 3]
    wedge_prob = facies[..., 6] if facies.shape[-1] > 6 else torch.zeros_like(eic)
    talik_prob = facies[..., 5] if facies.shape[-1] > 5 else torch.zeros_like(eic)
    frozen_prob = torch.clamp(ice_rich_prob + wedge_prob, 0.0, 1.0)
    phase_violation = talik_prob * F.relu(-temperature - 0.10)
    phase_violation = phase_violation + frozen_prob * F.relu(temperature + 0.20)

    range_terms = torch.stack(
        [
            _normalized_range_barrier(fields["eic_raw"], *cfg.eic_range, scale=0.10),
            _normalized_range_barrier(fields["unfrozen_water_raw"], *cfg.unfrozen_range, scale=0.10),
            _normalized_range_barrier(fields["temperature_raw"], *cfg.temperature_range, scale=2.0),
            _normalized_range_barrier(fields["log_resistivity_raw"], *cfg.log_resistivity_range, scale=2.0),
        ]
    )

    parts = {
        "unfrozen": _rms(unfrozen - theta_eq) / float(cfg.unfrozen_scale),
        "resistivity": _rms(log_rho - rho_eq) / float(cfg.resistivity_scale),
        "heat": _rms(heat_residual) / float(cfg.heat_scale),
        "phase": _rms(phase_violation) / float(cfg.phase_temperature_scale),
        "range": range_terms.mean(),
        "unfrozen_raw_rmse": _rms(unfrozen - theta_eq),
        "resistivity_raw_rmse": _rms(log_rho - rho_eq),
        "heat_raw_rmse": _rms(heat_residual),
    }
    total = (
        float(cfg.unfrozen_weight) * parts["unfrozen"]
        + float(cfg.resistivity_weight) * parts["resistivity"]
        + float(cfg.heat_weight) * parts["heat"]
        + float(cfg.phase_weight) * parts["phase"]
        + float(cfg.range_weight) * parts["range"]
    )
    parts["total_physics"] = total
    return total, parts


class GradientRatioBalancer:
    """Match physics-gradient magnitude to a fixed fraction of data gradients."""

    def __init__(
        self,
        target_ratio: float = 0.35,
        momentum: float = 0.90,
        minimum: float = 0.01,
        maximum: float = 3.0,
    ) -> None:
        self.target_ratio = float(target_ratio)
        self.momentum = float(momentum)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.value: float | None = None

    def update(
        self,
        data_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        reference_parameter: torch.Tensor,
    ) -> tuple[float, float, float]:
        data_grad = torch.autograd.grad(data_loss, reference_parameter, retain_graph=True, allow_unused=True)[0]
        physics_grad = torch.autograd.grad(physics_loss, reference_parameter, retain_graph=True, allow_unused=True)[0]
        data_norm = float(torch.linalg.vector_norm(data_grad.detach()).cpu()) if data_grad is not None else 0.0
        physics_norm = float(torch.linalg.vector_norm(physics_grad.detach()).cpu()) if physics_grad is not None else 0.0
        candidate = self.target_ratio * data_norm / max(physics_norm, 1.0e-12)
        candidate = min(self.maximum, max(self.minimum, candidate))
        self.value = candidate if self.value is None else self.momentum * self.value + (1.0 - self.momentum) * candidate
        return float(self.value), data_norm, physics_norm

    def weight(self, fallback: float = 0.10) -> float:
        return float(fallback if self.value is None else self.value)


@torch.enable_grad()
def guide_posterior_mean_latent(
    state_latents: torch.Tensor,
    decode: Callable[[torch.Tensor], torch.Tensor],
    n_facies: int,
    spacing: tuple[float, float, float],
    cfg: PermafrostPhysicsV2Config | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Guide only the posterior mean and preserve member-to-member anomalies.

    Optimizing one mean latent is memory efficient.  The resulting mean shift is
    added to every ensemble member, so stochastic spread is retained instead of
    being collapsed by per-member projection.
    """

    cfg = cfg or PermafrostPhysicsV2Config()
    original = state_latents.detach()
    mean_start = original.mean(dim=0, keepdim=True)
    with torch.no_grad():
        decoded_start = decode(mean_start).detach()
    guided = mean_start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([guided], lr=float(cfg.guidance_lr))
    history: list[dict[str, float]] = []

    for step in range(int(cfg.guidance_steps)):
        optimizer.zero_grad(set_to_none=True)
        decoded = decode(guided)
        physics, parts = normalized_permafrost_physics_loss(decoded, n_facies, spacing, cfg)
        latent_anchor = F.mse_loss(guided, mean_start)
        facies_anchor = F.mse_loss(decoded[:, :n_facies], decoded_start[:, :n_facies])
        continuous_anchor = F.mse_loss(decoded[:, n_facies:], decoded_start[:, n_facies:])
        objective = (
            physics
            + float(cfg.latent_anchor_weight) * latent_anchor
            + float(cfg.facies_anchor_weight) * facies_anchor
            + float(cfg.continuous_anchor_weight) * continuous_anchor
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_([guided], float(cfg.guidance_grad_clip))
        optimizer.step()
        row = {key: float(value.detach().cpu()) for key, value in parts.items()}
        row.update(
            {
                "step": float(step),
                "objective": float(objective.detach().cpu()),
                "latent_anchor": float(latent_anchor.detach().cpu()),
                "facies_anchor": float(facies_anchor.detach().cpu()),
                "continuous_anchor": float(continuous_anchor.detach().cpu()),
            }
        )
        history.append(row)

    shift = guided.detach() - mean_start
    return original + shift, history


@torch.enable_grad()
def guide_temperature_mean_field(
    decoded_members: torch.Tensor,
    n_facies: int,
    spacing: tuple[float, float, float],
    cfg: PermafrostPhysicsV2Config | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Reduce steady-heat residual in the ensemble mean without changing spread."""

    cfg = cfg or PermafrostPhysicsV2Config()
    original = decoded_members.detach()
    mean_decoded = original.mean(dim=0, keepdim=True)
    fields = decoded_volume_fields(mean_decoded, n_facies=n_facies)
    conductivity = estimate_thermal_conductivity_torch(
        fields["eic"], fields["facies_probability"]
    ).detach()
    temperature_start = fields["temperature"].detach()
    temperature = temperature_start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([temperature], lr=float(cfg.temperature_guidance_lr))
    history: list[dict[str, float]] = []
    best_temperature = temperature_start.clone()
    best_objective = float("inf")

    for step in range(int(cfg.temperature_guidance_steps)):
        optimizer.zero_grad(set_to_none=True)
        heat = _rms(steady_heat_residual_3d(temperature, conductivity, spacing))
        anchor = torch.mean(((temperature - temperature_start) / 1.0).square())
        objective = heat / float(cfg.heat_scale) + float(cfg.temperature_anchor_weight) * anchor
        objective.backward()
        torch.nn.utils.clip_grad_norm_([temperature], float(cfg.guidance_grad_clip))
        optimizer.step()
        with torch.no_grad():
            heat_after = _rms(steady_heat_residual_3d(temperature, conductivity, spacing))
            anchor_after = torch.mean(((temperature - temperature_start) / 1.0).square())
            objective_after = heat_after / float(cfg.heat_scale) + float(cfg.temperature_anchor_weight) * anchor_after
            if float(objective_after.cpu()) < best_objective:
                best_objective = float(objective_after.cpu())
                best_temperature = temperature.detach().clone()
        history.append(
            {
                "step": float(step),
                "heat_raw_rmse": float(heat.detach().cpu()),
                "temperature_anchor": float(anchor.detach().cpu()),
                "objective": float(objective.detach().cpu()),
                "objective_after": float(objective_after.detach().cpu()),
            }
        )

    shift_scaled = (best_temperature - temperature_start) / 10.0
    guided = original.clone()
    guided[:, n_facies + 1] = guided[:, n_facies + 1] + shift_scaled.expand_as(guided[:, n_facies + 1])
    return guided, history
