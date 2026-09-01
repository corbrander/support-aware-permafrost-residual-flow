from __future__ import annotations

import torch


def empirical_unfrozen_water(
    temperature: torch.Tensor,
    facies_probs: torch.Tensor | None = None,
    theta_sat: float = 0.42,
    theta_res: float = 0.06,
    a: float = 0.09,
    b: float = 0.45,
) -> torch.Tensor:
    frozen = theta_res + a / torch.pow(torch.clamp(torch.abs(temperature), min=0.08), b)
    theta = torch.where(temperature >= 0.0, torch.full_like(temperature, theta_sat), frozen)
    if facies_probs is not None:
        ice_prob = facies_probs[..., 6] if facies_probs.shape[-1] > 6 else 0.0
        peat_prob = facies_probs[..., 1] if facies_probs.shape[-1] > 1 else 0.0
        theta = theta * (1.0 - 0.65 * ice_prob) + 0.22 * peat_prob
    return torch.clamp(theta, 0.0, 0.8)


def unfrozen_water_loss(pred_theta: torch.Tensor, temperature: torch.Tensor, facies_probs: torch.Tensor | None = None) -> torch.Tensor:
    target = empirical_unfrozen_water(temperature, facies_probs)
    return torch.mean(torch.abs(pred_theta - target))

