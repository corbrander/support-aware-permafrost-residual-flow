from __future__ import annotations

import torch


def empirical_log_resistivity(
    eic: torch.Tensor,
    temperature: torch.Tensor,
    unfrozen_water: torch.Tensor,
    facies_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    log_rho = 5.0 + 2.4 * eic - 2.8 * unfrozen_water - 0.10 * temperature
    if facies_probs is not None:
        ice_prob = facies_probs[..., 6] if facies_probs.shape[-1] > 6 else 0.0
        talik_prob = facies_probs[..., 5] if facies_probs.shape[-1] > 5 else 0.0
        sand_prob = facies_probs[..., 4] if facies_probs.shape[-1] > 4 else 0.0
        log_rho = log_rho + 1.0 * ice_prob - 0.9 * talik_prob + 0.35 * sand_prob
    return log_rho


def ert_consistency_loss(
    pred_log_rho: torch.Tensor,
    obs_log_rho: torch.Tensor,
    sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = pred_log_rho - obs_log_rho
    if sigma is not None:
        residual = residual / torch.clamp(sigma, min=1e-3)
    return torch.mean(torch.abs(residual))

