from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from cold_recon.training.factorized_volume_codec import (
    N_ICE_STRUCTURE,
    N_LITHOLOGY,
    N_THERMAL_STATE,
)


@dataclass(frozen=True)
class ConstitutiveConsistencyConfig:
    eic_sigma: float = 0.08
    unfrozen_sigma: float = 0.06
    log_resistivity_sigma: float = 0.65
    semantic_sigma: float = 0.15
    eic_weight: float = 1.0
    unfrozen_weight: float = 0.6
    resistivity_weight: float = 0.3
    semantic_weight: float = 0.5


def gaussian_relation_penalty(
    value: torch.Tensor,
    expected: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigma, device=value.device, dtype=value.dtype).clamp_min(1.0e-4)
    return 0.5 * torch.mean(((value - expected) / sigma_tensor) ** 2 + 2.0 * torch.log(sigma_tensor))


def probabilistic_constitutive_loss(
    decoded: torch.Tensor,
    config: ConstitutiveConsistencyConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Uncertainty-aware cryogeological and thermodynamic consistency penalty."""

    cfg = config or ConstitutiveConsistencyConfig()
    offset_s = N_LITHOLOGY
    offset_i = offset_s + N_THERMAL_STATE
    offset_c = offset_i + N_ICE_STRUCTURE
    if decoded.shape[1] < offset_c + 4:
        raise ValueError("decoded tensor does not use the factorized state layout")
    p_l = torch.softmax(decoded[:, :offset_s], dim=1)
    p_s = torch.softmax(decoded[:, offset_s:offset_i], dim=1)
    p_i = torch.softmax(decoded[:, offset_i:offset_c], dim=1)
    eic = decoded[:, offset_c : offset_c + 1]
    temperature = 10.0 * decoded[:, offset_c + 1 : offset_c + 2]
    unfrozen = decoded[:, offset_c + 2 : offset_c + 3]
    log_resistivity = 10.0 * decoded[:, offset_c + 3 : offset_c + 4]

    lithology_eic_capacity = decoded.new_tensor([0.55, 0.70, 0.42, 0.60])[None, :, None, None, None]
    structure_eic_mean = decoded.new_tensor([0.08, 0.32, 0.68])[None, :, None, None, None]
    capacity = torch.sum(p_l * lithology_eic_capacity, dim=1, keepdim=True)
    expected_eic = torch.sum(p_i * structure_eic_mean, dim=1, keepdim=True).clamp_max(capacity)
    eic_relation = gaussian_relation_penalty(eic, expected_eic, float(cfg.eic_sigma))
    eic_bounds = torch.mean(F.relu(-eic) ** 2 + F.relu(eic - capacity) ** 2)

    residual_water = torch.sum(
        p_l * decoded.new_tensor([0.08, 0.05, 0.025, 0.04])[None, :, None, None, None],
        dim=1,
        keepdim=True,
    )
    available_porosity = torch.sum(
        p_l * decoded.new_tensor([0.70, 0.52, 0.35, 0.45])[None, :, None, None, None],
        dim=1,
        keepdim=True,
    )
    freezing = torch.sigmoid((temperature + 0.25) / 0.55)
    expected_unfrozen = residual_water + (available_porosity - residual_water) * freezing
    unfrozen_relation = gaussian_relation_penalty(
        unfrozen, expected_unfrozen, float(cfg.unfrozen_sigma)
    )

    lithology_intercept = torch.sum(
        p_l * decoded.new_tensor([5.0, 5.8, 4.7, 5.2])[None, :, None, None, None],
        dim=1,
        keepdim=True,
    )
    expected_log_resistivity = (
        lithology_intercept
        + 1.35 * eic
        + 0.10 * torch.relu(-temperature)
        - 1.80 * unfrozen
    )
    resistivity_relation = gaussian_relation_penalty(
        log_resistivity, expected_log_resistivity, float(cfg.log_resistivity_sigma)
    )

    thaw_probability = torch.sigmoid(temperature / 0.35)
    near_thaw_probability = torch.exp(-0.5 * (temperature / 0.45) ** 2)
    expected_state = torch.cat(
        [
            thaw_probability,
            (1.0 - thaw_probability) * (1.0 - near_thaw_probability),
            near_thaw_probability,
        ],
        dim=1,
    )
    expected_state = expected_state / expected_state.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    semantic_relation = gaussian_relation_penalty(p_s, expected_state, float(cfg.semantic_sigma))

    total = (
        float(cfg.eic_weight) * (eic_relation + eic_bounds)
        + float(cfg.unfrozen_weight) * unfrozen_relation
        + float(cfg.resistivity_weight) * resistivity_relation
        + float(cfg.semantic_weight) * semantic_relation
    )
    return total, {
        "total": total,
        "eic_relation": eic_relation,
        "eic_bounds": eic_bounds,
        "unfrozen_relation": unfrozen_relation,
        "resistivity_relation": resistivity_relation,
        "thermal_semantic": semantic_relation,
    }
