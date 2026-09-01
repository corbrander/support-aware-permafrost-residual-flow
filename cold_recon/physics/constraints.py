from __future__ import annotations

import torch
from torch.nn import functional as F

from cold_recon.physics.resistivity import empirical_log_resistivity
from cold_recon.physics.unfrozen_water import unfrozen_water_loss


def stratigraphic_tv_loss(facies_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(facies_logits, dim=-1)
    if probs.dim() < 4:
        return torch.tensor(0.0, device=probs.device)
    dx = torch.mean(torch.abs(probs[:, 1:, :, :, :] - probs[:, :-1, :, :, :]))
    dy = torch.mean(torch.abs(probs[:, :, 1:, :, :] - probs[:, :, :-1, :, :]))
    dz = torch.mean(torch.abs(probs[:, :, :, 1:, :] - probs[:, :, :, :-1, :]))
    return dx + dy + 0.5 * dz


def point_observation_loss(
    pred: dict[str, torch.Tensor],
    target_facies: torch.Tensor | None = None,
    target_eic: torch.Tensor | None = None,
    target_temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    if target_facies is not None:
        losses.append(F.cross_entropy(pred["facies_logits"].reshape(-1, pred["facies_logits"].shape[-1]), target_facies.reshape(-1)))
    if target_eic is not None:
        losses.append(F.l1_loss(pred["eic"], target_eic))
    if target_temperature is not None:
        losses.append(F.l1_loss(pred["temperature"], target_temperature))
    return sum(losses) if losses else torch.tensor(0.0, device=pred["eic"].device)


def physics_regularization(pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    facies_probs = torch.softmax(pred["facies_logits"], dim=-1)
    uw = unfrozen_water_loss(pred["unfrozen_water"], pred["temperature"], facies_probs)
    log_rho_emp = empirical_log_resistivity(pred["eic"], pred["temperature"], pred["unfrozen_water"], facies_probs)
    rho = torch.mean(torch.abs(pred["log_resistivity"] - log_rho_emp))
    return {"unfrozen_water": uw, "resistivity": rho}

