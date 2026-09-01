from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from cold_recon.models.site_adapter import SiteFiLMAdapter3D
from cold_recon.training.factorized_volume_codec import (
    N_ICE_STRUCTURE,
    N_LITHOLOGY,
    N_THERMAL_STATE,
)


@dataclass
class MaskedBoreholeAdapterCase:
    anchor: torch.Tensor
    encoded_context: torch.Tensor
    support_operator: torch.Tensor
    observed_eic: torch.Tensor
    sigma: torch.Tensor
    held_group_id: int


def _supported_eic(decoded: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    channel = N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE
    field = decoded[:, channel].reshape(decoded.shape[0], -1)
    return torch.sparse.mm(operator, field.T).T


def fit_masked_borehole_site_adapter(
    cases: list[MaskedBoreholeAdapterCase],
    *,
    autoencoder: nn.Module,
    bias_head: nn.Module,
    context_channels: int,
    steps: int = 100,
    learning_rate: float = 5.0e-3,
    weight_decay: float = 1.0e-5,
    seed: int = 42,
) -> tuple[SiteFiLMAdapter3D, list[dict[str, float]]]:
    """Fit a lightweight adapter only on masked complete-borehole targets."""

    if not cases:
        raise ValueError("at least one masked borehole case is required")
    device = cases[0].anchor.device
    torch.manual_seed(int(seed))
    adapter = SiteFiLMAdapter3D(int(context_channels)).to(device)
    for module in (autoencoder, bias_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    optimizer = AdamW(adapter.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    history: list[dict[str, float]] = []
    for step in range(1, int(steps) + 1):
        case = cases[(step - 1) % len(cases)]
        optimizer.zero_grad(set_to_none=True)
        adapted = adapter(case.encoded_context)
        bias, gate, _ = bias_head(adapted)
        decoded = autoencoder.decode(case.anchor + gate * bias)
        predicted = _supported_eic(decoded, case.support_operator)
        residual = (predicted - case.observed_eic[None, :]) / case.sigma[None, :].clamp_min(0.01)
        loss = torch.mean(torch.nn.functional.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            beta=2.0,
        ))
        regularization = sum(parameter.square().mean() for parameter in adapter.parameters())
        total = loss + 1.0e-4 * regularization
        total.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "held_group_id": float(case.held_group_id),
                "loss": float(loss.detach().cpu()),
            }
        )
    adapter.eval()
    return adapter, history


@torch.no_grad()
def adapter_eic_prediction(
    *,
    adapter: SiteFiLMAdapter3D,
    autoencoder: nn.Module,
    bias_head: nn.Module,
    anchor: torch.Tensor,
    encoded_context: torch.Tensor,
    support_operator: torch.Tensor,
) -> np.ndarray:
    adapted = adapter(encoded_context)
    bias, gate, _ = bias_head(adapted)
    decoded = autoencoder.decode(anchor + gate * bias)
    return _supported_eic(decoded, support_operator)[0].float().cpu().numpy()
