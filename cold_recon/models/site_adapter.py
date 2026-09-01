from __future__ import annotations

import torch
from torch import nn


class SiteFiLMAdapter3D(nn.Module):
    """Lightweight site-specific affine adapter for a frozen context tensor."""

    def __init__(self, channels: int, bottleneck: int = 16) -> None:
        super().__init__()
        self.summary = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(int(channels), int(bottleneck)),
            nn.SiLU(),
            nn.Linear(int(bottleneck), 2 * int(channels)),
        )
        nn.init.zeros_(self.summary[-1].weight)
        nn.init.zeros_(self.summary[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.summary(context).chunk(2, dim=1)
        return context * (1.0 + 0.05 * scale[:, :, None, None, None]) + 0.05 * shift[:, :, None, None, None]


def freeze_except_adapter(model: nn.Module, adapter: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)


def masked_observation_self_supervision_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    sigma: torch.Tensor,
    held_mask: torch.Tensor,
    min_sigma: float = 1.0e-3,
) -> torch.Tensor:
    normalized = (predicted - observed) / sigma.clamp_min(float(min_sigma))
    weights = held_mask.to(normalized.dtype)
    return torch.sum(weights * normalized.square()) / weights.sum().clamp_min(1.0)
