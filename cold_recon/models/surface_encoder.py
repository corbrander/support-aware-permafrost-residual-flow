from __future__ import annotations

import torch
from torch import nn


class SurfaceFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, surface_values: torch.Tensor) -> torch.Tensor:
        return self.net(surface_values)


def sample_surface_features_nearest(
    surface_tensor: torch.Tensor,
    ix: torch.Tensor,
    iy: torch.Tensor,
) -> torch.Tensor:
    """Return per-query surface features from [C, nx, ny] or [B, C, nx, ny]."""
    if surface_tensor.dim() == 3:
        return surface_tensor[:, ix, iy].transpose(0, 1)
    if surface_tensor.dim() == 4:
        b = surface_tensor.shape[0]
        return torch.stack([surface_tensor[j, :, ix[j], iy[j]].transpose(0, 1) for j in range(b)], dim=0)
    raise ValueError("surface_tensor must have shape [C,nx,ny] or [B,C,nx,ny]")

