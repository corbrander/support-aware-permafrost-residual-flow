from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / max(half - 1, 1)))
        emb = t.float()[:, None] * freq[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Denoiser3DUNet(nn.Module):
    def __init__(self, channels: int, cond_dim: int = 96, base: int = 32, time_dim: int = 64) -> None:
        super().__init__()
        self.time = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, base), nn.GELU())
        self.cond = nn.Sequential(nn.Linear(cond_dim, base), nn.GELU())
        self.in_conv = nn.Conv3d(channels, base, 3, padding=1)
        self.down = nn.Conv3d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(nn.GELU(), nn.Conv3d(base * 2, base * 2, 3, padding=1), nn.GELU())
        self.up = nn.ConvTranspose3d(base * 2, base, 4, stride=2, padding=1)
        self.out = nn.Conv3d(base * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = (self.time(t) + self.cond(cond))[:, :, None, None, None]
        e = self.in_conv(x) + emb
        m = self.mid(self.down(e))
        u = self.up(m)[..., : e.shape[-3], : e.shape[-2], : e.shape[-1]]
        return self.out(torch.cat([u, e], dim=1))

