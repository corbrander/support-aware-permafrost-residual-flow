from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / max(half - 1, 1)))
        emb = t.float()[:, None] * freq[None, :]
        if self.dim % 2:
            return torch.cat([torch.sin(emb), torch.cos(emb), torch.zeros_like(emb[:, :1])], dim=-1)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class SpectralConv3d(nn.Module):
    """Low-frequency 3D Fourier convolution used by the neural operator denoiser."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int, int] = (8, 8, 6)) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = tuple(int(max(1, m)) for m in modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        mx, my, mz = self.modes
        self.weight_pp = nn.Parameter(scale * torch.randn(in_channels, out_channels, mx, my, mz, dtype=torch.cfloat))
        self.weight_np = nn.Parameter(scale * torch.randn(in_channels, out_channels, mx, my, mz, dtype=torch.cfloat))
        self.weight_pn = nn.Parameter(scale * torch.randn(in_channels, out_channels, mx, my, mz, dtype=torch.cfloat))
        self.weight_nn = nn.Parameter(scale * torch.randn(in_channels, out_channels, mx, my, mz, dtype=torch.cfloat))

    @staticmethod
    def _contract(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcxyz,coxyz->boxyz", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        batch, _, sx, sy, sz = x.shape
        # CUDA FFT does not support bfloat16 for these 3-D sizes. Keep the
        # spectral path in float32/complex64 while surrounding convolutions can
        # still use autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
            out_ft = torch.zeros(
                batch,
                self.out_channels,
                sx,
                sy,
                sz // 2 + 1,
                device=x.device,
                dtype=torch.cfloat,
            )
            mx = min(self.modes[0], max(1, sx // 2))
            my = min(self.modes[1], max(1, sy // 2))
            mz = min(self.modes[2], sz // 2 + 1)
            out_ft[:, :, :mx, :my, :mz] = self._contract(x_ft[:, :, :mx, :my, :mz], self.weight_pp[:, :, :mx, :my, :mz])
            out_ft[:, :, -mx:, :my, :mz] = self._contract(x_ft[:, :, -mx:, :my, :mz], self.weight_np[:, :, :mx, :my, :mz])
            out_ft[:, :, :mx, -my:, :mz] = self._contract(x_ft[:, :, :mx, -my:, :mz], self.weight_pn[:, :, :mx, :my, :mz])
            out_ft[:, :, -mx:, -my:, :mz] = self._contract(x_ft[:, :, -mx:, -my:, :mz], self.weight_nn[:, :, :mx, :my, :mz])
            result = torch.fft.irfftn(out_ft, s=(sx, sy, sz), dim=(-3, -2, -1), norm="ortho")
        return result.to(dtype=output_dtype)


class FNOBlock3d(nn.Module):
    def __init__(self, width: int, modes: tuple[int, int, int], cond_dim: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes=modes)
        self.local = nn.Conv3d(width, width, 1)
        self.film = nn.Linear(cond_dim, width * 2)
        self.norm = nn.GroupNorm(num_groups=min(8, width), num_channels=width)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        y = self.spectral(x) + self.local(x)
        y = self.norm(y)
        y = y * (1.0 + 0.1 * scale[:, :, None, None, None]) + 0.1 * shift[:, :, None, None, None]
        return F.gelu(y)


class FNOTransformerHybrid(nn.Module):
    """Conditional Fourier neural-operator denoiser for latent 3D diffusion.

    The module keeps the same call signature as the existing U-Net denoiser:
    ``forward(x, t, cond) -> predicted_noise``. Global sparse-observation context
    and diffusion time are injected into FNO blocks with FiLM conditioning, while
    a compact Transformer over pooled latent tokens adds nonlocal cross-cell
    communication beyond the retained Fourier modes.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int = 96,
        width: int = 48,
        modes: tuple[int, int, int] = (8, 8, 6),
        depth: int = 4,
        time_dim: int = 64,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        token_grid: tuple[int, int, int] = (4, 4, 3),
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.width = int(width)
        self.token_grid = tuple(int(max(1, v)) for v in token_grid)
        self.time = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, width), nn.GELU())
        self.cond = nn.Sequential(nn.Linear(cond_dim, width), nn.GELU())
        self.in_proj = nn.Conv3d(channels, width, 1)
        self.blocks = nn.ModuleList([FNOBlock3d(width, modes=modes, cond_dim=width) for _ in range(int(depth))])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=int(transformer_heads),
            dim_feedforward=width * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.token_transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))
        self.token_norm = nn.LayerNorm(width)
        self.out = nn.Sequential(nn.Conv3d(width, width, 1), nn.GELU(), nn.Conv3d(width, channels, 1))

    def _transformer_residual(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(x, self.token_grid)
        batch, width, gx, gy, gz = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)
        cond_token = cond_emb[:, None, :]
        encoded = self.token_transformer(torch.cat([cond_token, tokens], dim=1))[:, 1:]
        encoded = self.token_norm(encoded)
        encoded_grid = encoded.transpose(1, 2).reshape(batch, width, gx, gy, gz)
        return F.interpolate(encoded_grid, size=x.shape[-3:], mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_emb = self.time(t) + self.cond(cond)
        h = self.in_proj(x)
        for idx, block in enumerate(self.blocks):
            h = h + block(h, cond_emb)
            if idx == len(self.blocks) // 2:
                h = h + self._transformer_residual(h, cond_emb)
        return self.out(h)
