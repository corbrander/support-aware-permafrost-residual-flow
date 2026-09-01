from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from cold_recon.models.fno_transformer import FNOBlock3d, SinusoidalTimeEmbedding
from cold_recon.models.prior_conditioned_diffusion_operator import GatedLocalSpectralBlock3d


class ChunkedSupportCrossAttention3D(nn.Module):
    """Cross-attend latent-grid queries to irregular support-aware tokens."""

    def __init__(
        self,
        token_dim: int,
        raster_channels: int,
        hidden_dim: int = 48,
        chunk_size: int = 256,
        support_extent_offset: int = 21,
        distance_scale: float = 0.18,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.chunk_size = int(chunk_size)
        self.support_extent_offset = int(support_extent_offset)
        self.distance_scale = float(distance_scale)
        self.token_encoder = nn.Sequential(
            nn.Linear(int(token_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.query = nn.Conv3d(int(raster_channels), int(hidden_dim), 1)
        self.key = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.value = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.output = nn.Conv3d(int(hidden_dim), int(raster_channels), 1)

    @staticmethod
    def _grid_coords(shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        axes = [torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype) for n in shape]
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return coords.reshape(-1, 3)

    def forward(
        self,
        raster: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, _, nx, ny, nz = raster.shape
        if tokens.ndim != 3 or tokens.shape[0] != batch:
            raise ValueError("tokens must have shape [batch, observations, token_dim]")
        encoded = self.token_encoder(tokens)
        keys = self.key(encoded)
        values = self.value(encoded)
        queries = self.query(raster).flatten(2).transpose(1, 2)
        token_coords = tokens[..., :3].clamp(0.0, 1.0)
        extent_start = self.support_extent_offset
        if tokens.shape[-1] >= extent_start + 3:
            token_extent = tokens[..., extent_start : extent_start + 3].clamp_min(0.0)
            support_radius = token_extent.mean(dim=-1) + float(self.distance_scale)
        else:
            support_radius = tokens.new_full(tokens.shape[:2], float(self.distance_scale))
        grid_coords = self._grid_coords((nx, ny, nz), raster.device, raster.dtype)
        outputs: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        for start in range(0, queries.shape[1], self.chunk_size):
            stop = min(start + self.chunk_size, queries.shape[1])
            query = queries[:, start:stop]
            scores = torch.einsum("bqh,bnh->bqn", query, keys) * scale
            delta = grid_coords[start:stop][None, :, None, :] - token_coords[:, None, :, :]
            distance2 = torch.sum(delta * delta, dim=-1)
            scores = scores - 0.5 * distance2 / support_radius[:, None, :].square().clamp_min(1.0e-4)
            if padding_mask is not None:
                scores = scores.masked_fill(padding_mask[:, None, :], -torch.inf)
            attention = torch.softmax(scores, dim=-1)
            attention = torch.nan_to_num(attention, nan=0.0)
            outputs.append(torch.einsum("bqn,bnh->bqh", attention, values))
        attended = torch.cat(outputs, dim=1).transpose(1, 2).reshape(batch, self.hidden_dim, nx, ny, nz)
        return self.output(attended)


class SupportAwareContextEncoder3D(nn.Module):
    def __init__(
        self,
        raster_in_channels: int,
        token_dim: int,
        context_channels: int = 32,
        attention_hidden: int = 48,
        attention_chunk: int = 256,
        support_extent_offset: int = 21,
        use_token_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.use_token_conditioning = bool(use_token_conditioning)
        self.raster = nn.Sequential(
            nn.Conv3d(int(raster_in_channels), 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, int(context_channels), 3, stride=2, padding=1),
            nn.GroupNorm(8 if int(context_channels) % 8 == 0 else 4, int(context_channels)),
            nn.SiLU(),
        )
        self.cross_attention = ChunkedSupportCrossAttention3D(
            token_dim=int(token_dim),
            raster_channels=int(context_channels),
            hidden_dim=int(attention_hidden),
            chunk_size=int(attention_chunk),
            support_extent_offset=int(support_extent_offset),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * int(context_channels), int(context_channels), 1),
            nn.GroupNorm(8 if int(context_channels) % 8 == 0 else 4, int(context_channels)),
            nn.SiLU(),
        )

    def forward(
        self,
        raster: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        raster_latent = self.raster(raster)
        if target_shape is not None and raster_latent.shape[-3:] != tuple(target_shape):
            raster_latent = F.interpolate(raster_latent, size=target_shape, mode="trilinear", align_corners=False)
        token_latent = (
            self.cross_attention(raster_latent, tokens, padding_mask)
            if self.use_token_conditioning
            else torch.zeros_like(raster_latent)
        )
        return self.fuse(torch.cat([raster_latent, token_latent], dim=1))


class SupportAwareResidualFlow3D(nn.Module):
    """Noise-conditioned support-aware latent residual-flow velocity model."""

    def __init__(
        self,
        *,
        latent_channels: int,
        raster_in_channels: int,
        token_dim: int,
        context_channels: int = 32,
        width: int = 40,
        modes: tuple[int, int, int] = (6, 6, 4),
        depth: int = 4,
        time_dim: int = 64,
        gated_fusion: bool = True,
        attention_hidden: int = 48,
        attention_chunk: int = 256,
        support_extent_offset: int = 21,
        use_token_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.context_encoder = SupportAwareContextEncoder3D(
            raster_in_channels,
            token_dim,
            context_channels=context_channels,
            attention_hidden=attention_hidden,
            attention_chunk=attention_chunk,
            support_extent_offset=support_extent_offset,
            use_token_conditioning=use_token_conditioning,
        )
        self.in_proj = nn.Conv3d(2 * int(latent_channels) + int(context_channels), int(width), 1)
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(int(time_dim)),
            nn.Linear(int(time_dim), int(width)),
            nn.SiLU(),
        )
        self.global_context = nn.Sequential(
            nn.Linear(int(latent_channels) + int(context_channels), int(width)),
            nn.SiLU(),
            nn.Linear(int(width), int(width)),
        )
        block_type = GatedLocalSpectralBlock3d if bool(gated_fusion) else FNOBlock3d
        self.blocks = nn.ModuleList(
            [block_type(int(width), modes=modes, cond_dim=int(width)) for _ in range(int(depth))]
        )
        self.out = nn.Sequential(
            nn.Conv3d(int(width), int(width), 1),
            nn.SiLU(),
            nn.Conv3d(int(width), int(latent_channels), 1),
        )

    def encode_context(
        self,
        raster: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        target_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        return self.context_encoder(raster, tokens, padding_mask, target_shape)

    def velocity_from_encoded(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        anchor: torch.Tensor,
        encoded_context: torch.Tensor,
    ) -> torch.Tensor:
        if encoded_context.shape[-3:] != state.shape[-3:]:
            encoded_context = F.interpolate(
                encoded_context, size=state.shape[-3:], mode="trilinear", align_corners=False
            )
        pooled = torch.cat(
            [anchor.mean(dim=(-3, -2, -1)), encoded_context.mean(dim=(-3, -2, -1))], dim=1
        )
        condition = self.time(time) + self.global_context(pooled)
        hidden = self.in_proj(torch.cat([state, anchor, encoded_context], dim=1))
        for block in self.blocks:
            hidden = hidden + block(hidden, condition)
        return self.out(hidden)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        anchor: torch.Tensor,
        raster: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encode_context(raster, tokens, padding_mask, target_shape=state.shape[-3:])
        return self.velocity_from_encoded(state, time, anchor, encoded)
