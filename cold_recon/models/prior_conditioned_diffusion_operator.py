from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from cold_recon.models.fno_transformer import FNOBlock3d, SinusoidalTimeEmbedding, SpectralConv3d


class GatedLocalSpectralBlock3d(nn.Module):
    """Fuse non-local Fourier and local thin-structure features with a learned gate.

    The gate varies by channel and voxel and is shifted by the global
    anchor/observation context.  This keeps the global receptive field of the
    Fourier branch while allowing narrow cryostratigraphic contacts to rely on
    a depthwise local branch when appropriate.
    """

    def __init__(self, width: int, modes: tuple[int, int, int], cond_dim: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes=modes)
        self.local = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1, groups=width),
            nn.Conv3d(width, width, 1),
        )
        self.gate = nn.Conv3d(width, width, 1)
        self.gate_context = nn.Linear(cond_dim, width)
        self.film = nn.Linear(cond_dim, width * 2)
        self.norm = nn.GroupNorm(num_groups=min(8, width), num_channels=width)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        spectral = self.spectral(x)
        local = self.local(x)
        gate = torch.sigmoid(
            self.gate(x) + self.gate_context(cond)[:, :, None, None, None]
        )
        y = gate * spectral + (1.0 - gate) * local
        scale, shift = self.film(cond).chunk(2, dim=-1)
        y = self.norm(y)
        y = y * (1.0 + 0.1 * scale[:, :, None, None, None])
        y = y + 0.1 * shift[:, :, None, None, None]
        return F.gelu(y)


class TypedContextEncoder3D(nn.Module):
    """Encode tree priors, typed observations and spatial covariates on the latent grid."""

    def __init__(self, in_channels: int, context_channels: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(int(in_channels), 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, int(context_channels), 3, stride=2, padding=1),
            nn.GroupNorm(6 if int(context_channels) % 6 == 0 else 4, int(context_channels)),
            nn.SiLU(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class PriorConditionedDiffusionOperator3D(nn.Module):
    """Fourier neural-operator denoiser for a prior-conditioned residual diffusion.

    ``x`` is the noised latent correction, ``anchor`` is the latent conditional
    mean produced from sparse observations, and ``context`` contains the tree
    prior, typed observation rasters, coordinates and surface covariates.  The
    retained Fourier modes make the denoiser an explicit 3-D neural operator;
    local 1x1 paths and FiLM conditioning retain small-scale and global context.
    """

    def __init__(
        self,
        latent_channels: int,
        context_in_channels: int,
        context_channels: int = 24,
        width: int = 32,
        modes: tuple[int, int, int] = (6, 6, 4),
        depth: int = 4,
        time_dim: int = 64,
        dropout: float = 0.05,
        gated_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.width = int(width)
        self.context_encoder = TypedContextEncoder3D(context_in_channels, context_channels=context_channels)
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
        self.gated_fusion = bool(gated_fusion)
        self.dropout = nn.Dropout3d(float(dropout))
        self.out = nn.Sequential(
            nn.Conv3d(int(width), int(width), 1),
            nn.SiLU(),
            nn.Conv3d(int(width), int(latent_channels), 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        anchor: torch.Tensor,
        context: torch.Tensor,
        context_is_encoded: bool = False,
    ) -> torch.Tensor:
        context_latent = context if context_is_encoded else self.context_encoder(context)
        if context_latent.shape[-3:] != x.shape[-3:]:
            context_latent = F.interpolate(context_latent, size=x.shape[-3:], mode="trilinear", align_corners=False)
        pooled = torch.cat(
            [anchor.mean(dim=(-3, -2, -1)), context_latent.mean(dim=(-3, -2, -1))], dim=1
        )
        cond = self.time(t) + self.global_context(pooled)
        h = self.in_proj(torch.cat([x, anchor, context_latent], dim=1))
        for block in self.blocks:
            h = h + self.dropout(block(h, cond))
        return self.out(h)

    def encode_context(self, context: torch.Tensor, target_shape: tuple[int, int, int] | None = None) -> torch.Tensor:
        encoded = self.context_encoder(context)
        if target_shape is not None and encoded.shape[-3:] != tuple(target_shape):
            encoded = F.interpolate(encoded, size=tuple(target_shape), mode="trilinear", align_corners=False)
        return encoded


def diffusion_schedule(
    timesteps: int,
    device: torch.device,
    beta_start: float = 1.0e-4,
    beta_end: float = 2.0e-2,
) -> dict[str, torch.Tensor]:
    betas = torch.linspace(float(beta_start), float(beta_end), int(timesteps), device=device)
    alphas = 1.0 - betas
    return {"betas": betas, "alphas": alphas, "alpha_bar": torch.cumprod(alphas, dim=0)}


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    a = alpha_bar[t].view(-1, 1, 1, 1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


def predict_x0(xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    a = alpha_bar[t].view(-1, 1, 1, 1, 1)
    return (xt - torch.sqrt(1.0 - a) * eps) / torch.sqrt(torch.clamp(a, min=1.0e-6))


@torch.no_grad()
def heun_sample_corrections(
    model: PriorConditionedDiffusionOperator3D,
    anchor: torch.Tensor,
    context: torch.Tensor,
    n_samples: int,
    sampling_steps: int = 10,
    time_scale: float = 79.0,
    seed: int = 42,
) -> torch.Tensor:
    """Integrate a conditional rectified-flow velocity field with Heun steps.

    Ten Heun steps use 20 network evaluations, matching the network-evaluation
    budget of the 20-step DDIM reference used in the controlled benchmark.
    """

    device = anchor.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    shape = (int(n_samples), *anchor.shape[1:])
    state = torch.randn(shape, device=device, generator=generator)
    anchor_b = anchor.expand(int(n_samples), -1, -1, -1, -1)
    context_latent = model.encode_context(context, target_shape=tuple(state.shape[-3:]))
    context_b = context_latent.expand(int(n_samples), -1, -1, -1, -1)
    model.eval()
    step_size = 1.0 / float(sampling_steps)
    for step in range(int(sampling_steps)):
        tau_0 = float(step) * step_size
        tau_1 = float(step + 1) * step_size
        t_0 = torch.full((int(n_samples),), tau_0 * float(time_scale), device=device)
        velocity_0 = model(state, t_0, anchor_b, context_b, context_is_encoded=True)
        predictor = state + step_size * velocity_0
        t_1 = torch.full((int(n_samples),), tau_1 * float(time_scale), device=device)
        velocity_1 = model(predictor, t_1, anchor_b, context_b, context_is_encoded=True)
        state = state + 0.5 * step_size * (velocity_0 + velocity_1)
    return state.clamp(-4.0, 4.0)


@torch.no_grad()
def ddim_sample_corrections(
    model: PriorConditionedDiffusionOperator3D,
    anchor: torch.Tensor,
    context: torch.Tensor,
    n_samples: int,
    timesteps: int = 80,
    sampling_steps: int = 20,
    eta: float = 0.20,
    seed: int = 42,
) -> torch.Tensor:
    """Sample latent corrections using a strided DDIM update."""

    device = anchor.device
    schedule = diffusion_schedule(timesteps, device=device)
    alpha_bar = schedule["alpha_bar"]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    shape = (int(n_samples), *anchor.shape[1:])
    x = torch.randn(shape, device=device, generator=generator)
    anchor_b = anchor.expand(int(n_samples), -1, -1, -1, -1)
    context_latent = model.encode_context(context, target_shape=tuple(x.shape[-3:]))
    context_b = context_latent.expand(int(n_samples), -1, -1, -1, -1)
    indices = torch.linspace(int(timesteps) - 1, 0, int(sampling_steps), device=device).round().long().unique_consecutive()
    model.eval()
    for step, current in enumerate(indices):
        t = torch.full((int(n_samples),), int(current.item()), device=device, dtype=torch.long)
        eps = model(x, t, anchor_b, context_b, context_is_encoded=True)
        x0 = predict_x0(x, eps, t, alpha_bar).clamp(-4.0, 4.0)
        if step == len(indices) - 1:
            x = x0
            break
        previous = int(indices[step + 1].item())
        a_t = alpha_bar[int(current.item())]
        a_prev = alpha_bar[previous]
        sigma = float(eta) * torch.sqrt(
            torch.clamp((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev), min=0.0)
        )
        direction = torch.sqrt(torch.clamp(1.0 - a_prev - sigma**2, min=0.0)) * eps
        noise = torch.randn(x.shape, device=device, generator=generator)
        x = torch.sqrt(a_prev) * x0 + direction + sigma * noise
    return x
