from __future__ import annotations

import torch
from torch.nn import functional as F


class GaussianDiffusion3D:
    def __init__(self, denoiser, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        self.denoiser = denoiser
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.registered = {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}

    def _to(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in self.registered.items()}

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        vals = self._to(x0.device)
        a = vals["alpha_bar"][t].view(-1, 1, 1, 1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise

    def training_loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_noise = self.denoiser(xt, t, cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...], cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        vals = self._to(device)
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            beta = vals["betas"][i]
            alpha = vals["alphas"][i]
            alpha_bar = vals["alpha_bar"][i]
            eps = self.denoiser(x, t, cond)
            x = (x - beta / torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha)
            if i > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x

