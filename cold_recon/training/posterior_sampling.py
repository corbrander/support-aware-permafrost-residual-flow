from __future__ import annotations

import torch


def hard_data_guidance_step(latent: torch.Tensor, loss: torch.Tensor, step_size: float = 1e-2) -> torch.Tensor:
    grad = torch.autograd.grad(loss, latent, retain_graph=True, allow_unused=False)[0]
    return latent - step_size * grad

