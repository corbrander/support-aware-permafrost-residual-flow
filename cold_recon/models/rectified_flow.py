from __future__ import annotations

import torch
from torch.nn import functional as F


class RectifiedFlow3D:
    def __init__(self, velocity_model, time_scale: int = 1000) -> None:
        self.velocity_model = velocity_model
        self.time_scale = int(time_scale)

    def training_loss(self, x0: torch.Tensor, x1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = torch.rand((x0.shape[0],), device=x0.device)
        xt = (1.0 - t.view(-1, 1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1, 1) * x1
        target_v = x1 - x0
        pred_v = self.velocity_model(xt, (t * self.time_scale).long(), cond)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, cond: torch.Tensor, steps: int = 16) -> torch.Tensor:
        if x0.shape[0] != cond.shape[0]:
            raise ValueError("x0 batch size must match cond batch size")
        steps = int(max(1, steps))
        x = x0
        dt = 1.0 / float(steps)
        for idx in range(steps):
            t_float = torch.full((x.shape[0],), (idx + 0.5) / steps, device=x.device)
            t = (t_float * self.time_scale).long()
            x = x + dt * self.velocity_model(x, t, cond)
        return x
