from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 3, num_features: int = 48, sigma: float = 6.0) -> None:
        super().__init__()
        b = torch.randn(in_dim, num_features) * sigma
        self.register_buffer("B", b)

    @property
    def out_dim(self) -> int:
        return int(self.B.shape[1] * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ImplicitCryoField(nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        surface_dim: int = 32,
        context_dim: int = 96,
        fourier_features: int = 48,
        hidden_dim: int = 192,
        num_layers: int = 5,
        n_facies: int = 7,
    ) -> None:
        super().__init__()
        self.fourier = FourierFeatures(coord_dim, fourier_features)
        in_dim = coord_dim + self.fourier.out_dim + surface_dim + context_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        self.trunk = nn.Sequential(*layers)
        self.facies_head = nn.Linear(hidden_dim, n_facies)
        self.eic_head = nn.Linear(hidden_dim, 1)
        self.temperature_head = nn.Linear(hidden_dim, 1)
        self.unfrozen_head = nn.Linear(hidden_dim, 1)
        self.log_resistivity_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        coords: torch.Tensor,
        surface_embedding: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if context.dim() == 2 and coords.dim() == 3:
            context = context[:, None, :].expand(-1, coords.shape[1], -1)
        ff = self.fourier(coords)
        h = torch.cat([coords, ff, surface_embedding, context], dim=-1)
        h = self.trunk(h)
        eic = torch.sigmoid(self.eic_head(h)).squeeze(-1) * 0.75
        temp = self.temperature_head(h).squeeze(-1)
        theta = torch.sigmoid(self.unfrozen_head(h)).squeeze(-1) * 0.8
        log_rho = self.log_resistivity_head(h).squeeze(-1) + 5.0
        return {
            "facies_logits": self.facies_head(h),
            "eic": eic,
            "temperature": temp,
            "unfrozen_water": theta,
            "log_resistivity": log_rho,
            "resistivity": torch.exp(torch.clamp(log_rho, 0.0, 12.0)),
        }


def prediction_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    losses: dict[str, torch.Tensor] = {}
    target_facies = target["facies"].reshape(-1).long()
    class_weight = None
    if float(weights.get("facies_balanced", 0.0)) > 0:
        n_classes = pred["facies_logits"].shape[-1]
        counts = torch.bincount(target_facies, minlength=n_classes).float()
        class_weight = target_facies.numel() / (n_classes * counts.clamp_min(1.0))
        class_weight = class_weight / class_weight[class_weight > 0].mean().clamp_min(1e-6)
        class_weight = torch.clamp(class_weight, max=8.0)
    losses["facies"] = F.cross_entropy(
        pred["facies_logits"].reshape(-1, pred["facies_logits"].shape[-1]),
        target_facies,
        weight=class_weight,
    )
    eic_weight = 1.0 + float(weights.get("eic_rich", 0.0)) * (
        target["eic"] > float(weights.get("eic_rich_threshold", 0.25))
    ).float()
    losses["eic"] = torch.mean(eic_weight * (pred["eic"] - target["eic"]).square())
    losses["temperature"] = F.mse_loss(pred["temperature"], target["temperature"])
    if "unfrozen_water" in target:
        losses["unfrozen_water"] = F.mse_loss(pred["unfrozen_water"], target["unfrozen_water"])
    if "log_resistivity" in target:
        losses["resistivity"] = F.mse_loss(pred["log_resistivity"], target["log_resistivity"])
    total = sum(float(weights.get(k, 1.0)) * v for k, v in losses.items())
    return total, losses
