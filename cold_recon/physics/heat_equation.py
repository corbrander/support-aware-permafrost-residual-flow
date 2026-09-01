from __future__ import annotations

import torch


def steady_heat_residual_3d(temperature: torch.Tensor, conductivity: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
    """Finite-difference residual div(k grad T) for [B, X, Y, Z] tensors."""
    if temperature.dim() == 3:
        temperature = temperature.unsqueeze(0)
    if conductivity.dim() == 3:
        conductivity = conductivity.unsqueeze(0)
    dx, dy, dz = spacing
    tx = torch.gradient(temperature, spacing=(dx,), dim=1)[0]
    ty = torch.gradient(temperature, spacing=(dy,), dim=2)[0]
    tz = torch.gradient(temperature, spacing=(dz,), dim=3)[0]
    qx = conductivity * tx
    qy = conductivity * ty
    qz = conductivity * tz
    div = (
        torch.gradient(qx, spacing=(dx,), dim=1)[0]
        + torch.gradient(qy, spacing=(dy,), dim=2)[0]
        + torch.gradient(qz, spacing=(dz,), dim=3)[0]
    )
    return div


def steady_heat_loss_3d(temperature: torch.Tensor, conductivity: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
    residual = steady_heat_residual_3d(temperature, conductivity, spacing)
    return torch.mean(residual.square())


def vertical_heat_loss_1d(temperature_profile: torch.Tensor, conductivity_profile: torch.Tensor, dz: float) -> torch.Tensor:
    dtdz = torch.gradient(temperature_profile, spacing=(dz,), dim=-1)[0]
    flux = conductivity_profile * dtdz
    residual = torch.gradient(flux, spacing=(dz,), dim=-1)[0]
    return torch.mean(residual.square())

