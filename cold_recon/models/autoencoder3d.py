from __future__ import annotations

import torch
from torch import nn


class Autoencoder3D(nn.Module):
    def __init__(self, in_channels: int = 11, latent_channels: int = 16, base: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(base, base * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(base * 2, latent_channels, 4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, base * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(base * 2, base, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(base, in_channels, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def fields_to_tensor(facies_onehot: torch.Tensor, continuous_fields: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([facies_onehot] + [f.unsqueeze(1) if f.dim() == 4 else f for f in continuous_fields], dim=1)

