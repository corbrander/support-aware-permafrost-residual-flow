from __future__ import annotations

import torch
from torch import nn

from cold_recon.models.implicit_field import ImplicitCryoField
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.models.surface_encoder import SurfaceFeatureEncoder


class COLDReconImplicitModel(nn.Module):
    def __init__(
        self,
        token_dim: int,
        surface_feature_dim: int = 8,
        obs_hidden_dim: int = 96,
        obs_layers: int = 2,
        obs_heads: int = 4,
        surface_hidden_dim: int = 32,
        fourier_features: int = 48,
        implicit_hidden_dim: int = 192,
        implicit_layers: int = 5,
        n_facies: int = 7,
    ) -> None:
        super().__init__()
        self.obs_encoder = ObsTransformerEncoder(token_dim, obs_hidden_dim, obs_layers, obs_heads)
        self.surface_encoder = SurfaceFeatureEncoder(surface_feature_dim, surface_hidden_dim, surface_hidden_dim)
        self.field = ImplicitCryoField(
            surface_dim=surface_hidden_dim,
            context_dim=obs_hidden_dim,
            fourier_features=fourier_features,
            hidden_dim=implicit_hidden_dim,
            num_layers=implicit_layers,
            n_facies=n_facies,
        )

    def forward(
        self,
        coords: torch.Tensor,
        surface_values: torch.Tensor,
        obs_tokens: torch.Tensor,
        obs_padding_mask: torch.Tensor | None = None,
        obs_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.obs_encoder(obs_tokens, obs_padding_mask, obs_attention_mask)
        surface_embedding = self.surface_encoder(surface_values)
        return self.field(coords, surface_embedding, context)


def build_model_from_config(config: dict) -> COLDReconImplicitModel:
    m = config["model"]
    return COLDReconImplicitModel(
        token_dim=int(m["token_dim"]),
        surface_feature_dim=int(m.get("surface_feature_dim", 8)),
        obs_hidden_dim=int(m.get("obs_hidden_dim", 96)),
        obs_layers=int(m.get("obs_layers", 2)),
        obs_heads=int(m.get("obs_heads", 4)),
        surface_hidden_dim=int(m.get("surface_hidden_dim", 32)),
        fourier_features=int(m.get("fourier_features", 48)),
        implicit_hidden_dim=int(m.get("implicit_hidden_dim", 192)),
        implicit_layers=int(m.get("implicit_layers", 5)),
        n_facies=int(m.get("n_facies", 7)),
    )
