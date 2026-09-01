from __future__ import annotations

import torch
from torch import nn


class ObsTransformerEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_proj(tokens)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=tokens.device)
        h = self.encoder(h, mask=attention_mask, src_key_padding_mask=key_padding_mask)
        if key_padding_mask is None:
            pooled = h.mean(dim=1)
        else:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)
