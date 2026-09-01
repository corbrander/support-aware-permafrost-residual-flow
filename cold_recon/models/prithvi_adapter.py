from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PrithviAdapter(nn.Module):
    """Remote-sensing surface encoder with a local fallback backend.

    The adapter can wrap an externally supplied Prithvi/terratorch-style
    backbone that returns a 4D feature map. If no backbone is supplied, it
    computes normalized multispectral/static-feature texture summaries and
    projects them with a 1x1 convolution. This keeps the COLD-Recon surface
    conditioning path runnable on local machines without large foundation-model
    weights, while preserving a narrow integration point for later replacement.
    """

    def __init__(
        self,
        input_channels: int = 8,
        output_channels: int = 32,
        backbone: nn.Module | None = None,
        external_channels: int | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.backbone = backbone
        self.external_channels = external_channels
        self.freeze_backbone = bool(freeze_backbone)
        if self.backbone is not None and self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        if self.backbone is None:
            self.projection = nn.Sequential(
                nn.Conv2d(self.input_channels * 3, self.output_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(self.output_channels, self.output_channels, kernel_size=1),
            )
        else:
            in_ch = int(external_channels or output_channels)
            self.projection = nn.Identity() if in_ch == self.output_channels else nn.Conv2d(in_ch, self.output_channels, kernel_size=1)

    @property
    def backend_name(self) -> str:
        return "external_prithvi" if self.backbone is not None else "local_texture_fallback"

    def available(self) -> bool:
        """Return True because the adapter always has a runnable fallback."""
        return True

    def _target_device(self) -> torch.device:
        for parameter in self.parameters():
            return parameter.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

    def _as_tensor(self, surface: torch.Tensor | np.ndarray | Mapping[str, np.ndarray]) -> torch.Tensor:
        if isinstance(surface, Mapping):
            arrays = [np.asarray(surface[name], dtype=np.float32) for name in sorted(surface)]
            surface = np.stack(arrays, axis=0)
        if isinstance(surface, np.ndarray):
            surface = torch.from_numpy(surface.astype(np.float32, copy=False))
        if not isinstance(surface, torch.Tensor):
            raise TypeError("surface must be a torch.Tensor, numpy array, or mapping of 2D arrays")
        x = surface.float()
        if x.dim() == 3:
            if x.shape[-1] == self.input_channels and x.shape[0] != self.input_channels:
                x = x.permute(2, 0, 1)
            elif x.shape[-1] <= 32 and x.shape[0] > 32:
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            if x.shape[-1] == self.input_channels and x.shape[1] != self.input_channels:
                x = x.permute(0, 3, 1, 2)
            elif x.shape[-1] <= 32 and x.shape[1] > 32:
                x = x.permute(0, 3, 1, 2)
        if x.dim() != 4:
            raise ValueError("surface must have shape [C,H,W], [H,W,C], [B,C,H,W], or [B,H,W,C]")
        return x

    @staticmethod
    def _extract_feature_map(backbone_output: Any) -> torch.Tensor:
        if isinstance(backbone_output, torch.Tensor):
            features = backbone_output
        elif isinstance(backbone_output, Mapping):
            for key in ("features", "feature_map", "last_hidden_state", "x"):
                value = backbone_output.get(key)
                if isinstance(value, torch.Tensor):
                    features = value
                    break
            else:
                raise ValueError("backbone output mapping does not contain a tensor feature map")
        elif isinstance(backbone_output, (tuple, list)) and backbone_output and isinstance(backbone_output[0], torch.Tensor):
            features = backbone_output[0]
        else:
            raise TypeError("backbone must return a tensor, tensor tuple/list, or mapping with tensor features")
        if features.dim() != 4:
            raise ValueError("PrithviAdapter expects a 4D backbone feature map [B,C,H,W]")
        return features

    def _fallback_features(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        norm = (x - mean) / std
        smooth = F.avg_pool2d(norm, kernel_size=3, stride=1, padding=1)
        grad_x = F.pad(norm[..., 1:, :] - norm[..., :-1, :], (0, 0, 0, 1))
        grad_y = F.pad(norm[..., :, 1:] - norm[..., :, :-1], (0, 1, 0, 0))
        texture = 0.5 * (grad_x.abs() + grad_y.abs())
        return torch.cat([norm, smooth, texture], dim=1)

    def forward(self, surface: torch.Tensor | np.ndarray | Mapping[str, np.ndarray]) -> torch.Tensor:
        x = self._as_tensor(surface)
        if self.backbone is None:
            features = self._fallback_features(x.to(self._target_device()))
        else:
            x = x.to(self._target_device())
            with torch.set_grad_enabled(not self.freeze_backbone):
                features = self._extract_feature_map(self.backbone(x))
        return self.projection(features)

    def encode(self, surface: torch.Tensor | np.ndarray | Mapping[str, np.ndarray]) -> torch.Tensor:
        return self.forward(surface)
