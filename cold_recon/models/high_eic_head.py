from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HighEICEventHead3D(nn.Module):
    """Independent multi-threshold event head for high excess-ice screening."""

    def __init__(
        self,
        in_channels: int,
        thresholds: tuple[float, ...] = (0.20, 0.30, 0.40),
        width: int = 24,
        raster_channels: int = 0,
    ) -> None:
        super().__init__()
        self.thresholds = tuple(float(value) for value in thresholds)
        self.net = nn.Sequential(
            nn.Conv3d(int(in_channels), int(width), 3, padding=1),
            nn.GroupNorm(min(6, int(width)), int(width)),
            nn.SiLU(),
            nn.Conv3d(int(width), len(self.thresholds), 1),
        )
        self.raster_channels = int(raster_channels)
        self.refinement = None
        if self.raster_channels > 0:
            refinement_channels = int(width)
            self.refinement = nn.Sequential(
                nn.Conv3d(
                    int(in_channels) + self.raster_channels,
                    refinement_channels,
                    1,
                ),
                nn.GroupNorm(min(6, refinement_channels), refinement_channels),
                nn.SiLU(),
                nn.Conv3d(
                    refinement_channels,
                    refinement_channels,
                    3,
                    padding=1,
                    groups=refinement_channels,
                ),
                nn.GroupNorm(min(6, refinement_channels), refinement_channels),
                nn.SiLU(),
                nn.Conv3d(refinement_channels, len(self.thresholds), 1),
            )

    def forward(
        self,
        features: torch.Tensor,
        raster: torch.Tensor | None = None,
        output_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        logits = self.net(features)
        target_shape = (
            tuple(int(value) for value in raster.shape[-3:])
            if raster is not None
            else output_shape
        )
        if target_shape is None:
            return logits
        logits = F.interpolate(
            logits,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )
        if raster is None or self.refinement is None:
            return logits
        if raster.shape[1] != self.raster_channels:
            raise ValueError(
                f"expected {self.raster_channels} raster channels, got {raster.shape[1]}"
            )
        upsampled_features = F.interpolate(
            features,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )
        return logits + self.refinement(
            torch.cat([upsampled_features, raster], dim=1)
        )


def monotone_event_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Project threshold probabilities so P(E>t) is non-increasing in t."""

    probabilities = torch.sigmoid(logits)
    if probabilities.shape[1] <= 1:
        return probabilities
    projected = [probabilities[:, :1]]
    for index in range(1, probabilities.shape[1]):
        projected.append(torch.minimum(projected[-1], probabilities[:, index : index + 1]))
    return torch.cat(projected, dim=1)


def focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(2, logits.ndim))
    true_positive = torch.sum(probabilities * targets, dim=dims)
    false_positive = torch.sum(probabilities * (1.0 - targets), dim=dims)
    false_negative = torch.sum((1.0 - probabilities) * targets, dim=dims)
    tversky = (true_positive + eps) / (
        true_positive + float(alpha) * false_positive + float(beta) * false_negative + eps
    )
    return torch.mean((1.0 - tversky) ** float(gamma))


def calibrated_event_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tversky_weight: float = 0.50,
    bce_weight: float = 0.50,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine rare-event overlap with a proper probability scoring loss."""

    tversky = focal_tversky_loss(logits, targets)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    total = float(tversky_weight) * tversky + float(bce_weight) * bce
    return total, {"total": total, "tversky": tversky, "bce": bce}
