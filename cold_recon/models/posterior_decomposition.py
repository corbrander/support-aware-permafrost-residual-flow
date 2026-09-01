from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class LocalBiasScaleHead3D(nn.Module):
    """Predict deterministic bias, a safe local gate, and anomaly scale."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 32) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv3d(int(in_channels), int(width), 3, padding=1),
            nn.GroupNorm(min(8, int(width)), int(width)),
            nn.SiLU(),
            nn.Conv3d(int(width), int(width), 3, padding=1),
            nn.SiLU(),
        )
        self.bias = nn.Conv3d(int(width), int(out_channels), 1)
        self.gate = nn.Conv3d(int(width), int(out_channels), 1)
        self.log_scale = nn.Conv3d(int(width), int(out_channels), 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        bias = self.bias(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        scale = F.softplus(self.log_scale(hidden)) + 1.0e-4
        return bias, gate, scale


def zero_mean_anomalies(anomalies: torch.Tensor) -> torch.Tensor:
    if anomalies.ndim < 2:
        raise ValueError("anomalies must have a leading ensemble dimension")
    return anomalies - anomalies.mean(dim=0, keepdim=True)


def reliability_adjusted_gate(
    learned_gate: torch.Tensor,
    ood_score: torch.Tensor | None = None,
    anchor_disagreement: torch.Tensor | None = None,
) -> torch.Tensor:
    gate = learned_gate.clamp(0.0, 1.0)
    if ood_score is not None:
        gate = gate * (1.0 - ood_score.clamp(0.0, 1.0))
    if anchor_disagreement is not None:
        gate = gate * (1.0 - anchor_disagreement.clamp(0.0, 1.0))
    return gate.clamp(0.0, 1.0)


def compose_safe_ensemble(
    anchor: torch.Tensor,
    deterministic_bias: torch.Tensor,
    anomalies: torch.Tensor,
    bias_gate: torch.Tensor,
    anomaly_scale: torch.Tensor,
    *,
    ood_score: torch.Tensor | None = None,
    anchor_disagreement: torch.Tensor | None = None,
    channel_bounds: dict[int, tuple[float, float]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose anchor + gated bias + locally scaled zero-mean anomalies.

    ``anomalies`` has shape ``[M, C, X, Y, Z]``. Other tensors may have a
    leading singleton batch and broadcast across the ensemble.
    """

    if anomalies.ndim != 5:
        raise ValueError("anomalies must have shape [ensemble, channels, x, y, z]")
    centered = zero_mean_anomalies(anomalies)
    gate = reliability_adjusted_gate(bias_gate, ood_score, anchor_disagreement)
    mean = anchor + gate * deterministic_bias
    ensemble = mean + anomaly_scale.clamp_min(1.0e-5) * centered
    if channel_bounds:
        channels = []
        for channel in range(ensemble.shape[1]):
            value = ensemble[:, channel : channel + 1]
            if channel in channel_bounds:
                lower, upper = channel_bounds[channel]
                value = value.clamp(float(lower), float(upper))
            channels.append(value)
        ensemble = torch.cat(channels, dim=1)
    diagnostics = {
        "centered_anomaly_mean_abs": centered.mean(dim=0).abs().mean(),
        "effective_bias_gate_mean": gate.mean(),
        "ensemble_mean": ensemble.mean(dim=0, keepdim=True),
        "ensemble_std": ensemble.std(dim=0, keepdim=True, unbiased=False),
    }
    return ensemble, diagnostics


@dataclass(frozen=True)
class NonInferiorityDecision:
    allow_bias: bool
    anchor_loss: float
    candidate_loss: float
    margin: float


def noninferiority_decision(
    anchor_loss: float,
    candidate_loss: float,
    margin: float,
) -> NonInferiorityDecision:
    """Allow a field bias head only when it meets a predeclared loss margin."""

    allow = float(candidate_loss) <= float(anchor_loss) + float(margin)
    return NonInferiorityDecision(
        allow_bias=bool(allow),
        anchor_loss=float(anchor_loss),
        candidate_loss=float(candidate_loss),
        margin=float(margin),
    )
