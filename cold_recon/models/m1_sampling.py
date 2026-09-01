from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from cold_recon.data.data_schema import OBS_TYPES
from cold_recon.models.high_eic_head import monotone_event_probabilities
from cold_recon.models.likelihood_guidance import ContinuousChannelSpec, SupportLikelihoodGuide
from cold_recon.models.posterior_decomposition import zero_mean_anomalies
from cold_recon.training.factorized_volume_codec import (
    N_ICE_STRUCTURE,
    N_LITHOLOGY,
    N_THERMAL_STATE,
)


def _guide_chunks(
    state: torch.Tensor,
    guide: SupportLikelihoodGuide,
    batch_size: int,
) -> tuple[torch.Tensor, float]:
    velocities: list[torch.Tensor] = []
    losses: list[float] = []
    for start in range(0, state.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), state.shape[0])
        velocity, loss = guide.velocity(state[start:stop])
        velocities.append(velocity)
        losses.append(float(loss.cpu()))
    return torch.cat(velocities, dim=0), float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def _guide_loss_chunks(
    state: torch.Tensor,
    guide: SupportLikelihoodGuide,
    batch_size: int,
) -> float:
    losses = [
        float(guide.loss(state[start : min(start + int(batch_size), state.shape[0])]).cpu())
        for start in range(0, state.shape[0], int(batch_size))
    ]
    return float(sum(losses) / max(len(losses), 1))


def sample_support_guided_ensemble(
    *,
    model: Any,
    bias_head: Any,
    event_head: Any,
    autoencoder: Any,
    anchor: torch.Tensor,
    raster: torch.Tensor,
    tokens: torch.Tensor,
    sample: dict,
    site_adapter: Any | None = None,
    n_members: int = 64,
    sampling_steps: int = 10,
    time_scale: float = 79.0,
    guidance_strength: float = 0.25,
    guidance_batch_size: int = 8,
    ood_gate_multiplier: float = 1.0,
    use_bias_decomposition: bool = True,
    seed: int = 42,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sample the final safe M1 posterior with stepwise support assimilation."""

    device = anchor.device
    model.eval()
    bias_head.eval()
    event_head.eval()
    with torch.no_grad():
        encoded = model.encode_context(raster, tokens, target_shape=anchor.shape[-3:])
        if site_adapter is not None:
            encoded = site_adapter(encoded)
        if bool(use_bias_decomposition):
            bias, gate, local_scale = bias_head(encoded)
            gate = gate * float(ood_gate_multiplier)
            effective_anchor = anchor + gate * bias
        else:
            bias = torch.zeros_like(anchor)
            gate = torch.zeros_like(anchor[:, :1])
            local_scale = torch.ones_like(anchor)
            effective_anchor = anchor
        event_logits = event_head(
            encoded,
            raster=raster,
            output_shape=tuple(int(value) for value in raster.shape[-3:]),
        )
        event_temperature = getattr(event_head, "calibration_temperature", None)
        if event_temperature is not None:
            event_logits = event_logits / event_temperature.to(
                event_logits.device, event_logits.dtype
            )
        event_probability = monotone_event_probabilities(event_logits)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    state = torch.randn(
        (int(n_members), *anchor.shape[1:]), device=device, generator=generator
    )
    state = zero_mean_anomalies(state)
    anchor_b = effective_anchor.expand(int(n_members), -1, -1, -1, -1)
    context_b = encoded.expand(int(n_members), -1, -1, -1, -1)
    guide = SupportLikelihoodGuide(
        autoencoder=autoencoder,
        anchor=effective_anchor,
        latent_scale=local_scale,
        sample=sample,
        continuous_channels={
            OBS_TYPES["borehole_eic"]: ContinuousChannelSpec(
                N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE, 1.0
            ),
            OBS_TYPES["borehole_temperature"]: ContinuousChannelSpec(
                N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE + 1, 10.0
            ),
            OBS_TYPES["nmr_unfrozen_water"]: ContinuousChannelSpec(
                N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE + 2, 1.0
            ),
            OBS_TYPES["ert_log_resistivity"]: ContinuousChannelSpec(
                N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE + 3, 10.0
            ),
        },
        alt_temperature_channel=ContinuousChannelSpec(
            N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE + 1,
            10.0,
        ),
        n_facies=0,
    )
    support_likelihood_initial = (
        _guide_loss_chunks(state, guide, int(guidance_batch_size))
        if float(guidance_strength) > 0.0
        else float("nan")
    )
    step_size = 1.0 / float(sampling_steps)
    history: list[dict[str, float]] = []
    cumulative_anchor_shift = torch.zeros_like(effective_anchor)
    for step in range(int(sampling_steps)):
        tau_0 = float(step) * step_size
        tau_1 = float(step + 1) * step_size
        with torch.no_grad():
            t_0 = torch.full((int(n_members),), tau_0 * float(time_scale), device=device)
            velocity_0 = model.velocity_from_encoded(state, t_0, anchor_b, context_b)
            predictor = state + step_size * velocity_0
            t_1 = torch.full((int(n_members),), tau_1 * float(time_scale), device=device)
            velocity_1 = model.velocity_from_encoded(predictor, t_1, anchor_b, context_b)
            state = state + 0.5 * step_size * (velocity_0 + velocity_1)
            state = zero_mean_anomalies(state)
        likelihood_before = float("nan")
        if float(guidance_strength) > 0.0:
            guide_velocity, likelihood_before = _guide_chunks(
                state, guide, int(guidance_batch_size)
            )
            guided_state = state + step_size * float(guidance_strength) * guide_velocity
            normalized_mean_shift = guided_state.mean(dim=0, keepdim=True)
            deterministic_shift = (
                float(ood_gate_multiplier) * local_scale * normalized_mean_shift
            )
            effective_anchor = (effective_anchor + deterministic_shift).detach()
            cumulative_anchor_shift = cumulative_anchor_shift + deterministic_shift
            guide.anchor = effective_anchor
            anchor_b = effective_anchor.expand(int(n_members), -1, -1, -1, -1)
            state = zero_mean_anomalies(guided_state).detach()
        history.append(
            {
                "step": float(step + 1),
                "tau": tau_1,
                "likelihood_before_centering": likelihood_before,
                "anomaly_mean_abs": float(state.mean(dim=0).abs().mean().cpu()),
                "anchor_shift_mean_abs": float(
                    cumulative_anchor_shift.abs().mean().cpu()
                ),
            }
        )

    support_likelihood_final = (
        _guide_loss_chunks(state, guide, int(guidance_batch_size))
        if float(guidance_strength) > 0.0
        else float("nan")
    )

    decoded_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(n_members), int(guidance_batch_size)):
            stop = min(start + int(guidance_batch_size), int(n_members))
            latent = effective_anchor + local_scale * state[start:stop]
            decoded_chunks.append(autoencoder.decode(latent))
    decoded = torch.cat(decoded_chunks, dim=0)
    continuous_offset = N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE
    decoded[:, continuous_offset] = decoded[:, continuous_offset].clamp(0.0, 0.90)
    decoded[:, continuous_offset + 1] = decoded[:, continuous_offset + 1].clamp(-1.2, 0.4)
    decoded[:, continuous_offset + 2] = decoded[:, continuous_offset + 2].clamp(0.0, 0.85)
    decoded[:, continuous_offset + 3] = decoded[:, continuous_offset + 3].clamp(0.0, 1.5)
    full_event_probability = F.interpolate(
        event_probability,
        size=decoded.shape[-3:],
        mode="trilinear",
        align_corners=False,
    )
    return decoded, {
        "history": history,
        "bias_gate_mean": float(gate.mean().cpu()),
        "local_scale_mean": float(local_scale.mean().cpu()),
        "anomaly_mean_abs": float(state.mean(dim=0).abs().mean().cpu()),
        "support_likelihood_initial": support_likelihood_initial,
        "support_likelihood_final": support_likelihood_final,
        "guidance_anchor_shift_mean_abs": float(
            cumulative_anchor_shift.abs().mean().cpu()
        ),
        "event_probability": full_event_probability.detach().cpu(),
    }
