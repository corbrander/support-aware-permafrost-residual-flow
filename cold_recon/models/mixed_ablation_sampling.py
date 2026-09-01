from __future__ import annotations

from typing import Any

import torch

from cold_recon.data.data_schema import OBS_TYPES
from cold_recon.models.likelihood_guidance import (
    ContinuousChannelSpec,
    SupportLikelihoodGuide,
)
from cold_recon.models.posterior_decomposition import zero_mean_anomalies
from cold_recon.training.mixed_volume_codec import N_CRYOFACIES


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
        losses.append(float(loss.detach().cpu()))
    return torch.cat(velocities, dim=0), float(sum(losses) / max(len(losses), 1))


def _guide_loss_chunks(
    state: torch.Tensor,
    guide: SupportLikelihoodGuide,
    batch_size: int,
) -> float:
    values = [
        float(
            guide.loss(
                state[start : min(start + int(batch_size), state.shape[0])]
            )
            .detach()
            .cpu()
        )
        for start in range(0, state.shape[0], int(batch_size))
    ]
    return float(sum(values) / max(len(values), 1))


def sample_mixed_ablation_ensemble(
    *,
    model: Any,
    autoencoder: Any,
    anchor: torch.Tensor,
    raster: torch.Tensor,
    tokens: torch.Tensor,
    sample: dict,
    n_members: int = 64,
    sampling_steps: int = 10,
    guidance_strength: float = 0.0,
    guidance_batch_size: int = 16,
    time_scale: float = 79.0,
    seed: int = 42,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = anchor.device
    model.eval()
    with torch.no_grad():
        encoded = model.encode_context(
            raster, tokens, target_shape=anchor.shape[-3:]
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    state = torch.randn(
        (int(n_members), *anchor.shape[1:]),
        device=device,
        generator=generator,
    )
    state = zero_mean_anomalies(state)
    effective_anchor = anchor
    anchor_batch = anchor.expand(int(n_members), -1, -1, -1, -1)
    context_batch = encoded.expand(int(n_members), -1, -1, -1, -1)
    latent_scale = torch.ones_like(anchor)
    guide = SupportLikelihoodGuide(
        autoencoder=autoencoder,
        anchor=effective_anchor,
        latent_scale=latent_scale,
        sample=sample,
        continuous_channels={
            OBS_TYPES["borehole_eic"]: ContinuousChannelSpec(
                N_CRYOFACIES, 1.0
            ),
            OBS_TYPES["borehole_temperature"]: ContinuousChannelSpec(
                N_CRYOFACIES + 1, 10.0
            ),
            OBS_TYPES["nmr_unfrozen_water"]: ContinuousChannelSpec(
                N_CRYOFACIES + 2, 1.0
            ),
            OBS_TYPES["ert_log_resistivity"]: ContinuousChannelSpec(
                N_CRYOFACIES + 3, 10.0
            ),
        },
        alt_temperature_channel=ContinuousChannelSpec(
            N_CRYOFACIES + 1, 10.0
        ),
        n_facies=N_CRYOFACIES,
    )
    initial = (
        _guide_loss_chunks(state, guide, int(guidance_batch_size))
        if float(guidance_strength) > 0.0
        else float("nan")
    )
    step_size = 1.0 / float(sampling_steps)
    cumulative_shift = torch.zeros_like(anchor)
    for step in range(int(sampling_steps)):
        tau_0 = float(step) * step_size
        tau_1 = float(step + 1) * step_size
        with torch.no_grad():
            t0 = torch.full(
                (int(n_members),), tau_0 * float(time_scale), device=device
            )
            velocity0 = model.velocity_from_encoded(
                state, t0, anchor_batch, context_batch
            )
            predictor = state + step_size * velocity0
            t1 = torch.full(
                (int(n_members),), tau_1 * float(time_scale), device=device
            )
            velocity1 = model.velocity_from_encoded(
                predictor, t1, anchor_batch, context_batch
            )
            state = zero_mean_anomalies(
                state + 0.5 * step_size * (velocity0 + velocity1)
            )
        if float(guidance_strength) > 0.0:
            guide_velocity, _ = _guide_chunks(
                state, guide, int(guidance_batch_size)
            )
            guided = state + step_size * float(guidance_strength) * guide_velocity
            shift = guided.mean(dim=0, keepdim=True)
            effective_anchor = (effective_anchor + shift).detach()
            cumulative_shift += shift
            guide.anchor = effective_anchor
            anchor_batch = effective_anchor.expand(
                int(n_members), -1, -1, -1, -1
            )
            state = zero_mean_anomalies(guided).detach()
    final = (
        _guide_loss_chunks(state, guide, int(guidance_batch_size))
        if float(guidance_strength) > 0.0
        else float("nan")
    )
    decoded_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(n_members), int(guidance_batch_size)):
            stop = min(start + int(guidance_batch_size), int(n_members))
            decoded_chunks.append(
                autoencoder.decode(effective_anchor + state[start:stop])
            )
    decoded = torch.cat(decoded_chunks, dim=0)
    decoded[:, N_CRYOFACIES] = decoded[:, N_CRYOFACIES].clamp(0.0, 0.90)
    decoded[:, N_CRYOFACIES + 1] = decoded[:, N_CRYOFACIES + 1].clamp(
        -1.2, 0.4
    )
    decoded[:, N_CRYOFACIES + 2] = decoded[:, N_CRYOFACIES + 2].clamp(
        0.0, 0.85
    )
    decoded[:, N_CRYOFACIES + 3] = decoded[:, N_CRYOFACIES + 3].clamp(
        0.0, 1.5
    )
    return decoded, {
        "anomaly_mean_abs": float(state.mean(dim=0).abs().mean().cpu()),
        "support_likelihood_initial": initial,
        "support_likelihood_final": final,
        "guidance_anchor_shift_mean_abs": float(
            cumulative_shift.abs().mean().cpu()
        ),
    }
