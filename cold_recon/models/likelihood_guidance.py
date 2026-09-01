from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cold_recon.data.data_schema import OBS_TYPES
from cold_recon.operators.support import build_error_covariance, build_observation_operator


@dataclass(frozen=True)
class ContinuousChannelSpec:
    channel: int
    physical_scale: float = 1.0


@dataclass
class _LikelihoodTerm:
    type_id: int
    indices: np.ndarray
    operator: torch.Tensor
    observed: torch.Tensor
    covariance_cholesky: torch.Tensor | None
    continuous_spec: ContinuousChannelSpec | None
    is_surface_crossing: bool = False


def _torch_sparse(matrix, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        device=device,
        check_invariants=False,
    ).coalesce()


class SupportLikelihoodGuide:
    """Differentiable support-aware likelihood for latent flow states."""

    def __init__(
        self,
        *,
        autoencoder: Any,
        anchor: torch.Tensor,
        latent_scale: torch.Tensor,
        sample: dict[str, Any],
        continuous_channels: dict[int, ContinuousChannelSpec],
        alt_temperature_channel: ContinuousChannelSpec | None = None,
        n_facies: int = 0,
        correlated_ert: bool = True,
        ert_length_scale_m: float = 6.0,
        min_sigma: float = 1.0e-3,
        gradient_clip: float = 5.0,
        balance_observation_types: bool = True,
    ) -> None:
        self.autoencoder = autoencoder
        self.anchor = anchor
        self.latent_scale = latent_scale
        self.sample = sample
        self.continuous_channels = dict(continuous_channels)
        self.alt_temperature_channel = alt_temperature_channel
        self.n_facies = int(n_facies)
        self.correlated_ert = bool(correlated_ert)
        self.ert_length_scale_m = float(ert_length_scale_m)
        self.min_sigma = float(min_sigma)
        self.gradient_clip = float(gradient_clip)
        self.balance_observation_types = bool(balance_observation_types)
        self._terms: dict[tuple[torch.device, torch.dtype], list[_LikelihoodTerm]] = {}

    def _index_groups(self) -> list[tuple[int, np.ndarray]]:
        observations = self.sample["observations"]
        valid = np.asarray(observations.mask, dtype=bool)
        supported = set(self.continuous_channels)
        if self.alt_temperature_channel is not None:
            supported.add(OBS_TYPES["alt"])
        if self.n_facies > 0:
            supported.add(OBS_TYPES["borehole_facies"])
        groups: list[tuple[int, np.ndarray]] = []
        for type_id in sorted(supported):
            indices = np.flatnonzero(valid & (observations.type_ids == type_id))
            if len(indices) == 0:
                continue
            if type_id == OBS_TYPES["ert_log_resistivity"] and self.correlated_ert:
                for group_id in np.unique(observations.group_ids[indices]):
                    if group_id < 0:
                        continue
                    subset = indices[observations.group_ids[indices] == group_id]
                    if len(subset):
                        groups.append((type_id, subset))
                ungrouped = indices[observations.group_ids[indices] < 0]
                if len(ungrouped):
                    groups.append((type_id, ungrouped))
            else:
                groups.append((type_id, indices))
        return groups

    def _build_terms(self, device: torch.device, dtype: torch.dtype) -> list[_LikelihoodTerm]:
        key = (device, dtype)
        if key in self._terms:
            return self._terms[key]
        observations = self.sample["observations"]
        terms: list[_LikelihoodTerm] = []
        for type_id, indices in self._index_groups():
            support = build_observation_operator(observations, self.sample["grid"], indices=indices)
            operator = _torch_sparse(support.matrix, device, dtype)
            observed = torch.as_tensor(observations.values[indices], device=device, dtype=dtype)
            continuous_spec = self.continuous_channels.get(type_id)
            is_surface_crossing = (
                type_id == OBS_TYPES["alt"]
                and self.alt_temperature_channel is not None
            )
            covariance_cholesky: torch.Tensor | None = None
            if continuous_spec is not None or is_surface_crossing:
                covariance = build_error_covariance(
                    observations,
                    indices,
                    correlated=(type_id == OBS_TYPES["ert_log_resistivity"] and self.correlated_ert),
                    length_scale=self.ert_length_scale_m,
                    min_sigma=self.min_sigma,
                )
                covariance_tensor = torch.as_tensor(covariance, device=device, dtype=dtype)
                covariance_cholesky = torch.linalg.cholesky(covariance_tensor)
            terms.append(
                _LikelihoodTerm(
                    type_id=type_id,
                    indices=indices,
                    operator=operator,
                    observed=observed,
                    covariance_cholesky=covariance_cholesky,
                    continuous_spec=continuous_spec,
                    is_surface_crossing=is_surface_crossing,
                )
            )
        self._terms[key] = terms
        return terms

    def decode(self, normalized_state: torch.Tensor) -> torch.Tensor:
        anchor = self.anchor.to(normalized_state.device, normalized_state.dtype)
        scale = self.latent_scale.to(normalized_state.device, normalized_state.dtype)
        latent = anchor.expand(normalized_state.shape[0], -1, -1, -1, -1) + normalized_state * scale
        return self.autoencoder.decode(latent)

    def loss(self, normalized_state: torch.Tensor) -> torch.Tensor:
        decoded = self.decode(normalized_state)
        terms = self._build_terms(decoded.device, decoded.dtype)
        batch = decoded.shape[0]
        n_voxels = int(np.prod(decoded.shape[-3:]))
        type_totals: dict[int, torch.Tensor] = {}
        type_weights: dict[int, int] = {}
        for term in terms:
            if term.is_surface_crossing:
                spec = self.alt_temperature_channel
                if spec is None:
                    raise RuntimeError("ALT likelihood term has no temperature channel")
                temperature = decoded[:, int(spec.channel)] * float(spec.physical_scale)
                z = np.asarray(self.sample["grid"]["z"], dtype=np.float64)
                dz = float(np.mean(np.diff(z))) if len(z) > 1 else float(
                    self.sample["grid"].get("dz", 1.0)
                )
                thaw_probability = torch.sigmoid(temperature / 0.25)
                alt_grid = torch.sum(thaw_probability, dim=-1) * dz
                alt_volume = alt_grid[..., None].expand(-1, -1, -1, temperature.shape[-1])
                predicted = torch.sparse.mm(
                    term.operator,
                    alt_volume.reshape(batch, n_voxels).T,
                ).T
                residual = predicted - term.observed[None, :]
                solved = torch.cholesky_solve(
                    residual.T, term.covariance_cholesky
                ).T
                type_totals[term.type_id] = type_totals.get(
                    term.type_id, decoded.new_zeros(())
                ) + torch.sum(residual * solved)
                type_weights[term.type_id] = type_weights.get(
                    term.type_id, 0
                ) + int(residual.numel())
                continue
            if term.continuous_spec is not None:
                spec = term.continuous_spec
                field = decoded[:, int(spec.channel)].reshape(batch, n_voxels)
                predicted = torch.sparse.mm(term.operator, field.T).T * float(spec.physical_scale)
                residual = predicted - term.observed[None, :]
                solved = torch.cholesky_solve(residual.T, term.covariance_cholesky).T
                type_totals[term.type_id] = type_totals.get(
                    term.type_id, decoded.new_zeros(())
                ) + torch.sum(residual * solved)
                type_weights[term.type_id] = type_weights.get(
                    term.type_id, 0
                ) + int(residual.numel())
                continue
            probabilities = torch.softmax(decoded[:, : self.n_facies], dim=1)
            probability_grid = probabilities.permute(0, 2, 3, 4, 1).reshape(
                batch, n_voxels, self.n_facies
            )
            columns = probability_grid.permute(1, 0, 2).reshape(n_voxels, batch * self.n_facies)
            supported = torch.sparse.mm(term.operator, columns)
            supported = supported.reshape(len(term.indices), batch, self.n_facies).permute(1, 0, 2)
            labels = term.observed.round().long().clamp(0, self.n_facies - 1)
            chosen = supported.gather(2, labels[None, :, None].expand(batch, -1, 1)).squeeze(-1)
            type_totals[term.type_id] = type_totals.get(
                term.type_id, decoded.new_zeros(())
            ) - 2.0 * torch.log(chosen.clamp_min(1.0e-7)).sum()
            type_weights[term.type_id] = type_weights.get(
                term.type_id, 0
            ) + int(chosen.numel())
        if not type_weights:
            return decoded.sum() * 0.0
        if self.balance_observation_types:
            normalized = [
                type_totals[type_id] / float(type_weights[type_id])
                for type_id in sorted(type_weights)
            ]
            return 0.5 * torch.stack(normalized).mean()
        total = torch.stack(list(type_totals.values())).sum()
        weight = sum(type_weights.values())
        return 0.5 * total / float(weight)

    def velocity(self, normalized_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            state = normalized_state.detach().requires_grad_(True)
            loss = self.loss(state)
            gradient = torch.autograd.grad(loss, state, retain_graph=False, create_graph=False)[0]
        flat = gradient.flatten(1)
        norms = torch.linalg.vector_norm(flat, dim=1).clamp_min(1.0e-8)
        factors = torch.clamp(float(self.gradient_clip) / norms, max=1.0)
        clipped = gradient * factors.view(-1, 1, 1, 1, 1)
        return -clipped.detach(), loss.detach()


def guided_heun_sample_corrections(
    model: Any,
    anchor: torch.Tensor,
    context: torch.Tensor,
    guide: SupportLikelihoodGuide,
    *,
    n_samples: int,
    sampling_steps: int = 10,
    time_scale: float = 79.0,
    guidance_strength: float = 0.25,
    seed: int = 42,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Heun flow integration followed by a likelihood proximal-gradient split."""

    device = anchor.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    state = torch.randn((int(n_samples), *anchor.shape[1:]), device=device, generator=generator)
    anchor_b = anchor.expand(int(n_samples), -1, -1, -1, -1)
    context_latent = model.encode_context(context, target_shape=tuple(state.shape[-3:]))
    context_b = context_latent.expand(int(n_samples), -1, -1, -1, -1)
    model.eval()
    step_size = 1.0 / float(sampling_steps)
    history: list[dict[str, float]] = []
    for step in range(int(sampling_steps)):
        tau_0 = float(step) * step_size
        tau_1 = float(step + 1) * step_size
        with torch.no_grad():
            t_0 = torch.full((int(n_samples),), tau_0 * float(time_scale), device=device)
            velocity_0 = model(state, t_0, anchor_b, context_b, context_is_encoded=True)
            predictor = state + step_size * velocity_0
            t_1 = torch.full((int(n_samples),), tau_1 * float(time_scale), device=device)
            velocity_1 = model(predictor, t_1, anchor_b, context_b, context_is_encoded=True)
            state = state + 0.5 * step_size * (velocity_0 + velocity_1)
        guide_velocity, before = guide.velocity(state)
        state = (state + step_size * float(guidance_strength) * guide_velocity).detach()
        with torch.no_grad():
            after = guide.loss(state)
        history.append(
            {
                "step": float(step + 1),
                "tau": tau_1,
                "likelihood_before": float(before.cpu()),
                "likelihood_after": float(after.cpu()),
            }
        )
    return state.clamp(-4.0, 4.0), history
