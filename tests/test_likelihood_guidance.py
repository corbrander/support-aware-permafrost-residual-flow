from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cold_recon.data.data_schema import OBS_TYPES, SUPPORT_TYPES, ObservationTable
from cold_recon.models.likelihood_guidance import (
    ContinuousChannelSpec,
    SupportLikelihoodGuide,
    guided_heun_sample_corrections,
)


class _IdentityAutoencoder:
    @staticmethod
    def decode(x: torch.Tensor) -> torch.Tensor:
        return x


class _ZeroFlow(nn.Module):
    @staticmethod
    def encode_context(context: torch.Tensor, target_shape=None) -> torch.Tensor:
        return context

    def forward(self, x, t, anchor, context, context_is_encoded=False):
        return torch.zeros_like(x)


def test_guided_heun_reduces_support_likelihood() -> None:
    grid = {
        "x": np.arange(3, dtype=np.float32),
        "y": np.arange(2, dtype=np.float32),
        "z": np.arange(2, dtype=np.float32),
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }
    observations = ObservationTable(
        coords=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        type_ids=np.array([OBS_TYPES["borehole_eic"]]),
        values=np.array([1.0], dtype=np.float32),
        sigma=np.array([0.2], dtype=np.float32),
        mask=np.ones(1, dtype=bool),
    )
    sample = {"grid": grid, "observations": observations}
    anchor = torch.zeros(1, 1, 3, 2, 2)
    context = torch.zeros(1, 1, 3, 2, 2)
    guide = SupportLikelihoodGuide(
        autoencoder=_IdentityAutoencoder(),
        anchor=anchor,
        latent_scale=torch.ones_like(anchor),
        sample=sample,
        continuous_channels={OBS_TYPES["borehole_eic"]: ContinuousChannelSpec(0, 1.0)},
        gradient_clip=50.0,
    )
    generator = torch.Generator().manual_seed(12)
    initial = torch.randn((4, 1, 3, 2, 2), generator=generator)
    initial_loss = float(guide.loss(initial))
    guided, history = guided_heun_sample_corrections(
        _ZeroFlow(),
        anchor,
        context,
        guide,
        n_samples=4,
        sampling_steps=8,
        guidance_strength=1.0,
        seed=12,
    )
    assert float(guide.loss(guided)) < initial_loss
    assert all(row["likelihood_after"] <= row["likelihood_before"] + 1e-6 for row in history)


def test_alt_surface_crossing_likelihood_has_useful_gradient() -> None:
    grid = {
        "x": np.arange(2, dtype=np.float32),
        "y": np.arange(2, dtype=np.float32),
        "z": np.arange(4, dtype=np.float32),
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }
    observations = ObservationTable(
        coords=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        type_ids=np.asarray([OBS_TYPES["alt"]]),
        values=np.asarray([3.0], dtype=np.float32),
        sigma=np.asarray([0.2], dtype=np.float32),
        mask=np.ones(1, dtype=bool),
        support_type_ids=np.asarray([SUPPORT_TYPES["surface_crossing"]]),
    )
    anchor = torch.zeros(1, 1, 2, 2, 4)
    guide = SupportLikelihoodGuide(
        autoencoder=_IdentityAutoencoder(),
        anchor=anchor,
        latent_scale=torch.ones_like(anchor),
        sample={"grid": grid, "observations": observations},
        continuous_channels={},
        alt_temperature_channel=ContinuousChannelSpec(0, 1.0),
        gradient_clip=50.0,
    )
    state = torch.zeros(2, 1, 2, 2, 4)
    velocity, before = guide.velocity(state)
    after = guide.loss(state + 0.01 * velocity)

    assert float(torch.linalg.vector_norm(velocity)) > 0.0
    assert float(after) < float(before)


def test_likelihood_balances_observation_types_not_record_counts() -> None:
    grid = {
        "x": np.arange(3, dtype=np.float32),
        "y": np.arange(3, dtype=np.float32),
        "z": np.arange(2, dtype=np.float32),
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }
    ert_coords = np.asarray(
        [[float(i % 3), float(i // 3), 0.0] for i in range(9)],
        dtype=np.float32,
    )
    observations = ObservationTable(
        coords=np.vstack([np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32), ert_coords]),
        type_ids=np.asarray(
            [OBS_TYPES["borehole_eic"]] + [OBS_TYPES["ert_log_resistivity"]] * 9
        ),
        values=np.asarray([2.0] + [0.5] * 9, dtype=np.float32),
        sigma=np.ones(10, dtype=np.float32),
        mask=np.ones(10, dtype=bool),
        group_ids=np.full(10, -1, dtype=np.int64),
    )
    anchor = torch.zeros(1, 1, 3, 3, 2)
    common = dict(
        autoencoder=_IdentityAutoencoder(),
        anchor=anchor,
        latent_scale=torch.ones_like(anchor),
        sample={"grid": grid, "observations": observations},
        continuous_channels={
            OBS_TYPES["borehole_eic"]: ContinuousChannelSpec(0, 1.0),
            OBS_TYPES["ert_log_resistivity"]: ContinuousChannelSpec(0, 1.0),
        },
        correlated_ert=False,
    )
    balanced = SupportLikelihoodGuide(
        **common, balance_observation_types=True
    ).loss(torch.zeros(1, 1, 3, 3, 2))
    record_weighted = SupportLikelihoodGuide(
        **common, balance_observation_types=False
    ).loss(torch.zeros(1, 1, 3, 3, 2))

    assert np.isclose(float(balanced), 1.0625, atol=1.0e-5)
    assert np.isclose(float(record_weighted), 0.3125, atol=1.0e-5)
