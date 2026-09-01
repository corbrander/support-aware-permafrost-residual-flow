from __future__ import annotations

import numpy as np
import torch

from cold_recon.data.data_schema import ObservationTable
from cold_recon.models.mixed_ablation_sampling import sample_mixed_ablation_ensemble
from cold_recon.training.mixed_volume_codec import (
    MIXED_CHANNELS,
    mixed_ensemble_to_posterior,
    mixed_reconstruction_loss,
    prior_to_mixed_tensor,
    sample_to_mixed_tensor,
)


def _fields() -> dict[str, np.ndarray]:
    shape = (4, 3, 2)
    return {
        "facies": (np.arange(np.prod(shape)).reshape(shape) % 7).astype(np.int16),
        "eic": np.full(shape, 0.25, dtype=np.float32),
        "temperature": np.full(shape, -2.0, dtype=np.float32),
        "unfrozen_water": np.full(shape, 0.12, dtype=np.float32),
        "resistivity": np.full(shape, 1000.0, dtype=np.float32),
    }


def test_mixed_codec_round_trip_shapes_and_loss() -> None:
    tensor = sample_to_mixed_tensor({"fields": _fields()})
    assert tensor.shape == (1, MIXED_CHANNELS, 4, 3, 2)
    loss, parts = mixed_reconstruction_loss(tensor, tensor)
    assert torch.isfinite(loss)
    assert {"cryofacies", "eic", "temperature", "unfrozen_water"}.issubset(parts)
    posterior = mixed_ensemble_to_posterior(tensor.repeat(3, 1, 1, 1, 1))
    assert posterior["cryofacies_mode"].shape == (4, 3, 2)
    assert posterior["eic_samples"].shape == (3, 4, 3, 2)


def test_prior_to_mixed_tensor_uses_log_resistivity_contract() -> None:
    fields = _fields()
    prior = {
        "facies": fields["facies"],
        "eic": fields["eic"],
        "temperature": fields["temperature"],
        "unfrozen_water": fields["unfrozen_water"],
        "log_resistivity": np.log(fields["resistivity"]),
    }
    encoded = prior_to_mixed_tensor(prior)
    expected = sample_to_mixed_tensor({"fields": fields})
    assert torch.allclose(encoded, expected)


def test_mixed_ablation_sampler_preserves_centered_anomalies() -> None:
    class DummyModel:
        def eval(self):
            return self

        def encode_context(self, raster, tokens, target_shape):
            return torch.zeros(
                raster.shape[0], 2, *target_shape, device=raster.device
            )

        def velocity_from_encoded(self, state, time, anchor, encoded_context):
            return torch.zeros_like(state)

    class IdentityAutoencoder:
        def decode(self, latent):
            return latent

    grid = {
        "x": np.arange(2, dtype=np.float32),
        "y": np.arange(2, dtype=np.float32),
        "z": np.arange(2, dtype=np.float32),
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }
    anchor = torch.zeros(1, MIXED_CHANNELS, 2, 2, 2)
    decoded, diagnostics = sample_mixed_ablation_ensemble(
        model=DummyModel(),
        autoencoder=IdentityAutoencoder(),
        anchor=anchor,
        raster=torch.zeros(1, 3, 2, 2, 2),
        tokens=torch.zeros(1, 1, 4),
        sample={
            "grid": grid,
            "observations": ObservationTable(
                coords=np.empty((0, 3), dtype=np.float32),
                type_ids=np.empty(0, dtype=np.int64),
                values=np.empty(0, dtype=np.float32),
                sigma=np.empty(0, dtype=np.float32),
                mask=np.empty(0, dtype=bool),
            ),
        },
        n_members=4,
        sampling_steps=2,
        guidance_strength=0.0,
        seed=3,
    )
    assert decoded.shape == (4, MIXED_CHANNELS, 2, 2, 2)
    assert diagnostics["anomaly_mean_abs"] < 1.0e-6
