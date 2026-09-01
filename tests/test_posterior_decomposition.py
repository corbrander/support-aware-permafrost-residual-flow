from __future__ import annotations

import torch

from cold_recon.models.high_eic_head import (
    HighEICEventHead3D,
    calibrated_event_loss,
    focal_tversky_loss,
    monotone_event_probabilities,
)
from cold_recon.models.posterior_decomposition import (
    compose_safe_ensemble,
    noninferiority_decision,
)


def test_safe_ensemble_centers_anomalies_and_falls_back_under_ood() -> None:
    anchor = torch.full((1, 2, 3, 3, 2), 0.4)
    bias = torch.full_like(anchor, 0.2)
    anomalies = torch.randn(8, 2, 3, 3, 2)
    gate = torch.ones_like(anchor)
    scale = torch.full_like(anchor, 0.5)
    ensemble, diagnostics = compose_safe_ensemble(
        anchor,
        bias,
        anomalies,
        gate,
        scale,
        ood_score=torch.ones_like(anchor),
    )
    torch.testing.assert_close(ensemble.mean(dim=0, keepdim=True), anchor, atol=1e-6, rtol=1e-6)
    assert diagnostics["centered_anomaly_mean_abs"].item() < 1e-6


def test_noninferiority_decision_disables_harmful_bias() -> None:
    assert noninferiority_decision(0.10, 0.105, 0.01).allow_bias
    assert not noninferiority_decision(0.10, 0.12, 0.01).allow_bias


def test_event_probabilities_are_monotone_and_loss_is_differentiable() -> None:
    logits = torch.randn(2, 3, 4, 4, 3, requires_grad=True)
    probabilities = monotone_event_probabilities(logits)
    assert torch.all(probabilities[:, 1] <= probabilities[:, 0])
    assert torch.all(probabilities[:, 2] <= probabilities[:, 1])
    targets = torch.zeros_like(logits)
    targets[:, :, 1:3, 1:3, 1:] = 1.0
    loss = focal_tversky_loss(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_calibrated_event_loss_penalizes_confident_false_positives() -> None:
    targets = torch.zeros(1, 3, 2, 2, 2)
    good, good_parts = calibrated_event_loss(torch.full_like(targets, -4.0), targets)
    bad, bad_parts = calibrated_event_loss(torch.full_like(targets, 4.0), targets)
    assert bad > good
    assert bad_parts["bce"] > good_parts["bce"]


def test_event_head_refines_latent_logits_with_full_resolution_raster() -> None:
    head = HighEICEventHead3D(6, raster_channels=5, width=6)
    latent = torch.randn(2, 6, 2, 3, 2, requires_grad=True)
    raster = torch.randn(2, 5, 8, 9, 6)

    logits = head(latent, raster=raster)

    assert logits.shape == (2, 3, 8, 9, 6)
    logits.mean().backward()
    assert latent.grad is not None
