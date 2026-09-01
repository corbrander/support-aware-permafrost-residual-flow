from __future__ import annotations

import torch

from cold_recon.training.probabilistic_constitutive import probabilistic_constitutive_loss


def test_probabilistic_constitutive_loss_is_finite_and_differentiable() -> None:
    decoded = torch.randn(2, 14, 5, 4, 3, requires_grad=True)
    loss, parts = probabilistic_constitutive_loss(decoded)
    loss.backward()
    assert torch.isfinite(loss)
    assert decoded.grad is not None
    assert {"eic_relation", "unfrozen_relation", "resistivity_relation", "thermal_semantic"}.issubset(parts)
