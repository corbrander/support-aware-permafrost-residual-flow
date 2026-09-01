from __future__ import annotations

import torch

from cold_recon.models.support_aware_residual_flow import SupportAwareResidualFlow3D


def test_support_aware_flow_shape_gradient_and_token_sensitivity() -> None:
    torch.manual_seed(3)
    model = SupportAwareResidualFlow3D(
        latent_channels=4,
        raster_in_channels=6,
        token_dim=20,
        context_channels=8,
        width=8,
        modes=(2, 2, 2),
        depth=1,
        attention_hidden=8,
        attention_chunk=16,
        support_extent_offset=12,
    )
    state = torch.randn(1, 4, 4, 4, 3, requires_grad=True)
    anchor = torch.randn_like(state)
    raster = torch.randn(1, 6, 16, 16, 12)
    tokens = torch.randn(1, 10, 20)
    tokens[..., :3] = torch.rand(1, 10, 3)
    tokens[..., 12:15] = torch.rand(1, 10, 3) * 0.2
    time = torch.tensor([20.0])
    first = model(state, time, anchor, raster, tokens)
    changed = tokens.clone()
    changed[..., 14] += 0.5
    second = model(state, time, anchor, raster, changed)
    assert first.shape == state.shape
    assert torch.isfinite(first).all()
    assert not torch.allclose(first, second)
    first.square().mean().backward()
    assert state.grad is not None


def test_raster_only_ablation_is_invariant_to_tokens() -> None:
    torch.manual_seed(7)
    model = SupportAwareResidualFlow3D(
        latent_channels=4,
        raster_in_channels=6,
        token_dim=20,
        context_channels=8,
        width=8,
        modes=(2, 2, 2),
        depth=1,
        attention_hidden=8,
        attention_chunk=16,
        support_extent_offset=12,
        use_token_conditioning=False,
    )
    state = torch.randn(1, 4, 4, 4, 3)
    anchor = torch.randn_like(state)
    raster = torch.randn(1, 6, 16, 16, 12)
    tokens = torch.randn(1, 10, 20)
    changed = tokens + 5.0
    time = torch.tensor([20.0])
    first = model(state, time, anchor, raster, tokens)
    second = model(state, time, anchor, raster, changed)
    assert torch.allclose(first, second)
