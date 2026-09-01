from __future__ import annotations

import torch

from cold_recon.models.cold_recon_model import COLDReconImplicitModel
from cold_recon.models.observation_tokenizer import ObservationGraphBuilder, build_observation_attention_mask
from cold_recon.models.observation_tokenizer import ObservationTokenizer
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.synthetic.cryo_synth_generator import generate_synthetic_sample


def test_observation_tokenizer_dimension() -> None:
    config = {
        "project": {"seed": 2},
        "grid": {"nx": 8, "ny": 8, "nz": 6, "dx": 1.0, "dy": 1.0, "dz": 0.5, "crs": "test"},
        "synthetic": {"n_boreholes": 2, "n_nmr_points": 5, "n_alt_points": 5, "n_ert_profiles": 1},
    }
    sample = generate_synthetic_sample(config, seed=2)
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    tokens = tokenizer.encode_numpy(sample["observations"])
    assert tokens.shape[0] == sample["observations"].n_obs
    assert tokens.shape[1] == 16


def test_observation_graph_builder_and_transformer_mask() -> None:
    config = {
        "project": {"seed": 3},
        "grid": {"nx": 8, "ny": 8, "nz": 6, "dx": 1.0, "dy": 1.0, "dz": 0.5, "crs": "test"},
        "synthetic": {"n_boreholes": 2, "n_nmr_points": 5, "n_alt_points": 5, "n_ert_profiles": 1},
    }
    sample = generate_synthetic_sample(config, seed=3)
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    tokens = tokenizer.encode_numpy(sample["observations"])
    graph = ObservationGraphBuilder(k_neighbors=4).fit_from_grid(sample["grid"]).build(sample["observations"])
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_weight.shape[0] == graph.edge_index.shape[1]
    assert graph.attention_mask.shape == (sample["observations"].n_obs, sample["observations"].n_obs)
    assert not graph.attention_mask.diagonal().any()
    assert torch.isfinite(torch.from_numpy(graph.edge_weight)).all()

    encoder = ObsTransformerEncoder(token_dim=tokens.shape[1], hidden_dim=16, num_layers=1, num_heads=4)
    pooled = encoder(
        torch.from_numpy(tokens).unsqueeze(0),
        torch.zeros((1, tokens.shape[0]), dtype=torch.bool),
        torch.from_numpy(graph.attention_mask),
    )
    assert pooled.shape == (1, 16)


def test_configured_observation_attention_mask_and_model_forward() -> None:
    config = {
        "project": {"seed": 4},
        "grid": {"nx": 6, "ny": 6, "nz": 5, "dx": 1.0, "dy": 1.0, "dz": 0.5, "crs": "test"},
        "synthetic": {"n_boreholes": 2, "n_nmr_points": 3, "n_alt_points": 3, "n_ert_profiles": 1},
        "observation_graph": {"enabled": True, "k_neighbors": 3, "length_scale_xyz": [0.2, 0.2, 0.3]},
    }
    sample = generate_synthetic_sample(config, seed=4)
    attention_mask = build_observation_attention_mask(config, sample["grid"], sample["observations"])
    assert attention_mask is not None
    assert attention_mask.dtype == torch.bool

    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    obs_tokens = tokenizer.encode_torch(sample["observations"]).unsqueeze(0)
    model = COLDReconImplicitModel(
        token_dim=tokenizer.token_dim,
        surface_feature_dim=8,
        obs_hidden_dim=16,
        obs_layers=1,
        obs_heads=4,
        surface_hidden_dim=8,
        fourier_features=4,
        implicit_hidden_dim=16,
        implicit_layers=2,
        n_facies=7,
    )
    coords = torch.rand(1, 5, 3)
    surface = torch.rand(1, 5, 8)
    padding = torch.zeros((1, obs_tokens.shape[1]), dtype=torch.bool)
    out = model(coords, surface, obs_tokens, padding, attention_mask)
    assert out["facies_logits"].shape == (1, 5, 7)
