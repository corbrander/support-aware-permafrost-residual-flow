from __future__ import annotations

from cold_recon.synthetic.cryo_synth_generator import generate_synthetic_sample


def test_synthetic_generator_small() -> None:
    config = {
        "project": {"seed": 1},
        "grid": {"nx": 16, "ny": 12, "nz": 10, "dx": 2.0, "dy": 2.0, "dz": 0.25, "crs": "test"},
        "synthetic": {"n_boreholes": 3, "n_nmr_points": 10, "n_alt_points": 8, "n_ert_profiles": 1},
    }
    sample = generate_synthetic_sample(config, seed=1)
    assert sample["fields"]["facies"].shape == (16, 12, 10)
    assert sample["fields"]["eic"].shape == (16, 12, 10)
    assert sample["fields"]["temperature"].shape == (16, 12, 10)
    assert sample["observations"].n_obs > 0

