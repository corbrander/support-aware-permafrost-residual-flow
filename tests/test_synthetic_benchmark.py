from __future__ import annotations

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.evaluation.synthetic_benchmark import aggregate_synthetic_benchmark, summarize_synthetic_sample


def _sample(seed: int = 1) -> dict:
    shape = (3, 3, 4)
    z = np.arange(shape[2], dtype=np.float32) * 0.5
    facies = np.zeros(shape, dtype=np.int16)
    facies[:, :, 2:] = 3
    eic = np.where(facies == 3, 0.4, 0.05).astype(np.float32)
    temp = np.broadcast_to(np.array([0.5, 0.1, -0.5, -1.0], dtype=np.float32), shape).copy()
    obs = ObservationTable(
        coords=np.zeros((4, 3), dtype=np.float32),
        type_ids=np.array(
            [
                OBS_TYPES["borehole_eic"],
                OBS_TYPES["borehole_temperature"],
                OBS_TYPES["ert_log_resistivity"],
                OBS_TYPES["alt"],
            ]
        ),
        values=np.ones(4, dtype=np.float32),
        sigma=np.ones(4, dtype=np.float32),
        mask=np.ones(4, dtype=bool),
    )
    return {
        "grid": {"x": np.arange(3), "y": np.arange(3), "z": z, "dx": 1.0, "dy": 1.0, "dz": 0.5},
        "fields": {
            "facies": facies,
            "ice_content": eic,
            "eic": eic,
            "temperature": temp,
            "unfrozen_water": np.full(shape, 0.1, dtype=np.float32),
            "resistivity": np.full(shape, 100.0, dtype=np.float32),
            "thermal_conductivity": np.ones(shape, dtype=np.float32),
            "heat_capacity": np.ones(shape, dtype=np.float32),
        },
        "surface_features": {},
        "observations": obs,
        "metadata": {"seed": seed, "site_id": f"s{seed}"},
    }


def test_summarize_synthetic_sample_contains_distribution_metrics() -> None:
    row = summarize_synthetic_sample(_sample(), sample_id="tiny")
    assert row["sample_id"] == "tiny"
    assert row["n_observations"] == 4.0
    assert row["facies_fraction_ice_rich_silt"] > 0.0
    assert row["obs_count_alt"] == 1.0
    assert row["ice_rich_fraction"] > 0.0


def test_aggregate_synthetic_benchmark_returns_mean_std() -> None:
    import pandas as pd

    rows = pd.DataFrame([summarize_synthetic_sample(_sample(1), "a"), summarize_synthetic_sample(_sample(2), "b")])
    summary = aggregate_synthetic_benchmark(rows)
    assert {"metric", "mean", "std", "min", "max", "n"}.issubset(summary.columns)
    assert "eic_mean" in set(summary["metric"])
