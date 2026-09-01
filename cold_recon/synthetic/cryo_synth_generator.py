from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cold_recon.data.data_schema import make_grid, save_sample_npz
from cold_recon.data.state_factorization import factorize_legacy_state
from cold_recon.synthetic.geophysics_forward import compute_resistivity, compute_unfrozen_water
from cold_recon.synthetic.ice_feature_generator import generate_ice_fields
from cold_recon.synthetic.observation_sampler import sample_observations, sampler_config_from_dict
from cold_recon.synthetic.stratigraphy_generator import generate_stratigraphy, generate_surface_features
from cold_recon.synthetic.thermal_forward import compute_temperature, compute_thermal_properties


def generate_synthetic_sample(config: dict[str, Any], seed: int | None = None, site_id: str = "synthetic") -> dict[str, Any]:
    if seed is None:
        seed = int(config.get("project", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    grid = make_grid(config)
    surface_features = generate_surface_features(grid, rng)
    facies, interfaces = generate_stratigraphy(grid, surface_features, rng)
    ice_content, eic = generate_ice_fields(facies, grid, interfaces, rng)
    temperature = compute_temperature(facies, grid, surface_features, interfaces)
    unfrozen_water = compute_unfrozen_water(temperature, facies)
    resistivity = compute_resistivity(facies, temperature, unfrozen_water, ice_content)
    thermal_conductivity, heat_capacity = compute_thermal_properties(facies, ice_content, unfrozen_water)
    fields = {
        "facies": facies.astype(np.int16),
        "ice_content": ice_content.astype(np.float32),
        "eic": eic.astype(np.float32),
        "temperature": temperature.astype(np.float32),
        "unfrozen_water": unfrozen_water.astype(np.float32),
        "resistivity": resistivity.astype(np.float32),
        "thermal_conductivity": thermal_conductivity.astype(np.float32),
        "heat_capacity": heat_capacity.astype(np.float32),
    }
    fields.update(factorize_legacy_state(facies, eic, temperature).as_fields())
    observations = sample_observations(fields, grid, rng, sampler_config_from_dict(config))
    return {
        "grid": grid,
        "fields": fields,
        "surface_features": surface_features,
        "observations": observations,
        "metadata": {
            "site_id": site_id,
            "source": "synthetic_cryo_generator",
            "synthetic": True,
            "seed": seed,
            "facies_names": {
                "0": "active_mineral",
                "1": "peat",
                "2": "silt",
                "3": "ice_rich_silt",
                "4": "sand_gravel",
                "5": "talik",
                "6": "wedge_ice",
            },
            "state_factorization": {
                "lithology": {"0": "peat", "1": "silt", "2": "sand_gravel", "3": "unspecified_mineral"},
                "thermal_state": {"0": "active_or_thawed", "1": "frozen", "2": "talik_or_near_thaw"},
                "ice_structure": {"0": "matrix_ice", "1": "lens_rich", "2": "massive_or_wedge_ice"},
            },
        },
    }


def save_synthetic_sample(path: str | Path, sample: dict[str, Any]) -> None:
    save_sample_npz(path, sample)
