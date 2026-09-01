from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cold_recon.data.state_factorization import factorize_legacy_state
from cold_recon.synthetic.cryo_synth_generator import generate_synthetic_sample
from cold_recon.synthetic.geophysics_forward import compute_resistivity, compute_unfrozen_water
from cold_recon.synthetic.observation_sampler import sample_observations, sampler_config_from_dict
from cold_recon.synthetic.thermal_forward import compute_thermal_properties


ID_GENERATOR_FAMILIES: tuple[str, ...] = (
    "horizontal_layered",
    "inclined_deformed",
    "wedge_network",
    "isolated_lenses",
    "talik",
    "fluvial_thermokarst",
    "mixed_sediment",
    "variable_active_layer",
    "combined",
)

OOD_GENERATOR_FAMILIES: tuple[str, ...] = (
    "abrupt_boundary",
    "altered_eic_coupling",
    "saline_low_resistivity_ice",
)

ALL_GENERATOR_FAMILIES = ID_GENERATOR_FAMILIES + OOD_GENERATOR_FAMILIES


@dataclass(frozen=True)
class ConstitutiveParameters:
    eic_scale: float
    temperature_offset_c: float
    vertical_gradient_scale: float
    resistivity_scale: float
    salinity_factor: float
    noise_multiplier: float
    outlier_fraction: float
    source_bias_log_resistivity: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in vars(self).items()}


def sample_constitutive_parameters(
    rng: np.random.Generator,
    family: str,
) -> ConstitutiveParameters:
    eic_scale = rng.uniform(0.75, 1.25)
    resistivity_scale = np.exp(rng.uniform(np.log(0.65), np.log(1.65)))
    salinity_factor = rng.uniform(0.75, 1.35)
    if family == "altered_eic_coupling":
        eic_scale = rng.uniform(0.85, 1.25)
    if family == "saline_low_resistivity_ice":
        resistivity_scale = rng.uniform(0.18, 0.45)
        salinity_factor = rng.uniform(1.8, 3.0)
    return ConstitutiveParameters(
        eic_scale=float(eic_scale),
        temperature_offset_c=float(rng.uniform(-1.25, 1.25)),
        vertical_gradient_scale=float(rng.uniform(0.75, 1.35)),
        resistivity_scale=float(resistivity_scale),
        salinity_factor=float(salinity_factor),
        noise_multiplier=float(rng.choice([0.5, 1.0, 2.0, 4.0])),
        outlier_fraction=float(rng.uniform(0.01, 0.05)),
        source_bias_log_resistivity=float(rng.normal(0.0, 0.08)),
    )


def _ellipsoid_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    ix, iy, iz = np.indices(shape, dtype=np.float32)
    return (
        ((ix - center[0]) / max(radii[0], 1.0)) ** 2
        + ((iy - center[1]) / max(radii[1], 1.0)) ** 2
        + ((iz - center[2]) / max(radii[2], 1.0)) ** 2
        <= 1.0
    )


def _warp_vertical(volume: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    out = np.empty_like(volume)
    nz = volume.shape[2]
    base = np.arange(nz)
    for ix in range(volume.shape[0]):
        for iy in range(volume.shape[1]):
            source = np.clip(base - int(offsets[ix, iy]), 0, nz - 1)
            out[ix, iy] = volume[ix, iy, source]
    return out


def _inclined_deformed(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    nx, ny, _ = fields["facies"].shape
    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny), indexing="ij")
    offsets = np.rint(3.5 * xx + 1.5 * np.sin(np.pi * yy + rng.uniform(-np.pi, np.pi))).astype(int)
    for name in ("facies", "eic", "ice_content", "temperature"):
        fields[name] = _warp_vertical(fields[name], offsets)


def _wedge_network(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    nx, ny, nz = fields["facies"].shape
    xx, yy, zz = np.indices((nx, ny, nz), dtype=np.float32)
    for _ in range(int(rng.integers(3, 7))):
        x0 = rng.uniform(0.1 * nx, 0.9 * nx)
        slope = rng.uniform(-0.45, 0.45)
        depth = rng.uniform(0.25 * nz, 0.55 * nz)
        distance = np.abs((xx - x0) - slope * (yy - 0.5 * ny))
        width = np.maximum(0.75, 2.5 * (1.0 - zz / depth))
        mask = (zz <= depth) & (distance <= width) & (fields["facies"] != 5)
        fields["facies"][mask] = 6
        fields["eic"][mask] = np.maximum(fields["eic"][mask], rng.uniform(0.50, 0.72))
        fields["ice_content"][mask] = np.maximum(fields["ice_content"][mask], 0.75)


def _isolated_lenses(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    shape = fields["facies"].shape
    nx, ny, nz = shape
    for _ in range(int(rng.integers(8, 18))):
        mask = _ellipsoid_mask(
            shape,
            (rng.uniform(0.05 * nx, 0.95 * nx), rng.uniform(0.05 * ny, 0.95 * ny), rng.uniform(0.15 * nz, 0.8 * nz)),
            (rng.uniform(3.0, 9.0), rng.uniform(3.0, 9.0), rng.uniform(2.0, 4.0)),
        )
        mask &= fields["facies"] != 5
        boost = rng.uniform(0.22, 0.48)
        fields["eic"][mask] = np.maximum(fields["eic"][mask], boost)
        fields["ice_content"][mask] = np.maximum(fields["ice_content"][mask], boost + 0.18)


def _talik(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    shape = fields["facies"].shape
    nx, ny, nz = shape
    mask = _ellipsoid_mask(
        shape,
        (rng.uniform(0.25 * nx, 0.75 * nx), rng.uniform(0.25 * ny, 0.75 * ny), rng.uniform(0.25 * nz, 0.45 * nz)),
        (rng.uniform(0.12 * nx, 0.25 * nx), rng.uniform(0.12 * ny, 0.25 * ny), rng.uniform(0.18 * nz, 0.35 * nz)),
    )
    fields["facies"][mask] = 5
    fields["temperature"][mask] = rng.uniform(0.15, 1.25)
    fields["eic"][mask] *= rng.uniform(0.0, 0.15)
    fields["ice_content"][mask] *= rng.uniform(0.0, 0.20)


def _fluvial_thermokarst(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    nx, ny, nz = fields["facies"].shape
    xx, yy, zz = np.indices((nx, ny, nz), dtype=np.float32)
    center = 0.5 * ny + 0.18 * ny * np.sin(2.0 * np.pi * xx / max(nx - 1, 1) + rng.uniform(-np.pi, np.pi))
    width = rng.uniform(0.06 * ny, 0.14 * ny)
    depth = rng.uniform(0.18 * nz, 0.38 * nz)
    channel = (np.abs(yy - center) <= width) & (zz <= depth)
    fields["facies"][channel] = 4
    fields["eic"][channel] *= 0.35
    pond = channel & (zz <= 0.65 * depth)
    fields["temperature"][pond] = np.maximum(fields["temperature"][pond], rng.uniform(-0.1, 0.8))


def _mixed_sediment(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    shape = fields["facies"].shape
    nx, ny, nz = shape
    for _ in range(int(rng.integers(8, 16))):
        mask = _ellipsoid_mask(
            shape,
            (rng.uniform(0, nx), rng.uniform(0, ny), rng.uniform(0.05 * nz, 0.95 * nz)),
            (rng.uniform(3, 10), rng.uniform(3, 10), rng.uniform(2, 8)),
        )
        fields["facies"][mask] = int(rng.choice([1, 2, 4], p=[0.15, 0.55, 0.30]))


def _variable_active_layer(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    nx, ny, nz = fields["facies"].shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    depth = 2.0 + 3.0 * (0.5 + 0.5 * np.sin(2.0 * np.pi * xx / max(nx, 1) + 0.7 * np.cos(2.0 * np.pi * yy / max(ny, 1))))
    depth += rng.normal(0.0, 0.35, size=depth.shape)
    zz = np.arange(nz)[None, None, :]
    shallow = zz <= depth[:, :, None]
    fields["temperature"][shallow] = np.maximum(fields["temperature"][shallow], 0.3)
    fields["temperature"][~shallow] = np.minimum(fields["temperature"][~shallow], -0.2)


def _abrupt_boundary(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    nx = fields["facies"].shape[0]
    split = int(rng.integers(max(1, nx // 3), max(2, 2 * nx // 3)))
    shift = int(rng.choice([-6, -5, 5, 6]))
    for name in ("facies", "eic", "ice_content", "temperature"):
        fields[name][split:] = np.roll(fields[name][split:], shift=shift, axis=2)


def _altered_eic_coupling(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    eligible = np.isin(fields["facies"], [1, 2, 4])
    selector = eligible & (rng.random(fields["facies"].shape) < 0.12)
    fields["eic"][selector] = rng.uniform(0.32, 0.65, size=int(selector.sum()))
    fields["ice_content"][selector] = np.maximum(fields["ice_content"][selector], fields["eic"][selector] + 0.16)
    inherited = (fields["facies"] == 3) & (rng.random(fields["facies"].shape) < 0.55)
    fields["eic"][inherited] *= rng.uniform(0.20, 0.55)


def _saline_low_resistivity_ice(fields: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    _isolated_lenses(fields, rng)


TRANSFORMS: dict[str, Callable[[dict[str, np.ndarray], np.random.Generator], None]] = {
    "horizontal_layered": lambda fields, rng: None,
    "inclined_deformed": _inclined_deformed,
    "wedge_network": _wedge_network,
    "isolated_lenses": _isolated_lenses,
    "talik": _talik,
    "fluvial_thermokarst": _fluvial_thermokarst,
    "mixed_sediment": _mixed_sediment,
    "variable_active_layer": _variable_active_layer,
    "abrupt_boundary": _abrupt_boundary,
    "altered_eic_coupling": _altered_eic_coupling,
    "saline_low_resistivity_ice": _saline_low_resistivity_ice,
}


def _apply_combined(fields: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[str, ...]:
    candidates = (
        "inclined_deformed",
        "wedge_network",
        "isolated_lenses",
        "talik",
        "fluvial_thermokarst",
        "mixed_sediment",
        "variable_active_layer",
    )
    selected = tuple(str(name) for name in rng.choice(candidates, size=3, replace=False))
    for name in selected:
        TRANSFORMS[name](fields, rng)
    return selected


def generate_multifamily_sample(
    config: dict[str, Any],
    *,
    seed: int,
    family: str,
    site_id: str = "synthetic",
    scene_id: str | None = None,
) -> dict[str, Any]:
    if family not in ALL_GENERATOR_FAMILIES:
        raise ValueError(f"unknown generator family: {family}")
    rng = np.random.default_rng(int(seed))
    local_config = copy.deepcopy(config)
    parameters = sample_constitutive_parameters(rng, family)
    synthetic = local_config.setdefault("synthetic", {})
    noise = synthetic.setdefault("noise", {})
    noise["multiplier"] = parameters.noise_multiplier
    noise["outlier_fraction"] = parameters.outlier_fraction
    noise["source_bias_log_resistivity"] = parameters.source_bias_log_resistivity
    noise["correlated_ert"] = True
    noise["ert_correlation_length_m"] = float(rng.uniform(3.0, 10.0))
    synthetic["n_boreholes"] = int(rng.integers(3, max(4, int(synthetic.get("n_boreholes", 8))) + 1))
    synthetic["n_ert_profiles"] = int(rng.integers(1, max(2, int(synthetic.get("n_ert_profiles", 2))) + 1))

    sample = generate_synthetic_sample(local_config, seed=int(seed), site_id=site_id)
    fields = {name: np.array(value, copy=True) for name, value in sample["fields"].items()}
    components: tuple[str, ...] = ()
    if family == "combined":
        components = _apply_combined(fields, rng)
    else:
        TRANSFORMS[family](fields, rng)

    eic = np.clip(fields["eic"] * parameters.eic_scale, 0.0, 0.85).astype(np.float32)
    ice_content = np.clip(np.maximum(fields["ice_content"], eic + 0.12), 0.0, 0.96).astype(np.float32)
    z_norm = np.linspace(0.0, 1.0, fields["temperature"].shape[2], dtype=np.float32)[None, None, :]
    temperature = fields["temperature"] * parameters.vertical_gradient_scale
    temperature = temperature + parameters.temperature_offset_c * (0.35 + 0.65 * z_norm)
    if family == "talik":
        temperature[fields["facies"] == 5] = np.maximum(temperature[fields["facies"] == 5], 0.15)
    unfrozen_water = compute_unfrozen_water(temperature, fields["facies"])
    resistivity = compute_resistivity(fields["facies"], temperature, unfrozen_water, ice_content)
    resistivity = resistivity * parameters.resistivity_scale / parameters.salinity_factor
    thermal_conductivity, heat_capacity = compute_thermal_properties(
        fields["facies"], ice_content, unfrozen_water
    )

    fields.update(
        {
            "ice_content": ice_content,
            "eic": eic,
            "temperature": temperature.astype(np.float32),
            "unfrozen_water": unfrozen_water.astype(np.float32),
            "resistivity": np.clip(resistivity, 1.0, None).astype(np.float32),
            "thermal_conductivity": thermal_conductivity.astype(np.float32),
            "heat_capacity": heat_capacity.astype(np.float32),
        }
    )
    fields.update(
        factorize_legacy_state(fields["facies"], fields["eic"], fields["temperature"]).as_fields()
    )
    sample["fields"] = fields
    sample["observations"] = sample_observations(
        fields, sample["grid"], rng, sampler_config_from_dict(local_config)
    )
    sample["metadata"].update(
        {
            "scene_id": scene_id or f"{family}_{int(seed):08d}",
            "generator_family": family,
            "combined_components": list(components),
            "constitutive_parameters": parameters.as_dict(),
            "observation_design": {
                "n_boreholes": int(synthetic["n_boreholes"]),
                "n_ert_profiles": int(synthetic["n_ert_profiles"]),
                "correlated_ert_noise": True,
            },
            "minimum_lens_vertical_radius_voxels": 2.0,
        }
    )
    return sample
