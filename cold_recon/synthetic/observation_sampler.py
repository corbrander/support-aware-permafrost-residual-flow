from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, SUPPORT_TYPES, ObservationTable
from cold_recon.operators.support import (
    apply_surface_crossing,
    build_observation_operator,
    sample_profile_correlated_noise,
)


@dataclass
class SamplerConfig:
    n_boreholes: int = 8
    borehole_depth_step: int = 2
    n_ert_profiles: int = 2
    ert_x_step: int = 2
    ert_z_step: int = 2
    n_nmr_points: int = 180
    n_alt_points: int = 160
    eic_noise: float = 0.03
    temperature_noise: float = 0.20
    log_resistivity_noise: float = 0.10
    unfrozen_water_noise: float = 0.02
    alt_noise: float = 0.08
    borehole_interval_m: float = 0.50
    ert_cell_width_m: float = 4.0
    ert_cell_thickness_m: float = 0.50
    nmr_kernel_radius_m: float = 1.0
    noise_multiplier: float = 1.0
    ert_correlation_length_m: float = 6.0
    correlated_ert_noise: bool = True
    outlier_fraction: float = 0.0
    outlier_scale: float = 5.0
    source_bias_log_resistivity: float = 0.0


def sampler_config_from_dict(config: dict) -> SamplerConfig:
    s = config.get("synthetic", {})
    noise = s.get("noise", {})
    return SamplerConfig(
        n_boreholes=int(s.get("n_boreholes", 8)),
        borehole_depth_step=int(s.get("borehole_depth_step", 2)),
        n_ert_profiles=int(s.get("n_ert_profiles", 2)),
        ert_x_step=int(s.get("ert_x_step", 2)),
        ert_z_step=int(s.get("ert_z_step", 2)),
        n_nmr_points=int(s.get("n_nmr_points", 180)),
        n_alt_points=int(s.get("n_alt_points", 160)),
        eic_noise=float(noise.get("eic", 0.03)),
        temperature_noise=float(noise.get("temperature", 0.20)),
        log_resistivity_noise=float(noise.get("log_resistivity", 0.10)),
        unfrozen_water_noise=float(noise.get("unfrozen_water", 0.02)),
        alt_noise=float(noise.get("alt", 0.08)),
        borehole_interval_m=float(s.get("borehole_interval_m", 0.50)),
        ert_cell_width_m=float(s.get("ert_cell_width_m", 4.0)),
        ert_cell_thickness_m=float(s.get("ert_cell_thickness_m", 0.50)),
        nmr_kernel_radius_m=float(s.get("nmr_kernel_radius_m", 1.0)),
        noise_multiplier=float(noise.get("multiplier", 1.0)),
        ert_correlation_length_m=float(noise.get("ert_correlation_length_m", 6.0)),
        correlated_ert_noise=bool(noise.get("correlated_ert", True)),
        outlier_fraction=float(noise.get("outlier_fraction", 0.0)),
        outlier_scale=float(noise.get("outlier_scale", 5.0)),
        source_bias_log_resistivity=float(noise.get("source_bias_log_resistivity", 0.0)),
    )


def _append(
    records: list[dict],
    coord: tuple[float, float, float],
    type_name: str,
    sigma: float,
    *,
    support_type: str = "point",
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    group_id: int = -1,
    quality: float = 1.0,
) -> None:
    records.append(
        {
            "coord": coord,
            "type_id": OBS_TYPES[type_name],
            "value": 0.0,
            "sigma": float(sigma),
            "mask": True,
            "support_type_id": SUPPORT_TYPES[support_type],
            "extent": extent,
            "orientation": orientation,
            "quality": float(quality),
            "site_id": 0,
            "source_id": OBS_TYPES[type_name],
            "group_id": int(group_id),
        }
    )


def active_layer_thickness(temperature: np.ndarray, z: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    thawed = temperature > threshold
    alt = np.zeros(temperature.shape[:2], dtype=np.float32)
    for i in range(temperature.shape[0]):
        for j in range(temperature.shape[1]):
            idx = np.where(thawed[i, j])[0]
            alt[i, j] = float(z[idx[-1]]) if len(idx) else 0.0
    return alt


def sample_observations(
    sample_fields: dict[str, np.ndarray],
    grid: dict,
    rng: np.random.Generator,
    config: SamplerConfig,
) -> ObservationTable:
    x = grid["x"]
    y = grid["y"]
    z = grid["z"]
    nx, ny, nz = len(x), len(y), len(z)
    records: list[dict] = []
    facies = sample_fields["facies"]
    eic = sample_fields["eic"]
    temp = sample_fields["temperature"]
    theta = sample_fields["unfrozen_water"]
    rho = sample_fields["resistivity"]

    bore_xy = np.column_stack(
        [
            rng.integers(0, nx, size=config.n_boreholes),
            rng.integers(0, ny, size=config.n_boreholes),
        ]
    )
    for borehole_id, (ix, iy) in enumerate(bore_xy):
        for iz in range(0, nz, max(config.borehole_depth_step, 1)):
            coord = (float(x[ix]), float(y[iy]), float(z[iz]))
            _append(
                records,
                coord,
                "borehole_facies",
                0.05,
                support_type="borehole_interval",
                extent=(0.0, 0.0, float(config.borehole_interval_m)),
                orientation=(0.0, 0.0, 1.0),
                group_id=1000 + borehole_id,
            )
            _append(
                records,
                coord,
                "borehole_eic",
                config.eic_noise,
                support_type="borehole_interval",
                extent=(0.0, 0.0, float(config.borehole_interval_m)),
                orientation=(0.0, 0.0, 1.0),
                group_id=1000 + borehole_id,
            )
            _append(
                records,
                coord,
                "borehole_temperature",
                config.temperature_noise,
                support_type="point",
                orientation=(0.0, 0.0, 1.0),
                group_id=1000 + borehole_id,
            )

    line_ys = np.linspace(0.2 * (ny - 1), 0.8 * (ny - 1), config.n_ert_profiles).round().astype(int)
    for profile_id, iy in enumerate(line_ys):
        for ix in range(0, nx, max(config.ert_x_step, 1)):
            for iz in range(0, nz, max(config.ert_z_step, 1)):
                coord = (float(x[ix]), float(y[iy]), float(z[iz]))
                _append(
                    records,
                    coord,
                    "ert_log_resistivity",
                    config.log_resistivity_noise,
                    support_type="ert_volume",
                    extent=(
                        float(config.ert_cell_width_m),
                        max(float(grid["dy"]), 0.5 * float(config.ert_cell_width_m)),
                        float(config.ert_cell_thickness_m),
                    ),
                    orientation=(1.0, 0.0, 0.0),
                    group_id=2000 + profile_id,
                    quality=0.85,
                )

    for _ in range(config.n_nmr_points):
        ix = int(rng.integers(0, nx))
        iy = int(rng.integers(0, ny))
        iz = int(rng.integers(0, max(2, int(0.7 * nz))))
        coord = (float(x[ix]), float(y[iy]), float(z[iz]))
        radius = float(config.nmr_kernel_radius_m)
        _append(
            records,
            coord,
            "nmr_unfrozen_water",
            config.unfrozen_water_noise,
            support_type="nmr_kernel",
            extent=(2.0 * radius, 2.0 * radius, max(float(grid["dz"]), radius)),
            quality=0.90,
        )

    for _ in range(config.n_alt_points):
        ix = int(rng.integers(0, nx))
        iy = int(rng.integers(0, ny))
        _append(
            records,
            (float(x[ix]), float(y[iy]), 0.0),
            "alt",
            config.alt_noise,
            support_type="surface_crossing",
            quality=0.80,
        )

    observations = ObservationTable(
        coords=np.array([r["coord"] for r in records], dtype=np.float32),
        type_ids=np.array([r["type_id"] for r in records], dtype=np.int64),
        values=np.array([r["value"] for r in records], dtype=np.float32),
        sigma=np.array([r["sigma"] for r in records], dtype=np.float32)
        * float(config.noise_multiplier),
        mask=np.array([r["mask"] for r in records], dtype=bool),
        support_type_ids=np.array([r["support_type_id"] for r in records], dtype=np.int64),
        support_extent=np.array([r["extent"] for r in records], dtype=np.float32),
        orientation=np.array([r["orientation"] for r in records], dtype=np.float32),
        quality=np.array([r["quality"] for r in records], dtype=np.float32),
        site_ids=np.array([r["site_id"] for r in records], dtype=np.int64),
        source_ids=np.array([r["source_id"] for r in records], dtype=np.int64),
        group_ids=np.array([r["group_id"] for r in records], dtype=np.int64),
    )

    true_values = np.zeros(observations.n_obs, dtype=np.float64)
    continuous_fields = {
        OBS_TYPES["borehole_eic"]: eic,
        OBS_TYPES["borehole_temperature"]: temp,
        OBS_TYPES["ert_log_resistivity"]: np.log(np.maximum(rho, 1.0)),
        OBS_TYPES["nmr_unfrozen_water"]: theta,
    }
    for type_id, field in continuous_fields.items():
        indices = np.flatnonzero(observations.type_ids == type_id)
        operator = build_observation_operator(observations, grid, indices=indices)
        true_values[indices] = operator.apply(field)

    facies_indices = np.flatnonzero(observations.type_ids == OBS_TYPES["borehole_facies"])
    if len(facies_indices):
        operator = build_observation_operator(observations, grid, indices=facies_indices)
        n_classes = int(np.max(facies)) + 1
        one_hot = np.eye(n_classes, dtype=np.float32)[facies]
        true_values[facies_indices] = np.argmax(operator.apply_probabilities(one_hot), axis=1)

    alt_indices = np.flatnonzero(observations.type_ids == OBS_TYPES["alt"])
    if len(alt_indices):
        alt = apply_surface_crossing(temp, z)
        ix = np.asarray([int(np.argmin(np.abs(x - observations.coords[i, 0]))) for i in alt_indices])
        iy = np.asarray([int(np.argmin(np.abs(y - observations.coords[i, 1]))) for i in alt_indices])
        true_values[alt_indices] = alt[ix, iy]

    values = true_values.copy()
    continuous = observations.type_ids != OBS_TYPES["borehole_facies"]
    non_ert = continuous & (observations.type_ids != OBS_TYPES["ert_log_resistivity"])
    values[non_ert] += rng.normal(0.0, observations.sigma[non_ert])
    ert_indices = np.flatnonzero(observations.type_ids == OBS_TYPES["ert_log_resistivity"])
    if len(ert_indices):
        if bool(config.correlated_ert_noise):
            for group_id in np.unique(observations.group_ids[ert_indices]):
                group_indices = ert_indices[observations.group_ids[ert_indices] == group_id]
                values[group_indices] += sample_profile_correlated_noise(
                    observations,
                    group_indices,
                    rng,
                    length_scale=float(config.ert_correlation_length_m),
                )
        else:
            values[ert_indices] += rng.normal(0.0, observations.sigma[ert_indices])
        values[ert_indices] += float(config.source_bias_log_resistivity)

    outlier_fraction = float(np.clip(config.outlier_fraction, 0.0, 0.05))
    candidate = np.flatnonzero(continuous)
    n_outliers = int(np.rint(outlier_fraction * len(candidate)))
    if n_outliers > 0:
        selected = rng.choice(candidate, size=n_outliers, replace=False)
        values[selected] += rng.normal(
            0.0,
            float(config.outlier_scale) * np.maximum(observations.sigma[selected], 1.0e-6),
        )
        observations.quality[selected] = np.minimum(observations.quality[selected], 0.25)

    observations.values = values.astype(np.float32)
    return observations
