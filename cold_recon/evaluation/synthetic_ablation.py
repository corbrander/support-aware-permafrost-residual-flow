from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from cold_recon.baselines.idw import idw_interpolate, reconstruct_idw
from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.synthetic.geophysics_forward import compute_resistivity, compute_unfrozen_water


@dataclass(frozen=True)
class AblationScenario:
    name: str
    geophysics: str
    include_alt: bool = True
    use_physics: bool = True


DEFAULT_SCENARIOS = (
    AblationScenario("no_geophysics", "none", include_alt=True, use_physics=True),
    AblationScenario("ert_only", "ert", include_alt=True, use_physics=True),
    AblationScenario("nmr_only", "nmr", include_alt=True, use_physics=True),
    AblationScenario("ert_nmr", "ert_nmr", include_alt=True, use_physics=True),
    AblationScenario("no_physics", "ert_nmr", include_alt=True, use_physics=False),
    AblationScenario("no_alt", "ert_nmr", include_alt=False, use_physics=True),
)


def _xy_keys(coords: np.ndarray) -> np.ndarray:
    return np.array([f"{x:.4f}_{y:.4f}" for x, y in coords[:, :2]], dtype=object)


def subset_synthetic_observations(
    observations: ObservationTable,
    n_boreholes: int,
    scenario: AblationScenario,
    seed: int = 0,
) -> ObservationTable:
    rng = np.random.default_rng(seed)
    borehole_type_ids = {
        OBS_TYPES["borehole_facies"],
        OBS_TYPES["borehole_eic"],
        OBS_TYPES["borehole_temperature"],
    }
    borehole_mask = np.isin(observations.type_ids, list(borehole_type_ids))
    borehole_keys = np.unique(_xy_keys(observations.coords[borehole_mask]))
    rng.shuffle(borehole_keys)
    selected_keys = set(borehole_keys[: min(n_boreholes, len(borehole_keys))].tolist())
    obs_keys = _xy_keys(observations.coords)
    keep = borehole_mask & np.array([key in selected_keys for key in obs_keys], dtype=bool)

    if scenario.geophysics in {"ert", "ert_nmr"}:
        keep |= observations.type_ids == OBS_TYPES["ert_log_resistivity"]
    if scenario.geophysics in {"nmr", "ert_nmr"}:
        keep |= observations.type_ids == OBS_TYPES["nmr_unfrozen_water"]
    if scenario.include_alt:
        keep |= observations.type_ids == OBS_TYPES["alt"]
    return observations.subset(np.where(keep)[0])


def _grid_points(grid: dict) -> np.ndarray:
    xx, yy, zz = np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def _facies_from_observations(observations: ObservationTable, query: np.ndarray, shape: tuple[int, int, int], n_facies: int) -> np.ndarray:
    mask = observations.type_ids == OBS_TYPES["borehole_facies"]
    if not np.any(mask):
        return np.zeros(shape, dtype=np.int16)
    coords = observations.coords[mask]
    values = observations.values[mask].astype(np.int64)
    tree = cKDTree(coords)
    k = min(7, len(coords))
    dist, idx = tree.query(query, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.power(dist + 1e-6, 2.0)
    votes = np.zeros((query.shape[0], n_facies), dtype=np.float32)
    for j in range(k):
        votes[np.arange(query.shape[0]), np.clip(values[idx[:, j]], 0, n_facies - 1)] += weights[:, j]
    return np.argmax(votes, axis=1).reshape(shape).astype(np.int16)


def _alt_temperature_constraint(
    temperature: np.ndarray,
    observations: ObservationTable,
    grid: dict,
) -> np.ndarray:
    alt_mask = observations.type_ids == OBS_TYPES["alt"]
    if not np.any(alt_mask):
        return temperature
    xx, yy = np.meshgrid(grid["x"], grid["y"], indexing="ij")
    surface = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float32)]).astype(np.float32)
    alt = idw_interpolate(observations.coords[alt_mask], observations.values[alt_mask], surface, k=min(8, int(np.sum(alt_mask))))
    alt = alt.reshape(len(grid["x"]), len(grid["y"]))
    z = grid["z"][None, None, :]
    alt3 = np.maximum(alt[:, :, None], 0.1)
    active_temp = 0.4 * (1.0 - z / alt3)
    frozen_temp = -0.15 - 0.30 * (z - alt3)
    constrained = temperature.copy()
    constrained = np.where(z <= alt3, np.maximum(constrained, active_temp), constrained)
    constrained = np.where(z > alt3, np.minimum(constrained, frozen_temp), constrained)
    return constrained.astype(np.float32)


def reconstruct_physics_fusion(
    sample: dict,
    observations: ObservationTable,
    scenario: AblationScenario,
    n_facies: int = 7,
) -> dict[str, np.ndarray]:
    grid = sample["grid"]
    query = _grid_points(grid)
    shape = (len(grid["x"]), len(grid["y"]), len(grid["z"]))
    idw = reconstruct_idw(observations, grid, n_facies=n_facies)
    facies = idw.get("facies")
    if facies is None:
        facies = _facies_from_observations(observations, query, shape, n_facies)
    eic = idw.get("eic")
    if eic is None:
        eic = np.zeros(shape, dtype=np.float32)
    temperature = idw.get("temperature")
    if temperature is None:
        temperature = -0.3 - 0.25 * grid["z"][None, None, :]
        temperature = np.broadcast_to(temperature, shape).astype(np.float32)

    if scenario.use_physics:
        temperature = _alt_temperature_constraint(temperature, observations, grid)

    theta = idw.get("unfrozen_water")
    if theta is None or scenario.use_physics:
        theta_phys = compute_unfrozen_water(temperature, facies)
        if theta is None:
            theta = theta_phys
        else:
            theta = 0.55 * theta.astype(np.float32) + 0.45 * theta_phys
    theta = np.clip(theta.astype(np.float32), 0.0, 0.9)

    log_rho = idw.get("log_resistivity")
    if log_rho is None or scenario.use_physics:
        ice_content = np.clip(eic + 0.18, 0.0, 0.95)
        rho_phys = compute_resistivity(facies, temperature, theta, ice_content)
        log_rho_phys = np.log(np.maximum(rho_phys, 1.0)).astype(np.float32)
        if log_rho is None:
            log_rho = log_rho_phys
        else:
            log_rho = 0.70 * log_rho.astype(np.float32) + 0.30 * log_rho_phys
    pred = {
        "facies": facies.astype(np.int16),
        "eic": np.clip(eic.astype(np.float32), 0.0, 0.75),
        "temperature": temperature.astype(np.float32),
        "unfrozen_water": theta,
        "log_resistivity": log_rho.astype(np.float32),
    }
    return pred


def run_synthetic_ablation(
    sample: dict,
    borehole_counts: list[int],
    scenarios: tuple[AblationScenario, ...] = DEFAULT_SCENARIOS,
    seed: int = 0,
    n_facies: int = 7,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for n_bh in borehole_counts:
        for scenario in scenarios:
            obs = subset_synthetic_observations(sample["observations"], n_bh, scenario, seed=seed + n_bh)
            pred = reconstruct_physics_fusion(sample, obs, scenario, n_facies=n_facies)
            metrics = synthetic_metrics(pred, sample["fields"], sample["grid"]["z"], n_facies=n_facies)
            rows.append(
                {
                    "model": "COLDReconPhysicsFusion",
                    "scenario": scenario.name,
                    "n_boreholes": int(n_bh),
                    "n_observations": int(obs.n_obs),
                    **metrics,
                }
            )
    return rows

