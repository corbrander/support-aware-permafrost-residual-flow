from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from cold_recon.data.data_schema import ObservationTable
from cold_recon.evaluation.uncertainty import facies_entropy
from cold_recon.physics.settlement import settlement_potential_numpy


@dataclass(frozen=True)
class VOIWeights:
    uncertainty: float = 0.35
    ice_rich_ambiguity: float = 0.20
    settlement_risk: float = 0.25
    differential_settlement: float = 0.10
    novelty: float = 0.10


def _robust_normalize(values: np.ndarray, lower_q: float = 5.0, upper_q: float = 95.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.nanpercentile(arr[finite], lower_q))
    high = float(np.nanpercentile(arr[finite], upper_q))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - low) / (high - low)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _upper_mask(z: np.ndarray, max_depth: float | None) -> np.ndarray:
    if max_depth is None:
        return np.ones(len(z), dtype=bool)
    mask = z <= float(max_depth)
    if not np.any(mask):
        mask[0] = True
    return mask


def _surface_mean(volume: np.ndarray, z: np.ndarray, max_depth: float | None) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError("Expected a 2D surface or 3D volume")
    return np.nanmean(arr[:, :, _upper_mask(z, max_depth)], axis=2).astype(np.float32)


def _surface_max(volume: np.ndarray, z: np.ndarray, max_depth: float | None) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError("Expected a 2D surface or 3D volume")
    return np.nanmax(arr[:, :, _upper_mask(z, max_depth)], axis=2).astype(np.float32)


def _settlement_from_posterior(posterior: dict[str, np.ndarray]) -> np.ndarray:
    if "settlement_potential" in posterior:
        return posterior["settlement_potential"].astype(np.float32)
    if "eic_mean" not in posterior or "temperature_mean" not in posterior:
        grid_shape = (len(posterior["grid_x"]), len(posterior["grid_y"]))
        return np.zeros(grid_shape, dtype=np.float32)
    dz = float(np.mean(np.diff(posterior["grid_z"]))) if len(posterior["grid_z"]) > 1 else 1.0
    return settlement_potential_numpy(posterior["eic_mean"], posterior["temperature_mean"] + 2.0, dz)


def _differential_settlement(settlement: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if settlement.shape[0] < 2 or settlement.shape[1] < 2:
        return np.zeros_like(settlement, dtype=np.float32)
    gx, gy = np.gradient(settlement.astype(np.float32), x.astype(np.float32), y.astype(np.float32), edge_order=1)
    return np.hypot(gx, gy).astype(np.float32)


def _distance_component(
    x: np.ndarray,
    y: np.ndarray,
    observations: ObservationTable | None,
    exclusion_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if observations is None or observations.n_obs == 0:
        ones = np.ones_like(xx, dtype=np.float32)
        return ones, np.zeros_like(xx, dtype=bool)
    obs_xy = observations.coords[:, :2]
    obs_xy = obs_xy[np.isfinite(obs_xy).all(axis=1)]
    if len(obs_xy) == 0:
        ones = np.ones_like(xx, dtype=np.float32)
        return ones, np.zeros_like(xx, dtype=bool)
    tree = cKDTree(obs_xy)
    dist, _ = tree.query(np.column_stack([xx.ravel(), yy.ravel()]), k=1)
    dist = dist.reshape(xx.shape).astype(np.float32)
    excluded = dist < float(exclusion_radius)
    return _robust_normalize(dist, 0.0, 95.0), excluded


def posterior_score_components(
    posterior: dict[str, np.ndarray],
    observations: ObservationTable | None = None,
    max_depth: float | None = 3.0,
    exclusion_radius: float = 3.0,
) -> dict[str, np.ndarray]:
    x = posterior["grid_x"].astype(np.float32)
    y = posterior["grid_y"].astype(np.float32)
    z = posterior["grid_z"].astype(np.float32)
    shape = (len(x), len(y))

    eic_unc = _surface_mean(posterior.get("eic_std", np.zeros((*shape, len(z)), dtype=np.float32)), z, max_depth)
    temp_unc = _surface_mean(posterior.get("temperature_std", np.zeros((*shape, len(z)), dtype=np.float32)), z, max_depth)
    if "facies_entropy" in posterior:
        fac_unc = _surface_mean(posterior["facies_entropy"], z, max_depth)
    elif "facies_probability" in posterior:
        fac_unc = _surface_mean(facies_entropy(posterior["facies_probability"]), z, max_depth)
    else:
        fac_unc = np.zeros(shape, dtype=np.float32)
    uncertainty = _robust_normalize(eic_unc) * 0.45 + _robust_normalize(temp_unc) * 0.25 + _robust_normalize(fac_unc) * 0.30

    ice_prob = posterior.get("ice_rich_probability", posterior.get("eic_mean", np.zeros((*shape, len(z)), dtype=np.float32)) > 0.30)
    ice_surface = _surface_max(np.asarray(ice_prob, dtype=np.float32), z, max_depth)
    ice_rich_ambiguity = _robust_normalize(ice_surface * (1.0 - ice_surface))

    settlement = _settlement_from_posterior(posterior)
    settlement_risk = _robust_normalize(settlement)
    differential = _robust_normalize(_differential_settlement(settlement, x, y))
    novelty, excluded = _distance_component(x, y, observations, exclusion_radius=exclusion_radius)
    return {
        "uncertainty": uncertainty.astype(np.float32),
        "ice_rich_ambiguity": ice_rich_ambiguity.astype(np.float32),
        "settlement_risk": settlement_risk.astype(np.float32),
        "differential_settlement": differential.astype(np.float32),
        "novelty": novelty.astype(np.float32),
        "excluded": excluded,
        "settlement_potential": settlement.astype(np.float32),
    }


def build_voi_score(components: dict[str, np.ndarray], weights: VOIWeights | None = None) -> np.ndarray:
    w = weights or VOIWeights()
    score = (
        w.uncertainty * components["uncertainty"]
        + w.ice_rich_ambiguity * components["ice_rich_ambiguity"]
        + w.settlement_risk * components["settlement_risk"]
        + w.differential_settlement * components["differential_settlement"]
        + w.novelty * components["novelty"]
    ).astype(np.float32)
    score = np.where(components.get("excluded", np.zeros_like(score, dtype=bool)), -np.inf, score)
    return score


def recommend_boreholes(
    score: np.ndarray,
    posterior: dict[str, np.ndarray],
    components: dict[str, np.ndarray],
    top_k: int = 8,
    min_spacing: float = 8.0,
    max_depth: float = 3.0,
) -> list[dict[str, float]]:
    x = posterior["grid_x"].astype(np.float32)
    y = posterior["grid_y"].astype(np.float32)
    work = np.array(score, dtype=np.float32, copy=True)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    picks: list[dict[str, float]] = []
    for rank in range(1, int(top_k) + 1):
        if not np.isfinite(work).any():
            break
        flat = int(np.nanargmax(work))
        ix, iy = np.unravel_index(flat, work.shape)
        value = float(work[ix, iy])
        if not np.isfinite(value):
            break
        picks.append(
            {
                "rank": float(rank),
                "x": float(x[ix]),
                "y": float(y[iy]),
                "recommended_depth_m": float(max_depth),
                "voi_score": value,
                "uncertainty": float(components["uncertainty"][ix, iy]),
                "ice_rich_ambiguity": float(components["ice_rich_ambiguity"][ix, iy]),
                "settlement_risk": float(components["settlement_risk"][ix, iy]),
                "differential_settlement": float(components["differential_settlement"][ix, iy]),
                "novelty": float(components["novelty"][ix, iy]),
                "settlement_potential": float(components["settlement_potential"][ix, iy]),
            }
        )
        suppress = np.hypot(xx - float(x[ix]), yy - float(y[iy])) < float(min_spacing)
        work[suppress] = -np.inf
    return picks


def recommend_ert_lines(
    score: np.ndarray,
    posterior: dict[str, np.ndarray],
    top_k: int = 4,
    min_spacing: float = 8.0,
) -> list[dict[str, float | str]]:
    x = posterior["grid_x"].astype(np.float32)
    y = posterior["grid_y"].astype(np.float32)
    candidates: list[dict[str, float | str]] = []
    finite_score = np.where(np.isfinite(score), score, np.nan)
    for iy, yv in enumerate(y):
        row = finite_score[:, iy]
        if np.isfinite(row).any():
            candidates.append(
                {
                    "orientation": "x",
                    "x_start": float(x[0]),
                    "y_start": float(yv),
                    "x_end": float(x[-1]),
                    "y_end": float(yv),
                    "line_score": float(np.nanmean(row)),
                    "max_score": float(np.nanmax(row)),
                }
            )
    for ix, xv in enumerate(x):
        col = finite_score[ix, :]
        if np.isfinite(col).any():
            candidates.append(
                {
                    "orientation": "y",
                    "x_start": float(xv),
                    "y_start": float(y[0]),
                    "x_end": float(xv),
                    "y_end": float(y[-1]),
                    "line_score": float(np.nanmean(col)),
                    "max_score": float(np.nanmax(col)),
                }
            )
    candidates.sort(key=lambda row: (float(row["line_score"]), float(row["max_score"])), reverse=True)
    selected: list[dict[str, float | str]] = []
    for row in candidates:
        if len(selected) >= int(top_k):
            break
        keep = True
        for prev in selected:
            if row["orientation"] != prev["orientation"]:
                continue
            if row["orientation"] == "x" and abs(float(row["y_start"]) - float(prev["y_start"])) < min_spacing:
                keep = False
            if row["orientation"] == "y" and abs(float(row["x_start"]) - float(prev["x_start"])) < min_spacing:
                keep = False
        if keep:
            row = dict(row)
            row["rank"] = float(len(selected) + 1)
            selected.append(row)
    return selected
