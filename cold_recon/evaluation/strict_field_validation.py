from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cold_recon.data.data_schema import (
    OBS_TYPES,
    SUPPORT_TYPES,
    ObservationTable,
)
from cold_recon.evaluation.uncertainty import ensemble_crps, interval_coverage


CONDITIONING_TYPES = frozenset(
    (OBS_TYPES["ert_log_resistivity"], OBS_TYPES["alt"])
)
QUERY_ALLOWED_KEYS = frozenset(
    {
        "query_coords",
        "query_ids",
        "query_group_ids",
        "query_site_ids",
        "query_times",
        "query_support_extent",
        "query_support_type_ids",
        "query_metadata_json",
    }
)
TARGET_ALLOWED_KEYS = frozenset(
    {
        "target_query_ids",
        "target_values",
        "target_sigma",
        "target_metadata_json",
    }
)

DEPLOYMENT_OOD_FEATURE_VERSION = "ert-alt-grid-physical-scale-v1"
DEPLOYMENT_GRID_SCALE_FEATURE_NAMES = (
    "grid_x_span_m",
    "grid_y_span_m",
    "grid_z_span_m",
    "grid_dx_m",
    "grid_dy_m",
    "grid_dz_m",
)


def deployment_grid_scale_features(grid: dict[str, Any]) -> np.ndarray:
    """Return absolute physical-domain features omitted by normalized OOD inputs.

    The legacy OOD features normalize coordinate extents by the local grid. That
    is appropriate for the frozen full-acquisition controller but can make a
    200-m field deployment resemble a much smaller synthetic scene. Deployment
    controllers therefore append the untransformed metre-scale domain spans and
    grid spacings under a versioned feature contract.
    """

    axes = [np.asarray(grid[name], dtype=np.float64) for name in ("x", "y", "z")]
    if any(axis.ndim != 1 or len(axis) == 0 for axis in axes):
        raise ValueError("deployment OOD grid axes must be non-empty and one-dimensional")
    spans = [float(np.ptp(axis)) for axis in axes]
    spacings = [
        float(np.median(np.diff(axis))) if len(axis) > 1 else float(grid.get(name, np.nan))
        for axis, name in zip(axes, ("dx", "dy", "dz"), strict=True)
    ]
    features = np.asarray([*spans, *spacings], dtype=np.float64)
    if not np.all(np.isfinite(features)) or np.any(features <= 0.0):
        raise ValueError("deployment OOD grid spans and spacings must be finite and positive")
    return features


def deployment_ood_features(
    observations: ObservationTable,
    grid: dict[str, Any],
    prior: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build versioned ERT+ALT deployment features with absolute grid scale."""

    # Imported lazily to keep this module's safety/schema helpers lightweight.
    from cold_recon.evaluation.ood_control import (
        observation_ood_features,
        scene_ood_features,
    )

    scale = deployment_grid_scale_features(grid)
    observation = np.concatenate(
        [observation_ood_features(observations, grid), scale]
    ).astype(np.float64)
    context = np.concatenate(
        [scene_ood_features(observations, grid, prior), scale]
    ).astype(np.float64)
    return observation, context


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array.astype(str)


def load_conditioning_observations(path: str | Path) -> ObservationTable:
    """Load an ERT+ALT-only file without touching values for rejected types.

    The type vector is inspected before ``obs_values`` is accessed.  A combined
    file containing even inactive NMR rows is rejected, so the prediction phase
    cannot silently see a held NMR value.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        required = {"obs_coords", "obs_type_ids", "obs_values", "obs_sigma", "obs_mask"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"conditioning file is missing keys: {missing}")
        type_ids = np.asarray(data["obs_type_ids"], dtype=np.int64)
        unexpected = sorted(set(type_ids.tolist()).difference(CONDITIONING_TYPES))
        if unexpected:
            raise ValueError(
                "conditioning file must physically contain only ERT and ALT rows; "
                f"found forbidden type ids {unexpected}"
            )
        if not CONDITIONING_TYPES.issubset(set(type_ids.tolist())):
            raise ValueError("conditioning file must contain both ERT and ALT rows")
        n = len(type_ids)

        def optional(name: str, default: np.ndarray) -> np.ndarray:
            return np.asarray(data[name]) if name in data.files else default

        observations = ObservationTable(
            coords=np.asarray(data["obs_coords"], dtype=np.float32),
            type_ids=type_ids,
            values=np.asarray(data["obs_values"], dtype=np.float32),
            sigma=np.asarray(data["obs_sigma"], dtype=np.float32),
            mask=np.asarray(data["obs_mask"], dtype=bool),
            times=optional("obs_times", np.full(n, np.nan, dtype=np.float32)),
            support_type_ids=optional(
                "obs_support_type_ids", np.full(n, SUPPORT_TYPES["point"], dtype=np.int64)
            ),
            support_extent=optional("obs_support_extent", np.zeros((n, 3), dtype=np.float32)),
            orientation=optional("obs_orientation", np.zeros((n, 3), dtype=np.float32)),
            quality=optional("obs_quality", np.ones(n, dtype=np.float32)),
            site_ids=optional("obs_site_ids", np.zeros(n, dtype=np.int64)),
            source_ids=optional("obs_source_ids", type_ids.copy()),
            group_ids=optional("obs_group_ids", np.full(n, -1, dtype=np.int64)),
        )
    if not np.all(np.isfinite(observations.values[observations.mask])):
        raise ValueError("active conditioning values must be finite")
    if np.any(observations.sigma[observations.mask] <= 0.0):
        raise ValueError("active conditioning sigma must be positive")
    return observations


@dataclass(frozen=True)
class BlindQueries:
    ids: np.ndarray
    observations: ObservationTable
    metadata: dict[str, Any]


def load_blind_queries(path: str | Path) -> BlindQueries:
    """Load geometry-only NMR queries from a strict key whitelist."""

    with np.load(Path(path), allow_pickle=False) as data:
        unexpected = sorted(set(data.files).difference(QUERY_ALLOWED_KEYS))
        if unexpected:
            raise ValueError(
                "query file contains non-geometry keys and is not safe for blind prediction: "
                f"{unexpected}"
            )
        required = {"query_coords", "query_ids", "query_group_ids", "query_times"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"query file is missing keys: {missing}")
        coords = np.asarray(data["query_coords"], dtype=np.float32)
        ids = _strings(data["query_ids"], "query_ids")
        groups = np.asarray(data["query_group_ids"], dtype=np.int64)
        times = np.asarray(data["query_times"], dtype=np.float32)
        n = len(ids)
        if coords.shape != (n, 3) or len(groups) != n or times.shape != (n,):
            raise ValueError("query coords, ids, group ids, and times have inconsistent lengths")
        if len(np.unique(ids)) != n:
            raise ValueError("query_ids must be unique")
        sites = (
            np.asarray(data["query_site_ids"], dtype=np.int64)
            if "query_site_ids" in data.files
            else np.zeros(n, dtype=np.int64)
        )
        extents = (
            np.asarray(data["query_support_extent"], dtype=np.float32)
            if "query_support_extent" in data.files
            else np.tile(np.asarray([1.0, 1.0, 0.25], dtype=np.float32), (n, 1))
        )
        support_types = (
            np.asarray(data["query_support_type_ids"], dtype=np.int64)
            if "query_support_type_ids" in data.files
            else np.full(n, SUPPORT_TYPES["nmr_kernel"], dtype=np.int64)
        )
        metadata = (
            json.loads(str(data["query_metadata_json"].item()))
            if "query_metadata_json" in data.files
            else {}
        )
    if len(sites) != n or len(support_types) != n or extents.shape != (n, 3):
        raise ValueError("query support metadata have inconsistent lengths")
    if np.any(support_types != SUPPORT_TYPES["nmr_kernel"]):
        raise ValueError("strict NMR evaluation requires nmr_kernel query supports")
    if np.any(extents <= 0.0) or not np.all(np.isfinite(coords)):
        raise ValueError("query coordinates and support extents must be finite and positive")
    observations = ObservationTable(
        coords=coords,
        type_ids=np.full(n, OBS_TYPES["nmr_unfrozen_water"], dtype=np.int64),
        values=np.zeros(n, dtype=np.float32),
        sigma=np.ones(n, dtype=np.float32),
        mask=np.ones(n, dtype=bool),
        times=times,
        support_type_ids=support_types,
        support_extent=extents,
        quality=np.ones(n, dtype=np.float32),
        site_ids=sites,
        source_ids=np.full(n, OBS_TYPES["nmr_unfrozen_water"], dtype=np.int64),
        group_ids=groups,
    )
    return BlindQueries(ids=ids, observations=observations, metadata=metadata)


def localize_xy(
    conditioning: ObservationTable, queries: BlindQueries
) -> tuple[ObservationTable, BlindQueries, np.ndarray]:
    """Apply one shared horizontal origin without using target values."""

    valid = np.asarray(conditioning.mask, dtype=bool)
    if not np.any(valid):
        raise ValueError("conditioning observations contain no active rows")
    origin = np.asarray(
        [
            float(np.nanmin(conditioning.coords[valid, 0])),
            float(np.nanmin(conditioning.coords[valid, 1])),
        ],
        dtype=np.float32,
    )
    cond = conditioning.subset(np.arange(conditioning.n_obs))
    cond.coords[:, :2] -= origin[None, :]
    query_obs = queries.observations.subset(np.arange(queries.observations.n_obs))
    query_obs.coords[:, :2] -= origin[None, :]
    return cond, BlindQueries(queries.ids.copy(), query_obs, dict(queries.metadata)), origin


def assert_queries_inside_grid(queries: BlindQueries, grid: dict[str, Any]) -> None:
    """Fail instead of clipping a blind query to an out-of-domain grid edge."""

    coords = queries.observations.coords
    extents = 0.5 * queries.observations.support_extent
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        axis = np.asarray(grid[axis_name], dtype=np.float64)
        spacing = float(np.mean(np.diff(axis))) if len(axis) > 1 else 0.0
        lower = coords[:, axis_index] - extents[:, axis_index]
        upper = coords[:, axis_index] + extents[:, axis_index]
        if np.any(lower < float(axis[0]) - spacing) or np.any(upper > float(axis[-1]) + spacing):
            raise ValueError(
                f"blind query support lies outside the conditioning-derived {axis_name} grid"
            )


def load_external_baseline(
    path: str | Path, query_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a prediction frozen on independent development sites.

    Eligibility is deliberately fail-closed: a baseline file must attest that
    it was frozen before unsealing and did not use blind target values.
    """

    allowed = {"baseline_query_ids", "baseline_prediction", "baseline_metadata_json"}
    with np.load(Path(path), allow_pickle=False) as data:
        unexpected = sorted(set(data.files).difference(allowed))
        if unexpected:
            raise ValueError(f"baseline file contains forbidden keys: {unexpected}")
        required = allowed
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"baseline file is missing keys: {missing}")
        ids = _strings(data["baseline_query_ids"], "baseline_query_ids")
        prediction = np.asarray(data["baseline_prediction"], dtype=np.float32)
        metadata = json.loads(str(data["baseline_metadata_json"].item()))
    if not np.array_equal(ids, query_ids):
        raise ValueError("baseline query ids do not exactly match blind query order")
    if prediction.shape != (len(query_ids),) or not np.all(np.isfinite(prediction)):
        raise ValueError("baseline_prediction must be a finite vector")
    eligible = (
        bool(metadata.get("frozen_before_target_unseal", False))
        and not bool(metadata.get("uses_blind_target", True))
        and str(metadata.get("development_construction_gate", "FAIL_CLOSED")) == "PASS"
        and str(metadata.get("site_generality_gate", "FAIL_CLOSED")) == "PASS"
    )
    metadata = {**metadata, "strict_gate_eligible": eligible}
    return prediction, metadata


def load_targets(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load target values. This function belongs exclusively to score phase."""

    with np.load(Path(path), allow_pickle=False) as data:
        unexpected = sorted(set(data.files).difference(TARGET_ALLOWED_KEYS))
        if unexpected:
            raise ValueError(f"target file contains unexpected keys: {unexpected}")
        required = {"target_query_ids", "target_values"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"target file is missing keys: {missing}")
        ids = _strings(data["target_query_ids"], "target_query_ids")
        values = np.asarray(data["target_values"], dtype=np.float32)
        sigma = (
            np.asarray(data["target_sigma"], dtype=np.float32)
            if "target_sigma" in data.files
            else np.full(len(ids), np.nan, dtype=np.float32)
        )
        metadata = (
            json.loads(str(data["target_metadata_json"].item()))
            if "target_metadata_json" in data.files
            else {}
        )
    if values.shape != (len(ids),) or sigma.shape != (len(ids),):
        raise ValueError("target arrays have inconsistent lengths")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("target_values must be finite volumetric fractions in [0, 1]")
    return ids, values, sigma, metadata


def _metrics(samples: np.ndarray, truth: np.ndarray, level: float) -> dict[str, float]:
    mean = np.mean(samples, axis=0)
    error = mean - truth
    coverage, width = interval_coverage(samples, truth, level=level)
    return {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "crps": ensemble_crps(samples, truth, max_points=None),
        f"coverage_{int(round(100 * level))}": coverage,
        f"mean_width_{int(round(100 * level))}": width,
    }


def score_predictions(
    *,
    prediction_members: np.ndarray,
    baseline_prediction: np.ndarray,
    truth: np.ndarray,
    group_ids: np.ndarray,
    site_ids: np.ndarray,
    learned_used: bool,
    strong_baseline_eligible: bool,
    level: float = 0.90,
    min_groups: int = 12,
    bootstrap_replicates: int = 5000,
    seed: int = 20260830,
) -> tuple[dict[str, Any], pd.DataFrame]:
    members = np.asarray(prediction_members, dtype=np.float32)
    target = np.asarray(truth, dtype=np.float32)
    baseline = np.asarray(baseline_prediction, dtype=np.float32)
    groups = np.asarray(group_ids, dtype=np.int64)
    sites = np.asarray(site_ids, dtype=np.int64)
    if members.ndim != 2 or members.shape[1] != len(target):
        raise ValueError("prediction_members must have shape [members, queries]")
    if baseline.shape != target.shape or groups.shape != target.shape or sites.shape != target.shape:
        raise ValueError("prediction, target, and grouping arrays must align")

    point = _metrics(members, target, level)
    baseline_error = baseline - target
    baseline_metrics = {
        "n": int(len(target)),
        "rmse": float(np.sqrt(np.mean(baseline_error**2))),
        "mae": float(np.mean(np.abs(baseline_error))),
        "bias": float(np.mean(baseline_error)),
    }
    rows: list[dict[str, Any]] = []
    group_improvements: list[float] = []
    for group_id in np.unique(groups):
        selected = groups == group_id
        local = _metrics(members[:, selected], target[selected], level)
        baseline_rmse = float(
            np.sqrt(np.mean((baseline[selected] - target[selected]) ** 2))
        )
        group_improvements.append(baseline_rmse - float(local["rmse"]))
        rows.append(
            {
                "site_id": int(sites[selected][0]),
                "group_id": int(group_id),
                **local,
                "baseline_rmse": baseline_rmse,
                "baseline_mae": float(
                    np.mean(np.abs(baseline[selected] - target[selected]))
                ),
                "rmse_improvement": baseline_rmse - float(local["rmse"]),
            }
        )
    group_frame = pd.DataFrame(rows).sort_values(["site_id", "group_id"])
    gains = np.asarray(group_improvements, dtype=np.float64)
    if len(gains):
        rng = np.random.default_rng(int(seed))
        index = rng.integers(0, len(gains), size=(int(bootstrap_replicates), len(gains)))
        boot = gains[index].mean(axis=1)
        gain_ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    else:
        gain_ci = [float("nan"), float("nan")]

    coverage_key = f"coverage_{int(round(100 * level))}"
    reasons: list[str] = []
    if not learned_used:
        reasons.append("OOD_FALLBACK_NO_LEARNED_CORRECTION")
    if not strong_baseline_eligible:
        reasons.append("NO_FROZEN_INDEPENDENT_STRONG_BASELINE")
    if len(gains) < int(min_groups):
        reasons.append(f"INSUFFICIENT_INDEPENDENT_GROUPS_{len(gains)}_LT_{int(min_groups)}")
    if not np.isfinite(gain_ci[0]) or gain_ci[0] <= 0.0:
        reasons.append("RMSE_IMPROVEMENT_CLUSTER_BOOTSTRAP_CI_NOT_POSITIVE")
    if not (0.80 <= float(point[coverage_key]) <= 1.0):
        reasons.append("NOMINAL_INTERVAL_COVERAGE_BELOW_0.80")
    status = "PASS" if not reasons else "FAIL_CLOSED"
    summary = {
        "field_validation_gate": status,
        "gate_reasons": reasons,
        "learned_correction_used": bool(learned_used),
        "strong_baseline_eligible": bool(strong_baseline_eligible),
        "independent_groups": int(len(gains)),
        "point_metrics": point,
        "baseline_point_metrics": baseline_metrics,
        "hole_balanced_rmse": float(group_frame["rmse"].mean()) if len(group_frame) else float("nan"),
        "hole_balanced_mae": float(group_frame["mae"].mean()) if len(group_frame) else float("nan"),
        "hole_balanced_baseline_rmse": (
            float(group_frame["baseline_rmse"].mean()) if len(group_frame) else float("nan")
        ),
        "hole_balanced_rmse_improvement": float(np.mean(gains)) if len(gains) else float("nan"),
        "hole_balanced_rmse_improvement_ci95": gain_ci,
        "interval_level": float(level),
        "minimum_groups_gate": int(min_groups),
        "bootstrap_unit": "complete query group (physical hole/profile)",
    }
    return summary, group_frame
