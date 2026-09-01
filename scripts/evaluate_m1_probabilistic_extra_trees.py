from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from cold_recon.baselines.random_forest import _grid_points_and_surface, _surface_at_obs
from cold_recon.data.data_schema import OBS_TYPES, load_sample_npz
from cold_recon.evaluation.block_conformal import interval_diagnostics
from cold_recon.evaluation.engineering_response import (
    DEFAULT_THAW_DEPTHS_M,
    engineering_response_metrics,
)
from cold_recon.evaluation.rare_structure_metrics import (
    binary_event_metrics,
    high_eic_object_metrics,
)
from cold_recon.evaluation.uncertainty import (
    ensemble_crps,
    interval_coverage,
)
from cold_recon.operators.support import (
    build_error_covariance,
    build_observation_operator,
    normalized_misfit,
)
from scripts.evaluate_m1_controlled import _bootstrap_mean_ci
from scripts.train_m1_support_guided_flow import _manifest_records


METHOD = "Bootstrap Extra Trees"


def _ensemble(
    sample: dict,
    *,
    seed: int,
    n_estimators: int,
    n_members: int,
    n_jobs: int,
) -> np.ndarray:
    observations = sample["observations"]
    mask = observations.mask & (
        observations.type_ids == OBS_TYPES["borehole_eic"]
    )
    if int(mask.sum()) < 4:
        raise ValueError("probabilistic Extra Trees requires at least four EIC rows")
    query, _, shape = _grid_points_and_surface(sample)
    training = _surface_at_obs(sample, observations.coords[mask])
    target = np.asarray(observations.values[mask], dtype=np.float32)
    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=2,
        max_features=1.0,
        bootstrap=True,
        n_jobs=int(n_jobs),
        random_state=int(seed),
    )
    model.fit(training, target)
    members = [
        estimator.predict(query).reshape(shape)
        for estimator in model.estimators_[: int(n_members)]
    ]
    return np.clip(np.asarray(members, dtype=np.float32), 0.0, 0.90)


def _block_scores(
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    block_shape: tuple[int, int, int],
    std_floor: float,
    within_block_quantile: float,
) -> np.ndarray:
    score = np.abs(np.asarray(truth) - np.asarray(mean)) / np.maximum(
        np.asarray(std), float(std_floor)
    )
    nx, ny, nz = score.shape
    bx, by, bz = (int(value) for value in block_shape)
    if nx % bx or ny % by or nz % bz:
        raise ValueError("block shape must exactly divide the controlled grid")
    blocks = (
        score.reshape(nx // bx, bx, ny // by, by, nz // bz, bz)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(-1, bx * by * bz)
    )
    return np.quantile(blocks, float(within_block_quantile), axis=1)


def _conformal_quantile(
    values: np.ndarray,
    *,
    level: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    adjusted = min(1.0, np.ceil((len(values) + 1) * float(level)) / len(values))
    return float(np.quantile(values, adjusted, method="higher"))


def _support_metrics(sample: dict, mean: np.ndarray) -> dict[str, float]:
    observations = sample["observations"]
    indices = np.flatnonzero(
        observations.mask
        & (observations.type_ids == OBS_TYPES["borehole_eic"])
    )
    operator = build_observation_operator(
        observations, sample["grid"], indices=indices
    )
    predicted = np.asarray(operator.apply(mean), dtype=np.float64)
    observed = np.asarray(observations.values[indices], dtype=np.float64)
    covariance = build_error_covariance(
        observations, indices, correlated=False
    )
    return {
        "support_nrmse_borehole_eic": normalized_misfit(
            predicted, observed, covariance
        ),
        "support_bias_borehole_eic": float(np.mean(predicted - observed)),
    }


def _fit_calibration(
    root: Path,
    records: list[dict],
    args: argparse.Namespace,
) -> tuple[float, pd.DataFrame]:
    block_rows: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int | str]] = []
    for index, record in enumerate(records):
        started = time.perf_counter()
        sample = load_sample_npz(root / record["relative_path"])
        members = _ensemble(
            sample,
            seed=int(record["seed"]) + int(args.seed_offset),
            n_estimators=int(args.n_estimators),
            n_members=int(args.members),
            n_jobs=int(args.n_jobs),
        )
        truth = np.asarray(sample["fields"]["eic"], dtype=np.float32)
        mean = members.mean(axis=0)
        std = members.std(axis=0)
        block_rows.append(
            _block_scores(
                truth,
                mean,
                std,
                tuple(args.block_shape),
                float(args.std_floor),
                float(args.within_block_quantile),
            )
        )
        diagnostics.append(
            {
                "scene_id": str(record["scene_id"]),
                "eic_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
                "raw_coverage": interval_coverage(members, truth, level=0.90)[0],
                "raw_width": interval_coverage(members, truth, level=0.90)[1],
            }
        )
        print(
            f"calibration {index + 1}/{len(records)} {record['scene_id']} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    quantile = _conformal_quantile(
        np.concatenate(block_rows), level=float(args.conformal_level)
    )
    return quantile, pd.DataFrame(diagnostics)


def _evaluate_test(
    root: Path,
    records: list[dict],
    quantile: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    response_rows: list[dict[str, float | int | str]] = []
    for index, record in enumerate(records):
        started = time.perf_counter()
        sample = load_sample_npz(root / record["relative_path"])
        members = _ensemble(
            sample,
            seed=int(record["seed"]) + int(args.seed_offset),
            n_estimators=int(args.n_estimators),
            n_members=int(args.members),
            n_jobs=int(args.n_jobs),
        )
        truth = np.asarray(sample["fields"]["eic"], dtype=np.float32)
        mean = members.mean(axis=0)
        std = members.std(axis=0)
        raw_coverage, raw_width = interval_coverage(members, truth, level=0.90)
        half_width = float(quantile) * np.maximum(std, float(args.std_floor))
        lower = np.clip(mean - half_width, 0.0, 0.90)
        upper = np.clip(mean + half_width, 0.0, 0.90)
        calibrated = interval_diagnostics(truth, lower, upper)
        row: dict[str, float | int | str] = {
            "scene_id": str(record["scene_id"]),
            "generator_family": str(record["generator_family"]),
            "method": METHOD,
            "seed": int(args.seed_offset),
            "eic_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
            "eic_coverage": float(raw_coverage),
            "eic_mean_width": float(raw_width),
            "eic_crps": ensemble_crps(
                members, truth, seed=int(record["seed"])
            ),
            "eic_calibrated_coverage": calibrated["coverage"],
            "eic_calibrated_mean_width": calibrated["mean_width"],
        }
        row.update(_support_metrics(sample, mean))
        if not bool(args.skip_event_metrics):
            for threshold in (0.20, 0.30, 0.40):
                probability = np.mean(members >= float(threshold), axis=0)
                prefix = f"high_eic_t{int(round(100 * threshold)):02d}"
                row.update(
                    {
                        f"{prefix}_{key}": value
                        for key, value in binary_event_metrics(
                            probability, truth >= threshold, probability_threshold=0.50
                        ).items()
                    }
                )
                row.update(
                    {
                        f"{prefix}_{key}": value
                        for key, value in high_eic_object_metrics(
                            probability,
                            truth,
                            eic_threshold=threshold,
                            probability_threshold=0.50,
                            dz=float(sample["grid"]["dz"]),
                        ).items()
                    }
                )
        row["end_to_end_wall_seconds"] = time.perf_counter() - started
        rows.append(row)
        for depth in args.engineering_response_depths:
            response_rows.append(
                engineering_response_metrics(
                    scene_id=str(record["scene_id"]),
                    method=METHOD,
                    seed=int(args.seed_offset),
                    truth_eic=truth,
                    candidate_eic_mean=mean,
                    candidate_eic_samples=members,
                    candidate_eic_std=std,
                    conformal_quantile=float(quantile),
                    grid=sample["grid"],
                    thaw_depth_m=float(depth),
                    screening_threshold_m=float(args.engineering_response_threshold),
                    decision_probability=float(args.engineering_response_probability),
                )
            )
        print(
            f"test {index + 1}/{len(records)} {record['scene_id']} "
            f"rmse={row['eic_rmse']:.4f} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(response_rows)


def _summary(detail: pd.DataFrame, response: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metrics = (
        "eic_rmse",
        "eic_coverage",
        "eic_mean_width",
        "eic_crps",
        "eic_calibrated_coverage",
        "eic_calibrated_mean_width",
        "support_nrmse_borehole_eic",
        "support_bias_borehole_eic",
        "high_eic_t30_f1",
        "high_eic_t30_object_f1",
    )
    for index, metric in enumerate(metrics):
        if metric not in detail.columns:
            continue
        mean, lower, upper = _bootstrap_mean_ci(
            detail[metric].astype(float).tolist(), int(seed) + index
        )
        rows.append(
            {
                "method": METHOD,
                "thaw_depth_m": float("nan"),
                "metric": metric,
                "mean": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_scenes": int(detail["scene_id"].nunique()),
            }
        )
    for (depth, metric), group in (
        response.melt(
            id_vars=["scene_id", "thaw_depth_m"],
            value_vars=[
                "response_rmse_m",
                "response_bias_m",
                "gradient_rmse_m_per_m",
                "sensitivity",
                "specificity",
                "raw_interval_mean_width_m",
                "conformal_envelope_mean_width_m",
            ],
            var_name="metric",
            value_name="value",
        ).groupby(["thaw_depth_m", "metric"])
    ):
        mean, lower, upper = _bootstrap_mean_ci(
            group["value"].astype(float).tolist(), int(seed) + int(10 * depth)
        )
        rows.append(
            {
                "method": METHOD,
                "thaw_depth_m": float(depth),
                "metric": str(metric),
                "mean": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_scenes": int(group["scene_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/m1_support_guided_benchmark/m1_scene_manifest.json",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--members", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed-offset", type=int, default=941)
    parser.add_argument("--max-validation-scenes", type=int, default=0)
    parser.add_argument("--max-test-scenes", type=int, default=0)
    parser.add_argument("--skip-event-metrics", action="store_true")
    parser.add_argument("--block-shape", type=int, nargs=3, default=(8, 8, 6))
    parser.add_argument("--std-floor", type=float, default=0.001)
    parser.add_argument("--within-block-quantile", type=float, default=0.90)
    parser.add_argument("--conformal-level", type=float, default=0.90)
    parser.add_argument(
        "--engineering-response-depths",
        type=float,
        nargs="+",
        default=DEFAULT_THAW_DEPTHS_M,
    )
    parser.add_argument("--engineering-response-threshold", type=float, default=0.30)
    parser.add_argument("--engineering-response-probability", type=float, default=0.50)
    parser.add_argument(
        "--output-dir",
        default="outputs/m1_support_guided/formal_probabilistic_extra_trees",
    )
    args = parser.parse_args()
    if int(args.members) > int(args.n_estimators):
        raise ValueError("members cannot exceed n_estimators")

    manifest_path = Path(args.manifest)
    root, validation, manifest = _manifest_records(manifest_path, "validation")
    _, test, _ = _manifest_records(manifest_path, "test_id")
    if int(args.max_validation_scenes) > 0:
        validation = validation[: int(args.max_validation_scenes)]
    if int(args.max_test_scenes) > 0:
        test = test[: int(args.max_test_scenes)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    quantile, validation_detail = _fit_calibration(root, validation, args)
    detail, response = _evaluate_test(root, test, quantile, args)
    summary = _summary(detail, response, int(args.seed_offset))
    validation_detail.to_csv(output / "m1_probabilistic_extra_trees_validation.csv", index=False)
    detail.to_csv(output / "m1_probabilistic_extra_trees_test_id_detail.csv", index=False)
    response.to_csv(output / "m1_probabilistic_extra_trees_test_id_response.csv", index=False)
    summary.to_csv(output / "m1_probabilistic_extra_trees_test_id_summary.csv", index=False)
    metadata = {
        "method": METHOD,
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "validation_scenes": int(validation_detail["scene_id"].nunique()),
        "test_scenes": int(detail["scene_id"].nunique()),
        "n_estimators": int(args.n_estimators),
        "posterior_members": int(args.members),
        "bootstrap": True,
        "min_samples_leaf": 2,
        "spatial_conformal": "validation-only 3-D block conformal",
        "conformal_quantile": float(quantile),
        "block_shape": list(args.block_shape),
        "independent_unit": "controlled scene",
    }
    (output / "m1_probabilistic_extra_trees_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
