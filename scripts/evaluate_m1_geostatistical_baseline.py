from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from cold_recon.baselines.support_conditioned_gaussian import (
    ConditionalGaussianConfig,
    support_conditioned_gaussian_ensemble,
    support_conditioned_gaussian_ensemble_batched,
)
from cold_recon.data.data_schema import OBS_TYPES, load_sample_npz
from cold_recon.evaluation.block_conformal import (
    interval_diagnostics,
    posterior_diagnostics,
)
from cold_recon.evaluation.rare_structure_metrics import high_eic_object_metrics
from cold_recon.evaluation.engineering_response import (
    DEFAULT_THAW_DEPTHS_M,
    engineering_response_metrics,
)
from cold_recon.operators.support import (
    build_error_covariance,
    build_observation_operator,
    normalized_misfit,
)
from cold_recon.training.factorized_volume_codec import (
    bounded_recenter_samples,
    tensor_to_factorized_fields,
)
from scripts.evaluate_m1_controlled import (
    _bootstrap_mean_ci,
    _fit_cached_spatial_conformal,
    _load_conformal_quantile,
)
from scripts.train_m1_support_guided_flow import (
    _load_or_build_prior,
    _manifest_records,
    _prior_tensor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/m1_support_guided_benchmark/m1_scene_manifest.json",
    )
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--scene-stride", type=int, default=1)
    parser.add_argument("--members", type=int, default=64)
    parser.add_argument("--rf-trees", type=int, default=24)
    parser.add_argument("--conditioning-iterations", type=int, default=20)
    parser.add_argument("--batched-conditioning", action="store_true")
    parser.add_argument("--conditioning-batch-size", type=int, default=8)
    parser.add_argument("--marginal-std", type=float, default=0.15)
    parser.add_argument(
        "--transform-space", choices=("raw", "logit"), default="raw"
    )
    parser.add_argument("--logit-marginal-std", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--bounded-sensitivity",
        action="store_true",
        help=(
            "Use the pre-registered physical EIC bounds by contracting "
            "anomalies about the support-conditioned Gaussian mean. The "
            "legacy option name is retained for command-ledger compatibility."
        ),
    )
    parser.add_argument("--seed", type=int, default=430)
    parser.add_argument(
        "--output-dir", default="outputs/m1_support_guided/tables"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conformal-file", default="")
    parser.add_argument("--fit-conformal", action="store_true")
    parser.add_argument("--conformal-level", type=float, default=0.90)
    parser.add_argument("--conformal-within-block-quantile", type=float, default=0.90)
    parser.add_argument("--conformal-std-floor", type=float, default=0.001)
    parser.add_argument("--engineering-response-audit", action="store_true")
    parser.add_argument("--engineering-response-only", action="store_true")
    parser.add_argument(
        "--engineering-response-depths",
        type=float,
        nargs="+",
        default=DEFAULT_THAW_DEPTHS_M,
    )
    parser.add_argument(
        "--engineering-response-threshold", type=float, default=0.30
    )
    parser.add_argument(
        "--engineering-response-probability", type=float, default=0.50
    )
    parser.add_argument(
        "--conformal-block-shape",
        type=int,
        nargs=3,
        default=(8, 8, 6),
        metavar=("BX", "BY", "BZ"),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root, records, manifest = _manifest_records(manifest_path, args.split)
    if int(args.max_scenes) > 0:
        records = records[: int(args.max_scenes)]
    if int(args.scene_stride) <= 0:
        raise ValueError("scene stride must be positive")
    if not 0 <= int(args.scene_offset) < int(args.scene_stride):
        raise ValueError("scene offset must satisfy 0 <= offset < stride")
    records = records[int(args.scene_offset) :: int(args.scene_stride)]
    requested_records = list(records)
    record_positions = {
        str(record["scene_id"]): index for index, record in enumerate(records)
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_suffix = "_logit" if args.transform_space == "logit" else ""
    detail_path = output_dir / (
        f"m1_geostatistical{transform_suffix}_{args.split}_detail.csv"
    )
    response_path = output_dir / (
        f"m1_geostatistical{transform_suffix}_{args.split}_engineering_response.csv"
    )
    conformal_cache_dir = output_dir / "conformal_cache" / (
        f"{args.split}{transform_suffix}"
    )
    if bool(args.fit_conformal):
        if args.split != "validation":
            raise ValueError("geostatistical conformal fitting is validation-only")
        conformal_cache_dir.mkdir(parents=True, exist_ok=True)
    conformal_quantile = _load_conformal_quantile(
        args.conformal_file,
        seed=int(args.seed),
        manifest_sha256=str(manifest["manifest_sha256"]),
    )
    rows: list[dict[str, float | int | str]] = []
    response_rows: list[dict[str, float | int | str]] = []
    if bool(args.resume) and detail_path.exists():
        existing = pd.read_csv(detail_path)
        if bool(args.engineering_response_only):
            retained_columns = [
                "scene_id",
                "generator_family",
                "seed",
                "n_observations",
                "eic_rmse",
                "anchor_eic_rmse",
                "invalid_eic_sample_fraction",
                "raw_gaussian_invalid_eic_sample_fraction",
            ]
            existing = existing[
                [column for column in retained_columns if column in existing.columns]
            ]
        rows = existing.to_dict(orient="records")
        completed = set(existing["scene_id"].astype(str))
        if bool(args.engineering_response_audit):
            if response_path.exists():
                response_existing = pd.read_csv(response_path)
                response_rows = response_existing.to_dict(orient="records")
                response_completed = set(response_existing["scene_id"].astype(str))
                completed &= response_completed
            else:
                completed = set()
        if bool(args.fit_conformal):
            completed = {
                scene_id
                for scene_id in completed
                if (conformal_cache_dir / f"{scene_id}_eic_score.npz").exists()
            }
            existing = existing[
                existing["scene_id"].astype(str).isin(completed)
            ]
            rows = existing.to_dict(orient="records")
        records = [record for record in records if record["scene_id"] not in completed]
    config = ConditionalGaussianConfig(
        marginal_std=(
            float(args.logit_marginal_std)
            if args.transform_space == "logit"
            else float(args.marginal_std)
        ),
        conditioning_iterations=int(args.conditioning_iterations),
        correlated_observation_errors=True,
    )
    for remaining_index, record in enumerate(records):
        scene_index = int(record_positions[str(record["scene_id"])])
        sample = load_sample_npz(root / record["relative_path"])
        observations = sample["observations"]
        indices = np.flatnonzero(
            observations.mask
            & (observations.type_ids == OBS_TYPES["borehole_eic"])
        )
        if len(indices) == 0:
            continue
        prior = _load_or_build_prior(
            sample,
            record,
            root / "prior_cache" / args.split,
            n_facies=7,
            rf_trees=int(args.rf_trees),
        )
        prior_fields = tensor_to_factorized_fields(_prior_tensor(prior))
        prior_eic = np.asarray(prior_fields["eic"], dtype=np.float32)
        gaussian_prior = prior_eic
        gaussian_observations = observations
        if args.transform_space == "logit":
            upper = 0.90
            epsilon = 0.005
            prior_probability = np.clip(prior_eic / upper, epsilon, 1.0 - epsilon)
            gaussian_prior = np.log(
                prior_probability / (1.0 - prior_probability)
            ).astype(np.float32)
            transformed_values = np.asarray(observations.values, dtype=np.float32).copy()
            transformed_sigma = np.asarray(observations.sigma, dtype=np.float32).copy()
            selected_values = np.clip(
                transformed_values[indices] / upper, epsilon, 1.0 - epsilon
            )
            transformed_values[indices] = np.log(
                selected_values / (1.0 - selected_values)
            )
            derivative = 1.0 / (
                upper * selected_values * (1.0 - selected_values)
            )
            transformed_sigma[indices] = np.maximum(
                transformed_sigma[indices] * derivative, 1.0e-4
            )
            gaussian_observations = replace(
                observations,
                values=transformed_values,
                sigma=transformed_sigma,
            )
        ensemble_function = (
            support_conditioned_gaussian_ensemble_batched
            if bool(args.batched_conditioning)
            else support_conditioned_gaussian_ensemble
        )
        ensemble_kwargs = {
            "n_members": int(args.members),
            "seed": int(args.seed) + scene_index,
            "config": config,
        }
        if bool(args.batched_conditioning):
            ensemble_kwargs["batch_size"] = int(args.conditioning_batch_size)
        ensemble = ensemble_function(
            gaussian_prior,
            gaussian_observations,
            sample["grid"],
            indices,
            **ensemble_kwargs,
        )
        if args.transform_space == "logit":
            ensemble = (0.90 * expit(ensemble)).astype(np.float32)
        raw_mean = ensemble.mean(axis=0).astype(np.float32)
        raw_invalid_fraction = float(np.mean((ensemble < 0.0) | (ensemble > 0.90)))
        if bool(args.bounded_sensitivity) and args.transform_space == "raw":
            ensemble = bounded_recenter_samples(
                ensemble,
                np.clip(raw_mean, 0.0, 0.90),
                0.0,
                0.90,
            )
        mean = ensemble.mean(axis=0).astype(np.float32)
        truth = np.asarray(sample["fields"]["eic"], dtype=np.float32)
        probability = np.mean(ensemble >= 0.30, axis=0).astype(np.float32)
        diagnostics = (
            {}
            if bool(args.engineering_response_only)
            else posterior_diagnostics(ensemble, truth)
        )
        calibrated_metrics: dict[str, float] = {}
        posterior_std = np.asarray(ensemble.std(axis=0), dtype=np.float32)
        if conformal_quantile is not None and not bool(args.engineering_response_only):
            half_width = float(conformal_quantile) * np.maximum(
                posterior_std, float(args.conformal_std_floor)
            )
            lower = np.clip(mean - half_width, 0.0, 0.90)
            upper = np.clip(mean + half_width, 0.0, 0.90)
            calibrated_metrics = {
                f"eic_calibrated_{key}": float(value)
                for key, value in interval_diagnostics(
                    truth, lower, upper
                ).items()
            }
        if bool(args.fit_conformal):
            score = np.abs(truth - mean) / np.maximum(
                posterior_std, float(args.conformal_std_floor)
            )
            np.savez_compressed(
                conformal_cache_dir / f"{record['scene_id']}_eic_score.npz",
                score=np.asarray(score, dtype=np.float32),
                block_shape=np.asarray(
                    args.conformal_block_shape, dtype=np.int32
                ),
            )
        object_metrics = (
            {}
            if bool(args.engineering_response_only)
            else high_eic_object_metrics(
                probability,
                truth,
                dz=float(sample["grid"]["dz"]),
            )
        )
        if bool(args.engineering_response_audit):
            method_name = (
                "Logit-Gaussian baseline"
                if args.transform_space == "logit"
                else "Bounded Gaussian baseline"
            )
            for thaw_depth_m in args.engineering_response_depths:
                response_rows.append(
                    engineering_response_metrics(
                        scene_id=str(record["scene_id"]),
                        method=method_name,
                        seed=int(args.seed),
                        truth_eic=truth,
                        candidate_eic_mean=mean,
                        candidate_eic_samples=ensemble,
                        candidate_eic_std=posterior_std,
                        conformal_quantile=conformal_quantile,
                        grid=sample["grid"],
                        thaw_depth_m=float(thaw_depth_m),
                        screening_threshold_m=float(
                            args.engineering_response_threshold
                        ),
                        decision_probability=float(
                            args.engineering_response_probability
                        ),
                    )
                )
            pd.DataFrame(response_rows).to_csv(response_path, index=False)
        row: dict[str, float | int | str] = {
            "scene_id": record["scene_id"],
            "generator_family": record["generator_family"],
            "seed": int(args.seed),
            "n_observations": int(len(indices)),
            "eic_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
            "anchor_eic_rmse": float(
                np.sqrt(np.mean((prior_fields["eic"] - truth) ** 2))
            ),
            "invalid_eic_sample_fraction": float(
                np.mean((ensemble < 0.0) | (ensemble > 0.90))
            ),
            "raw_gaussian_invalid_eic_sample_fraction": raw_invalid_fraction,
            **{f"eic_{key}": float(value) for key, value in diagnostics.items()},
            **calibrated_metrics,
            **{f"high_eic_{key}": float(value) for key, value in object_metrics.items()},
        }
        if not bool(args.engineering_response_only):
            operator = build_observation_operator(
                observations, sample["grid"], indices=indices
            )
            covariance = build_error_covariance(
                observations,
                indices,
                correlated=True,
            )
            row["support_nrmse_borehole_eic"] = normalized_misfit(
                operator.apply(mean), observations.values[indices], covariance
            )
        rows.append(row)
        pd.DataFrame(rows).to_csv(detail_path, index=False)
        print(
            f"{remaining_index + 1}/{len(records)} {record['scene_id']} "
            f"eic_rmse={rows[-1]['eic_rmse']:.4f} "
            f"anchor={rows[-1]['anchor_eic_rmse']:.4f}"
        )

    pd.DataFrame(rows).to_csv(detail_path, index=False)
    if bool(args.engineering_response_audit):
        pd.DataFrame(response_rows).to_csv(response_path, index=False)
    metric_names = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (float, int)) and key not in {"n_observations", "seed"}
    ] if rows else []
    summary_rows = []
    for metric in metric_names:
        mean, lower, upper = _bootstrap_mean_ci(
            [float(row[metric]) for row in rows], int(args.seed) + 71
        )
        summary_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_scenes": len(rows),
            }
        )
    summary_path = output_dir / (
        f"m1_geostatistical{transform_suffix}_{args.split}_summary.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metadata = {
        "method": (
            "logit-transformed covariance-weighted support-conditioned Gaussian random field"
            if args.transform_space == "logit"
            else "covariance-weighted support-conditioned Gaussian random field"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "split": args.split,
        "scenes": len(rows),
        "scene_offset": int(args.scene_offset),
        "scene_stride": int(args.scene_stride),
        "members": int(args.members),
        "seed": int(args.seed),
        "correlated_observation_errors": True,
        "batched_conditioning": bool(args.batched_conditioning),
        "conditioning_batch_size": int(args.conditioning_batch_size),
        "conformal_file": str(args.conformal_file),
        "conformal_quantile": conformal_quantile,
        "transform_space": str(args.transform_space),
        "physical_bounds": (
            "inverse-logit transform to the naturally bounded EIC interval (0, 0.90)"
            if args.transform_space == "logit"
            else
            "bounded anomaly contraction about the conditional Gaussian mean; primary A1"
            if bool(args.bounded_sensitivity)
            else "unconstrained latent Gaussian; invalid sample fraction is reported"
        ),
        "detail": str(detail_path),
        "summary": str(summary_path),
        "engineering_response_audit": bool(args.engineering_response_audit),
        "engineering_response_only": bool(args.engineering_response_only),
        "engineering_response_file": (
            str(response_path) if bool(args.engineering_response_audit) else ""
        ),
    }
    if bool(args.fit_conformal):
        calibrator, calibration_scenes, calibration_blocks = (
            _fit_cached_spatial_conformal(
                conformal_cache_dir,
                requested_records,
                level=float(args.conformal_level),
                within_block_quantile=float(
                    args.conformal_within_block_quantile
                ),
                std_floor=float(args.conformal_std_floor),
            )
        )
        conformal_path = output_dir / (
            f"m1_geostatistical{transform_suffix}_validation_seed{int(args.seed)}_"
            "spatial_conformal.json"
        )
        conformal_payload = {
            "method": "validation-only-spatial-block-conformal",
            "baseline": (
                "logit-transformed support-conditioned Gaussian random field"
                if args.transform_space == "logit"
                else "support-conditioned Gaussian random field"
            ),
            "checkpoint_seed": int(args.seed),
            "manifest_sha256": str(manifest["manifest_sha256"]),
            "fit_split": "validation",
            "level": float(args.conformal_level),
            "within_block_quantile": float(
                args.conformal_within_block_quantile
            ),
            "std_floor": float(args.conformal_std_floor),
            "block_shape_voxels": [
                int(value) for value in args.conformal_block_shape
            ],
            "calibration_scenes": int(calibration_scenes),
            "calibration_blocks": int(calibration_blocks),
            "global_quantile": float(calibrator.global_quantile),
            "cache_dir": str(conformal_cache_dir),
        }
        conformal_path.write_text(
            json.dumps(conformal_payload, indent=2), encoding="utf-8"
        )
        metadata["fitted_conformal_file"] = str(conformal_path)
        metadata["fitted_conformal_quantile"] = float(
            calibrator.global_quantile
        )
    metadata_path = output_dir / (
        f"m1_geostatistical{transform_suffix}_{args.split}_metadata.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
