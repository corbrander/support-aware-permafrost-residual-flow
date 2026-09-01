from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from cold_recon.data.data_schema import OBS_TYPES, load_sample_npz
from cold_recon.data.field_sample_builder import build_public_field_sample
from cold_recon.evaluation.ood_control import (
    MahalanobisOODController,
)
from cold_recon.evaluation.strict_field_validation import (
    DEPLOYMENT_GRID_SCALE_FEATURE_NAMES,
    DEPLOYMENT_OOD_FEATURE_VERSION,
    deployment_ood_features,
)
from cold_recon.utils.config import load_config
from scripts.build_tree_prior_residual_posterior import tree_prior_fields
from scripts.train_m1_support_guided_flow import _manifest_records


ACQUISITION_MASK = ("ert_log_resistivity", "alt")
ACQUISITION_TYPE_IDS = np.asarray(
    [OBS_TYPES[name] for name in ACQUISITION_MASK], dtype=np.int64
)


def _deployment_features(
    *,
    root: Path,
    record: dict,
    config: dict,
    cache_dir: Path,
    rf_trees: int,
    prior_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / record["split"] / f"{record['scene_id']}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            cached_version = (
                str(saved["feature_version"].item())
                if "feature_version" in saved.files
                else ""
            )
            if cached_version == DEPLOYMENT_OOD_FEATURE_VERSION:
                return (
                    np.asarray(saved["observation_features"], dtype=np.float64),
                    np.asarray(saved["context_features"], dtype=np.float64),
                )

    original = load_sample_npz(root / record["relative_path"])
    observations = original["observations"]
    selected = np.flatnonzero(
        observations.mask & np.isin(observations.type_ids, ACQUISITION_TYPE_IDS)
    )
    if len(selected) == 0:
        raise ValueError(f"scene {record['scene_id']} has no ERT+ALT observations")
    masked = observations.subset(selected)
    if not set(ACQUISITION_TYPE_IDS.tolist()).issubset(set(masked.type_ids.tolist())):
        raise ValueError(f"scene {record['scene_id']} is missing a deployment modality")

    # Recreate the exact deploy-time neutral field adapter. Original synthetic
    # volume truth is never passed to either the prior builder or feature code.
    sample = build_public_field_sample(
        masked,
        config,
        site_id=f"ood-cal-{record['scene_id']}",
    )
    prior = tree_prior_fields(
        sample,
        n_facies=int(config["model"]["n_facies"]),
        seed=int(prior_seed),
        rf_trees=int(rf_trees),
        rf_n_jobs=1,
    )
    observation, context = deployment_ood_features(masked, sample["grid"], prior)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        observation_features=observation,
        context_features=context,
        feature_version=np.asarray(DEPLOYMENT_OOD_FEATURE_VERSION),
    )
    return observation, context


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a target-independent OOD controller after applying the frozen "
            "ERT+ALT deployment acquisition mask to train and validation scenes."
        )
    )
    parser.add_argument("--config", default="configs/m1_support_guided.yaml")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--reference-scenes", type=int, default=500)
    parser.add_argument("--calibration-scenes", type=int, default=100)
    parser.add_argument("--rf-trees", type=int, default=24)
    parser.add_argument("--prior-seed", type=int, default=20260830)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel target-independent feature builders; output order remains manifest order.",
    )
    parser.add_argument(
        "--cache-dir",
        default=(
            "outputs/m1_support_guided/calibration/"
            "deployment_ert_alt_feature_cache_gridscale_v1"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/m1_support_guided/calibration/"
            "m1_dual_ood_controller_ert_alt_gridscale_v1_ref500_valcal.npz"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = Path(args.manifest or config["m1_training"]["manifest"])
    root, train_records, manifest = _manifest_records(manifest_path, "train")
    _, validation_records, _ = _manifest_records(manifest_path, "validation")
    train_records = train_records[: int(args.reference_scenes)]
    validation_records = validation_records[: int(args.calibration_scenes)]
    if len(train_records) < int(args.reference_scenes):
        raise ValueError("manifest does not contain the requested reference scenes")
    if len(validation_records) < int(args.calibration_scenes):
        raise ValueError("manifest does not contain the requested calibration scenes")
    cache_dir = Path(args.cache_dir)

    def extract(records: list[dict], label: str) -> tuple[np.ndarray, np.ndarray]:
        observations: list[np.ndarray] = []
        contexts: list[np.ndarray] = []

        def build(record: dict) -> tuple[np.ndarray, np.ndarray]:
            return _deployment_features(
                root=root,
                record=record,
                config=config,
                cache_dir=cache_dir,
                rf_trees=int(args.rf_trees),
                prior_seed=int(args.prior_seed),
            )

        workers = max(int(args.workers), 1)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(build, records)
            for index, (observation, context) in enumerate(results):
                observations.append(observation)
                contexts.append(context)
                if (index + 1) % 10 == 0:
                    print(f"{label} {index + 1}/{len(records)}", flush=True)
        return np.stack(observations), np.stack(contexts)

    train_observation, train_context = extract(train_records, "reference")
    calibration_observation, calibration_context = extract(
        validation_records, "calibration"
    )
    observation_controller = MahalanobisOODController(abstention_quantile=0.99)
    observation_controller.fit(train_observation).calibrate_reference_distances(
        calibration_observation
    )
    context_controller = MahalanobisOODController(abstention_quantile=0.99)
    context_controller.fit(train_context).calibrate_reference_distances(
        calibration_context
    )
    metadata = {
        "method": "validation-calibrated-max-observation-and-prior-context-ood",
        "controller_role": "preregistered-deployment-acquisition-mask",
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "reference_split": "train",
        "reference_scenes": len(train_records),
        "calibration_split": "validation",
        "calibration_scenes": len(validation_records),
        "acquisition_mask": list(ACQUISITION_MASK),
        "target_independent_features": True,
        "feature_version": DEPLOYMENT_OOD_FEATURE_VERSION,
        "grid_scale_feature_names": list(DEPLOYMENT_GRID_SCALE_FEATURE_NAMES),
        "grid_scale_feature_units": ["m"] * len(DEPLOYMENT_GRID_SCALE_FEATURE_NAMES),
        "grid_scale_feature_transform": "none (absolute physical metres)",
        "observation_feature_dimension": int(train_observation.shape[1]),
        "context_feature_dimension": int(train_context.shape[1]),
        "neutral_field_adapter": "build_public_field_sample",
        "rf_trees": int(args.rf_trees),
        "deployment_prior_seed": int(args.prior_seed),
        "feature_build_workers": max(int(args.workers), 1),
        "abstention_quantile": 0.99,
        "control_start_quantile": 0.95,
        "selection_boundary": (
            "The mask and threshold are fixed without field NMR target values. "
            "The original full-acquisition controller remains a mandatory diagnostic."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observation_mean=observation_controller.mean,
        observation_precision=observation_controller.precision,
        observation_reference_distances=observation_controller.reference_distances,
        context_mean=context_controller.mean,
        context_precision=context_controller.precision,
        context_reference_distances=context_controller.reference_distances,
        metadata=np.asarray(json.dumps(metadata)),
    )
    output.with_suffix(".json").write_text(
        json.dumps({**metadata, "artifact": str(output)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({**metadata, "artifact": str(output)}, indent=2))


if __name__ == "__main__":
    main()
