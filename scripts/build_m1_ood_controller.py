from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.ood_control import (
    MahalanobisOODController,
    observation_ood_features,
    scene_ood_features,
)
from scripts.train_m1_support_guided_flow import (
    _load_or_build_prior,
    _manifest_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/m1_support_guided_benchmark/m1_scene_manifest.json",
    )
    parser.add_argument("--reference-scenes", type=int, default=500)
    parser.add_argument("--calibration-split", default="validation")
    parser.add_argument("--rf-trees", type=int, default=24)
    parser.add_argument(
        "--output",
        default=(
            "outputs/m1_support_guided/calibration/"
            "m1_dual_ood_controller_ref500_valcal.npz"
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root, train_records, manifest = _manifest_records(manifest_path, "train")
    train_records = train_records[: int(args.reference_scenes)]
    _, calibration_records, _ = _manifest_records(
        manifest_path, args.calibration_split
    )

    def extract(records: list[dict], label: str) -> tuple[np.ndarray, np.ndarray]:
        observation_rows: list[np.ndarray] = []
        context_rows: list[np.ndarray] = []
        for index, record in enumerate(records):
            sample = load_sample_npz(root / record["relative_path"])
            prior = _load_or_build_prior(
                sample,
                record,
                root / "prior_cache" / record["split"],
                n_facies=7,
                rf_trees=int(args.rf_trees),
            )
            observation_rows.append(
                observation_ood_features(sample["observations"], sample["grid"])
            )
            context_rows.append(
                scene_ood_features(
                    sample["observations"], sample["grid"], prior
                )
            )
            if (index + 1) % 20 == 0:
                print(f"{label} {index + 1}/{len(records)}", flush=True)
        return np.stack(observation_rows), np.stack(context_rows)

    train_observation, train_context = extract(train_records, "reference")
    calibration_observation, calibration_context = extract(
        calibration_records, "calibration"
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
        "manifest_sha256": manifest["manifest_sha256"],
        "reference_split": "train",
        "reference_scenes": len(train_records),
        "calibration_split": args.calibration_split,
        "calibration_scenes": len(calibration_records),
        "rf_trees": int(args.rf_trees),
        "abstention_quantile": 0.99,
        "control_start_quantile": 0.95,
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
