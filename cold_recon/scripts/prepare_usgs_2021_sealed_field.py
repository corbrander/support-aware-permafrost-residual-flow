from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from cold_recon.data.usgs_2021_field_loader import (
    USGS_2021_NMR_TARGET_BASENAME,
    expand_usgs_2021_nmr_query_depths,
    is_usgs_2021_sealed_value_path,
    load_usgs_2021_prediction_inputs,
    write_usgs_2021_sealed_target_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare USGS 2021 ERT/ALT conditioning and NMR query geometry while "
            "keeping the NMR inversion target sealed."
        )
    )
    parser.add_argument("--root", default="data/raw/usgs_ert_nmr_2021")
    parser.add_argument(
        "--output", default="outputs/field_validation_v1/usgs_2021_sealed"
    )
    parser.add_argument("--max-ert-points-per-profile", type=int, default=1_000)
    parser.add_argument("--query-depth-step-m", type=float, default=0.125)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    target_manifest_path = output / "sealed_target_manifest.json"
    target_manifest = write_usgs_2021_sealed_target_manifest(root, target_manifest_path)
    if not bool(target_manifest["target"]["complete"]):
        raise RuntimeError(
            f"{USGS_2021_NMR_TARGET_BASENAME} is absent or incomplete; preparation stopped"
        )

    field = load_usgs_2021_prediction_inputs(
        root,
        max_ert_points_per_profile=args.max_ert_points_per_profile,
        require_release_inventory=True,
    )
    conditioning_path = output / "prediction_conditioning_ert_alt.npz"
    np.savez_compressed(
        conditioning_path,
        **field.conditioning.as_npz_dict("obs"),
        utm_origin_m=np.asarray(field.utm_origin_m, dtype=np.float64),
        crs=np.asarray(field.crs),
    )
    records_path = output / "prediction_conditioning_ert_alt_records.csv"
    field.conditioning_records.to_csv(records_path, index=False)
    geometry_path = output / "predeclared_nmr_query_geometry.csv"
    field.nmr_query_geometry.to_csv(geometry_path, index=False)
    query_grid = expand_usgs_2021_nmr_query_depths(
        field.nmr_query_geometry, depth_step_m=args.query_depth_step_m
    )
    query_grid_path = output / "predeclared_nmr_query_depth_grid.csv"
    query_grid.to_csv(query_grid_path, index=False)
    metadata_path = output / "prediction_safe_metadata.json"
    metadata_path.write_text(
        json.dumps(dict(field.metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    safe_paths = [
        conditioning_path,
        records_path,
        geometry_path,
        query_grid_path,
        metadata_path,
    ]
    if any(is_usgs_2021_sealed_value_path(path) for path in safe_paths):
        raise AssertionError("A sealed NMR value file was placed in prediction inputs")
    raw_prediction_sources = sorted(
        {
            Path(value)
            for value in [
                *field.conditioning_records["source_file"].dropna().astype(str).tolist(),
                *field.nmr_query_geometry["source_file"].dropna().astype(str).tolist(),
            ]
        },
        key=lambda path: str(path).lower(),
    )
    if any(is_usgs_2021_sealed_value_path(path) for path in raw_prediction_sources):
        raise AssertionError("A sealed NMR value file entered the raw prediction sources")
    manifest = {
        "schema_version": "usgs-2021-prediction-inputs-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "RETROSPECTIVE_COMPUTATIONAL_PRE_UNSEAL_PREPARATION",
        "blindness_scope": "CURRENT_PIPELINE_ONLY_NOT_PROSPECTIVE_COLLECTION",
        "prediction_inputs_contain_nmr_values": False,
        "conditioning_modalities": ["ERT inverted log10 resistivity", "manual ALT"],
        "query_information": "NMR borehole geometry and fixed depth grid only",
        "target_status": "SEALED_NOT_PARSED_BY_CURRENT_FIELD_VALIDATION_PIPELINE",
        "sealed_target_manifest_sha256": target_manifest["manifest_sha256"],
        "sealed_target_sha256": target_manifest["target"]["sha256"],
        "raw_prediction_sources": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in raw_prediction_sources
        ],
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in safe_paths
        ],
    }
    prediction_manifest_path = output / "prediction_input_manifest.json"
    prediction_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"sealed_target_manifest={target_manifest_path}")
    print(f"prediction_input_manifest={prediction_manifest_path}")
    print(f"conditioning_npz={conditioning_path}")
    print(f"n_conditioning={field.conditioning.n_obs}")
    print(f"n_nmr_query_sites={len(field.nmr_query_geometry)}")
    print(f"n_nmr_query_depths={len(query_grid)}")
    print("target_values_read=0")


if __name__ == "__main__":
    main()
