from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cold_recon.data.usgs_multiyear_blind_field import prepare_blind_field_units, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare blind USGS 2014/2016-17 ERT+ALT conditioning and NMR query geometry."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/field_validation_v1/usgs_multiyear/blind_units_v1_2")
    parser.add_argument("--max-ert-points-per-line", type=int, default=5000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty blind preparation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    units, metadata = prepare_blind_field_units(
        args.raw_dir, max_ert_points_per_line=int(args.max_ert_points_per_line)
    )
    manifests = []
    for unit in units:
        unit_dir = output / unit.unit_id
        unit_dir.mkdir(parents=True, exist_ok=False)
        conditioning_path = unit_dir / "prediction_conditioning_ert_alt.npz"
        np.savez_compressed(conditioning_path, **unit.conditioning.as_npz_dict("obs"))
        query_path = unit_dir / "blind_nmr_queries.npz"
        np.savez_compressed(query_path, **unit.query_arrays)
        query_records_path = unit_dir / "blind_query_design_records.csv"
        safe_columns = [
            "campaign", "BoreholeID", "collection_time", "depth_m", "utm_x_m", "utm_y_m",
            "line_id", "distance_to_ert_m", "source_file"
        ]
        unit.query_records[safe_columns].to_csv(query_records_path, index=False)
        pairing_path = unit_dir / "colocation_pairing.csv"
        unit.pairing_records.to_csv(pairing_path, index=False)
        manifest = {
            "unit_id": unit.unit_id,
            "campaign": unit.campaign,
            "line_id": unit.line_id,
            "target_value_columns_read": 0,
            "conditioning_npz": str(conditioning_path.resolve()),
            "conditioning_sha256": sha256_file(conditioning_path),
            "query_npz": str(query_path.resolve()),
            "query_sha256": sha256_file(query_path),
            "query_design_records": str(query_records_path.resolve()),
            "query_design_records_sha256": sha256_file(query_records_path),
            "complete_boreholes": int(len(unit.pairing_records)),
            "query_supports": int(len(unit.query_arrays["query_ids"])),
            "conditioning_observations": int(unit.conditioning.n_obs),
            "source_files": [
                {"path": path, "sha256": sha256_file(path)} for path in unit.source_files
            ],
        }
        manifest_path = unit_dir / "preprediction_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        manifests.append(manifest)
    metadata["units"] = manifests
    metadata_path = output / "PREPREDICTION_VALUE_BLIND_AUDIT.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "prepared_units": len(manifests),
        "prepared_complete_boreholes": metadata["prepared_complete_boreholes"],
        "eligible_campaigns": metadata["eligible_campaigns"],
        "target_value_columns_read": 0,
        "audit_sha256": sha256_file(metadata_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

