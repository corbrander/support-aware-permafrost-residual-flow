from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from cold_recon.data.cambridge_bay_eic_loader import (
    DOI,
    SITE_ID,
    cambridge_bay_eic_to_observations,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/cambridge_bay_2024_eic/pangaea_988280.tsv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/cambridge_bay_2024_eic_observations.npz",
    )
    parser.add_argument(
        "--table",
        default="outputs/tables/cambridge_bay_2024_eic_intervals.csv",
    )
    parser.add_argument(
        "--summary",
        default="outputs/tables/cambridge_bay_2024_eic_summary.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    table_path = Path(args.table)
    summary_path = Path(args.summary)
    observations, table = cambridge_bay_eic_to_observations(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **observations.as_npz_dict(),
        site_ids=table["site_id"].to_numpy(dtype=str),
        borehole_ids=table["borehole_id"].to_numpy(dtype=str),
        group_keys=table["group_key"].to_numpy(dtype=str),
        sample_ids=table["Sample ID"].astype(str).to_numpy(),
        latitude=table["Latitude"].to_numpy(dtype=np.float64),
        longitude=table["Longitude"].to_numpy(dtype=np.float64),
        depth_top_m=table["depth_top_m"].to_numpy(dtype=np.float32),
        depth_bottom_m=table["depth_bottom_m"].to_numpy(dtype=np.float32),
        coordinate_sources=table["coordinate_source"].to_numpy(dtype=str),
        source_doi=table["source_doi"].to_numpy(dtype=str),
    )
    table.to_csv(table_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "site_id": SITE_ID,
        "doi": DOI,
        "license": "CC BY 4.0",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "interval_records": int(len(table)),
        "physical_cores": int(table["borehole_id"].nunique()),
        "core_interval_counts": {
            str(key): int(value)
            for key, value in table["borehole_id"].value_counts().items()
        },
        "depth_range_m": [
            float(table["depth_top_m"].min()),
            float(table["depth_bottom_m"].max()),
        ],
        "eic_fraction_range": [
            float(table["eic_fraction"].min()),
            float(table["eic_fraction"].max()),
        ],
        "evidence_role": (
            "independent low-n external EIC site; retained as a three-core shadow audit "
            "and not used to tune gates or fit a site adapter"
        ),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "table": str(table_path),
        "table_sha256": _sha256(table_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
