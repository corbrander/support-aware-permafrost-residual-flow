from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cold_recon.data.usgs_multiyear_inventory import (
    USGS_MULTIYEAR_SOURCES,
    build_usgs_multiyear_inventory,
    read_usgs_multiyear_ert_conditioning,
    sealed_target_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a value-blind inventory for USGS 2014 and 2016-2017 field campaigns."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="outputs/field_validation_v1/usgs_multiyear")
    parser.add_argument("--max-points-per-line", type=int, default=2500)
    parser.add_argument(
        "--source",
        choices=["all", *USGS_MULTIYEAR_SOURCES],
        default="all",
        help="Prepare both campaigns or one named public source.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_keys = (
        tuple(USGS_MULTIYEAR_SOURCES)
        if args.source == "all"
        else (args.source,)
    )
    inventory = build_usgs_multiyear_inventory(args.raw_dir, source_keys=source_keys)
    sealed = sealed_target_manifest(inventory)
    inventory_path = out_dir / "sciencebase_value_blind_inventory.json"
    sealed_path = out_dir / "sealed_target_manifest.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    sealed_path.write_text(json.dumps(sealed, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(inventory["files"]).to_csv(out_dir / "sciencebase_file_inventory.csv", index=False)

    conditioning_status: dict[str, object]
    try:
        ert, conditioning_status = read_usgs_multiyear_ert_conditioning(
            args.raw_dir,
            inventory=inventory,
            source_keys=source_keys,
            max_points_per_line=args.max_points_per_line,
        )
        group_columns = [
            "source_key",
            "campaign",
            "site_id",
            "line_id",
            "date_group",
            "group_id",
            "source_file",
        ]
        group_index = (
            ert.groupby(group_columns, dropna=False)
            .agg(
                n_points=("log10_resistivity", "size"),
                utm_x_min_m=("utm_x_m", "min"),
                utm_x_max_m=("utm_x_m", "max"),
                utm_y_min_m=("utm_y_m", "min"),
                utm_y_max_m=("utm_y_m", "max"),
                depth_max_m=("depth_m", "max"),
            )
            .reset_index()
        )
        group_index.to_csv(out_dir / "conditioning_ert_group_index.csv", index=False)
    except FileNotFoundError as exc:
        conditioning_status = {
            "conditioning_only": True,
            "target_values_read": False,
            "available": False,
            "reason": str(exc),
        }
    summary = {
        "inventory_path": str(inventory_path),
        "sealed_target_manifest_path": str(sealed_path),
        "inventory_is_value_blind": True,
        "role_counts": inventory["role_counts"],
        "issues": inventory["issues"],
        "conditioning": conditioning_status,
    }
    (out_dir / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
