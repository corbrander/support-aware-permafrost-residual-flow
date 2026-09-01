from __future__ import annotations

import argparse
import json
from pathlib import Path

from cold_recon.data.public_sources import (
    PUBLIC_SOURCES,
    download_dataone_package_files,
    download_sciencebase_small_files,
    download_sciencebase_tree_small_files,
    fetch_sciencebase_metadata,
    query_dataone_package,
    write_download_instructions,
    write_public_source_manifest,
)
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-small-files", action="store_true")
    parser.add_argument("--include-children", action="store_true")
    parser.add_argument("--max-size-mb", type=float, default=50.0)
    parser.add_argument("--source", default="all", choices=["all", *PUBLIC_SOURCES.keys()])
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    raw_dir = Path(config["paths"]["raw_dir"])
    external_dir = Path(config["paths"]["external_dir"])
    manifest = write_public_source_manifest(external_dir / "public_sources.json")
    notes = write_download_instructions(external_dir)
    print(f"manifest={manifest}")
    print(f"instructions={notes}")
    targets = [args.source] if args.source != "all" else list(PUBLIC_SOURCES)
    report = {}
    for key in targets:
        item = PUBLIC_SOURCES[key]
        if "sciencebase_id" not in item:
            if "dataone_doi" in item:
                try:
                    package = query_dataone_package(key, raw_dir)
                    report[key] = {
                        "metadata": str(raw_dir / key / "dataone_package_index.json"),
                        "url": item["url"],
                        "dataone_num_found": int(package.get("response", {}).get("numFound", 0)),
                    }
                    print(f"fetched DataONE package {key}: {report[key]['metadata']}")
                    if args.download_small_files and not args.metadata_only:
                        report[key]["files"] = download_dataone_package_files(key, raw_dir, max_size_mb=args.max_size_mb)
                except Exception as exc:
                    report[key] = {"error": repr(exc), "url": item["url"]}
                    print(f"DataONE package fetch failed for {key}: {exc}")
            else:
                report[key] = {"metadata": "manual_or_authenticated_access", "url": item["url"]}
            continue
        try:
            path = fetch_sciencebase_metadata(key, raw_dir)
            report[key] = {"metadata": str(path), "url": item["url"]}
            print(f"fetched {key}: {path}")
            if args.download_small_files and not args.metadata_only:
                if args.include_children:
                    report[key]["files"] = download_sciencebase_tree_small_files(key, raw_dir, max_size_mb=args.max_size_mb)
                else:
                    report[key]["files"] = download_sciencebase_small_files(key, raw_dir, max_size_mb=args.max_size_mb)
        except Exception as exc:
            report[key] = {"error": repr(exc), "url": item["url"]}
            print(f"metadata fetch failed for {key}: {exc}")
    report_path = external_dir / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
