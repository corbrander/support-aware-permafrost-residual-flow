from __future__ import annotations

"""Download selected USGS campaign files without parsing sealed targets."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cold_recon.data.public_sources import _sciencebase_session


DEFAULT_ROLES = (
    "conditioning_ert_inverted",
    "sealed_nmr_inverted_target",
    "support_geometry",
    "sealed_alt_support_attributes",
    "metadata_dictionary",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_dir(role: str) -> str:
    if role == "conditioning_ert_inverted":
        return "conditioning_ert"
    if role.startswith("sealed_"):
        return "sealed_targets"
    if role == "support_geometry":
        return "support_geometry"
    return "metadata_safe"


def _download_one(
    record: dict[str, Any],
    *,
    raw_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    source_key = str(record["source_key"])
    role = str(record["role"])
    destination = raw_dir / source_key / _role_dir(role) / Path(str(record["file_name"])).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(record.get("size_bytes") or 0)
    if destination.is_file() and (expected_size <= 0 or destination.stat().st_size == expected_size):
        return {
            "source_key": source_key,
            "role": role,
            "file_name": destination.name,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "status": "existing_verified",
            "values_parsed": False,
        }
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    url = str(record.get("download_uri") or "")
    if not url:
        raise ValueError(f"missing download_uri for {source_key}/{destination.name}")
    with _sciencebase_session().get(url, stream=True, timeout=int(timeout)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    actual_size = temporary.stat().st_size
    if expected_size > 0 and actual_size != expected_size:
        temporary.unlink(missing_ok=True)
        raise IOError(
            f"size mismatch for {destination.name}: expected {expected_size}, got {actual_size}"
        )
    os.replace(temporary, destination)
    return {
        "source_key": source_key,
        "role": role,
        "file_name": destination.name,
        "path": str(destination),
        "bytes": actual_size,
        "sha256": _sha256(destination),
        "status": "downloaded_verified",
        "values_parsed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download allow-listed USGS multiyear conditioning/support/target bytes. "
            "Sealed target files are hashed but never parsed."
        )
    )
    parser.add_argument(
        "--inventory",
        default="outputs/field_validation_v1/usgs_multiyear/sciencebase_value_blind_inventory.json",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--report",
        default="outputs/field_validation_v1/usgs_multiyear/downloaded_file_hashes.json",
    )
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    roles = set(args.role or DEFAULT_ROLES)
    sources = set(args.source)
    records = [
        row
        for row in inventory.get("files", [])
        if row.get("role") in roles
        and (not sources or row.get("source_key") in sources)
        and row.get("download_uri")
    ]
    if not records:
        raise ValueError("no inventory files matched the requested source/role filters")

    def worker(record: dict[str, Any]) -> dict[str, Any]:
        return _download_one(record, raw_dir=Path(args.raw_dir), timeout=int(args.timeout))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        for index, result in enumerate(executor.map(worker, records), start=1):
            results.append(result)
            print(f"verified {index}/{len(records)} {result['source_key']} {result['file_name']}", flush=True)
    payload = {
        "schema_version": "usgs_multiyear_download_hashes_v1",
        "inventory": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "selected_roles": sorted(roles),
        "selected_sources": sorted(sources) if sources else "all",
        "sealed_values_parsed": False,
        "files": results,
        "total_bytes": int(sum(int(row["bytes"]) for row in results)),
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
