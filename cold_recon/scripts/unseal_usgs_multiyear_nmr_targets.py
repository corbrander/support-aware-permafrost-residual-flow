from __future__ import annotations

"""Unseal USGS NMR values only after every declared prediction hash verifies."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_date(values: pd.Series, campaign: str) -> pd.Series:
    if campaign == "2014":
        return pd.Series(
            pd.Timestamp("2014-08-30", tz="UTC"), index=values.index
        )
    raw = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = raw.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    parsed.loc[compact] = pd.to_datetime(
        raw.loc[compact], format="%Y%m%d", errors="coerce", utc=True
    )
    parsed.loc[~compact] = pd.to_datetime(raw.loc[~compact], errors="coerce", utc=True)
    return parsed


def _query_ids(frame: pd.DataFrame) -> np.ndarray:
    base = (
        frame["campaign"].astype(str)
        + "::"
        + frame["BoreholeID"].astype(str)
        + "::"
        + frame["collection_time"].dt.strftime("%Y-%m-%d")
        + "::"
        + frame["depth_m"].map(lambda value: f"{float(value):.4f}m")
    )
    counts: dict[str, int] = {}
    ids = []
    for value in base:
        ordinal = counts.get(value, 0)
        counts[value] = ordinal + 1
        ids.append(value if ordinal == 0 else f"{value}::{ordinal}")
    return np.asarray(ids, dtype="U")


def _read_target_container(path: Path, campaign: str) -> tuple[pd.DataFrame, dict]:
    header = pd.read_csv(path, nrows=0)
    columns = [str(column).strip() for column in header.columns]
    target_columns = [column for column in columns if column.casefold().startswith("totalwatercontent")]
    if len(target_columns) != 1:
        raise ValueError(f"{path.name} must have exactly one TotalWaterContent column")
    target_column = target_columns[0]
    safe = [
        column
        for column in columns
        if column
        in {
            "BoreholeID",
            "Lat_WGS84dd",
            "Lon_WGS84dd",
            "Lon_WGS84_dd",
            "CollectionDate",
            "Depth(cm)",
        }
    ]
    selected = [*safe, target_column]
    frame = pd.read_csv(path, usecols=selected)
    longitude = "Lon_WGS84dd" if "Lon_WGS84dd" in frame else "Lon_WGS84_dd"
    frame["BoreholeID"] = frame["BoreholeID"].astype(str).str.strip()
    frame["depth_m"] = pd.to_numeric(frame["Depth(cm)"], errors="coerce") / 100.0
    frame["latitude"] = pd.to_numeric(frame["Lat_WGS84dd"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame[longitude], errors="coerce")
    if "CollectionDate" in frame:
        frame["collection_time"] = _parse_date(frame["CollectionDate"], campaign)
    else:
        frame["collection_time"] = _parse_date(
            pd.Series("", index=frame.index), campaign
        )
    values = pd.to_numeric(frame[target_column], errors="coerce")
    unit_rule = "identity_fraction"
    if "(%)" in target_column:
        values = values / 100.0
        unit_rule = "divide_percent_by_100"
    frame["target_value"] = values
    frame["campaign"] = campaign
    valid = (
        frame["BoreholeID"].ne("")
        & np.isfinite(frame["depth_m"])
        & np.isfinite(frame["latitude"])
        & np.isfinite(frame["longitude"])
        & frame["collection_time"].notna()
        & np.isfinite(frame["target_value"])
    )
    frame = frame.loc[valid].copy()
    if np.any((frame["target_value"] < 0.0) | (frame["target_value"] > 1.0)):
        raise ValueError(f"{path.name} has target values outside [0, 1] after frozen conversion")
    audit = {
        "source_file": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "target_column_read": target_column,
        "unit_rule": unit_rule,
        "rows_read": int(len(frame)),
    }
    return frame, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify sealed predictions, then unseal matching USGS NMR target values."
    )
    parser.add_argument(
        "--units-dir",
        default="outputs/field_validation_v1/usgs_multiyear/blind_units_v1_2b",
    )
    args = parser.parse_args()
    units_root = Path(args.units_dir)
    units = sorted(path for path in units_root.iterdir() if path.is_dir())
    verified: list[tuple[Path, dict, dict]] = []
    # Complete the integrity phase for every unit before opening any target CSV.
    for unit in units:
        prep = json.loads((unit / "preprediction_manifest.json").read_text(encoding="utf-8"))
        prediction_path = unit / "strict_prediction/prediction_manifest.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        sealed = Path(prediction["sealed_prediction_file"])
        actual = sha256_file(sealed)
        if actual != prediction["sealed_prediction_sha256"]:
            raise ValueError(f"prediction seal mismatch for {unit.name}")
        if prediction.get("phase") != "PREDICT_SEALED" or prediction.get("truth_accessed"):
            raise ValueError(f"unit is not in sealed pre-unseal state: {unit.name}")
        verified.append((unit, prep, prediction))

    cache: dict[tuple[str, str], tuple[pd.DataFrame, dict]] = {}
    summaries = []
    for unit, prep, prediction in verified:
        with np.load(prep["query_npz"], allow_pickle=False) as query_data:
            query_ids = np.asarray(query_data["query_ids"]).astype(str)
        design = pd.read_csv(unit / "blind_query_design_records.csv")
        source_files = sorted(set(design["source_file"].astype(str)))
        frames = []
        audits = []
        for source_file in source_files:
            key = (source_file, prep["campaign"])
            if key not in cache:
                cache[key] = _read_target_container(Path(source_file), prep["campaign"])
            frame, audit = cache[key]
            frames.append(frame)
            audits.append(audit)
        target = pd.concat(frames, ignore_index=True)
        target = target.loc[target["BoreholeID"].astype(str).isin(set(design["BoreholeID"].astype(str)))].copy()
        target = target.sort_values(["BoreholeID", "collection_time", "depth_m"]).reset_index(drop=True)
        target_ids = _query_ids(target)
        if not np.array_equal(target_ids, query_ids):
            missing = sorted(set(query_ids).difference(target_ids))
            extra = sorted(set(target_ids).difference(query_ids))
            raise ValueError(
                f"unsealed target/query mismatch for {unit.name}: missing={missing}, extra={extra}"
            )
        output = unit / "unsealed_targets"
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite unsealed targets: {output}")
        output.mkdir(parents=True, exist_ok=True)
        target_path = output / "nmr_target_values.npz"
        metadata = {
            "protocol": "FieldValidation-v1.2-20260831",
            "prediction_hash_verified_before_target_access": True,
            "sealed_prediction_sha256": prediction["sealed_prediction_sha256"],
            "unit_id": unit.name,
            "source_audits": audits,
            "claim_boundary": "NMR support truth only; not EIC/core/topology validation",
        }
        np.savez_compressed(
            target_path,
            target_query_ids=target_ids.astype("U"),
            target_values=target["target_value"].to_numpy(dtype=np.float32),
            target_sigma=np.full(len(target), np.nan, dtype=np.float32),
            target_metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        manifest = {
            "phase": "TARGET_UNSEALED_AFTER_PREDICTION_HASH_VERIFICATION",
            "unit_id": unit.name,
            "prediction_hash_verified": prediction["sealed_prediction_sha256"],
            "target_file": str(target_path.resolve()),
            "target_sha256": sha256_file(target_path),
            "target_rows": int(len(target)),
            "target_min": float(target["target_value"].min()),
            "target_max": float(target["target_value"].max()),
            "source_audits": audits,
        }
        manifest_path = output / "UNSEAL_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append(manifest)
    print(json.dumps({
        "phase": "TARGETS_UNSEALED_AFTER_ALL_PREDICTION_SEALS_VERIFIED",
        "units": len(summaries),
        "target_rows": sum(item["target_rows"] for item in summaries),
        "manifests": summaries,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

