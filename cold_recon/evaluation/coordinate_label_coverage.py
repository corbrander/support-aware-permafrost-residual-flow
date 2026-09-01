from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoordinateLabelCoverageResult:
    site_audit: pd.DataFrame
    summary: dict[str, Any]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_bool_series(values: pd.Series) -> pd.Series:
    return values.map(_truthy).astype(bool)


def _nunique_pairs(df: pd.DataFrame, columns: list[str]) -> int:
    if df.empty or any(col not in df.columns for col in columns):
        return 0
    return int(df[columns].astype(str).drop_duplicates().shape[0])


def build_coordinate_label_coverage_audit(inventory: pd.DataFrame) -> CoordinateLabelCoverageResult:
    """Audit public coordinate and label density for EG-readiness.

    The audit is deliberately bounded. Georeferenced borehole intervals improve
    the surveyed-coordinate evidence, but vertical core labels are still not a
    dense public 3D truth volume or a prospective field validation.
    """
    if inventory.empty:
        return CoordinateLabelCoverageResult(
            site_audit=pd.DataFrame(),
            summary={
                "n_sites": 0,
                "n_units": 0,
                "readiness_status": "missing",
                "readiness_boundary": "missing ArcticData cryostratigraphy inventory",
            },
        )
    required = {"site", "borehole", "has_spatial_coordinates", "cryofacies_class", "has_eic_measurement", "wedge_ice_indicator"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"inventory missing required columns: {sorted(missing)}")

    df = inventory.copy()
    df["has_spatial_coordinates_bool"] = _to_bool_series(df["has_spatial_coordinates"])
    df["has_eic_measurement_bool"] = _to_bool_series(df["has_eic_measurement"])
    df["wedge_ice_indicator_bool"] = _to_bool_series(df["wedge_ice_indicator"])
    df["has_model_facies_label"] = df["cryofacies_class"].astype(str).str.strip().ne("") & df["cryofacies_class"].notna()
    if "high_eic" in df.columns:
        df["high_eic_bool"] = _to_bool_series(df["high_eic"])
    else:
        df["high_eic_bool"] = False

    rows: list[dict[str, object]] = []
    for site, group in df.groupby("site", sort=True):
        n_units = int(len(group))
        n_georef_units = int(group["has_spatial_coordinates_bool"].sum())
        georef_fraction = float(n_georef_units / n_units) if n_units else 0.0
        georef_boreholes = group[group["has_spatial_coordinates_bool"]]
        rows.append(
            {
                "site": str(site),
                "n_units": n_units,
                "n_boreholes": _nunique_pairs(group, ["site", "borehole"]),
                "n_georeferenced_units": n_georef_units,
                "n_georeferenced_boreholes": _nunique_pairs(georef_boreholes, ["site", "borehole"]),
                "georeferenced_unit_fraction": georef_fraction,
                "n_model_facies_units": int(group["has_model_facies_label"].sum()),
                "n_eic_measurements": int(group["has_eic_measurement_bool"].sum()),
                "n_high_eic_units": int(group["high_eic_bool"].sum()),
                "n_wedge_ice_units": int(group["wedge_ice_indicator_bool"].sum()),
                "latitude_min": float(pd.to_numeric(group.get("latitude"), errors="coerce").min(skipna=True)),
                "latitude_max": float(pd.to_numeric(group.get("latitude"), errors="coerce").max(skipna=True)),
                "longitude_min": float(pd.to_numeric(group.get("longitude"), errors="coerce").min(skipna=True)),
                "longitude_max": float(pd.to_numeric(group.get("longitude"), errors="coerce").max(skipna=True)),
            }
        )
    site_audit = pd.DataFrame(rows).sort_values(["n_georeferenced_units", "site"], ascending=[False, True]).reset_index(drop=True)

    n_units = int(len(df))
    n_georef_units = int(df["has_spatial_coordinates_bool"].sum())
    n_sites = int(df["site"].astype(str).nunique())
    n_sites_with_georef = int(site_audit["n_georeferenced_units"].gt(0).sum()) if not site_audit.empty else 0
    n_boreholes = _nunique_pairs(df, ["site", "borehole"])
    n_georef_boreholes = _nunique_pairs(df[df["has_spatial_coordinates_bool"]], ["site", "borehole"])
    n_eic = int(df["has_eic_measurement_bool"].sum())
    n_facies = int(df["has_model_facies_label"].sum())
    n_wedge = int(df["wedge_ice_indicator_bool"].sum())
    n_high_eic = int(df["high_eic_bool"].sum())
    georef_fraction = float(n_georef_units / n_units) if n_units else 0.0
    georef_borehole_fraction = float(n_georef_boreholes / n_boreholes) if n_boreholes else 0.0

    conditional_coordinate_evidence = (
        n_sites_with_georef >= 10
        and n_georef_units >= 1000
        and georef_fraction >= 0.75
        and n_eic >= 500
        and n_wedge >= 100
    )
    readiness_status = "conditional" if conditional_coordinate_evidence else "not_yet"
    summary = {
        "n_sites": n_sites,
        "n_sites_with_georeferenced_units": n_sites_with_georef,
        "n_boreholes": n_boreholes,
        "n_georeferenced_boreholes": n_georef_boreholes,
        "n_units": n_units,
        "n_georeferenced_units": n_georef_units,
        "georeferenced_unit_fraction": georef_fraction,
        "georeferenced_borehole_fraction": georef_borehole_fraction,
        "n_model_facies_units": n_facies,
        "n_eic_measurements": n_eic,
        "n_high_eic_units": n_high_eic,
        "n_wedge_ice_units": n_wedge,
        "readiness_status": readiness_status,
        "readiness_score": 0.5 if readiness_status == "conditional" else 0.0,
        "readiness_boundary": (
            "Public ArcticData provides substantial georeferenced vertical core labels, "
            "but these are still sparse borehole intervals rather than dense public 3D ground truth "
            "or a prospective validation campaign."
        ),
    }
    return CoordinateLabelCoverageResult(site_audit=site_audit, summary=summary)


def write_coordinate_label_coverage_outputs(
    result: CoordinateLabelCoverageResult,
    table_dir: Path,
    summary_path: Path | None = None,
) -> tuple[Path, Path]:
    import json

    table_dir.mkdir(parents=True, exist_ok=True)
    audit_path = table_dir / "coordinate_label_coverage_audit.csv"
    result.site_audit.to_csv(audit_path, index=False)
    out_summary = summary_path or table_dir / "coordinate_label_coverage_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path, out_summary
