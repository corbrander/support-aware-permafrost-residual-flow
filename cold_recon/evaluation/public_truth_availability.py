from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRUTH_REQUIREMENTS = (
    "public_access",
    "three_dimensional_support",
    "observed_or_curated_truth",
    "ground_ice_or_cryofacies_target",
)


@dataclass(frozen=True)
class PublicTruthAvailabilityResult:
    audit: pd.DataFrame
    summary: dict[str, Any]


def _truth_score(row: dict[str, Any]) -> int:
    return int(sum(bool(row.get(key, False)) for key in TRUTH_REQUIREMENTS))


def _local_source_rows(
    provenance: pd.DataFrame,
    availability: dict[str, Any] | None = None,
    coordinate_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    availability = availability or {}
    coordinate_summary = coordinate_summary or {}
    rows: list[dict[str, Any]] = []
    for _, source in provenance.iterrows():
        key = str(source.get("source_key", ""))
        dataset = str(source.get("dataset", ""))
        status = str(source.get("status", ""))
        role = str(source.get("role", ""))
        n_observations = int(float(source.get("n_observations", 0) or 0))
        n_inventory_rows = int(float(source.get("n_inventory_rows", 0) or 0))
        downloaded = status == "downloaded_processed"
        row = {
            "source_key": key,
            "dataset": dataset,
            "source_url": str(source.get("source_url", "")),
            "evidence_class": "local_processed_public_source" if downloaded else "documented_public_interface",
            "public_access": True,
            "downloaded_or_processed": downloaded,
            "n_observations": n_observations,
            "n_inventory_rows": n_inventory_rows,
            "role": role,
            "three_dimensional_support": False,
            "observed_or_curated_truth": False,
            "ground_ice_or_cryofacies_target": False,
            "dense_or_full_field": False,
            "full_field_3d_truth": False,
            "availability_status": "proxy_or_partial",
            "boundary": "",
        }
        if key == "usgs_ert_nmr":
            geo = availability.get("usgs_geophysics_products", {})
            row.update(
                {
                    "three_dimensional_support": False,
                    "observed_or_curated_truth": True,
                    "ground_ice_or_cryofacies_target": False,
                    "dense_or_full_field": False,
                    "n_observations": int(geo.get("n_total_tokens", n_observations) or 0),
                    "boundary": "ERT, NMR and thaw-depth observations are dense proxies, not direct full-field 3D ground-ice or cryofacies truth.",
                }
            )
        elif key == "usgs_eic_cores":
            eic = availability.get("usgs_eic_products", {})
            row.update(
                {
                    "three_dimensional_support": False,
                    "observed_or_curated_truth": True,
                    "ground_ice_or_cryofacies_target": True,
                    "dense_or_full_field": False,
                    "n_observations": int(eic.get("n_intervals", n_observations) or 0),
                    "boundary": "Direct EIC core intervals validate vertical samples, not dense 3D volumes.",
                }
            )
        elif key == "arcticdata_upper_permafrost_cryostratigraphy":
            row.update(
                {
                    "three_dimensional_support": False,
                    "observed_or_curated_truth": True,
                    "ground_ice_or_cryofacies_target": True,
                    "dense_or_full_field": False,
                    "n_observations": int(coordinate_summary.get("n_units", n_inventory_rows) or n_inventory_rows),
                    "boundary": "Georeferenced vertical cryostratigraphy and EIC labels are substantial but remain sparse borehole intervals.",
                }
            )
        elif key == "arcticdata_jago_ground_ice_2018":
            row.update(
                {
                    "three_dimensional_support": False,
                    "observed_or_curated_truth": True,
                    "ground_ice_or_cryofacies_target": True,
                    "dense_or_full_field": False,
                    "boundary": "Independent Jago EIC intervals are useful targeted validation, not a dense 3D truth volume.",
                }
            )
        elif key in {"calm", "gtnp_pangaea", "esa_cci"}:
            row.update(
                {
                    "observed_or_curated_truth": key != "esa_cci",
                    "ground_ice_or_cryofacies_target": False,
                    "dense_or_full_field": key == "esa_cci",
                    "boundary": "This source supports ALT, temperature or permafrost-state context, but not ground-ice or cryofacies full-field 3D truth.",
                }
            )
        else:
            row["boundary"] = "Covariate or interface source; useful for conditioning, not for ground-truth validation."
        row["truth_requirement_score"] = _truth_score(row)
        rows.append(row)
    return rows


def _candidate_rows() -> list[dict[str, Any]]:
    # Candidate sources are recorded as an availability audit, not silently treated as validated inputs.
    rows = [
        {
            "source_key": "utqiagvik_ert_gpr_thaw_probe_2021_2023",
            "dataset": "ERT, GPR and thaw-probe data for near-surface permafrost characterization in Utqiagvik, Alaska, 2021-2023",
            "source_url": "https://doi.org/10.5281/zenodo.17096203",
            "evidence_class": "external_candidate_proxy_source",
            "public_access": True,
            "downloaded_or_processed": False,
            "n_observations": 0,
            "n_inventory_rows": 0,
            "role": "Dense near-surface geophysical and thaw-probe candidate for future observation-design validation",
            "three_dimensional_support": False,
            "observed_or_curated_truth": True,
            "ground_ice_or_cryofacies_target": False,
            "dense_or_full_field": True,
            "full_field_3d_truth": False,
            "availability_status": "candidate_proxy_not_full_3d_truth",
            "boundary": "Dense ERT/GPR/thaw-probe measurements improve spatial proxy validation but do not provide direct dense 3D ground-ice or cryofacies truth.",
        },
        {
            "source_key": "permafrost_discovery_gateway_surface_features",
            "dataset": "Permafrost Discovery Gateway and related mapped Arctic surface-feature products",
            "source_url": "https://arcticdata.io/catalog/portals/permafrost",
            "evidence_class": "external_candidate_surface_label_source",
            "public_access": True,
            "downloaded_or_processed": False,
            "n_observations": 0,
            "n_inventory_rows": 0,
            "role": "Surface-feature labels and regional context for future spatial priors",
            "three_dimensional_support": False,
            "observed_or_curated_truth": True,
            "ground_ice_or_cryofacies_target": False,
            "dense_or_full_field": True,
            "full_field_3d_truth": False,
            "availability_status": "candidate_surface_proxy_not_3d_truth",
            "boundary": "Surface feature labels can constrain terrain and ice-wedge polygon context, but they are not subsurface 3D ground-ice truth.",
        },
    ]
    for row in rows:
        row["truth_requirement_score"] = _truth_score(row)
    return rows


def build_public_truth_availability_audit(
    provenance: pd.DataFrame,
    availability: dict[str, Any] | None = None,
    coordinate_summary: dict[str, Any] | None = None,
) -> PublicTruthAvailabilityResult:
    rows = _local_source_rows(provenance, availability=availability, coordinate_summary=coordinate_summary)
    rows.extend(_candidate_rows())
    audit = pd.DataFrame.from_records(rows)
    if audit.empty:
        summary = {
            "readiness_status": "not_yet",
            "n_sources_audited": 0,
            "n_full_field_3d_truth_sources": 0,
            "n_direct_ground_ice_or_cryofacies_sources": 0,
            "n_dense_proxy_sources": 0,
            "best_truth_requirement_score": 0,
            "readiness_boundary": "No public source audit was available.",
        }
        return PublicTruthAvailabilityResult(audit=audit, summary=summary)

    n_full = int(audit["full_field_3d_truth"].astype(bool).sum())
    n_direct = int(audit["ground_ice_or_cryofacies_target"].astype(bool).sum())
    n_dense_proxy = int((audit["dense_or_full_field"].astype(bool) & ~audit["full_field_3d_truth"].astype(bool)).sum())
    best_score = int(pd.to_numeric(audit["truth_requirement_score"], errors="coerce").fillna(0).max())
    best_sources = (
        audit[pd.to_numeric(audit["truth_requirement_score"], errors="coerce").fillna(0).eq(best_score)]["source_key"]
        .astype(str)
        .tolist()
    )
    summary = {
        "readiness_status": "pass" if n_full > 0 else "not_yet",
        "n_sources_audited": int(len(audit)),
        "n_full_field_3d_truth_sources": n_full,
        "n_direct_ground_ice_or_cryofacies_sources": n_direct,
        "n_dense_proxy_sources": n_dense_proxy,
        "best_truth_requirement_score": best_score,
        "best_available_sources": best_sources,
        "readiness_score": 1.0 if n_full > 0 else 0.0,
        "readiness_boundary": (
            "No audited public source simultaneously provides open access, dense three-dimensional spatial support, "
            "observed or curated truth status, and direct ground-ice/cryofacies/EIC targets. Current evidence combines "
            "direct sparse vertical labels with dense geophysical or surface proxies."
        ),
    }
    return PublicTruthAvailabilityResult(audit=audit, summary=summary)


def write_public_truth_availability_outputs(
    result: PublicTruthAvailabilityResult,
    table_dir: Path,
    summary_path: Path | None = None,
) -> tuple[Path, Path]:
    import json

    table_dir.mkdir(parents=True, exist_ok=True)
    audit_path = table_dir / "public_3d_truth_availability_audit.csv"
    result.audit.to_csv(audit_path, index=False)
    out_summary = summary_path or table_dir / "public_3d_truth_availability_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path, out_summary
