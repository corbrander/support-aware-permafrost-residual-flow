from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATUS_SCORE = {
    "pass": 1.0,
    "conditional": 0.5,
    "not_yet": 0.0,
    "missing": 0.0,
}


@dataclass(frozen=True)
class JournalReadinessResult:
    audit: pd.DataFrame
    summary: dict[str, Any]


def _finite(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _task_row(df: pd.DataFrame, task: str) -> pd.Series | None:
    if df.empty or "task" not in df.columns:
        return None
    rows = df[df["task"].astype(str).eq(task)]
    return rows.iloc[0] if not rows.empty else None


def _count_status(audit: pd.DataFrame, tier: str, status: str) -> int:
    if audit.empty:
        return 0
    rows = audit[audit["readiness_tier"].astype(str).eq(tier)]
    return int(rows["status"].astype(str).eq(status).sum())


def _score_for_status(status: str) -> float:
    return float(STATUS_SCORE.get(str(status), 0.0))


def _row(
    tier: str,
    criterion: str,
    status: str,
    evidence: str,
    current_value: str,
    required_value: str,
    boundary: str,
    artifacts: str,
) -> dict[str, object]:
    return {
        "readiness_tier": tier,
        "criterion": criterion,
        "status": status,
        "status_score": _score_for_status(status),
        "current_value": current_value,
        "required_value": required_value,
        "evidence": evidence,
        "boundary": boundary,
        "source_artifacts": artifacts,
    }


def build_journal_readiness_audit(
    gate_summary: dict[str, Any],
    real_benchmark: pd.DataFrame,
    external_generalization: pd.DataFrame,
    figure_atlas: pd.DataFrame,
    reproducibility_summary: dict[str, Any] | None = None,
    domain_support_summary: dict[str, Any] | None = None,
    coordinate_label_summary: dict[str, Any] | None = None,
    voi_backtest_summary: dict[str, Any] | None = None,
) -> JournalReadinessResult:
    """Build a bounded CG/EG readiness audit from existing reproducibility outputs.

    The audit is intentionally conservative: CG criteria can pass with a complete
    reproducible algorithm evidence chain, whereas EG criteria distinguish public
    transfer evidence from a stronger regional field-generalization claim.
    """
    repro = reproducibility_summary or {}
    domain = domain_support_summary or {}
    coordinate = coordinate_label_summary or {}
    voi = voi_backtest_summary or {}
    rows: list[dict[str, object]] = []
    passed_tasks = int(gate_summary.get("passed_tasks", 0) or 0)
    total_tasks = int(gate_summary.get("total_tasks", 0) or 0)
    sources = int(gate_summary.get("independent_public_sources_passed", 0) or 0)
    eic_sources = int(gate_summary.get("eic_sources_passed", 0) or 0)
    cg_gate_passed = _bool(gate_summary.get("cg_model_evidence_passed", False))
    has_repro_summary = bool(repro)
    repro_passed = _bool(repro.get("passed", False))
    missing_required = int(repro.get("n_missing_required", 0) or 0)
    n_artifacts = int(repro.get("n_artifacts", 0) or 0)

    claim_figures = 0
    excluded_figures = 0
    if not figure_atlas.empty:
        claim_figures = int(figure_atlas["copy_to_submission"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
        excluded_figures = int(figure_atlas["manuscript_status"].astype(str).eq("scope_boundary_excluded").sum())

    eic = _task_row(external_generalization, "EIC regression")
    facies = _task_row(external_generalization, "cryofacies")
    high_eic = _task_row(external_generalization, "high-EIC event")
    wedge = _task_row(external_generalization, "wedge-ice recall")
    eic_noninferior = _finite(eic["site_noninferior_rate"]) if eic is not None and "site_noninferior_rate" in eic else float("nan")
    eic_win = _finite(eic["site_win_rate"]) if eic is not None and "site_win_rate" in eic else float("nan")
    facies_win = _finite(facies["site_win_rate"]) if facies is not None and "site_win_rate" in facies else float("nan")
    high_eic_noninferior = _finite(high_eic["site_noninferior_rate"]) if high_eic is not None and "site_noninferior_rate" in high_eic else float("nan")
    wedge_noninferior = _finite(wedge["site_noninferior_rate"]) if wedge is not None and "site_noninferior_rate" in wedge else float("nan")

    cg_status = "pass" if cg_gate_passed else "missing"
    rows.append(
        _row(
            "CG algorithm article",
            "Three-source public-data evidence gate",
            cg_status,
            f"{sources} independent sources; {passed_tasks}/{total_tasks} validation tasks passed; {eic_sources} EIC sources.",
            f"sources={sources}; tasks={passed_tasks}/{total_tasks}; EIC sources={eic_sources}",
            "at least 3 independent sources, at least 7 passed tasks and at least 3 EIC sources",
            "Supports a computational geoscience algorithm manuscript; not by itself a regional map claim.",
            "outputs/tables/real_data_cg_gate.json; outputs/tables/real_data_cg_benchmark.csv",
        )
    )
    rows.append(
        _row(
            "CG algorithm article",
            "External transfer boundary is quantified",
            "pass" if external_generalization is not None and not external_generalization.empty and np.isfinite(eic_noninferior) and eic_noninferior >= 1.0 else "conditional",
            f"EIC non-inferiority rate={eic_noninferior:.2f}; EIC site win rate={eic_win:.2f}; facies site win rate={facies_win:.2f}.",
            f"EIC non-inferiority={eic_noninferior:.2f}; EIC wins={eic_win:.2f}",
            "site-wise wins, non-inferiority and failure labels reported",
            "Guarded non-inferior ties remain distinct from broad field-transfer wins.",
            "outputs/tables/external_generalization_audit.csv; outputs/tables/transfer_failure_site_diagnostics.csv",
        )
    )
    domain_all_ok = _bool(domain.get("all_sites_noninferior_all_evaluated_tasks", False))
    low_support = int(domain.get("n_low_support", 0) or 0)
    guarded_sites = domain.get("guarded_sites", [])
    rows.append(
        _row(
            "CG algorithm article",
            "Domain-support applicability gate",
            "pass" if domain_all_ok and low_support == 0 else ("conditional" if domain else "missing"),
            f"all evaluated site-tasks non-inferior={domain_all_ok}; low-support sites={low_support}; guarded sites={len(guarded_sites)}.",
            "no low-support transfer sites and all evaluated site-tasks non-inferior",
            "Train-side support scores and adaptive guard classes are reported before interpreting holdout outcomes.",
            "Guarded sites support non-inferiority claims, not unqualified transfer wins.",
            "outputs/tables/domain_support_site_audit.csv; outputs/tables/domain_support_summary.json",
        )
    )
    rows.append(
        _row(
            "CG algorithm article",
            "Rare cryostructure and high-EIC stress tests are separated",
            "pass"
            if np.isfinite(high_eic_noninferior)
            and high_eic_noninferior >= 1.0
            and np.isfinite(wedge_noninferior)
            and wedge_noninferior >= 1.0
            else "conditional",
            f"high-EIC non-inferiority={high_eic_noninferior:.2f}; wedge-recall non-inferiority={wedge_noninferior:.2f}.",
            f"high-EIC={high_eic_noninferior:.2f}; wedge={wedge_noninferior:.2f}",
            "high-EIC screening, wedge recall and precision trade-off reported separately",
            "This remains an operating-point analysis, not a solved rare-facies reconstruction proof.",
            "outputs/tables/arcticdata_wedge_operating_curve.csv; outputs/tables/synthetic_rare_cryostructure_audit.csv",
        )
    )
    rows.append(
        _row(
            "CG algorithm article",
            "Complete figure evidence chain",
            "pass" if claim_figures >= 50 and excluded_figures >= 1 else "conditional",
            f"{claim_figures} algorithm/public-data figure stems copied; {excluded_figures} scope-boundary figures excluded.",
            f"claim figures={claim_figures}; excluded={excluded_figures}",
            "all generated figures indexed with claim role and scope status",
            "Application figures are retained as boundary records, not used as claims.",
            "outputs/tables/figure_atlas.csv; paper/supplementary_figure_atlas.md",
        )
    )
    rows.append(
        _row(
            "CG algorithm article",
            "Reproducible package closure",
            "pass" if repro_passed and missing_required == 0 else ("conditional" if not has_repro_summary else "missing"),
            f"{n_artifacts} required artifacts audited; missing_required={missing_required}.",
            f"artifacts={n_artifacts}; missing={missing_required}",
            "zero missing required artifacts in reproducibility audit",
            "When the reproducibility summary is unavailable, this criterion remains conditional until the final audit is run.",
            "outputs/tables/reproducibility_summary.json; outputs/tables/reproducibility_manifest.csv",
        )
    )

    eg_public_status = "pass" if sources >= 3 and passed_tasks >= 8 else "conditional"
    rows.append(
        _row(
            "EG field-generalization claim",
            "Independent public-data breadth",
            eg_public_status,
            f"{sources} public sources and {passed_tasks}/{total_tasks} tasks passed.",
            f"sources={sources}; tasks={passed_tasks}/{total_tasks}",
            "multiple independent public sources with same-split baselines",
            "The breadth is sufficient for external algorithm stress testing but still small for regional field claims.",
            "outputs/tables/real_data_cg_benchmark.csv",
        )
    )
    rows.append(
        _row(
            "EG field-generalization claim",
            "Site-wise transfer robustness",
            "conditional" if np.isfinite(eic_noninferior) and eic_noninferior >= 1.0 else "not_yet",
            f"EIC non-inferiority={eic_noninferior:.2f}; EIC site wins={eic_win:.2f}; high-EIC non-inferiority={high_eic_noninferior:.2f}.",
            f"non-inferior={eic_noninferior:.2f}; wins={eic_win:.2f}",
            "high site win rates across independent regions, not only non-inferior guarded ties",
            "Current transfer evidence is guarded and heterogeneous; it supports EG-readiness, not an unqualified EG field claim.",
            "outputs/tables/external_generalization_audit.csv",
        )
    )
    rows.append(
        _row(
            "EG field-generalization claim",
            "Prospective applicability gate",
            "conditional" if domain and low_support == 0 else "not_yet",
            f"guarded local-prior sites={len(guarded_sites)}; all evaluated site-tasks non-inferior={domain_all_ok}.",
            "pre-registered applicability thresholds validated on future independent sites",
            "Current support scoring is retrospective over public holdouts; it is useful for deployment triage but not yet a prospective validation rule.",
            "The domain-support audit improves EG-readiness but does not replace prospective field validation.",
            "outputs/tables/domain_support_site_audit.csv; outputs/figures/domain_support_audit.*",
        )
    )
    coordinate_status = str(coordinate.get("readiness_status", "not_yet"))
    coordinate_current = (
        f"{int(coordinate.get('n_georeferenced_units', 0) or 0)} georeferenced ArcticData units across "
        f"{int(coordinate.get('n_sites_with_georeferenced_units', 0) or 0)} sites; "
        f"{int(coordinate.get('n_eic_measurements', 0) or 0)} EIC measurements; "
        f"{int(coordinate.get('n_wedge_ice_units', 0) or 0)} wedge-ice units."
        if coordinate
        else "partial public labels; ordered borehole convention for some branches"
    )
    rows.append(
        _row(
            "EG field-generalization claim",
            "Surveyed coordinates and dense labels",
            "conditional" if coordinate_status == "conditional" else "not_yet",
            (
                "Public ArcticData provides substantial georeferenced vertical cryostratigraphy labels, "
                "but these labels are still borehole intervals rather than dense gridded 3D truth."
                if coordinate
                else "USGS/Jago branches include partial public labels; processed releases lack the dense surveyed labels needed for regional 3D validation."
            ),
            coordinate_current,
            "surveyed spatial coordinates and denser cryofacies/EIC labels across independent field sites",
            (
                str(coordinate.get("readiness_boundary"))
                if coordinate
                else "This is a data-availability boundary, not a model failure."
            ),
            "outputs/tables/coordinate_label_coverage_audit.csv; outputs/tables/coordinate_label_coverage_summary.json",
        )
    )
    rows.append(
        _row(
            "EG field-generalization claim",
            "Prospective VOI validation",
            "conditional" if str(voi.get("readiness_status", "")) == "conditional" else "not_yet",
            (
                "Synthetic full-field retrospective VOI backtest shows error enrichment in high-ranked targets, "
                "but no prospective field campaign has tested the acquisitions."
                if voi
                else "VOI-ranked boreholes and ERT lines are generated, but no prospective field campaign has tested them."
            ),
            (
                f"composite top-VOI enrichment={_finite(voi.get('composite_top_voi_error_enrichment')):.2f}; "
                f"high-EIC top-VOI enrichment={_finite(voi.get('high_eic_top_voi_error_enrichment')):.2f}"
                if voi
                else "posterior-diagnostic hypotheses only"
            ),
            "prospective comparison of VOI-ranked observations against conventional site-investigation layouts",
            (
                str(voi.get("readiness_boundary"))
                if voi
                else "The VOI layer is reproducible observation design, not proven acquisition optimization."
            ),
            "outputs/tables/site_investigation_boreholes.csv; outputs/tables/site_investigation_ert_lines.csv; outputs/tables/voi_backtest_audit.csv",
        )
    )
    rows.append(
        _row(
            "EG field-generalization claim",
            "Full-field public 3D ground truth",
            "not_yet",
            "Full-field 3D validation remains synthetic because public releases provide partial labels and proxies.",
            "synthetic full-field truth; public partial labels",
            "independent 3D ground-truth volumes or dense validation grids",
            "The paper should not be framed as a regional ground-ice map or fully validated field model.",
            "outputs/tables/model_comparison.csv; outputs/tables/real_data_cg_benchmark.csv",
        )
    )

    audit = pd.DataFrame(rows)
    summary_rows = []
    for tier, group in audit.groupby("readiness_tier", sort=False):
        score = float(group["status_score"].mean()) if len(group) else 0.0
        summary_rows.append(
            {
                "tier": tier,
                "score": score,
                "n_criteria": int(len(group)),
                "n_pass": _count_status(audit, tier, "pass"),
                "n_conditional": _count_status(audit, tier, "conditional"),
                "n_not_yet": _count_status(audit, tier, "not_yet"),
                "ready_claim": bool(group["status"].astype(str).eq("pass").all()),
            }
        )
    summary = {
        "tiers": summary_rows,
        "cg_algorithm_article_ready": bool(
            audit[audit["readiness_tier"].eq("CG algorithm article")]["status"].astype(str).isin(["pass", "conditional"]).all()
            and _count_status(audit, "CG algorithm article", "pass") >= 4
        ),
        "eg_field_generalization_ready": bool(
            audit[audit["readiness_tier"].eq("EG field-generalization claim")]["status"].astype(str).eq("pass").all()
        ),
        "recommended_positioning": (
            "CG-plus algorithm manuscript with explicit EG-readiness evidence and boundaries; "
            "do not claim a completed EG regional field-generalization study."
        ),
    }
    return JournalReadinessResult(audit=audit, summary=summary)


def write_journal_readiness_outputs(
    result: JournalReadinessResult,
    table_dir: Path,
    summary_path: Path | None = None,
) -> tuple[Path, Path]:
    import json

    table_dir.mkdir(parents=True, exist_ok=True)
    audit_path = table_dir / "journal_readiness_audit.csv"
    result.audit.to_csv(audit_path, index=False)
    out_summary = summary_path or table_dir / "journal_readiness_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path, out_summary
