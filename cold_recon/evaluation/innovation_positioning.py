from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


EVIDENCE_COLUMNS = (
    "method_defined",
    "controlled_validation",
    "baseline_comparison",
    "public_data_evidence",
    "failure_boundary",
    "reproducibility_trace",
)


def _read_csv(table_dir: Path, name: str) -> pd.DataFrame:
    path = table_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(table_dir: Path, name: str) -> dict[str, Any]:
    path = table_dir / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick(df: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    if df.empty or column not in df.columns:
        return None
    rows = df[df[column].astype(str).eq(value)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "not available"
    except TypeError:
        pass
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.{digits}g}"


def _metric(row: pd.Series | None, key: str, digits: int = 4) -> str:
    if row is None or key not in row.index:
        return "not available"
    return _fmt(row[key], digits=digits)


def _coverage_score(row: dict[str, Any]) -> float:
    values = [float(row[col]) for col in EVIDENCE_COLUMNS]
    return float(sum(values) / len(values))


def build_innovation_positioning_audit(table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build an evidence-mapped novelty audit for the COLD-Recon algorithm article.

    The audit is intentionally claim-boundary oriented. It maps each innovation
    dimension to reproducible evidence and records what the evidence does not
    yet prove, so novelty is not asserted rhetorically.
    """

    model = _read_csv(table_dir, "model_comparison.csv")
    tokens = _read_csv(table_dir, "public_data_token_inventory.csv")
    physics = _read_csv(table_dir, "physics_consistency_metrics.csv")
    real_gate = _read_json(table_dir, "real_data_cg_gate.json")
    external = _read_csv(table_dir, "external_generalization_audit.csv")
    domain = _read_json(table_dir, "domain_support_summary.json")
    rare = _read_csv(table_dir, "synthetic_rare_cryostructure_audit.csv")
    hybrid = _read_csv(table_dir, "diffusion_rare_facies_hybrid_metrics.csv")
    voi_boreholes = _read_csv(table_dir, "site_investigation_boreholes.csv")
    voi_lines = _read_csv(table_dir, "site_investigation_ert_lines.csv")

    diffusion = _pick(model, "model", "COLDReconLatentDiffusion")
    fno = _pick(model, "model", "COLDReconFNOOperatorDiffusion")
    gb = _pick(model, "model", "GradientBoosting")
    trained = _pick(model, "model", "COLDReconLatentDiffusionPhysicsTrained")
    refined_phys = _pick(physics, "model", "COLDReconLatentDiffusionPhysicsRefined")
    diffusion_phys = _pick(physics, "model", "COLDReconLatentDiffusion")
    rare_trained = _pick(rare, "model", "COLDReconLatentDiffusionPhysicsTrained")
    rare_hybrid = _pick(hybrid, "model", "COLDReconLatentDiffusionRareFaciesHybrid")
    external_eic = _pick(external, "task", "EIC regression")
    external_wedge = _pick(external, "task", "wedge-ice recall")

    n_public_sources = int(tokens["source_key"].nunique()) if not tokens.empty and "source_key" in tokens.columns else 0
    n_token_types = int(tokens["observation_type"].nunique()) if not tokens.empty and "observation_type" in tokens.columns else 0
    n_public_tokens = int(pd.to_numeric(tokens.get("n_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    top_borehole = voi_boreholes.sort_values("rank").iloc[0] if not voi_boreholes.empty and "rank" in voi_boreholes.columns else None
    top_line = voi_lines.sort_values("rank").iloc[0] if not voi_lines.empty and "rank" in voi_lines.columns else None

    rows: list[dict[str, Any]] = [
        {
            "innovation_dimension": "Multi-source observation tokens",
            "plain_claim": "Heterogeneous borehole, EIC, geophysical and active-layer observations enter a single typed sparse-conditioning interface.",
            "evidence_summary": f"{n_public_tokens} processed public tokens across {n_public_sources} sources and {n_token_types} observation types.",
            "primary_table": "public_data_token_inventory.csv; synthetic_observation_consistency.csv",
            "primary_figure": "public_data_token_inventory.png; synthetic_observation_consistency.png",
            "method_defined": 1.0,
            "controlled_validation": 1.0,
            "baseline_comparison": 0.5,
            "public_data_evidence": 1.0,
            "failure_boundary": 0.5,
            "reproducibility_trace": 1.0,
            "current_maturity": 4.0,
            "eg_target_maturity": 5.0,
            "allowed_claim": "multi-source sparse-conditioning interface",
            "boundary": "The token interface standardizes evidence, but public data still provide partial labels rather than full 3D truth.",
        },
        {
            "innovation_dimension": "Conditional posterior neural operator",
            "plain_claim": "The reconstruction target is sampled as a posterior field, including latent-diffusion and FNO-Transformer operator variants.",
            "evidence_summary": (
                f"Latent diffusion mean IoU {_metric(diffusion, 'mean_iou')}; FNO-operator diffusion "
                f"{_metric(fno, 'mean_iou')}; Gradient Boosting baseline {_metric(gb, 'mean_iou')}."
            ),
            "primary_table": "model_comparison.csv; computational_footprint.csv",
            "primary_figure": "nature_figure_1_overview.*; computational_footprint_summary.*",
            "method_defined": 1.0,
            "controlled_validation": 1.0,
            "baseline_comparison": 1.0,
            "public_data_evidence": 0.5,
            "failure_boundary": 0.5,
            "reproducibility_trace": 1.0,
            "current_maturity": 3.5,
            "eg_target_maturity": 5.0,
            "allowed_claim": "posterior-generative neural-operator reconstruction prototype",
            "boundary": "Full-field performance is proven on synthetic truth; public field data validate partial labels and proxies.",
        },
        {
            "innovation_dimension": "Physics and calibration gates",
            "plain_claim": "Physical consistency and posterior calibration are treated as auditable gates rather than hidden post-processing.",
            "evidence_summary": (
                f"Physics-guided training mean IoU {_metric(trained, 'mean_iou')}; post-hoc projection reduces "
                f"unfrozen-water empirical MAE from {_metric(diffusion_phys, 'unfrozen_water_empirical_mae')} to "
                f"{_metric(refined_phys, 'unfrozen_water_empirical_mae')}."
            ),
            "primary_table": "physics_consistency_metrics.csv; uncertainty_calibration_metrics_calibrated.csv",
            "primary_figure": "nature_figure_1_overview.*; physics_consistency_summary.png",
            "method_defined": 1.0,
            "controlled_validation": 1.0,
            "baseline_comparison": 1.0,
            "public_data_evidence": 0.5,
            "failure_boundary": 1.0,
            "reproducibility_trace": 1.0,
            "current_maturity": 3.5,
            "eg_target_maturity": 5.0,
            "allowed_claim": "physics-checked posterior reconstruction",
            "boundary": "The implemented physics is diagnostic and empirical, not a full coupled thermo-hydrological simulator.",
        },
        {
            "innovation_dimension": "Rare cryostructure operating points",
            "plain_claim": "Rare ice-rich and wedge-ice structures are reported as operating-point trade-offs instead of being hidden by mean IoU.",
            "evidence_summary": (
                f"Synthetic rate-constrained high-EIC recall {_metric(rare_trained, 'rate_constrained_eic_recall')}; "
                f"rare-facies hybrid wedge recall {_metric(rare_hybrid, 'wedge_ice_recall')} and precision "
                f"{_metric(rare_hybrid, 'wedge_ice_precision')}; public wedge recall {_metric(external_wedge, 'model_value')}."
            ),
            "primary_table": "synthetic_rare_cryostructure_audit.csv; arcticdata_wedge_operating_points.csv",
            "primary_figure": "synthetic_rare_cryostructure_audit.*; arcticdata_wedge_operating_curve.*",
            "method_defined": 1.0,
            "controlled_validation": 1.0,
            "baseline_comparison": 1.0,
            "public_data_evidence": 1.0,
            "failure_boundary": 1.0,
            "reproducibility_trace": 1.0,
            "current_maturity": 4.0,
            "eg_target_maturity": 5.0,
            "allowed_claim": "recall-oriented rare-cryostructure handling",
            "boundary": "Precision and false-positive cost remain operating-point choices for site-specific field campaigns.",
        },
        {
            "innovation_dimension": "Public transfer applicability",
            "plain_claim": "Public multi-site transfer is scored with wins, non-inferiority, support classes and explicit guardrails.",
            "evidence_summary": (
                f"Real-data gate {real_gate.get('passed_tasks', 'not available')}/{real_gate.get('total_tasks', 'not available')} tasks; "
                f"EIC non-inferiority rate {_metric(external_eic, 'site_noninferior_rate')}; "
                f"{domain.get('n_model_supported', 'not available')} model-supported and "
                f"{domain.get('n_guarded_local_prior', 'not available')} guarded local-prior sites."
            ),
            "primary_table": "real_data_cg_benchmark.csv; external_generalization_audit.csv; domain_support_site_audit.csv",
            "primary_figure": "nature_figure_2_real_data_gate.*; external_generalization_audit.*; domain_support_audit.*",
            "method_defined": 1.0,
            "controlled_validation": 0.5,
            "baseline_comparison": 1.0,
            "public_data_evidence": 1.0,
            "failure_boundary": 1.0,
            "reproducibility_trace": 1.0,
            "current_maturity": 4.0,
            "eg_target_maturity": 5.0,
            "allowed_claim": "bounded public-data transfer evidence",
            "boundary": "Guarded local-prior sites support non-inferiority, not unqualified model-transfer wins or regional generalization.",
        },
        {
            "innovation_dimension": "Posterior observation design",
            "plain_claim": "Posterior uncertainty is converted into ranked borehole and ERT-line candidates for follow-up site investigation.",
            "evidence_summary": (
                f"Top borehole VOI {_metric(top_borehole, 'voi_score')}; top ERT line score {_metric(top_line, 'line_score')}."
            ),
            "primary_table": "site_investigation_boreholes.csv; site_investigation_ert_lines.csv",
            "primary_figure": "nature_figure_4_site_investigation.*",
            "method_defined": 1.0,
            "controlled_validation": 0.5,
            "baseline_comparison": 0.0,
            "public_data_evidence": 0.5,
            "failure_boundary": 1.0,
            "reproducibility_trace": 1.0,
            "current_maturity": 2.5,
            "eg_target_maturity": 5.0,
            "allowed_claim": "posterior-diagnostic observation recommendation",
            "boundary": "VOI-ranked targets are reproducible hypotheses and have not yet been prospectively field-tested.",
        },
    ]

    for row in rows:
        row["evidence_coverage_score"] = _coverage_score(row)
        row["maturity_gap_to_eg"] = float(row["eg_target_maturity"]) - float(row["current_maturity"])

    audit = pd.DataFrame(rows)
    summary = {
        "n_innovation_dimensions": int(len(audit)),
        "mean_evidence_coverage_score": float(audit["evidence_coverage_score"].mean()),
        "n_dimensions_with_public_data_evidence": int((audit["public_data_evidence"] > 0).sum()),
        "n_dimensions_with_full_boundary_audit": int((audit["failure_boundary"] >= 1.0).sum()),
        "largest_eg_maturity_gap_dimension": str(
            audit.sort_values("maturity_gap_to_eg", ascending=False).iloc[0]["innovation_dimension"]
        ),
        "positioning_boundary": (
            "This audit supports a bounded algorithmic novelty claim; it is not a literature-exhaustive priority claim "
            "and does not convert EG-readiness evidence into prospective regional field validation."
        ),
    }
    return audit, summary


def write_innovation_positioning_outputs(
    audit: pd.DataFrame,
    summary: dict[str, Any],
    table_dir: Path,
    summary_path: Path | None = None,
) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    audit_path = table_dir / "innovation_positioning_audit.csv"
    out_summary = summary_path or table_dir / "innovation_positioning_summary.json"
    audit.to_csv(audit_path, index=False)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit_path, out_summary
