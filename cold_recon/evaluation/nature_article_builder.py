from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cold_recon.evaluation.paper_builder import REFERENCE_BIBTEX, references_markdown, write_references_bib

PRIMARY_SUBMISSION_FIGURE_STEMS = {
    "nature_figure_1_overview",
    "nature_figure_2_real_data_gate",
    "nature_figure_3_cited_ground_ice",
    "nature_figure_4_site_investigation",
    "innovation_positioning_audit",
    "external_generalization_audit",
    "transfer_failure_attribution",
    "domain_support_audit",
    "journal_readiness_audit",
    "coordinate_label_coverage_audit",
    "voi_backtest_audit",
}


def read_table(table_dir: Path, name: str) -> pd.DataFrame:
    path = table_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pick_row(df: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    if df.empty or column not in df.columns:
        return None
    rows = df[df[column].astype(str) == value]
    if rows.empty:
        return None
    return rows.iloc[0]


def pick_model_target(df: pd.DataFrame, model: str, target: str) -> pd.Series | None:
    if df.empty or "model" not in df.columns or "target" not in df.columns:
        return None
    rows = df[df["model"].astype(str).eq(model) & df["target"].astype(str).eq(target)]
    if rows.empty:
        return None
    return rows.iloc[0]


def pick_numeric_row(df: pd.DataFrame, column: str, value: float, decimals: int = 6) -> pd.Series | None:
    if df.empty or column not in df.columns:
        return None
    numeric = pd.to_numeric(df[column], errors="coerce")
    rows = df[numeric.round(decimals).eq(round(float(value), decimals))]
    if rows.empty:
        return None
    return rows.iloc[0]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "not available"
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits == 0:
        return f"{numeric:,.0f}"
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.{digits}g}"


def metric(row: pd.Series | None, key: str, digits: int = 4) -> str:
    if row is None or key not in row.index:
        return "not available"
    return fmt(row[key], digits=digits)


def text_or_none(row: pd.Series | None, key: str) -> str:
    if row is None or key not in row.index or pd.isna(row[key]):
        return "none"
    text = str(row[key]).strip()
    return text if text else "none"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sentence_case_from_stem(stem: str) -> str:
    words = [word for word in stem.replace("-", "_").split("_") if word]
    preserve = {"3d": "3D", "eic": "EIC", "ert": "ERT", "fno": "FNO", "idw": "IDW", "nmr": "NMR", "usgs": "USGS", "voi": "VOI"}
    text = " ".join(preserve.get(word.lower(), word) for word in words)
    return text[:1].upper() + text[1:]


def _submission_caption(text: Any) -> str:
    caption = str(text).strip()
    replacements = {
        "supplemental observation": "additional observation",
        "supplemental boreholes": "additional boreholes",
        "supplemental ERT": "additional ERT",
        "Supplemental observation": "Additional observation",
        "Supplemental boreholes": "Additional boreholes",
        "Supplemental ERT": "Additional ERT",
    }
    for old, new in replacements.items():
        caption = caption.replace(old, new)
    return caption


def comprehensive_submission_figure_lines(table_dir: Path, start_number: int = 10) -> list[str]:
    atlas = read_table(table_dir, "figure_atlas.csv")
    if atlas.empty or "copy_to_submission" not in atlas.columns:
        return []

    rows = atlas[atlas["copy_to_submission"].map(_truthy)].copy()
    if "stem" in rows.columns:
        rows = rows[~rows["stem"].astype(str).isin(PRIMARY_SUBMISSION_FIGURE_STEMS)]
    if "manuscript_status" in rows.columns:
        rows = rows[~rows["manuscript_status"].astype(str).eq("scope_boundary_excluded")]
    if rows.empty:
        return []
    sort_columns = [col for col in ["category_order", "category_key", "stem"] if col in rows.columns]
    if sort_columns:
        rows = rows.sort_values(sort_columns, kind="stable")

    lines = [
        "### Comprehensive Evidence Figures",
        "",
        (
            "The following figures are part of the main submission draft. They include every algorithm, public-data, "
            "physics, uncertainty, rare-cryostructure, observation-design and boundary-audit figure marked "
            "`copy_to_submission=True` in `outputs/tables/figure_atlas.csv`, excluding only the primary narrative "
            "figures already shown above."
        ),
        "",
    ]
    figure_number = start_number
    for _, row in rows.iterrows():
        preferred_path = str(row.get("preferred_path", "")).strip()
        stem = str(row.get("stem", "")).strip()
        if not preferred_path or not stem:
            continue
        title = _sentence_case_from_stem(stem)
        caption = _submission_caption(row.get("caption", title))
        claim_role = str(row.get("claim_role", "")).strip()
        category = str(row.get("category", "")).strip()
        details = []
        if claim_role:
            details.append(f"Claim role: {claim_role}.")
        if category:
            details.append(f"Evidence class: {category}.")
        lines.extend(
            [
                f"![Figure {figure_number}. {title}.](../{preferred_path})",
                "",
                f"**Figure {figure_number} | {title}.** {caption} {' '.join(details)}".strip(),
                "",
            ]
        )
        figure_number += 1
    return lines


def _read_gate(table_dir: Path) -> dict[str, Any]:
    path = table_dir / "real_data_cg_gate.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_json(table_dir: Path, name: str) -> dict[str, Any]:
    path = table_dir / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _source_token_count(tokens: pd.DataFrame, source_key: str, observation_type: str) -> str:
    if tokens.empty:
        return "not available"
    rows = tokens[
        tokens["source_key"].astype(str).eq(source_key)
        & tokens["observation_type"].astype(str).eq(observation_type)
    ]
    if rows.empty:
        return "not available"
    return fmt(rows.iloc[0]["n_tokens"], digits=0)


DOMAIN_OUTCOME_COLUMNS = ("facies_outcome", "eic_outcome", "high_eic_outcome", "wedge_outcome")


def _domain_support_counts(domain_support: pd.DataFrame) -> dict[str, Any]:
    if domain_support.empty or "applicability_class" not in domain_support.columns:
        return {
            "model_supported": "not available",
            "guarded": "not available",
            "low": "not available",
            "all_noninferior": False,
        }
    classes = domain_support["applicability_class"].astype(str)
    outcome_cols = [col for col in DOMAIN_OUTCOME_COLUMNS if col in domain_support.columns]
    all_noninferior = False
    if outcome_cols:
        outcomes = domain_support[outcome_cols].astype(str)
        all_noninferior = bool(outcomes.isin(["win", "noninferior", "not_evaluated"]).all().all())
    guarded_sites = ""
    if "site" in domain_support.columns:
        guarded_sites = ", ".join(
            domain_support.loc[classes.eq("guarded local-prior"), "site"].astype(str).tolist()
        )
    return {
        "model_supported": int(classes.eq("model-supported transfer").sum()),
        "guarded": int(classes.eq("guarded local-prior").sum()),
        "low": int(classes.eq("low support").sum()),
        "all_noninferior": all_noninferior,
        "guarded_sites": guarded_sites or "none",
    }


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


REFERENCE_NUMBERS = {key: idx for idx, key in enumerate(REFERENCE_BIBTEX, start=1)}


def cite(*keys: str) -> str:
    numbers = [str(REFERENCE_NUMBERS[key]) for key in keys if key in REFERENCE_NUMBERS]
    return "[" + ",".join(numbers) + "]"


def build_nature_article(table_dir: Path, output_path: Path) -> Path:
    paper_dir = output_path.parent
    write_references_bib(paper_dir)

    model = read_table(table_dir, "model_comparison.csv")
    calib = read_table(table_dir, "uncertainty_calibration_metrics_calibrated.csv")
    physics = read_table(table_dir, "physics_consistency_metrics.csv")
    computational_footprint = read_table(table_dir, "computational_footprint.csv")
    innovation_positioning = read_table(table_dir, "innovation_positioning_audit.csv")
    gate = read_table(table_dir, "real_data_cg_benchmark.csv")
    wedge_operating_points = read_table(table_dir, "arcticdata_wedge_operating_points.csv")
    rare_cryostructure = read_table(table_dir, "synthetic_rare_cryostructure_audit.csv")
    rare_hybrid_metrics = read_table(table_dir, "diffusion_rare_facies_hybrid_metrics.csv")
    rare_hybrid_curve = read_table(table_dir, "rare_facies_hybrid_operating_curve.csv")
    uncertainty_alignment = read_table(table_dir, "posterior_uncertainty_alignment.csv")
    external_generalization = read_table(table_dir, "external_generalization_audit.csv")
    transfer_failure_diagnostics = read_table(table_dir, "transfer_failure_site_diagnostics.csv")
    transfer_failure_summary = read_table(table_dir, "transfer_failure_attribution_summary.csv")
    domain_support = read_table(table_dir, "domain_support_site_audit.csv")
    journal_readiness = read_table(table_dir, "journal_readiness_audit.csv")
    voi_backtest = read_table(table_dir, "voi_backtest_audit.csv")
    gate_summary = _read_gate(table_dir)
    coordinate_summary = _read_json(table_dir, "coordinate_label_coverage_summary.json")
    voi_summary = _read_json(table_dir, "voi_backtest_summary.json")
    tokens = read_table(table_dir, "public_data_token_inventory.csv")
    jago_summary = read_table(table_dir, "arcticdata_jago_ground_ice_observation_summary.csv")
    jago_comparison = read_table(table_dir, "arcticdata_jago_ground_ice_conditioned_diffusion_comparison.csv")
    usgs_eic_diff = read_table(table_dir, "usgs_eic_conditioned_diffusion_metrics.csv")
    usgs_real = read_table(table_dir, "usgs_real_conditioned_diffusion_metrics.csv")
    multisample = read_table(table_dir, "multisample_diffusion_cv_summary.csv")
    site_boreholes = read_table(table_dir, "site_investigation_boreholes.csv")
    site_lines = read_table(table_dir, "site_investigation_ert_lines.csv")

    diffusion = pick_row(model, "model", "COLDReconLatentDiffusion")
    fno = pick_row(model, "model", "COLDReconFNOOperatorDiffusion")
    flow = pick_row(model, "model", "COLDReconRectifiedFlow")
    physics_trained = pick_row(model, "model", "COLDReconLatentDiffusionPhysicsTrained")
    physics_refined = pick_row(model, "model", "COLDReconLatentDiffusionPhysicsRefined")
    footprint_latent = pick_row(computational_footprint, "model", "COLDReconLatentDiffusion")
    footprint_fno = pick_row(computational_footprint, "model", "COLDReconFNOOperatorDiffusion")
    footprint_trained = pick_row(computational_footprint, "model", "COLDReconLatentDiffusionPhysicsTrained")
    gb = pick_row(model, "model", "GradientBoosting")
    kriging = pick_row(model, "model", "KrigingGPR")
    unet = pick_row(model, "model", "SparseUNet3D")
    eic_cal = pick_row(calib, "target", "eic")
    temp_cal = pick_row(calib, "target", "temperature")
    water_cal = pick_row(calib, "target", "unfrozen_water")
    rho_cal = pick_row(calib, "target", "log_resistivity")
    refined_phys = pick_row(physics, "model", "COLDReconLatentDiffusionPhysicsRefined")
    diffusion_phys = pick_row(physics, "model", "COLDReconLatentDiffusion")
    jago_model = pick_row(jago_comparison, "model", "COLDReconJagoGroundIceConditionedDiffusion")
    jago_best = pick_row(jago_comparison, "model", "GlobalMean")
    jago_spatial = pick_row(jago_comparison, "model", "SpatialDepthIDW")
    wedge_recall_point = pick_row(wedge_operating_points, "operating_point", "current site-calibrated recall-first head")
    wedge_max_f1_point = pick_row(wedge_operating_points, "operating_point", "pooled max-F1 probability threshold")
    wedge_knn_point = pick_row(wedge_operating_points, "operating_point", "SpatialDepthKNN baseline")
    rare_trained = pick_row(rare_cryostructure, "model", "COLDReconLatentDiffusionPhysicsTrained")
    rare_implicit = pick_row(rare_cryostructure, "model", "COLDReconImplicit")
    rare_hybrid = pick_row(rare_hybrid_metrics, "model", "COLDReconLatentDiffusionRareFaciesHybrid")
    rare_hybrid_point = pick_numeric_row(rare_hybrid_curve, "eic_floor", 0.10)
    align_trained_eic = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsTrained", "eic")
    align_refined_water = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsRefined", "unfrozen_water")
    align_trained_facies = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsTrained", "facies")
    voi_composite = pick_row(voi_backtest, "target", "composite_error")
    voi_high_eic = pick_row(voi_backtest, "target", "high_eic_mismatch")
    voi_eic = pick_row(voi_backtest, "target", "eic_abs_error")
    external_facies = pick_row(external_generalization, "task", "cryofacies")
    external_eic = pick_row(external_generalization, "task", "EIC regression")
    external_wedge = pick_row(external_generalization, "task", "wedge-ice recall")
    external_high_eic = pick_row(external_generalization, "task", "high-EIC event")
    transfer_outcome_counts = pick_row(transfer_failure_summary, "signal", "EIC outcome counts")
    transfer_failure_counts = pick_row(transfer_failure_summary, "signal", "failure attribution counts")
    usgs_eic = usgs_eic_diff.iloc[0] if not usgs_eic_diff.empty else None
    usgs_real_row = usgs_real.iloc[0] if not usgs_real.empty else None
    jago_row = jago_summary.iloc[0] if not jago_summary.empty else None
    top_borehole = site_boreholes.sort_values("rank").iloc[0] if not site_boreholes.empty else None
    top_line = site_lines.sort_values("rank").iloc[0] if not site_lines.empty else None
    domain_counts = _domain_support_counts(domain_support)
    n_innovation_dimensions = int(len(innovation_positioning)) if not innovation_positioning.empty else 0
    innovation_coverage = (
        float(innovation_positioning["evidence_coverage_score"].mean())
        if not innovation_positioning.empty and "evidence_coverage_score" in innovation_positioning.columns
        else float("nan")
    )
    innovation_boundary_count = (
        int((pd.to_numeric(innovation_positioning["failure_boundary"], errors="coerce") >= 1.0).sum())
        if not innovation_positioning.empty and "failure_boundary" in innovation_positioning.columns
        else 0
    )
    domain_outcome_sentence = (
        "All evaluated site-task outcomes are non-inferior or better under this audit."
        if domain_counts["all_noninferior"]
        else "The domain-support audit is available as a boundary diagnostic, but not all evaluated site-tasks are non-inferior."
    )

    passed_sources = gate_summary.get("independent_public_sources_passed", "not available")
    passed_tasks = gate_summary.get("passed_tasks", "not available")
    total_tasks = gate_summary.get("total_tasks", "not available")
    arctic_eic = gate[
        gate["source"].astype(str).eq("ArcticData cryostratigraphy")
        & gate["task"].astype(str).eq("EIC regression")
    ].iloc[0] if not gate.empty and not gate[
        gate["source"].astype(str).eq("ArcticData cryostratigraphy")
        & gate["task"].astype(str).eq("EIC regression")
    ].empty else None
    arctic_wedge = gate[
        gate["source"].astype(str).eq("ArcticData cryostratigraphy")
        & gate["task"].astype(str).eq("wedge-ice recall")
    ].iloc[0] if not gate.empty and not gate[
        gate["source"].astype(str).eq("ArcticData cryostratigraphy")
        & gate["task"].astype(str).eq("wedge-ice recall")
    ].empty else None

    argument = (
        "In sparse permafrost site characterization, COLD-Recon treats three-dimensional cryostratigraphy "
        "as a multi-source conditional posterior-generation problem and shows bounded gains under synthetic "
        "full-field truth and three independent public validation sources."
    )

    lines = [
        "# Multi-source sparse-observation constrained probabilistic 3D permafrost reconstruction with a physics-guided conditional diffusion neural operator",
        "",
        "## Complete Submission Draft",
        "",
        f"**One-sentence argument.** {argument}",
        "",
        "### Summary",
        "",
        (
            "Subsurface ice controls the thermal, hydrological and geomorphic behaviour of permafrost landscapes, yet it is usually "
            "sampled through sparse boreholes and partial geophysical surveys. This makes site-scale cryostratigraphic reconstruction a "
            "posterior inference problem rather than a deterministic interpolation problem. We introduce COLD-Recon, a "
            "physics-guided conditional diffusion neural-operator workflow that converts multi-source borehole, electrical-resistivity, "
            "nuclear-magnetic-resonance and active-layer observations into probabilistic three-dimensional fields of "
            "cryofacies, excess-ice content, thermal state, unfrozen water and resistivity. On synthetic volumes with "
            f"full-field truth, the latent diffusion posterior reached a mean facies IoU of {metric(diffusion, 'mean_iou')}, "
            f"the FNO-Transformer denoiser reached {metric(fno, 'mean_iou')}, and physics-guided training reached "
            f"{metric(physics_trained, 'mean_iou')}. A post-hoc physics projection reduced unfrozen-water RMSE to "
            f"{metric(physics_refined, 'unfrozen_water_rmse')}. Public validation used USGS field and core data and "
            "two Arctic Data Center ground-ice releases. The real-data evidence gate passed "
            f"{passed_sources} sources and {passed_tasks}/{total_tasks} tasks, including independent Jago River EIC validation "
            "and a recall-oriented wedge-ice constraint head for rare cryostructure preservation. A posterior design diagnostic then "
            "ranks additional borehole and ERT-line candidates from uncertainty and ice-rich ambiguity; under synthetic full-field truth, "
            f"the top VOI decile enriched composite reconstruction error by {metric(voi_composite, 'top_voi_error_enrichment')} times. The result is a reproducible "
            "algorithmic framework for auditable sparse-data permafrost posterior reconstruction, not a regional ground-ice map or a "
            "stand-alone application workflow."
        ),
        "",
        "### Main Text",
        "",
        "#### Sparse ground-ice reconstruction as posterior generation",
        "",
        (
            "Site-scale permafrost characterization requires information about ground ice, cryofacies, temperature "
            "and unfrozen water in three dimensions. These variables determine thermal, hydrological and "
            "geophysical response, but they are rarely measured as dense volumetric fields. Boreholes provide high-confidence "
            "vertical evidence at isolated points, whereas electrical resistivity, nuclear magnetic resonance and active-layer "
            "measurements provide indirect or partial constraints. COLD-Recon therefore formulates site characterization as "
            "sampling p(M|O), where M is a gridded permafrost state and O is a heterogeneous set of sparse observations."
        ),
        "",
        (
            "The unresolved bottleneck is not simply missing interpolation machinery. A useful reconstruction must honour "
            "hard observations, represent uncertainty away from those observations, preserve rare but consequential facies "
            "such as wedge ice, and remain consistent with frozen-ground physical relations. A purely deterministic baseline "
            "can be accurate near measurements but gives little posterior structure; a purely generative model can produce "
            "plausible fields but may ignore site evidence. COLD-Recon addresses this gap through three linked design choices: "
            "a type-aware observation-token interface for sparse heterogeneous measurements, a conditional latent diffusion "
            "neural operator for posterior field generation, and auditable physics plus public-data evidence gates for checking "
            "where the reconstruction should and should not be trusted (Fig. 1)."
        ),
        "",
        (
            "To keep the novelty claim reviewable, the implementation also includes an innovation-positioning audit rather than "
            "relying on rhetorical priority claims. The audit maps "
            f"{fmt(n_innovation_dimensions, digits=0)} innovation dimensions to method definition, controlled validation, baseline comparison, "
            "public-data evidence, failure-boundary reporting and reproducibility traceability. Its mean evidence-coverage score is "
            f"{fmt(innovation_coverage)} and {fmt(innovation_boundary_count, digits=0)} dimensions have full boundary audits. "
            "This positioning supports a bounded algorithmic contribution: the new element is the auditable integration of sparse "
            "multi-source conditioning, posterior neural operators, physics and calibration gates, rare-structure operating points, "
            "public transfer applicability and posterior observation design (Fig. 5)."
        ),
        "",
        "#### A conditional diffusion workflow for sparse permafrost observations",
        "",
        (
            "Each observation is encoded as a token containing normalized spatial coordinates, observation type, value, "
            "uncertainty and mask information. Borehole facies and EIC intervals, ERT log-resistivity profiles, NMR water "
            "content and active-layer observations therefore enter the same conditioning interface. The synthetic training "
            "volumes include active layer, peat, mineral silt, ice-rich silt, sand and gravel, talik and wedge-ice facies, "
            "together with continuous EIC, temperature, unfrozen-water and resistivity fields. A 3D autoencoder maps these "
            "fields into a latent space, and the conditional denoiser samples posterior realizations from the tokenized "
            f"evidence {cite('ho2020ddpm', 'rombach2022latent')}."
        ),
        "",
        (
            "The current implementation evaluates several model families under the same synthetic target: IDW, Random Forest, "
            "Histogram Gradient Boosting, fixed-kernel Kriging/GPR, sparse-observation 3D U-Net, an implicit coordinate field, "
            "latent diffusion, an FNO-Transformer diffusion denoiser and a rectified-flow posterior. This comparison separates "
            "interpolation, tree ensemble, geostatistical, deterministic deep and posterior-generative behaviour. The strongest "
            f"classical baseline in mean facies IoU was Gradient Boosting ({metric(gb, 'mean_iou')}), whereas Kriging/GPR reached "
            f"{metric(kriging, 'mean_iou')} and the sparse 3D U-Net reached {metric(unet, 'mean_iou')}. Latent diffusion and "
            f"its operator/flow variants reached approximately {metric(diffusion, 'mean_iou')}-{metric(fno, 'mean_iou')} mean IoU "
            "while providing posterior ensembles rather than single deterministic fields."
        ),
        "",
        (
            "The comparison is also reported with computational footprint. The compact latent diffusion denoiser has "
            f"{metric(footprint_latent, 'total_params_m')} million trainable parameters, whereas the FNO-Transformer denoiser has "
            f"{metric(footprint_fno, 'total_params_m')} million. The physics-trained posterior keeps the compact architecture and "
            f"stores a prediction artifact of {metric(footprint_trained, 'prediction_mb')} MB. Reporting parameter count, checkpoint size, "
            "prediction footprint and posterior sample count prevents the high-parameter operator variant and the compact physics-trained "
            "variant from being treated as cost-equivalent."
        ),
        "",
        (
            "A separate synthetic rare-cryostructure audit prevents mean-IoU overstatement. For the physics-trained diffusion "
            f"posterior, a fixed high-EIC threshold gave recall {metric(rare_trained, 'raw_eic_recall')} and F1 "
            f"{metric(rare_trained, 'raw_eic_f1')}; constraining the high-EIC event rate to twice the observed borehole "
            f"high-EIC rate increased recall to {metric(rare_trained, 'rate_constrained_eic_recall')} and F1 to "
            f"{metric(rare_trained, 'rate_constrained_eic_f1')}. The same audit reports rare-facies recall "
            f"{metric(rare_trained, 'rare_facies_recall')} for the physics-trained posterior, while synthetic wedge-ice "
            f"facies recall remains {metric(rare_trained, 'facies_6_wedge_ice_recall')} for this diffusion variant and "
            f"{metric(rare_implicit, 'facies_6_wedge_ice_recall')} for the implicit field. Thus high-EIC screening is "
            "treated as an operating-point diagnostic, not as a solved wedge-facies reconstruction problem. A rare-facies "
            f"hybrid operating point that accepts implicit wedge proposals only where diffusion EIC exceeds 0.10 reports "
            f"wedge recall {metric(rare_hybrid, 'wedge_ice_recall')}, precision {metric(rare_hybrid, 'wedge_ice_precision')} "
            f"and mean IoU {metric(rare_hybrid, 'mean_iou')}, with {metric(rare_hybrid_point, 'gate_fraction')} of voxels "
            f"gated and mean-IoU change {metric(rare_hybrid_point, 'mean_iou_delta_vs_base')}. This converts the wedge miss "
            "into a measurable constraint trade-off rather than hiding it inside the main posterior score."
        ),
        "",
        "#### Physics and calibration diagnostics expose where the posterior is reliable",
        "",
        (
            "COLD-Recon uses physical consistency as a diagnostic and a correction target. The pipeline includes empirical "
            "unfrozen-water consistency, resistivity coupling, simplified heat residuals, physics-guided denoiser fine-tuning, "
            "latent-space guidance and post-hoc posterior projection. In the synthetic validation, physics-guided training "
            f"increased mean facies IoU to {metric(physics_trained, 'mean_iou')}. Post-hoc projection preserved the latent diffusion "
            f"facies IoU ({metric(physics_refined, 'mean_iou')}) but reduced unfrozen-water RMSE from "
            f"{metric(diffusion, 'unfrozen_water_rmse')} to {metric(physics_refined, 'unfrozen_water_rmse')} and reduced the "
            f"unfrozen-water empirical MAE from {metric(diffusion_phys, 'unfrozen_water_empirical_mae')} to "
            f"{metric(refined_phys, 'unfrozen_water_empirical_mae')} (Fig. 1)."
        ),
        "",
        (
            "Posterior calibration is deliberately reported rather than hidden. After post-hoc interval calibration, EIC, "
            f"temperature, unfrozen water and log-resistivity reached approximately 90% interval coverage ({metric(eic_cal, 'coverage_90')}, "
            f"{metric(temp_cal, 'coverage_90')}, {metric(water_cal, 'coverage_90')} and {metric(rho_cal, 'coverage_90')}, respectively). "
            "The unfrozen-water field required a bias-quantile fallback rather than simple spread scaling, indicating that interval "
            "coverage can be repaired diagnostically while mean-field physical interpretation still depends on stronger training-time physics."
        ),
        "",
        (
            "A separate uncertainty-error alignment audit tested whether posterior spread identifies where the reconstruction is wrong. "
            f"For the physics-trained posterior, EIC uncertainty had Spearman rank correlation {metric(align_trained_eic, 'spearman_uncertainty_error')} "
            f"with absolute EIC error, and the top uncertainty decile had {metric(align_trained_eic, 'top_uncertainty_error_enrichment')} times "
            "the global EIC error. After physics refinement, unfrozen-water uncertainty had rank correlation "
            f"{metric(align_refined_water, 'spearman_uncertainty_error')} and top-uncertainty error enrichment "
            f"{metric(align_refined_water, 'top_uncertainty_error_enrichment')}. Facies entropy was weaker, with top-entropy error enrichment "
            f"{metric(align_trained_facies, 'top_uncertainty_error_enrichment')}. The probabilistic claim is therefore target-specific: "
            "posterior uncertainty is informative for EIC error localization and some physics-refined continuous fields, but it is not treated as a universal reliability certificate."
        ),
        "",
        "#### Public data provide a three-source evidence gate",
        "",
        (
            f"The public validation package uses USGS Alaska permafrost geophysical data {cite('james2020usgs_geophysics')}, "
            f"USGS Arctic Coastal Plain core EIC data {cite('stephani2025usgs_eic')}, Arctic Data Center upper-permafrost "
            f"cryostratigraphy and ground-ice records {cite('kanevskiy2024upper_permafrost')}, and an independent Jago River "
            f"2018 ground-ice release {cite('kanevskiy2020jago_ground_ice')}. The processed token inventory contains "
            f"{_source_token_count(tokens, 'usgs_ert_nmr', 'ert_log_resistivity')} ERT log-resistivity tokens, "
            f"{_source_token_count(tokens, 'arcticdata_upper_permafrost_cryostratigraphy', 'borehole_facies')} ArcticData "
            f"cryofacies tokens, {_source_token_count(tokens, 'arcticdata_upper_permafrost_cryostratigraphy', 'borehole_eic')} "
            f"ArcticData EIC tokens, {_source_token_count(tokens, 'usgs_eic_cores', 'borehole_eic')} USGS core EIC tokens and "
            f"{_source_token_count(tokens, 'arcticdata_jago_ground_ice_2018', 'borehole_eic')} Jago EIC tokens (Fig. 2)."
        ),
        "",
        (
            "The evidence gate treats public validation as a set of task-specific checks instead of a single aggregate score. "
            f"ArcticData EIC regression improved RMSE from {metric(arctic_eic, 'baseline_value')} to "
            f"{metric(arctic_eic, 'model_value')}. The recall-first wedge-ice constraint head improved wedge-ice recall from "
            f"{metric(arctic_wedge, 'baseline_value')} to {metric(arctic_wedge, 'model_value')}, while the precision and "
            "false-positive trade-off remains an explicit operating-point choice. A separate operating-curve audit made that trade-off explicit: "
            f"the site-calibrated recall head reached pooled recall {metric(wedge_recall_point, 'recall')} and false-positive rate "
            f"{metric(wedge_recall_point, 'false_positive_rate')}, whereas a pooled max-F1 probability threshold reduced the false-positive "
            f"rate to {metric(wedge_max_f1_point, 'false_positive_rate')} at recall {metric(wedge_max_f1_point, 'recall')}. "
            "USGS EIC-conditioned diffusion evaluated "
            f"{metric(usgs_eic, 'holdout_eic_n', digits=0)} held-out intervals and reached hold-out EIC RMSE "
            f"{metric(usgs_eic, 'holdout_eic_rmse')} with high-EIC recall {metric(usgs_eic, 'holdout_eic_high_eic_recall')}. "
            f"The Jago high-EIC screen reached F1 {metric(jago_model, 'high_eic_f1')} and recall "
            f"{metric(jago_model, 'high_eic_recall')} using a calibrated prediction threshold of "
            f"{metric(jago_model, 'high_eic_prediction_threshold')}. "
            f"Together, the gate passed {passed_tasks}/{total_tasks} tasks across {passed_sources} independent public sources."
        ),
        "",
        (
            "A separate external-generalization audit exposes site-level heterogeneity within the ArcticData branch. "
            f"Across five compact public sites, the adaptive hybrid increased mean cryofacies accuracy from "
            f"{metric(external_facies, 'baseline_value')} to {metric(external_facies, 'model_value')}, with site win rate "
            f"{metric(external_facies, 'site_win_rate')} and non-inferiority rate {metric(external_facies, 'site_noninferior_rate')}. "
            f"For EIC, the evidence-gate comparison improved against the aggregate SpatialDepthIDW mean by "
            f"{metric(external_eic, 'evidence_gate_relative_improvement')}, but a stricter per-site best-simple audit gave site win rate "
            f"{metric(external_eic, 'site_win_rate')}, non-inferiority rate {metric(external_eic, 'site_noninferior_rate')}, and failure sites "
            f"{text_or_none(external_eic, 'failure_sites')}. "
            f"High-EIC event screening increased mean F1 from {metric(external_high_eic, 'baseline_value')} to "
            f"{metric(external_high_eic, 'model_value')}, with site win rate {metric(external_high_eic, 'site_win_rate')} "
            f"and non-inferiority rate {metric(external_high_eic, 'site_noninferior_rate')}. "
            f"The wedge recall head raised pooled recall from {metric(external_wedge, 'baseline_value')} to "
            f"{metric(external_wedge, 'model_value')}, while mean precision decreased from "
            f"{metric(external_wedge, 'secondary_baseline_value')} to {metric(external_wedge, 'secondary_model_value')}. "
            "Thus the public multi-site evidence is presented as auditable transfer with explicit guardrails and boundary conditions, not as a universal field-generalization claim (Fig. 6)."
        ),
        "",
        (
            "A transfer-failure attribution audit then asks whether the stricter EIC failures exposed by the earlier audit can be controlled without using hold-out truth. "
            f"The site outcomes are {metric(transfer_outcome_counts, 'interpretation')}; the failure labels are "
            f"{metric(transfer_failure_counts, 'interpretation')}. The compact-site spatial guard selects transfer_idw_adapter at compact, sparsely supported sites "
            "only when train-split cross-validation admits the local IDW prior. In the current five-site audit, this converts the exposed Itkillik and Tuktoyaktuk "
            "EIC failures into non-inferior ties, so the remaining boundary is heterogeneous site wins rather than hidden transfer failure (Fig. 7)."
        ),
        "",
        (
            "A domain-support audit separates train-side applicability signals from hold-out outcomes. Across the five public sites, "
            f"{fmt(domain_counts['model_supported'], digits=0)} sites are model-supported transfers, "
            f"{fmt(domain_counts['guarded'], digits=0)} compact sites use the guarded local-prior adapter, and "
            f"{fmt(domain_counts['low'], digits=0)} sites are low support. {domain_outcome_sentence} "
            "This converts external transfer from a pooled score into an applicability rule: COLD-Recon can report model-transfer wins where "
            "support is sufficient, and guarded non-inferiority where the site is compact and local support is sparse (Fig. 8)."
        ),
        "",
        (
            "The public coordinate-label coverage is stronger than the earlier boundary statement implied, but it is still not a completed EG field-generalization proof. "
            f"The ArcticData cryostratigraphy inventory contains {fmt(coordinate_summary.get('n_units'), digits=0)} vertical units, "
            f"{fmt(coordinate_summary.get('n_georeferenced_units'), digits=0)} georeferenced units across "
            f"{fmt(coordinate_summary.get('n_sites_with_georeferenced_units'), digits=0)} sites, "
            f"{fmt(coordinate_summary.get('n_eic_measurements'), digits=0)} EIC measurements and "
            f"{fmt(coordinate_summary.get('n_wedge_ice_units'), digits=0)} wedge-ice units. This moves the coordinate and dense-label item from missing evidence "
            "to conditional EG-readiness evidence, while retaining the core boundary: these are georeferenced vertical core intervals, not dense public 3D ground truth or a prospective field-validation campaign (Fig. 10)."
        ),
        "",
        "#### Independent Jago River data sharpen the boundary of the claim",
        "",
        (
            "The Jago River 2018 release is used as a third independent ground-ice/EIC source rather than as a regional "
            f"benchmark. It contributes {metric(jago_row, 'n_eic_tokens', digits=0)} EIC tokens from "
            f"{metric(jago_row, 'n_boreholes', digits=0)} ordered boreholes, with a mean EIC fraction of "
            f"{metric(jago_row, 'mean_eic_fraction')} and maximum EIC fraction {metric(jago_row, 'max_eic_fraction')}. "
            f"On the same hold-out split, the Jago-conditioned posterior reached EIC RMSE {metric(jago_model, 'eic_rmse')} "
            f"against {metric(jago_best, 'eic_rmse')} for the global-mean baseline, a small relative reduction of "
            f"{metric(jago_model, 'eic_rmse_reduction_vs_best_simple')} (Fig. 3)."
        ),
        "",
        (
            "This result is useful because it prevents overstatement. The Jago branch supports targeted EIC regression, but it "
            f"supports high-EIC screening only after train-split F2 calibration: hold-out F1 was {metric(jago_model, 'high_eic_f1')} "
            f"against {metric(jago_spatial, 'high_eic_f1')} for SpatialDepthIDW, with fixed-0.30 recall "
            f"{metric(jago_model, 'high_eic_recall_fixed_0p30')}. The public table lacks surveyed borehole coordinates in the "
            "processed CSV, so the workflow uses an ordered borehole convention rather than claiming surveyed spatial prediction. "
            "The paper therefore treats Jago as an independent small-sample constraint on the model, not as proof of regional transfer."
        ),
        "",
        "#### Posterior uncertainty is converted into an observation-design diagnostic",
        "",
        (
            "A probabilistic reconstruction should expose where the posterior remains weakly constrained. COLD-Recon therefore includes a "
            "posterior value-of-information diagnostic that scores candidate surface locations by posterior uncertainty, ice-rich ambiguity, "
            "thaw-sensitive EIC structure and novelty relative to existing observations. The resulting map ranks additional boreholes and "
            "ERT survey lines directly from the reconstruction posterior rather than from a separate heuristic map (Fig. 4)."
        ),
        "",
        (
            f"In the current USGS-conditioned posterior, the highest-ranked borehole target was at x={metric(top_borehole, 'x')} m, "
            f"y={metric(top_borehole, 'y')} m, with VOI score {metric(top_borehole, 'voi_score')}. The highest-ranked ERT line was "
            f"an {metric(top_line, 'orientation')} oriented line with mean line score {metric(top_line, 'line_score')}. These values should "
            "be read as audit-ready posterior-diagnostic hypotheses, because the weights are fixed and transparent rather than learned from a "
            "prospective field campaign. This distinction keeps the claim at the level of reproducible observation design for the reconstruction "
            "algorithm, not proven engineering optimality."
        ),
        "",
        (
            "A retrospective full-field VOI backtest adds a bounded validation layer for this design diagnostic. Using the synthetic "
            "physics-trained posterior and its known full-field truth, the top VOI decile enriched composite reconstruction error by "
            f"{metric(voi_composite, 'top_voi_error_enrichment')} times, high-EIC mismatch by {metric(voi_high_eic, 'top_voi_error_enrichment')} times, "
            f"and EIC absolute error by {metric(voi_eic, 'top_voi_error_enrichment')} times. The VOI score had Spearman correlation "
            f"{metric(voi_composite, 'spearman_voi_error')} with composite error. This converts the VOI layer from an untested recommendation map "
            "into a retrospective error-targeting audit, while still stopping short of a prospective field acquisition trial (Fig. 11)."
        ),
        "",
        "#### Discussion",
        "",
        (
            "COLD-Recon advances sparse permafrost characterization by joining posterior generation, observation-token "
            "conditioning, physical consistency diagnostics and public-data validation in a reproducible workflow. The strongest "
            "evidence is not any single metric, but the alignment between synthetic full-field reconstruction, physics diagnostics "
            "and a three-source evidence gate. The observation-design diagnostic adds a controlled way to inspect posterior blind spots "
            "without reframing the paper away from the reconstruction algorithm. This makes the framework suitable for auditable site-scale reconstruction "
            "where uncertainty, hard observations and rare cryostructures need to remain visible to the user."
        ),
        "",
        (
            "The present implementation also exposes clear boundaries. Synthetic experiments still dominate the full-field "
            "assessment because public permafrost releases rarely include complete 3D ground truth. The USGS branch validates "
            "EIC and geophysical proxies but not cryofacies labels. The coordinate-label coverage audit now shows substantial "
            "georeferenced ArcticData vertical labels, but these remain borehole intervals rather than gridded 3D validation volumes. The ArcticData wedge-ice head is intentionally recall-first, "
            "and the operating-curve audit shows how false-positive control trades against recall. The external-generalization audit shows that ArcticData EIC transfer remains heterogeneous under a stricter per-site best-simple comparator, and the compact-site spatial guard controls the previously exposed failures as non-inferior ties by using train-only support diagnostics. The domain-support audit makes that boundary explicit by separating model-supported transfers from guarded local-prior sites. The Jago River branch is independent but small, and its high-EIC screen is "
            "threshold-calibrated rather than a robust regional event detector. The synthetic rare-cryostructure and rare-facies hybrid audits further show that high-EIC operating points, wedge-facies recall and precision-cost trade-offs must be reported separately. The uncertainty-error alignment audit shows strong EIC error localization but weaker facies and geophysical alignment. The VOI layer now has a retrospective synthetic full-field backtest, but it has still not been prospectively tested in a field campaign. Finally, the unfrozen-water posterior needs post-hoc bias-quantile interval calibration, so calibrated coverage should not be read as a solved process model. These limitations are substantial, "
            "but they are now encoded as reproducible tests rather than left as qualitative caveats."
        ),
        "",
        (
            "A journal-readiness audit formalizes this boundary (Fig. 9). The CG algorithm criteria pass or remain conditional "
            "only on final reproducibility closure, whereas the EG field-generalization criteria now treat coordinate-label coverage and VOI backtesting as conditional evidence but retain a not-yet item for "
            "full-field public 3D ground truth. The "
            "recommended positioning is therefore a CG-plus algorithm manuscript with explicit EG-readiness evidence, not a completed "
            "regional field-generalization study."
        ),
        "",
        (
            "The next technical step is not simply a larger model. More valuable progress would come from multi-site training "
            "with public coordinate quality controls, explicit domain adaptation from synthetic to field records, richer public "
            "covariate ingestion, and physics-guided training objectives that improve calibration as well as mean error. Within "
            "the evidence available here, COLD-Recon should be read as a verified research prototype for conditional permafrost "
            "posterior reconstruction."
        ),
        "",
        "### Methods Summary",
        "",
        (
            "Synthetic cryostratigraphy volumes were generated on the configured grid with seven facies classes and continuous "
            "EIC, temperature, unfrozen-water and resistivity fields. Sparse borehole, ERT, NMR and active-layer observations "
            "were sampled from these fields and encoded as normalized observation tokens. Baselines and neural models were "
            "trained or evaluated through the command sequence recorded in the reproducibility audit. Public data were processed "
            "from USGS and Arctic Data Center releases into NPZ token files and CSV provenance tables. Real-data tasks used "
            "leave-one-borehole-out or same-split comparisons against simple baselines. Nature-style figures were generated "
            "from audited CSV/NPZ outputs using Python/matplotlib with editable SVG and PDF text and high-resolution TIFF export. "
            "Additional observation-design targets were generated by a fixed-weight VOI score from posterior uncertainty, ice-rich ambiguity, "
            "thaw-sensitive EIC structure and distance from existing observations. VOI was retrospectively backtested under synthetic full-field truth by comparing high-ranked surface targets with realized EIC, facies, high-EIC and wedge-miss error surfaces. Wedge-ice operating curves were generated from ArcticData hold-out "
            "posterior wedge probabilities and stored as audit tables. External multi-site generalization was audited from the ArcticData grouped-borehole holdout table by reporting site-wise facies deltas, EIC RMSE reductions against per-site best simple baselines, wedge recall/precision trade-offs and task-level site win rates. Transfer-failure attribution was computed from those site-wise deltas by labelling EIC outcomes, the best simple baseline, local IDW advantage over the global mean, compact hold-out support and transfer-readiness components. Domain-support applicability was audited from train-side support counts, adaptive EIC methods and the resulting holdout outcomes, keeping support scores separate from post-hoc performance. Coordinate-label coverage was audited from the processed ArcticData inventory by counting georeferenced vertical units, georeferenced boreholes, EIC measurements, high-EIC units and wedge-ice units by site. Synthetic rare-cryostructure audits were generated by comparing fixed high-EIC thresholds, observation-rate-constrained high-EIC thresholds and facies-level recall for rare cryostructure classes. A synthetic rare-facies hybrid operating curve was generated by accepting implicit wedge proposals only above swept diffusion-EIC floors and reporting wedge recall, precision, F1, gated voxel fraction and mean-IoU change. Computational footprint was audited from checkpoint, prediction NPZ and training-history artifacts. Innovation positioning was audited by mapping each claimed contribution to method definition, controlled validation, baseline comparison, public-data evidence, failure-boundary reporting and reproducibility traceability. Posterior uncertainty-error alignment was audited by comparing per-voxel posterior standard deviation or facies entropy with synthetic full-field absolute errors and misclassification indicators."
            " Journal-readiness criteria were computed from the public-data gate, site-wise external-generalization tables, domain-support audit, coordinate-label coverage audit, VOI backtest, figure atlas and reproducibility audit to separate algorithm-manuscript evidence from field-generalization gaps."
        ),
        "",
        "### Figure Captions",
        "",
        "![Figure 1. COLD-Recon overview.](../outputs/figures/nature_figure_1_overview.png)",
        "",
        (
            "**Figure 1 | COLD-Recon converts sparse permafrost observations into verified 3D posteriors.** "
            "a, Observation-token, conditional-posterior and physics-projection workflow. b, Synthetic mean facies IoU across "
            "interpolation, tree-ensemble, geostatistical, deterministic deep and posterior-generative models. c, Raw and "
            "post-hoc calibrated 90% posterior interval coverage. d, Unfrozen-water consistency errors for truth, baselines and "
            "COLD-Recon variants. Source data are in `outputs/source_data/nature_figure_1_source_data.csv`."
        ),
        "",
        "![Figure 2. Real-data evidence gate.](../outputs/figures/nature_figure_2_real_data_gate.png)",
        "",
        (
            "**Figure 2 | Real-data evidence gate across three public permafrost data sources.** "
            "a, Processed public observation-token inventory. b, Relative improvement for passed validation tasks. c, EIC RMSE "
            "for ArcticData, USGS core and Jago River holdouts against best simple baselines. d, Pass/fail gate matrix showing "
            "calibrated Jago high-EIC screening and recall-first wedge-ice handling. Source data are in `outputs/source_data/nature_figure_2_source_data.csv`."
        ),
        "",
        "![Figure 3. Cited ground-ice validation data.](../outputs/figures/nature_figure_3_cited_ground_ice.png)",
        "",
        (
            "**Figure 3 | Cited ground-ice records support cross-source EIC validation.** "
            "a, Measured EIC distributions from ArcticData, USGS core and Jago River records. b, Jago measured ground-ice "
            "observations by ordered borehole and depth. c, Jago observed-versus-predicted EIC for simple baselines and "
            "COLD-Recon. d, USGS core observed-versus-predicted EIC. Source data are in "
            "`outputs/source_data/nature_figure_3_source_data.csv`."
        ),
        "",
        "![Figure 4. Posterior value-of-information targets.](../outputs/figures/nature_figure_4_site_investigation.png)",
        "",
        (
            "**Figure 4 | Posterior value-of-information ranks additional observation targets.** "
            "a, VOI surface from the USGS-conditioned posterior, with existing observations, recommended boreholes and recommended "
            "ERT lines. b, Ranked additional borehole targets. c, Weighted component contributions for the highest-ranked "
            "boreholes. d, Ranked ERT lines, showing mean line score and maximum intersected cell score. Source data are in "
            "`outputs/source_data/nature_figure_4_source_data.csv`."
        ),
        "",
        "![Figure 5. Innovation positioning audit.](../outputs/figures/innovation_positioning_audit.png)",
        "",
        (
            "**Figure 5 | COLD-Recon innovation positioning is evidence-mapped rather than rhetorical.** "
            "a, Evidence coverage matrix linking each innovation dimension to method definition, controlled validation, baseline "
            "comparison, public-data evidence, boundary reporting and reproducibility traceability. b, Current evidence maturity "
            "against the prospective EG target. c, Coverage score by innovation dimension, showing which novelty claims are fully "
            "audited and which remain bounded by prospective-validation gaps. Source data are in "
            "`outputs/source_data/innovation_positioning_audit_source_data.csv`."
        ),
        "",
        "![Figure 6. External generalization audit.](../outputs/figures/external_generalization_audit.png)",
        "",
        (
            "**Figure 6 | Public multi-site holdouts expose transfer gains and boundaries.** "
            "a, Site-wise cryofacies accuracy deltas for the adaptive ArcticData hybrid against SpatialDepthKNN. "
            "b, Site-wise EIC RMSE reduction against the per-site best simple baseline. c, Wedge-ice recall and precision "
            "for the spatial KNN baseline and recall-oriented constraint head. d, Site-level win and non-inferiority rates "
            "by task. Source data are in `outputs/source_data/external_generalization_audit_source_data.csv`."
        ),
        "",
        "![Figure 7. Compact-site spatial guard audit.](../outputs/figures/transfer_failure_attribution.png)",
        "",
        (
            "**Figure 7 | Compact-site spatial guard controls EIC transfer failures.** "
            "a, EIC RMSE pairs for COLD-Recon and the per-site best simple baseline. b, SpatialDepthIDW advantage "
            "versus the COLD-Recon RMSE gap to the best simple baseline. c, Small-n Spearman associations with EIC "
            "RMSE reduction. d, Transfer-readiness components by site, with guarded compact sites retained as non-inferior EIC ties. Source data are in "
            "`outputs/source_data/transfer_failure_attribution_source_data.csv`."
        ),
        "",
        "![Figure 8. Domain-support audit.](../outputs/figures/domain_support_audit.png)",
        "",
        (
            "**Figure 8 | Domain-support audit separates train-side support from holdout outcomes.** "
            "a, Support-score components available before holdout scoring, with model-supported and guarded local-prior classes. "
            "b, Support score versus EIC RMSE reduction against the per-site best simple baseline. c, Holdout outcome matrix across "
            "facies, EIC, high-EIC screening and wedge recall, with not-evaluated site-task cells shown separately. Source data are in "
            "`outputs/source_data/domain_support_audit_source_data.csv`."
        ),
        "",
        "![Figure 9. Journal readiness audit.](../outputs/figures/journal_readiness_audit.png)",
        "",
        (
            "**Figure 9 | CG/EG readiness audit separates algorithm evidence from field-claim gaps.** "
            "a, Criterion-level readiness scores for CG algorithm and EG field-generalization claims. b, Mean readiness score "
            "by tier. c, Pass, conditional and not-yet criteria counts, showing that the present evidence supports a CG-plus "
            "algorithm manuscript while retaining explicit EG field-validation gaps. The coordinate-label criterion is conditional, "
            "VOI backtesting is conditional, and public 3D ground truth remains a not-yet item. Source data are in "
            "`outputs/source_data/journal_readiness_audit_source_data.csv`."
        ),
        "",
        "![Figure 10. Coordinate-label coverage audit.](../outputs/figures/coordinate_label_coverage_audit.png)",
        "",
        (
            "**Figure 10 | Public coordinate-label coverage improves EG-readiness but not full field validation.** "
            "a, Georeferenced and non-georeferenced vertical cryostratigraphic units for the highest-coverage public ArcticData sites. "
            "b, Site-wise coordinate coverage versus EIC measurement count, with marker size proportional to georeferenced boreholes. "
            "c, Aggregate facies, EIC, high-EIC, wedge-ice and georeferenced-unit counts. Source data are in "
            "`outputs/source_data/coordinate_label_coverage_audit_source_data.csv`."
        ),
        "",
        "![Figure 11. VOI backtest audit.](../outputs/figures/voi_backtest_audit.png)",
        "",
        (
            "**Figure 11 | Retrospective full-field VOI backtest supports bounded observation-design readiness.** "
            "a, Error enrichment in the top VOI decile relative to global error for composite, high-EIC, facies, wedge-miss and EIC targets. "
            "b, Spearman correlations between fixed VOI components and realized composite error, showing which components carry the targeting signal. "
            "c, Ranked borehole targets with realized composite error and VOI score. Source data are in "
            "`outputs/source_data/voi_backtest_audit_source_data.csv`."
        ),
        "",
        *comprehensive_submission_figure_lines(table_dir, start_number=12),
        "",
        "### Data and Code Availability",
        "",
        (
            "All generated data, processed public-data tokens, predictions, tables, figures and manuscript files are stored under "
            "`data/`, `outputs/` and `paper/`. Public datasets are cited in `paper/references.bib`; large or authentication-gated "
            "inputs are documented in `data/external/DOWNLOAD_INSTRUCTIONS.md`. The audited command sequence is recorded in "
            "`paper/reproducibility_audit.md`, with artifact hashes in `outputs/tables/reproducibility_manifest.csv`. VOI-ranked "
            "boreholes, ERT lines, the gridded score field, wedge-ice operating-curve audit tables and computational-footprint "
            "outputs are stored with the other audited outputs. Innovation-positioning outputs are stored in "
            "`outputs/tables/innovation_positioning_audit.csv`, `outputs/tables/innovation_positioning_summary.json`, "
            "`outputs/source_data/innovation_positioning_audit_source_data.csv` and `outputs/figures/innovation_positioning_audit.*`. "
            "External multi-site generalization outputs are stored in "
            "`outputs/tables/external_generalization_audit.csv`, `outputs/tables/external_generalization_site_deltas.csv` and "
            "`outputs/figures/external_generalization_audit.*`. Transfer-failure attribution outputs are stored in "
            "`outputs/tables/transfer_failure_site_diagnostics.csv`, `outputs/tables/transfer_failure_attribution_summary.csv`, "
            "`outputs/source_data/transfer_failure_attribution_source_data.csv` and `outputs/figures/transfer_failure_attribution.*`. "
            "Coordinate-label coverage outputs are stored in `outputs/tables/coordinate_label_coverage_audit.csv`, "
            "`outputs/tables/coordinate_label_coverage_summary.json`, `outputs/source_data/coordinate_label_coverage_audit_source_data.csv` "
            "and `outputs/figures/coordinate_label_coverage_audit.*`. "
            "VOI backtest outputs are stored in `outputs/tables/voi_backtest_audit.csv`, "
            "`outputs/tables/voi_backtest_summary.json`, `outputs/source_data/voi_backtest_audit_source_data.csv` and "
            "`outputs/figures/voi_backtest_audit.*`. "
            "Domain-support applicability outputs are stored in `outputs/tables/domain_support_site_audit.csv`, "
            "`outputs/tables/domain_support_summary.json`, `outputs/source_data/domain_support_audit_source_data.csv` and "
            "`outputs/figures/domain_support_audit.*`. "
            "Journal-readiness outputs are stored in `outputs/tables/journal_readiness_audit.csv`, "
            "`outputs/tables/journal_readiness_summary.json`, `outputs/source_data/journal_readiness_audit_source_data.csv` and "
            "`outputs/figures/journal_readiness_audit.*`."
        ),
        "",
        "### References",
        "",
        references_markdown(),
        "",
        "### Scope Notes",
        "",
        (
            "This complete submission draft keeps the complete atlas-selected evidence figure set in the main manuscript file. The longer generated "
            "technical draft `paper/cold_recon_manuscript_draft.md` remains a provenance-oriented technical document with expanded tables "
            "and command-level detail."
        ),
    ]
    return _write(output_path, lines)


def build_claim_evidence_audit(table_dir: Path, output_path: Path) -> Path:
    gate_summary = _read_gate(table_dir)
    gate = read_table(table_dir, "real_data_cg_benchmark.csv")
    model = read_table(table_dir, "model_comparison.csv")
    computational_footprint = read_table(table_dir, "computational_footprint.csv")
    innovation_positioning = read_table(table_dir, "innovation_positioning_audit.csv")
    wedge_operating_points = read_table(table_dir, "arcticdata_wedge_operating_points.csv")
    rare_cryostructure = read_table(table_dir, "synthetic_rare_cryostructure_audit.csv")
    rare_hybrid_metrics = read_table(table_dir, "diffusion_rare_facies_hybrid_metrics.csv")
    rare_hybrid_curve = read_table(table_dir, "rare_facies_hybrid_operating_curve.csv")
    uncertainty_alignment = read_table(table_dir, "posterior_uncertainty_alignment.csv")
    external_generalization = read_table(table_dir, "external_generalization_audit.csv")
    transfer_failure_summary = read_table(table_dir, "transfer_failure_attribution_summary.csv")
    domain_support = read_table(table_dir, "domain_support_site_audit.csv")
    journal_readiness = read_table(table_dir, "journal_readiness_audit.csv")
    voi_backtest = read_table(table_dir, "voi_backtest_audit.csv")
    coordinate_summary = _read_json(table_dir, "coordinate_label_coverage_summary.json")
    voi_summary = _read_json(table_dir, "voi_backtest_summary.json")
    site_boreholes = read_table(table_dir, "site_investigation_boreholes.csv")
    site_lines = read_table(table_dir, "site_investigation_ert_lines.csv")
    passed_sources = gate_summary.get("independent_public_sources_passed", "not available")
    passed_tasks = gate_summary.get("passed_tasks", "not available")
    total_tasks = gate_summary.get("total_tasks", "not available")
    diffusion = pick_row(model, "model", "COLDReconLatentDiffusion")
    fno = pick_row(model, "model", "COLDReconFNOOperatorDiffusion")
    physics_trained = pick_row(model, "model", "COLDReconLatentDiffusionPhysicsTrained")
    footprint_latent = pick_row(computational_footprint, "model", "COLDReconLatentDiffusion")
    footprint_fno = pick_row(computational_footprint, "model", "COLDReconFNOOperatorDiffusion")
    footprint_trained = pick_row(computational_footprint, "model", "COLDReconLatentDiffusionPhysicsTrained")
    innovation_dimensions = int(len(innovation_positioning)) if not innovation_positioning.empty else 0
    innovation_coverage = (
        float(innovation_positioning["evidence_coverage_score"].mean())
        if not innovation_positioning.empty and "evidence_coverage_score" in innovation_positioning.columns
        else float("nan")
    )
    innovation_boundaries = (
        int((pd.to_numeric(innovation_positioning["failure_boundary"], errors="coerce") >= 1.0).sum())
        if not innovation_positioning.empty and "failure_boundary" in innovation_positioning.columns
        else 0
    )
    rare_trained = pick_row(rare_cryostructure, "model", "COLDReconLatentDiffusionPhysicsTrained")
    rare_hybrid = pick_row(rare_hybrid_metrics, "model", "COLDReconLatentDiffusionRareFaciesHybrid")
    rare_hybrid_point = pick_numeric_row(rare_hybrid_curve, "eic_floor", 0.10)
    align_trained_eic = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsTrained", "eic")
    align_refined_water = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsRefined", "unfrozen_water")
    voi_composite = pick_row(voi_backtest, "target", "composite_error")
    voi_high_eic = pick_row(voi_backtest, "target", "high_eic_mismatch")
    external_facies = pick_row(external_generalization, "task", "cryofacies")
    external_eic = pick_row(external_generalization, "task", "EIC regression")
    external_wedge = pick_row(external_generalization, "task", "wedge-ice recall")
    external_high_eic = pick_row(external_generalization, "task", "high-EIC event")
    transfer_outcome_counts = pick_row(transfer_failure_summary, "signal", "EIC outcome counts")
    transfer_failure_counts = pick_row(transfer_failure_summary, "signal", "failure attribution counts")
    domain_counts = _domain_support_counts(domain_support)
    cg_readiness = journal_readiness[journal_readiness["readiness_tier"].astype(str).eq("CG algorithm article")] if not journal_readiness.empty else pd.DataFrame()
    eg_readiness = journal_readiness[journal_readiness["readiness_tier"].astype(str).eq("EG field-generalization claim")] if not journal_readiness.empty else pd.DataFrame()
    cg_not_yet = int(cg_readiness["status"].astype(str).eq("not_yet").sum()) if not cg_readiness.empty else 0
    eg_not_yet = int(eg_readiness["status"].astype(str).eq("not_yet").sum()) if not eg_readiness.empty else 0
    wedge_recall_point = pick_row(wedge_operating_points, "operating_point", "current site-calibrated recall-first head")
    wedge_max_f1_point = pick_row(wedge_operating_points, "operating_point", "pooled max-F1 probability threshold")
    top_borehole = site_boreholes.sort_values("rank").iloc[0] if not site_boreholes.empty else None
    top_line = site_lines.sort_values("rank").iloc[0] if not site_lines.empty else None
    jago_event = gate[
        gate["source"].astype(str).str.contains("Jago", na=False)
        & gate["task"].astype(str).eq("high-EIC event")
    ].iloc[0] if not gate.empty and not gate[
        gate["source"].astype(str).str.contains("Jago", na=False)
        & gate["task"].astype(str).eq("high-EIC event")
    ].empty else None
    wedge_event = gate[
        gate["source"].astype(str).eq("ArcticData cryostratigraphy")
        & gate["task"].astype(str).eq("wedge-ice recall")
    ].iloc[0] if not gate.empty and not gate[
        gate["source"].astype(str).eq("ArcticData cryostratigraphy")
        & gate["task"].astype(str).eq("wedge-ice recall")
    ].empty else None
    lines = [
        "# COLD-Recon Claim-Evidence Audit",
        "",
        "## Core Argument",
        "",
        (
            "COLD-Recon is positioned as a multi-source sparse-observation inverse-modelling workflow: heterogeneous "
            "permafrost observations are converted into conditioning tokens, sampled with a physics-guided conditional "
            "diffusion neural operator, and checked against synthetic full-field truth plus public-data task gates."
        ),
        "",
        "| Claim | Evidence | Status | Boundary |",
        "| --- | --- | --- | --- |",
        (
            "| COLD-Recon performs probabilistic 3D permafrost reconstruction from sparse heterogeneous observations. "
            "| Synthetic full-field validation, observation-token workflow and Nature Figure 1. "
            "| supported | Current full-field evidence is synthetic; public field data provide partial labels and proxies. |"
        ),
        (
            "| Neural posterior models outperform deterministic baselines in synthetic mean facies IoU while producing posterior ensembles. "
            f"| `outputs/tables/model_comparison.csv`: latent diffusion IoU {metric(diffusion, 'mean_iou')}, FNO-operator diffusion IoU {metric(fno, 'mean_iou')}, physics-guided training IoU {metric(physics_trained, 'mean_iou')}; Figure 1b. "
            "| supported | Not claimed as universal superiority for every continuous property or every public field task. |"
        ),
        (
            "| The paper's novelty is the auditable integration of six algorithmic components rather than an isolated model swap. "
            f"| `innovation_positioning_audit.csv`: {innovation_dimensions} innovation dimensions, mean evidence coverage {fmt(innovation_coverage)}, "
            f"{innovation_boundaries} dimensions with full boundary audits; Figure 5. "
            "| supported as positioning audit | This is not a literature-exhaustive priority claim and does not prove completed prospective EG field validation. |"
        ),
        (
            "| Posterior accuracy is reported with computational cost and artifact footprint. "
            f"| `computational_footprint.csv`: compact latent diffusion parameters {metric(footprint_latent, 'total_params_m')}M versus FNO-operator parameters {metric(footprint_fno, 'total_params_m')}M; physics-trained prediction footprint {metric(footprint_trained, 'prediction_mb')} MB; figure-atlas computational-footprint entry. "
            "| supported as audit | Parameter count and artifact size are not runtime benchmarks, but they prevent compact and high-parameter variants from being treated as cost-equivalent. |"
        ),
        (
            "| Mean facies IoU is not treated as sufficient evidence for rare cryostructure fidelity. "
            f"| `synthetic_rare_cryostructure_audit.csv`: physics-trained raw high-EIC recall {metric(rare_trained, 'raw_eic_recall')}, observation-rate-constrained recall {metric(rare_trained, 'rate_constrained_eic_recall')}, synthetic wedge-facies recall {metric(rare_trained, 'facies_6_wedge_ice_recall')}; figure-atlas rare-cryostructure entry. "
            "| supported as audit | High-EIC screening improves under an explicit operating point, but wedge-facies reconstruction remains a boundary. |"
        ),
        (
            "| Synthetic wedge-facies failure is converted into a measurable operating-point trade-off. "
            f"| `diffusion_rare_facies_hybrid_metrics.csv`: default hybrid wedge recall {metric(rare_hybrid, 'wedge_ice_recall')}, precision {metric(rare_hybrid, 'wedge_ice_precision')}, mean IoU {metric(rare_hybrid, 'mean_iou')}; `rare_facies_hybrid_operating_curve.csv`: EIC-floor 0.10 gates {metric(rare_hybrid_point, 'gate_fraction')} of voxels and changes mean IoU by {metric(rare_hybrid_point, 'mean_iou_delta_vs_base')}. "
            "| supported as operating point | This is a selectable synthetic rare-facies constraint, not a new regional wedge-ice map or a replacement for the main diffusion posterior. |"
        ),
        (
            "| Posterior uncertainty is evaluated as an error-localization signal, not only as calibrated interval width. "
            f"| `posterior_uncertainty_alignment.csv`: physics-trained EIC uncertainty-error Spearman {metric(align_trained_eic, 'spearman_uncertainty_error')} and top-uncertainty EIC error enrichment {metric(align_trained_eic, 'top_uncertainty_error_enrichment')}; physics-refined unfrozen-water Spearman {metric(align_refined_water, 'spearman_uncertainty_error')}. "
            "| supported as audit | Alignment is target-specific and weaker for facies/geophysical fields, so uncertainty is not claimed as a universal reliability certificate. |"
        ),
        (
            "| Physics projection improves implemented frozen-ground consistency diagnostics. "
            "| `outputs/tables/physics_consistency_metrics.csv` and Figure 1d. "
            "| supported | The implemented physics is simplified and diagnostic, not a full thermo-hydrological simulator. |"
        ),
        (
            f"| Public validation passes a three-source evidence gate. | `real_data_cg_gate.json`: {passed_sources} sources and {passed_tasks}/{total_tasks} tasks passed. "
            "| supported | The gate is task-specific; Jago high-EIC screening passes only with a train-split calibrated recall-oriented threshold. |"
        ),
        (
            "| Public multi-site transfer is audited with site-level wins, ties and failures. "
            f"| `external_generalization_audit.csv`: facies accuracy {metric(external_facies, 'model_value')} versus {metric(external_facies, 'baseline_value')} with site win rate {metric(external_facies, 'site_win_rate')}; EIC strict per-site best-simple failure sites {text_or_none(external_eic, 'failure_sites')} with non-inferiority rate {metric(external_eic, 'site_noninferior_rate')}; high-EIC F1 {metric(external_high_eic, 'model_value')} with non-inferiority rate {metric(external_high_eic, 'site_noninferior_rate')}; wedge recall {metric(external_wedge, 'model_value')} with precision {metric(external_wedge, 'secondary_model_value')}; Figure 6. "
            "| supported as bounded audit | EIC gains remain heterogeneous under stricter per-site baselines, so this is not claimed as universal field transfer. |"
        ),
        (
            "| EIC transfer failures are exposed and controlled by a compact-site spatial guard. "
            f"| `transfer_failure_attribution_summary.csv`: {metric(transfer_outcome_counts, 'interpretation')}; {metric(transfer_failure_counts, 'interpretation')}; Figure 7. "
            "| supported as diagnostic | Small n=5 exploratory audit; the guard is train-only and controls the exposed failures as non-inferior ties, but this is not a statistical proof of universal transfer. |"
        ),
        (
            "| Transfer applicability is audited from train-side support rather than inferred only from pooled holdout metrics. "
            f"| `domain_support_site_audit.csv`: {fmt(domain_counts['model_supported'], digits=0)} model-supported sites, "
            f"{fmt(domain_counts['guarded'], digits=0)} guarded local-prior sites, {fmt(domain_counts['low'], digits=0)} low-support sites; "
            f"{'all evaluated site-tasks are non-inferior or better' if domain_counts['all_noninferior'] else 'some evaluated site-tasks remain below the non-inferiority boundary'}; Figure 8. "
            "| supported as applicability audit | Guarded local-prior sites support non-inferiority, not unqualified model-transfer wins. |"
        ),
        (
            "| Public coordinate-label coverage is now audited as conditional EG-readiness evidence. "
            f"| `coordinate_label_coverage_summary.json`: {fmt(coordinate_summary.get('n_georeferenced_units'), digits=0)} georeferenced vertical units across {fmt(coordinate_summary.get('n_sites_with_georeferenced_units'), digits=0)} sites; {fmt(coordinate_summary.get('n_eic_measurements'), digits=0)} EIC measurements; {fmt(coordinate_summary.get('n_wedge_ice_units'), digits=0)} wedge-ice units; Figure 10. "
            "| supported as bounded data-coverage audit | These are georeferenced borehole intervals, not dense public 3D ground truth or prospective field validation. |"
        ),
        (
            "| Jago River 2018 provides a third independent ground-ice/EIC source. "
            "| `arcticdata_jago_ground_ice_observation_summary.csv`, Figure 3 and DOI 10.18739/A22J6853K. "
            "| supported | Treated as a small targeted validation, not a regional benchmark. |"
        ),
        (
            "| Wedge-ice recall handling is explicitly recall-oriented. "
            f"| `real_data_cg_benchmark.csv` wedge-ice recall task: recall {metric(wedge_event, 'model_value')} versus {metric(wedge_event, 'baseline_value')}; Figure 2d. "
            "| supported | Precision and false-positive control are treated as operating-point choices, not as solved field-cost optimization. |"
        ),
        (
            "| Wedge false-positive trade-offs are now auditable. "
            f"| `arcticdata_wedge_operating_points.csv`: recall-first pooled FPR {metric(wedge_recall_point, 'false_positive_rate')}; max-F1 threshold FPR {metric(wedge_max_f1_point, 'false_positive_rate')}; figure-atlas operating-curve entry. "
            "| supported as audit | The max-F1 threshold sacrifices recall, so it is a selectable operating point rather than the main safety-side gate. |"
        ),
        (
            "| Jago high-EIC screening closes the prior event-detection gap only under calibrated screening. "
            f"| `real_data_cg_benchmark.csv`: Jago high-EIC F1 {metric(jago_event, 'model_value')} versus {metric(jago_event, 'baseline_value')} for SpatialDepthIDW. "
            "| supported with boundary | This is a small hold-out split with an ordered-borehole convention and a training-calibrated threshold. |"
        ),
        (
            "| Posterior uncertainty can be converted into ranked additional observation targets. "
            f"| `site_investigation_boreholes.csv`: top borehole VOI {metric(top_borehole, 'voi_score')}; `site_investigation_ert_lines.csv`: top line score {metric(top_line, 'line_score')}; Figure 4. "
            "| supported as observation-design output | The VOI weights are fixed and transparent; prospective field-cost optimization is not yet validated. |"
        ),
        (
            "| VOI target ranking is retrospectively backtested under synthetic full-field truth. "
            f"| `voi_backtest_audit.csv`: composite top-VOI error enrichment {metric(voi_composite, 'top_voi_error_enrichment')}; high-EIC mismatch enrichment {metric(voi_high_eic, 'top_voi_error_enrichment')}; readiness {voi_summary.get('readiness_status', 'not available')}; Figure 11. "
            "| supported as bounded backtest | This is retrospective synthetic evidence, not a prospective field acquisition trial. |"
        ),
        (
            "| Unfrozen-water interval coverage requires bias-quantile calibration. "
            "| `uncertainty_calibration_metrics_calibrated.csv`, Figure 1c. "
            "| supported limitation | This improves interval coverage but constrains mechanistic interpretation of water-content posterior fields. |"
        ),
        (
            "| CG/EG positioning is audited rather than asserted rhetorically. "
            f"| `journal_readiness_audit.csv`: CG algorithm not-yet criteria {cg_not_yet}; EG field-generalization not-yet criteria {eg_not_yet}; Figure 9. "
            "| supported as claim-boundary audit | The audit supports a CG-plus algorithm manuscript with EG-readiness evidence, not a completed regional field-generalization claim. |"
        ),
        "",
        "## Reviewer Boundary Audit",
        "",
        "| Review dimension | Current evidence | Status | Revision implication |",
        "| --- | --- | --- | --- |",
        (
            "| Contribution and novelty | The paper combines multi-source observation tokens, conditional latent diffusion/FNO denoising, "
            "physics projection, rare-structure operating points, public-data transfer gates and VOI design, now mapped in the innovation-positioning audit. | pass | Keep the title, abstract and first two paragraphs centered on this auditable combination, not on isolated model names. |"
        ),
        (
            f"| Empirical effect | Synthetic mean-IoU gains are clear for posterior models; the public gate passes {passed_tasks}/{total_tasks} tasks across {passed_sources} sources. "
            "| pass with caveat | Report both wins and small effects, especially the modest Jago EIC RMSE reduction. |"
        ),
        (
            "| Evaluation completeness | Synthetic full-field truth is broad, public validation spans cryofacies, EIC, wedge-ice recall and high-EIC screening, and rare synthetic cryostructures have a separate operating-point audit. "
            "| pass for current algorithm article | EG-level regional generalization would still need more independent field sites with surveyed coordinates and denser labels. |"
        ),
        (
            "| Method-design soundness | Hard-data conditioning, fixed baselines, same-split comparisons and reproducibility manifests reduce hidden tuning bias. "
            "| pass with caveat | Keep Jago threshold calibration and wedge recall-first design explicit so reviewers do not read them as unconstrained detection claims. |"
        ),
        (
            "| Figure readiness | Four main figures have plotted source-data CSVs and SVG/PDF/TIFF/PNG exports. "
            "| pass | Final journal submission should still verify target-journal font sizes and figure panel specifications. |"
        ),
        "",
        "## Current Readiness Judgement",
        "",
        (
            "The evidence package is now strong enough for a CG-style computational geoscience algorithm manuscript: the model is runnable, "
            "the public-data gate is reproducible, major limitations are encoded in the claim map, and the readiness audit separates CG algorithm evidence "
            "from EG field-generalization gaps. The remaining gap to a stronger EG-style field-validation claim is not mainly wording; it is broader "
            "independent field validation with surveyed spatial coordinates, denser cryofacies/EIC labels and prospective VOI testing."
        ),
    ]
    return _write(output_path, lines)
