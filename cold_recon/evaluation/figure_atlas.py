from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


FIGURE_EXTENSIONS = (".png", ".svg", ".pdf", ".tiff")
PREFERRED_MARKDOWN_EXTENSIONS = (".png", ".svg", ".pdf", ".tiff")


@dataclass(frozen=True)
class FigureAtlasResult:
    table_csv: Path
    markdown: Path
    n_stems: int
    n_files: int
    n_submission_figures: int
    n_excluded: int
    n_previews: int


def _clean_text(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split())


def _site_from_arcticdata_stem(stem: str) -> str:
    site = stem.removeprefix("arcticdata_conditioned_diffusion_").removesuffix("_sections")
    return _clean_text(site).title()


def _classify_stem(stem: str) -> dict[str, str | bool | int]:
    lower = stem.lower()
    if lower.endswith("_preview"):
        return {
            "category_key": "qa_previews",
            "category": "QA previews and intermediate render checks",
            "order": 90,
            "manuscript_status": "qa_preview",
            "copy_to_submission": False,
            "caption": "Intermediate preview render retained for quality control, not as a manuscript claim figure.",
            "claim_role": "quality-control",
            "boundary_note": "Preview image; excluded from the claim evidence chain.",
        }
    if any(term in lower for term in ("foundation_reliability", "geotechnical_risk", "settlement")):
        return {
            "category_key": "scope_boundary_excluded",
            "category": "Scope boundary: application figures not used in this algorithm manuscript",
            "order": 80,
            "manuscript_status": "scope_boundary_excluded",
            "copy_to_submission": False,
            "caption": "Archived application-oriented figure retained in outputs but excluded from the COLD-Recon algorithm article.",
            "claim_role": "scope-boundary",
            "boundary_note": (
                "Excluded to keep this manuscript focused on probabilistic 3D permafrost structure "
                "reconstruction rather than separate risk, reliability or application-scenario papers."
            ),
        }
    if lower.startswith("nature_figure_"):
        return {
            "category_key": "main_figures",
            "category": "Main Nature-style article figures",
            "order": 0,
            "manuscript_status": "main_figure",
            "copy_to_submission": True,
            "caption": _main_figure_caption(stem),
            "claim_role": "main-text evidence",
            "boundary_note": "",
        }
    if lower in {"cold_recon_algorithm_schematic", "cold_recon_neural_operator_architecture"}:
        return {
            "category_key": "architecture_and_workflow",
            "category": "Model architecture and workflow",
            "order": 10,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": _architecture_caption(stem),
            "claim_role": "method definition",
            "boundary_note": "",
        }
    if lower == "innovation_positioning_audit":
        return {
            "category_key": "innovation_positioning",
            "category": "Innovation positioning and evidence map",
            "order": 12,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": "Evidence-mapped innovation audit linking COLD-Recon novelty claims to validation, boundaries and reproducibility artifacts.",
            "claim_role": "novelty and positioning audit",
            "boundary_note": "This figure supports bounded algorithmic positioning; it is not a literature-exhaustive priority claim or a prospective EG validation substitute.",
        }
    if lower.startswith("baseline_") or lower.startswith("fno_operator") or lower.startswith("rectified_flow"):
        return {
            "category_key": "model_comparisons_and_baselines",
            "category": "Baselines and neural-operator variants",
            "order": 25,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": _model_variant_caption(stem),
            "claim_role": "fair comparison",
            "boundary_note": "",
        }
    if (
        lower.startswith("figure_synthetic")
        or lower.startswith("synthetic_ensemble")
        or lower.startswith("synthetic_observation")
        or lower.startswith("volume_")
        or lower.startswith("borehole_")
        or lower.startswith("ablation_")
        or lower.startswith("observation_graph")
        or lower == "diffusion_posterior_sections"
    ):
        return {
            "category_key": "synthetic_validation",
            "category": "Synthetic reconstruction, ablation and observation consistency",
            "order": 20,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": _synthetic_caption(stem),
            "claim_role": "controlled validation",
            "boundary_note": "",
        }
    if (
        lower.startswith("diffusion_physics")
        or lower.startswith("diffusion_eic_std")
        or lower.startswith("diffusion_facies_entropy")
        or lower.startswith("uncertainty")
        or lower.startswith("posterior_")
        or lower.startswith("physics_consistency")
    ):
        return {
            "category_key": "physics_and_uncertainty",
            "category": "Physics guidance and posterior uncertainty",
            "order": 30,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": _physics_uncertainty_caption(stem),
            "claim_role": "uncertainty and physics audit",
            "boundary_note": "",
        }
    if lower.startswith("public_data"):
        return {
            "category_key": "public_data_validation",
            "category": "Public-data provenance and observation tokens",
            "order": 35,
            "manuscript_status": "supplementary_public_data_figure",
            "copy_to_submission": True,
            "caption": "Processed public observation-token inventory used to condition and validate COLD-Recon.",
            "claim_role": "data provenance",
            "boundary_note": "",
        }
    if lower.startswith("usgs_") or lower.startswith("arcticdata_"):
        return _public_data_classification(stem)
    if lower in {"external_generalization_audit", "transfer_failure_attribution"}:
        return {
            "category_key": "external_generalization",
            "category": "External generalization and transfer-boundary audits",
            "order": 45,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": _external_caption(stem),
            "claim_role": "stress test and failure-boundary audit",
            "boundary_note": "",
        }
    if lower == "coordinate_label_coverage_audit":
        return {
            "category_key": "external_generalization",
            "category": "External generalization and transfer-boundary audits",
            "order": 46,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": "Coordinate-label coverage audit showing that public ArcticData provides substantial georeferenced vertical cryostratigraphy labels while still falling short of dense public 3D ground truth.",
            "claim_role": "EG-readiness data-coverage audit",
            "boundary_note": "This audit upgrades coordinate-label evidence to a conditional EG-readiness component; it does not prove full regional field generalization.",
        }
    if lower == "journal_readiness_audit":
        return {
            "category_key": "journal_readiness",
            "category": "Journal readiness and claim-boundary audit",
            "order": 49,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": "CG/EG readiness audit separating completed algorithm-manuscript evidence from conditional field-generalization gaps.",
            "claim_role": "claim-boundary audit",
            "boundary_note": "This figure supports manuscript positioning; it does not convert conditional EG-readiness evidence into a completed regional field-validation claim.",
        }
    if lower == "domain_support_audit":
        return {
            "category_key": "domain_support",
            "category": "Domain-support and applicability audit",
            "order": 48,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": "Train-side site-support and applicability audit separating model-supported transfer from guarded local-prior non-inferiority.",
            "claim_role": "applicability-boundary audit",
            "boundary_note": "This audit is a deployment triage and transfer-boundary diagnostic; it is not a prospective field-validation substitute.",
        }
    if "rare" in lower or "wedge" in lower:
        return {
            "category_key": "rare_cryostructures",
            "category": "Rare cryostructure and wedge-ice operating points",
            "order": 50,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": _rare_caption(stem),
            "claim_role": "rare-event operating point",
            "boundary_note": "",
        }
    if lower.startswith("site_investigation"):
        return {
            "category_key": "observation_design",
            "category": "Posterior value-of-information observation design",
            "order": 55,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": "Posterior value-of-information recommendation map for supplemental boreholes and geophysical lines.",
            "claim_role": "observation design",
            "boundary_note": "",
        }
    if lower == "voi_backtest_audit":
        return {
            "category_key": "observation_design",
            "category": "Posterior value-of-information observation design",
            "order": 54,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": "Retrospective synthetic full-field value-of-information backtest showing whether high-ranked observation targets enrich realized reconstruction error.",
            "claim_role": "observation-design validation boundary",
            "boundary_note": "This audit supports bounded VOI readiness under synthetic truth; it is not a prospective field acquisition validation.",
        }
    if lower.startswith("computational_footprint"):
        return {
            "category_key": "model_comparisons_and_baselines",
            "category": "Baselines and neural-operator variants",
            "order": 25,
            "manuscript_status": "supplementary_algorithm_figure",
            "copy_to_submission": True,
            "caption": "Parameter count, checkpoint size, prediction footprint and posterior-sample cost audit.",
            "claim_role": "cost characteristic",
            "boundary_note": "",
        }
    return {
        "category_key": "other_algorithm_outputs",
        "category": "Other COLD-Recon algorithm outputs",
        "order": 60,
        "manuscript_status": "supplementary_algorithm_figure",
        "copy_to_submission": True,
        "caption": f"COLD-Recon output figure: {_clean_text(stem)}.",
        "claim_role": "supporting evidence",
        "boundary_note": "",
    }


def _main_figure_caption(stem: str) -> str:
    captions = {
        "nature_figure_1_overview": "Main Figure 1 summarizes the COLD-Recon workflow, controlled validation, uncertainty calibration and physics audit.",
        "nature_figure_2_real_data_gate": "Main Figure 2 reports the public-data evidence gate across independent cryofacies, EIC and wedge-ice validation tasks.",
        "nature_figure_3_cited_ground_ice": "Main Figure 3 connects cited public ground-ice observations to COLD-Recon EIC and cryostructure validation.",
        "nature_figure_4_site_investigation": "Main Figure 4 converts posterior uncertainty into value-of-information targets for supplemental observation design.",
    }
    return captions.get(stem, f"Main Nature-style COLD-Recon figure: {_clean_text(stem)}.")


def _architecture_caption(stem: str) -> str:
    if stem == "cold_recon_neural_operator_architecture":
        return "FNO-Transformer neural-operator denoiser architecture used inside the conditional diffusion posterior sampler."
    return "Sparse-observation-to-posterior COLD-Recon algorithm schematic and conditioning workflow."


def _model_variant_caption(stem: str) -> str:
    if "training_history" in stem:
        return f"Training history for the {_clean_text(stem).replace(' training history', '')} model variant."
    if stem.startswith("baseline_"):
        return f"Deterministic baseline reconstruction sections for {_clean_text(stem.removeprefix('baseline_').removesuffix('_sections'))}."
    return f"Posterior reconstruction sections or training diagnostics for {_clean_text(stem)}."


def _synthetic_caption(stem: str) -> str:
    captions = {
        "figure_synthetic_summary": "Synthetic truth, reconstruction and error sections for controlled full-field validation.",
        "synthetic_ensemble_benchmark": "Synthetic ensemble benchmark comparing reconstruction performance across generated cryostratigraphic scenes.",
        "synthetic_ensemble_facies_fractions": "Facies-fraction distribution audit for the synthetic benchmark ensemble.",
        "synthetic_observation_consistency": "Observation-consistency audit across sparse borehole, ERT, NMR and active-layer tokens.",
        "observation_graph_ablation": "Ablation comparing global token attention with kNN observation-graph attention.",
        "ablation_sparsity_curves": "Borehole sparsity and observation-source ablation curves.",
        "volume_truth_3d_overview": "Three-dimensional overview of the synthetic cryostratigraphic truth volume.",
        "volume_reconstruction_3d_overview": "Three-dimensional overview of the reconstructed posterior volume.",
        "borehole_profile_comparison": "Borehole-scale truth, sparse-observation and prediction profile comparison.",
    }
    return captions.get(stem, f"Controlled synthetic COLD-Recon validation figure: {_clean_text(stem)}.")


def _physics_uncertainty_caption(stem: str) -> str:
    captions = {
        "diffusion_eic_std_section": "Posterior standard-deviation section for excess-ice-content uncertainty.",
        "diffusion_facies_entropy_section": "Facies posterior entropy section for categorical cryostratigraphic uncertainty.",
        "uncertainty_reliability": "Raw posterior interval-reliability diagnostic under synthetic full-field truth.",
        "uncertainty_reliability_calibrated": "Post-hoc spread-calibrated posterior interval-reliability diagnostic.",
        "posterior_spread_scale_factors": "Field-wise posterior spread scale factors needed for calibrated uncertainty.",
        "posterior_uncertainty_alignment": "Rank alignment between posterior uncertainty and realized reconstruction error.",
        "physics_consistency_summary": "Physics-consistency audit for unfrozen water, resistivity and heat-balance proxies.",
    }
    return captions.get(stem, f"Physics-guided posterior or uncertainty diagnostic: {_clean_text(stem)}.")


def _public_data_classification(stem: str) -> dict[str, str | bool | int]:
    lower = stem.lower()
    if lower.startswith("arcticdata_conditioned_diffusion_") and lower.endswith("_sections"):
        caption = f"Per-site ArcticData conditioned posterior sections for {_site_from_arcticdata_stem(stem)}."
        category = "Per-site ArcticData conditioned reconstructions"
        category_key = "per_site_arcticdata_reconstructions"
        order = 42
    elif "wedge" in lower:
        caption = "ArcticData wedge-ice probability, recall and operating-point audit."
        category = "Rare cryostructure and wedge-ice operating points"
        category_key = "rare_cryostructures"
        order = 50
    elif "jago" in lower:
        caption = "Independent Jago River public ground-ice validation and conditioned posterior reconstruction."
        category = "Public-data validation and conditioned reconstructions"
        category_key = "public_data_validation"
        order = 40
    elif lower.startswith("usgs_"):
        caption = "USGS public-data validation, observation-token summary or conditioned posterior reconstruction."
        category = "Public-data validation and conditioned reconstructions"
        category_key = "public_data_validation"
        order = 40
    else:
        caption = "ArcticData public cryostratigraphy validation or conditioned posterior reconstruction."
        category = "Public-data validation and conditioned reconstructions"
        category_key = "public_data_validation"
        order = 40
    return {
        "category_key": category_key,
        "category": category,
        "order": order,
        "manuscript_status": "supplementary_public_data_figure",
        "copy_to_submission": True,
        "caption": caption,
        "claim_role": "public-data validation",
        "boundary_note": "",
    }


def _external_caption(stem: str) -> str:
    if stem == "transfer_failure_attribution":
        return "Compact-site spatial guard audit showing how exposed EIC transfer failures are controlled rather than hidden."
    return "External public multi-site generalization audit with site win rates, non-inferiority and exposed failure boundaries."


def _rare_caption(stem: str) -> str:
    if "hybrid" in stem:
        return "Synthetic rare-facies hybrid operating curve exposing wedge-recall and false-positive trade-offs."
    if "wedge" in stem:
        return "Wedge-ice operating curve exposing recall, precision and threshold choices under public-data holdout."
    return "Rare cryostructure operating-point audit separating high-EIC event screening from rare-facies recall."


def _iter_stem_groups(figure_dir: Path) -> Iterable[tuple[str, list[Path]]]:
    files = [p for p in figure_dir.iterdir() if p.is_file() and p.suffix.lower() in FIGURE_EXTENSIONS]
    by_stem: dict[str, list[Path]] = {}
    for path in files:
        by_stem.setdefault(path.stem, []).append(path)
    for stem in sorted(by_stem):
        yield stem, sorted(by_stem[stem], key=lambda p: FIGURE_EXTENSIONS.index(p.suffix.lower()))


def build_figure_atlas(root: Path, figure_dir: Path | None = None, table_dir: Path | None = None, paper_dir: Path | None = None) -> FigureAtlasResult:
    root = root.resolve()
    figure_dir = (figure_dir or root / "outputs" / "figures").resolve()
    table_dir = (table_dir or root / "outputs" / "tables").resolve()
    paper_dir = (paper_dir or root / "paper").resolve()
    if not figure_dir.exists():
        raise FileNotFoundError(f"Missing figure directory: {figure_dir}")

    rows: list[dict[str, object]] = []
    n_files = 0
    for stem, files in _iter_stem_groups(figure_dir):
        n_files += len(files)
        formats = [path.suffix.lower().lstrip(".") for path in files]
        by_suffix = {path.suffix.lower(): path for path in files}
        preferred = next((by_suffix[ext] for ext in PREFERRED_MARKDOWN_EXTENSIONS if ext in by_suffix), files[0])
        meta = _classify_stem(stem)
        rows.append(
            {
                "stem": stem,
                "category_key": meta["category_key"],
                "category": meta["category"],
                "category_order": int(meta["order"]),
                "manuscript_status": meta["manuscript_status"],
                "claim_role": meta["claim_role"],
                "copy_to_submission": bool(meta["copy_to_submission"]),
                "caption": meta["caption"],
                "boundary_note": meta["boundary_note"],
                "formats": ";".join(formats),
                "file_count": len(files),
                "preferred_path": preferred.relative_to(root).as_posix(),
                "all_paths": ";".join(path.relative_to(root).as_posix() for path in files),
            }
        )

    atlas = pd.DataFrame(rows).sort_values(["category_order", "category", "stem"]).reset_index(drop=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    table_csv = table_dir / "figure_atlas.csv"
    markdown = paper_dir / "supplementary_figure_atlas.md"
    atlas.to_csv(table_csv, index=False)
    markdown.write_text(render_figure_atlas_markdown(atlas), encoding="utf-8")

    return FigureAtlasResult(
        table_csv=table_csv,
        markdown=markdown,
        n_stems=int(len(atlas)),
        n_files=int(n_files),
        n_submission_figures=int(atlas["copy_to_submission"].sum()) if not atlas.empty else 0,
        n_excluded=int((atlas["manuscript_status"] == "scope_boundary_excluded").sum()) if not atlas.empty else 0,
        n_previews=int((atlas["manuscript_status"] == "qa_preview").sum()) if not atlas.empty else 0,
    )


def render_figure_atlas_markdown(atlas: pd.DataFrame) -> str:
    n_stems = len(atlas)
    n_claim = int(atlas["copy_to_submission"].sum()) if not atlas.empty else 0
    n_excluded = int((atlas["manuscript_status"] == "scope_boundary_excluded").sum()) if not atlas.empty else 0
    n_previews = int((atlas["manuscript_status"] == "qa_preview").sum()) if not atlas.empty else 0
    lines = [
        "# COLD-Recon Supplementary Figure Atlas",
        "",
        "**One-sentence argument.** COLD-Recon is evaluated as an algorithmic reconstruction system: the four main figures carry the concise Nature-style narrative, while this atlas maps the complete generated figure set to model design, controlled validation, public-data validation, uncertainty, physics checks, external-transfer boundaries and observation design.",
        "",
        "## Figure Contract",
        "",
        "- Core conclusion: the generated figure set supports a bounded algorithm manuscript for multi-source sparse-observation constrained probabilistic 3D permafrost reconstruction.",
        "- Archetype: quantitative and image-plate supplementary atlas.",
        "- Backend: Python-generated tables and Markdown over audited figure files.",
        "- Source data: `outputs/figures` and `outputs/tables/figure_atlas.csv`.",
        "- Reviewer risk: overloading the main text; handled by keeping four main figures and moving complete evidence coverage into this structured atlas.",
        "",
        "## Coverage Summary",
        "",
        f"- Figure stems indexed: `{n_stems}`.",
        f"- Algorithm and public-data claim figures copied to the submission supplement: `{n_claim}`.",
        f"- Scope-boundary application figures retained but not used as claims: `{n_excluded}`.",
        f"- QA preview renders retained outside the claim chain: `{n_previews}`.",
        "",
        "| category | figures | copied to submission |",
        "| --- | ---: | ---: |",
    ]
    if not atlas.empty:
        summary = (
            atlas.groupby(["category_order", "category"], as_index=False)
            .agg(figures=("stem", "count"), copied=("copy_to_submission", "sum"))
            .sort_values(["category_order", "category"])
        )
        for _, row in summary.iterrows():
            lines.append(f"| {row['category']} | {int(row['figures'])} | {int(row['copied'])} |")
    lines += ["", "## Indexed Figures", ""]

    for (_, category), group in atlas.groupby(["category_order", "category"], sort=True):
        lines += [f"### {category}", ""]
        show_images = bool(group["copy_to_submission"].any())
        for _, row in group.iterrows():
            stem = str(row["stem"])
            path = str(row["preferred_path"])
            link = f"../{path}"
            formats = str(row["formats"])
            status = str(row["manuscript_status"])
            role = str(row["claim_role"])
            caption = str(row["caption"])
            note = str(row["boundary_note"])
            lines.append(f"#### {stem}")
            lines.append("")
            lines.append(f"- Status: `{status}`; role: `{role}`; formats: `{formats}`.")
            lines.append(f"- Caption: {caption}")
            if note:
                lines.append(f"- Boundary note: {note}")
            if show_images and bool(row["copy_to_submission"]) and path.lower().endswith((".png", ".svg")):
                lines.append("")
                lines.append(f"![{stem}]({link})")
            else:
                lines.append(f"- Preferred asset: `{path}`.")
            lines.append("")
    return "\n".join(lines)
