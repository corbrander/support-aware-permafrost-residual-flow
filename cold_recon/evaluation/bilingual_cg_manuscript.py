from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cold_recon.evaluation.cg_article_builder import _build_docx, _fmt
from cold_recon.evaluation.paper_builder import REFERENCE_SUMMARIES, write_references_bib


PACKAGE_DIR = Path("paper/cg_bilingual_manuscript")
ARTICLE_EN = "cold_recon_cg_manuscript_EN"
ARTICLE_CN = "cold_recon_cg_manuscript_CN"


@dataclass(frozen=True)
class BilingualCGManuscriptResult:
    package_dir: Path
    english_md: Path
    chinese_md: Path
    english_docx: Path
    chinese_docx: Path
    figure_manifest: Path
    alignment_table: Path
    package_zip: Path
    n_figures: int


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
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


def _metric(row: pd.Series | None, key: str, digits: int = 4) -> str:
    if row is None or key not in row.index:
        return "not available"
    return _fmt(row[key], digits=digits)


def _metrics(project_root: Path) -> dict[str, Any]:
    table_dir = project_root / "outputs/tables"
    model = _read_csv(table_dir / "model_comparison.csv")
    gate = _read_csv(table_dir / "real_data_cg_benchmark.csv")
    tokens = _read_csv(table_dir / "public_data_token_inventory.csv")
    footprint = _read_csv(table_dir / "computational_footprint.csv")
    uncertainty = _read_csv(table_dir / "posterior_uncertainty_alignment.csv")
    readiness = _read_json(table_dir / "journal_readiness_summary.json")
    cg_gate = _read_json(table_dir / "real_data_cg_gate.json")
    coordinate = _read_json(table_dir / "coordinate_label_coverage_summary.json")
    voi = _read_json(table_dir / "voi_backtest_summary.json")

    latent = _pick(model, "model", "COLDReconLatentDiffusion")
    physics_trained = _pick(model, "model", "COLDReconLatentDiffusionPhysicsTrained")
    fno = _pick(model, "model", "COLDReconFNOOperatorDiffusion")
    gradient = _pick(model, "model", "GradientBoosting")
    rare_hybrid = _pick(model, "model", "COLDReconLatentDiffusionRareFaciesHybrid")
    compact = _pick(footprint, "model", "COLDReconLatentDiffusionPhysicsTrained")
    fno_footprint = _pick(footprint, "model", "COLDReconFNOOperatorDiffusion")

    eic_unc = uncertainty[
        uncertainty.get("model", pd.Series(dtype=str)).astype(str).eq("COLDReconLatentDiffusionPhysicsTrained")
        & uncertainty.get("target", pd.Series(dtype=str)).astype(str).eq("eic")
    ]
    uw_unc = uncertainty[
        uncertainty.get("model", pd.Series(dtype=str)).astype(str).eq("COLDReconLatentDiffusionPhysicsRefined")
        & uncertainty.get("target", pd.Series(dtype=str)).astype(str).eq("unfrozen_water")
    ]

    def token_count(source_key: str, observation_type: str) -> str:
        if tokens.empty:
            return "not available"
        rows = tokens[
            tokens["source_key"].astype(str).eq(source_key)
            & tokens["observation_type"].astype(str).eq(observation_type)
        ]
        if rows.empty:
            return "not available"
        return _fmt(rows.iloc[0]["n_tokens"], digits=0)

    def gate_metric(task: str, metric: str) -> pd.Series | None:
        if gate.empty:
            return None
        rows = gate[gate["task"].astype(str).eq(task) & gate["metric"].astype(str).eq(metric)]
        if rows.empty:
            return None
        return rows.iloc[0]

    cg_tier = {}
    eg_tier = {}
    for tier in readiness.get("tiers", []):
        if tier.get("tier") == "CG algorithm article":
            cg_tier = tier
        if tier.get("tier") == "EG field-generalization claim":
            eg_tier = tier

    arctic_eic = gate_metric("EIC regression", "eic_rmse_mean")
    arctic_facies = gate_metric("cryofacies", "facies_accuracy_mean")
    wedge = gate_metric("wedge-ice recall", "wedge_ice_recall_mean")
    jago = gate[(gate.get("source", pd.Series(dtype=str)).astype(str).str.contains("Jago", na=False))] if not gate.empty else pd.DataFrame()

    return {
        "gradient_iou": _metric(gradient, "mean_iou"),
        "latent_iou": _metric(latent, "mean_iou"),
        "physics_iou": _metric(physics_trained, "mean_iou"),
        "fno_iou": _metric(fno, "mean_iou"),
        "physics_eic_rmse": _metric(physics_trained, "eic_rmse"),
        "rare_wedge_recall": _metric(rare_hybrid, "wedge_ice_recall"),
        "rare_wedge_precision": _metric(rare_hybrid, "wedge_ice_precision"),
        "compact_params_m": _metric(compact, "total_params_m"),
        "compact_artifact_mb": _metric(compact, "prediction_mb"),
        "fno_params_m": _metric(fno_footprint, "total_params_m"),
        "eic_unc_spearman": _fmt(eic_unc.iloc[0]["spearman_uncertainty_error"] if not eic_unc.empty else None),
        "uw_unc_spearman": _fmt(uw_unc.iloc[0]["spearman_uncertainty_error"] if not uw_unc.empty else None),
        "public_sources": cg_gate.get("independent_public_sources_passed", "not available"),
        "passed_tasks": cg_gate.get("passed_tasks", "not available"),
        "total_tasks": cg_gate.get("total_tasks", "not available"),
        "eic_sources": cg_gate.get("eic_sources_passed", "not available"),
        "facies_sources": cg_gate.get("facies_sources_passed", "not available"),
        "usgs_ert": token_count("usgs_ert_nmr", "ert_log_resistivity"),
        "usgs_nmr": token_count("usgs_ert_nmr", "nmr_unfrozen_water"),
        "usgs_alt": token_count("usgs_ert_nmr", "alt"),
        "usgs_eic": token_count("usgs_eic_cores", "borehole_eic"),
        "arctic_facies_tokens": token_count("arcticdata_upper_permafrost_cryostratigraphy", "borehole_facies"),
        "arctic_eic_tokens": token_count("arcticdata_upper_permafrost_cryostratigraphy", "borehole_eic"),
        "jago_eic_tokens": token_count("arcticdata_jago_ground_ice_2018", "borehole_eic"),
        "arctic_facies_value": _metric(arctic_facies, "model_value"),
        "arctic_facies_baseline": _metric(arctic_facies, "baseline_value"),
        "arctic_eic_value": _metric(arctic_eic, "model_value"),
        "arctic_eic_baseline": _metric(arctic_eic, "baseline_value"),
        "wedge_value": _metric(wedge, "model_value"),
        "wedge_baseline": _metric(wedge, "baseline_value"),
        "jago_f1": _fmt(jago[jago["task"].astype(str).eq("high-EIC event")].iloc[0]["model_value"] if not jago.empty and any(jago["task"].astype(str).eq("high-EIC event")) else None),
        "coord_units": coordinate.get("n_georeferenced_units", "not available"),
        "coord_sites": coordinate.get("n_sites_with_georeferenced_units", "not available"),
        "coord_eic": coordinate.get("n_eic_measurements", "not available"),
        "coord_wedge": coordinate.get("n_wedge_ice_units", "not available"),
        "voi_composite": _fmt(voi.get("composite_top_voi_error_enrichment")),
        "voi_high_eic": _fmt(voi.get("high_eic_top_voi_error_enrichment")),
        "voi_spearman": _fmt(voi.get("composite_spearman_voi_error")),
        "cg_score": _fmt(cg_tier.get("score") if cg_tier else None),
        "cg_pass": cg_tier.get("n_pass", "not available"),
        "cg_criteria": cg_tier.get("n_criteria", "not available"),
        "eg_score": _fmt(eg_tier.get("score") if eg_tier else None),
        "eg_conditional": eg_tier.get("n_conditional", "not available"),
        "eg_not_yet": eg_tier.get("n_not_yet", "not available"),
    }


FIGURES: list[dict[str, Any]] = [
    {
        "number": 0,
        "stem": "graphical_abstract",
        "source": "generated",
        "section": "Graphical abstract",
        "en_title": "COLD-Recon evidence chain",
        "zh_title": "COLD-Recon 证据链",
        "en_claim": "COLD-Recon turns sparse multi-source observations into auditable posterior 3D permafrost reconstructions.",
        "zh_claim": "COLD-Recon 将多源稀疏观测转化为可审计的三维冻土后验重构。",
    },
    {
        "number": 1,
        "stem": "workflow_and_benchmark",
        "source": "outputs/figures/nature_figure_1_overview.png",
        "section": "Method overview and synthetic validation",
        "en_title": "COLD-Recon workflow and primary synthetic benchmark",
        "zh_title": "COLD-Recon 工作流与核心合成基准",
        "en_claim": "The method links sparse observation tokens, posterior generation, calibration and physics checking in one workflow.",
        "zh_claim": "该方法把稀疏观测 token、后验生成、校准和物理检查整合到同一工作流。",
    },
    {
        "number": 2,
        "stem": "algorithm_schematic",
        "source": "outputs/figures/cold_recon_algorithm_schematic.png",
        "section": "Method",
        "en_title": "Sparse-observation conditioning schematic",
        "zh_title": "稀疏观测条件化示意图",
        "en_claim": "Heterogeneous borehole, geophysical and thaw-depth observations are encoded through a shared conditioning interface.",
        "zh_claim": "钻孔、地球物理和融化深度观测通过统一条件接口进入模型。",
    },
    {
        "number": 3,
        "stem": "neural_operator_architecture",
        "source": "outputs/figures/cold_recon_neural_operator_architecture.png",
        "section": "Method",
        "en_title": "Conditional diffusion neural-operator architecture",
        "zh_title": "条件扩散神经算子架构",
        "en_claim": "The FNO-Transformer denoiser provides the operator variant tested against the compact latent diffusion posterior.",
        "zh_claim": "FNO-Transformer 去噪器构成与紧凑潜在扩散后验对比的神经算子变体。",
    },
    {
        "number": 4,
        "stem": "synthetic_full_field_validation",
        "source": "outputs/figures/figure_synthetic_summary.png",
        "section": "Synthetic full-field validation",
        "en_title": "Synthetic truth, reconstruction and error fields",
        "zh_title": "合成真值、重构结果与误差场",
        "en_claim": "Synthetic full-field truth provides the only complete volumetric target for facies and continuous-field scoring.",
        "zh_claim": "合成全场真值提供了当前唯一完整的体素级相类与连续场评价目标。",
    },
    {
        "number": 5,
        "stem": "three_dimensional_volume_evidence",
        "source": "composite",
        "sources": ["outputs/figures/volume_truth_3d_overview.png", "outputs/figures/volume_reconstruction_3d_overview.png"],
        "section": "Synthetic full-field validation",
        "en_title": "Three-dimensional truth and posterior reconstruction",
        "zh_title": "三维真值与后验重构体",
        "en_claim": "The reconstruction is explicitly volumetric rather than a set of isolated vertical profiles.",
        "zh_claim": "重构对象是显式三维体，而不是孤立的一维钻孔剖面。",
    },
    {
        "number": 6,
        "stem": "sparsity_and_observation_ablation",
        "source": "outputs/figures/ablation_sparsity_curves.png",
        "section": "Ablation",
        "en_title": "Borehole sparsity and observation-source ablation",
        "zh_title": "钻孔稀疏性与观测源消融",
        "en_claim": "Performance changes systematically with observation density and source availability.",
        "zh_claim": "模型性能随观测密度和观测源可用性系统变化。",
    },
    {
        "number": 7,
        "stem": "physics_consistency",
        "source": "outputs/figures/physics_consistency_summary.png",
        "section": "Physics and uncertainty",
        "en_title": "Physics-consistency diagnostics",
        "zh_title": "冻土物理一致性诊断",
        "en_claim": "Physics checks expose whether posterior fields remain compatible with unfrozen-water, resistivity and heat-balance proxies.",
        "zh_claim": "物理诊断检验后验场是否与未冻水、电阻率和热平衡代理关系一致。",
    },
    {
        "number": 8,
        "stem": "uncertainty_error_alignment",
        "source": "outputs/figures/posterior_uncertainty_alignment.png",
        "section": "Physics and uncertainty",
        "en_title": "Posterior uncertainty-error alignment",
        "zh_title": "后验不确定性与误差对齐",
        "en_claim": "Posterior spread is useful only when it localizes realized reconstruction error.",
        "zh_claim": "只有当后验离散度能够定位实际重构误差时，不确定性才具有诊断价值。",
    },
    {
        "number": 9,
        "stem": "public_data_token_inventory",
        "source": "outputs/figures/public_data_token_inventory.png",
        "section": "Public data",
        "en_title": "Processed public observation-token inventory",
        "zh_title": "公开数据观测 token 清单",
        "en_claim": "The public-data evidence chain is based on explicit processed observation counts and provenance.",
        "zh_claim": "公开数据证据链建立在可追溯的观测 token 数量和来源之上。",
    },
    {
        "number": 10,
        "stem": "public_data_evidence_gate",
        "source": "outputs/figures/nature_figure_2_real_data_gate.png",
        "section": "Public validation",
        "en_title": "Public-data evidence gate",
        "zh_title": "公开数据证据门控",
        "en_claim": "The algorithm passes task-wise public validation across independent sources without claiming dense 3D truth.",
        "zh_claim": "算法通过多个独立公开源的任务级验证，但不把这些数据夸大为密集三维真值。",
    },
    {
        "number": 11,
        "stem": "ground_ice_validation_records",
        "source": "outputs/figures/nature_figure_3_cited_ground_ice.png",
        "section": "Public validation",
        "en_title": "Ground-ice validation records",
        "zh_title": "地下冰验证记录",
        "en_claim": "USGS, ArcticData and Jago records constrain EIC and cryostructure behaviour across sources.",
        "zh_claim": "USGS、ArcticData 和 Jago 数据共同约束 EIC 与冰结构行为。",
    },
    {
        "number": 12,
        "stem": "independent_eic_holdouts",
        "source": "composite",
        "sources": ["outputs/figures/usgs_eic_holdout_validation.png", "outputs/figures/arcticdata_jago_ground_ice_holdout_validation.png"],
        "section": "Public validation",
        "en_title": "Independent USGS and Jago EIC holdouts",
        "zh_title": "USGS 与 Jago 独立 EIC 留出验证",
        "en_claim": "Independent EIC holdouts test whether the conditional posterior improves over simple same-split baselines.",
        "zh_claim": "独立 EIC 留出验证检验条件后验是否优于同划分简单基线。",
    },
    {
        "number": 13,
        "stem": "rare_cryostructure_operating_points",
        "source": "composite",
        "sources": [
            "outputs/figures/arcticdata_wedge_operating_curve.png",
            "outputs/figures/rare_facies_hybrid_operating_curve.png",
            "outputs/figures/synthetic_rare_cryostructure_audit.png",
        ],
        "section": "Rare cryostructures",
        "en_title": "Rare cryostructure operating points",
        "zh_title": "稀有冰结构操作点",
        "en_claim": "Wedge and high-EIC performance must be reported as recall-precision operating points rather than hidden inside mean IoU.",
        "zh_claim": "楔状冰和高 EIC 表现必须作为召回率-精度操作点报告，不能隐藏在平均 IoU 中。",
    },
    {
        "number": 14,
        "stem": "transfer_and_domain_boundary",
        "source": "composite",
        "sources": ["outputs/figures/external_generalization_audit.png", "outputs/figures/domain_support_audit.png"],
        "section": "Transfer boundary",
        "en_title": "External transfer and domain-support boundary",
        "zh_title": "外部迁移与域支持边界",
        "en_claim": "Transfer evidence is treated as bounded applicability evidence, not as a completed regional mapping claim.",
        "zh_claim": "迁移证据被限定为适用性边界证据，而不是已完成的区域制图结论。",
    },
    {
        "number": 15,
        "stem": "coordinate_label_coverage",
        "source": "outputs/figures/coordinate_label_coverage_audit.png",
        "section": "Transfer boundary",
        "en_title": "Public coordinate-label coverage",
        "zh_title": "公开坐标标签覆盖度",
        "en_claim": "Public ArcticData labels are substantial and georeferenced, but still sparse vertical intervals rather than dense 3D truth.",
        "zh_claim": "ArcticData 公开标签具有较大规模和地理坐标，但仍是稀疏垂向区间而非密集三维真值。",
    },
    {
        "number": 16,
        "stem": "voi_targets_and_backtest",
        "source": "composite",
        "sources": ["outputs/figures/nature_figure_4_site_investigation.png", "outputs/figures/voi_backtest_audit.png"],
        "section": "Observation design",
        "en_title": "Posterior observation-design targets and retrospective VOI backtest",
        "zh_title": "后验观测设计靶区与回顾性 VOI 验证",
        "en_claim": "Posterior uncertainty can prioritize additional boreholes and ERT lines, but the validation is retrospective rather than prospective.",
        "zh_claim": "后验不确定性可排序新增钻孔和 ERT 测线，但当前验证仍是回顾性而非前瞻性。",
    },
]


TABLE_FILES = (
    "model_comparison.csv",
    "real_data_cg_benchmark.csv",
    "real_data_cg_gate.json",
    "public_data_token_inventory.csv",
    "public_data_provenance.csv",
    "physics_consistency_metrics.csv",
    "posterior_uncertainty_alignment.csv",
    "arcticdata_wedge_operating_curve.csv",
    "rare_facies_hybrid_operating_curve.csv",
    "synthetic_rare_cryostructure_audit.csv",
    "external_generalization_audit.csv",
    "domain_support_site_audit.csv",
    "coordinate_label_coverage_audit.csv",
    "coordinate_label_coverage_summary.json",
    "voi_backtest_audit.csv",
    "voi_backtest_summary.json",
    "journal_readiness_audit.csv",
    "journal_readiness_summary.json",
)
TEXT_REPLACEMENTS = {
    "settlement_potential": "thaw_sensitive_eic_proxy",
    "settlement": "thaw_sensitive_eic_proxy",
    "Settlement": "Thaw-sensitive EIC proxy",
    "supplementary_figure_atlas": "figure_atlas",
    "supplementary_figures": "figures",
    "supplementary": "article",
    "Supplementary": "Article",
    "supplemental": "additional",
    "Supplemental": "Additional",
    "scope_boundary_excluded": "excluded_from_cg_article",
}


def _clean_text(text: str) -> str:
    cleaned = text
    for old, new in TEXT_REPLACEMENTS.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [_clean_text(str(column)) for column in cleaned.columns]
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].map(lambda value: _clean_text(str(value)) if pd.notna(value) else value)
    return cleaned


def _write_copy_script(script_path: Path, source: Path, destination_name: str, upstream: str) -> None:
    relative_source = source.as_posix()
    script_path.write_text(
        f'''from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT_ROOT / {relative_source!r}
DESTINATION = Path(__file__).resolve().parents[1] / "figures" / {destination_name!r}
UPSTREAM_GENERATOR = {upstream!r}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, DESTINATION)
    print(f"png={{DESTINATION}}")
    print(f"source={{SOURCE}}")
    print(f"upstream_generator={{UPSTREAM_GENERATOR}}")


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )


def _write_composite_script(script_path: Path, sources: list[Path], destination_name: str, columns: int = 2) -> None:
    source_list = [path.as_posix() for path in sources]
    script_path.write_text(
        f'''from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCES = [{", ".join(repr(item) for item in source_list)}]
DESTINATION = Path(__file__).resolve().parents[1] / "figures" / {destination_name!r}
COLUMNS = {columns}


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    images = []
    for item in SOURCES:
        path = PROJECT_ROOT / item
        if not path.exists():
            raise FileNotFoundError(path)
        images.append(Image.open(path).convert("RGB"))

    target_w = 1250
    padding = 38
    label_h = 42
    resized = []
    for img in images:
        scale = target_w / img.width
        resized.append(img.resize((target_w, max(1, int(img.height * scale))), Image.Resampling.LANCZOS))

    rows = (len(resized) + COLUMNS - 1) // COLUMNS
    row_heights = []
    for row in range(rows):
        start = row * COLUMNS
        row_imgs = resized[start:start + COLUMNS]
        row_heights.append(max(img.height for img in row_imgs) + label_h)

    canvas_w = COLUMNS * target_w + (COLUMNS + 1) * padding
    canvas_h = sum(row_heights) + (rows + 1) * padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(30)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    y = padding
    for row in range(rows):
        x = padding
        row_imgs = resized[row * COLUMNS:(row + 1) * COLUMNS]
        for col, img in enumerate(row_imgs):
            idx = row * COLUMNS + col
            draw.text((x, y), labels[idx], fill=(20, 20, 20), font=font)
            canvas.paste(img, (x, y + label_h))
            x += target_w + padding
        y += row_heights[row] + padding

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(DESTINATION, "PNG", optimize=True)
    print(f"png={{DESTINATION}}")


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )


def _write_graphical_abstract_script(script_path: Path, destination_name: str) -> None:
    script_path.write_text(
        f'''from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DESTINATION = Path(__file__).resolve().parents[1] / "figures" / {destination_name!r}


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=220)
    ax.set_axis_off()
    colors = ["#d7e8f7", "#e8f3df", "#f6e6d7", "#e8ddf2"]
    titles = [
        "Sparse public\\nobservations",
        "Typed observation\\ntokens",
        "Conditional diffusion\\nneural operator",
        "Audited 3D posterior",
    ]
    subtitles = [
        "Borehole facies/EIC\\nERT, NMR and ALT",
        "Location, depth, type,\\nvalue and uncertainty",
        "Latent volume sampling\\nwith physics guidance",
        "3D cryofacies, EIC,\\nuncertainty and VOI",
    ]
    xs = [0.06, 0.30, 0.54, 0.78]
    for idx, x in enumerate(xs):
        box = FancyBboxPatch(
            (x, 0.34),
            0.16,
            0.36,
            boxstyle="round,pad=0.025,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#394b59",
            facecolor=colors[idx],
        )
        ax.add_patch(box)
        ax.text(x + 0.08, 0.61, titles[idx], ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + 0.08, 0.45, subtitles[idx], ha="center", va="center", fontsize=9)
        if idx < len(xs) - 1:
            arrow = FancyArrowPatch(
                (x + 0.17, 0.52),
                (xs[idx + 1] - 0.02, 0.52),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.5,
                color="#394b59",
            )
            ax.add_patch(arrow)
    ax.text(
        0.5,
        0.86,
        "COLD-Recon: probabilistic 3D permafrost reconstruction from multi-source sparse observations",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color="#12324a",
    )
    ax.text(
        0.5,
        0.16,
        "CG-ready evidence: synthetic full-field benchmark + three-source public validation + explicit EG boundary",
        ha="center",
        va="center",
        fontsize=11,
        color="#394b59",
    )
    fig.savefig(DESTINATION, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"png={{DESTINATION}}")


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )


def _upstream_command(stem: str) -> str:
    mapping = {
        "workflow_and_benchmark": "python -m cold_recon.scripts.46_make_nature_main_figures --config configs/synth_default.yaml",
        "algorithm_schematic": "python -m cold_recon.scripts.33_make_algorithm_summary --config configs/synth_default.yaml",
        "neural_operator_architecture": "python -m cold_recon.scripts.33_make_algorithm_summary --config configs/synth_default.yaml",
        "synthetic_full_field_validation": "python -m cold_recon.scripts.31_make_diagnostic_visualizations --config configs/synth_default.yaml",
        "sparsity_and_observation_ablation": "python -m cold_recon.scripts.07_ablation --config configs/synth_default.yaml --boreholes 2,4,8",
        "physics_consistency": "python -m cold_recon.scripts.19_evaluate_physics_consistency --config configs/synth_default.yaml",
        "uncertainty_error_alignment": "python -m cold_recon.scripts.51_audit_posterior_uncertainty_alignment --config configs/synth_default.yaml",
        "public_data_token_inventory": "python -m cold_recon.scripts.35_make_public_data_provenance --config configs/synth_default.yaml",
        "public_data_evidence_gate": "python -m cold_recon.scripts.46_make_nature_main_figures --config configs/synth_default.yaml",
        "ground_ice_validation_records": "python -m cold_recon.scripts.46_make_nature_main_figures --config configs/synth_default.yaml",
        "coordinate_label_coverage": "python -m cold_recon.scripts.60_audit_coordinate_label_coverage --config configs/synth_default.yaml",
    }
    return mapping.get(stem, "See figure script and outputs/tables/reproducibility_summary.json")


def _prepare_figures(project_root: Path, package_dir: Path) -> pd.DataFrame:
    fig_dir = package_dir / "figures"
    script_dir = package_dir / "figure_scripts"
    fig_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for fig in FIGURES:
        number = int(fig["number"])
        filename = f"Fig{number:02d}_{fig['stem']}.png"
        script_path = script_dir / f"Fig{number:02d}_{fig['stem']}.py"
        if fig["source"] == "generated":
            _write_graphical_abstract_script(script_path, filename)
        elif fig["source"] == "composite":
            sources = [Path(item) for item in fig["sources"]]
            columns = 3 if len(sources) == 3 else 2
            _write_composite_script(script_path, sources, filename, columns=columns)
        else:
            source = Path(fig["source"])
            _write_copy_script(script_path, source, filename, _upstream_command(str(fig["stem"])))
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        records.append(
            {
                "figure_number": number,
                "stem": fig["stem"],
                "section": fig["section"],
                "en_title": fig["en_title"],
                "zh_title": fig["zh_title"],
                "en_claim": fig["en_claim"],
                "zh_claim": fig["zh_claim"],
                "package_png": (fig_dir / filename).as_posix(),
                "figure_script": script_path.as_posix(),
                "source": fig["source"],
                "source_paths": ";".join(fig.get("sources", [fig["source"]]) if fig["source"] != "generated" else []),
            }
        )
    return pd.DataFrame.from_records(records)


def _figure_block(lang: str, row: pd.Series) -> list[str]:
    number = int(row["figure_number"])
    title = str(row["en_title"] if lang == "en" else row["zh_title"])
    claim = str(row["en_claim"] if lang == "en" else row["zh_claim"])
    rel = Path("figures") / Path(str(row["package_png"])).name
    if lang == "en":
        if number == 0:
            caption = f"**Graphical abstract | {title}.** {claim}"
        else:
            caption = f"**Figure {number} | {title}.** {claim}"
        return [f"![Figure {number}. {title}.]({rel.as_posix()})", "", caption, ""]
    if number == 0:
        caption = f"**图文摘要 | {title}。** {claim}"
    else:
        caption = f"**图 {number} | {title}。** {claim}"
    return [f"![图 {number}. {title}.]({rel.as_posix()})", "", caption, ""]


def _references_lines(lang: str) -> list[str]:
    if lang == "en":
        lines = ["## References", ""]
        for idx, item in enumerate(REFERENCE_SUMMARIES, start=1):
            lines.append(f"{idx}. {item}")
        return lines
    zh_intro = [
        "## 参考文献",
        "",
        "以下文献条目与 `references.bib` 中的 BibTeX 记录一致，包含公开数据产品和核心方法参考。",
    ]
    for idx, item in enumerate(REFERENCE_SUMMARIES, start=1):
        zh_intro.append(f"{idx}. {item}")
    return zh_intro


def _english_manuscript(metrics: dict[str, Any], manifest: pd.DataFrame) -> list[str]:
    lines = [
        "# Physics-guided conditional diffusion neural operator for probabilistic 3D permafrost reconstruction from sparse multi-source observations",
        "",
        "## Abstract",
        "",
        (
            "Permafrost subsurface characterization is a sparse inverse problem: boreholes, core intervals and partial "
            "geophysical profiles are observed, but the scientific target is a volumetric state containing cryofacies, "
            "excess-ice content (EIC), temperature, unfrozen water and resistivity. We present COLD-Recon, a physics-guided "
            "conditional diffusion neural-operator workflow that treats this state as a posterior distribution conditioned "
            "on typed multi-source observation tokens. In synthetic full-field tests, the physics-trained posterior reached "
            f"mean facies IoU {metrics['physics_iou']}, compared with {metrics['gradient_iou']} for the strongest tree "
            f"baseline and {metrics['fno_iou']} for the FNO-Transformer operator variant, with EIC RMSE "
            f"{metrics['physics_eic_rmse']}. Public sparse-label validation passed {metrics['passed_tasks']}/"
            f"{metrics['total_tasks']} tasks across {metrics['public_sources']} independent sources, including "
            f"{metrics['eic_sources']} EIC sources and {metrics['facies_sources']} cryofacies source. The same posterior "
            "supports rare-cryostructure operating points, target-specific uncertainty-error alignment and a retrospective "
            f"value-of-information audit in which the top ranked decile enriched composite reconstruction error by "
            f"{metrics['voi_composite']}. The evidence supports a Computational Geoscience algorithm manuscript for "
            "auditable probabilistic 3D permafrost reconstruction, while retaining the boundary that public data do not yet "
            "provide independent dense 3D ground truth."
        ),
        "",
    ]
    lines.extend(_figure_block("en", manifest.iloc[0]))
    lines.extend(
        [
            "## Introduction",
            "",
            (
                "Ground ice and cryostratigraphy govern the thermal, hydrological and mechanical behaviour of permafrost "
                "terrain, but they are rarely observed as complete three-dimensional volumes. Direct evidence is usually "
                "concentrated in boreholes and core intervals, whereas electrical resistivity, nuclear magnetic resonance "
                "and active-layer observations provide indirect constraints with different spatial support. This mismatch "
                "makes deterministic interpolation an incomplete formulation: it can fill missing space, but it cannot "
                "represent the posterior ambiguity that remains away from observations or expose where the reconstruction "
                "should be trusted."
            ),
            "",
            (
                "The methodological difficulty is not only data sparsity. The observations differ in physical meaning, "
                "support volume, uncertainty and target variable, so a useful reconstruction method must condition on "
                "heterogeneous evidence without pretending that all measurements are equivalent point samples. It must also "
                "separate what can be verified with dense synthetic truth from what can only be checked against sparse public "
                "field labels. A complete algorithm paper therefore needs three linked components: a probabilistic model, "
                "a fair synthetic benchmark and an external sparse-label audit with explicit limits."
            ),
            "",
            (
                "COLD-Recon addresses this gap by treating permafrost reconstruction as conditional generation. The central "
                "object is a posterior distribution over gridded 3D frozen-ground states rather than a single best-estimate "
                "map. This manuscript makes four contributions: a shared observation-token interface for heterogeneous "
                "field evidence; a compact latent conditional diffusion posterior with a neural-operator comparison; "
                "physics, uncertainty and rare-structure diagnostics tied to measured benchmark quantities; and a public "
                "sparse-label evidence ladder that separates completed algorithmic support from the stronger field "
                "generalization claim. The graphical abstract summarizes this evidence chain and the intended scope of the "
                "article."
            ),
            "",
            "## Methods",
            "",
            "### Task formulation and observation tokens",
            "",
            (
                "Each observation is converted to a fixed-width token containing normalized x-y-z coordinates, acquisition "
                "time when available, a one-hot observation type, the scaled measurement value, reported or assigned "
                "uncertainty and a validity mask. Borehole facies and EIC intervals, ERT log-resistivity, NMR "
                "unfrozen-water proxies and active-layer measurements therefore enter a common conditioning interface "
                "without being forced into identical raster channels. Figure 1 shows how these tokens drive a conditional "
                "posterior sampler and how synthetic validation, calibration and physics checks are connected to the "
                "reconstruction workflow."
            ),
            "",
        ]
    )
    for idx in [1, 2]:
        lines.extend(_figure_block("en", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "The reconstruction target is a gridded 3D state containing categorical cryofacies and continuous fields "
                "that include EIC, temperature, unfrozen water and resistivity. The conditioning interface does not require "
                "each source to observe every target. Instead, the token mask records which quantity is observed and the "
                "model learns a posterior over the full state conditioned on the available subset. This design lets the same "
                "sampler operate under borehole-only, geophysical-only and mixed-source settings. In the current benchmark, "
                "the state tensor has seven facies channels and four continuous channels; missing observations are handled "
                "by masks rather than by replacing unobserved variables with artificial values."
            ),
            "",
            (
                "The token encoder is a compact Transformer that pools variable-length observation sets into a conditioning "
                "embedding. Padding masks keep batches with different token counts comparable, and an optional local "
                "observation graph restricts attention to spatial neighbours while preserving same-type observation links. "
                "This keeps the conditioning pathway explicit: a borehole interval, a resistivity sample and an active-layer "
                "measurement remain distinguishable tokens until they are summarized for the denoiser."
            ),
            "",
            "### Conditional diffusion neural operator",
            "",
        ]
    )
    lines.extend(_figure_block("en", manifest.iloc[3]))
    lines.extend(
        [
            (
                "The generative component operates in the latent space of a 3D autoencoder. A denoiser is trained to "
                "predict Gaussian noise added to latent volumes along the diffusion schedule, conditioned on the pooled "
                "observation-token embedding. At inference, repeated reverse-diffusion trajectories are decoded to produce "
                "posterior samples of facies probabilities and continuous fields rather than a single deterministic map."
            ),
            "",
            (
                "The compact physics-trained posterior uses "
                f"{metrics['compact_params_m']} million parameters and writes a {metrics['compact_artifact_mb']} MB "
                f"prediction artifact, whereas the FNO-Transformer operator variant uses {metrics['fno_params_m']} million "
                "parameters. The FNO-Transformer denoiser combines low-frequency 3D Fourier convolution, FiLM conditioning "
                "from the observation embedding and a small Transformer over pooled latent tokens. We evaluated IDW, Random "
                "Forest, Gradient Boosting, Kriging/GPR, sparse 3D U-Net, implicit coordinate fields, latent diffusion, "
                "FNO-Transformer diffusion and rectified flow under the same synthetic target so that deterministic "
                "interpolation, deterministic neural prediction and posterior generation could be compared directly."
            ),
            "",
            (
                "Physics enters the workflow through training and post-sampling audit terms rather than through an "
                "unverified claim of physical realism. The physics-trained objective combines the diffusion noise-prediction "
                "loss with decoded-volume penalties for unfrozen-water consistency, empirical log-resistivity consistency, "
                "steady heat-balance residuals and physical range barriers, plus latent, facies and continuous-field anchor "
                "terms. These terms discourage implausible couplings among EIC, temperature, unfrozen water and resistivity, "
                "while the final claims remain tied to measured benchmark and audit metrics."
            ),
            "",
            "### Evaluation design",
            "",
            (
                "The evaluation is organized as an evidence ladder. Synthetic full-field volumes provide dense targets for "
                "voxel-level reconstruction metrics, ablations and uncertainty-error checks. Public borehole and "
                "geophysical releases then test whether the same conditioning interface improves the sparse cryofacies, EIC "
                "and rare-structure tasks that those records can actually label, using task-specific baselines on the same "
                "splits. The final readiness audit separates the completed Computational Geoscience algorithm claim from "
                "the stronger field-generalization claim, which would require independent dense 3D ground truth. This design "
                "prevents sparse public labels from being overinterpreted as full-volume validation."
            ),
            "",
            "## Results",
            "",
            "### Synthetic full-field reconstruction and ablation",
            "",
            (
                "Synthetic full-field truth provides the complete volumetric target required for facies IoU, EIC error and "
                "continuous-field diagnostics. The strongest classical baseline reached mean facies IoU "
                f"{metrics['gradient_iou']}. The compact latent diffusion posterior reached {metrics['latent_iou']}, the "
                f"FNO-Transformer operator posterior reached {metrics['fno_iou']}, and the physics-trained posterior reached "
                f"{metrics['physics_iou']} with EIC RMSE {metrics['physics_eic_rmse']}. Figures 4 and 5 show that the "
                "evaluation is volumetric: the method is tested against gridded truth and reconstructed 3D fields, not only "
                "against isolated vertical profiles."
            ),
            "",
        ]
    )
    for idx in [4, 5, 6]:
        lines.extend(_figure_block("en", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "The ablation result is important because sparse reconstruction can improve for the wrong reason if a model "
                "only memorizes a smooth spatial prior. The borehole-density and source-removal tests therefore ask whether "
                "performance changes when information is actually withheld. The observed sensitivity to observation density "
                "and source availability supports the interpretation that COLD-Recon uses the conditioning tokens rather "
                "than only reproducing an unconditional training prior."
            ),
            "",
            (
                "The synthetic improvement should be read as an incremental reconstruction gain, not as a claim that the "
                "inverse problem has been solved. Its importance is that the gain occurs under a controlled full-field target "
                "and is accompanied by lower EIC error, uncertainty diagnostics and a compact artifact footprint. The "
                "FNO-Transformer result is therefore useful as a neural-operator stress test, while the compact "
                "physics-trained posterior remains the main reported model because it gives the best benchmark score with a "
                "substantially smaller parameter count."
            ),
            "",
            "### Physics consistency and posterior uncertainty",
            "",
            (
                "Physics and uncertainty diagnostics are treated as evidence rather than decoration. The posterior must "
                "preserve plausible relationships among EIC, unfrozen water, temperature and resistivity, and its uncertainty "
                "must localize realized errors under synthetic truth. For the physics-trained posterior, EIC uncertainty had "
                f"Spearman correlation {metrics['eic_unc_spearman']} with absolute EIC error; after physics refinement, "
                f"unfrozen-water uncertainty had correlation {metrics['uw_unc_spearman']} with absolute unfrozen-water error. "
                "Figures 7 and 8 therefore support a target-specific uncertainty claim, not a blanket reliability statement."
            ),
            "",
        ]
    )
    for idx in [7, 8]:
        lines.extend(_figure_block("en", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "These diagnostics do not claim that every posterior standard deviation is perfectly calibrated. They show "
                "a narrower and more useful result: for the reported targets, posterior spread was aligned with realized "
                "error strongly enough to guide model checking and observation design. This distinction is central for "
                "using generative reconstruction in permafrost applications, where uncertainty maps can otherwise look "
                "convincing without being tied to error."
            ),
            "",
            "### Public sparse-label validation",
            "",
            (
                "The public-data evidence chain was built from explicit processed tokens and source provenance. The current "
                f"inventory contains {metrics['usgs_ert']} ERT log-resistivity tokens, {metrics['usgs_nmr']} NMR tokens, "
                f"{metrics['usgs_alt']} active-layer tokens, {metrics['usgs_eic']} USGS EIC intervals, "
                f"{metrics['arctic_facies_tokens']} ArcticData facies tokens, {metrics['arctic_eic_tokens']} ArcticData EIC "
                f"tokens and {metrics['jago_eic_tokens']} Jago EIC tokens. This inventory is shown before the validation "
                "metrics because the public-data claim depends on what was actually processed."
            ),
            "",
        ]
    )
    lines.extend(_figure_block("en", manifest.iloc[9]))
    lines.extend(
        [
            (
                f"The task-wise public evidence gate passed {metrics['passed_tasks']}/{metrics['total_tasks']} checks across "
                f"{metrics['public_sources']} independent sources. ArcticData cryofacies accuracy improved from "
                f"{metrics['arctic_facies_baseline']} to {metrics['arctic_facies_value']}, ArcticData EIC RMSE improved from "
                f"{metrics['arctic_eic_baseline']} to {metrics['arctic_eic_value']}, and the wedge recall head improved "
                f"mean recall from {metrics['wedge_baseline']} to {metrics['wedge_value']}. The Jago River branch provides "
                f"a third independent EIC/ground-ice source and reached high-EIC F1 {metrics['jago_f1']} on the current "
                "hold-out split. Figures 10 to 12 connect these numbers to the underlying validation records."
            ),
            "",
        ]
    )
    for idx in [10, 11, 12]:
        lines.extend(_figure_block("en", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "The public validation is deliberately phrased as sparse-label validation. It tests whether conditioning on "
                "multi-source observations improves the tasks that the public records can actually support. It does not "
                "convert those records into a dense 3D truth volume. This prevents the common overclaim in subsurface "
                "machine learning, where a map-like output is evaluated only at sparse points but discussed as if the whole "
                "volume had been independently observed."
            ),
            "",
            "### Rare cryostructures, transfer boundary and observation design",
            "",
            (
                "Rare cryostructures are evaluated separately from mean IoU because high-EIC and wedge-ice behaviour can be "
                "scientifically important even when their voxel fraction is small. The synthetic rare-facies hybrid reached "
                f"wedge recall {metrics['rare_wedge_recall']} and precision {metrics['rare_wedge_precision']}, while the "
                "public ArcticData wedge head is deliberately reported as a recall-oriented operating point. Figure 13 makes "
                "this recall-precision trade-off explicit."
            ),
            "",
        ]
    )
    lines.extend(_figure_block("en", manifest.iloc[13]))
    lines.extend(
        [
            (
                "External transfer is presented as bounded algorithmic evidence rather than as a completed regional field "
                "generalization claim. Across the public ArcticData branch, transfer audits separate model-supported sites "
                "from guarded local-prior cases. The coordinate-label audit shows that the public ArcticData inventory "
                f"contains {metrics['coord_units']} georeferenced vertical units across {metrics['coord_sites']} sites, "
                f"including {metrics['coord_eic']} EIC measurements and {metrics['coord_wedge']} wedge-ice units. However, "
                "these are still sparse vertical labels rather than dense public 3D ground truth."
            ),
            "",
        ]
    )
    for idx in [14, 15]:
        lines.extend(_figure_block("en", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "Finally, the posterior is converted into a value-of-information diagnostic for additional boreholes and ERT "
                "lines. The synthetic retrospective audit showed that the top VOI decile enriched composite reconstruction "
                f"error by {metrics['voi_composite']} and high-EIC mismatch by {metrics['voi_high_eic']}, with Spearman "
                f"correlation {metrics['voi_spearman']} between VOI score and composite error. Figure 16 therefore supports "
                "VOI as a posterior blind-spot diagnostic, while retaining the boundary that it is not yet a prospective field "
                "acquisition trial."
            ),
            "",
        ]
    )
    lines.extend(_figure_block("en", manifest.iloc[16]))
    lines.extend(
        [
            "## Discussion",
            "",
            (
                "The evidence supports COLD-Recon as a Computational Geoscience-ready algorithmic framework for "
                "probabilistic 3D permafrost reconstruction. Its strength is not a single accuracy number, but the agreement "
                "between synthetic full-field validation, physically interpretable posterior checks, uncertainty-error "
                "alignment, public multi-source validation and explicit rare-structure operating points. The approach is "
                "designed for sparse site characterization workflows where hard observations, posterior uncertainty and rare "
                "cryostructures must remain visible to the analyst."
            ),
            "",
            (
                "The main methodological advance is the alignment between the conditioning representation and the geological "
                "observation problem. Boreholes, ERT, NMR and thaw-depth observations are not forced into a single rasterized "
                "input channel with identical support. They remain typed observations that condition a posterior over a full "
                "3D state. This is why the paper can connect reconstruction accuracy, physics checks, uncertainty and VOI "
                "without changing the basic data interface between experiments."
            ),
            "",
            (
                "A practical consequence is that the output should be read as an auditable posterior product. The posterior "
                "mean provides one reconstruction, but the spread, rare-structure operating point and VOI score identify "
                "where additional observations would most change the interpretation. This is a different product from a "
                "deterministic regional permafrost map, and the manuscript is intentionally framed around that distinction."
            ),
            "",
            (
                "The results also show where the approach remains fragile. Wedge-ice detection is best reported as an "
                "operating point, because recall and precision move against each other under class imbalance. Transfer "
                "evidence is useful but conditional, because public georeferenced labels are sparse vertical intervals and "
                "not independent dense volumes. VOI is promising as a blind-spot diagnostic, but the reported evidence is "
                "retrospective and synthetic rather than a prospective field campaign."
            ),
            "",
            (
                "The main limitation is also explicit. Public permafrost releases provide direct vertical labels, EIC intervals "
                "and dense geophysical proxies, but not independent dense 3D ground-truth volumes. Full-field reconstruction "
                "metrics therefore remain synthetic, and public validation must be interpreted as sparse-label and proxy "
                "evidence. The journal-readiness audit records the algorithm tier as "
                f"{metrics['cg_pass']}/{metrics['cg_criteria']} criteria passed with score {metrics['cg_score']}, whereas "
                f"the field-generalization tier remains score {metrics['eg_score']} with {metrics['eg_conditional']} conditional "
                f"criteria and {metrics['eg_not_yet']} not-yet criterion. This manuscript therefore makes a completed algorithm "
                "claim, not a completed regional mapping claim."
            ),
            "",
            "## Conclusion",
            "",
            (
                "COLD-Recon provides a reproducible method for multi-source sparse-observation constrained probabilistic "
                "3D permafrost reconstruction. The current evidence justifies a Computational Geoscience algorithm article: "
                "the model is benchmarked under synthetic full-field truth, checked against public data from three independent "
                "sources, audited for physics and uncertainty, stress-tested for rare cryostructures, and linked to posterior "
                "observation design. Future progress should focus on prospective field campaigns and public dense 3D validation "
                "datasets that can move the remaining EG boundary from explicit limitation to completed field evidence."
            ),
            "",
            "## Data and code availability",
            "",
            (
                "The manuscript package contains the English and Chinese manuscripts, 17 PNG figures, one Python reproduction "
                "script per figure, source-data files, copied audit tables, BibTeX references and a figure-text alignment "
                "table. Public data provenance is retained through the copied tables and source-data files, and the package "
                "can be rebuilt with `python -m cold_recon.scripts.63_build_bilingual_cg_manuscript --config "
                "configs/synth_default.yaml` from the project root."
            ),
            "",
        ]
    )
    lines.extend(_references_lines("en"))
    return lines


def _chinese_manuscript(metrics: dict[str, Any], manifest: pd.DataFrame) -> list[str]:
    lines = [
        "# 多源稀疏观测约束的三维冻土结构概率重构：一种冻土物理引导的条件扩散神经算子",
        "",
        "## 摘要",
        "",
        (
            "冻土地下结构通常只能通过稀疏钻孔和局部地球物理测线间接认识，但研究对象却是三维空间中的冻土相、"
            "超量冰含量、温度、未冻水和电阻率场。本文提出 COLD-Recon，一种冻土物理引导的条件扩散神经算子"
            "工作流，将该问题表述为由多源稀疏观测 token 条件化的三维后验生成。合成全场真值实验中，物理训练"
            f"后的后验模型达到平均相类 IoU {metrics['physics_iou']}，高于最强树模型基线的 {metrics['gradient_iou']}，"
            f"并与 FNO-Transformer 神经算子变体的 {metrics['fno_iou']} 处于同一性能量级。公开数据验证在 "
            f"{metrics['public_sources']} 个独立公开源上通过 {metrics['passed_tasks']}/{metrics['total_tasks']} 个任务，"
            f"其中包括 {metrics['eic_sources']} 个 EIC 数据源和 {metrics['facies_sources']} 个冻土相数据源。模型进一步报告"
            f"稀有冰结构操作点、后验不确定性与误差对齐关系，以及回顾性 VOI 审计；其中 VOI 排名前 10% 区域对综合重构"
            f"误差的富集倍数为 {metrics['voi_composite']}。这些证据支持一篇 Computational Geoscience 算法论文，"
            "但不把当前公开数据夸大为已完成的区域三维地下冰泛化验证，因为独立密集三维真值仍然缺失。"
        ),
        "",
    ]
    lines.extend(_figure_block("zh", manifest.iloc[0]))
    lines.extend(
        [
            "## 引言",
            "",
            (
                "地下冰与冻土沉积结构控制冻土区的热、水文和地貌行为，但它们很少以完整三维体的形式被观测到。"
                "直接证据通常集中在钻孔和岩心区间，电阻率、核磁共振和活动层厚度观测则提供空间支撑不同的间接约束。"
                "这种观测与目标之间的不匹配，使简单确定性插值不足以支撑三维冻土结构重构：插值可以填补空间空白，"
                "但不能表达远离观测处仍然存在的后验不确定性，也难以指出重构结果在哪些区域更可靠。"
            ),
            "",
            (
                "这个困难不只是样本少。不同观测源具有不同的物理含义、空间支撑、不确定性和目标变量，因此一个有用的"
                "重构方法必须能够条件化异质证据，而不能把所有测量都等同为同一种点样本。它还必须区分两类证据：合成"
                "全场真值能够支持体素级指标和消融实验，公开野外数据则更多支持稀疏标签和代理量验证。完整的算法论文因此"
                "需要同时给出概率模型、公平合成基准和带有边界声明的公开数据审计。"
            ),
            "",
            (
                "COLD-Recon 将冻土结构重构视为条件生成问题。其核心对象不是一张单一最优三维图，而是给定多源观测后的"
                "三维冻土状态后验分布。本文的算法贡献包括：统一的多源稀疏观测 token 接口、潜在空间条件扩散神经算子、"
                "冻土物理和校准诊断、稀有冰结构操作点、公开数据证据门控，以及用于新增观测设计的后验 VOI 诊断。"
                "图文摘要概括了这条证据链和本文的算法定位。"
            ),
            "",
            "## 方法",
            "",
            "### 任务定义与观测 token",
            "",
            (
                "每一个观测被编码为包含空间位置、深度支撑、观测类型、数值、不确定性和掩膜信息的 token。钻孔冻土相"
                "和 EIC 区间、ERT 对数电阻率、NMR 未冻水代理量以及活动层厚度观测因此可以进入同一个条件化接口。"
                "图 1 展示了这些 token 如何驱动条件后验采样，并把合成验证、校准和物理诊断连接成完整工作流。"
            ),
            "",
        ]
    )
    for idx in [1, 2]:
        lines.extend(_figure_block("zh", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "重构目标是一个三维格网状态，其中同时包含离散冻土相和 EIC、温度、未冻水、电阻率等连续场。条件化接口"
                "不要求每个数据源都观测所有目标变量，而是用 token 掩膜记录当前观测对应的物理量。模型学习的是在可用观测"
                "子集条件下的完整三维状态后验，因此同一采样器可以处理仅钻孔、仅地球物理和多源混合观测场景。"
            ),
            "",
            "### 条件扩散神经算子",
            "",
        ]
    )
    lines.extend(_figure_block("zh", manifest.iloc[3]))
    lines.extend(
        [
            (
                "生成模型在冻土三维体的潜在表示中运行。紧凑的物理训练后验模型包含 "
                f"{metrics['compact_params_m']} 百万个参数，预测文件大小为 {metrics['compact_artifact_mb']} MB；"
                f"FNO-Transformer 神经算子变体包含 {metrics['fno_params_m']} 百万个参数。本文在同一合成目标上评价 "
                "IDW、随机森林、梯度提升、Kriging/GPR、稀疏观测 3D U-Net、隐式坐标场、潜在扩散、FNO-Transformer "
                "扩散和 rectified flow，从而把确定性插值、确定性神经预测和后验生成放在同一证据框架下比较。"
            ),
            "",
            (
                "神经算子变体使用 Fourier 式空间混合和 token 条件化 Transformer 结构。本文把它作为算子基线报告，而"
                "不是把它当作唯一模型，因为核心科学问题是条件后验生成是否能在相同输入约束下改善稀疏三维重构。物理信息"
                "通过训练和审计项进入流程，用于惩罚 EIC、未冻水、温度和电阻率之间不合理的耦合；最终论断仍然只绑定到"
                "已报告的基准和审计指标。"
            ),
            "",
            "### 评价设计",
            "",
            (
                "评价按证据阶梯组织。合成全场体提供体素级重构指标和消融实验所需的密集目标；公开钻孔和地球物理数据"
                "进一步检验同一条件化接口是否能在独立来源上改善冻土相、EIC 和稀有冰结构任务。最后的 readiness 审计"
                "把已经完成的 Computational Geoscience 算法论断与更强的场地泛化论断分开，后者仍需要独立密集三维真值。"
            ),
            "",
            "## 结果",
            "",
            "### 合成全场重构与消融",
            "",
            (
                "合成全场真值提供了评价冻土相 IoU、EIC 误差和连续场诊断所需的完整体素目标。最强经典基线的平均相类 "
                f"IoU 为 {metrics['gradient_iou']}；紧凑潜在扩散后验达到 {metrics['latent_iou']}，FNO-Transformer "
                f"神经算子后验达到 {metrics['fno_iou']}，物理训练后验达到 {metrics['physics_iou']}，其 EIC RMSE 为 "
                f"{metrics['physics_eic_rmse']}。图 4 和图 5 说明这里评价的是显式三维体，而不是少数孤立钻孔剖面。"
            ),
            "",
        ]
    )
    for idx in [4, 5, 6]:
        lines.extend(_figure_block("zh", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "消融结果之所以重要，是因为稀疏重构模型如果只记住平滑空间先验，也可能在表面上取得较好指标。钻孔密度和"
                "观测源移除实验检验了信息被真正拿掉时性能是否随之变化。模型对观测密度和观测源可用性的系统响应支持这样"
                "的解释：COLD-Recon 的结果来自条件 token 的使用，而不只是无条件训练先验的复现。"
            ),
            "",
            "### 物理一致性与后验不确定性",
            "",
            (
                "物理一致性与不确定性诊断被作为证据链的一部分，而不是附加装饰。后验场需要维持 EIC、未冻水、温度和"
                "电阻率之间的合理关系，其不确定性也应在合成真值下能够定位实际误差。物理训练后验的 EIC 不确定性与"
                f"EIC 绝对误差的 Spearman 相关为 {metrics['eic_unc_spearman']}；物理修正后，未冻水不确定性与未冻水"
                f"绝对误差的相关为 {metrics['uw_unc_spearman']}。因此，图 7 和图 8 支持的是目标变量特定的不确定性诊断，"
                "而不是对所有变量的一般可靠性承诺。"
            ),
            "",
        ]
    )
    for idx in [7, 8]:
        lines.extend(_figure_block("zh", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "这些诊断并不声称每一个后验标准差都已经完美校准。它们支持的是一个更窄、更有用的结论：在报告的目标变量"
                "上，后验离散度与实际误差具有足够对应关系，可以用于模型检查和观测设计。这个区分很重要，因为在冻土应用中，"
                "不确定性图如果没有与误差挂钩，也可能看起来合理但缺乏诊断价值。"
            ),
            "",
            "### 公开稀疏标签验证",
            "",
            (
                "公开数据证据链首先建立在明确的处理后 token 数量和来源记录上。当前清单包含 "
                f"{metrics['usgs_ert']} 个 ERT 对数电阻率 token、{metrics['usgs_nmr']} 个 NMR token、"
                f"{metrics['usgs_alt']} 个活动层 token、{metrics['usgs_eic']} 个 USGS EIC 区间、"
                f"{metrics['arctic_facies_tokens']} 个 ArcticData 冻土相 token、{metrics['arctic_eic_tokens']} 个 "
                f"ArcticData EIC token，以及 {metrics['jago_eic_tokens']} 个 Jago EIC token。先报告该清单，是因为公开"
                "数据结论必须对应到实际处理过的数据。"
            ),
            "",
        ]
    )
    lines.extend(_figure_block("zh", manifest.iloc[9]))
    lines.extend(
        [
            (
                f"任务级公开证据门控在 {metrics['public_sources']} 个独立公开源上通过 "
                f"{metrics['passed_tasks']}/{metrics['total_tasks']} 个检查。ArcticData 冻土相准确率从 "
                f"{metrics['arctic_facies_baseline']} 提高到 {metrics['arctic_facies_value']}；ArcticData EIC RMSE "
                f"从 {metrics['arctic_eic_baseline']} 降至 {metrics['arctic_eic_value']}；楔状冰召回头将平均召回率从 "
                f"{metrics['wedge_baseline']} 提高到 {metrics['wedge_value']}。Jago River 分支提供第三个独立 EIC/"
                f"地下冰数据源，并在当前留出划分上达到高 EIC F1 {metrics['jago_f1']}。图 10 至图 12 将这些指标与"
                "具体公开验证记录相对应。"
            ),
            "",
        ]
    )
    for idx in [10, 11, 12]:
        lines.extend(_figure_block("zh", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "公开验证被刻意表述为稀疏标签验证。它检验的是多源观测条件化是否改善了公开记录实际能够支持的任务，而不是"
                "把这些记录转化成密集三维真值。这样可以避免地下机器学习中常见的过度表述：模型输出看起来像完整三维地图，"
                "但评价只发生在稀疏点或稀疏区间上，却被讨论成整个三维体已经被独立验证。"
            ),
            "",
            "### 稀有冰结构、迁移边界与观测设计",
            "",
            (
                "稀有冰结构需要从平均 IoU 中分离出来评价，因为高 EIC 与楔状冰在体素比例很小时仍可能具有重要的冻土学意义。"
                f"合成稀有相混合操作点的楔状冰召回率为 {metrics['rare_wedge_recall']}，精度为 "
                f"{metrics['rare_wedge_precision']}；公开 ArcticData 楔状冰头则被明确报告为偏召回的操作点。图 13 "
                "直接展示了召回率与误报/精度之间的权衡。"
            ),
            "",
        ]
    )
    lines.extend(_figure_block("zh", manifest.iloc[13]))
    lines.extend(
        [
            (
                "外部迁移结果被解释为有边界的算法适用性证据，而不是已完成的区域泛化结论。ArcticData 分支的迁移审计"
                "区分了模型支持的站点和需要局部先验保护的站点。坐标标签覆盖审计表明，公开 ArcticData 清单包含 "
                f"{metrics['coord_units']} 个具备地理坐标的垂向单元，分布在 {metrics['coord_sites']} 个站点，其中包括 "
                f"{metrics['coord_eic']} 个 EIC 测量和 {metrics['coord_wedge']} 个楔状冰单元。然而，这些数据仍是"
                "稀疏垂向标签，而不是密集公开三维真值。"
            ),
            "",
        ]
    )
    for idx in [14, 15]:
        lines.extend(_figure_block("zh", manifest.iloc[idx]))
    lines.extend(
        [
            (
                "最后，后验结果被转化为新增钻孔和 ERT 测线的 VOI 诊断。合成回顾性审计显示，VOI 排名前 10% 的区域对"
                f"综合重构误差的富集倍数为 {metrics['voi_composite']}，对高 EIC 错配的富集倍数为 "
                f"{metrics['voi_high_eic']}，VOI 分数与综合误差的 Spearman 相关为 {metrics['voi_spearman']}。"
                "因此，图 16 支持 VOI 作为后验盲区诊断工具，但不把它说成已经完成前瞻性野外采集优化。"
            ),
            "",
        ]
    )
    lines.extend(_figure_block("zh", manifest.iloc[16]))
    lines.extend(
        [
            "## 讨论",
            "",
            (
                "综合证据支持 COLD-Recon 作为一套达到 CG 算法论文要求的三维冻土概率重构框架。其核心强度不在于某一个"
                "孤立精度指标，而在于合成全场验证、物理可解释后验诊断、不确定性与误差对齐、公开多源验证以及稀有冰结构"
                "操作点之间形成了相互支撑的证据链。该方法适用于稀疏场地勘察场景，因为在这类场景中，硬观测、后验不确定性"
                "和稀有冰结构都必须对分析者保持可见。"
            ),
            "",
            (
                "主要方法贡献在于条件化表示与冻土观测问题之间的匹配。钻孔、ERT、NMR 和活动层厚度没有被强行压成具有相同"
                "空间支撑的栅格通道，而是保持为有类型的观测 token，去条件化完整三维状态后验。因此，本文能够在同一数据接口"
                "下连接重构精度、物理检查、不确定性和 VOI。"
            ),
            "",
            (
                "结果也暴露了方法仍然脆弱的地方。楔状冰检测更适合报告为操作点，因为在类别极不平衡条件下召回率和精度会互相"
                "牵制。迁移证据有价值但仍是有条件的，因为公开地理坐标标签是稀疏垂向区间，而不是独立密集三维体。VOI 可作为"
                "后验盲区诊断，但当前证据是合成回顾性审计，而不是前瞻性野外采集试验。"
            ),
            "",
            (
                "本文的主要限制同样明确。公开冻土数据能够提供直接垂向标签、EIC 区间和较密集地球物理代理量，但尚不能提供"
                "独立密集三维地下冰或冻土相真值。因此，全场重构指标仍然依赖合成真值，公开数据验证应被解释为稀疏标签和代理"
                f"证据。当前 readiness 审计中，CG 算法层级为 {metrics['cg_pass']}/{metrics['cg_criteria']} 项通过，"
                f"得分 {metrics['cg_score']}；EG 场地泛化层级得分 {metrics['eg_score']}，其中 "
                f"{metrics['eg_conditional']} 项为条件证据，{metrics['eg_not_yet']} 项尚未满足。本文因此主张的是"
                "完整算法证据，而不是完成区域三维制图泛化。"
            ),
            "",
            "## 结论",
            "",
            (
                "COLD-Recon 提供了一种可复现的多源稀疏观测约束三维冻土概率重构方法。当前证据足以支撑一篇 "
                "Computational Geoscience 算法文章：模型在合成全场真值上与基线比较，在三个独立公开数据源上验证，"
                "接受物理一致性和不确定性审计，对稀有冰结构进行独立操作点评估，并与后验观测设计相连接。下一步最关键的"
                "工作不是简单扩大模型，而是通过前瞻性野外验证和公开密集三维真值数据集，将当前明确保留的 EG 边界转化为"
                "完成的场地证据。"
            ),
            "",
            "## 数据和代码可用性",
            "",
            (
                "本包包含中英文手稿、17 张 PNG 图、每张图对应的 Python 复现脚本、source data、复制的审计表、BibTeX "
                "参考文献和图文对应表。公开数据来源通过复制的表格和 source-data 文件保留；在项目根目录运行 "
                "`python -m cold_recon.scripts.63_build_bilingual_cg_manuscript --config configs/synth_default.yaml` "
                "可以重新生成该投稿包。"
            ),
            "",
        ]
    )
    lines.extend(_references_lines("zh"))
    return lines


def _copy_tables(project_root: Path, package_dir: Path) -> None:
    table_dir = package_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLE_FILES:
        src = project_root / "outputs/tables" / name
        if src.exists():
            dst = table_dir / src.name
            if src.suffix.lower() == ".csv":
                _clean_df(pd.read_csv(src)).to_csv(dst, index=False)
            elif src.suffix.lower() == ".json":
                dst.write_text(_clean_text(src.read_text(encoding="utf-8")), encoding="utf-8")
            else:
                shutil.copy2(src, dst)
    source_dir = package_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    selected_source_data = {
        "nature_figure_1_source_data.csv",
        "nature_figure_2_source_data.csv",
        "nature_figure_3_source_data.csv",
        "nature_figure_4_source_data.csv",
        "posterior_uncertainty_alignment_source_data.csv",
        "coordinate_label_coverage_audit_source_data.csv",
        "voi_backtest_audit_source_data.csv",
        "external_generalization_audit_source_data.csv",
        "domain_support_audit_source_data.csv",
    }
    for name in sorted(selected_source_data):
        src = project_root / "outputs/source_data" / name
        if src.exists():
            _clean_df(pd.read_csv(src)).to_csv(source_dir / src.name, index=False)


def _write_readme(package_dir: Path, n_figures: int) -> Path:
    path = package_dir / "README.md"
    lines = [
        "# COLD-Recon bilingual CG manuscript package",
        "",
        "This package contains the revised Chinese and English Computational Geoscience manuscript.",
        "",
        f"It includes {n_figures} PNG figures, one Python script per figure, bilingual Markdown and DOCX manuscripts, source data, audited tables, and a figure-text alignment table.",
        "",
        "The figure set is curated for manuscript relevance. It excludes application-oriented risk or foundation-reliability material and does not use unrelated generated figures.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _zip_package(package_dir: Path) -> Path:
    zip_path = package_dir / "cold_recon_bilingual_cg_manuscript.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path == zip_path or path.is_dir():
                continue
            zf.write(path, path.relative_to(package_dir))
    return zip_path


def build_bilingual_cg_manuscript(project_root: Path = Path("."), package_dir: Path = PACKAGE_DIR) -> BilingualCGManuscriptResult:
    project_root = project_root.resolve()
    package_dir = (project_root / package_dir).resolve() if not package_dir.is_absolute() else package_dir
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest = _prepare_figures(project_root, package_dir)
    metrics = _metrics(project_root)
    english_md = package_dir / f"{ARTICLE_EN}.md"
    chinese_md = package_dir / f"{ARTICLE_CN}.md"
    english_md.write_text("\n".join(_english_manuscript(metrics, manifest)).rstrip() + "\n", encoding="utf-8")
    chinese_md.write_text("\n".join(_chinese_manuscript(metrics, manifest)).rstrip() + "\n", encoding="utf-8")

    english_docx = package_dir / f"{ARTICLE_EN}.docx"
    chinese_docx = package_dir / f"{ARTICLE_CN}.docx"
    _build_docx(english_md, english_docx)
    _build_docx(chinese_md, chinese_docx)

    manifest_path = package_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    alignment = manifest[["figure_number", "stem", "section", "en_claim", "zh_claim", "package_png", "figure_script"]].copy()
    alignment_path = package_dir / "figure_text_alignment.csv"
    alignment.to_csv(alignment_path, index=False)
    write_references_bib(package_dir)
    _copy_tables(project_root, package_dir)
    _write_readme(package_dir, len(manifest))
    zip_path = _zip_package(package_dir)
    return BilingualCGManuscriptResult(
        package_dir=package_dir,
        english_md=english_md,
        chinese_md=chinese_md,
        english_docx=english_docx,
        chinese_docx=chinese_docx,
        figure_manifest=manifest_path,
        alignment_table=alignment_path,
        package_zip=zip_path,
        n_figures=len(manifest),
    )
