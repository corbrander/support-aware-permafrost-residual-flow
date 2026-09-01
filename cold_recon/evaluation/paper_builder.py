from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REFERENCE_BIBTEX: dict[str, str] = {
    "james2020usgs_geophysics": """@misc{james2020usgs_geophysics,
  author = {James, Stephanie R. and Minsley, Burke J. and Pastick, Neal J. and Sullivan, Taylor D.},
  title = {Alaska permafrost characterization: Geophysical and related field data collected from 2019-2020},
  year = {2020},
  publisher = {U.S. Geological Survey data release},
  doi = {10.5066/P9I6VUQV},
  url = {https://doi.org/10.5066/P9I6VUQV}
}""",
    "stephani2025usgs_eic": """@misc{stephani2025usgs_eic,
  author = {Stephani, Eva and Baxter, Brian and Pitcher, Lincoln and Sullivan, Taylor and Baughman, Caitlin and Rosenkrans, Hannah and Bray, Maegan and Giesche, Anna and Blauvelt, Daniel},
  title = {Measurements and photographs of permafrost cores drilled on the Arctic Coastal Plain, Alaska},
  year = {2025},
  publisher = {U.S. Geological Survey data release},
  doi = {10.5066/P13AEEH7},
  url = {https://doi.org/10.5066/P13AEEH7}
}""",
    "kanevskiy2024upper_permafrost": """@misc{kanevskiy2024upper_permafrost,
  author = {Kanevskiy, Mikhail},
  title = {Cryostratigraphy and ground-ice content of the upper permafrost in Alaska and Northern Canada, 2018-2023},
  year = {2024},
  publisher = {NSF Arctic Data Center},
  doi = {10.18739/A2QR4NS3D},
  url = {https://doi.org/10.18739/A2QR4NS3D}
}""",
    "kanevskiy2020jago_ground_ice": """@misc{kanevskiy2020jago_ground_ice,
  author = {Kanevskiy, Mikhail},
  title = {Cryostratigraphy and ground-ice content of the upper permafrost at the Jago River study site, Northern Alaska, July-August 2018},
  year = {2020},
  publisher = {NSF Arctic Data Center},
  doi = {10.18739/A22J6853K},
  url = {https://doi.org/10.18739/A22J6853K}
}""",
    "brown1998calm": """@misc{brown1998calm,
  author = {Brown, Jerry},
  title = {Circumpolar Active-Layer Monitoring (CALM) Program: Description and data},
  year = {1998},
  publisher = {National Snow and Ice Data Center},
  url = {https://nsidc.org/data/ggd313/versions/1}
}""",
    "gtnp2021magt": """@misc{gtnp2021magt,
  author = {{GTN-P}},
  title = {Long-term mean annual ground temperature data for permafrost},
  year = {2021},
  publisher = {PANGAEA},
  doi = {10.1594/PANGAEA.930669},
  url = {https://doi.org/10.1594/PANGAEA.930669}
}""",
    "esa2025permafrost": """@misc{esa2025permafrost,
  author = {{ESA Climate Change Initiative}},
  title = {Permafrost active layer thickness for the Northern Hemisphere, v5.0, 1997-2023},
  year = {2025},
  publisher = {Centre for Environmental Data Analysis},
  doi = {10.5285/a6fbedd8ee5b472c8e84e55f746c1704},
  url = {https://climate.esa.int/en/projects/permafrost/data/}
}""",
    "porter2022arcticdem": """@misc{porter2022arcticdem,
  author = {Porter, Claire and others},
  title = {ArcticDEM - Strips, Version 4.1},
  year = {2022},
  publisher = {Harvard Dataverse},
  doi = {10.7910/DVN/C98DVS},
  url = {https://www.pgc.umn.edu/data/arcticdem/}
}""",
    "claverie2018hls": """@article{claverie2018hls,
  author = {Claverie, Martin and Ju, Junchang and Masek, Jeffrey G. and Dungan, Jennifer L. and Vermote, Eric F. and Roger, Jean-Claude and Skakun, Sergii V. and Justice, Christopher},
  title = {The Harmonized Landsat and Sentinel-2 surface reflectance data set},
  journal = {Remote Sensing of Environment},
  year = {2018},
  volume = {219},
  pages = {145--161},
  doi = {10.1016/j.rse.2018.09.002}
}""",
    "poggio2021soilgrids": """@article{poggio2021soilgrids,
  author = {Poggio, Laura and de Sousa, Luis M. and Batjes, Niels H. and Heuvelink, Gerard B. M. and Kempen, Bas and Ribeiro, Eloi and Rossiter, David},
  title = {SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty},
  journal = {SOIL},
  year = {2021},
  volume = {7},
  pages = {217--240},
  doi = {10.5194/soil-7-217-2021}
}""",
    "munozsabater2021era5land": """@article{munozsabater2021era5land,
  author = {Munoz-Sabater, Joaquin and Dutra, Emanuel and Agusti-Panareda, Anna and Albergel, Clement and Arduini, Gabriele and Balsamo, Gianpaolo and Boussetta, Souhail and Choulga, Margarita and Harrigan, Shaun and Hersbach, Hans and Martens, Brecht and Miralles, Diego G. and Piles, Maria and Rodriguez-Fernandez, Nemesio J. and Zsoter, Ervin and Buontempo, Carlo and Thepaut, Jean-Noel},
  title = {ERA5-Land: a state-of-the-art global reanalysis dataset for land applications},
  journal = {Earth System Science Data},
  year = {2021},
  volume = {13},
  pages = {4349--4383},
  doi = {10.5194/essd-13-4349-2021}
}""",
    "breiman2001randomforest": """@article{breiman2001randomforest,
  author = {Breiman, Leo},
  title = {Random Forests},
  journal = {Machine Learning},
  year = {2001},
  volume = {45},
  pages = {5--32},
  doi = {10.1023/A:1010933404324}
}""",
    "vaswani2017attention": """@inproceedings{vaswani2017attention,
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  title = {Attention is All You Need},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017},
  url = {https://arxiv.org/abs/1706.03762}
}""",
    "tancik2020fourier": """@inproceedings{tancik2020fourier,
  author = {Tancik, Matthew and Srinivasan, Pratul P. and Mildenhall, Ben and Fridovich-Keil, Sara and Raghavan, Nithin and Singhal, Utkarsh and Ramamoorthi, Ravi and Barron, Jonathan T. and Ng, Ren},
  title = {Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
  url = {https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html}
}""",
    "ho2020ddpm": """@inproceedings{ho2020ddpm,
  author = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  title = {Denoising Diffusion Probabilistic Models},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
  doi = {10.48550/arXiv.2006.11239}
}""",
    "rombach2022latent": """@inproceedings{rombach2022latent,
  author = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bjorn},
  title = {High-Resolution Image Synthesis with Latent Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2022},
  pages = {10684--10695},
  doi = {10.1109/CVPR52688.2022.01042}
}""",
    "gneiting2007proper": """@article{gneiting2007proper,
  author = {Gneiting, Tilmann and Raftery, Adrian E.},
  title = {Strictly Proper Scoring Rules, Prediction, and Estimation},
  journal = {Journal of the American Statistical Association},
  year = {2007},
  volume = {102},
  number = {477},
  pages = {359--378},
  doi = {10.1198/016214506000001437}
}""",
}


REFERENCE_SUMMARIES: list[str] = [
    "James et al. (2020), USGS Alaska permafrost ERT/NMR/thaw-depth data, DOI: 10.5066/P9I6VUQV.",
    "Stephani et al. (2025), USGS Arctic Coastal Plain permafrost core EIC data, DOI: 10.5066/P13AEEH7.",
    "Kanevskiy et al. (2024), Arctic Data Center upper-permafrost cryostratigraphy and ground-ice data, DOI: 10.18739/A2QR4NS3D.",
    "Kanevskiy (2020), Arctic Data Center Jago River 2018 cryostratigraphy and ground-ice data, DOI: 10.18739/A22J6853K.",
    "Brown (1998), Circumpolar Active-Layer Monitoring Program data, NSIDC GGD313.",
    "GTN-P (2021), long-term mean annual ground temperature permafrost borehole data, DOI: 10.1594/PANGAEA.930669.",
    "ESA CCI Permafrost (2025), Northern Hemisphere active-layer thickness v5.0, DOI: 10.5285/a6fbedd8ee5b472c8e84e55f746c1704.",
    "Porter et al. (2022), ArcticDEM strips v4.1, DOI: 10.7910/DVN/C98DVS.",
    "Claverie et al. (2018), Harmonized Landsat and Sentinel-2 surface reflectance, DOI: 10.1016/j.rse.2018.09.002.",
    "Poggio et al. (2021), SoilGrids 2.0, DOI: 10.5194/soil-7-217-2021.",
    "Munoz-Sabater et al. (2021), ERA5-Land, DOI: 10.5194/essd-13-4349-2021.",
    "Breiman (2001), Random Forests, DOI: 10.1023/A:1010933404324.",
    "Vaswani et al. (2017), attention/Transformer sequence modeling.",
    "Tancik et al. (2020), Fourier feature mappings for coordinate MLPs.",
    "Ho et al. (2020), denoising diffusion probabilistic models, DOI: 10.48550/arXiv.2006.11239.",
    "Rombach et al. (2022), latent diffusion models, DOI: 10.1109/CVPR52688.2022.01042.",
    "Gneiting and Raftery (2007), proper scoring rules including CRPS, DOI: 10.1198/016214506000001437.",
]


def read_table(table_dir: Path, name: str) -> pd.DataFrame:
    path = table_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def df_to_markdown(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "Not generated."
    view = df.copy()
    if columns is not None:
        view = view[[col for col in columns if col in view.columns]]
    if max_rows is not None:
        view = view.head(max_rows)
    cols = [str(col) for col in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt_value(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


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


def metric_text(row: pd.Series | None, key: str, default: str = "not available") -> str:
    if row is None or key not in row.index or pd.isna(row[key]):
        return default
    return fmt_value(row[key])


def _figure(path: str, caption: str) -> str:
    return f"![{caption}]({path})\n\n*{caption}*"


def write_references_bib(paper_dir: Path) -> Path:
    paper_dir.mkdir(parents=True, exist_ok=True)
    out = paper_dir / "references.bib"
    out.write_text("\n\n".join(REFERENCE_BIBTEX.values()) + "\n", encoding="utf-8")
    return out


def references_markdown() -> str:
    return "\n".join(f"{idx}. {line}" for idx, line in enumerate(REFERENCE_SUMMARIES, start=1))


def build_manuscript(table_dir: Path, output_path: Path) -> Path:
    write_references_bib(output_path.parent)
    model = read_table(table_dir, "model_comparison.csv")
    architecture = read_table(table_dir, "model_architecture_summary.csv")
    computational_footprint = read_table(table_dir, "computational_footprint.csv")
    innovation_positioning = read_table(table_dir, "innovation_positioning_audit.csv")
    ablation = read_table(table_dir, "ablation_metrics.csv")
    calib = read_table(table_dir, "uncertainty_calibration_metrics.csv")
    calib_scaled = read_table(table_dir, "uncertainty_calibration_metrics_calibrated.csv")
    spread_factors = read_table(table_dir, "posterior_spread_calibration_factors.csv")
    physics = read_table(table_dir, "physics_consistency_metrics.csv")
    synthetic_summary = read_table(table_dir, "synthetic_ensemble_summary.csv")
    synthetic_benchmark = read_table(table_dir, "synthetic_ensemble_benchmark.csv")
    obs_consistency = read_table(table_dir, "synthetic_observation_consistency.csv")
    graph_ablation = read_table(table_dir, "observation_graph_ablation.csv")
    rare_cryostructure = read_table(table_dir, "synthetic_rare_cryostructure_audit.csv")
    rare_hybrid_metrics = read_table(table_dir, "diffusion_rare_facies_hybrid_metrics.csv")
    rare_hybrid_curve = read_table(table_dir, "rare_facies_hybrid_operating_curve.csv")
    uncertainty_alignment = read_table(table_dir, "posterior_uncertainty_alignment.csv")
    usgs_eic = read_table(table_dir, "usgs_eic_summary.csv")
    usgs_eic_holdout = read_table(table_dir, "usgs_eic_holdout_metrics.csv")
    usgs_eic_diffusion = read_table(table_dir, "usgs_eic_conditioned_diffusion_metrics.csv")
    jago_summary = read_table(table_dir, "arcticdata_jago_ground_ice_observation_summary.csv")
    jago_holdout = read_table(table_dir, "arcticdata_jago_ground_ice_eic_holdout_metrics.csv")
    jago_diffusion = read_table(table_dir, "arcticdata_jago_ground_ice_conditioned_diffusion_metrics.csv")
    jago_comparison = read_table(table_dir, "arcticdata_jago_ground_ice_conditioned_diffusion_comparison.csv")
    real_data_cg = read_table(table_dir, "real_data_cg_benchmark.csv")
    wedge_operating_points = read_table(table_dir, "arcticdata_wedge_operating_points.csv")
    external_generalization = read_table(table_dir, "external_generalization_audit.csv")
    external_site_deltas = read_table(table_dir, "external_generalization_site_deltas.csv")
    transfer_failure_diagnostics = read_table(table_dir, "transfer_failure_site_diagnostics.csv")
    transfer_failure_summary = read_table(table_dir, "transfer_failure_attribution_summary.csv")
    domain_support = read_table(table_dir, "domain_support_site_audit.csv")
    usgs_geo = read_table(table_dir, "usgs_geophysics_summary.csv")
    usgs_holdout = read_table(table_dir, "usgs_field_holdout_metrics.csv")
    usgs_real = read_table(table_dir, "usgs_real_conditioned_diffusion_metrics.csv")
    public_provenance = read_table(table_dir, "public_data_provenance.csv")
    public_tokens = read_table(table_dir, "public_data_token_inventory.csv")
    site_boreholes = read_table(table_dir, "site_investigation_boreholes.csv")
    site_lines = read_table(table_dir, "site_investigation_ert_lines.csv")

    diff_row = pick_row(model, "model", "COLDReconLatentDiffusion")
    fno_row = pick_row(model, "model", "COLDReconFNOOperatorDiffusion")
    flow_row = pick_row(model, "model", "COLDReconRectifiedFlow")
    trained_row = pick_row(model, "model", "COLDReconLatentDiffusionPhysicsTrained")
    guided_row = pick_row(model, "model", "COLDReconLatentDiffusionPhysicsGuided")
    refined_row = pick_row(model, "model", "COLDReconLatentDiffusionPhysicsRefined")
    footprint_latent = pick_row(computational_footprint, "model", "COLDReconLatentDiffusion")
    footprint_fno = pick_row(computational_footprint, "model", "COLDReconFNOOperatorDiffusion")
    footprint_trained = pick_row(computational_footprint, "model", "COLDReconLatentDiffusionPhysicsTrained")
    footprint_hybrid = pick_row(computational_footprint, "model", "COLDReconLatentDiffusionRareFaciesHybrid")
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
    implicit_row = pick_row(model, "model", "COLDReconImplicit")
    rf_row = pick_row(model, "model", "RandomForest")
    gb_row = pick_row(model, "model", "GradientBoosting")
    kriging_row = pick_row(model, "model", "KrigingGPR")
    unet_row = pick_row(model, "model", "SparseUNet3D")
    eic_row = pick_row(calib, "target", "eic")
    alt_row = pick_row(calib, "target", "active_layer_thickness")
    ice_row = pick_row(calib, "target", "ice_rich_probability")
    facies_row = pick_row(calib, "target", "facies")
    eic_scaled_row = pick_row(calib_scaled, "target", "eic")
    temp_scaled_row = pick_row(calib_scaled, "target", "temperature")
    unfrozen_scaled_row = pick_row(calib_scaled, "target", "unfrozen_water")
    log_rho_scaled_row = pick_row(calib_scaled, "target", "log_resistivity")
    implicit_phys_row = pick_row(physics, "model", "COLDReconImplicit")
    diffusion_phys_row = pick_row(physics, "model", "COLDReconLatentDiffusion")
    fno_phys_row = pick_row(physics, "model", "COLDReconFNOOperatorDiffusion")
    flow_phys_row = pick_row(physics, "model", "COLDReconRectifiedFlow")
    trained_phys_row = pick_row(physics, "model", "COLDReconLatentDiffusionPhysicsTrained")
    guided_phys_row = pick_row(physics, "model", "COLDReconLatentDiffusionPhysicsGuided")
    refined_phys_row = pick_row(physics, "model", "COLDReconLatentDiffusionPhysicsRefined")
    usgs_phys_row = pick_row(physics, "model", "USGSRealConditionedDiffusion")
    usgs_eic_phys_row = pick_row(physics, "model", "USGSEICConditionedDiffusion")
    usgs_real_row = usgs_real.iloc[0] if not usgs_real.empty else None
    eic_global_row = pick_row(usgs_eic_holdout, "model", "GlobalMean")
    eic_spatial_row = pick_row(usgs_eic_holdout, "model", "SpatialDepthIDW")
    eic_diffusion_row = usgs_eic_diffusion.iloc[0] if not usgs_eic_diffusion.empty else None
    jago_global_row = pick_row(jago_comparison, "model", "GlobalMean")
    jago_spatial_row = pick_row(jago_comparison, "model", "SpatialDepthIDW")
    jago_model_row = pick_row(jago_comparison, "model", "COLDReconJagoGroundIceConditionedDiffusion")
    jago_diffusion_row = jago_diffusion.iloc[0] if not jago_diffusion.empty else None
    cg_passed = (
        real_data_cg["passed"].map(lambda value: str(value).strip().lower() == "true")
        if not real_data_cg.empty and "passed" in real_data_cg.columns
        else pd.Series(dtype=bool)
    )
    cg_passed_tasks = int(cg_passed.sum()) if not cg_passed.empty else 0
    cg_total_tasks = int(len(real_data_cg))
    cg_passed_sources = int(real_data_cg.loc[cg_passed, "source"].nunique()) if not cg_passed.empty and "source" in real_data_cg.columns else 0
    wedge_recall_point = pick_row(wedge_operating_points, "operating_point", "current site-calibrated recall-first head")
    wedge_max_f1_point = pick_row(wedge_operating_points, "operating_point", "pooled max-F1 probability threshold")
    wedge_knn_point = pick_row(wedge_operating_points, "operating_point", "SpatialDepthKNN baseline")
    external_facies = pick_row(external_generalization, "task", "cryofacies")
    external_eic = pick_row(external_generalization, "task", "EIC regression")
    external_wedge = pick_row(external_generalization, "task", "wedge-ice recall")
    external_high_eic = pick_row(external_generalization, "task", "high-EIC event")
    transfer_outcome_counts = pick_row(transfer_failure_summary, "signal", "EIC outcome counts")
    transfer_failure_counts = pick_row(transfer_failure_summary, "signal", "failure attribution counts")
    domain_classes = (
        domain_support["applicability_class"].astype(str)
        if not domain_support.empty and "applicability_class" in domain_support.columns
        else pd.Series(dtype=str)
    )
    domain_model_supported = int(domain_classes.eq("model-supported transfer").sum()) if not domain_classes.empty else 0
    domain_guarded = int(domain_classes.eq("guarded local-prior").sum()) if not domain_classes.empty else 0
    domain_low = int(domain_classes.eq("low support").sum()) if not domain_classes.empty else 0
    domain_outcome_cols = [
        col for col in ("facies_outcome", "eic_outcome", "high_eic_outcome", "wedge_outcome") if col in domain_support.columns
    ]
    domain_all_noninferior = (
        bool(domain_support[domain_outcome_cols].astype(str).isin(["win", "noninferior", "not_evaluated"]).all().all())
        if domain_outcome_cols
        else False
    )
    rare_trained_row = pick_row(rare_cryostructure, "model", "COLDReconLatentDiffusionPhysicsTrained")
    rare_implicit_row = pick_row(rare_cryostructure, "model", "COLDReconImplicit")
    rare_hybrid_row = pick_row(rare_hybrid_metrics, "model", "COLDReconLatentDiffusionRareFaciesHybrid")
    rare_hybrid_default = (
        rare_hybrid_curve[pd.to_numeric(rare_hybrid_curve["eic_floor"], errors="coerce").round(6).eq(0.1)].iloc[0]
        if not rare_hybrid_curve.empty and "eic_floor" in rare_hybrid_curve.columns and not rare_hybrid_curve[pd.to_numeric(rare_hybrid_curve["eic_floor"], errors="coerce").round(6).eq(0.1)].empty
        else None
    )
    align_trained_eic = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsTrained", "eic")
    align_refined_eic = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsRefined", "eic")
    align_refined_water = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsRefined", "unfrozen_water")
    align_trained_facies = pick_model_target(uncertainty_alignment, "COLDReconLatentDiffusionPhysicsTrained", "facies")
    synthetic_summary_view = (
        synthetic_summary[
            synthetic_summary["metric"].isin(
                [
                    "n_observations",
                    "eic_mean",
                    "ice_rich_fraction",
                    "temperature_mean",
                    "active_layer_mean",
                    "truth_heat_residual_rmse",
                    "truth_unfrozen_water_empirical_mae",
                    "truth_log_resistivity_empirical_mae",
                ]
            )
        ]
        if not synthetic_summary.empty and "metric" in synthetic_summary.columns
        else synthetic_summary
    )
    rare_view = (
        rare_cryostructure[
            rare_cryostructure["model"].isin(
                [
                    "GradientBoosting",
                    "COLDReconImplicit",
                    "COLDReconLatentDiffusion",
                    "COLDReconFNOOperatorDiffusion",
                    "COLDReconRectifiedFlow",
                    "COLDReconLatentDiffusionPhysicsTrained",
                    "COLDReconLatentDiffusionRareFaciesHybrid",
                    "COLDReconLatentDiffusionPhysicsRefined",
                ]
            )
        ]
        if not rare_cryostructure.empty and "model" in rare_cryostructure.columns
        else rare_cryostructure
    )
    uncertainty_alignment_view = (
        uncertainty_alignment[
            uncertainty_alignment["model"].isin(
                [
                    "COLDReconLatentDiffusion",
                    "COLDReconFNOOperatorDiffusion",
                    "COLDReconRectifiedFlow",
                    "COLDReconLatentDiffusionPhysicsTrained",
                    "COLDReconLatentDiffusionPhysicsRefined",
                    "COLDReconLatentDiffusionCalibrated",
                ]
            )
        ]
        if not uncertainty_alignment.empty and "model" in uncertainty_alignment.columns
        else uncertainty_alignment
    )
    obs_consistency_view = (
        obs_consistency[
            obs_consistency["model"].isin(
                [
                    "truth",
                    "IDW",
                    "RandomForest",
                    "GradientBoosting",
                    "KrigingGPR",
                    "SparseUNet3D",
                    "COLDReconImplicit",
                    "COLDReconLatentDiffusion",
                    "COLDReconFNOOperatorDiffusion",
                    "COLDReconRectifiedFlow",
                    "COLDReconLatentDiffusionPhysicsTrained",
                    "COLDReconLatentDiffusionPhysicsRefined",
                ]
            )
        ]
        if not obs_consistency.empty and "model" in obs_consistency.columns
        else obs_consistency
    )

    lines: list[str] = [
        "# Multi-source sparse-observation constrained probabilistic 3D permafrost reconstruction with a physics-guided conditional diffusion neural operator",
        "",
        "## Abstract",
        "",
        (
            "Sparse boreholes and geophysical surveys rarely provide the volumetric information needed for permafrost "
            "site characterization. We present COLD-Recon, a physics-guided conditional latent diffusion framework "
            "inspired by diffusion and latent diffusion models [@ho2020ddpm; @rombach2022latent] for "
            "probabilistic three-dimensional reconstruction of cryostratigraphy, excess ice content, thermal state, "
            "unfrozen water, and resistivity from heterogeneous site-investigation data. "
            "The method tokenizes borehole, ERT, NMR, and active-layer observations, combines them with environmental "
            "surface features, reconstructs implicit or voxelized permafrost property fields, and samples posterior "
            "realizations in a learned latent space. Differentiable permafrost constraints, posterior hard-data "
            "guidance, physics-guided denoiser fine-tuning, latent-space physics guidance, and posterior physics projection link the generated fields to unfrozen-water physics, simplified heat consistency, geophysical "
            "responses and heat-consistency diagnostics. Synthetic experiments provide full 3D ground-truth "
            f"validation, where the latent diffusion model reaches mean facies IoU {metric_text(diff_row, 'mean_iou')} "
            f"and the FNO-Transformer neural-operator denoiser reaches {metric_text(fno_row, 'mean_iou')}, "
            f"while the rectified-flow posterior reaches {metric_text(flow_row, 'mean_iou')}, "
            f"and physics-guided fine-tuning reaches {metric_text(trained_row, 'mean_iou')}, "
            f"and a physics-refined posterior reduces unfrozen-water RMSE to {metric_text(refined_row, 'unfrozen_water_rmse')}. Public USGS Alaska ERT/NMR/thaw-depth data are used "
            "[@james2020usgs_geophysics] "
            f"for field validation, with real-token conditioned diffusion giving ERT RMSE "
            f"{metric_text(usgs_real_row, 'ert_log_resistivity_rmse')} and ALT RMSE {metric_text(usgs_real_row, 'alt_rmse')}. "
            f"USGS core EIC leave-one-borehole-out validation gives ordered spatial-depth IDW MAE {metric_text(eic_spatial_row, 'mae')} "
            f"against a global-mean MAE {metric_text(eic_global_row, 'mae')}. "
            f"An EIC-core conditioned diffusion posterior obtains hold-out EIC RMSE {metric_text(eic_diffusion_row, 'holdout_eic_rmse')}. "
            "A third independent Arctic Data Center Jago River ground-ice source "
            "[@kanevskiy2020jago_ground_ice] contributes 29 EIC tokens; with conservative EIC guidance, "
            f"the Jago-conditioned posterior obtains hold-out EIC RMSE {metric_text(jago_model_row, 'eic_rmse')} "
            f"against the best simple baseline RMSE {metric_text(jago_global_row, 'eic_rmse')}. "
            f"The real-data evidence gate passes {cg_passed_sources} public sources and {cg_passed_tasks}/{cg_total_tasks} tasks, "
            "including a train-split calibrated Jago high-EIC screening head that is reported as recall-oriented rather than precision-optimized. "
            "The posterior ensemble provides calibrated uncertainty maps and observation-consistency diagnostics for sparse-data reconstruction."
        ),
        "",
        "## 1. Introduction",
        "",
        (
            "Permafrost site characterization requires 3D information about cryofacies, excess ice, temperature, "
            "unfrozen water, and geophysical state. Conventional interpolation of sparse boreholes is not "
            "sufficient when the subsurface contains ice-rich layers, taliks, wedge ice, and strong lateral variability. "
            "COLD-Recon addresses this inverse problem as posterior generation, estimating p(M|O), where M is the "
            "3D permafrost state and O denotes heterogeneous sparse observations."
        ),
        "",
        "The contributions of this implementation are:",
        "",
        "1. A reproducible synthetic cryostratigraphy generator with 3D full-field truth and sparse borehole/ERT/NMR/ALT sampling.",
        "2. A unified observation-token interface for heterogeneous permafrost site-investigation data.",
        "3. Conditional implicit and latent diffusion reconstruction models with permafrost physics utilities.",
        "4. Gradient Boosting, fixed-kernel Kriging/GPR, and sparse-observation 3D U-Net baselines for tree-ensemble, geostatistical, and deterministic deep comparison.",
        "5. A runnable FNO-Transformer hybrid denoiser that makes the diffusion model a neural-operator variant.",
        "6. A warm-start rectified-flow posterior sampler as a flow-matching generative alternative.",
        "7. Public USGS and Arctic Data Center data loaders with hold-out validation workflows for thaw depth, NMR, ERT, cryofacies, and EIC records.",
        "8. A cross-source real-data evidence gate requiring three independent public validation sources, plus posterior uncertainty, calibration, and sparse-observation consistency diagnostics.",
        "",
        "Innovation-positioning audit:",
        "",
        df_to_markdown(
            innovation_positioning,
            columns=[
                "innovation_dimension",
                "evidence_summary",
                "evidence_coverage_score",
                "current_maturity",
                "maturity_gap_to_eg",
                "allowed_claim",
                "boundary",
            ],
        ),
        "",
        (
            "This audit is used to raise the manuscript's positioning without overclaiming. "
            f"It maps {len(innovation_positioning)} innovation dimensions to method definition, controlled validation, baseline comparison, public-data evidence, "
            f"failure-boundary reporting and reproducibility traceability. The mean evidence-coverage score is {fmt_value(innovation_coverage)}, "
            f"and {innovation_boundary_count} dimensions have full boundary audits. The remaining maturity gaps identify prospective EG validation needs rather than being hidden in the prose."
        ),
        "",
        _figure(
            "../outputs/figures/innovation_positioning_audit.png",
            "Figure S0. Innovation-positioning audit mapping each COLD-Recon novelty claim to evidence coverage, current maturity, EG target maturity and claim boundaries.",
        ),
        "",
        "## 2. Public Data",
        "",
        "The current reproducible validation uses public USGS and Arctic Data Center datasets [@james2020usgs_geophysics; @stephani2025usgs_eic; @kanevskiy2024upper_permafrost; @kanevskiy2020jago_ground_ice] and keeps interfaces for additional CALM, GTN-P, ESA CCI, ArcticDEM, HLS, SoilGrids, and ERA5-Land products [@brown1998calm; @gtnp2021magt; @esa2025permafrost; @porter2022arcticdem; @claverie2018hls; @poggio2021soilgrids; @munozsabater2021era5land].",
        "",
        df_to_markdown(
            pd.DataFrame(
                [
                    {
                        "dataset": "USGS Alaska permafrost characterization",
                        "variables": "ERT inverted models, NMR water content, thaw depth",
                        "role": "field observation tokens and hold-out validation",
                        "citation": "[@james2020usgs_geophysics]",
                        "local_output": "data/processed/usgs_geophysics_observations.npz",
                    },
                    {
                        "dataset": "USGS Arctic Coastal Plain cores",
                        "variables": "borehole EIC intervals and drilling locations",
                        "role": "independent EIC summary and leave-one-borehole-out core validation target",
                        "citation": "[@stephani2025usgs_eic]",
                        "local_output": "data/processed/usgs_eic_observations.npz",
                    },
                    {
                        "dataset": "ArcticData upper-permafrost cryostratigraphy",
                        "variables": "cryofacies, sample EIC, ground-ice classes",
                        "role": "multi-site cryofacies, EIC and wedge-ice recall holdout validation target",
                        "citation": "[@kanevskiy2024upper_permafrost]",
                        "local_output": "data/processed/arcticdata_cryostratigraphy_observations.npz",
                    },
                    {
                        "dataset": "ArcticData Jago River 2018 ground ice",
                        "variables": "sample excess-ice content, moisture content, dry density",
                        "role": "third independent ground-ice/EIC validation source",
                        "citation": "[@kanevskiy2020jago_ground_ice]",
                        "local_output": "data/processed/arcticdata_jago_ground_ice_observations.npz",
                    },
                    {
                        "dataset": "CALM / GTN-P / ESA CCI / ArcticDEM / HLS / SoilGrids / ERA5-Land",
                        "variables": "ALT, ground temperature, permafrost extent, terrain, surface, soil, climate",
                        "role": "standardized loaders and future regional priors",
                        "citation": "[@brown1998calm; @gtnp2021magt; @esa2025permafrost; @porter2022arcticdem; @claverie2018hls; @poggio2021soilgrids; @munozsabater2021era5land]",
                        "local_output": "data/external/DOWNLOAD_INSTRUCTIONS.md",
                    },
                ]
            )
        ),
        "",
        "Local public-data provenance and processed-token inventory:",
        "",
        df_to_markdown(
            public_provenance,
            columns=[
                "source_key",
                "status",
                "raw_file_count",
                "raw_total_mb",
                "downloaded_file_count",
                "skipped_large_file_count",
                "n_observations",
                "n_tokens",
                "source_url",
            ],
            max_rows=12,
        ),
        "",
        df_to_markdown(
            public_tokens,
            columns=["source_key", "observation_type", "n_tokens"],
            max_rows=12,
        ),
        "",
        _figure("../outputs/figures/public_data_token_inventory.png", "Figure S0. Public-data token inventory used by the current COLD-Recon validation."),
        "",
        "USGS EIC summary:",
        "",
        df_to_markdown(usgs_eic),
        "",
        "USGS EIC leave-one-borehole-out validation:",
        "",
        df_to_markdown(
            usgs_eic_holdout,
            columns=[
                "model",
                "n",
                "n_boreholes",
                "mae",
                "rmse",
                "pearson_r",
                "high_eic_accuracy",
                "high_eic_recall",
                "rmse_reduction_vs_global_mean",
            ],
        ),
        "",
        "USGS EIC-conditioned latent diffusion validation:",
        "",
        df_to_markdown(
            usgs_eic_diffusion,
            columns=[
                "train_n",
                "holdout_n",
                "train_eic_rmse",
                "holdout_eic_rmse",
                "holdout_eic_high_eic_accuracy",
                "holdout_eic_high_eic_recall",
            ],
        ),
        "",
        "USGS geophysics token summary:",
        "",
        df_to_markdown(usgs_geo),
        "",
        "Jago River 2018 ground-ice token summary:",
        "",
        df_to_markdown(jago_summary),
        "",
        "## 3. Method",
        "",
        _figure(
            "../outputs/figures/nature_figure_1_overview.png",
            "Figure 1. COLD-Recon converts sparse permafrost observations into verified 3D posteriors. Panel a summarizes the observation-token, conditional-posterior, and physics-projection workflow; panel b compares synthetic facies IoU across baselines and generative variants; panel c shows raw and post-hoc calibrated 90% posterior coverage; panel d reports unfrozen-water consistency errors.",
        ),
        "",
        _figure("../outputs/figures/cold_recon_algorithm_schematic.png", "Figure S0a. COLD-Recon algorithm schematic from sparse observations to posterior 3D cryostratigraphy."),
        "",
        "Checkpoint-derived architecture summary:",
        "",
        df_to_markdown(
            architecture,
            columns=[
                "model",
                "role",
                "component_params",
                "obs_encoder_params",
                "total_params",
                "latent_shape",
                "n_facies",
                "notes",
            ],
        ),
        "",
        "Computational footprint audit:",
        "",
        df_to_markdown(
            computational_footprint,
            columns=[
                "model",
                "role",
                "total_params_m",
                "checkpoint_mb",
                "prediction_mb",
                "posterior_samples",
                "training_epochs",
                "mean_iou",
                "eic_rmse",
            ],
            max_rows=14,
        ),
        "",
        (
            "This audit reports model complexity and artifact footprint alongside performance. The compact latent diffusion "
            f"posterior uses {metric_text(footprint_latent, 'total_params_m')} million trainable parameters, whereas the FNO-Transformer "
            f"operator variant uses {metric_text(footprint_fno, 'total_params_m')} million parameters. The physics-trained posterior "
            f"uses the same compact latent-diffusion architecture with prediction footprint {metric_text(footprint_trained, 'prediction_mb')} MB, "
            f"and the rare-facies hybrid is reported as a post-hoc operating point with prediction footprint {metric_text(footprint_hybrid, 'prediction_mb')} MB. "
            "The footprint table therefore prevents accuracy comparisons from treating compact and high-parameter posterior variants as cost-equivalent."
        ),
        "",
        _figure(
            "../outputs/figures/computational_footprint_summary.png",
            "Figure S0b. Computational footprint audit showing mean-IoU versus parameter count, checkpoint and prediction artifact sizes, posterior sample counts and training-history metadata.",
        ),
        "",
        "### 3.1 Observation Tokenization",
        "",
        (
            "Each observation is represented as o_i=(x_i,y_i,z_i,t_i,type_i,value_i,sigma_i,mask_i). "
            "The ObservationTokenizer normalizes coordinates and values on the target grid and emits fixed-width tokens. "
            "An ObservationGraphBuilder additionally constructs a normalized spatial k-nearest-neighbor graph and an optional local "
            "Transformer attention mask, so borehole, ERT, NMR, and ALT tokens can be aggregated through neighborhood-constrained "
            "attention rather than only through unconstrained global token mixing. An ObsTransformerEncoder aggregates the sparse tokens "
            "into a conditioning vector for implicit and diffusion models, "
            "following the Transformer attention pattern [@vaswani2017attention]."
        ),
        "",
        "### 3.2 Synthetic Cryostratigraphy and Forward Observations",
        "",
        (
            "Synthetic volumes include active layers, peat, mineral silt, ice-rich silt, sand/gravel, taliks, and wedge ice. "
            "Forward fields include EIC, temperature, unfrozen water, resistivity, thermal conductivity, and heat capacity. "
            "Sparse observations emulate borehole intervals, ERT profiles, NMR points, and active-layer measurements. "
            "Diagnostic 3D overviews and borehole profile panels are generated to inspect volumetric cryostratigraphy and hard-data consistency."
        ),
        "",
        "Synthetic ensemble distribution summary:",
        "",
        df_to_markdown(synthetic_summary_view, columns=["metric", "mean", "std", "min", "max", "n"]),
        "",
        (
            f"The benchmark uses {len(synthetic_benchmark) if not synthetic_benchmark.empty else 'not available'} fixed-seed synthetic volumes "
            "to document data-generator variability before model fitting. This separates generator-distribution evidence from the "
            "single-volume reconstruction experiment used for checkpoint training."
        ),
        "",
        _figure("../outputs/figures/synthetic_ensemble_benchmark.png", "Figure S1. Fixed-seed synthetic ensemble distribution benchmark."),
        "",
        _figure("../outputs/figures/synthetic_ensemble_facies_fractions.png", "Figure S2. Facies volume fractions across fixed-seed synthetic benchmark samples."),
        "",
        _figure("../outputs/figures/volume_truth_3d_overview.png", "Figure S2d. Synthetic truth 3D overview of ice-rich voxels, cryofacies, and near-thaw thermal structure."),
        "",
        "### 3.3 Conditional Reconstruction Models",
        "",
        (
            "The deterministic COLD-Recon implicit model maps query coordinates, surface features, and observation-token context "
            "to facies logits and continuous permafrost properties. A histogram Gradient Boosting baseline provides an XGBoost/NGBoost-style "
            "tree-ensemble comparator without heavy optional dependencies. A fixed-kernel Kriging/GPR baseline provides ordinary and indicator-kriging-style "
            "geostatistical interpolation without external geostatistics dependencies. A sparse-observation 3D U-Net baseline rasterizes observation "
            "values and masks with surface covariates into a voxel conditioning tensor and predicts the same facies/EIC/thermal/geophysical fields. "
            "The probabilistic model first trains a 3D autoencoder for "
            "multi-channel permafrost fields and then learns a conditional denoiser in latent space [@ho2020ddpm; @rombach2022latent]. "
            "The neural-operator variant replaces the U-Net denoiser with an FNO-Transformer hybrid that applies low-mode 3D spectral "
            "convolutions, FiLM conditioning from sparse observation tokens and diffusion time, and a compact Transformer over pooled "
            "latent tokens. A companion rectified-flow objective learns a conditional velocity field between warm latent-noise sources "
            "and the encoded cryostratigraphic target, giving a flow-matching posterior sampler with the same observation-token conditioning. "
            "The coordinate MLP uses Fourier feature mappings to reduce spectral bias [@tancik2020fourier]. Posterior sampling produces "
            "facies probabilities, EIC/temperature/unfrozen-water/resistivity ensembles, entropy, and ice-rich exceedance probability."
        ),
        "",
        "### 3.4 Physics Guidance",
        "",
        (
            "Physics modules implement unfrozen-water consistency, simplified heat residuals, and empirical resistivity coupling. "
            "Physics-guided denoiser fine-tuning adds these differentiable losses to the diffusion "
            "noise-prediction objective through an autoencoder-decoded x0 estimate. A latent-space physics guidance operator "
            "then backpropagates the same losses through the autoencoder decoder during sampling with facies/eic anchors, while "
            "a post-hoc physics-refinement operator projects diffusion posterior samples toward the implemented unfrozen-water, "
            "resistivity, and smoothed heat-consistency relations while preserving the sampled facies ensemble. For public field data, "
            "a field-scale physical proxy is fused with latent diffusion outputs using hard temperature guidance and high-weight "
            "proxy conditioning to preserve observed active-layer structure."
        ),
        "",
        "## 4. Experiments",
        "",
        "### 4.1 Synthetic 3D Reconstruction",
        "",
        df_to_markdown(
            model,
            columns=[
                "model",
                "mean_iou",
                "eic_rmse",
                "temperature_rmse",
                "unfrozen_water_rmse",
                "log_resistivity_rmse",
                "alt_mae",
                "ice_rich_recall",
            ],
        ),
        "",
        (
            f"Compared with the Random Forest baseline [@breiman2001randomforest] mean IoU {metric_text(rf_row, 'mean_iou')}, "
            f"the Gradient Boosting baseline mean IoU {metric_text(gb_row, 'mean_iou')}, "
            f"the Kriging/GPR baseline mean IoU {metric_text(kriging_row, 'mean_iou')}, "
            f"and the sparse-observation 3D U-Net baseline mean IoU {metric_text(unet_row, 'mean_iou')}, the implicit model obtains "
            f"{metric_text(implicit_row, 'mean_iou')} and the latent diffusion posterior obtains "
            f"{metric_text(diff_row, 'mean_iou')}. The FNO-Transformer operator denoiser obtains mean IoU "
            f"{metric_text(fno_row, 'mean_iou')} with EIC RMSE {metric_text(fno_row, 'eic_rmse')}, providing a runnable neural-operator "
            "diffusion variant with comparable synthetic reconstruction quality. The warm-start rectified-flow posterior obtains mean IoU "
            f"{metric_text(flow_row, 'mean_iou')} with EIC RMSE {metric_text(flow_row, 'eic_rmse')}. Physics-guided denoiser fine-tuning improves mean IoU to "
            f"{metric_text(trained_row, 'mean_iou')} with temperature RMSE {metric_text(trained_row, 'temperature_rmse')} "
            f"and unfrozen-water RMSE {metric_text(trained_row, 'unfrozen_water_rmse')}. Latent-space physics guidance obtains mean IoU "
            f"{metric_text(guided_row, 'mean_iou')} while reducing temperature RMSE to "
            f"{metric_text(guided_row, 'temperature_rmse')} and log-resistivity RMSE to "
            f"{metric_text(guided_row, 'log_resistivity_rmse')}. Physics refinement preserves the diffusion facies IoU "
            f"({metric_text(refined_row, 'mean_iou')}) while reducing temperature RMSE from "
            f"{metric_text(diff_row, 'temperature_rmse')} to {metric_text(refined_row, 'temperature_rmse')}, "
            f"unfrozen-water RMSE from {metric_text(diff_row, 'unfrozen_water_rmse')} to "
            f"{metric_text(refined_row, 'unfrozen_water_rmse')}, and log-resistivity RMSE from "
            f"{metric_text(diff_row, 'log_resistivity_rmse')} to {metric_text(refined_row, 'log_resistivity_rmse')}."
        ),
        "",
        _figure("../outputs/figures/diffusion_posterior_sections.png", "Figure S3e. Latent diffusion posterior mean and facies mode on the synthetic validation volume."),
        "",
        _figure("../outputs/figures/volume_reconstruction_3d_overview.png", "Figure S2e. Physics-refined posterior 3D overview of ice-rich, facies, and near-thaw structure."),
        "",
        _figure("../outputs/figures/borehole_profile_comparison.png", "Figure S2f. Borehole profiles comparing synthetic truth, sparse observations, and posterior reconstruction."),
        "",
        _figure("../outputs/figures/baseline_gradient_boosting_sections.png", "Figure S2a. Histogram Gradient Boosting baseline sections."),
        "",
        _figure("../outputs/figures/baseline_kriging_sections.png", "Figure S2b. Fixed-kernel Kriging/GPR geostatistical baseline sections."),
        "",
        _figure("../outputs/figures/baseline_unet3d_sections.png", "Figure S2c. Sparse-observation 3D U-Net deterministic baseline sections."),
        "",
        _figure("../outputs/figures/fno_operator_diffusion_sections.png", "Figure S3f. FNO-Transformer operator diffusion posterior sections on the synthetic validation volume."),
        "",
        _figure("../outputs/figures/fno_operator_diffusion_training_history.png", "Figure S3a. FNO-Transformer operator diffusion training curve."),
        "",
        _figure("../outputs/figures/rectified_flow_sections.png", "Figure S3b. Warm-start rectified-flow posterior sections on the synthetic validation volume."),
        "",
        _figure("../outputs/figures/rectified_flow_training_history.png", "Figure S3c. Rectified-flow training curve."),
        "",
        "Synthetic rare-cryostructure operating-point audit:",
        "",
        df_to_markdown(
            rare_view,
            columns=[
                "model",
                "raw_eic_recall",
                "raw_eic_f1",
                "rate_constrained_eic_threshold",
                "rate_constrained_eic_recall",
                "rate_constrained_eic_f1",
                "rare_facies_recall",
                "facies_3_ice_rich_silt_recall",
                "facies_6_wedge_ice_recall",
            ],
            max_rows=8,
        ),
        "",
        (
            "This audit separates high-EIC event screening from full rare-facies reconstruction. For the physics-trained diffusion "
            f"posterior, the fixed EIC > 0.30 rule gives high-EIC recall {metric_text(rare_trained_row, 'raw_eic_recall')} "
            f"and F1 {metric_text(rare_trained_row, 'raw_eic_f1')}; an observation-rate-constrained threshold of "
            f"{metric_text(rare_trained_row, 'rate_constrained_eic_threshold')} raises recall to "
            f"{metric_text(rare_trained_row, 'rate_constrained_eic_recall')} and F1 to "
            f"{metric_text(rare_trained_row, 'rate_constrained_eic_f1')}. The same row reports rare-facies recall "
            f"{metric_text(rare_trained_row, 'rare_facies_recall')}, ice-rich-silt recall "
            f"{metric_text(rare_trained_row, 'facies_3_ice_rich_silt_recall')} and wedge-ice recall "
            f"{metric_text(rare_trained_row, 'facies_6_wedge_ice_recall')}. The implicit field retains wedge-ice recall "
            f"{metric_text(rare_implicit_row, 'facies_6_wedge_ice_recall')}, whereas the diffusion variants currently expose "
            "wedge facies as a boundary condition rather than a solved reconstruction target."
        ),
        "",
        _figure(
            "../outputs/figures/synthetic_rare_cryostructure_audit.png",
            "Figure S3g. Synthetic rare cryostructure audit comparing fixed and observation-rate-constrained high-EIC event operating points, plus rare facies and wedge-ice recall.",
        ),
        "",
        "Rare-facies hybrid operating point:",
        "",
        df_to_markdown(
            rare_hybrid_metrics,
            columns=[
                "model",
                "mean_iou",
                "wedge_ice_recall",
                "wedge_ice_precision",
                "wedge_ice_f1",
                "wedge_ice_predicted_rate",
                "eic_floor",
                "gate_probability",
            ],
            max_rows=8,
        ),
        "",
        df_to_markdown(
            rare_hybrid_curve,
            columns=[
                "eic_floor",
                "gate_fraction",
                "mean_iou",
                "mean_iou_delta_vs_base",
                "wedge_ice_recall",
                "wedge_ice_precision",
                "wedge_ice_f1",
                "wedge_ice_predicted_rate",
            ],
            max_rows=12,
        ),
        "",
        (
            "To convert the synthetic wedge-facies miss into an explicit operating point, the rare-facies hybrid accepts implicit "
            "wedge proposals only where the physics-trained diffusion EIC posterior exceeds a specified floor. At the default "
            f"EIC floor 0.10, the hybrid reports mean IoU {metric_text(rare_hybrid_row, 'mean_iou')}, wedge recall "
            f"{metric_text(rare_hybrid_row, 'wedge_ice_recall')} and wedge precision "
            f"{metric_text(rare_hybrid_row, 'wedge_ice_precision')}. The same threshold-sweep row gates "
            f"{metric_text(rare_hybrid_default, 'gate_fraction')} of voxels and changes mean IoU by "
            f"{metric_text(rare_hybrid_default, 'mean_iou_delta_vs_base')} relative to the physics-trained posterior. This is "
            "reported as a selectable rare-facies constraint, not as a hidden replacement for the diffusion posterior."
        ),
        "",
        _figure(
            "../outputs/figures/rare_facies_hybrid_operating_curve.png",
            "Figure S3h. Rare-facies hybrid operating curve showing wedge recall, precision, F1, gated voxel fraction and mean-IoU change as the EIC acceptance floor varies.",
        ),
        "",
        "Sparse multi-source observation consistency:",
        "",
        df_to_markdown(
            obs_consistency_view,
            columns=["model", "source", "n", "rmse", "normalized_rmse", "accuracy"],
            max_rows=80,
        ),
        "",
        (
            "This diagnostic evaluates agreement at the exact sparse observation locations. IDW is expected to be near-exact "
            "for continuous observations because it directly interpolates the same observations; diffusion models are evaluated "
            "as learned conditional generators rather than hard interpolants."
        ),
        "",
        _figure("../outputs/figures/synthetic_observation_consistency.png", "Figure S3. Synthetic multi-source sparse-observation consistency by model and source."),
        "",
        "### 4.1b Observation Graph Ablation",
        "",
        (
            "To verify the sparse-observation graph pathway independently from the larger generative models, we run a short controlled "
            "implicit-model ablation with identical initialization and training budget. The comparison is intended as an implementation "
            "and sensitivity diagnostic, not as a final performance ranking; it checks whether replacing unconstrained global token "
            "mixing with kNN neighborhood-constrained attention is reproducible and quantitatively traceable."
        ),
        "",
        df_to_markdown(
            graph_ablation,
            columns=[
                "scenario",
                "graph_enabled",
                "epochs",
                "k_neighbors",
                "mean_iou",
                "eic_rmse",
                "temperature_rmse",
                "unfrozen_water_rmse",
                "ice_rich_recall",
            ],
        ),
        "",
        _figure("../outputs/figures/observation_graph_ablation.png", "Figure S3d. Short-run observation graph ablation comparing global token attention and kNN graph-constrained attention."),
        "",
        "### 4.2 Sparsity and Observation-Source Ablation",
        "",
        df_to_markdown(
            ablation,
            columns=["scenario", "n_boreholes", "n_observations", "mean_iou", "eic_rmse", "temperature_rmse", "alt_mae"],
            max_rows=18,
        ),
        "",
        _figure("../outputs/figures/ablation_sparsity_curves.png", "Figure S4. Synthetic borehole sparsity and observation-source ablation."),
        "",
        "### 4.3 Posterior Calibration",
        "",
        "Posterior calibration is evaluated with empirical central-interval coverage, CRPS, and probability scores [@gneiting2007proper].",
        "",
        df_to_markdown(
            calib,
            columns=["target", "kind", "rmse", "crps", "coverage_90", "width_90", "brier", "nll", "accuracy"],
        ),
        "",
        (
            f"EIC CRPS is {metric_text(eic_row, 'crps')} with 90% coverage {metric_text(eic_row, 'coverage_90')}; "
            f"active-layer thickness CRPS is {metric_text(alt_row, 'crps')} with 90% coverage {metric_text(alt_row, 'coverage_90')}. "
            f"Ice-rich probability has Brier score {metric_text(ice_row, 'brier')}, and facies probability has NLL "
            f"{metric_text(facies_row, 'nll')}. The very low coverage for some continuous thermal/geophysical fields indicates "
            "that posterior spread is still under-calibrated and should be improved in the next model revision."
        ),
        "",
        _figure("../outputs/figures/uncertainty_reliability.png", "Figure S5a. Posterior reliability curves on synthetic full-field truth."),
        "",
        "Post-hoc interval calibration factors:",
        "",
        df_to_markdown(spread_factors),
        "",
        "Post-hoc calibrated continuous-field reliability:",
        "",
        df_to_markdown(
            calib_scaled,
            columns=["target", "calibration", "rmse", "crps", "coverage_90", "width_90", "mean_std"],
        ),
        "",
        (
            f"After post-hoc interval calibration, EIC 90% coverage changes to {metric_text(eic_scaled_row, 'coverage_90')} "
            f"with CRPS {metric_text(eic_scaled_row, 'crps')}, temperature coverage changes to "
            f"{metric_text(temp_scaled_row, 'coverage_90')}, unfrozen-water coverage changes to "
            f"{metric_text(unfrozen_scaled_row, 'coverage_90')}, and log-resistivity coverage changes to "
            f"{metric_text(log_rho_scaled_row, 'coverage_90')}. The unfrozen-water row requires bias-quantile calibration, so "
            "coverage repair is treated as a posterior diagnostic rather than proof that the mean water-content field is physically solved."
        ),
        "",
        _figure("../outputs/figures/uncertainty_reliability_calibrated.png", "Figure S5b. Post-hoc calibrated posterior reliability curves."),
        "",
        _figure("../outputs/figures/posterior_spread_scale_factors.png", "Figure S5c. Field-wise interval calibration factors required for 90% target coverage."),
        "",
        "### 4.3b Posterior Uncertainty-Error Alignment",
        "",
        (
            "Interval coverage alone does not show whether posterior uncertainty usefully localizes reconstruction errors. We therefore "
            "audit per-voxel posterior standard deviation, or facies entropy for categorical output, against synthetic full-field "
            "absolute error and misclassification indicators."
        ),
        "",
        df_to_markdown(
            uncertainty_alignment_view,
            columns=[
                "model",
                "target",
                "spearman_uncertainty_error",
                "top_uncertainty_error_enrichment",
                "bottom_uncertainty_error_ratio",
                "top_uncertainty_captures_top_error_rate",
            ],
            max_rows=18,
        ),
        "",
        (
            "The strongest alignment is target-specific. For the physics-trained diffusion posterior, EIC uncertainty has "
            f"Spearman rank correlation {metric_text(align_trained_eic, 'spearman_uncertainty_error')} with absolute EIC error, "
            f"and the top uncertainty decile has {metric_text(align_trained_eic, 'top_uncertainty_error_enrichment')} times the "
            "global EIC error. The physics-refined posterior preserves similar EIC localization "
            f"({metric_text(align_refined_eic, 'top_uncertainty_error_enrichment')} enrichment) and improves unfrozen-water alignment "
            f"to Spearman {metric_text(align_refined_water, 'spearman_uncertainty_error')} with enrichment "
            f"{metric_text(align_refined_water, 'top_uncertainty_error_enrichment')}. Facies entropy is weaker, with physics-trained "
            f"top-entropy error enrichment {metric_text(align_trained_facies, 'top_uncertainty_error_enrichment')}. These values support "
            "uncertainty as an auditable reconstruction diagnostic while preventing an overbroad reliability claim."
        ),
        "",
        _figure(
            "../outputs/figures/posterior_uncertainty_alignment.png",
            "Figure S5e. Posterior uncertainty-error alignment audit showing EIC error enrichment in high-uncertainty voxels, rank correlations for continuous fields, and the weaker facies-entropy boundary.",
        ),
        "",
        "### 4.4 Physics Consistency Diagnostics",
        "",
        df_to_markdown(
            physics,
            columns=[
                "model",
                "domain",
                "unfrozen_water_empirical_mae",
                "log_resistivity_empirical_mae",
                "heat_residual_rmse",
                "stratigraphic_tv_xy",
            ],
        ),
        "",
        (
            "These diagnostics evaluate whether generated fields obey the implemented frozen-ground consistency relations rather "
            "than only matching held-out observations. The implicit model obtains unfrozen-water empirical MAE "
            f"{metric_text(implicit_phys_row, 'unfrozen_water_empirical_mae')} and heat residual RMSE "
            f"{metric_text(implicit_phys_row, 'heat_residual_rmse')}, while the latent diffusion posterior obtains "
            f"{metric_text(diffusion_phys_row, 'unfrozen_water_empirical_mae')} and "
            f"{metric_text(diffusion_phys_row, 'heat_residual_rmse')}. The FNO-Transformer operator posterior obtains "
            f"{metric_text(fno_phys_row, 'unfrozen_water_empirical_mae')} and "
            f"{metric_text(fno_phys_row, 'heat_residual_rmse')}. The rectified-flow posterior obtains "
            f"{metric_text(flow_phys_row, 'unfrozen_water_empirical_mae')} and "
            f"{metric_text(flow_phys_row, 'heat_residual_rmse')}. Physics-guided fine-tuning changes these diagnostics to "
            f"{metric_text(trained_phys_row, 'unfrozen_water_empirical_mae')} and "
            f"{metric_text(trained_phys_row, 'heat_residual_rmse')}; latent-space physics guidance changes them to "
            f"{metric_text(guided_phys_row, 'unfrozen_water_empirical_mae')} and "
            f"{metric_text(guided_phys_row, 'heat_residual_rmse')}; post-hoc physics refinement reduces them to "
            f"{metric_text(refined_phys_row, 'unfrozen_water_empirical_mae')} and "
            f"{metric_text(refined_phys_row, 'heat_residual_rmse')}, respectively. The real-token USGS posterior has heat residual RMSE "
            f"{metric_text(usgs_phys_row, 'heat_residual_rmse')}. The EIC-core conditioned posterior has heat residual RMSE "
            f"{metric_text(usgs_eic_phys_row, 'heat_residual_rmse')}, providing field-scale physical consistency checks."
        ),
        "",
        _figure("../outputs/figures/physics_consistency_summary.png", "Figure S5d. Physics consistency diagnostics across synthetic and real-conditioned reconstructions."),
        "",
        "### 4.5 Public USGS Field Validation",
        "",
        (
            "USGS Arctic Coastal Plain EIC core intervals are first evaluated with leave-one-borehole-out prediction. "
            "The public release masks geographic coordinates as proprietary, so this validation uses a reproducible ordered "
            "borehole index plus depth coordinate rather than claiming surveyed spatial accuracy."
        ),
        "",
        df_to_markdown(
            usgs_eic_holdout,
            columns=[
                "model",
                "n",
                "mae",
                "rmse",
                "normalized_rmse",
                "pearson_r",
                "high_eic_accuracy",
                "high_eic_recall",
            ],
        ),
        "",
        (
            f"Spatial-depth IDW reduces EIC MAE from {metric_text(eic_global_row, 'mae')} for the global-mean baseline "
            f"to {metric_text(eic_spatial_row, 'mae')}, while its RMSE is {metric_text(eic_spatial_row, 'rmse')} "
            f"versus {metric_text(eic_global_row, 'rmse')} for the global mean. This pattern indicates that depth and "
            "ordered-neighbor information improves typical interval error but still misses some high-contrast ice-rich intervals."
        ),
        "",
        _figure("../outputs/figures/usgs_eic_holdout_validation.png", "Figure S4. USGS EIC core leave-one-borehole-out validation."),
        "",
        "USGS EIC-conditioned latent diffusion validation:",
        "",
        df_to_markdown(
            usgs_eic_diffusion,
            columns=[
                "train_n",
                "condition_n",
                "holdout_n",
                "train_eic_rmse",
                "holdout_eic_rmse",
                "holdout_eic_normalized_rmse",
                "holdout_eic_high_eic_accuracy",
                "holdout_eic_high_eic_recall",
            ],
        ),
        "",
        (
            f"The EIC-conditioned diffusion posterior uses {metric_text(eic_diffusion_row, 'condition_n')} training EIC tokens "
            f"from {metric_text(eic_diffusion_row, 'train_boreholes')} boreholes and evaluates on "
            f"{metric_text(eic_diffusion_row, 'holdout_n')} EIC intervals from "
            f"{metric_text(eic_diffusion_row, 'holdout_boreholes')} held-out boreholes. Hold-out EIC RMSE is "
            f"{metric_text(eic_diffusion_row, 'holdout_eic_rmse')}, compared with training-token RMSE "
            f"{metric_text(eic_diffusion_row, 'train_eic_rmse')}."
        ),
        "",
        _figure("../outputs/figures/usgs_eic_conditioned_diffusion_sections.png", "Figure S5. USGS EIC-conditioned diffusion posterior sections."),
        "",
        "Field-scale hold-out validation from the interpolation/proxy reconstruction:",
        "",
        df_to_markdown(usgs_holdout),
        "",
        "Real-token conditioned latent diffusion validation:",
        "",
        df_to_markdown(usgs_real),
        "",
        _figure("../outputs/figures/usgs_real_conditioned_diffusion_sections.png", "Figure S5e. Real-token conditioned USGS diffusion posterior sections."),
        "",
        "### 4.6 Cross-source ArcticData and Jago River validation",
        "",
        (
            "The Arctic Data Center cryostratigraphy package supplies multi-site cryofacies, EIC and wedge-ice recall tests, while the "
            "Jago River 2018 package supplies a smaller but independent ground-ice table. The Jago package contains "
            f"{metric_text(jago_summary.iloc[0] if not jago_summary.empty else None, 'n_eic_tokens')} EIC observation tokens from "
            f"{metric_text(jago_summary.iloc[0] if not jago_summary.empty else None, 'n_boreholes')} ordered boreholes. "
            "Because the public table does not provide surveyed borehole coordinates in the CSV, this validation uses the same "
            "ordered-borehole coordinate convention as the coordinate-masked USGS EIC workflow."
        ),
        "",
        _figure(
            "../outputs/figures/nature_figure_2_real_data_gate.png",
            "Figure 2. Real-data evidence gate across three public permafrost data sources. Panel a summarizes processed public observation tokens; panel b reports relative improvement for passed validation tasks; panel c compares EIC RMSE across ArcticData, USGS core, and Jago River holdouts; panel d gives the pass/fail evidence matrix, including calibrated Jago high-EIC screening and recall-first wedge-ice handling.",
        ),
        "",
        "Jago River EIC leave-one-borehole-out simple baselines:",
        "",
        df_to_markdown(
            jago_holdout,
            columns=[
                "model",
                "n",
                "n_boreholes",
                "mae",
                "rmse",
                "pearson_r",
                "high_eic_accuracy",
                "high_eic_recall",
                "high_eic_f1",
            ],
        ),
        "",
        _figure("../outputs/figures/arcticdata_jago_ground_ice_holdout_validation.png", "Figure S6. Jago River 2018 ground-ice measurements and leave-one-borehole-out EIC baselines."),
        "",
        "Jago River same-split EIC-conditioned diffusion comparison:",
        "",
        df_to_markdown(
            jago_comparison,
            columns=[
                "model",
                "eic_n",
                "eic_mae",
                "eic_rmse",
                "eic_pearson_r",
                "high_eic_accuracy",
                "high_eic_recall",
                "high_eic_f1",
                "high_eic_f2",
                "high_eic_prediction_threshold",
                "high_eic_recall_fixed_0p30",
                "eic_rmse_reduction_vs_best_simple",
            ],
        ),
        "",
        (
            f"With conservative EIC guidance, the Jago-conditioned posterior obtains hold-out EIC RMSE "
            f"{metric_text(jago_model_row, 'eic_rmse')} compared with {metric_text(jago_global_row, 'eic_rmse')} for the best simple baseline in the same split. "
            f"The relative RMSE reduction is {metric_text(jago_model_row, 'eic_rmse_reduction_vs_best_simple')}. "
            f"A train-split F2-calibrated screening threshold of {metric_text(jago_model_row, 'high_eic_prediction_threshold')} raises hold-out high-EIC F1 to "
            f"{metric_text(jago_model_row, 'high_eic_f1')} with recall {metric_text(jago_model_row, 'high_eic_recall')}, compared with "
            f"{metric_text(jago_spatial_row, 'high_eic_f1')} for SpatialDepthIDW. The fixed 0.30 threshold has recall "
            f"{metric_text(jago_model_row, 'high_eic_recall_fixed_0p30')}, so the claim is a calibrated, recall-oriented screening result on a small split rather than robust regional event detection."
        ),
        "",
        _figure(
            "../outputs/figures/nature_figure_3_cited_ground_ice.png",
            "Figure 3. Cited ground-ice records support cross-source EIC validation. Panel a compares EIC distributions from ArcticData, USGS core, and Jago River records; panel b shows measured Jago ground-ice observations; panel c compares Jago observed and predicted EIC for simple baselines and COLD-Recon; panel d compares USGS core EIC predictions.",
        ),
        "",
        _figure("../outputs/figures/arcticdata_jago_ground_ice_conditioned_diffusion_sections.png", "Figure S7. Jago River EIC-conditioned posterior sections."),
        "",
        "Cross-source real-data evidence gate:",
        "",
        df_to_markdown(
            real_data_cg,
            columns=[
                "source",
                "task",
                "metric",
                "model",
                "model_value",
                "baseline",
                "baseline_value",
                "relative_improvement",
                "passed",
            ],
        ),
        "",
        (
            f"The cross-source gate passes {cg_passed_sources} independent public sources and {cg_passed_tasks}/{cg_total_tasks} tasks. "
            "The passed tasks include ArcticData cryofacies, ArcticData EIC regression, ArcticData wedge-ice recall, ArcticData high-EIC F1, "
            "USGS EIC regression, USGS high-EIC F1, Jago EIC regression and Jago high-EIC screening. The gate remains task-specific: the wedge head is recall-first, and the Jago event result depends on a training-split calibrated threshold with fixed-threshold audit columns retained in the comparison table."
        ),
        "",
        "External multi-site generalization audit:",
        "",
        df_to_markdown(
            external_generalization,
            columns=[
                "task",
                "metric",
                "model_value",
                "baseline",
                "baseline_value",
                "site_win_rate",
                "site_noninferior_rate",
                "failure_sites",
            ],
        ),
        "",
        df_to_markdown(
            external_site_deltas,
            columns=[
                "site",
                "facies_delta",
                "eic_rmse_reduction_vs_best_simple",
                "eic_best_simple_baseline",
                "adaptive_eic_method",
                "adaptive_eic_transfer_guard_reason",
                "wedge_recall_delta",
                "wedge_precision_delta",
            ],
        ),
        "",
        (
            "The external-generalization audit uses the same ArcticData grouped-borehole holdouts but reports site-level deltas rather than only pooled task scores. "
            f"Facies accuracy improves from {metric_text(external_facies, 'baseline_value')} to {metric_text(external_facies, 'model_value')} with site win rate "
            f"{metric_text(external_facies, 'site_win_rate')} and non-inferiority rate {metric_text(external_facies, 'site_noninferior_rate')}. "
            f"EIC regression now has site win rate {metric_text(external_eic, 'site_win_rate')} and non-inferiority rate "
            f"{metric_text(external_eic, 'site_noninferior_rate')} under the stricter per-site best-simple comparator, with failure sites "
            f"reported as {metric_text(external_eic, 'failure_sites', 'none')}. The evidence-gate aggregate SpatialDepthIDW comparison gives relative improvement "
            f"{metric_text(external_eic, 'evidence_gate_relative_improvement')}, and the site-level table exposes where the adaptive hybrid "
            "uses the compact-site spatial guard rather than hiding ties inside the aggregate score. "
            f"High-EIC event F1 increases from {metric_text(external_high_eic, 'baseline_value')} to "
            f"{metric_text(external_high_eic, 'model_value')}, with site win rate {metric_text(external_high_eic, 'site_win_rate')} "
            f"and non-inferiority rate {metric_text(external_high_eic, 'site_noninferior_rate')}. "
            f"Wedge recall increases from {metric_text(external_wedge, 'baseline_value')} to {metric_text(external_wedge, 'model_value')}, while precision decreases from "
            f"{metric_text(external_wedge, 'secondary_baseline_value')} to {metric_text(external_wedge, 'secondary_model_value')}."
        ),
        "",
        _figure(
            "../outputs/figures/external_generalization_audit.png",
            "Figure S7b. Public multi-site ArcticData holdouts expose aggregate transfer gains and site-level boundaries. Panel a shows cryofacies accuracy deltas; panel b shows EIC RMSE reductions against per-site best simple baselines; panel c compares wedge recall and precision; panel d reports task-level site win and non-inferiority rates.",
        ),
        "",
        "Transfer failure attribution audit:",
        "",
        df_to_markdown(
            transfer_failure_diagnostics,
            columns=[
                "site",
                "eic_transfer_outcome",
                "adaptive_eic_method",
                "adaptive_eic_transfer_guard_reason",
                "eic_model_gap_vs_best_simple",
                "spatial_idw_advantage_vs_global",
                "transfer_readiness_score",
                "failure_attribution",
            ],
        ),
        "",
        df_to_markdown(
            transfer_failure_summary,
            columns=[
                "signal",
                "n_sites",
                "spearman",
                "pearson",
                "interpretation",
            ],
            max_rows=12,
        ),
        "",
        (
            "The transfer-failure attribution audit is a diagnostic extension of the external-generalization test. "
            f"The outcome count is {metric_text(transfer_outcome_counts, 'interpretation')}. The failure-attribution count is "
            f"{metric_text(transfer_failure_counts, 'interpretation')}. The compact-site spatial guard selects the transfer_idw_adapter only "
            "where training support is compact and train-split cross-validation admits the local IDW prior. In the current five-site audit, "
            "Itkillik and Tuktoyaktuk move from strict failures to non-inferior ties with zero EIC gap against the per-site best simple "
            "baseline. This is reported as a bounded algorithmic guard for sparse-site reconstruction, not as a separate application study."
        ),
        "",
        _figure(
            "../outputs/figures/transfer_failure_attribution.png",
            "Figure S7c. Transfer-failure attribution audit. Panel a pairs COLD-Recon and per-site best-simple EIC RMSE; panel b relates the COLD-Recon gap to the best simple baseline with the SpatialDepthIDW advantage over the global mean; panel c reports small-n associations with EIC RMSE reduction; panel d shows transfer-readiness components by site.",
        ),
        "",
        "Domain-support applicability audit:",
        "",
        df_to_markdown(
            domain_support,
            columns=[
                "site",
                "support_score",
                "applicability_class",
                "facies_outcome",
                "eic_outcome",
                "high_eic_outcome",
                "wedge_outcome",
                "recommended_action",
            ],
        ),
        "",
        (
            "The domain-support audit turns the public multi-site transfer result into an explicit applicability rule. "
            f"In the current five-site audit, {domain_model_supported} sites are model-supported transfers, {domain_guarded} compact sites "
            f"are guarded local-prior cases, and {domain_low} sites are low support. "
            f"{'All evaluated site-task outcomes are non-inferior or better.' if domain_all_noninferior else 'At least one evaluated site-task remains below the non-inferiority boundary.'} "
            "The support score uses train-side observation, group and borehole support, while the outcome columns are kept separate so the audit does not infer applicability from holdout performance alone."
        ),
        "",
        _figure(
            "../outputs/figures/domain_support_audit.png",
            "Figure S7d. Domain-support applicability audit. Panel a reports train-side support components; panel b relates support score to EIC RMSE reduction; panel c reports facies, EIC, high-EIC and wedge task outcomes by site.",
        ),
        "",
        "Wedge-ice recall operating-curve audit:",
        "",
        df_to_markdown(
            wedge_operating_points,
            columns=[
                "operating_point",
                "model",
                "threshold",
                "recall",
                "precision",
                "false_positive_rate",
                "f1",
                "mean_site_recall",
                "mean_site_precision",
                "mean_site_false_positive_rate",
            ],
            max_rows=8,
        ),
        "",
        (
            "The wedge task is deliberately evaluated as an operating-point problem rather than a single universal classifier. "
            f"The current site-calibrated recall-first head obtains pooled recall {metric_text(wedge_recall_point, 'recall')} "
            f"against {metric_text(wedge_knn_point, 'recall')} for SpatialDepthKNN, but its pooled false-positive rate is "
            f"{metric_text(wedge_recall_point, 'false_positive_rate')}. A pooled max-F1 posterior-probability threshold of "
            f"{metric_text(wedge_max_f1_point, 'threshold')} reduces the pooled false-positive rate to "
            f"{metric_text(wedge_max_f1_point, 'false_positive_rate')} with precision "
            f"{metric_text(wedge_max_f1_point, 'precision')} and recall {metric_text(wedge_max_f1_point, 'recall')}. "
            "This supporting audit does not replace the recall-first gate; it documents the false-positive cost that a field user would tune for a specific campaign."
        ),
        "",
        _figure(
            "../outputs/figures/arcticdata_wedge_operating_curve.png",
            "Figure S8. ArcticData wedge-ice recall operating curve. Panel a shows the posterior wedge-probability recall-precision curve across thresholds; panel b compares recall, precision and false-positive rate for the KNN baseline, recall-first constraint head and probability-threshold operating points.",
        ),
        "",
        "### 4.7 Posterior value-of-information as a reconstruction diagnostic",
        "",
        (
            "The reconstruction posterior is also converted into a reproducible value-of-information (VOI) diagnostic for the next "
            "observation candidates. The score combines posterior uncertainty, ice-rich ambiguity, thaw-sensitive EIC structure "
            "and distance from existing observations, while excluding cells within the configured near-observation radius. This module "
            "is reported as an auditable posterior diagnostic, not as a field-verified proof that the selected boreholes or ERT lines "
            "are globally optimal."
        ),
        "",
        _figure(
            "../outputs/figures/nature_figure_4_site_investigation.png",
            "Figure 4. Posterior value-of-information converts probabilistic reconstruction uncertainty into supplemental observation targets. Panel a maps the VOI surface with recommended boreholes and ERT lines; panel b ranks the borehole targets; panel c decomposes the weighted VOI score for the highest-ranked boreholes; panel d ranks ERT survey-line candidates.",
        ),
        "",
        "Recommended supplemental boreholes:",
        "",
        df_to_markdown(
            site_boreholes.rename(
                columns={
                    "settlement_risk": "thaw_sensitive_eic_proxy",
                    "differential_settlement": "eic_gradient_proxy",
                }
            ),
            columns=[
                "rank",
                "x",
                "y",
                "recommended_depth_m",
                "voi_score",
                "uncertainty",
                "thaw_sensitive_eic_proxy",
                "eic_gradient_proxy",
                "novelty",
            ],
            max_rows=8,
        ),
        "",
        "Recommended supplemental ERT lines:",
        "",
        df_to_markdown(
            site_lines,
            columns=["rank", "orientation", "x_start", "y_start", "x_end", "y_end", "line_score", "max_score"],
            max_rows=8,
        ),
        "",
        (
            "This observation-design result closes the loop between reconstruction and data acquisition without reframing the paper away "
            "from the reconstruction algorithm: high-uncertainty and high-ambiguity posterior regions are translated into ranked observation "
            "candidates with source data and weights preserved for audit. In the current implementation, the VOI weights are fixed by a "
            "transparent default rather than learned from a prospective field trial; future campaigns should treat these recommendations "
            "as testable posterior-diagnostic hypotheses."
        ),
        "",
        "## 5. Discussion",
        "",
        (
            "The experiments show that COLD-Recon can join synthetic full-field training, sparse multi-source observation tokens, "
            "conditional latent diffusion, public-data validation, and posterior uncertainty analysis in one reproducible pipeline. "
            "The current posterior is useful for facies, EIC, active-layer, thermal-state, and geophysical-property reconstruction, and the "
            "three-source gate gives a stricter real-data check than a single-dataset demonstration. Physics-guided denoiser "
            "fine-tuning is now represented as a train-time objective rather than only a post-processing step; latent-space "
            "guidance offers a conservative decoder-level correction, and post-hoc physics refinement gives a stronger "
            "continuous-field projection without changing the sampled facies ensemble. The VOI layer extends the posterior into "
            "a ranked next-observation diagnostic while keeping the decision weights visible. Calibration diagnostics still show "
            "under-dispersion for several continuous fields, and the external-generalization audit shows that ArcticData EIC transfer is heterogeneous under per-site best-simple baselines. The compact-site spatial guard now controls the previously exposed EIC transfer failures by deferring to a local IDW adapter at compact, sparsely supported sites. The domain-support audit makes the same boundary actionable by separating model-supported transfers from guarded local-prior non-inferiority. Jago high-EIC screening remains a small-sample, threshold-calibrated result with false-positive trade-offs. "
            "A journal-readiness audit now separates the completed CG-style algorithm evidence package from conditional EG-style field-generalization evidence, so the manuscript can state the current position without converting guarded external validation into an overbroad regional claim. "
            "This motivates larger multi-sample physics-guided diffusion training, "
            "domain adaptation from synthetic to real sites, broader multi-site public-data training, and field trials that compare VOI-ranked observations with conventional survey layouts."
        ),
        "",
        "## 6. Conclusions",
        "",
        (
            "COLD-Recon reframes permafrost site characterization as probabilistic 3D cryostratigraphic reconstruction from sparse "
            "observations. The implemented system produces posterior samples, field-scale public-data reconstructions, uncertainty "
            "metrics, physics-consistency diagnostics, observation-consistency tables, VOI-ranked supplemental observation targets, and manuscript-ready reconstruction figures. It is therefore a concrete, "
            "runnable basis for the COLD-Recon algorithm manuscript."
        ),
        "",
        "## Data and Code Availability",
        "",
        (
            "All synthetic data, processed public-data tokens, model checkpoints, predictions, tables, and figures are generated "
            "under the local `data/` and `outputs/` directories. Public USGS and Arctic Data Center data are downloaded into `data/raw/`; datasets "
            "requiring authentication or license steps are documented in `data/external/DOWNLOAD_INSTRUCTIONS.md`. The full "
            "reproducibility command sequence is listed in `paper/methods_and_results_mvp.md`. Nature-style main-figure source data are "
            "written to `outputs/source_data/nature_figure_*_source_data.csv`, and the audited artifact manifest "
            "is written to `outputs/tables/reproducibility_manifest.csv` with a Markdown report in `paper/reproducibility_audit.md`. "
            "Supplemental observation-design outputs are written to `outputs/tables/site_investigation_boreholes.csv`, "
            "`outputs/tables/site_investigation_ert_lines.csv` and `outputs/predictions/site_investigation_voi_score.npz`. "
            "Rare cryostructure diagnostics are written to `outputs/tables/synthetic_rare_cryostructure_audit.csv` and "
            "`outputs/figures/synthetic_rare_cryostructure_audit.*`; rare-facies hybrid diagnostics are written to "
            "`outputs/tables/diffusion_rare_facies_hybrid_metrics.csv`, `outputs/tables/rare_facies_hybrid_operating_curve.csv` and "
            "`outputs/figures/rare_facies_hybrid_operating_curve.*`. Posterior uncertainty-error alignment diagnostics are written to "
            "`outputs/tables/posterior_uncertainty_alignment.csv` and `outputs/figures/posterior_uncertainty_alignment.*`. "
            "Computational footprint diagnostics are written to `outputs/tables/computational_footprint.csv` and "
            "`outputs/figures/computational_footprint_summary.*`. Innovation-positioning diagnostics are written to "
            "`outputs/tables/innovation_positioning_audit.csv`, `outputs/tables/innovation_positioning_summary.json`, "
            "`outputs/source_data/innovation_positioning_audit_source_data.csv` and `outputs/figures/innovation_positioning_audit.*`. "
            "External multi-site generalization diagnostics are written to "
            "`outputs/tables/external_generalization_audit.csv`, `outputs/tables/external_generalization_site_deltas.csv` and "
            "`outputs/figures/external_generalization_audit.*`. Transfer-failure attribution diagnostics are written to "
            "`outputs/tables/transfer_failure_site_diagnostics.csv`, `outputs/tables/transfer_failure_attribution_summary.csv`, "
            "`outputs/source_data/transfer_failure_attribution_source_data.csv` and `outputs/figures/transfer_failure_attribution.*`. "
            "Domain-support applicability diagnostics are written to `outputs/tables/domain_support_site_audit.csv`, "
            "`outputs/tables/domain_support_summary.json`, `outputs/source_data/domain_support_audit_source_data.csv` and "
            "`outputs/figures/domain_support_audit.*`. "
            "Journal-readiness diagnostics are written to `outputs/tables/journal_readiness_audit.csv`, "
            "`outputs/tables/journal_readiness_summary.json`, `outputs/source_data/journal_readiness_audit_source_data.csv` "
            "and `outputs/figures/journal_readiness_audit.*`."
        ),
        "",
        "## Current Limitations",
        "",
        (
            "This draft reflects the current runnable research prototype. It still needs expanded multi-site training, stronger "
            "posterior calibration, journal-specific reference formatting, and additional high-resolution public covariates before "
            "submission. The external-generalization and transfer-failure attribution audits show that EIC gains are not uniform across public sites even after strict failures are controlled as non-inferior ties, and the rare-cryostructure and rare-facies hybrid audits also show that high-EIC event screening, wedge-facies recall and precision-cost trade-offs must remain separate claims."
        ),
        "",
        "## References",
        "",
        "Citation keys in the text resolve through `paper/references.bib`. The bibliography entries generated with this draft are:",
        "",
        references_markdown(),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
