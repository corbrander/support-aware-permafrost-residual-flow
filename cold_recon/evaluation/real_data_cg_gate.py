from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RealDataGateConfig:
    min_independent_sources: int = 3
    min_passed_tasks: int = 7
    min_eic_sources: int = 3
    min_facies_sources: int = 1


WEDGE_RECALL_MODEL = "COLDReconArcticDataWedgeRecallHead"


def _normalize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def _row(df: pd.DataFrame, model: str) -> pd.Series:
    rows = df[df["model"] == model]
    if rows.empty:
        raise ValueError(f"Missing model row: {model}")
    return rows.iloc[0]


def _finite(value: object) -> float:
    out = float(value)
    return out if np.isfinite(out) else float("nan")


def build_real_data_cg_benchmark(
    arctic_summary: pd.DataFrame,
    usgs_eic_comparison: pd.DataFrame,
    jago_eic_comparison: pd.DataFrame | None = None,
    config: RealDataGateConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or RealDataGateConfig()
    arctic_summary = _normalize_model_names(arctic_summary)
    usgs_eic_comparison = _normalize_model_names(usgs_eic_comparison)
    if jago_eic_comparison is not None:
        jago_eic_comparison = _normalize_model_names(jago_eic_comparison)
    arctic_adaptive = _row(arctic_summary, "COLDReconArcticDataAdaptiveHybrid")
    arctic_wedge_recall = _row(arctic_summary, WEDGE_RECALL_MODEL)
    arctic_knn = _row(arctic_summary, "SpatialDepthKNN")
    arctic_global = _row(arctic_summary, "GlobalMean")
    arctic_idw = _row(arctic_summary, "SpatialDepthIDW")
    usgs_model = _row(usgs_eic_comparison, "COLDReconUSGSEICConditionedDiffusion")
    usgs_simple = usgs_eic_comparison[usgs_eic_comparison["model"].isin(["GlobalMean", "DepthIDW", "SpatialDepthIDW"])]
    if usgs_simple.empty:
        raise ValueError("USGS comparison is missing simple EIC baselines")
    usgs_best_simple = usgs_simple.loc[usgs_simple["eic_rmse"].astype(float).idxmin()]
    usgs_spatial = _row(usgs_eic_comparison, "SpatialDepthIDW")

    arctic_best_eic_rmse = min(_finite(arctic_global["eic_rmse_mean"]), _finite(arctic_idw["eic_rmse_mean"]))
    rows: list[dict[str, Any]] = [
        {
            "source": "ArcticData cryostratigraphy",
            "task": "cryofacies",
            "metric": "facies_accuracy_mean",
            "higher_is_better": True,
            "model": "COLDReconArcticDataAdaptiveHybrid",
            "model_value": _finite(arctic_adaptive["facies_accuracy_mean"]),
            "baseline": "SpatialDepthKNN",
            "baseline_value": _finite(arctic_knn["facies_accuracy_mean"]),
            "relative_improvement": _finite(arctic_adaptive["facies_accuracy_mean"]) / _finite(arctic_knn["facies_accuracy_mean"]) - 1.0,
            "site_win_rate": _finite(arctic_adaptive.get("facies_win_rate_vs_spatial_knn", np.nan)),
            "passed": bool(
                _finite(arctic_adaptive["facies_accuracy_mean"]) > _finite(arctic_knn["facies_accuracy_mean"])
                and _finite(arctic_adaptive.get("facies_win_rate_vs_spatial_knn", np.nan)) >= 0.5
            ),
        },
        {
            "source": "ArcticData cryostratigraphy",
            "task": "EIC regression",
            "metric": "eic_rmse_mean",
            "higher_is_better": False,
            "model": "COLDReconArcticDataAdaptiveHybrid",
            "model_value": _finite(arctic_adaptive["eic_rmse_mean"]),
            "baseline": "best(GlobalMean,SpatialDepthIDW)",
            "baseline_value": arctic_best_eic_rmse,
            "relative_improvement": 1.0 - _finite(arctic_adaptive["eic_rmse_mean"]) / arctic_best_eic_rmse,
            "site_win_rate": _finite(arctic_adaptive.get("eic_rmse_win_rate_vs_best_simple_baseline", np.nan)),
            "site_noninferior_rate": _finite(arctic_adaptive.get("eic_rmse_noninferior_rate_vs_best_simple_baseline", np.nan)),
            "passed": bool(
                _finite(arctic_adaptive["eic_rmse_mean"]) < arctic_best_eic_rmse
                and _finite(arctic_adaptive.get("eic_rmse_win_rate_vs_best_simple_baseline", np.nan)) >= 0.5
                and (
                    not np.isfinite(_finite(arctic_adaptive.get("eic_rmse_noninferior_rate_vs_best_simple_baseline", np.nan)))
                    or _finite(arctic_adaptive.get("eic_rmse_noninferior_rate_vs_best_simple_baseline", np.nan)) >= 0.8
                )
            ),
        },
        {
            "source": "ArcticData cryostratigraphy",
            "task": "wedge-ice recall",
            "metric": "wedge_ice_recall_mean",
            "higher_is_better": True,
            "model": WEDGE_RECALL_MODEL,
            "model_value": _finite(arctic_wedge_recall["wedge_ice_recall_mean"]),
            "baseline": "SpatialDepthKNN",
            "baseline_value": _finite(arctic_knn["wedge_ice_recall_mean"]),
            "relative_improvement": _finite(arctic_wedge_recall["wedge_ice_recall_mean"]) / _finite(arctic_knn["wedge_ice_recall_mean"]) - 1.0,
            "site_win_rate": _finite(arctic_wedge_recall.get("wedge_win_rate_vs_spatial_knn", np.nan)),
            "site_noninferior_rate": _finite(arctic_wedge_recall.get("wedge_noninferior_rate_vs_spatial_knn", np.nan)),
            "passed": bool(
                _finite(arctic_wedge_recall["wedge_ice_recall_mean"]) > _finite(arctic_knn["wedge_ice_recall_mean"])
                and _finite(arctic_wedge_recall.get("wedge_noninferior_rate_vs_spatial_knn", np.nan)) >= 0.8
            ),
        },
        {
            "source": "ArcticData cryostratigraphy",
            "task": "high-EIC event",
            "metric": "high_eic_f1_mean",
            "higher_is_better": True,
            "model": "COLDReconArcticDataAdaptiveHybrid",
            "model_value": _finite(arctic_adaptive["high_eic_f1_mean"]),
            "baseline": "SpatialDepthIDW",
            "baseline_value": _finite(arctic_idw["high_eic_f1_mean"]),
            "relative_improvement": _finite(arctic_adaptive["high_eic_f1_mean"]) / _finite(arctic_idw["high_eic_f1_mean"]) - 1.0,
            "site_win_rate": _finite(arctic_adaptive.get("high_eic_f1_win_rate_vs_spatial_idw", np.nan)),
            "site_noninferior_rate": _finite(arctic_adaptive.get("high_eic_f1_noninferior_rate_vs_spatial_idw", np.nan)),
            "passed": bool(
                _finite(arctic_adaptive["high_eic_f1_mean"]) > _finite(arctic_idw["high_eic_f1_mean"])
                and (
                    not np.isfinite(_finite(arctic_adaptive.get("high_eic_f1_noninferior_rate_vs_spatial_idw", np.nan)))
                    or _finite(arctic_adaptive.get("high_eic_f1_noninferior_rate_vs_spatial_idw", np.nan)) >= 0.8
                )
            ),
        },
        {
            "source": "USGS EIC cores",
            "task": "EIC regression",
            "metric": "eic_rmse",
            "higher_is_better": False,
            "model": "COLDReconUSGSEICConditionedDiffusion",
            "model_value": _finite(usgs_model["eic_rmse"]),
            "baseline": str(usgs_best_simple["model"]),
            "baseline_value": _finite(usgs_best_simple["eic_rmse"]),
            "relative_improvement": 1.0 - _finite(usgs_model["eic_rmse"]) / _finite(usgs_best_simple["eic_rmse"]),
            "site_win_rate": np.nan,
            "passed": bool(_finite(usgs_model["eic_rmse"]) < _finite(usgs_best_simple["eic_rmse"])),
        },
        {
            "source": "USGS EIC cores",
            "task": "high-EIC event",
            "metric": "high_eic_f1",
            "higher_is_better": True,
            "model": "COLDReconUSGSEICConditionedDiffusion",
            "model_value": _finite(usgs_model["high_eic_f1"]),
            "baseline": "SpatialDepthIDW",
            "baseline_value": _finite(usgs_spatial["high_eic_f1"]),
            "relative_improvement": _finite(usgs_model["high_eic_f1"]) / _finite(usgs_spatial["high_eic_f1"]) - 1.0,
            "site_win_rate": np.nan,
            "passed": bool(_finite(usgs_model["high_eic_f1"]) > _finite(usgs_spatial["high_eic_f1"])),
        },
    ]
    if jago_eic_comparison is not None and not jago_eic_comparison.empty:
        jago_model = _row(jago_eic_comparison, "COLDReconJagoGroundIceConditionedDiffusion")
        jago_simple = jago_eic_comparison[jago_eic_comparison["model"].isin(["GlobalMean", "DepthIDW", "SpatialDepthIDW"])]
        if jago_simple.empty:
            raise ValueError("Jago comparison is missing simple EIC baselines")
        jago_best_simple = jago_simple.loc[jago_simple["eic_rmse"].astype(float).idxmin()]
        jago_spatial = _row(jago_eic_comparison, "SpatialDepthIDW")
        jago_model_rmse = _finite(jago_model["eic_rmse"])
        jago_best_rmse = _finite(jago_best_simple["eic_rmse"])
        jago_model_f1 = _finite(jago_model.get("high_eic_f1", np.nan))
        jago_spatial_f1 = _finite(jago_spatial.get("high_eic_f1", np.nan))
        rows.extend(
            [
                {
                    "source": "ArcticData Jago River 2018 ground ice",
                    "task": "EIC regression",
                    "metric": "eic_rmse",
                    "higher_is_better": False,
                    "model": "COLDReconJagoGroundIceConditionedDiffusion",
                    "model_value": jago_model_rmse,
                    "baseline": str(jago_best_simple["model"]),
                    "baseline_value": jago_best_rmse,
                    "relative_improvement": 1.0 - jago_model_rmse / jago_best_rmse if np.isfinite(jago_best_rmse) and jago_best_rmse > 0.0 else np.nan,
                    "site_win_rate": np.nan,
                    "passed": bool(np.isfinite(jago_model_rmse) and np.isfinite(jago_best_rmse) and jago_model_rmse < jago_best_rmse),
                },
                {
                    "source": "ArcticData Jago River 2018 ground ice",
                    "task": "high-EIC event",
                    "metric": "high_eic_f1",
                    "higher_is_better": True,
                    "model": "COLDReconJagoGroundIceConditionedDiffusion",
                    "model_value": jago_model_f1,
                    "baseline": "SpatialDepthIDW",
                    "baseline_value": jago_spatial_f1,
                    "relative_improvement": jago_model_f1 / jago_spatial_f1 - 1.0 if np.isfinite(jago_spatial_f1) and jago_spatial_f1 > 0.0 else np.nan,
                    "site_win_rate": np.nan,
                    "passed": bool(np.isfinite(jago_model_f1) and np.isfinite(jago_spatial_f1) and jago_model_f1 > jago_spatial_f1),
                },
            ]
        )
    benchmark = pd.DataFrame(rows)
    passed = benchmark["passed"].astype(bool)
    sources = sorted(benchmark.loc[passed, "source"].unique().tolist())
    eic_sources = sorted(benchmark.loc[passed & benchmark["task"].str.contains("EIC", case=False), "source"].unique().tolist())
    facies_sources = sorted(benchmark.loc[passed & benchmark["task"].eq("cryofacies"), "source"].unique().tolist())
    gate = {
        "independent_public_sources_passed": len(sources),
        "passed_tasks": int(passed.sum()),
        "total_tasks": int(len(benchmark)),
        "eic_sources_passed": len(eic_sources),
        "facies_sources_passed": len(facies_sources),
        "min_independent_sources": int(cfg.min_independent_sources),
        "min_passed_tasks": int(cfg.min_passed_tasks),
        "min_eic_sources": int(cfg.min_eic_sources),
        "min_facies_sources": int(cfg.min_facies_sources),
        "cg_model_evidence_passed": bool(
            len(sources) >= cfg.min_independent_sources
            and int(passed.sum()) >= cfg.min_passed_tasks
            and len(eic_sources) >= cfg.min_eic_sources
            and len(facies_sources) >= cfg.min_facies_sources
        ),
        "cg_plus_remaining_limitations": [
            "USGS branch validates EIC but not cryofacies labels.",
            "ArcticData wedge-ice recall head is recall-oriented; the operating-curve audit exposes precision/false-positive tradeoffs, but operating-point selection remains a modelling choice.",
            "Jago River adds an independent ground-ice/EIC source, but its small borehole table should be treated as a targeted validation rather than a regional benchmark.",
        ],
    }
    return benchmark, gate
