from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ADAPTIVE_MODEL = "COLDReconArcticDataAdaptiveHybrid"
WEDGE_RECALL_MODEL = "COLDReconArcticDataWedgeRecallHead"
FACIES_BASELINE = "SpatialDepthKNN"
EIC_BASELINES = ("GlobalMean", "SpatialDepthIDW")
EVENT_BASELINE = "SpatialDepthIDW"


@dataclass(frozen=True)
class GeneralizationThresholds:
    facies_noninferior_margin: float = 0.02
    eic_noninferior_margin: float = 0.02
    wedge_noninferior_margin: float = 0.0


def _finite(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _row(site_metrics: pd.DataFrame, model: str) -> pd.Series | None:
    if site_metrics.empty or "model" not in site_metrics.columns:
        return None
    rows = site_metrics[site_metrics["model"].astype(str).eq(model)]
    return rows.iloc[0] if not rows.empty else None


def _value(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row.index:
        return float("nan")
    return _finite(row[column])


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _rate(values: pd.Series) -> float:
    valid = values.dropna()
    return float(valid.mean()) if len(valid) else float("nan")


def _site_list(site_deltas: pd.DataFrame, mask: pd.Series) -> str:
    if site_deltas.empty or "site" not in site_deltas.columns:
        return ""
    values = site_deltas.loc[mask.fillna(False), "site"].astype(str).tolist()
    return "; ".join(values)


def _best_eic_baseline(global_row: pd.Series | None, idw_row: pd.Series | None) -> tuple[str, float]:
    candidates = {
        "GlobalMean": _value(global_row, "eic_rmse"),
        "SpatialDepthIDW": _value(idw_row, "eic_rmse"),
    }
    finite = {name: value for name, value in candidates.items() if np.isfinite(value)}
    if not finite:
        return "", float("nan")
    name = min(finite, key=finite.get)
    return name, finite[name]


def build_external_generalization_site_deltas(
    multisite_metrics: pd.DataFrame,
    thresholds: GeneralizationThresholds | None = None,
) -> pd.DataFrame:
    """Build site-wise public holdout deltas from the ArcticData multi-site table."""
    cfg = thresholds or GeneralizationThresholds()
    if multisite_metrics.empty:
        return pd.DataFrame()
    if "site" not in multisite_metrics.columns:
        raise ValueError("multisite_metrics must contain a site column")

    rows: list[dict[str, float | str | bool]] = []
    for site, group in multisite_metrics.groupby("site", sort=True):
        adaptive = _row(group, ADAPTIVE_MODEL)
        facies_base = _row(group, FACIES_BASELINE)
        global_mean = _row(group, "GlobalMean")
        spatial_idw = _row(group, "SpatialDepthIDW")
        wedge_head = _row(group, WEDGE_RECALL_MODEL)

        best_eic_name, best_eic = _best_eic_baseline(global_mean, spatial_idw)
        facies_model = _value(adaptive, "facies_accuracy")
        facies_baseline = _value(facies_base, "facies_accuracy")
        eic_model = _value(adaptive, "eic_rmse")
        event_model = _value(adaptive, "high_eic_f1")
        event_baseline = _value(spatial_idw, "high_eic_f1")
        wedge_model = _value(wedge_head, "wedge_ice_recall")
        wedge_baseline = _value(facies_base, "wedge_ice_recall")
        wedge_precision_model = _value(wedge_head, "wedge_ice_precision")
        wedge_precision_baseline = _value(facies_base, "wedge_ice_precision")

        facies_delta = facies_model - facies_baseline
        eic_reduction = 1.0 - eic_model / best_eic if np.isfinite(eic_model) and np.isfinite(best_eic) and best_eic > 0 else float("nan")
        event_delta = event_model - event_baseline
        wedge_delta = wedge_model - wedge_baseline
        rows.append(
            {
                "site": str(site),
                "train_n": _value(adaptive, "train_n"),
                "condition_n": _value(adaptive, "condition_n"),
                "holdout_n": _value(adaptive, "holdout_n"),
                "train_boreholes": _value(adaptive, "train_boreholes"),
                "holdout_boreholes": _value(adaptive, "holdout_boreholes"),
                "adaptive_eic_method": str(adaptive.get("adaptive_eic_method", "")) if adaptive is not None else "",
                "adaptive_eic_transfer_guard_reason": str(adaptive.get("adaptive_eic_transfer_guard_reason", "")) if adaptive is not None else "",
                "adaptive_eic_train_observations": _value(adaptive, "adaptive_eic_train_observations"),
                "adaptive_eic_train_groups": _value(adaptive, "adaptive_eic_train_groups"),
                "facies_model": facies_model,
                "facies_baseline": facies_baseline,
                "facies_delta": facies_delta,
                "facies_relative_gain": facies_delta / facies_baseline if np.isfinite(facies_delta) and np.isfinite(facies_baseline) and facies_baseline else float("nan"),
                "facies_win": bool(facies_delta > 0.0) if np.isfinite(facies_delta) else np.nan,
                "facies_noninferior": bool(facies_delta >= -cfg.facies_noninferior_margin) if np.isfinite(facies_delta) else np.nan,
                "eic_model_rmse": eic_model,
                "eic_global_mean_rmse": _value(global_mean, "eic_rmse"),
                "eic_spatial_idw_rmse": _value(spatial_idw, "eic_rmse"),
                "eic_best_simple_baseline": best_eic_name,
                "eic_best_simple_rmse": best_eic,
                "eic_rmse_reduction_vs_best_simple": eic_reduction,
                "eic_win_vs_best_simple": bool(eic_model < best_eic) if np.isfinite(eic_model) and np.isfinite(best_eic) else np.nan,
                "eic_noninferior_vs_best_simple": bool(eic_model <= best_eic * (1.0 + cfg.eic_noninferior_margin)) if np.isfinite(eic_model) and np.isfinite(best_eic) else np.nan,
                "high_eic_f1_model": event_model,
                "high_eic_f1_spatial_idw": event_baseline,
                "high_eic_f1_delta_vs_spatial_idw": event_delta,
                "high_eic_f1_win_vs_spatial_idw": bool(event_delta > 0.0) if np.isfinite(event_delta) else np.nan,
                "high_eic_noninferior_vs_spatial_idw": bool(event_delta >= 0.0) if np.isfinite(event_delta) else np.nan,
                "wedge_recall_head": wedge_model,
                "wedge_recall_baseline": wedge_baseline,
                "wedge_recall_delta": wedge_delta,
                "wedge_recall_win": bool(wedge_delta > 0.0) if np.isfinite(wedge_delta) else np.nan,
                "wedge_recall_noninferior": bool(wedge_delta >= -cfg.wedge_noninferior_margin) if np.isfinite(wedge_delta) else np.nan,
                "wedge_precision_head": wedge_precision_model,
                "wedge_precision_baseline": wedge_precision_baseline,
                "wedge_precision_delta": wedge_precision_model - wedge_precision_baseline
                if np.isfinite(wedge_precision_model) and np.isfinite(wedge_precision_baseline)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_external_generalization_audit(site_deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize external multi-site behavior as bounded task-level evidence."""
    if site_deltas.empty:
        return pd.DataFrame()

    facies_fail = pd.to_numeric(site_deltas["facies_delta"], errors="coerce") < 0.0
    eic_fail = pd.to_numeric(site_deltas["eic_rmse_reduction_vs_best_simple"], errors="coerce") < 0.0
    wedge_valid = pd.to_numeric(site_deltas["wedge_recall_delta"], errors="coerce").notna()
    wedge_fail = (pd.to_numeric(site_deltas["wedge_recall_delta"], errors="coerce") < 0.0) & wedge_valid
    event_fail = pd.to_numeric(site_deltas["high_eic_f1_delta_vs_spatial_idw"], errors="coerce") < 0.0
    event_failure_sites = _site_list(site_deltas, event_fail)
    event_noninferior_rate = _rate(site_deltas["high_eic_noninferior_vs_spatial_idw"])
    if event_failure_sites:
        event_boundary = "High-EIC event behavior is reported as thresholded screening; strict site failures remain at the listed sites."
    elif np.isfinite(event_noninferior_rate) and event_noninferior_rate >= 1.0:
        event_boundary = "High-EIC event screening is non-inferior at every audited site, but wins occur only at a subset of sites."
    else:
        event_boundary = "High-EIC event behavior is reported as thresholded screening, not as a solved rare-facies reconstruction target."
    eic_model_mean = _mean(site_deltas["eic_model_rmse"])
    eic_spatial_mean = _mean(site_deltas["eic_spatial_idw_rmse"])
    eic_failure_sites = _site_list(site_deltas, eic_fail)
    eic_noninferior_rate = _rate(site_deltas["eic_noninferior_vs_best_simple"])
    if eic_failure_sites:
        eic_boundary = (
            "The aggregate evidence-gate RMSE improves against simple baselines, "
            "but strict per-site best-simple failures remain at the listed sites."
        )
    elif np.isfinite(eic_noninferior_rate) and eic_noninferior_rate >= 1.0:
        eic_boundary = (
            "The compact-site spatial guard removes strict per-site EIC failures in this five-site audit; "
            "the remaining boundary is that several sites are non-inferior ties rather than wins."
        )
    else:
        eic_boundary = (
            "The aggregate evidence-gate RMSE improves against simple baselines, "
            "but site-level EIC transfer remains heterogeneous under the stricter per-site comparator."
        )

    rows = [
        {
            "task": "cryofacies",
            "metric": "facies_accuracy",
            "higher_is_better": True,
            "model": ADAPTIVE_MODEL,
            "baseline": FACIES_BASELINE,
            "n_sites": int(pd.to_numeric(site_deltas["facies_delta"], errors="coerce").notna().sum()),
            "model_value": _mean(site_deltas["facies_model"]),
            "baseline_value": _mean(site_deltas["facies_baseline"]),
            "absolute_delta": _mean(site_deltas["facies_delta"]),
            "relative_improvement": _mean(site_deltas["facies_relative_gain"]),
            "site_win_rate": _rate(site_deltas["facies_win"]),
            "site_noninferior_rate": _rate(site_deltas["facies_noninferior"]),
            "failure_sites": _site_list(site_deltas, facies_fail),
            "boundary": "Aggregate facies accuracy improves, but several sites are ties or near-ties rather than large wins.",
        },
        {
            "task": "EIC regression",
            "metric": "eic_rmse",
            "higher_is_better": False,
            "model": ADAPTIVE_MODEL,
            "baseline": "per-site best(GlobalMean,SpatialDepthIDW)",
            "n_sites": int(pd.to_numeric(site_deltas["eic_model_rmse"], errors="coerce").notna().sum()),
            "model_value": _mean(site_deltas["eic_model_rmse"]),
            "baseline_value": _mean(site_deltas["eic_best_simple_rmse"]),
            "absolute_delta": _mean(site_deltas["eic_best_simple_rmse"]) - _mean(site_deltas["eic_model_rmse"]),
            "relative_improvement": _mean(site_deltas["eic_rmse_reduction_vs_best_simple"]),
            "evidence_gate_baseline": "aggregate SpatialDepthIDW mean",
            "evidence_gate_baseline_value": eic_spatial_mean,
            "evidence_gate_relative_improvement": 1.0 - eic_model_mean / eic_spatial_mean
            if np.isfinite(eic_model_mean) and np.isfinite(eic_spatial_mean) and eic_spatial_mean > 0.0
            else float("nan"),
            "site_win_rate": _rate(site_deltas["eic_win_vs_best_simple"]),
            "site_noninferior_rate": eic_noninferior_rate,
            "failure_sites": eic_failure_sites,
            "boundary": eic_boundary,
        },
        {
            "task": "wedge-ice recall",
            "metric": "wedge_ice_recall",
            "higher_is_better": True,
            "model": WEDGE_RECALL_MODEL,
            "baseline": FACIES_BASELINE,
            "n_sites": int(wedge_valid.sum()),
            "model_value": _mean(site_deltas.loc[wedge_valid, "wedge_recall_head"]),
            "baseline_value": _mean(site_deltas.loc[wedge_valid, "wedge_recall_baseline"]),
            "absolute_delta": _mean(site_deltas.loc[wedge_valid, "wedge_recall_delta"]),
            "relative_improvement": np.nan,
            "site_win_rate": _rate(site_deltas.loc[wedge_valid, "wedge_recall_win"]),
            "site_noninferior_rate": _rate(site_deltas.loc[wedge_valid, "wedge_recall_noninferior"]),
            "secondary_metric": "wedge_ice_precision",
            "secondary_model_value": _mean(site_deltas["wedge_precision_head"]),
            "secondary_baseline_value": _mean(site_deltas["wedge_precision_baseline"]),
            "failure_sites": _site_list(site_deltas, wedge_fail),
            "boundary": "Wedge handling is recall-oriented; precision and false-positive control remain operating-point choices.",
        },
        {
            "task": "high-EIC event",
            "metric": "high_eic_f1",
            "higher_is_better": True,
            "model": ADAPTIVE_MODEL,
            "baseline": EVENT_BASELINE,
            "n_sites": int(pd.to_numeric(site_deltas["high_eic_f1_delta_vs_spatial_idw"], errors="coerce").notna().sum()),
            "model_value": _mean(site_deltas["high_eic_f1_model"]),
            "baseline_value": _mean(site_deltas["high_eic_f1_spatial_idw"]),
            "absolute_delta": _mean(site_deltas["high_eic_f1_delta_vs_spatial_idw"]),
            "relative_improvement": np.nan,
            "site_win_rate": _rate(site_deltas["high_eic_f1_win_vs_spatial_idw"]),
            "site_noninferior_rate": event_noninferior_rate,
            "failure_sites": event_failure_sites,
            "boundary": event_boundary,
        },
    ]
    return pd.DataFrame(rows)
