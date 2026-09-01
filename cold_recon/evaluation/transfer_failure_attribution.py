from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = (
    "train_n",
    "condition_n",
    "holdout_n",
    "train_boreholes",
    "holdout_boreholes",
    "adaptive_eic_train_observations",
    "adaptive_eic_train_groups",
    "facies_delta",
    "eic_model_rmse",
    "eic_global_mean_rmse",
    "eic_spatial_idw_rmse",
    "eic_best_simple_rmse",
    "eic_rmse_reduction_vs_best_simple",
    "high_eic_f1_delta_vs_spatial_idw",
    "wedge_recall_delta",
    "wedge_precision_delta",
)


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(lambda value: _truthy(value) if pd.notna(value) else np.nan).astype(object)


def _truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _outcome(row: pd.Series) -> str:
    if _truthy(row.get("eic_win_vs_best_simple", False)):
        return "win"
    if _truthy(row.get("eic_noninferior_vs_best_simple", False)):
        return "noninferior"
    return "failure"


def _failure_attribution(row: pd.Series, train_median: float, holdout_median: float) -> str:
    if _outcome(row) != "failure":
        return "no_eic_failure"
    reasons: list[str] = []
    if str(row.get("eic_best_simple_baseline", "")) == "SpatialDepthIDW":
        reasons.append("spatial_idw_best_simple")
    spatial_advantage = float(row.get("spatial_idw_advantage_vs_global", np.nan))
    if np.isfinite(spatial_advantage) and spatial_advantage > 0.02:
        reasons.append("strong_spatial_local_continuity")
    holdout_n = float(row.get("holdout_n", np.nan))
    if np.isfinite(holdout_n) and np.isfinite(holdout_median) and holdout_n <= holdout_median:
        reasons.append("compact_holdout")
    train_n = float(row.get("train_n", np.nan))
    if np.isfinite(train_n) and np.isfinite(train_median) and train_n <= train_median:
        reasons.append("limited_training_support")
    return "; ".join(reasons) if reasons else "unassigned_eic_failure"


def _readiness_score(row: pd.Series) -> float:
    values = [
        row.get("facies_noninferior"),
        row.get("eic_noninferior_vs_best_simple"),
        row.get("high_eic_noninferior_vs_spatial_idw"),
        row.get("wedge_recall_noninferior"),
    ]
    clean = [float(_truthy(value)) for value in values if pd.notna(value)]
    return float(np.mean(clean)) if clean else float("nan")


def _corr(values: pd.Series, target: pd.Series, method: str) -> float:
    values = pd.to_numeric(values, errors="coerce")
    target = pd.to_numeric(target, errors="coerce")
    valid = values.notna() & target.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    if values[valid].nunique() < 2 or target[valid].nunique() < 2:
        return float("nan")
    out = values[valid].corr(target[valid], method=method)
    return float(out) if pd.notna(out) else float("nan")


def build_transfer_failure_site_diagnostics(site_deltas: pd.DataFrame) -> pd.DataFrame:
    """Convert site-level generalization deltas into domain-transfer diagnostics."""
    if site_deltas.empty:
        return pd.DataFrame()
    if "site" not in site_deltas.columns:
        raise ValueError("site_deltas must contain a site column")
    out = site_deltas.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in [
        "facies_win",
        "facies_noninferior",
        "eic_win_vs_best_simple",
        "eic_noninferior_vs_best_simple",
        "high_eic_f1_win_vs_spatial_idw",
        "wedge_recall_win",
        "wedge_recall_noninferior",
    ]:
        if col in out.columns:
            out[col] = _bool_series(out[col])

    out["eic_model_gap_vs_best_simple"] = out["eic_model_rmse"] - out["eic_best_simple_rmse"]
    out["eic_relative_gap_vs_best_simple"] = out["eic_model_gap_vs_best_simple"] / out["eic_best_simple_rmse"]
    out["spatial_idw_advantage_vs_global"] = out["eic_global_mean_rmse"] - out["eic_spatial_idw_rmse"]
    out["holdout_fraction"] = out["holdout_n"] / (out["train_n"] + out["holdout_n"])
    if "adaptive_eic_method" in out.columns:
        out["guarded_by_transfer_adapter"] = out["adaptive_eic_method"].astype(str).eq("transfer_idw_adapter")
    else:
        out["guarded_by_transfer_adapter"] = False
    out["high_eic_noninferior_vs_spatial_idw"] = out["high_eic_f1_delta_vs_spatial_idw"].map(
        lambda value: bool(value >= 0.0) if pd.notna(value) else np.nan
    ).astype(object)
    out["eic_transfer_outcome"] = out.apply(_outcome, axis=1)
    train_median = float(out["train_n"].median()) if out["train_n"].notna().any() else float("nan")
    holdout_median = float(out["holdout_n"].median()) if out["holdout_n"].notna().any() else float("nan")
    out["failure_attribution"] = out.apply(lambda row: _failure_attribution(row, train_median, holdout_median), axis=1)
    out["transfer_readiness_score"] = out.apply(_readiness_score, axis=1)
    return out


def build_transfer_failure_attribution_summary(site_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize exploratory associations with EIC transfer success."""
    if site_diagnostics.empty:
        return pd.DataFrame()
    target = pd.to_numeric(site_diagnostics["eic_rmse_reduction_vs_best_simple"], errors="coerce")
    signals = {
        "training observations": "train_n",
        "holdout observations": "holdout_n",
        "holdout boreholes": "holdout_boreholes",
        "spatial IDW advantage over global mean": "spatial_idw_advantage_vs_global",
        "facies accuracy delta": "facies_delta",
        "transfer readiness score": "transfer_readiness_score",
    }
    rows: list[dict[str, float | str]] = []
    for label, column in signals.items():
        if column not in site_diagnostics.columns:
            continue
        values = pd.to_numeric(site_diagnostics[column], errors="coerce")
        valid = values.notna() & target.notna()
        spearman = _corr(values, target, "spearman")
        pearson = _corr(values, target, "pearson")
        rows.append(
            {
                "signal": label,
                "column": column,
                "target": "eic_rmse_reduction_vs_best_simple",
                "n_sites": int(valid.sum()),
                "spearman": spearman,
                "pearson": pearson,
                "interpretation": (
                    "exploratory small-n association; used for transfer diagnostics, not hypothesis testing"
                ),
            }
        )

    outcome_counts = site_diagnostics["eic_transfer_outcome"].value_counts(dropna=False).to_dict()
    rows.append(
        {
            "signal": "EIC outcome counts",
            "column": "eic_transfer_outcome",
            "target": "eic_rmse_reduction_vs_best_simple",
            "n_sites": int(len(site_diagnostics)),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "interpretation": "; ".join(f"{key}={value}" for key, value in sorted(outcome_counts.items())),
        }
    )
    failure_counts: dict[str, int] = {}
    for value in site_diagnostics.loc[site_diagnostics["eic_transfer_outcome"].eq("failure"), "failure_attribution"]:
        for reason in str(value).split("; "):
            if reason and reason != "no_eic_failure":
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
    rows.append(
        {
            "signal": "failure attribution counts",
            "column": "failure_attribution",
            "target": "eic_rmse_reduction_vs_best_simple",
            "n_sites": int((site_diagnostics["eic_transfer_outcome"] == "failure").sum()),
            "spearman": float("nan"),
            "pearson": float("nan"),
            "interpretation": "; ".join(f"{key}={value}" for key, value in sorted(failure_counts.items()))
            if failure_counts
            else "no EIC failures",
        }
    )
    return pd.DataFrame(rows)
