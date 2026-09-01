from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DomainSupportResult:
    site_audit: pd.DataFrame
    summary: dict[str, Any]


def _finite(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _norm_log(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").clip(lower=0.0)
    if vals.notna().sum() == 0:
        return pd.Series(np.zeros(len(vals)), index=values.index)
    max_val = float(vals.max())
    if max_val <= 0.0:
        return pd.Series(np.zeros(len(vals)), index=values.index)
    return np.log1p(vals).divide(np.log1p(max_val)).fillna(0.0)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _outcome_label(win: object, noninferior: object) -> str:
    if _is_missing(win) and _is_missing(noninferior):
        return "not_evaluated"
    win_bool = _truthy(win)
    noninferior_bool = _truthy(noninferior)
    if win_bool:
        return "win"
    if noninferior_bool:
        return "noninferior"
    return "failure"


def _outcome_score(label: str) -> float:
    return {"win": 1.0, "noninferior": 0.5, "failure": 0.0, "not_evaluated": np.nan}.get(label, np.nan)


def build_domain_support_audit(site_deltas: pd.DataFrame) -> DomainSupportResult:
    """Audit when public-site transfer is model-supported, guarded or not ready.

    Support scores use train-side quantities and selected adaptive method labels.
    Outcome columns use holdout metrics only to evaluate the support rule after
    the fact, keeping deployment-time support and post-hoc performance separate.
    """
    if site_deltas.empty:
        empty = pd.DataFrame()
        return DomainSupportResult(site_audit=empty, summary={"n_sites": 0, "applicability_boundary": "missing site deltas"})
    required = {"site", "train_n", "train_boreholes", "adaptive_eic_method"}
    missing = required.difference(site_deltas.columns)
    if missing:
        raise ValueError(f"site_deltas missing required columns: {sorted(missing)}")

    df = site_deltas.copy()
    train_obs = pd.to_numeric(df.get("adaptive_eic_train_observations", df["train_n"]), errors="coerce")
    train_groups = pd.to_numeric(df.get("adaptive_eic_train_groups", df["train_boreholes"]), errors="coerce")
    train_boreholes = pd.to_numeric(df["train_boreholes"], errors="coerce")
    holdout_fraction = pd.to_numeric(df.get("holdout_n", 0.0), errors="coerce") / (
        pd.to_numeric(df.get("train_n", 0.0), errors="coerce") + pd.to_numeric(df.get("holdout_n", 0.0), errors="coerce")
    )
    holdout_fraction = holdout_fraction.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    obs_score = _norm_log(train_obs)
    group_score = _norm_log(train_groups)
    borehole_score = _norm_log(train_boreholes)
    split_score = (1.0 - holdout_fraction).clip(lower=0.0, upper=1.0)
    support_score = (0.35 * obs_score + 0.25 * group_score + 0.25 * borehole_score + 0.15 * split_score).clip(0.0, 1.0)

    guarded = df["adaptive_eic_method"].astype(str).eq("transfer_idw_adapter")
    guard_reason = df.get("adaptive_eic_transfer_guard_reason", pd.Series([""] * len(df))).astype(str)
    applicability_class = []
    action = []
    for idx, score in support_score.items():
        is_guarded = bool(guarded.loc[idx])
        if is_guarded:
            applicability_class.append("guarded local-prior")
            action.append("Use the train-only local IDW adapter and report non-inferiority rather than model-transfer wins.")
        elif float(score) >= 0.82:
            applicability_class.append("model-supported transfer")
            action.append("Use the adaptive COLD-Recon hybrid and report site-wise wins/non-inferiority.")
        elif float(score) >= 0.68:
            applicability_class.append("moderate support")
            action.append("Use the adaptive hybrid with explicit site-wise monitoring.")
        else:
            applicability_class.append("low support")
            action.append("Require additional local boreholes or geophysical lines before claiming transfer.")

    rows: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        facies_label = _outcome_label(row.get("facies_win", np.nan), row.get("facies_noninferior", np.nan))
        eic_label = _outcome_label(row.get("eic_win_vs_best_simple", np.nan), row.get("eic_noninferior_vs_best_simple", np.nan))
        high_eic_label = _outcome_label(row.get("high_eic_f1_win_vs_spatial_idw", np.nan), row.get("high_eic_noninferior_vs_spatial_idw", np.nan))
        wedge_label = _outcome_label(row.get("wedge_recall_win", np.nan), row.get("wedge_recall_noninferior", np.nan))
        labels = [facies_label, eic_label, high_eic_label, wedge_label]
        outcome_scores = [_outcome_score(label) for label in labels]
        rows.append(
            {
                "site": str(row["site"]),
                "support_score": float(support_score.loc[idx]),
                "train_observation_score": float(obs_score.loc[idx]),
                "train_group_score": float(group_score.loc[idx]),
                "train_borehole_score": float(borehole_score.loc[idx]),
                "split_support_score": float(split_score.loc[idx]),
                "train_n": _finite(row.get("train_n")),
                "holdout_n": _finite(row.get("holdout_n")),
                "train_boreholes": _finite(row.get("train_boreholes")),
                "adaptive_eic_method": str(row.get("adaptive_eic_method", "")),
                "guard_reason": str(guard_reason.loc[idx]),
                "guarded_by_transfer_adapter": bool(guarded.loc[idx]),
                "applicability_class": applicability_class[len(rows)],
                "recommended_action": action[len(rows)],
                "facies_outcome": facies_label,
                "eic_outcome": eic_label,
                "high_eic_outcome": high_eic_label,
                "wedge_outcome": wedge_label,
                "outcome_score_mean": float(np.nanmean(outcome_scores)),
                "eic_rmse_reduction_vs_best_simple": _finite(row.get("eic_rmse_reduction_vs_best_simple")),
                "spatial_idw_advantage_vs_global": _finite(row.get("spatial_idw_advantage_vs_global")),
                "facies_delta": _finite(row.get("facies_delta")),
                "wedge_precision_delta": _finite(row.get("wedge_precision_delta")),
            }
        )
    site_audit = pd.DataFrame(rows).sort_values(["support_score", "site"], ascending=[False, True]).reset_index(drop=True)
    evaluated = site_audit[["facies_outcome", "eic_outcome", "high_eic_outcome", "wedge_outcome"]].apply(
        lambda col: ~col.astype(str).eq("not_evaluated")
    )
    noninferior_all = site_audit[["facies_outcome", "eic_outcome", "high_eic_outcome", "wedge_outcome"]].apply(
        lambda col: col.astype(str).isin(["win", "noninferior", "not_evaluated"])
    )
    guarded_sites = site_audit[site_audit["guarded_by_transfer_adapter"].astype(bool)]["site"].tolist()
    support_outcome_corr = float("nan")
    if len(site_audit) >= 3 and site_audit["support_score"].nunique() > 1:
        support_outcome_corr = float(site_audit["support_score"].corr(site_audit["outcome_score_mean"], method="spearman"))
    summary = {
        "n_sites": int(len(site_audit)),
        "n_model_supported": int(site_audit["applicability_class"].eq("model-supported transfer").sum()),
        "n_moderate_support": int(site_audit["applicability_class"].eq("moderate support").sum()),
        "n_guarded_local_prior": int(site_audit["applicability_class"].eq("guarded local-prior").sum()),
        "n_low_support": int(site_audit["applicability_class"].eq("low support").sum()),
        "all_sites_noninferior_all_evaluated_tasks": bool(noninferior_all.all(axis=None)) if not site_audit.empty else False,
        "n_not_evaluated_site_tasks": int((~evaluated).sum().sum()) if not site_audit.empty else 0,
        "guarded_sites": guarded_sites,
        "support_outcome_spearman": support_outcome_corr,
        "applicability_boundary": (
            "This audit separates train-side support diagnostics from holdout outcomes; "
            "guarded local-prior sites support non-inferiority claims but not unqualified model-transfer wins."
        ),
    }
    return DomainSupportResult(site_audit=site_audit, summary=summary)


def write_domain_support_outputs(result: DomainSupportResult, table_dir: Path, summary_path: Path | None = None) -> tuple[Path, Path]:
    import json

    table_dir.mkdir(parents=True, exist_ok=True)
    audit_path = table_dir / "domain_support_site_audit.csv"
    result.site_audit.to_csv(audit_path, index=False)
    out_summary = summary_path or table_dir / "domain_support_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path, out_summary
