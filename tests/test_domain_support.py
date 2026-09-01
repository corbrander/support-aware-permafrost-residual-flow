from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cold_recon.evaluation.domain_support import build_domain_support_audit, write_domain_support_outputs


def test_domain_support_separates_guarded_sites_from_model_supported_sites(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "site": "large site",
                "train_n": 300,
                "holdout_n": 50,
                "train_boreholes": 40,
                "adaptive_eic_train_observations": 300,
                "adaptive_eic_train_groups": 25,
                "adaptive_eic_method": "spatial_raw_guarded_ensemble",
                "adaptive_eic_transfer_guard_reason": "cv_selected",
                "facies_win": True,
                "facies_noninferior": True,
                "eic_win_vs_best_simple": True,
                "eic_noninferior_vs_best_simple": True,
                "high_eic_f1_win_vs_spatial_idw": True,
                "high_eic_noninferior_vs_spatial_idw": True,
                "wedge_recall_win": True,
                "wedge_recall_noninferior": True,
                "eic_rmse_reduction_vs_best_simple": 0.05,
            },
            {
                "site": "compact site",
                "train_n": 120,
                "holdout_n": 40,
                "train_boreholes": 10,
                "adaptive_eic_train_observations": 120,
                "adaptive_eic_train_groups": 8,
                "adaptive_eic_method": "transfer_idw_adapter",
                "adaptive_eic_transfer_guard_reason": "compact_site_spatial_guard",
                "facies_win": False,
                "facies_noninferior": True,
                "eic_win_vs_best_simple": False,
                "eic_noninferior_vs_best_simple": True,
                "high_eic_f1_win_vs_spatial_idw": False,
                "high_eic_noninferior_vs_spatial_idw": True,
                "wedge_recall_win": False,
                "wedge_recall_noninferior": True,
                "eic_rmse_reduction_vs_best_simple": 0.0,
            },
        ]
    )
    result = build_domain_support_audit(df)
    audit = result.site_audit

    assert result.summary["n_sites"] == 2
    assert result.summary["n_guarded_local_prior"] == 1
    assert result.summary["all_sites_noninferior_all_evaluated_tasks"] is True
    compact = audit[audit["site"].eq("compact site")].iloc[0]
    assert compact["applicability_class"] == "guarded local-prior"
    assert compact["eic_outcome"] == "noninferior"

    audit_path, summary_path = write_domain_support_outputs(result, tmp_path)
    assert audit_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_guarded_local_prior"] == 1
