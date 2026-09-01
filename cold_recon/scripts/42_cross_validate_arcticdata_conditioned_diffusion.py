from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cold_recon.utils.config import ensure_dirs, load_config


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "site"


def _eligible_sites(token_index: pd.DataFrame, min_eic_tokens: int, min_boreholes: int, max_span_m: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for site, group in token_index.groupby("site", sort=True):
        x_span = float(group["x"].max() - group["x"].min())
        y_span = float(group["y"].max() - group["y"].min())
        eic_tokens = int((group["type_id"].astype(int) == 1).sum())
        boreholes = int(group["borehole"].nunique())
        rows.append(
            {
                "site": str(site),
                "slug": _slug(str(site)),
                "n_tokens": int(len(group)),
                "eic_tokens": eic_tokens,
                "facies_tokens": int((group["type_id"].astype(int) == 0).sum()),
                "boreholes": boreholes,
                "x_span_m": x_span,
                "y_span_m": y_span,
                "max_span_m": max(x_span, y_span),
                "public_coord_tokens": int((group["coordinate_source"] == "public_lat_lon_site_local").sum()),
                "eligible": eic_tokens >= min_eic_tokens and boreholes >= min_boreholes and max(x_span, y_span) <= max_span_m,
            }
        )
    sites = pd.DataFrame(rows)
    sites["selection_score"] = sites["eic_tokens"] * 3 + sites["facies_tokens"] + sites["boreholes"] - 0.01 * sites["max_span_m"]
    return sites.sort_values(["eligible", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in metrics.columns
        if col not in {"model", "site", "source_metrics_csv", "source_predictions_csv"} and pd.api.types.is_numeric_dtype(metrics[col])
    ]
    rows: list[dict[str, object]] = []
    for model, group in metrics.groupby("model", sort=False):
        row: dict[str, object] = {"model": str(model), "n_sites": int(group["site"].nunique())}
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            if values.notna().any():
                row[f"{col}_mean"] = float(values.mean(skipna=True))
                row[f"{col}_median"] = float(values.median(skipna=True))
        rows.append(row)
    summary = pd.DataFrame(rows)
    knn = metrics[metrics["model"] == "SpatialDepthKNN"].set_index("site")
    global_mean = metrics[metrics["model"] == "GlobalMean"].set_index("site")
    idw = metrics[metrics["model"] == "SpatialDepthIDW"].set_index("site")

    def _win_rate(model_df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str, higher_is_better: bool) -> float:
        common = model_df.index.intersection(baseline_df.index)
        if not len(common):
            return float("nan")
        model_values = pd.to_numeric(model_df.loc[common, metric], errors="coerce")
        baseline_values = pd.to_numeric(baseline_df.loc[common, metric], errors="coerce")
        finite = model_values.notna() & baseline_values.notna()
        if not finite.any():
            return float("nan")
        if higher_is_better:
            return float((model_values[finite] > baseline_values[finite]).mean())
        return float((model_values[finite] < baseline_values[finite]).mean())

    def _noninferior_rate(model_df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str, higher_is_better: bool) -> float:
        common = model_df.index.intersection(baseline_df.index)
        if not len(common):
            return float("nan")
        model_values = pd.to_numeric(model_df.loc[common, metric], errors="coerce")
        baseline_values = pd.to_numeric(baseline_df.loc[common, metric], errors="coerce")
        finite = model_values.notna() & baseline_values.notna()
        if not finite.any():
            return float("nan")
        if higher_is_better:
            return float((model_values[finite] >= baseline_values[finite]).mean())
        return float((model_values[finite] <= baseline_values[finite]).mean())

    cold_models = [
        "COLDReconArcticDataHybridCalibrated",
        "COLDReconArcticDataAdaptiveHybrid",
        "COLDReconArcticDataWedgeRecallHead",
    ]
    for col in [
        "facies_win_rate_vs_spatial_knn",
        "facies_noninferior_rate_vs_spatial_knn",
        "wedge_win_rate_vs_spatial_knn",
        "wedge_noninferior_rate_vs_spatial_knn",
        "eic_rmse_win_rate_vs_global_mean",
        "eic_rmse_win_rate_vs_spatial_idw",
        "eic_rmse_win_rate_vs_best_simple_baseline",
        "eic_rmse_noninferior_rate_vs_best_simple_baseline",
        "high_eic_f1_win_rate_vs_spatial_idw",
        "high_eic_f1_noninferior_rate_vs_spatial_idw",
    ]:
        summary[col] = np.nan
    for model_name in cold_models:
        model_df = metrics[metrics["model"] == model_name].set_index("site")
        if model_df.empty:
            continue
        summary.loc[summary["model"] == model_name, "facies_win_rate_vs_spatial_knn"] = _win_rate(model_df, knn, "facies_accuracy", True)
        summary.loc[summary["model"] == model_name, "facies_noninferior_rate_vs_spatial_knn"] = _noninferior_rate(model_df, knn, "facies_accuracy", True)
        summary.loc[summary["model"] == model_name, "wedge_win_rate_vs_spatial_knn"] = _win_rate(model_df, knn, "wedge_ice_recall", True)
        summary.loc[summary["model"] == model_name, "wedge_noninferior_rate_vs_spatial_knn"] = _noninferior_rate(model_df, knn, "wedge_ice_recall", True)
        summary.loc[summary["model"] == model_name, "eic_rmse_win_rate_vs_global_mean"] = _win_rate(model_df, global_mean, "eic_rmse", False)
        summary.loc[summary["model"] == model_name, "eic_rmse_win_rate_vs_spatial_idw"] = _win_rate(model_df, idw, "eic_rmse", False)
        summary.loc[summary["model"] == model_name, "high_eic_f1_win_rate_vs_spatial_idw"] = _win_rate(model_df, idw, "high_eic_f1", True)
        summary.loc[summary["model"] == model_name, "high_eic_f1_noninferior_rate_vs_spatial_idw"] = _noninferior_rate(model_df, idw, "high_eic_f1", True)
        common = model_df.index.intersection(global_mean.index).intersection(idw.index)
        if len(common):
            model_values = pd.to_numeric(model_df.loc[common, "eic_rmse"], errors="coerce")
            global_values = pd.to_numeric(global_mean.loc[common, "eic_rmse"], errors="coerce")
            idw_values = pd.to_numeric(idw.loc[common, "eic_rmse"], errors="coerce")
            finite = model_values.notna() & global_values.notna() & idw_values.notna()
            if finite.any():
                best_simple = np.minimum(global_values[finite].to_numpy(dtype=float), idw_values[finite].to_numpy(dtype=float))
                model_array = model_values[finite].to_numpy(dtype=float)
                summary.loc[summary["model"] == model_name, "eic_rmse_win_rate_vs_best_simple_baseline"] = float((model_array < best_simple).mean())
                summary.loc[summary["model"] == model_name, "eic_rmse_noninferior_rate_vs_best_simple_baseline"] = float((model_array <= best_simple * 1.02).mean())

    hybrid = summary["model"] == "COLDReconArcticDataHybridCalibrated"
    summary["hybrid_facies_win_rate_vs_spatial_knn"] = np.nan
    summary["hybrid_wedge_win_rate_vs_spatial_knn"] = np.nan
    summary["hybrid_eic_rmse_win_rate_vs_global_mean"] = np.nan
    summary.loc[hybrid, "hybrid_facies_win_rate_vs_spatial_knn"] = summary.loc[hybrid, "facies_win_rate_vs_spatial_knn"]
    summary.loc[hybrid, "hybrid_wedge_win_rate_vs_spatial_knn"] = summary.loc[hybrid, "wedge_win_rate_vs_spatial_knn"]
    summary.loc[hybrid, "hybrid_eic_rmse_win_rate_vs_global_mean"] = summary.loc[hybrid, "eic_rmse_win_rate_vs_global_mean"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--token-index", default="outputs/tables/arcticdata_cryostratigraphy_token_index.csv")
    parser.add_argument("--max-sites", type=int, default=5)
    parser.add_argument("--min-eic-tokens", type=int, default=20)
    parser.add_argument("--min-boreholes", type=int, default=4)
    parser.add_argument("--max-site-span-m", type=float, default=5000.0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--target-shape", default="64,64,48")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    table_dir = Path(config["paths"]["tables_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    token_index = pd.read_csv(args.token_index)
    sites = _eligible_sites(token_index, int(args.min_eic_tokens), int(args.min_boreholes), float(args.max_site_span_m))
    selected = sites[sites["eligible"]].head(int(args.max_sites)).copy()
    if selected.empty:
        raise ValueError("No eligible compact ArcticData sites found for multi-site conditioned diffusion")

    site_path = table_dir / "arcticdata_conditioned_diffusion_multisite_sites.csv"
    selected.to_csv(site_path, index=False)
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        site = str(row["site"])
        prefix = f"arcticdata_conditioned_diffusion_{row['slug']}"
        cmd = [
            sys.executable,
            "-m",
            "cold_recon.scripts.41_condition_arcticdata_diffusion",
            "--config",
            args.config,
            "--site",
            site,
            "--output-prefix",
            prefix,
            "--samples",
            str(int(args.samples)),
            "--target-shape",
            str(args.target_shape),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        print(f"running site={site} prefix={prefix}")
        subprocess.run(cmd, check=True)
        metrics_path = table_dir / f"{prefix}_metrics.csv"
        predictions_path = table_dir / f"{prefix}_holdout_predictions.csv"
        metrics = pd.read_csv(metrics_path)
        metrics["source_metrics_csv"] = metrics_path.as_posix()
        metrics["source_predictions_csv"] = predictions_path.as_posix()
        all_metrics.append(metrics)
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            predictions["site"] = site
            predictions["source_predictions_csv"] = predictions_path.as_posix()
            all_predictions.append(predictions)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    predictions_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    summary = _summarize(metrics_all)
    metrics_path = table_dir / "arcticdata_conditioned_diffusion_multisite_metrics.csv"
    predictions_path = table_dir / "arcticdata_conditioned_diffusion_multisite_predictions.csv"
    summary_path = table_dir / "arcticdata_conditioned_diffusion_multisite_summary.csv"
    metrics_all.to_csv(metrics_path, index=False)
    predictions_all.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"sites={site_path}")
    print(f"metrics={metrics_path}")
    print(f"predictions={predictions_path}")
    print(f"summary={summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
