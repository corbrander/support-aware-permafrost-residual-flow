from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cold_recon.data.data_schema import FACIES_NAMES, OBS_TYPE_NAMES, load_sample_npz
from cold_recon.evaluation.metrics import alt_from_temperature
from cold_recon.evaluation.physics_consistency import physics_consistency_metrics, sample_truth_fields
from cold_recon.physics.settlement import settlement_potential_numpy


FACIES_FRACTION_PREFIX = "facies_fraction_"
OBS_COUNT_PREFIX = "obs_count_"


def summarize_synthetic_sample(sample: dict[str, Any], sample_id: str | None = None) -> dict[str, float | str]:
    fields = sample["fields"]
    grid = sample["grid"]
    metadata = sample.get("metadata", {})
    facies = np.asarray(fields["facies"], dtype=np.int16)
    eic = np.asarray(fields["eic"], dtype=np.float32)
    temperature = np.asarray(fields["temperature"], dtype=np.float32)
    unfrozen = np.asarray(fields["unfrozen_water"], dtype=np.float32)
    log_rho = np.log(np.maximum(np.asarray(fields["resistivity"], dtype=np.float32), 1.0))
    alt = alt_from_temperature(temperature, np.asarray(grid["z"], dtype=np.float32))
    dz = float(grid["dz"])
    settlement = settlement_potential_numpy(eic, temperature + 2.0, dz)
    spacing = tuple(float(x) for x in grid.get("spacing", (grid["dx"], grid["dy"], grid["dz"])))
    physics = physics_consistency_metrics(sample_truth_fields(sample), spacing=spacing)
    obs = sample["observations"]
    row: dict[str, float | str] = {
        "sample_id": sample_id or str(metadata.get("site_id", "")),
        "seed": float(metadata.get("seed", np.nan)),
        "n_observations": float(obs.n_obs),
        "eic_mean": float(np.mean(eic)),
        "eic_p95": float(np.percentile(eic, 95.0)),
        "ice_rich_fraction": float(np.mean(eic > 0.30)),
        "temperature_mean": float(np.mean(temperature)),
        "temperature_min": float(np.min(temperature)),
        "temperature_max": float(np.max(temperature)),
        "active_layer_mean": float(np.mean(alt)),
        "active_layer_p95": float(np.percentile(alt, 95.0)),
        "unfrozen_water_mean": float(np.mean(unfrozen)),
        "log_resistivity_mean": float(np.mean(log_rho)),
        "log_resistivity_std": float(np.std(log_rho)),
        "settlement_potential_mean": float(np.mean(settlement)),
        "settlement_potential_p95": float(np.percentile(settlement, 95.0)),
        "truth_heat_residual_rmse": float(physics["heat_residual_rmse"]),
        "truth_unfrozen_water_empirical_mae": float(physics["unfrozen_water_empirical_mae"]),
        "truth_log_resistivity_empirical_mae": float(physics["log_resistivity_empirical_mae"]),
    }
    for facies_id, name in FACIES_NAMES.items():
        row[f"{FACIES_FRACTION_PREFIX}{name}"] = float(np.mean(facies == facies_id))
    for type_id, name in OBS_TYPE_NAMES.items():
        row[f"{OBS_COUNT_PREFIX}{name}"] = float(np.sum(obs.type_ids == type_id))
    return row


def summarize_synthetic_paths(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        sample = load_sample_npz(path)
        rows.append(summarize_synthetic_sample(sample, sample_id=path.stem))
    return pd.DataFrame(rows)


def aggregate_synthetic_benchmark(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    skip = {"sample_id"}
    numeric_cols = [col for col in rows.columns if col not in skip and pd.api.types.is_numeric_dtype(rows[col])]
    records = []
    for col in numeric_cols:
        values = rows[col].dropna().astype(float)
        if values.empty:
            continue
        records.append(
            {
                "metric": col,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "n": int(values.shape[0]),
            }
        )
    return pd.DataFrame(records)


def write_synthetic_benchmark_tables(rows: pd.DataFrame, summary: pd.DataFrame, table_dir: Path) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    detail_path = table_dir / "synthetic_ensemble_benchmark.csv"
    summary_path = table_dir / "synthetic_ensemble_summary.csv"
    rows.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {"detail": detail_path, "summary": summary_path}
