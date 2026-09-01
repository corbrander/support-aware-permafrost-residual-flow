from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable
from cold_recon.evaluation.metrics import alt_from_temperature


SOURCE_SPECS: tuple[tuple[str, int, str], ...] = (
    ("borehole_facies", OBS_TYPES["borehole_facies"], "facies"),
    ("borehole_eic", OBS_TYPES["borehole_eic"], "eic"),
    ("borehole_temperature", OBS_TYPES["borehole_temperature"], "temperature"),
    ("ert_log_resistivity", OBS_TYPES["ert_log_resistivity"], "log_resistivity"),
    ("nmr_unfrozen_water", OBS_TYPES["nmr_unfrozen_water"], "unfrozen_water"),
    ("alt", OBS_TYPES["alt"], "active_layer_thickness"),
)


def nearest_grid_indices(coords: np.ndarray, grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(grid.get("grid_x", grid.get("x")), dtype=np.float32)
    y = np.asarray(grid.get("grid_y", grid.get("y")), dtype=np.float32)
    z = np.asarray(grid.get("grid_z", grid.get("z")), dtype=np.float32)
    ix = np.abs(x[None, :] - coords[:, 0:1]).argmin(axis=1)
    iy = np.abs(y[None, :] - coords[:, 1:2]).argmin(axis=1)
    iz = np.abs(z[None, :] - coords[:, 2:3]).argmin(axis=1)
    return ix, iy, iz


def canonical_prediction_fields(prediction: dict[str, np.ndarray], truth_fields: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    aliases = {
        "facies": ("facies_mode", "facies"),
        "eic": ("eic_mean", "eic"),
        "temperature": ("temperature_mean", "temperature"),
        "unfrozen_water": ("unfrozen_water_mean", "unfrozen_water"),
        "log_resistivity": ("log_resistivity_mean", "log_resistivity"),
    }
    for out_key, names in aliases.items():
        for name in names:
            if name in prediction:
                fields[out_key] = np.asarray(prediction[name])
                break
    if "log_resistivity" not in fields and "resistivity" in prediction:
        fields["log_resistivity"] = np.log(np.maximum(np.asarray(prediction["resistivity"], dtype=np.float32), 1.0))
    if truth_fields is not None:
        for key in ("facies", "eic", "temperature", "unfrozen_water"):
            if key not in fields and key in truth_fields:
                fields[key] = np.asarray(truth_fields[key])
        if "log_resistivity" not in fields and "resistivity" in truth_fields:
            fields["log_resistivity"] = np.log(np.maximum(np.asarray(truth_fields["resistivity"], dtype=np.float32), 1.0))
    return fields


def predicted_observation_values(
    fields: dict[str, np.ndarray],
    grid: dict[str, Any],
    observations: ObservationTable,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = next((item for item in SOURCE_SPECS if item[0] == source_name), None)
    if spec is None:
        raise KeyError(f"Unknown source_name={source_name}")
    _, type_id, field_name = spec
    mask = observations.type_ids == type_id
    obs_idx = np.where(mask)[0]
    if obs_idx.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    coords = observations.coords[obs_idx]
    ix, iy, iz = nearest_grid_indices(coords, grid)
    if field_name == "active_layer_thickness":
        z = np.asarray(grid.get("grid_z", grid.get("z")), dtype=np.float32)
        pred_grid = alt_from_temperature(fields["temperature"], z)
        pred = pred_grid[ix, iy]
    else:
        pred_grid = fields[field_name]
        pred = pred_grid[ix, iy, iz]
    return np.asarray(pred, dtype=np.float32), observations.values[obs_idx], observations.sigma[obs_idx]


def evaluate_observation_consistency_by_source(
    prediction: dict[str, np.ndarray],
    sample: dict[str, Any],
    model_name: str,
) -> list[dict[str, float | str]]:
    fields = canonical_prediction_fields(prediction, sample.get("fields"))
    rows: list[dict[str, float | str]] = []
    for source_name, _, _ in SOURCE_SPECS:
        if source_name == "borehole_facies" and "facies" not in fields:
            continue
        if source_name != "borehole_facies":
            required = "temperature" if source_name == "alt" else next(item[2] for item in SOURCE_SPECS if item[0] == source_name)
            if required not in fields:
                continue
        pred, obs, sigma = predicted_observation_values(fields, sample["grid"], sample["observations"], source_name)
        if pred.size == 0:
            continue
        row: dict[str, float | str] = {"model": model_name, "source": source_name, "n": float(pred.size)}
        if source_name == "borehole_facies":
            target = obs.astype(np.int64)
            pred_cls = np.rint(pred).astype(np.int64)
            row["accuracy"] = float(np.mean(pred_cls == target))
            row["error_rate"] = float(1.0 - row["accuracy"])
        else:
            err = pred - obs
            sigma_safe = np.where(sigma > 0.0, sigma, np.nan)
            row["bias"] = float(np.mean(err))
            row["mae"] = float(np.mean(np.abs(err)))
            row["rmse"] = float(np.sqrt(np.mean(err**2)))
            row["normalized_rmse"] = float(np.sqrt(np.nanmean((err / sigma_safe) ** 2))) if np.any(sigma > 0.0) else float("nan")
        rows.append(row)
    return rows


def observation_consistency_table(predictions: list[tuple[str, dict[str, np.ndarray]]], sample: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name, prediction in predictions:
        rows.extend(evaluate_observation_consistency_by_source(prediction, sample, model_name))
    return pd.DataFrame(rows)
