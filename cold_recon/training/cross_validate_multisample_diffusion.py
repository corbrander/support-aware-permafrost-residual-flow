from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np

from cold_recon.training.train_multisample_diffusion import synthetic_sample_paths, train_multisample_diffusion


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def aggregate_cv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        numeric_cols = []
        for key in model_rows[0]:
            if key in {"model", "holdout_sample", "fold"}:
                continue
            if any(_as_float(row.get(key)) is not None for row in model_rows):
                numeric_cols.append(key)
        for key in numeric_cols:
            values = [_as_float(row.get(key)) for row in model_rows]
            finite = np.asarray([v for v in values if v is not None], dtype=np.float64)
            if finite.size == 0:
                continue
            summary.append(
                {
                    "model": model,
                    "metric": key,
                    "mean": float(finite.mean()),
                    "std": float(finite.std(ddof=0)),
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "n": int(finite.size),
                }
            )
    return summary


def paired_improvement_rows(rows: list[dict[str, Any]], baseline: str = "IDWObservationProxy", candidate: str = "COLDReconMultiSampleDiffusion") -> list[dict[str, Any]]:
    by_fold_model = {(str(row.get("fold")), str(row.get("model"))): row for row in rows}
    folds = sorted({str(row.get("fold")) for row in rows})
    metrics = [
        "mean_iou",
        "iou_6",
        "eic_rmse",
        "ice_rich_recall",
        "temperature_rmse",
        "unfrozen_water_rmse",
        "log_resistivity_rmse",
        "borehole_facies_accuracy",
    ]
    out: list[dict[str, Any]] = []
    for fold in folds:
        base = by_fold_model.get((fold, baseline))
        cand = by_fold_model.get((fold, candidate))
        if base is None or cand is None:
            continue
        row: dict[str, Any] = {"fold": fold, "holdout_sample": cand.get("holdout_sample", "")}
        for metric in metrics:
            b = _as_float(base.get(metric))
            c = _as_float(cand.get(metric))
            if b is None or c is None:
                continue
            row[f"{metric}_candidate"] = c
            row[f"{metric}_baseline"] = b
            row[f"{metric}_delta"] = c - b
            if metric.endswith("_rmse"):
                row[f"{metric}_reduction"] = b - c
        out.append(row)
    return out


def cross_validate_multisample_diffusion(
    config: dict,
    n_samples: int | None = None,
    folds: list[int] | None = None,
    epochs: int | None = None,
    samples: int | None = None,
    max_condition_tokens: int | None = None,
    device: str | None = None,
) -> dict[str, Path | list[dict[str, Any]]]:
    cfg = config.get("multisample_diffusion", {})
    paths = synthetic_sample_paths(config, n_samples=n_samples if n_samples is not None else cfg.get("n_samples"))
    fold_indices = folds if folds is not None else list(range(len(paths)))
    if not fold_indices:
        raise ValueError("At least one fold is required")
    base_pred = Path(cfg.get("posterior_path", Path(config["paths"]["predictions_dir"]) / "multisample_diffusion_holdout.npz"))
    base_ckpt = Path(cfg.get("checkpoint", Path(config["paths"]["checkpoints_dir"]) / "multisample_latent_diffusion.pt"))
    base_metrics = Path(cfg.get("metrics_path", Path(config["paths"]["tables_dir"]) / "multisample_diffusion_holdout_metrics.csv"))
    base_history = Path(cfg.get("history_path", Path(config["paths"]["tables_dir"]) / "multisample_diffusion_history.csv"))
    detail_rows: list[dict[str, Any]] = []
    for fold in fold_indices:
        fold = int(fold)
        fold_cfg = copy.deepcopy(config)
        fold_tag = f"fold_{fold:02d}"
        fold_ms = fold_cfg.setdefault("multisample_diffusion", {})
        fold_ms["checkpoint"] = str(base_ckpt.with_name(f"{base_ckpt.stem}_{fold_tag}{base_ckpt.suffix}"))
        fold_ms["posterior_path"] = str(base_pred.with_name(f"{base_pred.stem}_{fold_tag}{base_pred.suffix}"))
        fold_ms["metrics_path"] = str(base_metrics.with_name(f"{base_metrics.stem}_{fold_tag}{base_metrics.suffix}"))
        fold_ms["history_path"] = str(base_history.with_name(f"{base_history.stem}_{fold_tag}{base_history.suffix}"))
        result = train_multisample_diffusion(
            fold_cfg,
            sample_paths=paths,
            holdout_index=fold,
            epochs=epochs,
            samples=samples,
            max_condition_tokens=max_condition_tokens,
            device=device,
        )
        for row in _read_csv_rows(Path(result["metrics_path"])):
            detail_rows.append({"fold": fold, **row})
    summary_rows = aggregate_cv_rows(detail_rows)
    improvement_rows = paired_improvement_rows(detail_rows)
    table_dir = Path(config["paths"]["tables_dir"])
    detail_path = Path(cfg.get("cv_detail_path", table_dir / "multisample_diffusion_cv_metrics.csv"))
    summary_path = Path(cfg.get("cv_summary_path", table_dir / "multisample_diffusion_cv_summary.csv"))
    improvement_path = Path(cfg.get("cv_improvement_path", table_dir / "multisample_diffusion_cv_improvement.csv"))
    _write_csv(detail_path, detail_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(improvement_path, improvement_rows)
    return {
        "detail_path": detail_path,
        "summary_path": summary_path,
        "improvement_path": improvement_path,
        "detail_rows": detail_rows,
        "summary_rows": summary_rows,
        "improvement_rows": improvement_rows,
    }
