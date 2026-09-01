from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FootprintModel:
    model: str
    prediction: str | None = None
    checkpoint: str | None = None
    params_from: str | None = None
    history: str | None = None
    role: str = ""
    note: str = ""


DEFAULT_MODELS: tuple[FootprintModel, ...] = (
    FootprintModel("IDW", prediction="baseline_idw.npz", role="deterministic baseline"),
    FootprintModel("RandomForest", prediction="baseline_random_forest.npz", role="deterministic baseline"),
    FootprintModel("GradientBoosting", prediction="baseline_gradient_boosting.npz", role="deterministic baseline"),
    FootprintModel("KrigingGPR", prediction="baseline_kriging.npz", role="deterministic baseline"),
    FootprintModel("SparseUNet3D", prediction="baseline_unet3d.npz", params_from="SparseUNet3D", history="baseline_unet3d_history.csv", role="deterministic neural baseline"),
    FootprintModel("COLDReconImplicit", prediction="implicit_prediction.npz", params_from="COLDReconImplicit", role="coordinate implicit field"),
    FootprintModel("COLDReconLatentDiffusion", prediction="diffusion_posterior.npz", params_from="COLDReconLatentDiffusion", history="diffusion_posterior_history.csv", role="posterior generator"),
    FootprintModel("COLDReconFNOOperatorDiffusion", prediction="fno_operator_diffusion_posterior.npz", params_from="COLDReconFNOOperatorDiffusion", history="fno_operator_diffusion_history.csv", role="operator posterior generator"),
    FootprintModel("COLDReconRectifiedFlow", prediction="rectified_flow_posterior.npz", params_from="COLDReconRectifiedFlow", history="rectified_flow_history.csv", role="flow posterior generator"),
    FootprintModel("COLDReconLatentDiffusionPhysicsTrained", prediction="diffusion_posterior_physics_trained.npz", checkpoint="outputs/checkpoints/latent_diffusion_physics_trained.pt", params_from="COLDReconLatentDiffusion", history="diffusion_physics_trained_history.csv", role="physics-trained posterior generator"),
    FootprintModel("COLDReconLatentDiffusionRareFaciesHybrid", prediction="diffusion_posterior_rare_facies_hybrid.npz", params_from="COLDReconLatentDiffusion", role="post-hoc rare-facies operating point", note="inherits physics-trained diffusion and implicit-proposal artifacts"),
    FootprintModel("COLDReconLatentDiffusionPhysicsGuided", prediction="diffusion_posterior_physics_guided.npz", params_from="COLDReconLatentDiffusion", history="diffusion_physics_guided_history.csv", role="latent-guided posterior generator"),
    FootprintModel("COLDReconLatentDiffusionPhysicsRefined", prediction="diffusion_posterior_physics_refined.npz", params_from="COLDReconLatentDiffusion", role="post-hoc physics projection"),
)


METRIC_TABLES: tuple[str, ...] = (
    "baseline_metrics.csv",
    "baseline_unet3d_metrics.csv",
    "implicit_metrics.csv",
    "diffusion_posterior_metrics.csv",
    "fno_operator_diffusion_metrics.csv",
    "rectified_flow_metrics.csv",
    "diffusion_physics_trained_metrics.csv",
    "diffusion_rare_facies_hybrid_metrics.csv",
    "diffusion_physics_guided_metrics.csv",
    "diffusion_physics_refined_metrics.csv",
)


def _file_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return float("nan")
    return float(path.stat().st_size / (1024.0 * 1024.0))


def _history_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"training_epochs": float("nan"), "final_train_loss": float("nan"), "history_rows": 0.0}
    df = pd.read_csv(path)
    if df.empty:
        return {"training_epochs": float("nan"), "final_train_loss": float("nan"), "history_rows": 0.0}
    epoch = pd.to_numeric(df["epoch"], errors="coerce") if "epoch" in df.columns else pd.Series(dtype=float)
    loss = pd.to_numeric(df["loss"], errors="coerce") if "loss" in df.columns else pd.Series(dtype=float)
    return {
        "training_epochs": float(np.nanmax(epoch) + 1.0) if not epoch.empty and np.isfinite(np.nanmax(epoch)) else float(len(df)),
        "final_train_loss": float(loss.dropna().iloc[-1]) if not loss.dropna().empty else float("nan"),
        "history_rows": float(len(df)),
    }


def _npz_summary(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {
            "posterior_samples": float("nan"),
            "prediction_arrays": 0.0,
            "largest_array": "",
            "largest_array_mb": float("nan"),
        }
    with np.load(path, allow_pickle=False) as data:
        samples = []
        largest_name = ""
        largest_bytes = -1
        for name in data.files:
            arr = data[name]
            if name.endswith("_samples") and arr.ndim >= 4:
                samples.append(float(arr.shape[0]))
            if arr.nbytes > largest_bytes:
                largest_name = name
                largest_bytes = int(arr.nbytes)
        return {
            "posterior_samples": float(np.nanmax(samples)) if samples else float("nan"),
            "prediction_arrays": float(len(data.files)),
            "largest_array": largest_name,
            "largest_array_mb": float(largest_bytes / (1024.0 * 1024.0)) if largest_bytes >= 0 else float("nan"),
        }


def build_computational_footprint(
    table_dir: Path,
    prediction_dir: Path,
    root: Path,
    models: Iterable[FootprintModel] = DEFAULT_MODELS,
) -> pd.DataFrame:
    root = root.resolve()
    table_dir = (root / table_dir).resolve() if not table_dir.is_absolute() else table_dir.resolve()
    prediction_dir = (root / prediction_dir).resolve() if not prediction_dir.is_absolute() else prediction_dir.resolve()
    architecture_path = table_dir / "model_architecture_summary.csv"
    architecture = pd.read_csv(architecture_path) if architecture_path.exists() else pd.DataFrame()
    metrics = _read_model_metrics(table_dir)

    arch_by_model = {str(row["model"]): row for _, row in architecture.iterrows()} if "model" in architecture.columns else {}
    rows: list[dict[str, float | str]] = []
    for spec in models:
        arch = arch_by_model.get(spec.params_from or spec.model)
        metric_rows = metrics[metrics["model"].astype(str).eq(spec.model)] if "model" in metrics.columns else pd.DataFrame()
        metric = metric_rows.iloc[0] if not metric_rows.empty else None
        prediction_path = prediction_dir / spec.prediction if spec.prediction else None
        if spec.checkpoint:
            checkpoint_path = root / spec.checkpoint
        elif arch is not None and "checkpoint" in arch.index and pd.notna(arch["checkpoint"]):
            checkpoint_path = root / str(arch["checkpoint"])
        else:
            checkpoint_path = None

        row: dict[str, float | str] = {
            "model": spec.model,
            "role": spec.role,
            "parameter_source_model": spec.params_from or spec.model,
            "prediction_file": str(prediction_path.resolve().relative_to(root)) if prediction_path and prediction_path.exists() else "",
            "checkpoint_file": str(checkpoint_path.resolve().relative_to(root)) if checkpoint_path and checkpoint_path.exists() else "",
            "component_params": float(arch["component_params"]) if arch is not None and "component_params" in arch.index and pd.notna(arch["component_params"]) else float("nan"),
            "obs_encoder_params": float(arch["obs_encoder_params"]) if arch is not None and "obs_encoder_params" in arch.index and pd.notna(arch["obs_encoder_params"]) else float("nan"),
            "total_params": float(arch["total_params"]) if arch is not None and "total_params" in arch.index and pd.notna(arch["total_params"]) else float("nan"),
            "latent_shape": str(arch["latent_shape"]) if arch is not None and "latent_shape" in arch.index and pd.notna(arch["latent_shape"]) else "",
            "checkpoint_mb": _file_mb(checkpoint_path),
            "prediction_mb": _file_mb(prediction_path),
            "mean_iou": float(metric["mean_iou"]) if metric is not None and "mean_iou" in metric.index and pd.notna(metric["mean_iou"]) else float("nan"),
            "eic_rmse": float(metric["eic_rmse"]) if metric is not None and "eic_rmse" in metric.index and pd.notna(metric["eic_rmse"]) else float("nan"),
            "unfrozen_water_rmse": float(metric["unfrozen_water_rmse"]) if metric is not None and "unfrozen_water_rmse" in metric.index and pd.notna(metric["unfrozen_water_rmse"]) else float("nan"),
            "note": spec.note,
        }
        row.update(_npz_summary(prediction_path) if prediction_path else {})
        if spec.history:
            row.update(_history_summary(table_dir / spec.history))
        else:
            row.update({"training_epochs": float("nan"), "final_train_loss": float("nan"), "history_rows": 0.0})
        rows.append(row)

    out = pd.DataFrame(rows)
    out["total_params_m"] = out["total_params"] / 1_000_000.0
    out["artifact_mb"] = out["checkpoint_mb"].fillna(0.0) + out["prediction_mb"].fillna(0.0)
    out.loc[out["checkpoint_mb"].isna() & out["prediction_mb"].isna(), "artifact_mb"] = float("nan")
    return out


def _read_model_metrics(table_dir: Path) -> pd.DataFrame:
    metrics_path = table_dir / "model_comparison.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    frames: list[pd.DataFrame] = []
    for name in METRIC_TABLES:
        path = table_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if name == "diffusion_rare_facies_hybrid_metrics.csv" and "model" in df.columns:
            df = df[df["model"].astype(str).eq("COLDReconLatentDiffusionRareFaciesHybrid")].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
