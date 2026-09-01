from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WEDGE_CLASS_ID = 6
RECALL_MODEL = "COLDReconArcticDataWedgeRecallHead"
BASELINE_MODEL = "SpatialDepthKNN"


def normalize_wedge_model_names(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.copy()


def slugify_site(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "site"


def binary_wedge_metrics(observed_wedge: np.ndarray, predicted_wedge: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed_wedge, dtype=bool)
    predicted = np.asarray(predicted_wedge, dtype=bool)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted arrays must have the same shape")
    tp = int(np.sum(observed & predicted))
    fp = int(np.sum(~observed & predicted))
    fn = int(np.sum(observed & ~predicted))
    tn = int(np.sum(~observed & ~predicted))
    n = int(observed.size)
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else np.nan
    beta = 2.0
    f2 = (
        (1.0 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and beta * beta * precision + recall > 0
        else np.nan
    )
    return {
        "n": float(n),
        "positives": float(np.sum(observed)),
        "predicted_positives": float(np.sum(predicted)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "accuracy": float((tp + tn) / n) if n else np.nan,
        "recall": float(recall) if np.isfinite(recall) else np.nan,
        "precision": float(precision) if np.isfinite(precision) else np.nan,
        "specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        "false_positive_rate": float(fpr) if np.isfinite(fpr) else np.nan,
        "f1": float(f1) if np.isfinite(f1) else np.nan,
        "f2": float(f2) if np.isfinite(f2) else np.nan,
        "prevalence": float(np.mean(observed)) if n else np.nan,
        "positive_prediction_rate": float(np.mean(predicted)) if n else np.nan,
    }


def _nearest_indices(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    return np.abs(grid[:, None] - values[None, :]).argmin(axis=0)


def _prefix_from_group(group: pd.DataFrame, site: str) -> str:
    if "source_predictions_csv" in group.columns:
        values = group["source_predictions_csv"].dropna().astype(str)
        if len(values):
            stem = Path(values.iloc[0]).stem
            return stem.removesuffix("_holdout_predictions")
    return f"arcticdata_conditioned_diffusion_{slugify_site(site)}"


def load_wedge_probability_scores(
    predictions: pd.DataFrame,
    prediction_dir: Path,
    recall_model: str = RECALL_MODEL,
) -> pd.DataFrame:
    predictions = normalize_wedge_model_names(predictions)
    rows = predictions[predictions["model"].astype(str).eq(recall_model)].copy()
    if rows.empty:
        raise ValueError(f"No prediction rows found for {recall_model}")
    out_frames: list[pd.DataFrame] = []
    for site, group in rows.groupby("site", sort=True):
        prefix = _prefix_from_group(group, str(site))
        posterior_path = prediction_dir / f"{prefix}.npz"
        if not posterior_path.exists():
            raise FileNotFoundError(f"Missing posterior NPZ for site {site}: {posterior_path}")
        posterior = np.load(posterior_path, allow_pickle=False)
        if "facies_probability" not in posterior.files:
            raise KeyError(f"{posterior_path} does not contain facies_probability")
        coords = group[["x", "y", "z"]].astype(float).to_numpy()
        ix = _nearest_indices(posterior["grid_x"], coords[:, 0])
        iy = _nearest_indices(posterior["grid_y"], coords[:, 1])
        iz = _nearest_indices(posterior["grid_z"], coords[:, 2])
        probability = np.asarray(posterior["facies_probability"][ix, iy, iz, WEDGE_CLASS_ID], dtype=float)
        frame = pd.DataFrame(
            {
                "site": str(site),
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "observed_class": pd.to_numeric(group["observed"], errors="coerce").round().astype("Int64").astype(float),
                "observed_wedge": pd.to_numeric(group["observed"], errors="coerce").round().astype(int).to_numpy() == WEDGE_CLASS_ID,
                "current_recall_head_class": pd.to_numeric(group["predicted"], errors="coerce").round().astype("Int64").astype(float),
                "current_recall_head_wedge": pd.to_numeric(group["predicted"], errors="coerce").round().astype(int).to_numpy() == WEDGE_CLASS_ID,
                "wedge_probability": probability,
                "posterior_npz": posterior_path.as_posix(),
            }
        )
        out_frames.append(frame)
    return pd.concat(out_frames, ignore_index=True)


def threshold_curve(
    scores: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
    group_by_site: bool = True,
) -> pd.DataFrame:
    thresholds_arr = np.asarray(list(thresholds) if thresholds is not None else np.linspace(0.0, 0.95, 20), dtype=float)
    rows: list[dict[str, object]] = []

    def add_rows(scope: str, frame: pd.DataFrame) -> None:
        observed = frame["observed_wedge"].astype(bool).to_numpy()
        probability = frame["wedge_probability"].astype(float).to_numpy()
        for threshold in thresholds_arr:
            metrics = binary_wedge_metrics(observed, probability >= float(threshold))
            rows.append({"scope": scope, "threshold": float(threshold), **metrics})

    add_rows("all_sites_pooled", scores)
    if group_by_site and "site" in scores.columns:
        for site, group in scores.groupby("site", sort=True):
            add_rows(str(site), group)
    return pd.DataFrame(rows)


def _site_mean_metrics_from_threshold(scores: pd.DataFrame, threshold: float) -> dict[str, float]:
    rows = []
    for _, group in scores.groupby("site", sort=True):
        rows.append(binary_wedge_metrics(group["observed_wedge"].astype(bool).to_numpy(), group["wedge_probability"].astype(float).to_numpy() >= float(threshold)))
    metrics = pd.DataFrame(rows)
    return {f"mean_site_{col}": float(pd.to_numeric(metrics[col], errors="coerce").mean(skipna=True)) for col in metrics.columns if col not in {"n", "tp", "fp", "fn", "tn"}}


def _site_mean_metrics_from_predictions(frame: pd.DataFrame) -> dict[str, float]:
    rows = []
    for _, group in frame.groupby("site", sort=True):
        observed = pd.to_numeric(group["observed"], errors="coerce").round().astype(int).to_numpy() == WEDGE_CLASS_ID
        predicted = pd.to_numeric(group["predicted"], errors="coerce").round().astype(int).to_numpy() == WEDGE_CLASS_ID
        rows.append(binary_wedge_metrics(observed, predicted))
    metrics = pd.DataFrame(rows)
    return {f"mean_site_{col}": float(pd.to_numeric(metrics[col], errors="coerce").mean(skipna=True)) for col in metrics.columns if col not in {"n", "tp", "fp", "fn", "tn"}}


def _prediction_point(predictions: pd.DataFrame, model: str, label: str) -> dict[str, object]:
    frame = predictions[predictions["model"].astype(str).eq(model)].copy()
    if frame.empty:
        raise ValueError(f"No prediction rows found for {model}")
    observed = pd.to_numeric(frame["observed"], errors="coerce").round().astype(int).to_numpy() == WEDGE_CLASS_ID
    predicted = pd.to_numeric(frame["predicted"], errors="coerce").round().astype(int).to_numpy() == WEDGE_CLASS_ID
    return {
        "operating_point": label,
        "model": model,
        "threshold": np.nan,
        "threshold_source": "discrete_prediction",
        **binary_wedge_metrics(observed, predicted),
        **_site_mean_metrics_from_predictions(frame),
    }


def operating_points(
    scores: pd.DataFrame,
    curve: pd.DataFrame,
    predictions: pd.DataFrame,
    min_precision: float = 0.20,
) -> pd.DataFrame:
    predictions = normalize_wedge_model_names(predictions)
    pooled = curve[curve["scope"].eq("all_sites_pooled")].copy()
    rows: list[dict[str, object]] = [
        _prediction_point(predictions, BASELINE_MODEL, "SpatialDepthKNN baseline"),
        _prediction_point(predictions, RECALL_MODEL, "current site-calibrated recall-first head"),
    ]
    baseline_recall = float(rows[0].get("recall", np.nan))

    def selected_row(label: str, frame: pd.DataFrame, source: str) -> dict[str, object]:
        row = frame.iloc[0].to_dict()
        threshold = float(row["threshold"])
        return {
            "operating_point": label,
            "model": "PosteriorWedgeProbability",
            "threshold": threshold,
            "threshold_source": source,
            **{key: row[key] for key in row if key not in {"scope", "threshold"}},
            **_site_mean_metrics_from_threshold(scores, threshold),
        }

    recall_pool = pooled[pooled["recall"].notna()].copy()
    feasible = recall_pool[pd.to_numeric(recall_pool["precision"], errors="coerce").fillna(0.0) >= float(min_precision)].copy()
    if feasible.empty:
        feasible = recall_pool
    feasible = feasible.sort_values(["recall", "threshold", "precision", "accuracy"], ascending=[False, False, False, False])
    rows.append(selected_row("pooled recall-first probability threshold", feasible, f"max_recall_precision_ge_{min_precision:g}"))

    f1_pool = pooled[pd.to_numeric(pooled["f1"], errors="coerce").notna()].copy()
    f1_pool = f1_pool.sort_values(["f1", "precision", "recall", "threshold"], ascending=[False, False, False, False])
    rows.append(selected_row("pooled max-F1 probability threshold", f1_pool, "max_f1"))

    guarded = pooled[pd.to_numeric(pooled["precision"], errors="coerce").notna()].copy()
    if np.isfinite(baseline_recall):
        guarded = guarded[pd.to_numeric(guarded["recall"], errors="coerce") >= baseline_recall]
    if guarded.empty:
        guarded = f1_pool
    guarded = guarded.sort_values(["precision", "f1", "recall", "threshold"], ascending=[False, False, False, False])
    rows.append(selected_row("precision-guarded probability threshold", guarded, "max_precision_at_or_above_baseline_recall"))
    return pd.DataFrame(rows)


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False


def save_wedge_operating_figure(curve: pd.DataFrame, points: pd.DataFrame, figure_dir: Path, stem: str = "arcticdata_wedge_operating_curve") -> list[Path]:
    apply_publication_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    pooled = curve[curve["scope"].eq("all_sites_pooled")].copy().sort_values("threshold")
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    sc = ax_a.scatter(pooled["recall"], pooled["precision"], c=pooled["threshold"], cmap="viridis", s=24, edgecolors="black", linewidths=0.25)
    ax_a.plot(pooled["recall"], pooled["precision"], color="#767676", lw=0.8, alpha=0.7)
    for _, row in points.iterrows():
        if str(row["model"]) == "PosteriorWedgeProbability":
            ax_a.scatter(float(row["recall"]), float(row["precision"]), marker="D", s=36, color="#B64342", edgecolors="black", linewidths=0.3, zorder=5)
    ax_a.set_xlabel("wedge recall")
    ax_a.set_ylabel("wedge precision")
    ax_a.set_xlim(-0.03, 1.03)
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.grid(color="0.9", lw=0.55)
    ax_a.text(-0.16, 1.04, "a", transform=ax_a.transAxes, fontsize=8, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax_a, fraction=0.046, pad=0.03)
    cbar.set_label("threshold", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.8)

    show = points[
        points["operating_point"].isin(
            [
                "SpatialDepthKNN baseline",
                "current site-calibrated recall-first head",
                "pooled max-F1 probability threshold",
                "precision-guarded probability threshold",
            ]
        )
    ].copy()
    labels = {
        "SpatialDepthKNN baseline": "KNN",
        "current site-calibrated recall-first head": "recall-first",
        "pooled max-F1 probability threshold": "max F1",
        "precision-guarded probability threshold": "precision-guarded",
    }
    show["label"] = show["operating_point"].map(labels)
    x = np.arange(len(show))
    width = 0.25
    metrics = [("recall", "#0F4D92"), ("precision", "#42949E"), ("false_positive_rate", "#B64342")]
    for i, (metric, color) in enumerate(metrics):
        ax_b.bar(x + (i - 1) * width, pd.to_numeric(show[metric], errors="coerce"), width=width, color=color, label=metric.replace("_", " "))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(show["label"], rotation=22, ha="right")
    ax_b.set_ylim(0, 1.03)
    ax_b.set_ylabel("pooled hold-out metric")
    ax_b.grid(axis="y", color="0.9", lw=0.55)
    ax_b.legend(fontsize=5.8, loc="upper right")
    ax_b.text(-0.13, 1.04, "b", transform=ax_b.transAxes, fontsize=8, fontweight="bold")
    fig.suptitle("Wedge-ice recall audit exposes the recall-precision operating curve", y=1.03, fontsize=9)

    paths: list[Path] = []
    for ext in ("svg", "pdf", "png", "tiff"):
        path = figure_dir / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def build_wedge_operating_audit(
    predictions_path: Path,
    prediction_dir: Path,
    table_dir: Path,
    figure_dir: Path,
    thresholds: Iterable[float] | None = None,
) -> dict[str, Path]:
    predictions = normalize_wedge_model_names(pd.read_csv(predictions_path))
    scores = load_wedge_probability_scores(predictions, prediction_dir=prediction_dir)
    curve = threshold_curve(scores, thresholds=thresholds)
    points = operating_points(scores, curve, predictions)
    table_dir.mkdir(parents=True, exist_ok=True)
    score_path = table_dir / "arcticdata_wedge_probability_holdout_scores.csv"
    curve_path = table_dir / "arcticdata_wedge_operating_curve.csv"
    points_path = table_dir / "arcticdata_wedge_operating_points.csv"
    scores.to_csv(score_path, index=False)
    curve.to_csv(curve_path, index=False)
    points.to_csv(points_path, index=False)
    figure_paths = save_wedge_operating_figure(curve, points, figure_dir)
    return {
        "scores": score_path,
        "curve": curve_path,
        "points": points_path,
        "figure_svg": figure_paths[0],
        "figure_pdf": figure_paths[1],
        "figure_png": figure_paths[2],
        "figure_tiff": figure_paths[3],
    }
