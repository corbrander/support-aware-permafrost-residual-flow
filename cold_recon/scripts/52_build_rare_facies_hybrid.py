from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.evaluation.rare_cryostructure import binary_event_metrics
from cold_recon.evaluation.rare_facies_hybrid import (
    apply_rare_facies_hybrid,
    load_npz_dict,
    rare_facies_hybrid_metrics,
    rare_facies_hybrid_operating_curve,
)
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "teal": "#42949E",
    "red": "#B64342",
    "gold": "#D4AE24",
    "neutral": "#767676",
    "black": "#272727",
}


def _style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False


def _panel(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def _parse_floors(text: str) -> list[float]:
    floors = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not floors:
        raise ValueError("--curve-floors must contain at least one numeric value")
    return floors


def _field(data: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    if name in data:
        return np.asarray(data[name])
    mean_name = f"{name}_mean"
    if mean_name in data:
        return np.asarray(data[mean_name])
    return None


def _facies_mode(data: dict[str, np.ndarray]) -> np.ndarray:
    if "facies_mode" in data:
        return np.asarray(data["facies_mode"], dtype=np.int16)
    if "facies" in data:
        return np.asarray(data["facies"], dtype=np.int16)
    if "facies_probability" in data:
        return np.argmax(np.asarray(data["facies_probability"], dtype=np.float32), axis=-1).astype(np.int16)
    if "facies_samples" in data:
        samples = np.asarray(data["facies_samples"], dtype=np.int16)
        n_classes = int(np.nanmax(samples)) + 1
        counts = np.stack([(samples == cls).sum(axis=0) for cls in range(n_classes)], axis=-1)
        return np.argmax(counts, axis=-1).astype(np.int16)
    raise KeyError("prediction must contain facies_mode, facies, facies_probability, or facies_samples")


def _prediction_metrics(
    data: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    z: np.ndarray,
    model_name: str,
    n_facies: int,
    rare_class: int,
    ice_threshold: float,
) -> dict[str, float | str]:
    pred: dict[str, np.ndarray] = {"facies": _facies_mode(data)}
    for field in ("eic", "temperature", "unfrozen_water", "log_resistivity"):
        values = _field(data, field)
        if values is not None:
            pred[field] = values.astype(np.float32)
    row: dict[str, float | str] = {"model": model_name}
    row.update(synthetic_metrics(pred, truth, z, n_facies=n_facies, ice_threshold=ice_threshold))
    rare = binary_event_metrics(np.asarray(truth["facies"], dtype=np.int16) == int(rare_class), pred["facies"] == int(rare_class))
    row["wedge_ice_recall"] = rare["recall"]
    row["wedge_ice_precision"] = rare["precision"]
    row["wedge_ice_f1"] = rare["f1"]
    row["wedge_ice_predicted_rate"] = rare["predicted_rate"]
    return row


def save_operating_curve_figure(
    curve: pd.DataFrame,
    metrics: pd.DataFrame,
    fig_dir: Path,
    stem: str = "rare_facies_hybrid_operating_curve",
) -> list[Path]:
    if curve.empty:
        raise ValueError("rare-facies hybrid operating curve is empty")
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    x = pd.to_numeric(curve["eic_floor"], errors="coerce").to_numpy(dtype=float)
    recall = pd.to_numeric(curve["wedge_ice_recall"], errors="coerce").to_numpy(dtype=float)
    precision = pd.to_numeric(curve["wedge_ice_precision"], errors="coerce").to_numpy(dtype=float)
    f1 = pd.to_numeric(curve["wedge_ice_f1"], errors="coerce").to_numpy(dtype=float)
    mean_delta = pd.to_numeric(curve["mean_iou_delta_vs_base"], errors="coerce").to_numpy(dtype=float)
    gate_fraction = pd.to_numeric(curve["gate_fraction"], errors="coerce").to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 0.92], wspace=0.42, hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    ax_a.plot(x, recall, marker="o", color=PALETTE["red"], lw=1.2, label="wedge recall")
    ax_a.plot(x, precision, marker="s", color=PALETTE["teal"], lw=1.2, label="wedge precision")
    ax_a.plot(x, f1, marker="^", color=PALETTE["gold"], lw=1.2, label="wedge F1")
    ax_a.set_xlabel("EIC floor for accepting implicit wedge proposal")
    ax_a.set_ylabel("event score")
    ax_a.set_ylim(0.0, max(0.62, float(np.nanmax([recall, precision, f1])) + 0.05))
    ax_a.grid(color="0.9", lw=0.55)
    ax_a.legend(fontsize=6.0, loc="upper right")
    _panel(ax_a, "a")

    ax_b.plot(x, mean_delta, marker="o", color=PALETTE["cold"], lw=1.2, label="mean IoU delta")
    ax_b.axhline(0.0, color="0.45", lw=0.75, ls="--")
    ax_b.set_xlabel("EIC floor")
    ax_b.set_ylabel("mean facies IoU delta")
    ax_b.grid(color="0.9", lw=0.55)
    ax_b2 = ax_b.twinx()
    ax_b2.plot(x, gate_fraction, marker="D", color=PALETTE["neutral"], lw=1.0, label="gated voxels")
    ax_b2.set_ylabel("gated voxel fraction")
    lines = [line for line in ax_b.get_lines() + ax_b2.get_lines() if not line.get_label().startswith("_")]
    ax_b.legend(lines, [line.get_label() for line in lines], fontsize=6.0, loc="lower right")
    _panel(ax_b, "b")

    metric_names = ["mean_iou", "wedge_ice_recall", "wedge_ice_precision"]
    metric_labels = ["mean IoU", "wedge recall", "wedge precision"]
    model_order = [
        "COLDReconLatentDiffusionPhysicsTrained",
        "COLDReconImplicit",
        "COLDReconLatentDiffusionRareFaciesHybrid",
    ]
    present = [model for model in model_order if model in set(metrics["model"].astype(str))]
    colors = [PALETTE["cold"], PALETTE["neutral"], PALETTE["red"]]
    bar_x = np.arange(len(metric_names), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(present)) if present else np.array([])
    short = {
        "COLDReconLatentDiffusionPhysicsTrained": "physics-trained",
        "COLDReconImplicit": "implicit proposal",
        "COLDReconLatentDiffusionRareFaciesHybrid": "rare-facies hybrid",
    }
    for offset, model, color in zip(offsets, present, colors):
        row = metrics[metrics["model"].astype(str).eq(model)].iloc[0]
        values = [float(pd.to_numeric(row.get(name, np.nan), errors="coerce")) for name in metric_names]
        ax_c.bar(bar_x + offset, values, width=width, label=short.get(model, model), color=color)
    ax_c.set_xticks(bar_x)
    ax_c.set_xticklabels(metric_labels, rotation=18, ha="right")
    ax_c.set_ylim(0.0, 1.02)
    ax_c.set_ylabel("score")
    ax_c.grid(axis="y", color="0.9", lw=0.55)
    ax_c.legend(fontsize=6.0, loc="upper right")
    _panel(ax_c, "c")

    fig.suptitle("Rare-facies hybrid exposes the wedge-recall versus mean-IoU trade-off", fontsize=9, y=0.995)
    paths: list[Path] = []
    for ext in ("svg", "pdf", "png", "tiff"):
        path = fig_dir / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--diffusion-posterior", default=None)
    parser.add_argument("--rare-source", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--eic-floor", type=float, default=0.10)
    parser.add_argument("--gate-probability", type=float, default=0.95)
    parser.add_argument("--curve-floors", default="0.0,0.1,0.2,0.3,0.4,0.5")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    pred_dir = Path(config["paths"]["predictions_dir"])
    posterior_path = Path(args.diffusion_posterior) if args.diffusion_posterior else pred_dir / "diffusion_posterior_physics_trained.npz"
    rare_source_path = Path(args.rare_source) if args.rare_source else pred_dir / "implicit_prediction.npz"
    output_path = Path(args.output) if args.output else pred_dir / "diffusion_posterior_rare_facies_hybrid.npz"

    posterior = load_npz_dict(posterior_path)
    rare_source = load_npz_dict(rare_source_path)
    rare_source_facies = _facies_mode(rare_source)
    n_facies = int(config["model"].get("n_facies", 7))
    rare_class = 6
    ice_threshold = float(config["evaluation"].get("ice_rich_threshold", 0.30))

    hybrid = apply_rare_facies_hybrid(
        posterior,
        rare_source_facies,
        n_facies=n_facies,
        rare_class=rare_class,
        eic_floor=float(args.eic_floor),
        gate_probability=float(args.gate_probability),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **hybrid)

    z = sample["grid"]["z"]
    truth = sample["fields"]
    rows = [
        _prediction_metrics(
            posterior,
            truth,
            z,
            model_name="COLDReconLatentDiffusionPhysicsTrained",
            n_facies=n_facies,
            rare_class=rare_class,
            ice_threshold=ice_threshold,
        ),
        _prediction_metrics(
            rare_source,
            truth,
            z,
            model_name="COLDReconImplicit",
            n_facies=n_facies,
            rare_class=rare_class,
            ice_threshold=ice_threshold,
        ),
        rare_facies_hybrid_metrics(
            hybrid,
            truth,
            z,
            model_name="COLDReconLatentDiffusionRareFaciesHybrid",
            n_facies=n_facies,
            rare_class=rare_class,
            ice_threshold=ice_threshold,
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics["eic_floor"] = np.nan
    metrics.loc[metrics["model"].eq("COLDReconLatentDiffusionRareFaciesHybrid"), "eic_floor"] = float(args.eic_floor)
    metrics["gate_probability"] = np.nan
    metrics.loc[metrics["model"].eq("COLDReconLatentDiffusionRareFaciesHybrid"), "gate_probability"] = float(args.gate_probability)
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = table_dir / "diffusion_rare_facies_hybrid_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    curve = rare_facies_hybrid_operating_curve(
        posterior,
        rare_source_facies,
        truth,
        z,
        eic_floors=_parse_floors(args.curve_floors),
        n_facies=n_facies,
        rare_class=rare_class,
        gate_probability=float(args.gate_probability),
        ice_threshold=ice_threshold,
    )
    curve_path = table_dir / "rare_facies_hybrid_operating_curve.csv"
    curve.to_csv(curve_path, index=False)
    figure_paths = save_operating_curve_figure(curve, metrics, fig_dir)

    print(f"posterior={output_path}")
    print(f"metrics={metrics_path}")
    print(f"curve={curve_path}")
    for path in figure_paths:
        print(f"figure={path}")
    cols = ["model", "mean_iou", "wedge_ice_recall", "wedge_ice_precision", "wedge_ice_f1", "eic_floor"]
    print(metrics[[col for col in cols if col in metrics.columns]].to_string(index=False))
    print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
