from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from m1_figure_style import apply_m1_style, enforce_m1_typography


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "m1_support_guided" / "tables"
REGISTRY = ROOT / "configs" / "m1_experiment_registry.yaml"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure4_a0_a11_ablation"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure4"

INK = "#263238"
BLUE = "#377eb8"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
TEAL = "#1b9e77"
GREY = "#9aa5ab"
RED = "#b23a48"
IDS = [f"A{index}" for index in range(12)]

UNBOUNDED = [
    (430, TABLES / "m1_geostatistical_test_id_detail.csv"),
    (
        431,
        ROOT
        / "outputs"
        / "m1_support_guided"
        / "formal_geostat_seed431"
        / "m1_geostatistical_test_id_detail.csv",
    ),
    (
        432,
        ROOT
        / "outputs"
        / "m1_support_guided"
        / "formal_geostat_seed432"
        / "m1_geostatistical_test_id_detail.csv",
    ),
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(0.0, 1.035, f"({letter})", transform=ax.transAxes, fontsize=8.4, fontweight="normal")
    ax.text(0.10, 1.035, title, transform=ax.transAxes, fontsize=7.8, color=INK)


def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    summary_path = TABLES / "m1_a0_a11_ablation_summary.csv"
    detail_path = TABLES / "m1_a0_a11_ablation_detail.csv"
    missing = [path for path in (summary_path, detail_path, REGISTRY) if not path.exists()]
    missing.extend(path for _, path in UNBOUNDED if not path.exists())
    if missing:
        raise FileNotFoundError(
            "Figure 4 cannot be built before the complete ablation and geostatistical artifacts exist: "
            + ", ".join(str(path) for path in missing)
        )
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    if sorted(summary["ablation_id"].unique(), key=lambda value: int(value[1:])) != IDS:
        raise RuntimeError("A0-A11 summary is incomplete.")
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    frames = []
    for seed, path in UNBOUNDED:
        current = pd.read_csv(path).copy()
        current["seed"] = seed
        frames.append(current)
    sensitivity = pd.concat(frames, ignore_index=True)
    if len(sensitivity) != 300:
        raise RuntimeError("Unbounded Gaussian sensitivity must contain 300 seed-scene rows.")
    return summary, detail, registry, sensitivity


def metric_row(summary: pd.DataFrame, ablation_id: str, metric: str) -> pd.Series | None:
    selected = summary.loc[
        (summary["ablation_id"] == ablation_id) & (summary["metric"] == metric)
    ]
    return None if selected.empty else selected.iloc[0]


def plot_eic(ax: plt.Axes, summary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    panel_title(ax, "a", "Fixed-budget EIC reconstruction error")
    rows = [metric_row(summary, ablation_id, "eic_rmse") for ablation_id in IDS]
    if any(row is None for row in rows):
        raise RuntimeError("Every A0-A11 row must report EIC RMSE.")
    mean = np.asarray([float(row["mean"]) for row in rows if row is not None])
    low = np.asarray([float(row["ci95_lower"]) for row in rows if row is not None])
    high = np.asarray([float(row["ci95_upper"]) for row in rows if row is not None])
    y = np.arange(len(IDS))[::-1]
    colors = [GREY, ORANGE, *([BLUE] * 9), RED]
    for position, value, lower, upper, color in zip(y, mean, low, high, colors, strict=True):
        ax.errorbar(
            value,
            position,
            xerr=np.asarray([[value - lower], [upper - value]]),
            fmt="o",
            color=color,
            ecolor=color,
            ms=4.1,
            lw=0.8,
            capsize=1.8,
        )
    unbounded_mean = float(sensitivity["eic_rmse"].mean())
    invalid = float(sensitivity["invalid_eic_sample_fraction"].mean())
    a1_y = y[1]
    ax.scatter(
        [unbounded_mean],
        [a1_y - 0.28],
        marker="D",
        s=24,
        facecolor="none",
        edgecolor=PURPLE,
        linewidth=0.9,
        label=f"unbounded Gaussian sensitivity ({invalid:.0%} invalid)",
    )
    ax.set_yticks(y, IDS)
    ax.set_xlabel("EIC RMSE (lower is better)")
    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.legend(loc="lower right", fontsize=5.5)


def plot_probabilistic(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_title(ax, "b", "Probabilistic score and common support residual")
    label_offsets = {
        "A1": (4, 2),
        "A7": (8, 5),
        "A8": (-18, 7),
        "A9": (8, 5),
        "A10": (8, -12),
        "A11": (8, 7),
    }
    clustered_points: list[tuple[float, float]] = []
    for ablation_id in IDS:
        crps = metric_row(summary, ablation_id, "eic_crps")
        support = metric_row(summary, ablation_id, "support_nrmse_borehole_eic")
        if crps is None or support is None:
            continue
        color = RED if ablation_id == "A11" else (ORANGE if ablation_id == "A1" else BLUE)
        size = 35 if ablation_id == "A11" else 20
        x = float(crps["mean"])
        y = float(support["mean"])
        ax.scatter(x, y, s=size, color=color, zorder=3)
        if ablation_id in {"A2", "A3", "A4", "A5", "A6"}:
            clustered_points.append((x, y))
            continue
        offset = label_offsets.get(ablation_id, (3, 2))
        ax.annotate(
            ablation_id,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.5,
            color=color,
        )
    if clustered_points:
        cluster_x = float(np.mean([point[0] for point in clustered_points]))
        cluster_y = float(np.mean([point[1] for point in clustered_points]))
        ax.annotate(
            "A2--A6",
            (cluster_x, cluster_y),
            xytext=(4, -9),
            textcoords="offset points",
            fontsize=5.5,
            color=BLUE,
        )
    ax.axhline(1.0, color=INK, lw=0.7, ls="--", label="declared noise scale")
    ax.set_ylim(0.94, 2.36)
    ax.set_xlabel("EIC CRPS (lower is better)")
    ax.set_ylabel("Borehole-EIC support NRMSE")
    ax.grid(color="0.92", lw=0.5)
    ax.legend(loc="best", fontsize=5.5)


def component_matrix(registry: dict) -> tuple[np.ndarray, list[str]]:
    columns = [
        "Factorized\nstate",
        "Explicit\nsupport",
        "Profile\ncovariance",
        "Token\nencoder",
        "Noise\nconditioning",
        "Safe bias /\nanomaly",
        "Probabilistic\nconstitutive",
        "Stepwise\nguidance",
        "Safety /\ncalibration",
    ]
    matrix = np.zeros((len(IDS), len(columns)), dtype=float)
    for row_index, ablation_id in enumerate(IDS):
        config = registry["ablations"][ablation_id]
        state = str(config.get("state", ""))
        matrix[row_index] = [
            float("factorized" in state.lower()),
            float(config.get("support") == "explicit"),
            float(config.get("covariance") == "profile_correlated"),
            float(bool(config.get("token_conditioning", False))),
            float(bool(config.get("noise_conditioning", False))),
            float(bool(config.get("bias_anomaly_decomposition", False))),
            float(bool(config.get("probabilistic_constitutive", False))),
            float(bool(config.get("guided_sampling", False))),
            float(
                any(
                    bool(config.get(key, False))
                    for key in ("high_eic_event_head", "ood_safe_fallback", "block_conformal")
                )
                or "calibration" in config
            ),
        ]
    return matrix, columns


def plot_components(ax: plt.Axes, registry: dict) -> None:
    panel_title(ax, "c", "Predeclared component sequence")
    matrix, columns = component_matrix(registry)
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#f1f3f4", TEAL]), vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(IDS)), IDS)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(IDS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def plot_difference(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_title(ax, "d", "Paired difference from tree prior")
    rows = [metric_row(summary, ablation_id, "eic_rmse_difference_vs_tree") for ablation_id in IDS]
    mean = np.asarray([float(row["mean"]) for row in rows if row is not None])
    low = np.asarray([float(row["ci95_lower"]) for row in rows if row is not None])
    high = np.asarray([float(row["ci95_upper"]) for row in rows if row is not None])
    labels = [ablation_id for ablation_id, row in zip(IDS, rows, strict=True) if row is not None]
    y = np.arange(len(labels))[::-1]
    for position, value, lower, upper, label in zip(y, mean, low, high, labels, strict=True):
        color = RED if label == "A11" else (TEAL if value < 0 else ORANGE)
        ax.errorbar(
            value,
            position,
            xerr=np.asarray([[value - lower], [upper - value]]),
            fmt="o",
            color=color,
            ecolor=color,
            ms=4.0,
            lw=0.8,
            capsize=1.8,
        )
    ax.axvline(0, color=INK, lw=0.75)
    ax.axvline(0.005, color=RED, lw=0.8, ls="--", label="NI margin +0.005")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Model minus tree-prior EIC RMSE")
    ax.grid(axis="x", color="0.92", lw=0.5)
    ax.legend(loc="lower right", fontsize=5.5)


def main() -> None:
    apply_m1_style()
    summary, detail, registry, sensitivity = require_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SOURCE / "figure4_ablation_summary.csv", index=False)
    detail.to_csv(SOURCE / "figure4_ablation_detail.csv", index=False)
    sensitivity.to_csv(SOURCE / "figure4_unbounded_gaussian_sensitivity.csv", index=False)
    matrix, columns = component_matrix(registry)
    pd.DataFrame(matrix, index=IDS, columns=columns).to_csv(SOURCE / "figure4_component_matrix.csv")
    (SOURCE / "metadata.json").write_text(
        json.dumps(
            {
                "registry": str(REGISTRY.relative_to(ROOT)),
                "a4_checkpoint_reuse": "A4 changes covariance-aware guidance at inference and reuses the A3 checkpoint.",
                "a10_checkpoint_reuse": "A10 adds stepwise likelihood guidance and reuses the A9 checkpoint.",
                "primary_a1": "Bounded Gaussian ensemble with validation-only spatial block conformal calibration.",
                "sensitivity_a1": "The unbounded Gaussian result is displayed separately with its invalid-sample fraction.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 132 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.15, wspace=0.37, hspace=0.48)
    plot_eic(axes[0, 0], summary, sensitivity)
    plot_probabilistic(axes[0, 1], summary)
    plot_components(axes[1, 0], registry)
    plot_difference(axes[1, 1], summary)
    enforce_m1_typography(fig)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        OUTPUT.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(OUTPUT), "source": str(SOURCE)}, indent=2))


if __name__ == "__main__":
    main()
