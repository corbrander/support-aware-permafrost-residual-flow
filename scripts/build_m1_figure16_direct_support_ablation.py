from __future__ import annotations

from pathlib import Path
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m1_figure_style import INK, apply_m1_style, export_m1_figure, panel_title


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "m1_support_guided" / "formal_direct_support_ablation"
SUPPORT_FILE = BASE / "support_aware_seed41" / "m1_test_id_seed41_detail.csv"
NEAREST_FILE = BASE / "nearest_seed41" / "m1_test_id_seed41_detail.csv"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure16_direct_support_ablation"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_direct_support_ablation"

BLUE = "#377eb8"
ORANGE = "#d95f02"
GREY = "#65727a"
LIGHT = "#d8dee2"

SUPPORTS = [
    ("borehole_eic", "EIC\ninterval"),
    ("borehole_temperature", "Temperature\ninterval"),
    ("ert_log_resistivity", "ERT\nvolume"),
    ("nmr_unfrozen_water", "NMR\nkernel"),
    ("alt", "ALT\ncrossing"),
]


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, values.size, size=(5000, values.size))].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def _bars_with_ci(
    ax: plt.Axes,
    support: pd.DataFrame,
    nearest: pd.DataFrame,
    prefix: str,
    *,
    ylabel: str,
    yline: float | None = None,
) -> list[dict[str, float | str]]:
    x = np.arange(len(SUPPORTS), dtype=float)
    width = 0.36
    rows: list[dict[str, float | str]] = []
    for method_index, (frame, label, color) in enumerate(
        ((support, "Support-aware", BLUE), (nearest, "Nearest-voxel", ORANGE))
    ):
        means: list[float] = []
        errors: list[list[float]] = []
        for index, (key, display) in enumerate(SUPPORTS):
            metric = f"{prefix}_{key}"
            mean, low, high = _bootstrap(frame[metric].to_numpy(), 1600 + 20 * method_index + index)
            means.append(mean)
            errors.append([mean - low, high - mean])
            rows.append(
                {
                    "method": label,
                    "support": display.replace("\n", " "),
                    "metric": prefix,
                    "mean": mean,
                    "ci95_lower": low,
                    "ci95_upper": high,
                }
            )
        positions = x + (method_index - 0.5) * width
        ax.bar(positions, means, width=width, color=color, alpha=0.88, label=label)
        ax.errorbar(
            positions,
            means,
            yerr=np.asarray(errors).T,
            fmt="none",
            ecolor=INK,
            elinewidth=0.7,
            capsize=2.2,
        )
    ax.set_xticks(x, ["EIC int.", "Temp. int.", "ERT vol.", "NMR ker.", "ALT cross."])
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right")
        label.set_fontsize(8.0)
    ax.set_ylabel(ylabel)
    if yline is not None:
        ax.axhline(yline, color=GREY, ls="--", lw=0.8)
    return rows


def main() -> None:
    apply_m1_style(base_font_size=9.4)
    support = pd.read_csv(SUPPORT_FILE)
    nearest = pd.read_csv(NEAREST_FILE)
    if len(support) != 100 or len(nearest) != 100:
        raise RuntimeError(
            "Figure 16 requires the complete 100-scene ID evaluation for both "
            f"branches; found support-aware={len(support)}, nearest-voxel={len(nearest)}"
        )
    for frame in (support, nearest):
        for threshold in (20, 30, 40):
            metric = f"high_eic_t{threshold}_object_f1"
            truth_count = f"high_eic_t{threshold}_truth_object_count"
            zero_match = frame[metric].isna() & frame[truth_count].gt(0)
            frame.loc[zero_match, metric] = 0.0
    paired = support.merge(
        nearest,
        on="scene_id",
        suffixes=("_support", "_nearest"),
        validate="one_to_one",
    )
    if len(paired) != 100:
        raise RuntimeError(
            "Direct-support branches do not contain the same 100 scene IDs; "
            f"matched {len(paired)}"
        )
    if not np.allclose(
        paired["anchor_eic_rmse_support"],
        paired["anchor_eic_rmse_nearest"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Direct-support branches do not share the same tree anchor")
    paired["eic_rmse_difference_support_minus_nearest"] = (
        paired["eic_rmse_support"] - paired["eic_rmse_nearest"]
    )

    fig, axes = plt.subplots(2, 3, figsize=(8.2, 6.15))

    ax = axes[0, 0]
    ax.scatter(
        paired["eic_rmse_nearest"],
        paired["eic_rmse_support"],
        s=17,
        color=BLUE,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
    )
    low = float(min(paired["eic_rmse_nearest"].min(), paired["eic_rmse_support"].min()))
    high = float(max(paired["eic_rmse_nearest"].max(), paired["eic_rmse_support"].max()))
    pad = 0.04 * (high - low)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=GREY, ls="--", lw=0.85)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Nearest-voxel EIC RMSE")
    ax.set_ylabel("Support-aware EIC RMSE")
    better = 100.0 * float((paired["eic_rmse_difference_support_minus_nearest"] < 0).mean())
    ax.text(0.97, 0.05, f"Support-aware lower in {better:.0f}%", transform=ax.transAxes, ha="right")
    panel_title(ax, "A", "Paired whole-volume error")

    ax = axes[0, 1]
    source_rows = _bars_with_ci(
        ax,
        support,
        nearest,
        "support_nrmse",
        ylabel="NRMSE at original support",
        yline=1.0,
    )
    ax.legend(loc="upper left", ncol=1)
    panel_title(ax, "B", "Original-support residual")

    ax = axes[0, 2]
    source_rows += _bars_with_ci(
        ax,
        support,
        nearest,
        "support_standardized_bias",
        ylabel="Signed bias / declared sigma",
        yline=0.0,
    )
    panel_title(ax, "C", "Support-scale signed bias")

    x = np.arange(2, dtype=float)
    width = 0.36
    ax = axes[1, 0]
    for method_index, (frame, label, color) in enumerate(
        ((support, "Support-aware", BLUE), (nearest, "Nearest-voxel", ORANGE))
    ):
        values = [float(frame["eic_coverage"].mean()), float(frame["eic_calibrated_coverage"].mean())]
        ax.bar(x + (method_index - 0.5) * width, values, width=width, color=color, alpha=0.88, label=label)
    ax.axhline(0.90, color=GREY, ls="--", lw=0.8)
    ax.set_xticks(x, ["Raw", "Calibrated"])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Voxelwise coverage")
    panel_title(ax, "D", "Validation-only coverage")

    ax = axes[1, 1]
    for method_index, (frame, label, color) in enumerate(
        ((support, "Support-aware", BLUE), (nearest, "Nearest-voxel", ORANGE))
    ):
        values = [float(frame["eic_mean_width"].mean()), float(frame["eic_calibrated_mean_width"].mean())]
        ax.bar(x + (method_index - 0.5) * width, values, width=width, color=color, alpha=0.88, label=label)
    ax.set_xticks(x, ["Raw", "Calibrated"])
    ax.set_ylabel("Mean EIC interval width")
    panel_title(ax, "E", "Coverage cost")

    ax = axes[1, 2]
    thresholds = np.array([0.20, 0.30, 0.40])
    for frame, method, color in (
        (support, "Support-aware", BLUE),
        (nearest, "Nearest-voxel", ORANGE),
    ):
        voxel = [float(frame[f"high_eic_t{int(t * 100):02d}_f1"].mean()) for t in thresholds]
        objects = [float(frame[f"high_eic_t{int(t * 100):02d}_object_f1"].mean()) for t in thresholds]
        ax.plot(thresholds, voxel, color=color, marker="o", lw=1.1, label=f"{method}, voxel")
        ax.plot(thresholds, objects, color=color, marker="s", ls="--", lw=1.1, label=f"{method}, object")
    ax.set_xticks(thresholds)
    ax.set_ylim(0.0, 0.78)
    ax.set_xlabel("High-EIC threshold")
    ax.set_ylabel("F1")
    ax.legend(loc="upper right")
    panel_title(ax, "F", "High-EIC retention")

    for axis in axes.flat:
        axis.grid(axis="y", color=LIGHT, lw=0.45, alpha=0.65)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.105, top=0.945, wspace=0.34, hspace=0.41)
    export_m1_figure(fig, OUTPUT)

    SOURCE.mkdir(parents=True, exist_ok=True)
    paired.to_csv(SOURCE / "figure16_paired_scene_metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(SOURCE / "figure16_support_summary.csv", index=False)

    summary_rows: list[dict[str, float | str | int]] = []
    paired_metrics = [
        ("Whole-volume EIC RMSE", "eic_rmse"),
        ("Raw EIC coverage", "eic_coverage"),
        ("Raw EIC width", "eic_mean_width"),
        ("Calibrated EIC coverage", "eic_calibrated_coverage"),
        ("Calibrated EIC width", "eic_calibrated_mean_width"),
        ("High-EIC voxel F1 at 0.30", "high_eic_t30_f1"),
        ("High-EIC object F1 at 0.30", "high_eic_t30_object_f1"),
    ]
    paired_metrics += [
        (f"{label.replace(chr(10), ' ')} support NRMSE", f"support_nrmse_{key}")
        for key, label in SUPPORTS
    ]
    paired_metrics += [
        (f"{label.replace(chr(10), ' ')} collapsed-voxel NRMSE", f"voxel_nrmse_{key}")
        for key, label in SUPPORTS
    ]
    paired_metrics += [
        (
            f"{label.replace(chr(10), ' ')} support standardized bias",
            f"support_standardized_bias_{key}",
        )
        for key, label in SUPPORTS
    ]
    for index, (label, metric) in enumerate(paired_metrics):
        support_values = paired[f"{metric}_support"].to_numpy(dtype=float)
        nearest_values = paired[f"{metric}_nearest"].to_numpy(dtype=float)
        difference = support_values - nearest_values
        support_mean, support_low, support_high = _bootstrap(support_values, 2100 + index)
        nearest_mean, nearest_low, nearest_high = _bootstrap(nearest_values, 2200 + index)
        diff_mean, diff_low, diff_high = _bootstrap(difference, 2300 + index)
        summary_rows.append(
            {
                "metric": label,
                "support_aware_mean": support_mean,
                "support_aware_ci95_lower": support_low,
                "support_aware_ci95_upper": support_high,
                "nearest_voxel_mean": nearest_mean,
                "nearest_voxel_ci95_lower": nearest_low,
                "nearest_voxel_ci95_upper": nearest_high,
                "paired_support_minus_nearest": diff_mean,
                "paired_ci95_lower": diff_low,
                "paired_ci95_upper": diff_high,
                "n_scenes": len(paired),
            }
        )
    summary_file = BASE / "m1_direct_support_ablation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
    nearest_conformal = (
        BASE / "nearest_seed41" / "m1_validation_seed41_spatial_conformal.json"
    )
    support_conformal = (
        BASE / "support_aware_seed41" / "m1_validation_seed41_spatial_conformal.json"
    )
    nearest_calibration = json.loads(nearest_conformal.read_text(encoding="utf-8"))
    support_calibration = json.loads(support_conformal.read_text(encoding="utf-8"))
    (SOURCE / "figure16_metadata.json").write_text(
        json.dumps(
            {
                "design": "matched single-fit seed-41 support-representation ablation",
                "test_scenes": len(paired),
                "posterior_members": 64,
                "sampling_steps": 10,
                "guidance_strength": 2.0,
                "ood_control_disabled_for_component_isolation": True,
                "support_aware_validation_conformal_quantile": support_calibration[
                    "global_quantile"
                ],
                "nearest_voxel_validation_conformal_quantile": nearest_calibration[
                    "global_quantile"
                ],
                "maximum_anchor_rmse_difference": float(
                    np.max(
                        np.abs(
                            paired["anchor_eic_rmse_support"]
                            - paired["anchor_eic_rmse_nearest"]
                        )
                    )
                ),
                "paired_difference_direction": "support-aware minus nearest-voxel",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copy2(
        nearest_conformal,
        SOURCE / "nearest_voxel_validation_conformal.json",
    )
    shutil.copy2(
        support_conformal,
        SOURCE / "support_aware_validation_conformal.json",
    )
    reuse_manifest = BASE / "support_aware_seed41" / "m1_validation_reuse_manifest.json"
    if reuse_manifest.exists():
        shutil.copy2(reuse_manifest, SOURCE / reuse_manifest.name)
    print(BASE / "m1_direct_support_ablation_summary.csv")
    print(OUTPUT)


if __name__ == "__main__":
    main()
