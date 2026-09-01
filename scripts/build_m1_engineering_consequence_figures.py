from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from m1_figure_style import INK, apply_m1_style, export_m1_figure, panel_title


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "source_data" / "m1_figure5" / "test_id_combined_00008_seed41.npz"
CONFORMAL = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_controlled_selected_guidance"
    / "m1_validation_seed41_spatial_conformal.json"
)
FIGURE14 = (
    ROOT / "paper" / "engineering_geology_manuscript" / "figures" / "m1_final"
    / "figure14_engineering_response_propagation"
)
FIGURE15 = (
    ROOT / "paper" / "engineering_geology_manuscript" / "figures" / "m1_final"
    / "figure15_engineering_decision_sensitivity"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_engineering_consequence"

BLUE = "#377eb8"
TEAL = "#1b9e77"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
RED = "#b23a48"
GOLD = "#d6a632"
GREY = "#607d8b"

THAW_DEPTH_M = 6.0
SCREENING_THRESHOLD_M = 0.30
POSTERIOR_DECISION_PROBABILITY = 0.50


def potential_settlement(eic: np.ndarray, depth_mask: np.ndarray, dz: float) -> np.ndarray:
    """Excess-ice-only potential settlement under complete thaw and drainage."""
    return np.sum(np.asarray(eic)[..., depth_mask], axis=-1) * float(dz)


def gradient(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    gx, gy = np.gradient(np.asarray(field, dtype=float), float(dx), float(dy), edge_order=1)
    return np.hypot(gx, gy)


def rmse(candidate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(candidate) - np.asarray(truth)) ** 2)))


def map_panel(fig, ax, data, x, y, letter, title, label, *, cmap, vmin, vmax, contour=None):
    image = ax.imshow(
        np.asarray(data).T,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if contour is not None:
        ax.contour(x, y, np.asarray(contour).T, levels=[SCREENING_THRESHOLD_M], colors="white", linewidths=0.75)
    ax.set_xlabel("Distance, x (m)")
    ax.set_ylabel("Distance, y (m)")
    ax.set_xticks([0, 64, 126])
    ax.set_yticks([0, 64, 126])
    panel_title(ax, letter, title, x=0.0, y=1.025, title_offset=0.14, fontsize=9.6)
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.052, pad=0.23)
    bar.set_label(label, labelpad=1)
    bar.ax.tick_params(length=2, pad=1)
    return image


def load_payload():
    if not ARCHIVE.exists() or not CONFORMAL.exists():
        raise FileNotFoundError("Locked volumetric archive or conformal artifact is missing")
    values = np.load(ARCHIVE, allow_pickle=False)
    calibration = json.loads(CONFORMAL.read_text(encoding="utf-8"))
    x = values["x"].astype(float)
    y = values["y"].astype(float)
    z = values["z"].astype(float)
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    dz = float(np.mean(np.diff(z)))
    depth_mask = z < THAW_DEPTH_M
    if not np.isclose(np.sum(depth_mask) * dz, THAW_DEPTH_M):
        raise RuntimeError("The prescribed thawed thickness is not represented exactly by the grid")

    truth = potential_settlement(values["truth_eic"], depth_mask, dz)
    anchor = potential_settlement(values["anchor_eic"], depth_mask, dz)
    samples = potential_settlement(values["posterior_eic_samples"], depth_mask, dz)
    posterior = samples.mean(axis=0)
    raw_lower = np.quantile(samples, 0.05, axis=0)
    raw_upper = np.quantile(samples, 0.95, axis=0)
    probability = np.mean(samples > SCREENING_THRESHOLD_M, axis=0)

    q = float(calibration["global_quantile"])
    eic_mean = values["posterior_eic_mean"].astype(float)
    eic_std = values["posterior_eic_std"].astype(float)
    eic_lower = np.clip(eic_mean - q * eic_std, 0.0, 0.90)
    eic_upper = np.clip(eic_mean + q * eic_std, 0.0, 0.90)
    propagated_lower = potential_settlement(eic_lower, depth_mask, dz)
    propagated_upper = potential_settlement(eic_upper, depth_mask, dz)
    propagated_width = propagated_upper - propagated_lower

    truth_decision = truth > SCREENING_THRESHOLD_M
    anchor_decision = anchor > SCREENING_THRESHOLD_M
    posterior_decision = probability >= POSTERIOR_DECISION_PROBABILITY
    outcome = np.zeros_like(truth, dtype=np.int16)
    outcome[posterior_decision & truth_decision] = 1
    outcome[posterior_decision & ~truth_decision] = 2
    outcome[~posterior_decision & truth_decision] = 3

    return {
        "values": values,
        "calibration": calibration,
        "x": x,
        "y": y,
        "z": z,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "depth_mask": depth_mask,
        "truth": truth,
        "anchor": anchor,
        "posterior": posterior,
        "samples": samples,
        "probability": probability,
        "raw_lower": raw_lower,
        "raw_upper": raw_upper,
        "propagated_lower": propagated_lower,
        "propagated_upper": propagated_upper,
        "propagated_width": propagated_width,
        "truth_decision": truth_decision,
        "anchor_decision": anchor_decision,
        "posterior_decision": posterior_decision,
        "outcome": outcome,
    }


def metrics(payload) -> dict[str, float | int | str]:
    truth = payload["truth"]
    anchor = payload["anchor"]
    posterior = payload["posterior"]
    truth_decision = payload["truth_decision"]
    anchor_decision = payload["anchor_decision"]
    posterior_decision = payload["posterior_decision"]
    truth_gradient = gradient(truth, payload["dx"], payload["dy"])
    anchor_gradient = gradient(anchor, payload["dx"], payload["dy"])
    posterior_gradient = gradient(posterior, payload["dx"], payload["dy"])
    raw_covered = (truth >= payload["raw_lower"]) & (truth <= payload["raw_upper"])
    propagated_covered = (
        (truth >= payload["propagated_lower"]) & (truth <= payload["propagated_upper"])
    )
    tp = int(np.sum(posterior_decision & truth_decision))
    tn = int(np.sum(~posterior_decision & ~truth_decision))
    positives = max(int(np.sum(truth_decision)), 1)
    negatives = max(int(np.sum(~truth_decision)), 1)
    anchor_rmse = rmse(anchor, truth)
    posterior_rmse = rmse(posterior, truth)
    return {
        "scene_id": str(payload["values"]["scene_id"].item()),
        "model_seed": int(payload["values"]["checkpoint_seed"].item()),
        "posterior_members": int(payload["samples"].shape[0]),
        "prescribed_thawed_thickness_m": THAW_DEPTH_M,
        "screening_threshold_m": SCREENING_THRESHOLD_M,
        "posterior_decision_probability": POSTERIOR_DECISION_PROBABILITY,
        "truth_mean_potential_settlement_m": float(np.mean(truth)),
        "anchor_mean_potential_settlement_m": float(np.mean(anchor)),
        "posterior_mean_potential_settlement_m": float(np.mean(posterior)),
        "anchor_potential_settlement_rmse_m": anchor_rmse,
        "posterior_potential_settlement_rmse_m": posterior_rmse,
        "posterior_relative_rmse_reduction_vs_anchor": float(1.0 - posterior_rmse / anchor_rmse),
        "truth_flagged_area_fraction": float(np.mean(truth_decision)),
        "anchor_flagged_area_fraction": float(np.mean(anchor_decision)),
        "posterior_flagged_area_fraction": float(np.mean(posterior_decision)),
        "anchor_to_posterior_decision_change_fraction": float(np.mean(anchor_decision != posterior_decision)),
        "posterior_decision_sensitivity": float(tp / positives),
        "posterior_decision_specificity": float(tn / negatives),
        "truth_differential_gradient_p95": float(np.quantile(truth_gradient, 0.95)),
        "anchor_differential_gradient_p95": float(np.quantile(anchor_gradient, 0.95)),
        "posterior_differential_gradient_p95": float(np.quantile(posterior_gradient, 0.95)),
        "anchor_differential_gradient_rmse": rmse(anchor_gradient, truth_gradient),
        "posterior_differential_gradient_rmse": rmse(posterior_gradient, truth_gradient),
        "raw_settlement_interval_cellwise_coverage": float(np.mean(raw_covered)),
        "raw_settlement_interval_mean_width_m": float(np.mean(payload["raw_upper"] - payload["raw_lower"])),
        "propagated_conformal_eic_envelope_cellwise_coverage": float(np.mean(propagated_covered)),
        "propagated_conformal_eic_envelope_mean_width_m": float(np.mean(payload["propagated_width"])),
        "conformal_eic_quantile": float(payload["calibration"]["global_quantile"]),
    }


def figure14(payload, summary):
    apply_m1_style(base_font_size=8.6)
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 138 / 25.4))
    fig.subplots_adjust(left=0.070, right=0.985, top=0.965, bottom=0.140, wspace=0.30, hspace=0.60)
    vmax = float(np.quantile(payload["truth"], 0.99))
    common = dict(x=payload["x"], y=payload["y"], cmap="cividis", vmin=0.0, vmax=vmax)
    map_panel(fig, axes[0, 0], payload["truth"], letter="A", title="Controlled response truth",
              label="Excess-ice potential settlement (m)", **common)
    map_panel(fig, axes[0, 1], payload["anchor"], letter="B", title="Tree-anchor response",
              label="Excess-ice potential settlement (m)", **common)
    map_panel(fig, axes[0, 2], payload["posterior"], letter="C", title="Posterior-mean response",
              label="Excess-ice potential settlement (m)", **common)
    map_panel(
        fig, axes[1, 0], payload["probability"], payload["x"], payload["y"], "D",
        "Raw threshold-exceedance probability", "P(S_e > 0.30 m)", cmap="viridis", vmin=0.0, vmax=1.0,
        contour=payload["truth"],
    )
    map_panel(
        fig, axes[1, 1], payload["propagated_width"], payload["x"], payload["y"], "E",
        "Calibrated-EIC propagation width", "Diagnostic S_e envelope width (m)", cmap="magma", vmin=0.0,
        vmax=float(np.quantile(payload["propagated_width"], 0.99)),
    )
    cmap = ListedColormap(["#d9d9d9", TEAL, ORANGE, RED])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    image = axes[1, 2].imshow(
        payload["outcome"].T, origin="lower",
        extent=[payload["x"].min(), payload["x"].max(), payload["y"].min(), payload["y"].max()],
        aspect="equal", interpolation="nearest", cmap=cmap, norm=norm,
    )
    axes[1, 2].set_xlabel("Distance, x (m)")
    axes[1, 2].set_ylabel("Distance, y (m)")
    axes[1, 2].set_xticks([0, 64, 126])
    axes[1, 2].set_yticks([0, 64, 126])
    panel_title(axes[1, 2], "F", "Posterior screening outcome", x=0.0, y=1.025, title_offset=0.14, fontsize=9.6)
    bar = fig.colorbar(image, ax=axes[1, 2], orientation="horizontal", fraction=0.052, pad=0.23, ticks=[0, 1, 2, 3])
    bar.ax.set_xticklabels(["Low", "Hit", "False +", "Miss"])
    bar.set_label("Outcome at S_e = 0.30 m and P = 0.50", labelpad=1)
    fig.text(
        0.5, 0.020,
        f"Illustrative prospectively locked scene | prescribed newly thawed thickness: {THAW_DEPTH_M:.0f} m | posterior response RMSE {summary['posterior_potential_settlement_rmse_m']:.3f} m "
        f"(anchor {summary['anchor_potential_settlement_rmse_m']:.3f} m) | white contour in (D): controlled Se = 0.30 m",
        ha="center", va="bottom", fontsize=7.7, color=INK,
    )
    export_m1_figure(fig, FIGURE14)


def threshold_curves(payload):
    thresholds = np.linspace(0.05, 1.20, 48)
    rows = []
    for threshold in thresholds:
        truth = payload["truth"] > threshold
        anchor = payload["anchor"] > threshold
        probability = np.mean(payload["samples"] > threshold, axis=0)
        posterior = probability >= POSTERIOR_DECISION_PROBABILITY
        pos = max(int(np.sum(truth)), 1)
        neg = max(int(np.sum(~truth)), 1)
        rows.append(
            {
                "threshold_m": float(threshold),
                "truth_flagged_fraction": float(np.mean(truth)),
                "anchor_flagged_fraction": float(np.mean(anchor)),
                "posterior_flagged_fraction": float(np.mean(posterior)),
                "anchor_sensitivity": float(np.sum(anchor & truth) / pos),
                "posterior_sensitivity": float(np.sum(posterior & truth) / pos),
                "anchor_specificity": float(np.sum(~anchor & ~truth) / neg),
                "posterior_specificity": float(np.sum(~posterior & ~truth) / neg),
            }
        )
    return pd.DataFrame(rows)


def figure15(payload, summary, curves):
    apply_m1_style(base_font_size=8.6)
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 136 / 25.4))
    fig.subplots_adjust(left=0.075, right=0.980, top=0.965, bottom=0.105, wspace=0.34, hspace=0.46)
    truth_g = gradient(payload["truth"], payload["dx"], payload["dy"])
    anchor_g = gradient(payload["anchor"], payload["dx"], payload["dy"])
    posterior_g = gradient(payload["posterior"], payload["dx"], payload["dy"])
    vmax = float(np.quantile(truth_g, 0.99))
    for ax, data, letter, title in (
        (axes[0, 0], truth_g, "A", "Controlled differential response"),
        (axes[0, 1], anchor_g, "B", "Tree-anchor differential response"),
        (axes[0, 2], posterior_g, "C", "Posterior differential response"),
    ):
        map_panel(fig, ax, data, payload["x"], payload["y"], letter, title,
                  "Settlement-gradient magnitude (m/m)", cmap="magma", vmin=0.0, vmax=vmax)

    ax = axes[1, 0]
    panel_title(ax, "D", "Threshold-dependent flagged area", x=0.0, y=1.025, title_offset=0.14, fontsize=9.6)
    ax.plot(curves["threshold_m"], curves["truth_flagged_fraction"], color=INK, lw=1.3, label="Controlled truth")
    ax.plot(curves["threshold_m"], curves["anchor_flagged_fraction"], color=GREY, lw=1.2, ls="--", label="Tree anchor")
    ax.plot(curves["threshold_m"], curves["posterior_flagged_fraction"], color=TEAL, lw=1.3, label="Posterior P >= 0.50")
    ax.axvline(SCREENING_THRESHOLD_M, color=RED, lw=0.8, ls=":")
    ax.set_xlabel("Screening threshold, S_e (m)")
    ax.set_ylabel("Flagged area fraction")
    ax.set_ylim(0, 1)
    ax.grid(color="0.91", lw=0.45)
    ax.legend(loc="upper right", fontsize=7.4)

    ax = axes[1, 1]
    panel_title(ax, "E", "Threshold-dependent discrimination", x=0.0, y=1.025, title_offset=0.14, fontsize=9.6)
    ax.plot(curves["threshold_m"], curves["posterior_sensitivity"], color=TEAL, lw=1.3, label="Posterior sensitivity")
    ax.plot(curves["threshold_m"], curves["posterior_specificity"], color=ORANGE, lw=1.3, label="Posterior specificity")
    ax.plot(curves["threshold_m"], curves["anchor_sensitivity"], color=GREY, lw=1.0, ls="--", label="Anchor sensitivity")
    ax.plot(curves["threshold_m"], curves["anchor_specificity"], color=INK, lw=1.0, ls=":", label="Anchor specificity")
    ax.axvline(SCREENING_THRESHOLD_M, color=RED, lw=0.8, ls=":")
    ax.set_xlabel("Screening threshold, S_e (m)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.02)
    ax.grid(color="0.91", lw=0.45)
    ax.legend(loc="lower right", fontsize=7.0)

    ax = axes[1, 2]
    panel_title(ax, "F", "Uncertainty after response propagation", x=0.0, y=1.025, title_offset=0.14, fontsize=9.6)
    labels = ["Raw 64-member\ninterval", "Propagated calibrated-\nEIC envelope"]
    coverage = [summary["raw_settlement_interval_cellwise_coverage"], summary["propagated_conformal_eic_envelope_cellwise_coverage"]]
    width = [summary["raw_settlement_interval_mean_width_m"], summary["propagated_conformal_eic_envelope_mean_width_m"]]
    xpos = np.arange(2)
    bars = ax.bar(xpos, coverage, color=[BLUE, PURPLE], alpha=0.75, width=0.58)
    ax.axhline(0.90, color=RED, ls="--", lw=0.8, label="Nominal 0.90")
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("Descriptive cellwise coverage")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="0.91", lw=0.45)
    ax2 = ax.twinx()
    ax2.plot(xpos, width, color=ORANGE, marker="o", ms=5.0, lw=1.1, label="Mean width")
    ax2.set_ylabel("Mean response width (m)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    for b, value in zip(bars, coverage, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=7.5)
    for xval, value in zip(xpos, width, strict=True):
        ax2.text(xval, value + 0.06, f"{value:.3f} m", ha="center", color=ORANGE, fontsize=7.3)
    ax.text(0.02, 0.03, "Not a settlement-level conformal guarantee", transform=ax.transAxes, fontsize=7.1, color=RED)

    fig.text(
        0.5, 0.018,
        f"At Se = 0.30 m: sensitivity {summary['posterior_decision_sensitivity']:.3f}, specificity {summary['posterior_decision_specificity']:.3f}; "
        f"posterior differential-gradient p95 {summary['posterior_differential_gradient_p95']:.3f} versus controlled {summary['truth_differential_gradient_p95']:.3f} m/m",
        ha="center", va="bottom", fontsize=7.7, color=INK,
    )
    export_m1_figure(fig, FIGURE15)


def save_source(payload, summary, curves):
    SOURCE.mkdir(parents=True, exist_ok=True)
    x, y = payload["x"], payload["y"]
    frame = pd.DataFrame(
        {
            "x_m": np.repeat(x, len(y)),
            "y_m": np.tile(y, len(x)),
            "truth_potential_settlement_m": payload["truth"].reshape(-1),
            "anchor_potential_settlement_m": payload["anchor"].reshape(-1),
            "posterior_potential_settlement_mean_m": payload["posterior"].reshape(-1),
            "raw_exceedance_probability_s_gt_0p30": payload["probability"].reshape(-1),
            "raw_interval_lower_m": payload["raw_lower"].reshape(-1),
            "raw_interval_upper_m": payload["raw_upper"].reshape(-1),
            "propagated_calibrated_eic_lower_m": payload["propagated_lower"].reshape(-1),
            "propagated_calibrated_eic_upper_m": payload["propagated_upper"].reshape(-1),
            "posterior_screening_outcome_code": payload["outcome"].reshape(-1),
        }
    )
    frame.to_csv(SOURCE / "engineering_response_grid.csv", index=False)
    curves.to_csv(SOURCE / "engineering_threshold_sensitivity.csv", index=False)
    (SOURCE / "engineering_response_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = {
        "source_archive": str(ARCHIVE.relative_to(ROOT)),
        "conformal_artifact": str(CONFORMAL.relative_to(ROOT)),
        "physical_definition": "S_e(x,y;D) = integral from 0 to D of volumetric EIC dz",
        "assumptions": [
            "the prescribed top 6 m of the reconstructed volume thaws completely",
            "meltwater from excess ice drains and the corresponding excess-ice volume is lost",
            "pore-ice phase-change strain, soil-skeleton compression, loading, drainage time and thermal timing are excluded",
        ],
        "screening_rule": "flag when raw posterior P(S_e > 0.30 m) >= 0.50",
        "threshold_status": "illustrative investigation threshold, not an allowable design settlement",
        "uncertainty_status": "the propagated calibrated-EIC envelope is diagnostic and is not a conformal interval for settlement",
        "outcome_codes": {"0": "correct low", "1": "detected high", "2": "false alert", "3": "missed high"},
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    payload = load_payload()
    summary = metrics(payload)
    curves = threshold_curves(payload)
    save_source(payload, summary, curves)
    figure14(payload, summary)
    figure15(payload, summary, curves)
    print(json.dumps({"figure14": str(FIGURE14), "figure15": str(FIGURE15), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
