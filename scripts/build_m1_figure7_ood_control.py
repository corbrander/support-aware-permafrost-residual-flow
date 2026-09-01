from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m1_figure_style import apply_m1_style, enforce_m1_typography


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "m1_support_guided" / "tables"
CONTROLLED = ROOT / "outputs" / "m1_support_guided" / "formal_controlled_selected_guidance"
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure11_ood_control"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure11_ood_control"

INK = "#263238"
BLUE = "#377eb8"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
RED = "#b23a48"
TEAL = "#1b9e77"
GREY = "#9aa5ab"

SPLITS = [
    ("test_id", "ID", BLUE, "m1_test_id_three_seed"),
    ("ood_abrupt_boundary", "Abrupt\ngeometry", ORANGE, "m1_ood_abrupt_boundary_three_seed"),
    ("ood_altered_eic_coupling", "Altered\ncoupling", PURPLE, "m1_ood_altered_eic_coupling_three_seed"),
    ("ood_saline_low_resistivity_ice", "Saline\nlow-resistivity", RED, "m1_ood_saline_low_resistivity_ice_three_seed"),
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
    ax.text(0.0, 1.035, f"({letter})", transform=ax.transAxes, fontsize=8.4, fontweight="normal", ha="left")
    ax.text(0.10, 1.035, title, transform=ax.transAxes, fontsize=7.8, color=INK, ha="left")


def load_details() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split, label, _, _ in SPLITS:
        for seed in (41, 42, 43):
            path = CONTROLLED / f"m1_{split}_seed{seed}_detail.csv"
            frame = pd.read_csv(path)
            expected = 100 if split == "test_id" else 50
            if len(frame) != expected or frame["scene_id"].nunique() != expected:
                raise RuntimeError(f"Incomplete or duplicated formal detail: {path}")
            frame = frame.copy()
            frame["evaluation_family"] = label.replace("\n", " ")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metric_row(path: Path, metric: str) -> pd.Series:
    rows = pd.read_csv(path)
    matched = rows.loc[rows["metric"] == metric]
    if matched.empty:
        raise KeyError(f"{metric} not found in {path}")
    return matched.iloc[0]


def build_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for split, label, _, stem in SPLITS:
        summary_path = TABLES / f"{stem}_summary.csv"
        eic = metric_row(summary_path, "eic_rmse")
        anchor = metric_row(summary_path, "anchor_eic_rmse")
        subset = details.loc[details["split"] == split]
        rows.append(
            {
                "split": split,
                "label": label.replace("\n", " "),
                "eic_rmse_mean": float(eic["mean"]),
                "eic_rmse_ci95_lower": float(eic["ci95_lower"]),
                "eic_rmse_ci95_upper": float(eic["ci95_upper"]),
                "anchor_eic_rmse_mean": float(anchor["mean"]),
                "anchor_eic_rmse_ci95_lower": float(anchor["ci95_lower"]),
                "anchor_eic_rmse_ci95_upper": float(anchor["ci95_upper"]),
                "ood_score_mean": float(subset["ood_score"].mean()),
                "abstain_fraction": float(subset["ood_abstain"].astype(float).mean()),
                "exact_fallback_fraction": float(subset["exact_anchor_fallback_applied"].astype(float).mean()),
                "scene_seed_rows": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def plot_error(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_title(ax, "a", "Whole-volume EIC error")
    x = np.arange(len(summary), dtype=float)
    width = 0.16
    for offset, prefix, color, marker, label in [
        (-width / 2, "eic_rmse", BLUE, "o", "controlled model"),
        (width / 2, "anchor_eic_rmse", GREY, "s", "tree anchor"),
    ]:
        mean = summary[f"{prefix}_mean"].to_numpy(float)
        lower = summary[f"{prefix}_ci95_lower"].to_numpy(float)
        upper = summary[f"{prefix}_ci95_upper"].to_numpy(float)
        ax.errorbar(x + offset, mean, yerr=np.vstack((mean - lower, upper - mean)), fmt=marker, ms=4.2, color=color, ecolor=color, capsize=2, lw=0.9, label=label)
    ax.set_xticks(x, [label for _, label, _, _ in SPLITS])
    ax.set_ylabel("EIC RMSE")
    ax.set_ylim(0.075, 0.195)
    ax.grid(axis="y", color="0.91", lw=0.55)
    ax.legend(loc="upper left", fontsize=6.0)


def plot_scores(ax: plt.Axes, details: pd.DataFrame) -> None:
    panel_title(ax, "b", "Validation-locked OOD score")
    values = [details.loc[details["split"] == split, "ood_score"].to_numpy(float) for split, _, _, _ in SPLITS]
    boxes = ax.boxplot(values, patch_artist=True, widths=0.58, whis=(5, 95), showfliers=False, medianprops={"color": "white", "lw": 1.1})
    for box, (_, _, color, _) in zip(boxes["boxes"], SPLITS, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.82)
        box.set_edgecolor(color)
    for element in ("whiskers", "caps"):
        for artist in boxes[element]:
            artist.set_color("0.35")
            artist.set_linewidth(0.7)
    ax.axhline(0.95, color=ORANGE, lw=0.9, ls="--", label="control starts (0.95)")
    ax.axhline(0.99, color=RED, lw=0.9, ls=":", label="fallback (0.99)")
    ax.set_xticks(np.arange(1, 5), [label for _, label, _, _ in SPLITS])
    ax.set_ylabel("dual-max OOD score")
    ax.set_ylim(0, 1.035)
    ax.legend(loc="lower right", fontsize=5.8)


def plot_response(ax: plt.Axes, details: pd.DataFrame) -> None:
    panel_title(ax, "c", "Local safety response")
    order = np.argsort(details["ood_score"].to_numpy(float))
    score = details["ood_score"].to_numpy(float)[order]
    gate = details["ood_bias_gate_multiplier"].to_numpy(float)[order]
    inflation = details["ood_interval_inflation"].to_numpy(float)[order]
    bins = np.linspace(0, 1, 31)
    centers = 0.5 * (bins[:-1] + bins[1:])
    group = np.digitize(score, bins) - 1
    gate_med = np.array([np.median(gate[group == index]) if np.any(group == index) else np.nan for index in range(len(centers))])
    inf_med = np.array([np.median(inflation[group == index]) if np.any(group == index) else np.nan for index in range(len(centers))])
    ax.plot(centers, gate_med, color=TEAL, lw=1.5, label="bias-gate multiplier")
    ax.set_xlabel("dual-max OOD score")
    ax.set_ylabel("bias-gate multiplier", color=TEAL)
    ax.tick_params(axis="y", labelcolor=TEAL)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)
    twin = ax.twinx()
    twin.plot(centers, inf_med, color=PURPLE, lw=1.5, label="interval inflation")
    twin.set_ylabel("interval inflation factor", color=PURPLE)
    twin.tick_params(axis="y", labelcolor=PURPLE)
    twin.set_ylim(0.9, max(3.0, float(np.nanmax(inf_med)) * 1.08))
    ax.axvline(0.95, color=ORANGE, lw=0.8, ls="--")
    ax.axvline(0.99, color=RED, lw=0.8, ls=":")
    lines = ax.get_lines()[:1] + twin.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=5.8)


def plot_fallback(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_title(ax, "d", "Abstention and exact mean fallback")
    x = np.arange(len(summary), dtype=float)
    width = 0.31
    abstain = summary["abstain_fraction"].to_numpy(float)
    fallback = summary["exact_fallback_fraction"].to_numpy(float)
    ax.bar(x - width / 2, abstain, width, color=RED, alpha=0.82, label="abstained")
    ax.bar(x + width / 2, fallback, width, color=INK, alpha=0.75, label="exact anchor fallback")
    for index, value in enumerate(fallback):
        ax.text(index, value + 0.015, f"{value:.0%}", ha="center", va="bottom", fontsize=5.9)
    ax.set_xticks(x, [label for _, label, _, _ in SPLITS])
    ax.set_ylabel("scene-seed fraction")
    ax.set_ylim(0, max(0.42, float(fallback.max()) + 0.08))
    ax.grid(axis="y", color="0.91", lw=0.55)
    ax.legend(loc="upper left", fontsize=5.8)
    ax.text(0.99, 0.95, "54/54 abstentions\nused exact fallback", transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=6.0, fontweight="normal")


def main() -> None:
    apply_m1_style()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    details = load_details()
    summary = build_summary(details)
    selected = details[
        [
            "scene_id",
            "split",
            "seed",
            "eic_rmse",
            "anchor_eic_rmse",
            "ood_score",
            "ood_risk",
            "ood_bias_gate_multiplier",
            "ood_interval_inflation",
            "ood_abstain",
            "exact_anchor_fallback_applied",
        ]
    ].copy()
    selected.to_csv(SOURCE / "figure11_scene_level.csv", index=False)
    summary.to_csv(SOURCE / "figure11_summary.csv", index=False)
    audit_path = TABLES / "m1_ood_feature_audit_ref500_valcal.csv"
    pd.read_csv(audit_path).to_csv(SOURCE / "figure11_ood_feature_audit.csv", index=False)
    metadata = {
        "source_root": str(CONTROLLED.relative_to(ROOT)),
        "ood_feature_audit": str(audit_path.relative_to(ROOT)),
        "controller_fit": "500 training scenes; score calibration on 100 validation scenes",
        "control_threshold": 0.95,
        "exact_fallback_threshold": 0.99,
        "posterior_members": 64,
        "sampling_steps": 10,
        "guidance_strength": 2.0,
        "note": "Thresholds were locked before ID and OOD evaluation.",
    }
    (SOURCE / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 118 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.925, top=0.94, bottom=0.105, wspace=0.34, hspace=0.42)
    plot_error(axes[0, 0], summary)
    plot_scores(axes[0, 1], details)
    plot_response(axes[1, 0], details)
    plot_fallback(axes[1, 1], summary)
    enforce_m1_typography(fig)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(OUTPUT), "source": str(SOURCE), "rows": int(len(details)), "summary": summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
