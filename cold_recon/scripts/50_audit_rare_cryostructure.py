from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.rare_cryostructure import build_rare_cryostructure_audit
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "baseline": "#767676",
    "cold": "#0F4D92",
    "cold_light": "#3775BA",
    "teal": "#42949E",
    "red": "#B64342",
    "gold": "#D4AE24",
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


def _short_model(name: str) -> str:
    return {
        "GradientBoosting": "GB",
        "COLDReconLatentDiffusion": "Latent diff.",
        "COLDReconFNOOperatorDiffusion": "FNO diff.",
        "COLDReconRectifiedFlow": "Rectified flow",
        "COLDReconLatentDiffusionPhysicsTrained": "Physics-trained",
        "COLDReconLatentDiffusionRareFaciesHybrid": "Rare-facies hybrid",
        "COLDReconLatentDiffusionPhysicsRefined": "Physics-refined",
    }.get(name, name)


def _panel(ax, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def save_rare_figure(audit, fig_dir: Path, stem: str = "synthetic_rare_cryostructure_audit") -> list[Path]:
    _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        "GradientBoosting",
        "COLDReconLatentDiffusion",
        "COLDReconFNOOperatorDiffusion",
        "COLDReconRectifiedFlow",
        "COLDReconLatentDiffusionPhysicsTrained",
        "COLDReconLatentDiffusionRareFaciesHybrid",
        "COLDReconLatentDiffusionPhysicsRefined",
    ]
    view = audit[audit["model"].isin(selected)].copy()
    view["order"] = view["model"].map({m: i for i, m in enumerate(selected)})
    view = view.sort_values("order")

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.48)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    y = np.arange(len(view))
    width = 0.34
    raw_f1 = view["raw_eic_f1"].astype(float).to_numpy()
    rate_f1 = view["rate_constrained_eic_f1"].astype(float).to_numpy()
    ax_a.barh(y - width / 2, raw_f1, height=width, color=PALETTE["baseline"], label="raw EIC > 0.30")
    ax_a.barh(y + width / 2, rate_f1, height=width, color=PALETTE["cold"], label="obs-rate threshold")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([_short_model(str(m)) for m in view["model"]])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("high-EIC event F1")
    ax_a.set_xlim(0.0, max(0.38, float(np.nanmax([raw_f1, rate_f1])) + 0.04))
    ax_a.grid(axis="x", color="0.9", lw=0.6)
    ax_a.legend(fontsize=6.2, loc="lower right")
    _panel(ax_a, "a")

    raw_rec = view["raw_eic_recall"].astype(float).to_numpy()
    rate_rec = view["rate_constrained_eic_recall"].astype(float).to_numpy()
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    for marker, (_, row) in zip(markers, view.iterrows()):
        ax_b.scatter(
            float(row["raw_eic_recall"]),
            float(row["rate_constrained_eic_recall"]),
            s=32,
            marker=marker,
            color=PALETTE["cold"],
            edgecolors="black",
            linewidths=0.3,
            label=_short_model(str(row["model"])),
        )
    lim = max(0.55, float(np.nanmax([raw_rec, rate_rec])) + 0.05)
    ax_b.plot([0, lim], [0, lim], color="0.5", lw=0.8, ls="--")
    ax_b.set_xlim(0, lim)
    ax_b.set_ylim(0, lim)
    ax_b.set_xlabel("raw high-EIC recall")
    ax_b.set_ylabel("obs-rate high-EIC recall")
    ax_b.grid(color="0.9", lw=0.55)
    ax_b.legend(fontsize=5.0, loc="lower right", handletextpad=0.25, borderpad=0.2, labelspacing=0.25)
    _panel(ax_b, "b")

    rare_cols = [
        ("rare_facies_recall", "rare facies", PALETTE["teal"]),
        ("facies_3_ice_rich_silt_recall", "ice-rich silt", PALETTE["gold"]),
        ("facies_6_wedge_ice_recall", "wedge ice", PALETTE["red"]),
    ]
    labels = [label for _, label, _ in rare_cols]
    models_c = [
        "COLDReconLatentDiffusionPhysicsTrained",
        "COLDReconLatentDiffusionRareFaciesHybrid",
    ]
    models_c = [model for model in models_c if model in set(audit["model"].astype(str))]
    if not models_c:
        models_c = [str(view["model"].iloc[-1])]
    x = np.arange(len(labels), dtype=float)
    width_c = 0.34 if len(models_c) > 1 else 0.62
    offsets = np.linspace(-width_c / 2, width_c / 2, len(models_c)) if len(models_c) > 1 else np.array([0.0])
    colors_c = [PALETTE["cold"], PALETTE["red"], PALETTE["teal"]]
    for offset, model, color in zip(offsets, models_c, colors_c):
        row = audit[audit["model"].astype(str).eq(model)].iloc[0]
        vals = [float(row[col]) for col, _, _ in rare_cols]
        ax_c.bar(x + offset, vals, color=color, width=width_c, label=_short_model(model))
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylim(0, 1.02)
    ax_c.set_ylabel("facies recall")
    ax_c.set_title("facies recall operating point", fontsize=7)
    ax_c.tick_params(axis="x", rotation=20)
    ax_c.grid(axis="y", color="0.9", lw=0.55)
    ax_c.legend(fontsize=5.5, loc="upper right")
    _panel(ax_c, "c")

    fig.suptitle("Rare cryostructure audit separates high-EIC detection from facies reconstruction", fontsize=9, y=0.995)
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
    parser.add_argument("--observation-rate-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    sample = load_sample_npz(args.sample or config["training"]["sample_path"])
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    pred_dir = Path(config["paths"]["predictions_dir"])
    audit = build_rare_cryostructure_audit(
        pred_dir,
        sample["fields"],
        sample["observations"],
        eic_threshold=float(config["evaluation"].get("ice_rich_threshold", 0.30)),
        observation_rate_multiplier=float(args.observation_rate_multiplier),
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "synthetic_rare_cryostructure_audit.csv"
    audit.to_csv(table_path, index=False)
    figure_paths = save_rare_figure(audit, fig_dir)
    print(f"audit={table_path}")
    for path in figure_paths:
        print(f"figure={path}")
    cols = [
        "model",
        "raw_eic_recall",
        "raw_eic_f1",
        "rate_constrained_eic_recall",
        "rate_constrained_eic_f1",
        "rare_facies_recall",
        "facies_6_wedge_ice_recall",
    ]
    print(audit[[col for col in cols if col in audit.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
