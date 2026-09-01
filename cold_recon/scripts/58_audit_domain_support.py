from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.evaluation.domain_support import build_domain_support_audit, write_domain_support_outputs
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "green": "#2E9E44",
    "amber": "#C9A227",
    "red": "#B64342",
    "teal": "#42949E",
    "neutral": "#767676",
    "neutral_light": "#D8D8D8",
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


def _panel(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _short_site(site: str) -> str:
    return {
        "Anaktuvuk Fire August": "Anaktuvuk\nAug.",
        "Anaktuvuk River Fire June": "Anaktuvuk\nJune",
        "Prudhoe Bay August September": "Prudhoe\nBay",
        "Itkillik June July": "Itkillik",
        "Tuktoyaktuk September": "Tuktoyaktuk",
    }.get(str(site), str(site).replace(" ", "\n", 1))


def _class_color(value: str) -> str:
    value = str(value)
    if value == "model-supported transfer":
        return PALETTE["green"]
    if value == "moderate support":
        return PALETTE["cold"]
    if value == "guarded local-prior":
        return PALETTE["amber"]
    return PALETTE["red"]


def _outcome_value(value: str) -> float:
    return {"failure": 0.0, "noninferior": 0.5, "win": 1.0, "not_evaluated": np.nan}.get(str(value), np.nan)


def _save(fig: plt.Figure, fig_dir: Path, stem: str) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
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


def save_domain_support_figure(site_audit: pd.DataFrame, summary: dict, fig_dir: Path) -> list[Path]:
    if site_audit.empty:
        raise ValueError("domain support audit is empty")
    _style()
    view = site_audit.sort_values("support_score", ascending=True).reset_index(drop=True)
    y = np.arange(len(view))
    labels = view["site"].map(_short_site)

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.45)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    components = [
        ("train_observation_score", "EIC obs", PALETTE["cold"]),
        ("train_group_score", "groups", PALETTE["teal"]),
        ("train_borehole_score", "boreholes", PALETTE["green"]),
        ("split_support_score", "split", PALETTE["amber"]),
    ]
    left = np.zeros(len(view))
    for col, label, color in components:
        vals = view[col].astype(float).to_numpy()
        weighted = vals * {"train_observation_score": 0.35, "train_group_score": 0.25, "train_borehole_score": 0.25, "split_support_score": 0.15}[col]
        ax_a.barh(y, weighted, left=left, color=color, edgecolor=PALETTE["black"], linewidth=0.25, label=label)
        left += weighted
    for yi, score, klass in zip(y, view["support_score"].astype(float), view["applicability_class"]):
        ax_a.text(score + 0.015, yi, str(klass).replace(" ", "\n", 1), va="center", ha="left", fontsize=5.4, color=_class_color(str(klass)))
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=6)
    ax_a.set_xlim(0, 1.18)
    ax_a.set_xlabel("train-side support score")
    ax_a.set_title("support components available before holdout scoring")
    ax_a.grid(axis="x", color="0.9", lw=0.55)
    ax_a.legend(fontsize=5.5, loc="lower right")
    _panel(ax_a, "a")

    x = view["support_score"].astype(float).to_numpy()
    eic = 100.0 * view["eic_rmse_reduction_vs_best_simple"].astype(float).to_numpy()
    colors = [_class_color(v) for v in view["applicability_class"]]
    markers = ["s" if bool(v) else "o" for v in view["guarded_by_transfer_adapter"]]
    for xi, yi, color, marker, site in zip(x, eic, colors, markers, view["site"]):
        ax_b.scatter(xi, yi, s=42, marker=marker, color=color, edgecolor=PALETTE["black"], linewidth=0.35)
        ax_b.text(xi + 0.008, yi, _short_site(str(site)).replace("\n", " "), fontsize=5.2, va="center")
    ax_b.axhline(0, color=PALETTE["black"], lw=0.7)
    ax_b.set_xlabel("support score")
    ax_b.set_ylabel("EIC RMSE reduction (%)")
    ax_b.set_title("support versus EIC outcome")
    ax_b.grid(color="0.9", lw=0.55)
    _panel(ax_b, "b", x=-0.16)

    tasks = ["facies_outcome", "eic_outcome", "high_eic_outcome", "wedge_outcome"]
    task_labels = ["facies", "EIC", "high-EIC", "wedge"]
    mat = np.asarray([[ _outcome_value(row[col]) for col in tasks] for _, row in view.iterrows()], dtype=float)
    cmap = matplotlib.colors.ListedColormap([PALETTE["red"], PALETTE["amber"], PALETTE["green"]])
    cmap.set_bad(color=PALETTE["neutral_light"])
    norm = matplotlib.colors.BoundaryNorm([-0.01, 0.25, 0.75, 1.01], cmap.N)
    ax_c.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm, aspect="auto")
    ax_c.set_xticks(np.arange(len(task_labels)))
    ax_c.set_xticklabels(task_labels, fontsize=6)
    ax_c.set_yticks(np.arange(len(view)))
    ax_c.set_yticklabels(labels, fontsize=6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            label = "NA" if not np.isfinite(mat[i, j]) else {1.0: "win", 0.5: "tie", 0.0: "fail"}.get(float(mat[i, j]), "")
            ax_c.text(j, i, label, ha="center", va="center", fontsize=5.4, color=PALETTE["black"])
    ax_c.set_title("holdout outcome matrix")
    _panel(ax_c, "c", x=-0.16)

    fig.suptitle("Domain-support audit separates deployable support signals from holdout outcomes", fontsize=9, y=0.995)
    return _save(fig, fig_dir, "domain_support_audit")


def _write_source_data(path: Path, site_audit: pd.DataFrame, summary: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    site = site_audit.copy()
    site.insert(0, "record_type", "site_support")
    summary_rows = pd.DataFrame([summary])
    summary_rows.insert(0, "record_type", "summary")
    pd.concat([site, summary_rows], ignore_index=True, sort=False).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    source_dir = Path("outputs/source_data")
    site_path = table_dir / "transfer_failure_site_diagnostics.csv"
    if not site_path.exists():
        site_path = table_dir / "external_generalization_site_deltas.csv"
    if not site_path.exists():
        raise FileNotFoundError("Missing transfer/domain site diagnostics")

    result = build_domain_support_audit(pd.read_csv(site_path))
    audit_path, summary_path = write_domain_support_outputs(result, table_dir)
    source_path = _write_source_data(source_dir / "domain_support_audit_source_data.csv", result.site_audit, result.summary)
    figures = save_domain_support_figure(result.site_audit, result.summary, fig_dir)

    print(f"audit={audit_path}")
    print(f"summary={summary_path}")
    print(f"source_data={source_path}")
    for path in figures:
        print(f"figure={path}")
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
