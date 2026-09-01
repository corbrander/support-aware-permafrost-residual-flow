from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.evaluation.journal_readiness import build_journal_readiness_audit, write_journal_readiness_outputs
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "cold": "#0F4D92",
    "green": "#2E9E44",
    "amber": "#C9A227",
    "red": "#B64342",
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


def _panel(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _short_criterion(text: str) -> str:
    mapping = {
        "Three-source public-data evidence gate": "public\ngate",
        "External transfer boundary is quantified": "transfer\nboundary",
        "Rare cryostructure and high-EIC stress tests are separated": "rare\nstress",
        "Complete figure evidence chain": "figure\nchain",
        "Reproducible package closure": "repro\nclosure",
        "Independent public-data breadth": "public\nbreadth",
        "Site-wise transfer robustness": "site\nrobustness",
        "Surveyed coordinates and dense labels": "dense\nlabels",
        "Prospective VOI validation": "prospective\nVOI",
        "Full-field public 3D ground truth": "3D public\ntruth",
    }
    return mapping.get(text, text.replace(" ", "\n", 1))


def _status_color(status: str) -> str:
    if status == "pass":
        return PALETTE["green"]
    if status == "conditional":
        return PALETTE["amber"]
    if status == "not_yet":
        return PALETTE["red"]
    return PALETTE["neutral"]


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


def save_journal_readiness_figure(audit: pd.DataFrame, summary: dict, fig_dir: Path) -> list[Path]:
    if audit.empty:
        raise ValueError("journal readiness audit is empty")
    _style()
    audit = audit.copy()
    tiers = list(dict.fromkeys(audit["readiness_tier"].astype(str).tolist()))
    criteria = audit["criterion"].astype(str).tolist()
    scores = audit["status_score"].astype(float).to_numpy()
    statuses = audit["status"].astype(str).tolist()

    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.45)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    y = np.arange(len(audit))
    ax_a.barh(y, scores, color=[_status_color(s) for s in statuses], edgecolor=PALETTE["black"], linewidth=0.25)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([_short_criterion(c) for c in criteria], fontsize=6)
    ax_a.set_xlim(0, 1.05)
    ax_a.set_xlabel("readiness score")
    ax_a.set_title("criterion-level readiness")
    ax_a.grid(axis="x", color="0.9", lw=0.55)
    ax_a.invert_yaxis()
    for idx, (score, status) in enumerate(zip(scores, statuses)):
        ax_a.text(score + 0.02, idx, status.replace("_", " "), va="center", ha="left", fontsize=5.5)
    for tier in tiers:
        idxs = np.where(audit["readiness_tier"].astype(str).to_numpy() == tier)[0]
        if len(idxs):
            ax_a.axhline(max(idxs) + 0.5, color="0.82", lw=0.7)
            ax_a.text(1.02, float(idxs.mean()), tier, transform=ax_a.get_yaxis_transform(), ha="left", va="center", fontsize=6, color=PALETTE["neutral"])
    _panel(ax_a, "a")

    tier_summary = pd.DataFrame(summary.get("tiers", []))
    if tier_summary.empty:
        tier_summary = audit.groupby("readiness_tier", as_index=False).agg(score=("status_score", "mean"))
    x = np.arange(len(tier_summary))
    colors = [PALETTE["green"] if bool(v) else PALETTE["amber"] for v in tier_summary.get("ready_claim", pd.Series([False] * len(tier_summary)))]
    ax_b.bar(x, tier_summary["score"].astype(float), color=colors, edgecolor=PALETTE["black"], linewidth=0.25)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str(t).replace(" ", "\n", 1) for t in tier_summary["tier"]], fontsize=6)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("mean score")
    ax_b.set_title("tier summary")
    ax_b.grid(axis="y", color="0.9", lw=0.55)
    for xi, val in zip(x, tier_summary["score"].astype(float)):
        ax_b.text(xi, float(val) + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=6)
    _panel(ax_b, "b", x=-0.16)

    status_counts = audit.groupby(["readiness_tier", "status"]).size().unstack(fill_value=0)
    for status in ("pass", "conditional", "not_yet", "missing"):
        if status not in status_counts.columns:
            status_counts[status] = 0
    left = np.zeros(len(status_counts))
    yy = np.arange(len(status_counts))
    for status in ("pass", "conditional", "not_yet", "missing"):
        vals = status_counts[status].to_numpy()
        ax_c.barh(yy, vals, left=left, color=_status_color(status), edgecolor=PALETTE["black"], linewidth=0.25, label=status.replace("_", " "))
        left += vals
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels([str(v).replace(" ", "\n", 1) for v in status_counts.index], fontsize=6)
    ax_c.set_xlabel("criteria")
    ax_c.set_title("claim boundary")
    ax_c.legend(fontsize=5.5, loc="lower right")
    ax_c.grid(axis="x", color="0.9", lw=0.55)
    _panel(ax_c, "c", x=-0.16)

    fig.suptitle("COLD-Recon readiness audit separates CG algorithm evidence from EG field-claim gaps", fontsize=9, y=0.995)
    return _save(fig, fig_dir, "journal_readiness_audit")


def _write_source_data(path: Path, audit: pd.DataFrame, summary: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_rows = audit.copy()
    audit_rows.insert(0, "record_type", "criterion")
    summary_rows = pd.DataFrame(summary.get("tiers", []))
    if not summary_rows.empty:
        summary_rows.insert(0, "record_type", "tier_summary")
    pd.concat([audit_rows, summary_rows], ignore_index=True, sort=False).to_csv(path, index=False)
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

    result = build_journal_readiness_audit(
        gate_summary=_read_json(table_dir / "real_data_cg_gate.json"),
        real_benchmark=_read_csv(table_dir / "real_data_cg_benchmark.csv"),
        external_generalization=_read_csv(table_dir / "external_generalization_audit.csv"),
        figure_atlas=_read_csv(table_dir / "figure_atlas.csv"),
        reproducibility_summary=_read_json(table_dir / "reproducibility_summary.json"),
        domain_support_summary=_read_json(table_dir / "domain_support_summary.json"),
        coordinate_label_summary=_read_json(table_dir / "coordinate_label_coverage_summary.json"),
        voi_backtest_summary=_read_json(table_dir / "voi_backtest_summary.json"),
    )
    audit_path, summary_path = write_journal_readiness_outputs(result, table_dir)
    source_path = _write_source_data(source_dir / "journal_readiness_audit_source_data.csv", result.audit, result.summary)
    figure_paths = save_journal_readiness_figure(result.audit, result.summary, fig_dir)

    print(f"audit={audit_path}")
    print(f"summary={summary_path}")
    print(f"source_data={source_path}")
    for path in figure_paths:
        print(f"figure={path}")
    print(f"cg_algorithm_article_ready={result.summary['cg_algorithm_article_ready']}")
    print(f"eg_field_generalization_ready={result.summary['eg_field_generalization_ready']}")
    print(f"recommended_positioning={result.summary['recommended_positioning']}")


if __name__ == "__main__":
    main()
