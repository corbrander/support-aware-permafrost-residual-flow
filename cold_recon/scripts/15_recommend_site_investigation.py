from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.data_schema import observations_from_npz
from cold_recon.evaluation.site_investigation import (
    VOIWeights,
    build_voi_score,
    posterior_score_components,
    recommend_boreholes,
    recommend_ert_lines,
)
from cold_recon.utils.config import ensure_dirs, load_config


def _load_observations(path: str | None):
    if not path:
        return None
    obs_path = Path(path)
    if not obs_path.exists():
        return None
    return observations_from_npz(np.load(obs_path, allow_pickle=False))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["rank", "x", "y", "recommended_depth_m", "orientation", "x_start", "y_start", "x_end", "y_end", "voi_score", "line_score"]
    ordered = [name for name in preferred if name in fieldnames] + [name for name in fieldnames if name not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _plot_recommendations(
    score: np.ndarray,
    posterior: dict[str, np.ndarray],
    observations,
    boreholes: list[dict[str, float]],
    lines: list[dict[str, float | str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = posterior["grid_x"]
    y = posterior["grid_y"]
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    finite = np.where(np.isfinite(score), score, np.nan)
    im = ax.imshow(
        finite.T,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="auto",
        cmap="magma",
    )
    fig.colorbar(im, ax=ax, label="VOI score")
    if observations is not None and observations.n_obs:
        coords = observations.coords[np.isfinite(observations.coords[:, :2]).all(axis=1), :2]
        if len(coords):
            step = max(1, len(coords) // 1200)
            ax.scatter(coords[::step, 0], coords[::step, 1], s=4, c="white", alpha=0.28, linewidths=0, label="existing observations")
    for line in lines:
        ax.plot(
            [float(line["x_start"]), float(line["x_end"])],
            [float(line["y_start"]), float(line["y_end"])],
            color="#00d4ff",
            linewidth=2.0,
            alpha=0.95,
        )
        ax.text(float(line["x_start"]), float(line["y_start"]), f"L{int(line['rank'])}", color="#00d4ff", fontsize=8)
    if boreholes:
        bx = [row["x"] for row in boreholes]
        by = [row["y"] for row in boreholes]
        ax.scatter(bx, by, s=52, c="#78ff63", edgecolors="black", linewidths=0.6, marker="^", label="recommended boreholes")
        for row in boreholes:
            ax.text(row["x"], row["y"], str(int(row["rank"])), color="black", fontsize=8, ha="center", va="center")
    ax.set_title("Recommended supplemental site investigation")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--posterior", default="outputs/predictions/usgs_real_conditioned_diffusion.npz")
    parser.add_argument("--observations", default="data/processed/usgs_geophysics_observations.npz")
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--exclusion-radius", type=float, default=3.0)
    parser.add_argument("--boreholes", type=int, default=8)
    parser.add_argument("--ert-lines", type=int, default=4)
    parser.add_argument("--min-spacing", type=float, default=8.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.35)
    parser.add_argument("--ice-ambiguity-weight", type=float, default=0.20)
    parser.add_argument("--settlement-weight", type=float, default=0.25)
    parser.add_argument("--differential-weight", type=float, default=0.10)
    parser.add_argument("--novelty-weight", type=float, default=0.10)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    posterior = dict(np.load(args.posterior, allow_pickle=False))
    observations = _load_observations(args.observations)
    components = posterior_score_components(
        posterior,
        observations=observations,
        max_depth=float(args.max_depth),
        exclusion_radius=float(args.exclusion_radius),
    )
    score = build_voi_score(
        components,
        weights=VOIWeights(
            uncertainty=float(args.uncertainty_weight),
            ice_rich_ambiguity=float(args.ice_ambiguity_weight),
            settlement_risk=float(args.settlement_weight),
            differential_settlement=float(args.differential_weight),
            novelty=float(args.novelty_weight),
        ),
    )
    boreholes = recommend_boreholes(
        score,
        posterior,
        components,
        top_k=int(args.boreholes),
        min_spacing=float(args.min_spacing),
        max_depth=float(args.max_depth),
    )
    lines = recommend_ert_lines(score, posterior, top_k=int(args.ert_lines), min_spacing=float(args.min_spacing))
    table_dir = Path(config["paths"]["tables_dir"])
    borehole_path = table_dir / "site_investigation_boreholes.csv"
    line_path = table_dir / "site_investigation_ert_lines.csv"
    _write_csv(borehole_path, boreholes)
    _write_csv(line_path, lines)
    score_path = Path(config["paths"]["predictions_dir"]) / "site_investigation_voi_score.npz"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        score_path,
        score=score.astype(np.float32),
        grid_x=posterior["grid_x"].astype(np.float32),
        grid_y=posterior["grid_y"].astype(np.float32),
        excluded=components.get("excluded", np.zeros_like(score, dtype=bool)).astype(bool),
        weights=np.array(
            [
                float(args.uncertainty_weight),
                float(args.ice_ambiguity_weight),
                float(args.settlement_weight),
                float(args.differential_weight),
                float(args.novelty_weight),
            ],
            dtype=np.float32,
        ),
        weight_names=np.array(
            ["uncertainty", "ice_rich_ambiguity", "settlement_risk", "differential_settlement", "novelty"],
            dtype="U32",
        ),
        **{f"component_{k}": v.astype(np.float32) for k, v in components.items() if k != "excluded"},
    )
    fig_path = Path(config["paths"]["figures_dir"]) / "site_investigation_recommendations.png"
    _plot_recommendations(score, posterior, observations, boreholes, lines, fig_path)
    print(f"boreholes={borehole_path}")
    print(f"ert_lines={line_path}")
    print(f"score={score_path}")
    print(f"figure={fig_path}")
    if boreholes:
        first = boreholes[0]
        print(f"top_borehole=x:{first['x']:.3f}, y:{first['y']:.3f}, score:{first['voi_score']:.6f}")
    if lines:
        first_line = lines[0]
        print(f"top_ert_line={first_line['orientation']} score:{float(first_line['line_score']):.6f}")


if __name__ == "__main__":
    main()
