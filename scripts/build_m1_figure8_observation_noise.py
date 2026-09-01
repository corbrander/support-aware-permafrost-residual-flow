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
OUTPUT = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "figures"
    / "m1_final"
    / "figure9_observation_deletion_and_noise"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_figure09_observation_noise"

INK = "#263238"
BLUE = "#377eb8"
ORANGE = "#d95f02"
PURPLE = "#756bb1"
TEAL = "#1b9e77"
GREY = "#9aa5ab"
RED = "#b23a48"

MODES = [
    ("no_ert", "No ERT"),
    ("no_nmr", "No NMR"),
    ("no_temperature", "No temperature"),
    ("boreholes_only", "Boreholes only"),
    ("half_boreholes", "Half boreholes"),
    ("sparse_boreholes", "Sparse boreholes"),
]
SUPPORT = [
    "support_nrmse_borehole_eic",
    "support_nrmse_borehole_temperature",
    "support_nrmse_ert_log_resistivity",
    "support_nrmse_nmr_unfrozen_water",
    "support_nrmse_alt",
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


def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    deletion_path = TABLES / "m1_observation_deletion_three_seed_detail.csv"
    noise_path = TABLES / "m1_noise_three_seed_summary.csv"
    missing = [path for path in (deletion_path, noise_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Figure 8 cannot be built before the three-seed post-lock aggregates exist: "
            + ", ".join(str(path) for path in missing)
        )
    deletion = pd.read_csv(deletion_path)
    expected_modes = {"all", *[mode for mode, _ in MODES]}
    if set(deletion["observation_mode"].unique()) != expected_modes:
        raise RuntimeError("Observation-deletion aggregate is missing one or more predeclared modes.")
    if deletion["model_seed"].nunique() != 3 or deletion["scene_id"].nunique() != 100:
        raise RuntimeError("Observation-deletion detail is not the complete three-seed, 100-scene suite.")
    return deletion, pd.read_csv(noise_path)


def hierarchical_ci(frame: pd.DataFrame, column: str, seed: int) -> tuple[float, float, float]:
    selected = frame[["model_seed", "scene_id", column]].dropna()
    seeds = np.asarray(sorted(selected["model_seed"].unique()))
    grouped = {
        model_seed: selected.loc[selected["model_seed"] == model_seed, column].to_numpy(float)
        for model_seed in seeds
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(5000, dtype=float)
    for index in range(len(draws)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        means = []
        for model_seed in sampled_seeds:
            values = grouped[model_seed]
            means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        draws[index] = float(np.mean(means))
    point = float(selected.groupby("model_seed")[column].mean().mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def paired_mode_effects(deletion: pd.DataFrame) -> pd.DataFrame:
    full = deletion.loc[deletion["observation_mode"] == "all"].copy()
    full["support_score"] = np.nanmean(np.log1p(full[SUPPORT].to_numpy(float)), axis=1)
    rows: list[dict[str, float | str]] = []
    for index, (mode, label) in enumerate(MODES):
        current = deletion.loc[deletion["observation_mode"] == mode].copy()
        current["support_score"] = np.nanmean(np.log1p(current[SUPPORT].to_numpy(float)), axis=1)
        paired = current.merge(
            full[["model_seed", "scene_id", "eic_rmse", "support_score"]],
            on=["model_seed", "scene_id"],
            suffixes=("_deleted", "_all"),
            validate="one_to_one",
        )
        if len(paired) != 300:
            raise RuntimeError(f"Incomplete paired deletion mode: {mode}")
        paired["delta_eic_rmse"] = paired["eic_rmse_deleted"] - paired["eic_rmse_all"]
        paired["delta_support_score"] = paired["support_score_deleted"] - paired["support_score_all"]
        for metric in ("delta_eic_rmse", "delta_support_score"):
            mean, low, high = hierarchical_ci(paired, metric, 81_000 + 100 * index + len(rows))
            rows.append(
                {
                    "mode": mode,
                    "label": label,
                    "metric": metric,
                    "mean": mean,
                    "ci95_lower": low,
                    "ci95_upper": high,
                    "abstain_fraction": float(current["ood_abstain"].astype(float).mean()),
                    "exact_fallback_fraction": float(
                        current["exact_anchor_fallback_applied"].astype(float).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def forest(ax: plt.Axes, effects: pd.DataFrame, metric: str, letter: str, title: str, xlabel: str) -> None:
    panel_title(ax, letter, title)
    rows = effects.loc[effects["metric"] == metric].set_index("mode").loc[[mode for mode, _ in MODES]]
    means = rows["mean"].to_numpy(float)
    low = rows["ci95_lower"].to_numpy(float)
    high = rows["ci95_upper"].to_numpy(float)
    y = np.arange(len(MODES))[::-1]
    ax.errorbar(
        means,
        y,
        xerr=np.vstack((means - low, high - means)),
        fmt="o",
        color=BLUE if metric == "delta_eic_rmse" else TEAL,
        ms=4.0,
        lw=0.9,
        capsize=2,
    )
    ax.axvline(0, color=INK, lw=0.75)
    ax.set_yticks(y, [label for _, label in MODES])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="0.92", lw=0.5)
    # Reserve a dedicated annotation strip to the right of zero.  This keeps
    # controller-fallback labels from covering the point estimates or their
    # confidence intervals, especially when every support-score change is
    # negative and zero would otherwise sit at the right axis boundary.
    data_min = float(min(np.min(low), 0.0))
    data_max = float(max(np.max(high), 0.0))
    span = max(data_max - data_min, np.finfo(float).eps)
    ax.set_xlim(data_min - 0.08 * span, data_max + 0.32 * span)
    for position, fraction in zip(y, rows["exact_fallback_fraction"].to_numpy(float), strict=True):
        if fraction > 0:
            ax.text(
                0.985,
                position,
                f"fallback {fraction:.0%}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=5.5,
                color=RED,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.6},
            )


def noise_row(noise: pd.DataFrame, multiplier: float, metric: str) -> pd.Series:
    selected = noise.loc[(noise["multiplier"] == multiplier) & (noise["metric"] == metric)]
    if selected.empty:
        raise KeyError(f"Missing noise result: multiplier={multiplier}, metric={metric}")
    return selected.iloc[0]


def plot_spread(ax: plt.Axes, noise: pd.DataFrame) -> None:
    panel_title(ax, "c", "Declared noise and posterior spread")
    multipliers = np.asarray(sorted(noise["multiplier"].unique()), dtype=float)
    rows = [noise_row(noise, value, "posterior_spread_mean") for value in multipliers]
    mean = np.asarray([float(row["mean"]) for row in rows])
    low = np.asarray([float(row["ci95_lower"]) for row in rows])
    high = np.asarray([float(row["ci95_upper"]) for row in rows])
    ax.plot(multipliers, mean, marker="o", color=PURPLE, lw=1.3, ms=4.0)
    ax.fill_between(multipliers, low, high, color=PURPLE, alpha=0.18, linewidth=0)
    ax.axvline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(multipliers, [f"{value:g}x" for value in multipliers])
    ax.set_xlabel("Declared uncertainty multiplier")
    ax.set_ylabel("Mean posterior spread")
    ax.grid(color="0.92", lw=0.5)


def plot_response(ax: plt.Axes, noise: pd.DataFrame) -> None:
    panel_title(ax, "d", "Locality of the noise response")
    multipliers = np.asarray(sorted(noise["multiplier"].unique()), dtype=float)
    for metric, label, color, marker in [
        ("mean_shift_from_nominal", "global mean shift", ORANGE, "o"),
        ("distant_shift_from_nominal", "distant mean shift", BLUE, "s"),
    ]:
        rows = [noise_row(noise, value, metric) for value in multipliers]
        mean = np.asarray([float(row["mean"]) for row in rows])
        low = np.asarray([float(row["ci95_lower"]) for row in rows])
        high = np.asarray([float(row["ci95_upper"]) for row in rows])
        ax.plot(multipliers, mean, marker=marker, color=color, lw=1.2, ms=3.7, label=label)
        ax.fill_between(multipliers, low, high, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(multipliers, [f"{value:g}x" for value in multipliers])
    ax.set_xlabel("Declared uncertainty multiplier")
    ax.set_ylabel("Absolute EIC mean shift")
    ax.grid(color="0.92", lw=0.5)
    twin = ax.twinx()
    radius = np.asarray([float(noise_row(noise, value, "influence_radius_m")["mean"]) for value in multipliers])
    twin.plot(multipliers, radius, color=TEAL, marker="^", lw=1.2, ms=3.8, label="influence radius")
    twin.set_ylabel("Influence radius (m)", color=TEAL)
    twin.tick_params(axis="y", labelcolor=TEAL)
    lines = ax.get_lines()[:2] + twin.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], fontsize=5.6, loc="best")


def main() -> None:
    apply_m1_style()
    deletion, noise = require_inputs()
    effects = paired_mode_effects(deletion)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    effects.to_csv(SOURCE / "figure9_paired_deletion_effects.csv", index=False)
    noise.to_csv(SOURCE / "figure9_noise_summary.csv", index=False)
    deletion[
        [
            "model_seed",
            "scene_id",
            "observation_mode",
            "eic_rmse",
            "ood_abstain",
            "exact_anchor_fallback_applied",
            *SUPPORT,
        ]
    ].to_csv(SOURCE / "figure9_deletion_scene_level.csv", index=False)
    (SOURCE / "metadata.json").write_text(
        json.dumps(
            {
                "deletion_pairing": "Same model seed and scene; the tree anchor is rebuilt after deletion.",
                "posterior_members_deletion": 64,
                "sampling_steps_deletion": 10,
                "noise_members": 16,
                "noise_sampling_steps": 5,
                "fallback_annotation": "Exact fallback fractions are shown rather than interpreting fallback equality as learned-model robustness.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 124 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.91, top=0.94, bottom=0.105, wspace=0.45, hspace=0.44)
    forest(axes[0, 0], effects, "delta_eic_rmse", "a", "Information loss and EIC error", "Deleted minus all-data EIC RMSE")
    forest(
        axes[0, 1],
        effects,
        "delta_support_score",
        "b",
        "Active-support score shift",
        "Deleted minus all-data active-support score",
    )
    plot_spread(axes[1, 0], noise)
    plot_response(axes[1, 1], noise)
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
