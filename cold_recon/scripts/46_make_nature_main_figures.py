from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.data.data_schema import OBS_TYPES
from cold_recon.utils.config import ensure_dirs, load_config


PALETTE = {
    "baseline": "#767676",
    "baseline_light": "#CFCECE",
    "cold": "#0F4D92",
    "cold_light": "#3775BA",
    "green": "#2E9E44",
    "red": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#C9A227",
    "black": "#272727",
}


def _apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False


def _read_table(table_dir: Path, name: str) -> pd.DataFrame:
    path = table_dir / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _panel(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _save(fig: plt.Figure, figure_dir: Path, stem: str, dpi: int = 600) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "pdf", "png", "tiff"):
        out = figure_dir / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(out, **kwargs)
        paths.append(out)
    plt.close(fig)
    return paths


def _write_source(path: Path, frames: Iterable[pd.DataFrame]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [df for df in frames if df is not None and not df.empty]
    out = pd.concat(clean, ignore_index=True, sort=False) if clean else pd.DataFrame()
    out.to_csv(path, index=False)
    return path


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _workflow_panel(ax: plt.Axes) -> pd.DataFrame:
    ax.set_axis_off()
    boxes = [
        ("Field\nobservations", "boreholes\nERT/NMR/ALT", 0.03, 0.58, 0.25, 0.26, PALETTE["baseline_light"]),
        ("Observation\ntokens", "x, y, z, type\nvalue, sigma", 0.38, 0.58, 0.22, 0.26, "#DCEAF7"),
        ("Conditional\nposterior", "latent diffusion\nneural operator", 0.71, 0.58, 0.25, 0.26, "#CFE3F6"),
        ("Physics\nprojection", "thermal water\nresistivity", 0.19, 0.13, 0.25, 0.26, "#DDF3DE"),
        ("Verified\noutputs", "3D cryofacies\nEIC posterior", 0.57, 0.13, 0.28, 0.26, "#E8D7E7"),
    ]
    rows = []
    for title, body, x, y, w, h, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=PALETTE["black"], lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.73, title, ha="center", va="center", fontsize=5.4, fontweight="bold", linespacing=0.86)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=5.1, linespacing=0.98)
        rows.append({"panel": "1a", "step": title, "description": body.replace("\n", "; ")})
    arrows = [((0.28, 0.71), (0.38, 0.71)), ((0.60, 0.71), (0.71, 0.71)), ((0.82, 0.58), (0.70, 0.39)), ((0.44, 0.26), (0.57, 0.26))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 0.9, "color": PALETTE["black"]})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return pd.DataFrame(rows)


def make_figure_1(table_dir: Path, figure_dir: Path, source_dir: Path) -> list[Path]:
    model = _read_table(table_dir, "model_comparison.csv")
    calib = _read_table(table_dir, "uncertainty_calibration_metrics.csv")
    calib_scaled = _read_table(table_dir, "uncertainty_calibration_metrics_calibrated.csv")
    physics = _read_table(table_dir, "physics_consistency_metrics.csv")

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    src_a = _workflow_panel(ax_a)
    _panel(ax_a, "a", x=-0.02)

    selected = [
        "IDW",
        "RandomForest",
        "GradientBoosting",
        "KrigingGPR",
        "SparseUNet3D",
        "COLDReconImplicit",
        "COLDReconLatentDiffusion",
        "COLDReconFNOOperatorDiffusion",
        "COLDReconRectifiedFlow",
        "COLDReconLatentDiffusionPhysicsTrained",
    ]
    labels = {
        "RandomForest": "RF",
        "GradientBoosting": "GB",
        "KrigingGPR": "Kriging",
        "SparseUNet3D": "3D U-Net",
        "COLDReconImplicit": "Implicit",
        "COLDReconLatentDiffusion": "Latent diff.",
        "COLDReconFNOOperatorDiffusion": "FNO diff.",
        "COLDReconRectifiedFlow": "Rectified flow",
        "COLDReconLatentDiffusionPhysicsTrained": "Physics-trained",
    }
    view = model[model["model"].isin(selected)].copy()
    view["order"] = view["model"].map({m: i for i, m in enumerate(selected)})
    view = view.sort_values("order")
    y = np.arange(len(view))
    colors = [PALETTE["baseline"] if not str(m).startswith("COLDRecon") else PALETTE["cold"] for m in view["model"]]
    ax_b.barh(y, view["mean_iou"].astype(float), color=colors, height=0.74)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([labels.get(str(m), str(m)) for m in view["model"]])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("mean facies IoU")
    ax_b.set_xlim(0, max(0.62, float(view["mean_iou"].max()) + 0.05))
    ax_b.grid(axis="x", color="0.9", lw=0.6)
    _panel(ax_b, "b")

    targets = ["eic", "temperature", "unfrozen_water", "log_resistivity"]
    before = calib[calib["target"].isin(targets)][["target", "coverage_90"]].copy()
    before["calibration"] = "raw"
    after = calib_scaled[calib_scaled["target"].isin(targets)][["target", "coverage_90"]].copy()
    after["calibration"] = "posthoc calibrated"
    cov = pd.concat([before, after], ignore_index=True)
    x = np.arange(len(targets))
    width = 0.36
    raw_vals = [float(before.loc[before["target"] == t, "coverage_90"].iloc[0]) if t in set(before["target"]) else np.nan for t in targets]
    scaled_vals = [float(after.loc[after["target"] == t, "coverage_90"].iloc[0]) if t in set(after["target"]) else np.nan for t in targets]
    ax_c.bar(x - width / 2, raw_vals, width=width, color=PALETTE["baseline"], label="raw")
    ax_c.bar(x + width / 2, scaled_vals, width=width, color=PALETTE["cold_light"], label="posthoc")
    ax_c.axhline(0.9, color=PALETTE["red"], lw=0.9, ls="--")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["EIC", "Temp.", "Water", "Log rho"])
    ax_c.set_ylabel("90% interval coverage")
    ax_c.set_ylim(0, 1.02)
    ax_c.legend(loc="lower right", fontsize=6)
    ax_c.grid(axis="y", color="0.9", lw=0.6)
    _panel(ax_c, "c")

    phys_sel = [
        "truth",
        "IDW",
        "GradientBoosting",
        "COLDReconLatentDiffusion",
        "COLDReconLatentDiffusionPhysicsTrained",
        "COLDReconLatentDiffusionPhysicsRefined",
    ]
    phys = physics[physics["model"].isin(phys_sel)].copy()
    phys["order"] = phys["model"].map({m: i for i, m in enumerate(phys_sel)})
    phys = phys.sort_values("order")
    short = {
        "COLDReconLatentDiffusion": "Latent diff.",
        "COLDReconLatentDiffusionPhysicsTrained": "Physics-trained",
        "COLDReconLatentDiffusionPhysicsRefined": "Physics-refined",
        "GradientBoosting": "GB",
    }
    ax_d.barh(np.arange(len(phys)), phys["unfrozen_water_empirical_mae"].astype(float), color=[PALETTE["baseline"] if not str(m).startswith("COLDRecon") else PALETTE["teal"] for m in phys["model"]])
    ax_d.set_yticks(np.arange(len(phys)))
    ax_d.set_yticklabels([short.get(str(m), str(m)) for m in phys["model"]])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("unfrozen-water consistency MAE")
    ax_d.grid(axis="x", color="0.9", lw=0.6)
    _panel(ax_d, "d")

    fig.suptitle("COLD-Recon converts sparse permafrost observations into verified 3D posteriors", y=0.995, fontsize=9)
    src_b = view.assign(panel="1b")[["panel", "model", "mean_iou", "eic_rmse", "temperature_rmse", "unfrozen_water_rmse"]]
    src_c = cov.assign(panel="1c")
    src_d = phys.assign(panel="1d")[["panel", "model", "domain", "unfrozen_water_empirical_mae", "heat_residual_rmse", "log_resistivity_empirical_mae"]]
    _write_source(source_dir / "nature_figure_1_source_data.csv", [src_a, src_b, src_c, src_d])
    return _save(fig, figure_dir, "nature_figure_1_overview")


def make_figure_2(table_dir: Path, figure_dir: Path, source_dir: Path) -> list[Path]:
    tokens = _read_table(table_dir, "public_data_token_inventory.csv")
    gate = _read_table(table_dir, "real_data_cg_benchmark.csv")
    fig = plt.figure(figsize=(7.2, 6.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.46)
    ax_a = fig.add_subplot(gs[:, 0])
    right = gs[0, 1].subgridspec(3, 1, height_ratios=[1.35, 0.82, 0.62], hspace=0.82)
    ax_b = fig.add_subplot(right[0, 0])
    ax_c = fig.add_subplot(right[1, 0])
    ax_d = fig.add_subplot(right[2, 0])

    source_labels = {
        "arcticdata_jago_ground_ice_2018": "Jago 2018",
        "arcticdata_upper_permafrost_cryostratigraphy": "ArcticData cryo.",
        "usgs_eic_cores": "USGS cores",
        "usgs_ert_nmr": "USGS ERT/NMR",
    }
    obs_labels = {
        "borehole_eic": "EIC",
        "borehole_facies": "facies",
        "alt": "ALT",
        "ert_log_resistivity": "ERT log rho",
        "nmr_unfrozen_water": "NMR water",
    }
    view = tokens.copy()
    view["label"] = [
        f"{source_labels.get(str(src), str(src))} - {obs_labels.get(str(obs), str(obs))}"
        for src, obs in zip(view["source_key"], view["observation_type"])
    ]
    view = view.sort_values("n_tokens", ascending=True)
    color_map = {
        "usgs_ert_nmr": PALETTE["cold_light"],
        "usgs_eic_cores": PALETTE["gold"],
        "arcticdata_upper_permafrost_cryostratigraphy": PALETTE["green"],
        "arcticdata_jago_ground_ice_2018": PALETTE["violet"],
    }
    y = np.arange(len(view))
    counts = view["n_tokens"].astype(int).to_numpy()
    ax_a.barh(y, counts, color=[color_map.get(str(src), PALETTE["baseline"]) for src in view["source_key"]], height=0.70)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(view["label"])
    ax_a.set_xscale("log")
    ax_a.set_xlabel("observation tokens (log scale)")
    ax_a.grid(axis="x", color="0.9", which="both", lw=0.6)
    for yi, count in zip(y, counts):
        ax_a.text(max(count * 1.08, 1.1), yi, f"{count:,}", va="center", fontsize=6)
    ax_a.set_xlim(1, max(counts) * 3.0)
    _panel(ax_a, "a")

    gate_view = gate.copy()
    gate_source_labels = {
        "ArcticData cryostratigraphy": "ArcticData",
        "USGS EIC cores": "USGS cores",
        "ArcticData Jago River 2018 ground ice": "Jago 2018",
    }
    gate_task_labels = {
        "cryofacies": "cryofacies",
        "EIC regression": "EIC regression",
        "wedge-ice recall": "wedge recall",
        "high-EIC event": "high-EIC event",
    }
    gate_view["label"] = [
        f"{gate_source_labels.get(str(src), str(src))} {gate_task_labels.get(str(task), str(task))}"
        for src, task in zip(gate_view["source"], gate_view["task"])
    ]
    gate_view["relative_improvement"] = gate_view["relative_improvement"].astype(float)
    gate_view = gate_view.sort_values("relative_improvement")
    colors = [PALETTE["green"] if str(p).lower() == "true" else PALETTE["red"] for p in gate_view["passed"]]
    yy = np.arange(len(gate_view))
    ax_b.axvline(0, color=PALETTE["black"], lw=0.8)
    ax_b.barh(yy, gate_view["relative_improvement"], color=colors, height=0.70)
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels(gate_view["label"], fontsize=5.2)
    ax_b.set_xlabel("relative improvement")
    ax_b.grid(axis="x", color="0.9", lw=0.6)
    _panel(ax_b, "b")

    eic = gate[gate["task"].eq("EIC regression")].copy()
    eic["short_source"] = eic["source"].map(
        {
            "ArcticData cryostratigraphy": "ArcticData",
            "USGS EIC cores": "USGS cores",
            "ArcticData Jago River 2018 ground ice": "Jago 2018",
        }
    ).fillna(eic["source"])
    x = np.arange(len(eic))
    width = 0.36
    ax_c.bar(x - width / 2, eic["baseline_value"].astype(float), width=width, color=PALETTE["baseline"], label="best simple")
    ax_c.bar(x + width / 2, eic["model_value"].astype(float), width=width, color=PALETTE["cold"], label="COLD-Recon")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(eic["short_source"], rotation=18, ha="right")
    ax_c.set_ylabel("EIC RMSE")
    ax_c.legend(fontsize=6)
    ax_c.grid(axis="y", color="0.9", lw=0.6)
    _panel(ax_c, "c")

    source_order = ["ArcticData cryostratigraphy", "USGS EIC cores", "ArcticData Jago River 2018 ground ice"]
    source_short = {
        "ArcticData cryostratigraphy": "ArcticData",
        "USGS EIC cores": "USGS",
        "ArcticData Jago River 2018 ground ice": "Jago 2018",
    }
    task_order = ["cryofacies", "EIC regression", "wedge-ice recall", "high-EIC event"]
    task_short = {
        "cryofacies": "cryo.",
        "EIC regression": "EIC",
        "wedge-ice recall": "wedge",
        "high-EIC event": "high EIC",
    }
    status = np.full((len(source_order), len(task_order)), np.nan, dtype=float)
    panel_d_rows: list[dict[str, object]] = []
    for i, source in enumerate(source_order):
        for j, task in enumerate(task_order):
            rows = gate[(gate["source"].astype(str) == source) & (gate["task"].astype(str) == task)]
            if rows.empty:
                panel_d_rows.append({"panel": "2d", "source": source, "task": task, "status": "not_tested"})
                continue
            passed = str(rows.iloc[0]["passed"]).strip().lower() == "true"
            status[i, j] = 1.0 if passed else 0.0
            panel_d_rows.append({"panel": "2d", "source": source, "task": task, "status": "pass" if passed else "fail"})
    cmap = matplotlib.colors.ListedColormap([PALETTE["red"], PALETTE["green"]])
    cmap.set_bad("#EFEFEF")
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax_d.imshow(np.ma.masked_invalid(status), cmap=cmap, norm=norm, aspect="auto")
    ax_d.set_xticks(np.arange(len(task_order)))
    ax_d.set_xticklabels([task_short[t] for t in task_order], rotation=30, ha="right", fontsize=5.6)
    ax_d.set_yticks(np.arange(len(source_order)))
    ax_d.set_yticklabels([source_short[s] for s in source_order], fontsize=5.8)
    for i in range(status.shape[0]):
        for j in range(status.shape[1]):
            if np.isnan(status[i, j]):
                label = "-"
            else:
                label = "P" if status[i, j] > 0.5 else "F"
            ax_d.text(j, i, label, ha="center", va="center", fontsize=5.4, color=PALETTE["black"])
    ax_d.set_title("gate matrix", fontsize=7)
    ax_d.tick_params(length=0)
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    _panel(ax_d, "d", x=-0.08, y=1.04)

    fig.suptitle("Three public data sources support the real-data evidence gate", y=0.995, fontsize=9)
    _write_source(
        source_dir / "nature_figure_2_source_data.csv",
        [view.assign(panel="2a"), gate_view.assign(panel="2b"), eic.assign(panel="2c"), pd.DataFrame(panel_d_rows)],
    )
    return _save(fig, figure_dir, "nature_figure_2_real_data_gate")


def _load_observation_npz_values(path: Path, obs_type: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = np.load(path, allow_pickle=False)
    keep = data["obs_type_ids"].astype(int) == int(obs_type)
    return pd.DataFrame(
        {
            "value": data["obs_values"][keep].astype(float),
            "z": data["obs_coords"][keep, 2].astype(float),
        }
    )


def make_figure_3(root: Path, table_dir: Path, figure_dir: Path, source_dir: Path) -> list[Path]:
    arctic = _read_table(table_dir, "arcticdata_cryostratigraphy_token_index.csv")
    jago = _read_table(table_dir, "arcticdata_jago_ground_ice_token_index.csv")
    jago_pred = _read_table(table_dir, "arcticdata_jago_ground_ice_conditioned_diffusion_holdout_predictions.csv")
    usgs_pred = _read_table(table_dir, "usgs_eic_conditioned_diffusion_holdout_predictions.csv")
    usgs_values = _load_observation_npz_values(root / "data" / "processed" / "usgs_eic_observations.npz", OBS_TYPES["borehole_eic"])

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    dist_frames = []
    if not arctic.empty:
        dist_frames.append(pd.DataFrame({"source": "ArcticData", "eic": arctic.loc[arctic["type_id"].astype(int) == OBS_TYPES["borehole_eic"], "value"].astype(float)}))
    if not usgs_values.empty:
        dist_frames.append(pd.DataFrame({"source": "USGS cores", "eic": usgs_values["value"].astype(float)}))
    if not jago.empty:
        dist_frames.append(pd.DataFrame({"source": "Jago 2018", "eic": jago["value"].astype(float)}))
    dist = pd.concat(dist_frames, ignore_index=True)
    order = ["ArcticData", "USGS cores", "Jago 2018"]
    data = [dist.loc[dist["source"] == src, "eic"].dropna().to_numpy() for src in order]
    bp = ax_a.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False, medianprops={"color": PALETTE["black"]})
    for patch, color in zip(bp["boxes"], [PALETTE["green"], PALETTE["gold"], PALETTE["violet"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(10)
    for i, vals in enumerate(data, start=1):
        if vals.size:
            sample = vals if vals.size <= 250 else rng.choice(vals, size=250, replace=False)
            ax_a.scatter(rng.normal(i, 0.04, size=sample.size), sample, s=5, color=PALETTE["black"], alpha=0.20, linewidths=0)
    ax_a.set_ylabel("measured EIC fraction")
    ax_a.set_ylim(-0.03, 1.03)
    ax_a.grid(axis="y", color="0.9", lw=0.6)
    _panel(ax_a, "a")

    if not jago.empty:
        jago = jago.copy()
        boreholes = {name: i for i, name in enumerate(sorted(jago["BOREHOLE_ID"].astype(str).unique()))}
        sc = ax_b.scatter(
            jago["BOREHOLE_ID"].map(boreholes),
            jago["z"].astype(float),
            c=jago["value"].astype(float),
            s=28,
            cmap="viridis",
            vmin=0,
            vmax=max(0.6, float(jago["value"].max())),
            edgecolors="black",
            linewidths=0.2,
        )
        ax_b.invert_yaxis()
        ax_b.set_xlabel("ordered Jago borehole")
        ax_b.set_ylabel("depth (m)")
        ax_b.set_title("Jago measured ground ice", fontsize=8)
        fig.colorbar(sc, ax=ax_b, label="EIC", fraction=0.046, pad=0.02)
    _panel(ax_b, "b")

    if not jago_pred.empty:
        show_models = ["GlobalMean", "SpatialDepthIDW", "COLDReconJagoGroundIceConditionedDiffusion"]
        colors = {"GlobalMean": PALETTE["baseline"], "SpatialDepthIDW": PALETTE["gold"], "COLDReconJagoGroundIceConditionedDiffusion": PALETTE["cold"]}
        for model_name in show_models:
            group = jago_pred[jago_pred["model"] == model_name]
            if group.empty:
                continue
            ax_c.scatter(group["observed_eic"], group["predicted_eic"], s=18, alpha=0.75, label=model_name.replace("COLDReconJagoGroundIceConditionedDiffusion", "COLD-Recon"), color=colors[model_name])
        ax_c.plot([0, 0.65], [0, 0.65], color=PALETTE["black"], lw=0.8, ls="--")
        ax_c.set_xlim(-0.02, 0.65)
        ax_c.set_ylim(-0.02, 0.65)
        ax_c.set_xlabel("observed EIC")
        ax_c.set_ylabel("predicted EIC")
        ax_c.legend(fontsize=5.8)
        ax_c.grid(color="0.9", lw=0.6)
    _panel(ax_c, "c")

    if not usgs_pred.empty:
        show_models = ["SpatialDepthIDW", "COLDReconUSGSEICConditionedDiffusion"]
        colors = {"SpatialDepthIDW": PALETTE["gold"], "COLDReconUSGSEICConditionedDiffusion": PALETTE["cold"]}
        for model_name in show_models:
            group = usgs_pred[usgs_pred["model"] == model_name]
            if group.empty:
                continue
            if len(group) > 250:
                group = group.sample(250, random_state=11)
            ax_d.scatter(group["observed_eic"], group["predicted_eic"], s=8, alpha=0.45, label=model_name.replace("COLDReconUSGSEICConditionedDiffusion", "COLD-Recon"), color=colors[model_name], linewidths=0)
        ax_d.plot([0, 1], [0, 1], color=PALETTE["black"], lw=0.8, ls="--")
        ax_d.set_xlim(-0.03, 1.03)
        ax_d.set_ylim(-0.03, 1.03)
        ax_d.set_xlabel("observed EIC")
        ax_d.set_ylabel("predicted EIC")
        ax_d.legend(fontsize=5.8)
        ax_d.grid(color="0.9", lw=0.6)
    _panel(ax_d, "d")

    fig.suptitle("Cited ground-ice data are explicitly plotted and traced to model validation", y=0.995, fontsize=9)
    _write_source(
        source_dir / "nature_figure_3_source_data.csv",
        [
            dist.assign(panel="3a"),
            jago.assign(panel="3b") if not jago.empty else pd.DataFrame(),
            jago_pred.assign(panel="3c"),
            usgs_pred.assign(panel="3d"),
        ],
    )
    return _save(fig, figure_dir, "nature_figure_3_cited_ground_ice")


def _load_observation_xy(path: Path, max_points: int = 1400) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = np.load(path, allow_pickle=False)
    if "obs_coords" not in data.files:
        return pd.DataFrame()
    coords = np.asarray(data["obs_coords"], dtype=float)
    keep = np.isfinite(coords[:, :2]).all(axis=1)
    coords = coords[keep, :2]
    if coords.size == 0:
        return pd.DataFrame()
    if len(coords) > max_points:
        idx = np.linspace(0, len(coords) - 1, max_points).astype(int)
        coords = coords[idx]
    return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})


def make_figure_4(root: Path, table_dir: Path, figure_dir: Path, source_dir: Path) -> list[Path]:
    boreholes = _read_table(table_dir, "site_investigation_boreholes.csv")
    lines = _read_table(table_dir, "site_investigation_ert_lines.csv")
    score_path = table_dir.parent / "predictions" / "site_investigation_voi_score.npz"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing VOI score file: {score_path}")
    voi = np.load(score_path, allow_pickle=False)
    score = np.asarray(voi["score"], dtype=float)
    excluded = np.asarray(voi["excluded"], dtype=bool) if "excluded" in voi.files else ~np.isfinite(score)
    x = np.asarray(voi["grid_x"], dtype=float) if "grid_x" in voi.files else np.arange(score.shape[0], dtype=float)
    y = np.asarray(voi["grid_y"], dtype=float) if "grid_y" in voi.files else np.arange(score.shape[1], dtype=float)
    obs_xy = _load_observation_xy(root / "data" / "processed" / "usgs_geophysics_observations.npz")

    component_cols = ["uncertainty", "ice_rich_ambiguity", "settlement_risk", "differential_settlement", "novelty"]
    component_labels = {
        "uncertainty": "uncertainty",
        "ice_rich_ambiguity": "ice-rich ambiguity",
        "settlement_risk": "thaw-sensitive EIC",
        "differential_settlement": "EIC gradient",
        "novelty": "distance from data",
    }
    component_colors = {
        "uncertainty": PALETTE["cold"],
        "ice_rich_ambiguity": PALETTE["gold"],
        "settlement_risk": PALETTE["red"],
        "differential_settlement": PALETTE["teal"],
        "novelty": PALETTE["violet"],
    }
    if "weight_names" in voi.files and "weights" in voi.files:
        weights = {str(name): float(weight) for name, weight in zip(voi["weight_names"], voi["weights"])}
    else:
        weights = {
            "uncertainty": 0.35,
            "ice_rich_ambiguity": 0.20,
            "settlement_risk": 0.25,
            "differential_settlement": 0.10,
            "novelty": 0.10,
        }

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.38, 0.78, 0.82], height_ratios=[1.05, 0.95], wspace=0.42, hspace=0.50)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    heat = np.ma.masked_where(~np.isfinite(score) | excluded, score)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#EFEFEF")
    im = ax_a.imshow(
        heat.T,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=max(0.7, float(np.nanmax(np.where(np.isfinite(score), score, np.nan)))),
    )
    if not obs_xy.empty:
        ax_a.scatter(obs_xy["x"], obs_xy["y"], s=1.5, c="#F7F7F7", alpha=0.075, linewidths=0, label="existing observations")
    if not lines.empty:
        for _, row in lines.iterrows():
            ax_a.plot(
                [float(row["x_start"]), float(row["x_end"])],
                [float(row["y_start"]), float(row["y_end"])],
                color="#A7E7F2",
                lw=1.1,
                alpha=0.95,
            )
            tx = 0.5 * (float(row["x_start"]) + float(row["x_end"]))
            ty = 0.5 * (float(row["y_start"]) + float(row["y_end"]))
            ax_a.text(
                tx,
                ty,
                f"L{int(float(row['rank']))}",
                color="#083D4F",
                fontsize=5.2,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "alpha": 0.62, "edgecolor": "none", "pad": 0.25},
            )
    if not boreholes.empty:
        ax_a.scatter(
            boreholes["x"].astype(float),
            boreholes["y"].astype(float),
            s=34,
            c="#DDF3DE",
            edgecolors=PALETTE["black"],
            linewidths=0.55,
            marker="^",
            label="recommended boreholes",
            zorder=5,
        )
        for _, row in boreholes.iterrows():
            ax_a.text(float(row["x"]), float(row["y"]), str(int(float(row["rank"]))), fontsize=4.8, ha="center", va="center", color=PALETTE["black"], zorder=6)
    ax_a.set_xlabel("local x (m)")
    ax_a.set_ylabel("local y (m)")
    ax_a.set_title("posterior value-of-information surface", fontsize=8)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.045, pad=0.025)
    cb.set_label("VOI score", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.8)
    _panel(ax_a, "a", x=-0.05, y=1.02)

    bore_view = boreholes.sort_values("rank").copy() if not boreholes.empty else pd.DataFrame()
    if not bore_view.empty:
        labels = [f"B{int(float(rank))}" for rank in bore_view["rank"]]
        y_pos = np.arange(len(bore_view))
        ax_b.barh(y_pos, bore_view["voi_score"].astype(float), color=PALETTE["cold"], height=0.72)
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(labels)
        ax_b.invert_yaxis()
        ax_b.set_xlabel("VOI score")
        ax_b.set_xlim(0, max(0.75, float(bore_view["voi_score"].astype(float).max()) + 0.06))
        ax_b.grid(axis="x", color="0.9", lw=0.6)
        for yi, row in zip(y_pos, bore_view.itertuples(index=False)):
            ax_b.text(float(row.voi_score) + 0.01, yi, f"x={float(row.x):.0f}, y={float(row.y):.0f}", va="center", fontsize=5.6)
    ax_b.set_title("ranked observation boreholes", fontsize=8)
    _panel(ax_b, "b")

    top_components = bore_view.head(6).copy()
    if not top_components.empty:
        labels = [f"B{int(float(rank))}" for rank in top_components["rank"]]
        yy = np.arange(len(top_components))
        left = np.zeros(len(top_components), dtype=float)
        for col in component_cols:
            if col not in top_components.columns:
                continue
            values = top_components[col].astype(float).to_numpy() * float(weights.get(col, 0.0))
            ax_c.barh(yy, values, left=left, color=component_colors[col], height=0.70, label=component_labels[col])
            left += values
        ax_c.set_yticks(yy)
        ax_c.set_yticklabels(labels)
        ax_c.invert_yaxis()
        ax_c.set_xlabel("weighted contribution")
        ax_c.set_xlim(0, max(0.75, float(left.max()) + 0.04))
        ax_c.grid(axis="x", color="0.92", lw=0.55)
        ax_c.legend(fontsize=4.6, loc="upper left", bbox_to_anchor=(0.0, -0.28), handlelength=0.9, borderaxespad=0.1)
    ax_c.set_title("score components", fontsize=8)
    _panel(ax_c, "c")

    line_view = lines.sort_values("rank").copy() if not lines.empty else pd.DataFrame()
    if not line_view.empty:
        labels = [f"L{int(float(row.rank))} {str(row.orientation)}" for row in line_view.itertuples(index=False)]
        yy = np.arange(len(line_view))
        ax_d.barh(yy, line_view["line_score"].astype(float), color=PALETTE["teal"], height=0.70, label="mean line score")
        ax_d.scatter(line_view["max_score"].astype(float), yy, marker="D", s=18, color=PALETTE["black"], label="max cell")
        ax_d.set_yticks(yy)
        ax_d.set_yticklabels(labels)
        ax_d.invert_yaxis()
        ax_d.set_xlabel("VOI score")
        ax_d.set_xlim(0, max(0.75, float(line_view["max_score"].astype(float).max()) + 0.05))
        ax_d.grid(axis="x", color="0.92", lw=0.55)
        ax_d.legend(fontsize=5.0, loc="upper left", bbox_to_anchor=(0.0, -0.28), handlelength=1.0, borderaxespad=0.1)
    ax_d.set_title("ranked observation lines", fontsize=8)
    _panel(ax_d, "d")

    fig.suptitle("Posterior uncertainty is converted into auditable observation targets", y=0.995, fontsize=9)

    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid_source = pd.DataFrame(
        {
            "panel": "4a",
            "x": xx.ravel(),
            "y": yy.ravel(),
            "voi_score": score.ravel(),
            "excluded": excluded.ravel(),
        }
    )
    for col in component_cols + ["settlement_potential"]:
        key = f"component_{col}"
        if key in voi.files:
            grid_source[col] = np.asarray(voi[key], dtype=float).ravel()
    bore_source = boreholes.assign(panel="4b_4c") if not boreholes.empty else pd.DataFrame()
    if not bore_source.empty:
        for col in component_cols:
            if col in bore_source.columns:
                bore_source[f"weighted_{col}"] = bore_source[col].astype(float) * float(weights.get(col, 0.0))
    line_source = lines.assign(panel="4d") if not lines.empty else pd.DataFrame()
    source_rename = {
        "settlement_risk": "thaw_sensitive_eic_proxy",
        "differential_settlement": "eic_gradient_proxy",
        "settlement_potential": "thaw_sensitive_eic_raw",
        "weighted_settlement_risk": "weighted_thaw_sensitive_eic_proxy",
        "weighted_differential_settlement": "weighted_eic_gradient_proxy",
    }
    grid_source = grid_source.rename(columns=source_rename)
    bore_source = bore_source.rename(columns=source_rename)
    weight_component_names = {
        "settlement_risk": "thaw_sensitive_eic_proxy",
        "differential_settlement": "eic_gradient_proxy",
    }
    weight_source = pd.DataFrame(
        {
            "panel": "4c_weights",
            "component": [weight_component_names.get(str(name), str(name)) for name in weights.keys()],
            "weight": list(weights.values()),
        }
    )
    _write_source(source_dir / "nature_figure_4_source_data.csv", [grid_source, bore_source, line_source, weight_source])
    return _save(fig, figure_dir, "nature_figure_4_site_investigation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    root = Path(".")
    table_dir = Path(config["paths"]["tables_dir"])
    figure_dir = Path(config["paths"]["figures_dir"])
    source_dir = Path(config["paths"].get("source_data_dir", "outputs/source_data"))
    _apply_style()
    generated = []
    generated.extend(make_figure_1(table_dir, figure_dir, source_dir))
    generated.extend(make_figure_2(table_dir, figure_dir, source_dir))
    generated.extend(make_figure_3(root, table_dir, figure_dir, source_dir))
    generated.extend(make_figure_4(root, table_dir, figure_dir, source_dir))
    for path in generated:
        print(f"figure={path}")
    for path in sorted(source_dir.glob("nature_figure_*_source_data.csv")):
        print(f"source_data={path}")


if __name__ == "__main__":
    main()
