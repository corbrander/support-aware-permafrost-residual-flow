from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cold_recon.data.usgs_eic_loader import load_usgs_eic_cores
from cold_recon.data.usgs_eic_loader import (
    prepare_usgs_eic_observation_table,
    read_usgs_eic_tables,
    usgs_eic_to_observations,
)
from cold_recon.data.usgs_ert_nmr_loader import load_usgs_ert_nmr
from cold_recon.data.usgs_ert_nmr_loader import (
    combine_observation_tables,
    read_usgs_ert_inverted_models,
    read_usgs_nmr_inverted,
    read_usgs_thaw_depths,
    usgs_ert_to_observations,
    usgs_nmr_to_observations,
    usgs_thaw_depth_to_observations,
)
from cold_recon.evaluation.field_validation import describe_real_data_validation
from cold_recon.models.observation_tokenizer import ObservationTokenizer
from cold_recon.utils.config import ensure_dirs, load_config


def _write_usgs_eic_products(config: dict) -> dict:
    raw = Path(config["paths"]["raw_dir"])
    processed = Path(config["paths"]["processed_dir"])
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    eic, _ = read_usgs_eic_tables(raw)
    observation_index = prepare_usgs_eic_observation_table(raw)
    observations = usgs_eic_to_observations(raw)
    tokenizer = ObservationTokenizer(n_types=9)
    tokens = tokenizer.encode_numpy(observations)
    obs_path = processed / "usgs_eic_observations.npz"
    token_path = processed / "usgs_eic_tokens.npz"
    np.savez_compressed(
        obs_path,
        **observations.as_npz_dict(),
        site_ids=np.full(len(observation_index), "USGS Arctic Coastal Plain", dtype=str),
        borehole_ids=observation_index["BOREHOLE_ID"].to_numpy(dtype=str),
        source_files=np.full(
            len(observation_index), "permafrostCores_borehole_EIC.csv", dtype=str
        ),
        coordinate_sources=observation_index["coordinate_source"].to_numpy(dtype=str),
        depth_top_m=observation_index["DEPTH_TOP"].to_numpy(dtype=np.float32),
        depth_bottom_m=observation_index["DEPTH_BOTTOM"].to_numpy(dtype=np.float32),
        source_doi=np.full(len(observation_index), "10.5066/P13AEEH7", dtype=str),
    )
    public_coordinate_index = prepare_usgs_eic_observation_table(
        raw, coordinate_mode="public_lat_lon"
    )
    public_coordinate_observations = usgs_eic_to_observations(
        raw, coordinate_mode="public_lat_lon"
    )
    public_coordinate_obs_path = (
        processed / "usgs_eic_public_coordinates_observations.npz"
    )
    np.savez_compressed(
        public_coordinate_obs_path,
        **public_coordinate_observations.as_npz_dict(),
        site_ids=np.full(
            len(public_coordinate_index), "USGS Utqiagvik public-coordinate subset", dtype=str
        ),
        borehole_ids=public_coordinate_index["BOREHOLE_ID"].to_numpy(dtype=str),
        source_files=np.full(
            len(public_coordinate_index),
            "permafrostCores_borehole_EIC.csv",
            dtype=str,
        ),
        coordinate_sources=public_coordinate_index["coordinate_source"].to_numpy(
            dtype=str
        ),
        depth_top_m=public_coordinate_index["DEPTH_TOP"].to_numpy(dtype=np.float32),
        depth_bottom_m=public_coordinate_index["DEPTH_BOTTOM"].to_numpy(dtype=np.float32),
        source_doi=np.full(
            len(public_coordinate_index), "10.5066/P13AEEH7", dtype=str
        ),
    )
    np.savez_compressed(token_path, tokens=tokens)
    summary = {
        "n_boreholes": int(eic["BOREHOLE_ID"].nunique()),
        "n_intervals": int(eic.shape[0]),
        "depth_min_m": float(eic["DEPTH_TOP"].min()),
        "depth_max_m": float(eic["DEPTH_BOTTOM"].max()),
        "eic_mean_fraction": float((eic["EXCESS_ICE_CONTENT"] / 100.0).mean()),
        "eic_max_fraction": float((eic["EXCESS_ICE_CONTENT"] / 100.0).max()),
        "high_eic_fraction_intervals": float((eic["EXCESS_ICE_CONTENT"] >= 30.0).mean()),
        "n_intervals_above_model_eic_ceiling_0_90": int(
            (eic["EXCESS_ICE_CONTENT"] > 90.0).sum()
        ),
        "n_public_coordinate_boreholes": int(
            public_coordinate_index["BOREHOLE_ID"].nunique()
        ),
        "n_public_coordinate_intervals": int(len(public_coordinate_index)),
    }
    summary_path = table_dir / "usgs_eic_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    borehole_order = {name: i for i, name in enumerate(sorted(eic["BOREHOLE_ID"].unique()))}
    xs = eic["BOREHOLE_ID"].map(borehole_order).to_numpy(dtype=float)
    depths = 0.5 * (eic["DEPTH_TOP"].to_numpy(dtype=float) + eic["DEPTH_BOTTOM"].to_numpy(dtype=float))
    values = eic["EXCESS_ICE_CONTENT"].to_numpy(dtype=float) / 100.0
    sc = ax.scatter(xs, depths, c=values, s=24, cmap="viridis", vmin=0.0, vmax=max(0.5, values.max()))
    ax.invert_yaxis()
    ax.set_xlabel("Borehole index")
    ax.set_ylabel("Depth (m)")
    ax.set_title("USGS Arctic Coastal Plain permafrost core EIC")
    fig.colorbar(sc, ax=ax, label="EIC fraction")
    fig_path = fig_dir / "usgs_eic_borehole_profiles.png"
    fig.savefig(fig_path, dpi=180, facecolor="white")
    plt.close(fig)
    return {
        "observations_npz": str(obs_path),
        "public_coordinate_observations_npz": str(public_coordinate_obs_path),
        "tokens_npz": str(token_path),
        "summary_csv": str(summary_path),
        "figure": str(fig_path),
        **summary,
    }


def _write_usgs_geophysics_products(config: dict) -> dict:
    raw = Path(config["paths"]["raw_dir"])
    processed = Path(config["paths"]["processed_dir"])
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    thaw_df = read_usgs_thaw_depths(raw)
    nmr_df = read_usgs_nmr_inverted(raw)
    ert_df = read_usgs_ert_inverted_models(raw, max_points_per_model=1200)
    thaw_obs = usgs_thaw_depth_to_observations(raw)
    nmr_obs = usgs_nmr_to_observations(raw)
    ert_obs = usgs_ert_to_observations(raw, max_points_per_model=1200)
    observations = combine_observation_tables([thaw_obs, nmr_obs, ert_obs])
    tokenizer = ObservationTokenizer(n_types=9)
    tokens = tokenizer.encode_numpy(observations)

    obs_path = processed / "usgs_geophysics_observations.npz"
    token_path = processed / "usgs_geophysics_tokens.npz"
    np.savez_compressed(obs_path, **observations.as_npz_dict())
    np.savez_compressed(token_path, tokens=tokens)

    summary = {
        "n_thaw_depth": int(thaw_obs.n_obs),
        "n_nmr_inverted": int(nmr_obs.n_obs),
        "n_ert_model": int(ert_obs.n_obs),
        "n_total_tokens": int(observations.n_obs),
        "thaw_depth_mean_m": float(thaw_df["ThawDepth_m"].mean()),
        "nmr_total_water_mean_fraction": float(nmr_df["TotalWaterContent(fraction)"].mean()),
        "ert_log10_res_mean": float(ert_df["Log10Res"].mean()),
    }
    summary_path = table_dir / "usgs_geophysics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    sc0 = axes[0].scatter(thaw_df["local_x_m"], thaw_df["local_y_m"], c=thaw_df["ThawDepth_m"], s=16, cmap="viridis")
    axes[0].set_title("USGS thaw depth / ALT")
    axes[0].set_xlabel("local x (m)")
    axes[0].set_ylabel("local y (m)")
    fig.colorbar(sc0, ax=axes[0], label="thaw depth (m)")

    sc1 = axes[1].scatter(nmr_df["local_x_m"], nmr_df["Depth_m"], c=nmr_df["TotalWaterContent(fraction)"], s=18, cmap="Blues")
    axes[1].invert_yaxis()
    axes[1].set_title("USGS borehole NMR inverted water")
    axes[1].set_xlabel("local x (m)")
    axes[1].set_ylabel("depth (m)")
    fig.colorbar(sc1, ax=axes[1], label="water content")

    first_profile = str(ert_df["ProfileID"].iloc[0])
    ert_plot = ert_df[ert_df["ProfileID"] == first_profile]
    sc2 = axes[2].scatter(ert_plot["local_x_m"], ert_plot["Depth_m"], c=ert_plot["Log10Res"], s=4, cmap="magma")
    axes[2].invert_yaxis()
    axes[2].set_title(f"USGS ERT model {first_profile}")
    axes[2].set_xlabel("distance (m)")
    axes[2].set_ylabel("depth (m)")
    fig.colorbar(sc2, ax=axes[2], label="log10 resistivity")

    fig_path = fig_dir / "usgs_geophysics_observation_summary.png"
    fig.savefig(fig_path, dpi=180, facecolor="white")
    plt.close(fig)
    return {
        "observations_npz": str(obs_path),
        "tokens_npz": str(token_path),
        "summary_csv": str(summary_path),
        "figure": str(fig_path),
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    raw = Path(config["paths"]["raw_dir"])
    report = {
        "validation_note": describe_real_data_validation(),
        "usgs_ert_nmr": load_usgs_ert_nmr(raw),
        "usgs_eic_cores": load_usgs_eic_cores(raw),
    }
    if report["usgs_eic_cores"].get("eic_csv"):
        report["usgs_eic_products"] = _write_usgs_eic_products(config)
    if report["usgs_ert_nmr"].get("n_ert_model_zips", 0) > 0 and report["usgs_ert_nmr"].get("n_nmr_inverted_csv", 0) > 0:
        report["usgs_geophysics_products"] = _write_usgs_geophysics_products(config)
    out = Path(config["paths"]["tables_dir"]) / "real_data_availability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
