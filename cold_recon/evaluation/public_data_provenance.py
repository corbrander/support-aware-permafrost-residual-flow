from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cold_recon.data.data_schema import OBS_TYPE_NAMES
from cold_recon.data.public_sources import PUBLIC_SOURCES
from cold_recon.evaluation.reproducibility import sha256_file


USED_PUBLIC_DATASETS: dict[str, dict[str, str]] = {
    "usgs_ert_nmr": {
        "raw_subdir": "usgs_ert_nmr",
        "observations_npz": "usgs_geophysics_observations.npz",
        "tokens_npz": "usgs_geophysics_tokens.npz",
        "summary_csv": "usgs_geophysics_summary.csv",
        "notes": "Processed into thaw-depth, borehole NMR, and ERT observation tokens for field validation.",
    },
    "usgs_eic_cores": {
        "raw_subdir": "usgs_eic_cores",
        "observations_npz": "usgs_eic_observations.npz",
        "tokens_npz": "usgs_eic_tokens.npz",
        "summary_csv": "usgs_eic_summary.csv",
        "notes": "Processed into borehole EIC observation tokens for independent core validation.",
    },
    "arcticdata_upper_permafrost_cryostratigraphy": {
        "raw_subdir": "arcticdata_upper_permafrost_cryostratigraphy",
        "observations_npz": "arcticdata_cryostratigraphy_observations.npz",
        "tokens_npz": "arcticdata_cryostratigraphy_tokens.npz",
        "inventory_csv": "arcticdata_cryostratigraphy_inventory.csv",
        "summary_csv": "arcticdata_cryostratigraphy_observation_summary.csv",
        "notes": "Downloaded from Arctic Data Center / DataONE and processed into external cryostratigraphy facies and ground-ice observation tokens.",
    },
    "arcticdata_jago_ground_ice_2018": {
        "raw_subdir": "arcticdata_jago_ground_ice_2018",
        "observations_npz": "arcticdata_jago_ground_ice_observations.npz",
        "tokens_npz": "arcticdata_jago_ground_ice_tokens.npz",
        "inventory_csv": "arcticdata_jago_ground_ice_inventory.csv",
        "summary_csv": "arcticdata_jago_ground_ice_observation_summary.csv",
        "notes": "Downloaded from Arctic Data Center / DataONE and processed into an independent Jago River ground-ice/EIC validation source.",
    },
}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _path_from_config(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _file_inventory(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, int(path.stat().st_size)
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), int(sum(p.stat().st_size for p in files))


def _npz_count(path: Path, preferred_keys: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    data = np.load(path, allow_pickle=True)
    for key in preferred_keys:
        if key in data:
            return int(data[key].shape[0])
    return 0


def _npz_type_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    if "obs_type_ids" not in data:
        return {}
    type_ids = data["obs_type_ids"].astype(int)
    unique, counts = np.unique(type_ids, return_counts=True)
    return {OBS_TYPE_NAMES.get(int(t), f"type_{int(t)}"): int(c) for t, c in zip(unique, counts)}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _download_summary(report: dict[str, Any], source_key: str) -> dict[str, Any]:
    source_report = report.get(source_key, {})
    files_report = source_report.get("files", {})
    downloaded_count = 0
    downloaded_bytes = 0
    skipped_count = 0
    skipped_large_count = 0
    skipped_large_bytes = 0

    def consume_file_block(block: dict[str, Any]) -> None:
        nonlocal downloaded_count, downloaded_bytes, skipped_count, skipped_large_count, skipped_large_bytes
        for item in block.get("downloaded", []) or []:
            downloaded_count += 1
            downloaded_bytes += int(item.get("size") or 0)
        for item in block.get("skipped", []) or []:
            skipped_count += 1
            size = int(item.get("size") or 0)
            if item.get("reason") == "larger_than_limit":
                skipped_large_count += 1
                skipped_large_bytes += size

    root_block = files_report.get("root")
    if isinstance(files_report.get("downloaded"), list) or isinstance(files_report.get("skipped"), list):
        consume_file_block(files_report)
    if isinstance(root_block, dict):
        consume_file_block(root_block)
    for child in files_report.get("children", []) or []:
        child_files = child.get("files", {})
        if isinstance(child_files, dict):
            consume_file_block(child_files)

    return {
        "downloaded_file_count": downloaded_count,
        "downloaded_total_mb": downloaded_bytes / (1024 * 1024),
        "skipped_file_count": skipped_count,
        "skipped_large_file_count": skipped_large_count,
        "skipped_large_total_mb": skipped_large_bytes / (1024 * 1024),
    }


def build_public_data_tables(config: dict[str, Any], root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    paths = config.get("paths", {})
    raw_dir = _path_from_config(root, paths.get("raw_dir", "data/raw"))
    processed_dir = _path_from_config(root, paths.get("processed_dir", "data/processed"))
    external_dir = _path_from_config(root, paths.get("external_dir", "data/external"))
    table_dir = _path_from_config(root, paths.get("tables_dir", "outputs/tables"))

    download_report = _load_json(external_dir / "download_report.json")
    availability = _load_json(table_dir / "real_data_availability.json")

    provenance_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for source_key, source in PUBLIC_SOURCES.items():
        used = USED_PUBLIC_DATASETS.get(source_key)
        raw_subdir = used["raw_subdir"] if used else source_key
        raw_path = raw_dir / raw_subdir
        raw_count, raw_bytes = _file_inventory(raw_path)
        download_stats = _download_summary(download_report, source_key)

        obs_path = processed_dir / used["observations_npz"] if used and "observations_npz" in used else None
        token_path = processed_dir / used["tokens_npz"] if used and "tokens_npz" in used else None
        inventory_path = processed_dir / used["inventory_csv"] if used and "inventory_csv" in used else None
        summary_path = table_dir / used["summary_csv"] if used else None
        has_obs = obs_path is not None and obs_path.exists()
        has_tokens = token_path is not None and token_path.exists()
        has_inventory = inventory_path is not None and inventory_path.is_file()
        n_observations = _npz_count(obs_path, ("obs_coords", "observations", "tokens")) if has_obs else 0
        n_tokens = _npz_count(token_path, ("tokens", "obs_coords")) if has_tokens else 0
        n_inventory_rows = int(len(pd.read_csv(inventory_path))) if has_inventory else 0
        type_counts = _npz_type_counts(obs_path) if has_obs else {}
        processed_sha = sha256_file(obs_path)[:16] if has_obs else (sha256_file(inventory_path)[:16] if has_inventory else "")
        token_sha = sha256_file(token_path)[:16] if has_tokens else ""
        if n_observations > 0 and n_tokens > 0:
            status = "downloaded_processed"
        elif n_inventory_rows > 0:
            status = "downloaded_inventory"
        else:
            status = "raw_downloaded" if raw_count else "interface_documented"

        notes = used["notes"] if used else "Loader/download interface documented; not part of the current processed validation token set."
        if source_key == "usgs_ert_nmr" and download_stats["skipped_large_file_count"]:
            notes += f" {download_stats['skipped_large_file_count']} oversized raw file(s) documented but skipped by download size limit."
        if source_key in availability and isinstance(availability[source_key], dict):
            notes += f" availability={availability[source_key].get('available', False)}."

        provenance_rows.append(
            {
                "source_key": source_key,
                "dataset": source["name"],
                "source_url": source["url"],
                "role": source["role"],
                "status": status,
                "raw_dir": _rel(raw_path, root),
                "raw_file_count": raw_count,
                "raw_total_mb": raw_bytes / (1024 * 1024),
                **download_stats,
                "observations_npz": _rel(obs_path, root) if obs_path is not None else "",
                "tokens_npz": _rel(token_path, root) if token_path is not None else "",
                "inventory_csv": _rel(inventory_path, root) if has_inventory else "",
                "summary_csv": _rel(summary_path, root) if summary_path is not None else "",
                "n_observations": n_observations,
                "n_tokens": n_tokens,
                "n_inventory_rows": n_inventory_rows,
                "processed_sha256_16": processed_sha,
                "token_sha256_16": token_sha,
                "observation_type_counts": json.dumps(type_counts, ensure_ascii=False, sort_keys=True),
                "notes": notes,
            }
        )

        for obs_type, count in type_counts.items():
            token_rows.append(
                {
                    "source_key": source_key,
                    "dataset": source["name"],
                    "observation_type": obs_type,
                    "n_tokens": count,
                    "tokens_npz": _rel(token_path, root) if token_path is not None else "",
                    "observations_npz": _rel(obs_path, root) if obs_path is not None else "",
                }
            )

    provenance = pd.DataFrame(provenance_rows)
    token_inventory = pd.DataFrame(token_rows).sort_values(["source_key", "observation_type"]).reset_index(drop=True)
    return provenance, token_inventory


def write_public_data_tables(config: dict[str, Any], root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    table_dir = _path_from_config(root, config.get("paths", {}).get("tables_dir", "outputs/tables"))
    figure_dir = _path_from_config(root, config.get("paths", {}).get("figures_dir", "outputs/figures"))
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    provenance, token_inventory = build_public_data_tables(config, root=root)
    provenance_csv = table_dir / "public_data_provenance.csv"
    provenance_json = table_dir / "public_data_provenance.json"
    token_csv = table_dir / "public_data_token_inventory.csv"
    figure_path = figure_dir / "public_data_token_inventory.png"

    provenance.to_csv(provenance_csv, index=False)
    token_inventory.to_csv(token_csv, index=False)
    provenance_json.write_text(json.dumps(provenance.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    plot_public_data_token_inventory(token_inventory, figure_path)
    return {
        "provenance_csv": provenance_csv,
        "provenance_json": provenance_json,
        "token_inventory_csv": token_csv,
        "token_inventory_figure": figure_path,
    }


def plot_public_data_token_inventory(token_inventory: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if token_inventory.empty:
        ax.text(0.5, 0.5, "No processed public-data tokens", ha="center", va="center")
        ax.set_axis_off()
    else:
        view = token_inventory.copy()
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
        view["label"] = [
            f"{source_labels.get(str(src), str(src))} - {obs_labels.get(str(obs), str(obs))}"
            for src, obs in zip(view["source_key"], view["observation_type"])
        ]
        view = view.sort_values("n_tokens", ascending=True).reset_index(drop=True)
        colors = {
            "usgs_ert_nmr": "#4c78a8",
            "usgs_eic_cores": "#f58518",
            "arcticdata_upper_permafrost_cryostratigraphy": "#54a24b",
            "arcticdata_jago_ground_ice_2018": "#b279a2",
        }
        bar_colors = [colors.get(str(src), "#777777") for src in view["source_key"]]
        y = np.arange(len(view))
        counts = view["n_tokens"].astype(int).to_numpy()
        ax.barh(y, counts, color=bar_colors, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels(view["label"])
        ax.set_xscale("log")
        ax.set_xlabel("processed observation tokens (log scale)")
        ax.set_title("Public-data token inventory for COLD-Recon validation")
        ax.grid(axis="x", alpha=0.25, which="both")
        for yi, count in zip(y, counts):
            ax.text(max(count * 1.08, 1.1), yi, f"{count:,}", va="center", fontsize=8)
        ax.set_xlim(1, max(counts) * 3.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path
