from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
import pandas as pd

from cold_recon.data.arcticdata_cryostratigraphy_loader import (
    arcticdata_cryostratigraphy_to_observations,
    write_arcticdata_cryostratigraphy_inventory,
)
from cold_recon.data.data_schema import OBS_TYPE_NAMES
from cold_recon.models.observation_tokenizer import ObservationTokenizer
from cold_recon.utils.config import ensure_dirs, load_config


def _write_summary(path: Path, observations, token_index: pd.DataFrame, inventory: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    type_counts = {
        OBS_TYPE_NAMES.get(int(type_id), f"type_{int(type_id)}"): int(count)
        for type_id, count in zip(*np.unique(observations.type_ids, return_counts=True))
    }
    summary = {
        "n_inventory_rows": int(len(inventory)),
        "n_observation_tokens": int(observations.n_obs),
        "n_facies_tokens": int(type_counts.get("borehole_facies", 0)),
        "n_eic_tokens": int(type_counts.get("borehole_eic", 0)),
        "n_sites": int(token_index["site"].nunique()) if not token_index.empty else 0,
        "n_boreholes": int(token_index[["site", "borehole"]].drop_duplicates().shape[0]) if not token_index.empty else 0,
        "n_public_lat_lon_tokens": int((token_index["coordinate_source"] == "public_lat_lon_site_local").sum()) if not token_index.empty else 0,
        "n_ordered_index_tokens": int((token_index["coordinate_source"] == "ordered_borehole_index_site_local").sum()) if not token_index.empty else 0,
        "max_depth_m": float(token_index["z"].max()) if not token_index.empty else float("nan"),
        "mean_eic_fraction": float(token_index.loc[token_index["type_id"] == 1, "value"].mean()) if not token_index.empty and np.any(token_index["type_id"] == 1) else float("nan"),
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _site_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unnamed_site"


def _archive_payload(observations, token_index: pd.DataFrame) -> dict:
    return {
        **observations.as_npz_dict(),
        "site_ids": token_index["site"].to_numpy(dtype=str),
        "borehole_ids": token_index["borehole"].to_numpy(dtype=str),
        "group_keys": (
            token_index["site"].astype(str)
            + "::"
            + token_index["borehole"].astype(str)
        ).to_numpy(dtype=str),
        "source_files": token_index["source_file"].to_numpy(dtype=str),
        "cryofacies_classes": token_index["cryofacies_class"].to_numpy(dtype=str),
        "coordinate_sources": token_index["coordinate_source"].to_numpy(dtype=str),
        "depth_top_m": token_index["depth_top_m"].to_numpy(dtype=np.float32),
        "depth_bottom_m": token_index["depth_bottom_m"].to_numpy(dtype=np.float32),
        "source_doi": np.full(
            len(token_index), "10.18739/A2QR4NS3D", dtype=str
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    paths = config["paths"]
    inventory_path = Path(paths["processed_dir"]) / "arcticdata_cryostratigraphy_inventory.csv"
    if not inventory_path.exists():
        write_arcticdata_cryostratigraphy_inventory(config)
    inventory = pd.read_csv(inventory_path)
    observations, token_index = arcticdata_cryostratigraphy_to_observations(inventory)
    tokenizer = ObservationTokenizer(n_types=9)
    tokens = tokenizer.encode_numpy(observations)

    processed = Path(paths["processed_dir"])
    table_dir = Path(paths["tables_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    obs_path = processed / "arcticdata_cryostratigraphy_observations.npz"
    token_path = processed / "arcticdata_cryostratigraphy_tokens.npz"
    token_index_path = table_dir / "arcticdata_cryostratigraphy_token_index.csv"
    summary_path = table_dir / "arcticdata_cryostratigraphy_observation_summary.csv"

    np.savez_compressed(obs_path, **_archive_payload(observations, token_index))
    np.savez_compressed(token_path, tokens=tokens)
    token_index.to_csv(token_index_path, index=False)
    _write_summary(summary_path, observations, token_index, inventory)

    site_dir = processed / "arcticdata_cryostratigraphy_sites"
    site_dir.mkdir(parents=True, exist_ok=True)
    registry_rows: list[dict] = []
    used_slugs: set[str] = set()
    for site, site_index in token_index.groupby("site", sort=True):
        slug = _site_slug(str(site))
        if slug in used_slugs:
            raise ValueError(f"Duplicate Arctic site slug: {slug}")
        used_slugs.add(slug)
        selected = site_index.index.to_numpy(dtype=np.int64)
        site_observations = observations.subset(selected)
        site_index = site_index.reset_index(drop=True)
        site_path = site_dir / f"{slug}_observations.npz"
        np.savez_compressed(
            site_path, **_archive_payload(site_observations, site_index)
        )
        eic = site_index[site_index["type_id"] == 1]
        registry_rows.append(
            {
                "site_id": str(site),
                "site_slug": slug,
                "observations_npz": str(site_path),
                "n_tokens": int(len(site_index)),
                "n_eic_intervals": int(len(eic)),
                "n_eic_boreholes": int(eic["borehole"].nunique()),
                "n_source_files": int(site_index["source_file"].nunique()),
                "coordinate_sources": "|".join(
                    sorted(site_index["coordinate_source"].astype(str).unique())
                ),
                "eligible_within_site_loo": bool(eic["borehole"].nunique() >= 8),
            }
        )
    site_registry_path = table_dir / "arcticdata_cryostratigraphy_site_registry.csv"
    pd.DataFrame(registry_rows).to_csv(site_registry_path, index=False)

    print(f"observations={obs_path}")
    print(f"tokens={token_path}")
    print(f"token_index={token_index_path}")
    print(f"summary={summary_path}")
    print(f"site_registry={site_registry_path}")
    print(f"n_observation_tokens={observations.n_obs}")


if __name__ == "__main__":
    main()
