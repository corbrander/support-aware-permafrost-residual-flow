from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from cold_recon.data.data_schema import OBS_TYPE_NAMES
from cold_recon.data.jago_ground_ice_loader import (
    jago_ground_ice_to_observations,
    write_jago_ground_ice_inventory,
)
from cold_recon.models.observation_tokenizer import ObservationTokenizer
from cold_recon.utils.config import ensure_dirs, load_config


def _write_observation_summary(path: Path, observations, token_index: pd.DataFrame, inventory: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    type_counts = {
        OBS_TYPE_NAMES.get(int(type_id), f"type_{int(type_id)}"): int(count)
        for type_id, count in zip(*np.unique(observations.type_ids, return_counts=True))
    }
    summary = {
        "n_inventory_rows": int(len(inventory)),
        "n_observation_tokens": int(observations.n_obs),
        "n_eic_tokens": int(type_counts.get("borehole_eic", 0)),
        "n_boreholes": int(token_index["BOREHOLE_ID"].nunique()) if not token_index.empty else 0,
        "n_high_eic_tokens": int((token_index["value"] >= 0.30).sum()) if not token_index.empty else 0,
        "mean_eic_fraction": float(token_index["value"].mean()) if not token_index.empty else float("nan"),
        "max_eic_fraction": float(token_index["value"].max()) if not token_index.empty else float("nan"),
        "max_depth_m": float(token_index["z"].max()) if not token_index.empty else float("nan"),
        "coordinate_source": "ordered_borehole_index",
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--borehole-spacing-m", type=float, default=20.0)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    paths = config["paths"]
    outputs = write_jago_ground_ice_inventory(config)
    inventory = pd.read_csv(outputs["inventory_csv"])
    observations, token_index = jago_ground_ice_to_observations(inventory, borehole_spacing_m=float(args.borehole_spacing_m))
    tokenizer = ObservationTokenizer(n_types=9)
    tokens = tokenizer.encode_numpy(observations)

    processed = Path(paths["processed_dir"])
    table_dir = Path(paths["tables_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    obs_path = processed / "arcticdata_jago_ground_ice_observations.npz"
    token_path = processed / "arcticdata_jago_ground_ice_tokens.npz"
    token_index_path = table_dir / "arcticdata_jago_ground_ice_token_index.csv"
    summary_path = table_dir / "arcticdata_jago_ground_ice_observation_summary.csv"

    np.savez_compressed(
        obs_path,
        **observations.as_npz_dict(),
        site_ids=token_index["site"].to_numpy(dtype=str),
        borehole_ids=token_index["BOREHOLE_ID"].to_numpy(dtype=str),
        group_keys=(
            token_index["site"].astype(str)
            + "::"
            + token_index["BOREHOLE_ID"].astype(str)
        ).to_numpy(dtype=str),
        source_files=token_index["source_file"].to_numpy(dtype=str),
        layers=token_index["layer"].to_numpy(dtype=str),
        layer_groups=token_index["layer_group"].to_numpy(dtype=str),
        coordinate_sources=token_index["coordinate_source"].to_numpy(dtype=str),
        depth_top_m=token_index["DEPTH_TOP"].to_numpy(dtype=np.float32),
        depth_bottom_m=token_index["DEPTH_BOTTOM"].to_numpy(dtype=np.float32),
        source_doi=token_index["source_doi"].to_numpy(dtype=str),
    )
    np.savez_compressed(token_path, tokens=tokens)
    token_index.to_csv(token_index_path, index=False)
    _write_observation_summary(summary_path, observations, token_index, inventory)

    print(f"inventory={outputs['inventory_csv']}")
    print(f"summary={outputs['summary_csv']}")
    print(f"observations={obs_path}")
    print(f"tokens={token_path}")
    print(f"token_index={token_index_path}")
    print(f"observation_summary={summary_path}")
    print(f"n_observation_tokens={observations.n_obs}")


if __name__ == "__main__":
    main()
