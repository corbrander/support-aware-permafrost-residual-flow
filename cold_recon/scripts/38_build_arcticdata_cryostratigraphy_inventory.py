from __future__ import annotations

import argparse

from cold_recon.data.arcticdata_cryostratigraphy_loader import write_arcticdata_cryostratigraphy_inventory
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = write_arcticdata_cryostratigraphy_inventory(config)
    print(f"inventory={result['inventory_csv']}")
    print(f"summary={result['summary_csv']}")


if __name__ == "__main__":
    main()
