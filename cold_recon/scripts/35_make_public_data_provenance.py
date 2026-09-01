from __future__ import annotations

import argparse

from cold_recon.evaluation.public_data_provenance import write_public_data_tables
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    outputs = write_public_data_tables(config, root=".")
    for key, path in outputs.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
