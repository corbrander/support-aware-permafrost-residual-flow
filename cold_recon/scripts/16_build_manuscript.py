from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.evaluation.paper_builder import build_manuscript
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--output", default="paper/cold_recon_manuscript_draft.md")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    output = build_manuscript(Path(config["paths"]["tables_dir"]), Path(args.output))
    print(f"manuscript={output}")


if __name__ == "__main__":
    main()
