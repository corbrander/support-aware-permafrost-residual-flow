from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.evaluation.architecture_summary import plot_algorithm_schematic, summarize_architecture
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    table_path = Path(config["paths"]["tables_dir"]) / "model_architecture_summary.csv"
    figure_path = Path(config["paths"]["figures_dir"]) / "cold_recon_algorithm_schematic.png"
    table = summarize_architecture(".")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    plot_algorithm_schematic(figure_path)
    print(f"architecture_summary={table_path}")
    print(f"algorithm_schematic={figure_path}")


if __name__ == "__main__":
    main()
