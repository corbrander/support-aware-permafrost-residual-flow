from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.evaluation.figure_atlas import build_figure_atlas
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = build_figure_atlas(
        Path("."),
        figure_dir=Path(config["paths"]["figures_dir"]),
        table_dir=Path(config["paths"]["tables_dir"]),
        paper_dir=Path("paper"),
    )
    print(f"figure_atlas_csv={result.table_csv}")
    print(f"figure_atlas_markdown={result.markdown}")
    print(f"figure_stems={result.n_stems}")
    print(f"figure_files={result.n_files}")
    print(f"submission_figures={result.n_submission_figures}")
    print(f"scope_boundary_excluded={result.n_excluded}")
    print(f"qa_previews={result.n_previews}")


if __name__ == "__main__":
    main()
