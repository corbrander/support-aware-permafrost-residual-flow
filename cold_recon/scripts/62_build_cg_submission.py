from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.evaluation.cg_article_builder import build_cg_submission
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--output-dir", default="paper/cg_algorithm_submission")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    result = build_cg_submission(Path("."), Path(args.output_dir))
    print(f"package_dir={result.package_dir}")
    print(f"article_md={result.article_md}")
    print(f"article_docx={result.article_docx}")
    print(f"figure_manifest={result.figure_manifest}")
    print(f"n_figures={result.n_figures}")
    print(f"n_scripts={result.n_scripts}")
    print(f"zip={result.package_zip}")


if __name__ == "__main__":
    main()
