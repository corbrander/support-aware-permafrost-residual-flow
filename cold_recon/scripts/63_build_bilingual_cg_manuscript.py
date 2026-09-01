from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.evaluation.bilingual_cg_manuscript import build_bilingual_cg_manuscript
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--output-dir", default="paper/cg_bilingual_manuscript")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = build_bilingual_cg_manuscript(Path("."), Path(args.output_dir))
    print(f"package_dir={result.package_dir}")
    print(f"english_md={result.english_md}")
    print(f"chinese_md={result.chinese_md}")
    print(f"english_docx={result.english_docx}")
    print(f"chinese_docx={result.chinese_docx}")
    print(f"figure_manifest={result.figure_manifest}")
    print(f"alignment_table={result.alignment_table}")
    print(f"n_figures={result.n_figures}")
    print(f"zip={result.package_zip}")


if __name__ == "__main__":
    main()
