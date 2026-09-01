from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.evaluation.submission_package import make_submission_package
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = make_submission_package(Path("."))
    print(f"article_docx={result.article_docx}")
    print(f"package_dir={result.package_dir}")
    print(f"package_readme={result.package_readme}")
    print(f"package_zip={result.package_zip}")


if __name__ == "__main__":
    main()
