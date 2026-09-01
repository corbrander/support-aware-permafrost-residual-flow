from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.evaluation.reproducibility import write_audit_outputs
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--no-hash", action="store_true", help="Skip SHA-256 hashing for faster existence-only checks.")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = write_audit_outputs(
        Path("."),
        Path(config["paths"]["tables_dir"]),
        Path("paper"),
        hash_files=not args.no_hash,
    )
    summary = result["summary_data"]
    print(f"manifest={result['manifest']}")
    print(f"summary={result['summary']}")
    print(f"report={result['report']}")
    print(f"passed={summary['passed']}")
    print(f"missing_required={summary['n_missing_required']}")


if __name__ == "__main__":
    main()
