from __future__ import annotations

import argparse

from cold_recon.training.cross_validate_multisample_diffusion import cross_validate_multisample_diffusion
from cold_recon.utils.config import ensure_dirs, load_config


def _folds(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--folds", default=None, help="Comma-separated holdout fold indices. Default: all samples.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-condition-tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = cross_validate_multisample_diffusion(
        config,
        n_samples=args.n_samples,
        folds=_folds(args.folds),
        epochs=args.epochs,
        samples=args.samples,
        max_condition_tokens=args.max_condition_tokens,
        device=args.device,
    )
    print(f"detail={result['detail_path']}")
    print(f"summary={result['summary_path']}")
    print(f"improvement={result['improvement_path']}")


if __name__ == "__main__":
    main()
