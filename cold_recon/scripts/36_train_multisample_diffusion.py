from __future__ import annotations

import argparse

from cold_recon.training.train_multisample_diffusion import synthetic_sample_paths, train_multisample_diffusion
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--holdout-index", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-condition-tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    paths = synthetic_sample_paths(config, n_samples=args.n_samples)
    result = train_multisample_diffusion(
        config,
        sample_paths=paths,
        holdout_index=args.holdout_index,
        epochs=args.epochs,
        samples=args.samples,
        max_condition_tokens=args.max_condition_tokens,
        device=args.device,
    )
    print(f"checkpoint={result['checkpoint']}")
    print(f"posterior={result['posterior_path']}")
    print(f"metrics={result['metrics_path']}")
    print(f"history={result['history_path']}")


if __name__ == "__main__":
    main()
