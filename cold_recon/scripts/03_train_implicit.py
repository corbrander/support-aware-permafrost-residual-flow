from __future__ import annotations

import argparse

from cold_recon.training.train_implicit import train_implicit_model
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = train_implicit_model(
        config,
        sample_path=args.sample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"checkpoint={result['checkpoint']}")
    print(f"prediction={result['prediction_path']}")


if __name__ == "__main__":
    main()

