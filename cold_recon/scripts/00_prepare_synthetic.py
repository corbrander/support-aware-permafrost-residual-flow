from __future__ import annotations

import argparse
from pathlib import Path

from cold_recon.synthetic.cryo_synth_generator import generate_synthetic_sample, save_synthetic_sample
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    base_seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    out_dir = Path(config["paths"]["synthetic_dir"])
    for i in range(args.n_samples):
        sample = generate_synthetic_sample(config, seed=base_seed + i, site_id=f"synthetic_{i:04d}")
        out_path = out_dir / f"sample_{i:04d}.npz"
        save_synthetic_sample(out_path, sample)
        print(f"saved {out_path} n_obs={sample['observations'].n_obs}")


if __name__ == "__main__":
    main()

