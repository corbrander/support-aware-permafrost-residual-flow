from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cold_recon.evaluation.real_data_cg_gate import build_real_data_cg_benchmark
from cold_recon.utils.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--arctic-summary", default=None)
    parser.add_argument("--usgs-comparison", default=None)
    parser.add_argument("--jago-comparison", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    table_dir = Path(config["paths"]["tables_dir"])
    arctic_path = Path(args.arctic_summary) if args.arctic_summary else table_dir / "arcticdata_conditioned_diffusion_multisite_summary.csv"
    usgs_path = Path(args.usgs_comparison) if args.usgs_comparison else table_dir / "usgs_eic_conditioned_diffusion_comparison.csv"
    jago_path = Path(args.jago_comparison) if args.jago_comparison else table_dir / "arcticdata_jago_ground_ice_conditioned_diffusion_comparison.csv"
    arctic_summary = pd.read_csv(arctic_path)
    usgs_comparison = pd.read_csv(usgs_path)
    jago_comparison = pd.read_csv(jago_path) if jago_path.exists() else None
    benchmark, gate = build_real_data_cg_benchmark(arctic_summary, usgs_comparison, jago_comparison)
    benchmark_path = table_dir / "real_data_cg_benchmark.csv"
    gate_path = table_dir / "real_data_cg_gate.json"
    benchmark.to_csv(benchmark_path, index=False)
    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"benchmark={benchmark_path}")
    print(f"gate={gate_path}")
    print(benchmark.to_string(index=False))
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
