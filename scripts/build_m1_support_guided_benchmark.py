from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from cold_recon.data.benchmark_manifest import build_m1_manifest, write_m1_manifest
from cold_recon.data.data_schema import load_sample_npz, save_sample_npz
from cold_recon.synthetic.multifamily_generator import generate_multifamily_sample
from cold_recon.utils.config import load_config


def materialize_manifest(
    manifest: dict,
    config: dict,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict:
    start = time.time()
    generated = 0
    reused = 0
    failures: list[dict[str, str]] = []
    for index, record in enumerate(manifest["records"], start=1):
        path = output_dir / record["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            try:
                existing = load_sample_npz(path)
                metadata = existing.get("metadata", {})
                if (
                    metadata.get("scene_id") == record["scene_id"]
                    and metadata.get("generator_family") == record["generator_family"]
                ):
                    reused += 1
                    continue
            except Exception:
                pass
        try:
            sample = generate_multifamily_sample(
                config,
                seed=int(record["seed"]),
                family=str(record["generator_family"]),
                scene_id=str(record["scene_id"]),
                site_id=f"synthetic_{record['split']}",
            )
            sample["metadata"]["split"] = record["split"]
            sample["metadata"]["manifest_sha256"] = manifest["manifest_sha256"]
            save_sample_npz(path, sample)
            generated += 1
        except Exception as exc:
            failures.append({"scene_id": str(record["scene_id"]), "error": repr(exc)})
        if index == 1 or index % 25 == 0:
            print(f"{index}/{len(manifest['records'])}: generated={generated} reused={reused} failures={len(failures)}")
    summary = {
        "manifest_sha256": manifest["manifest_sha256"],
        "total": len(manifest["records"]),
        "generated": generated,
        "reused": reused,
        "failures": failures,
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "m1_materialization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--output-dir", default="data/m1_support_guided_benchmark")
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test-id", type=int, default=100)
    parser.add_argument("--ood-per-family", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20260714)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = build_m1_manifest(
        train_count=int(args.train),
        validation_count=int(args.validation),
        id_test_count=int(args.test_id),
        ood_count_per_family=int(args.ood_per_family),
        base_seed=int(args.base_seed),
    )
    json_path, csv_path = write_m1_manifest(manifest, output_dir)
    print(f"manifest: {json_path}")
    print(f"manifest table: {csv_path}")
    if not args.manifest_only:
        summary = materialize_manifest(
            manifest,
            load_config(args.config),
            output_dir,
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
