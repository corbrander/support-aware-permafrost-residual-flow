from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np

from cold_recon.data.data_schema import load_sample_npz
from scripts.build_tree_prior_residual_posterior import tree_prior_fields


def _build_one(task: tuple[str, dict, int, int, int]) -> dict[str, str | float]:
    root_string, record, n_facies, rf_trees, rf_jobs = task
    root = Path(root_string)
    cache_dir = root / "prior_cache" / record["split"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{record['scene_id']}_rf{int(rf_trees)}.npz"
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            required = {"facies", "eic", "temperature", "unfrozen_water", "log_resistivity"}
            if required.issubset(data.files):
                return {"scene_id": record["scene_id"], "status": "reused", "seconds": 0.0}
        except Exception:
            pass
    start = time.time()
    sample = load_sample_npz(root / record["relative_path"])
    prior = tree_prior_fields(
        sample,
        n_facies=int(n_facies),
        seed=int(record["seed"]) + 91,
        rf_trees=int(rf_trees),
        rf_n_jobs=int(rf_jobs),
    )
    np.savez_compressed(cache_path, **{name: np.asarray(value) for name, value in prior.items()})
    return {
        "scene_id": record["scene_id"],
        "status": "generated",
        "seconds": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/m1_support_guided_benchmark/m1_scene_manifest.json")
    parser.add_argument("--splits", nargs="*", default=["train", "validation", "test_id"])
    parser.add_argument("--include-ood", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rf-jobs", type=int, default=4)
    parser.add_argument("--rf-trees", type=int, default=24)
    parser.add_argument("--n-facies", type=int, default=7)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_splits = set(args.splits)
    records = [
        record
        for record in manifest["records"]
        if record["split"] in selected_splits
        or (bool(args.include_ood) and record["split"].startswith("ood_"))
    ]
    missing = [record["relative_path"] for record in records if not (manifest_path.parent / record["relative_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"benchmark not materialized; first missing scene: {missing[0]}")
    tasks = [
        (
            str(manifest_path.parent),
            record,
            int(args.n_facies),
            int(args.rf_trees),
            int(args.rf_jobs),
        )
        for record in records
    ]
    generated = reused = failures = 0
    details: list[dict] = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(_build_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                details.append(result)
                if result["status"] == "generated":
                    generated += 1
                else:
                    reused += 1
            except Exception as exc:
                failures += 1
                details.append({"status": "failed", "error": repr(exc)})
            if index == 1 or index % 10 == 0:
                print(f"{index}/{len(tasks)} generated={generated} reused={reused} failures={failures}")
    summary = {
        "manifest_sha256": manifest["manifest_sha256"],
        "records": len(records),
        "generated": generated,
        "reused": reused,
        "failures": failures,
        "workers": int(args.workers),
        "rf_jobs_per_worker": int(args.rf_jobs),
        "rf_trees": int(args.rf_trees),
        "elapsed_seconds": time.time() - start,
        "details": details,
    }
    split_tag = "_".join(sorted(selected_splits))
    if bool(args.include_ood):
        split_tag = f"{split_tag}_with_ood"
    output = (
        manifest_path.parent
        / "prior_cache"
        / f"prior_cache_summary_{split_tag}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                **{key: value for key, value in summary.items() if key != "details"},
                "summary": str(output),
            },
            indent=2,
        )
    )
    if failures:
        raise RuntimeError(f"{failures} prior-cache tasks failed")


if __name__ == "__main__":
    main()
