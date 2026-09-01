from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from cold_recon.baselines.random_forest import (
    _grid_points_and_surface,
    _surface_at_obs,
)
from cold_recon.data.data_schema import OBS_TYPES, load_sample_npz
from cold_recon.evaluation.engineering_response import (
    DEFAULT_THAW_DEPTHS_M,
    engineering_response_metrics,
)
from scripts.evaluate_m1_controlled import _bootstrap_mean_ci
from scripts.train_m1_support_guided_flow import _manifest_records


def _fit_predict_rf_prefixes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    *,
    counts: tuple[int, ...],
    seed: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    model = RandomForestRegressor(
        n_estimators=int(counts[0]),
        min_samples_leaf=2,
        n_jobs=int(n_jobs),
        random_state=int(seed),
        warm_start=True,
    )
    predictions: dict[str, np.ndarray] = {}
    for count in counts:
        model.set_params(n_estimators=int(count))
        model.fit(x_train, y_train)
        predictions[f"RF-{int(count)}"] = model.predict(x_query).astype(np.float32)
    return predictions


def _fit_predict_alternatives(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    *,
    seed: int,
    extra_trees: int,
    n_jobs: int,
    include_extra: bool,
    include_gradient_boosting: bool,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    if include_extra:
        model = ExtraTreesRegressor(
            n_estimators=int(extra_trees),
            min_samples_leaf=2,
            n_jobs=int(n_jobs),
            random_state=int(seed) + 101,
        )
        model.fit(x_train, y_train)
        predictions[f"Extra Trees-{int(extra_trees)}"] = model.predict(x_query).astype(
            np.float32
        )
    if include_gradient_boosting:
        model = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0e-3,
            min_samples_leaf=8,
            random_state=int(seed) + 11,
        )
        model.fit(x_train, y_train)
        predictions["Gradient boosting"] = model.predict(x_query).astype(np.float32)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/m1_support_guided_benchmark/m1_scene_manifest.json",
    )
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--rf-tree-counts", type=int, nargs="+", default=(24, 100, 300, 500)
    )
    parser.add_argument("--extra-trees", type=int, default=300)
    parser.add_argument("--skip-extra", action="store_true")
    parser.add_argument("--skip-gradient-boosting", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=730)
    parser.add_argument(
        "--output-dir",
        default="outputs/m1_support_guided/formal_anchor_sensitivity",
    )
    parser.add_argument(
        "--engineering-response-depths",
        type=float,
        nargs="+",
        default=DEFAULT_THAW_DEPTHS_M,
    )
    parser.add_argument("--engineering-response-threshold", type=float, default=0.30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    counts = tuple(sorted(set(int(value) for value in args.rf_tree_counts)))
    if not counts or counts[0] <= 0:
        raise ValueError("RF tree counts must be positive")

    manifest_path = Path(args.manifest)
    root, records, manifest = _manifest_records(manifest_path, args.split)
    if int(args.max_scenes) > 0:
        records = records[: int(args.max_scenes)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"m1_anchor_sensitivity_{args.split}_detail.csv"
    response_path = output_dir / f"m1_anchor_sensitivity_{args.split}_response.csv"
    summary_path = output_dir / f"m1_anchor_sensitivity_{args.split}_summary.csv"

    detail_rows: list[dict[str, float | int | str]] = []
    response_rows: list[dict[str, float | int | str]] = []
    completed: set[str] = set()
    if bool(args.resume) and detail_path.exists() and response_path.exists():
        detail_existing = pd.read_csv(detail_path)
        response_existing = pd.read_csv(response_path)
        detail_rows = detail_existing.to_dict(orient="records")
        response_rows = response_existing.to_dict(orient="records")
        required_methods = {f"RF-{count}" for count in counts}
        if not bool(args.skip_extra):
            required_methods.add(f"Extra Trees-{int(args.extra_trees)}")
        if not bool(args.skip_gradient_boosting):
            required_methods.add("Gradient boosting")
        for scene_id, group in detail_existing.groupby("scene_id"):
            if required_methods.issubset(set(group["method"].astype(str))):
                completed.add(str(scene_id))

    pending = [record for record in records if str(record["scene_id"]) not in completed]
    for index, record in enumerate(pending):
        started = time.perf_counter()
        sample = load_sample_npz(root / record["relative_path"])
        observations = sample["observations"]
        mask = observations.mask & (
            observations.type_ids == OBS_TYPES["borehole_eic"]
        )
        if int(np.sum(mask)) < 4:
            continue
        x_query, _, shape = _grid_points_and_surface(sample)
        x_train = _surface_at_obs(sample, observations.coords[mask])
        y_train = np.asarray(observations.values[mask], dtype=np.float32)
        seed = int(record["seed"]) + 91
        predictions = _fit_predict_rf_prefixes(
            x_train,
            y_train,
            x_query,
            counts=counts,
            seed=seed,
            n_jobs=int(args.n_jobs),
        )
        predictions.update(
            _fit_predict_alternatives(
                x_train,
                y_train,
                x_query,
                seed=seed,
                extra_trees=int(args.extra_trees),
                n_jobs=int(args.n_jobs),
                include_extra=not bool(args.skip_extra),
                include_gradient_boosting=not bool(args.skip_gradient_boosting),
            )
        )
        truth = np.asarray(sample["fields"]["eic"], dtype=np.float32)
        for method, flat_prediction in predictions.items():
            prediction = np.clip(flat_prediction.reshape(shape), 0.0, 0.90)
            detail_rows.append(
                {
                    "scene_id": str(record["scene_id"]),
                    "generator_family": str(record["generator_family"]),
                    "method": str(method),
                    "seed": int(args.seed),
                    "n_eic_observations": int(np.sum(mask)),
                    "eic_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                }
            )
            for thaw_depth_m in args.engineering_response_depths:
                response_rows.append(
                    engineering_response_metrics(
                        scene_id=str(record["scene_id"]),
                        method=str(method),
                        seed=int(args.seed),
                        truth_eic=truth,
                        candidate_eic_mean=prediction,
                        grid=sample["grid"],
                        thaw_depth_m=float(thaw_depth_m),
                        screening_threshold_m=float(
                            args.engineering_response_threshold
                        ),
                    )
                )
        pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
        pd.DataFrame(response_rows).to_csv(response_path, index=False)
        print(
            f"{index + 1}/{len(pending)} {record['scene_id']} "
            f"methods={len(predictions)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    detail = pd.DataFrame(detail_rows)
    response = pd.DataFrame(response_rows)
    summary_rows: list[dict[str, float | int | str]] = []
    if not detail.empty:
        for method, group in detail.groupby("method", sort=False):
            mean, lower, upper = _bootstrap_mean_ci(
                group["eic_rmse"].astype(float).tolist(), int(args.seed) + 1
            )
            summary_rows.append(
                {
                    "method": str(method),
                    "thaw_depth_m": float("nan"),
                    "metric": "eic_rmse",
                    "mean": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "n_scenes": int(group["scene_id"].nunique()),
                }
            )
        reference_method = f"RF-{counts[0]}"
        detail_pivot = detail.pivot(
            index="scene_id", columns="method", values="eic_rmse"
        )
        if reference_method in detail_pivot.columns:
            for method in detail_pivot.columns:
                if str(method) == reference_method:
                    continue
                difference = (
                    detail_pivot[str(method)] - detail_pivot[reference_method]
                ).dropna()
                mean, lower, upper = _bootstrap_mean_ci(
                    difference.astype(float).tolist(), int(args.seed) + 19
                )
                summary_rows.append(
                    {
                        "method": str(method),
                        "thaw_depth_m": float("nan"),
                        "metric": f"paired_eic_rmse_difference_vs_{reference_method}",
                        "mean": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "n_scenes": int(difference.size),
                    }
                )
    if not response.empty:
        metrics = (
            "response_rmse_m",
            "response_bias_m",
            "gradient_rmse_m_per_m",
            "sensitivity",
            "specificity",
        )
        for (method, depth), group in response.groupby(
            ["method", "thaw_depth_m"], sort=False
        ):
            for metric in metrics:
                values = group[metric].astype(float).to_numpy()
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                mean, lower, upper = _bootstrap_mean_ci(
                    values.tolist(), int(args.seed) + int(round(10 * float(depth)))
                )
                summary_rows.append(
                    {
                        "method": str(method),
                        "thaw_depth_m": float(depth),
                        "metric": str(metric),
                        "mean": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "n_scenes": int(group["scene_id"].nunique()),
                    }
                )
        response_pivot = response.pivot(
            index=["scene_id", "thaw_depth_m"],
            columns="method",
            values="response_rmse_m",
        )
        reference_method = f"RF-{counts[0]}"
        if reference_method in response_pivot.columns:
            for method in response_pivot.columns:
                if str(method) == reference_method:
                    continue
                for depth in sorted(
                    response_pivot.index.get_level_values("thaw_depth_m").unique()
                ):
                    difference = (
                        response_pivot.xs(depth, level="thaw_depth_m")[str(method)]
                        - response_pivot.xs(depth, level="thaw_depth_m")[
                            reference_method
                        ]
                    ).dropna()
                    mean, lower, upper = _bootstrap_mean_ci(
                        difference.astype(float).tolist(),
                        int(args.seed) + 23 + int(round(float(depth) * 10)),
                    )
                    summary_rows.append(
                        {
                            "method": str(method),
                            "thaw_depth_m": float(depth),
                            "metric": f"paired_response_rmse_difference_vs_{reference_method}",
                            "mean": mean,
                            "ci95_lower": lower,
                            "ci95_upper": upper,
                            "n_scenes": int(difference.size),
                        }
                    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metadata = {
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "split": str(args.split),
        "scenes": int(detail["scene_id"].nunique()) if not detail.empty else 0,
        "rf_tree_counts": list(counts),
        "extra_trees": 0 if bool(args.skip_extra) else int(args.extra_trees),
        "gradient_boosting": not bool(args.skip_gradient_boosting),
        "independent_unit": "controlled scene",
        "detail": str(detail_path),
        "response": str(response_path),
        "summary": str(summary_path),
    }
    (output_dir / f"m1_anchor_sensitivity_{args.split}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
