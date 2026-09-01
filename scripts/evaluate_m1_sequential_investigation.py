from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cold_recon.data.data_schema import OBS_TYPES, load_sample_npz
from cold_recon.evaluation.sequential_acquisition import (
    BoreholeCandidate,
    run_sequential_backtest,
    summarize_policy_regret,
)
from cold_recon.models.m1_sampling import sample_support_guided_ensemble
from cold_recon.training.factorized_volume_codec import factorized_ensemble_to_posterior
from scripts.evaluate_m1_controlled import load_bundle
from scripts.train_m1_support_guided_flow import (
    _context_raster,
    _prior_tensor,
    _subsample_tokens,
    _token_tensor,
)
from scripts.build_tree_prior_residual_posterior import tree_prior_fields


BOREHOLE_TYPES = {
    OBS_TYPES["borehole_facies"],
    OBS_TYPES["borehole_eic"],
    OBS_TYPES["borehole_temperature"],
}


def _candidate_boreholes(sample: dict) -> list[BoreholeCandidate]:
    observations = sample["observations"]
    borehole = np.isin(observations.type_ids, list(BOREHOLE_TYPES)) & (observations.group_ids >= 0)
    candidates: list[BoreholeCandidate] = []
    x = np.asarray(sample["grid"]["x"])
    y = np.asarray(sample["grid"]["y"])
    for group_id in np.unique(observations.group_ids[borehole]):
        coords = observations.coords[borehole & (observations.group_ids == group_id)]
        center = coords.mean(axis=0)
        candidates.append(
            BoreholeCandidate(
                candidate_id=int(group_id),
                x_index=int(np.argmin(np.abs(x - center[0]))),
                y_index=int(np.argmin(np.abs(y - center[1]))),
            )
        )
    return candidates


def _initial_space_filling(candidates: list[BoreholeCandidate], count: int) -> tuple[int, ...]:
    if len(candidates) <= int(count):
        return tuple(candidate.candidate_id for candidate in candidates)
    selected = [candidates[0]]
    while len(selected) < int(count):
        available = [candidate for candidate in candidates if candidate not in selected]
        choice = max(
            available,
            key=lambda candidate: min(
                np.hypot(candidate.x_index - other.x_index, candidate.y_index - other.y_index)
                for other in selected
            ),
        )
        selected.append(choice)
    return tuple(candidate.candidate_id for candidate in selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--policies", default="random,grid_space_filling,farthest,variance,entropy,high_eic_probability,composite,expected_loss")
    parser.add_argument("--initial-boreholes", type=int, default=3)
    parser.add_argument("--additions", type=int, default=5)
    parser.add_argument("--posterior-members", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=3)
    parser.add_argument("--guidance-strength", type=float, default=2.0)
    parser.add_argument("--rf-trees", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/m1_support_guided/tables/sequential_investigation.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    saved, config, autoencoder, model, bias_head, event_head = load_bundle(Path(args.checkpoint), device)
    manifest_path = Path(args.manifest or config["m1_training"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_records = [record for record in manifest["records"] if record["split"] == args.split]
    split_positions = {
        str(record["scene_id"]): index for index, record in enumerate(split_records)
    }
    required_candidates = int(args.initial_boreholes) + int(args.additions)
    candidate_counts: dict[str, int] = {}
    eligible_records: list[dict] = []
    for record in split_records:
        eligibility_sample = load_sample_npz(
            manifest_path.parent / record["relative_path"]
        )
        count = len(_candidate_boreholes(eligibility_sample))
        candidate_counts[str(record["scene_id"])] = int(count)
        if count >= required_candidates:
            eligible_records.append(record)
    if len(eligible_records) < int(args.max_scenes):
        raise RuntimeError(
            "Sequential protocol requires "
            f"{required_candidates} borehole candidates per scene, but only "
            f"{len(eligible_records)} eligible {args.split} scenes are available "
            f"for the requested {int(args.max_scenes)}."
        )
    records = eligible_records[: int(args.max_scenes)]
    record_positions = {
        str(record["scene_id"]): split_positions[str(record["scene_id"])]
        for record in records
    }
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    regret_path = output.with_name(output.stem + "_regret.csv")
    detail_rows: list[dict] = []
    regret_rows: list[dict] = []
    if bool(args.resume) and output.exists() and regret_path.exists():
        existing_detail = pd.read_csv(output)
        existing_regret = pd.read_csv(regret_path)
        valid_scene_ids = {str(record["scene_id"]) for record in records}
        existing_detail = existing_detail.loc[
            existing_detail["scene_id"].astype(str).isin(valid_scene_ids)
        ].copy()
        existing_regret = existing_regret.loc[
            existing_regret["scene_id"].astype(str).isin(valid_scene_ids)
        ].copy()
        expected_steps = set(range(int(args.additions) + 1))
        expected_policies = set(policies)
        complete_detail = {
            str(scene_id)
            for scene_id, subset in existing_detail.groupby("scene_id")
            if len(subset) == len(policies) * len(expected_steps)
            and set(subset["policy"].astype(str)) == expected_policies
            and set(subset["step"].astype(int)) == expected_steps
        }
        complete_regret = {
            str(scene_id)
            for scene_id, subset in existing_regret.groupby("scene_id")
            if len(subset) == len(policies)
            and set(subset["policy"].astype(str)) == expected_policies
        }
        complete = complete_detail & complete_regret
        existing_detail = existing_detail.loc[
            existing_detail["scene_id"].astype(str).isin(complete)
        ].copy()
        existing_regret = existing_regret.loc[
            existing_regret["scene_id"].astype(str).isin(complete)
        ].copy()
        detail_rows = existing_detail.to_dict(orient="records")
        regret_rows = existing_regret.to_dict(orient="records")
        records = [record for record in records if record["scene_id"] not in complete]
    for remaining_index, record in enumerate(records):
        scene_index = int(record_positions[str(record["scene_id"])])
        sample = load_sample_npz(manifest_path.parent / record["relative_path"])
        candidates = _candidate_boreholes(sample)
        initial = _initial_space_filling(candidates, int(args.initial_boreholes))
        posterior_cache: dict[tuple[int, ...], dict[str, np.ndarray]] = {}

        def reconstruct(active_ids: tuple[int, ...]) -> dict[str, np.ndarray]:
            if active_ids in posterior_cache:
                return posterior_cache[active_ids]
            observations = sample["observations"].subset(np.arange(sample["observations"].n_obs))
            borehole = np.isin(observations.type_ids, list(BOREHOLE_TYPES))
            observations.mask[borehole & ~np.isin(observations.group_ids, active_ids)] = False
            local_sample = dict(sample)
            local_sample["observations"] = observations
            prior = tree_prior_fields(
                local_sample,
                n_facies=int(config["model"]["n_facies"]),
                seed=int(args.seed) + scene_index + len(active_ids),
                rf_trees=int(args.rf_trees),
            )
            prior_tensor = _prior_tensor(prior).to(device)
            with torch.no_grad():
                anchor = autoencoder.encode(prior_tensor)
            raster = _context_raster(local_sample, prior_tensor.cpu(), observations).to(device)
            token_obs = _subsample_tokens(
                observations,
                int(config["model"]["max_condition_tokens"]),
                np.random.default_rng(int(args.seed) + scene_index),
            )
            tokens = _token_tensor(token_obs, local_sample, config, device)
            decoded, _ = sample_support_guided_ensemble(
                model=model,
                bias_head=bias_head,
                event_head=event_head,
                autoencoder=autoencoder,
                anchor=anchor,
                raster=raster,
                tokens=tokens,
                sample=local_sample,
                n_members=int(args.posterior_members),
                sampling_steps=int(args.sampling_steps),
                guidance_strength=float(args.guidance_strength),
                seed=int(args.seed) + scene_index,
            )
            posterior = factorized_ensemble_to_posterior(decoded)
            event_probability = np.clip(
                posterior["ice_rich_probability"], 0.0, 1.0
            )
            posterior["engineering_risk"] = np.minimum(
                1.0 * (1.0 - event_probability),
                5.0 * event_probability,
            ).astype(np.float32)
            posterior_cache[active_ids] = posterior
            return posterior

        policy_rows = {}
        for policy in policies:
            steps = run_sequential_backtest(
                reconstruct,
                sample["fields"],
                candidates,
                initial_candidate_ids=initial,
                additions=int(args.additions),
                policy=policy,
                seed=int(args.seed) + scene_index,
            )
            policy_rows[policy] = steps
            detail_rows.extend(
                {
                    "scene_id": record["scene_id"],
                    "model_seed": int(saved["seed"]),
                    **vars(step),
                }
                for step in steps
            )
        regret_rows.extend(
            {
                "scene_id": record["scene_id"],
                "model_seed": int(saved["seed"]),
                **row,
            }
            for row in summarize_policy_regret(policy_rows)
        )
        pd.DataFrame(detail_rows).to_csv(output, index=False)
        pd.DataFrame(regret_rows).to_csv(regret_path, index=False)
        print(f"{remaining_index + 1}/{len(records)} {record['scene_id']} reconstructions={len(posterior_cache)}")

    detail_frame = pd.DataFrame(detail_rows)
    detail_frame.to_csv(output, index=False)
    regret_frame = pd.DataFrame(regret_rows)
    regret_frame.to_csv(regret_path, index=False)

    def bootstrap_rows(
        frame: pd.DataFrame,
        grouping: list[str],
        metrics: list[str],
        seed_offset: int,
    ) -> list[dict]:
        summary: list[dict] = []
        for group_values, subset in frame.groupby(grouping):
            group_values = group_values if isinstance(group_values, tuple) else (group_values,)
            group_metadata = dict(zip(grouping, group_values, strict=True))
            for metric_index, metric in enumerate(metrics):
                values = subset[metric].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                rng = np.random.default_rng(int(args.seed) + seed_offset + metric_index)
                draw = rng.integers(0, len(values), size=(2000, len(values)))
                means = values[draw].mean(axis=1)
                summary.append(
                    {
                        **group_metadata,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "ci95_lower": float(np.quantile(means, 0.025)),
                        "ci95_upper": float(np.quantile(means, 0.975)),
                        "n_scenes": int(len(values)),
                    }
                )
        return summary

    trajectory_summary = bootstrap_rows(
        detail_frame,
        ["policy", "step"],
        [
            "engineering_loss",
            "eic_rmse",
            "high_eic_error_rate",
            "mean_interval_width",
            "expected_decision_loss",
            "realized_decision_loss",
            "false_negative_rate",
            "false_positive_rate",
        ],
        10_000,
    )
    regret_summary = bootstrap_rows(
        regret_frame,
        ["policy"],
        [
            "initial_loss",
            "final_loss",
            "loss_reduction",
            "avoided_decision_loss",
            "regret",
        ],
        20_000,
    )
    trajectory_summary_path = output.with_name(output.stem + "_summary.csv")
    regret_summary_path = output.with_name(output.stem + "_regret_summary.csv")
    pd.DataFrame(trajectory_summary).to_csv(trajectory_summary_path, index=False)
    pd.DataFrame(regret_summary).to_csv(regret_summary_path, index=False)
    metadata = {
        "detail": str(output),
        "trajectory_summary": str(trajectory_summary_path),
        "regret": str(regret_path),
        "regret_summary": str(regret_summary_path),
        "scenes": int(detail_frame["scene_id"].nunique()),
        "model_seed": int(saved["seed"]),
        "protocol": "actual reconstruct-select-update cycles",
        "eligibility_rule": (
            f"At least {required_candidates} unique borehole candidates: "
            f"{int(args.initial_boreholes)} initial plus {int(args.additions)} additions."
        ),
        "eligible_scene_ids": [
            str(record["scene_id"])
            for record in eligible_records[: int(args.max_scenes)]
        ],
        "candidate_counts": {
            str(record["scene_id"]): candidate_counts[str(record["scene_id"])]
            for record in eligible_records[: int(args.max_scenes)]
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
