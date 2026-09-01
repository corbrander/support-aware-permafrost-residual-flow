from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cold_recon.data.data_schema import ObservationTable, load_sample_npz
from cold_recon.data.support_raster import PreparedSupportRaster
from cold_recon.models.m1_sampling import sample_support_guided_ensemble
from cold_recon.training.factorized_volume_codec import factorized_ensemble_to_posterior
from scripts.evaluate_m1_controlled import load_bundle, support_misfit_by_type
from scripts.train_m1_support_guided_flow import (
    _context_raster,
    _load_or_build_prior,
    _manifest_records,
    _prior_tensor,
    _subsample_tokens,
    _token_tensor,
)


def _with_sigma_multiplier(observations: ObservationTable, multiplier: float) -> ObservationTable:
    out = observations.subset(np.arange(observations.n_obs))
    out.sigma = (out.sigma * float(multiplier)).astype(np.float32)
    return out


def _distance_to_observations(sample: dict) -> np.ndarray:
    grid = sample["grid"]
    obs = sample["observations"]
    xx, yy, zz = np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij")
    distance = np.full(xx.shape, np.inf, dtype=np.float32)
    for coord in obs.coords[obs.mask][:: max(1, int(np.sum(obs.mask) // 512))]:
        current = np.sqrt((xx - coord[0]) ** 2 + (yy - coord[1]) ** 2 + (zz - coord[2]) ** 2)
        distance = np.minimum(distance, current)
    return distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--multipliers", default="0.5,1,2,4")
    parser.add_argument("--posterior-members", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=5)
    parser.add_argument("--guidance-strength", type=float, default=2.0)
    parser.add_argument("--rf-trees", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/m1_support_guided/tables/noise_response.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    saved, config, autoencoder, model, bias_head, event_head = load_bundle(Path(args.checkpoint), device)
    manifest_path = Path(args.manifest or config["m1_training"]["manifest"])
    root, records, _ = _manifest_records(manifest_path, args.split)
    records = records[: int(args.max_scenes)]
    record_positions = {
        str(record["scene_id"]): index for index, record in enumerate(records)
    }
    multipliers = [float(value) for value in args.multipliers.split(",")]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if bool(args.resume) and output.exists():
        existing = pd.read_csv(output)
        rows = existing.to_dict(orient="records")
        counts = existing.groupby("scene_id")["multiplier"].nunique()
        complete = set(counts[counts >= len(multipliers)].index.astype(str))
        records = [record for record in records if record["scene_id"] not in complete]
    for remaining_index, record in enumerate(records):
        scene_index = int(record_positions[str(record["scene_id"])])
        sample = load_sample_npz(root / record["relative_path"])
        prior = _load_or_build_prior(
            sample,
            record,
            root / "prior_cache" / args.split,
            int(config["model"]["n_facies"]),
            int(args.rf_trees),
        )
        prior_tensor = _prior_tensor(prior).to(device)
        with torch.no_grad():
            anchor = autoencoder.encode(prior_tensor)
        prepared = PreparedSupportRaster.prepare(sample["observations"], sample["grid"])
        distance = _distance_to_observations(sample)
        distant = distance >= np.nanmedian(distance)
        posteriors: dict[float, dict[str, np.ndarray]] = {}
        for multiplier in multipliers:
            observations = _with_sigma_multiplier(sample["observations"], multiplier)
            local_sample = dict(sample)
            local_sample["observations"] = observations
            raster = _context_raster(
                local_sample,
                prior_tensor.cpu(),
                observations,
                prepared_support=prepared,
            ).to(device)
            token_obs = _subsample_tokens(
                observations,
                int(config["model"]["max_condition_tokens"]),
                np.random.default_rng(8000 + scene_index),
            )
            tokens = _token_tensor(token_obs, local_sample, config, device)
            decoded, diagnostics = sample_support_guided_ensemble(
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
                seed=9000 + scene_index,
            )
            posterior = factorized_ensemble_to_posterior(decoded)
            posteriors[multiplier] = posterior
            misfit = support_misfit_by_type(posterior, local_sample)
            rows.append(
                {
                    "scene_id": record["scene_id"],
                    "model_seed": int(saved["seed"]),
                    "multiplier": multiplier,
                    "posterior_spread_mean": float(np.mean(posterior["eic_std"])),
                    "support_nrmse_eic": misfit.get("support_nrmse_borehole_eic", np.nan),
                    "bias_gate_mean": diagnostics["bias_gate_mean"],
                }
            )
        reference = posteriors[min(multipliers, key=lambda value: abs(value - 1.0))]["eic_mean"]
        for row in rows[-len(multipliers) :]:
            current = posteriors[float(row["multiplier"])]["eic_mean"]
            shift = np.abs(current - reference)
            row["mean_shift_from_nominal"] = float(np.mean(shift))
            row["distant_shift_from_nominal"] = float(np.mean(shift[distant]))
            if np.any(shift > 0.0):
                influential = shift >= 0.50 * float(np.max(shift))
                row["influence_radius_m"] = float(np.max(distance[influential]))
            else:
                row["influence_radius_m"] = 0.0
        pd.DataFrame(rows).to_csv(output, index=False)
        print(
            f"{remaining_index + 1}/{len(records)} {record['scene_id']} "
            f"noise levels={len(multipliers)}"
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    summary_rows: list[dict] = []
    for multiplier, subset in frame.groupby("multiplier"):
        for metric in (
            "posterior_spread_mean",
            "support_nrmse_eic",
            "mean_shift_from_nominal",
            "distant_shift_from_nominal",
            "influence_radius_m",
            "bias_gate_mean",
        ):
            values = subset[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                mean = lower = upper = float("nan")
            else:
                rng = np.random.default_rng(17_000 + int(round(100 * multiplier)))
                draw = rng.integers(0, len(values), size=(2000, len(values)))
                means = values[draw].mean(axis=1)
                mean = float(values.mean())
                lower, upper = (float(value) for value in np.quantile(means, [0.025, 0.975]))
            summary_rows.append(
                {
                    "multiplier": float(multiplier),
                    "metric": metric,
                    "mean": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "n_scenes": int(len(subset)),
                }
            )
    summary_path = output.with_name(output.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    monotone_scenes = 0
    for _, subset in frame.groupby("scene_id"):
        ordered = subset.sort_values("multiplier")["posterior_spread_mean"].to_numpy()
        monotone_scenes += int(np.all(np.diff(ordered) >= -1.0e-6))
    metadata = {
        "output": str(output),
        "summary": str(summary_path),
        "rows": len(rows),
        "scenes": int(frame["scene_id"].nunique()),
        "model_seed": int(saved["seed"]),
        "spread_monotonic_scene_fraction": monotone_scenes
        / max(len(record_positions), 1),
        "interpretation": "sigma multipliers alter declared uncertainty while observation values are fixed",
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
