from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.block_conformal import posterior_diagnostics
from cold_recon.models.autoencoder3d import Autoencoder3D
from cold_recon.models.mixed_ablation_sampling import (
    sample_mixed_ablation_ensemble,
)
from cold_recon.models.support_aware_residual_flow import SupportAwareResidualFlow3D
from cold_recon.training.mixed_volume_codec import (
    MIXED_CHANNELS,
    mixed_ensemble_to_posterior,
    prior_to_mixed_tensor,
)
from scripts.evaluate_m1_controlled import (
    _bootstrap_mean_ci,
    _rmse,
    categorical_iou,
    support_misfit_by_type,
)
from scripts.train_m1_mixed_ablation_flow import _mixed_context_raster
from scripts.train_m1_support_guided_flow import (
    _load_or_build_prior,
    _manifest_records,
    _subsample_tokens,
    _token_tensor,
)


def _load_bundle(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    auto_saved = torch.load(
        saved["autoencoder_checkpoint"], map_location="cpu", weights_only=False
    )
    autoencoder = Autoencoder3D(
        in_channels=MIXED_CHANNELS,
        latent_channels=int(auto_saved["latent_channels"]),
        base=int(auto_saved["base_channels"]),
    ).to(device)
    autoencoder.load_state_dict(auto_saved["model_state"])
    autoencoder.eval()
    variant = saved["variant"]
    cfg = config["model"]
    model = SupportAwareResidualFlow3D(
        latent_channels=int(cfg["latent_channels"]),
        raster_in_channels=int(saved["raster_channels"]),
        token_dim=int(cfg["token_dim"]),
        context_channels=int(cfg["context_channels"]),
        width=int(cfg["width"]),
        modes=tuple(int(value) for value in cfg["modes"]),
        depth=int(cfg["depth"]),
        attention_hidden=int(cfg["attention_hidden"]),
        attention_chunk=int(cfg["attention_chunk"]),
        support_extent_offset=int(cfg["support_extent_offset"]),
        use_token_conditioning=bool(variant["token_conditioning"]),
    ).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return saved, config, autoencoder, model


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    saved, config, autoencoder, model = _load_bundle(Path(args.checkpoint), device)
    manifest_path = Path(args.manifest or config["m1_training"]["manifest"])
    root, records, manifest = _manifest_records(manifest_path, args.split)
    if int(args.max_scenes) > 0:
        records = records[: int(args.max_scenes)]
    positions = {record["scene_id"]: i for i, record in enumerate(records)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_id = str(args.ablation_id or saved["ablation_id"]).upper()
    detail_path = output_dir / f"m1_{ablation_id.lower()}_{args.split}_seed{saved['seed']}_detail.csv"
    rows: list[dict[str, Any]] = []
    if bool(args.resume) and detail_path.exists():
        existing = pd.read_csv(detail_path)
        rows = existing.to_dict(orient="records")
        completed = set(existing["scene_id"].astype(str))
        records = [record for record in records if record["scene_id"] not in completed]
    for remaining_index, record in enumerate(records):
        scene_index = int(positions[record["scene_id"]])
        sample = load_sample_npz(root / record["relative_path"])
        prior = _load_or_build_prior(
            sample,
            record,
            root / "prior_cache" / args.split,
            int(config["model"]["n_facies"]),
            int(args.rf_trees),
        )
        prior_tensor = prior_to_mixed_tensor(prior)
        anchor = autoencoder.encode(prior_tensor.to(device))
        raster = _mixed_context_raster(
            sample,
            prior_tensor,
            sample["observations"],
            support_raster=str(saved["variant"]["support_raster"]),
        ).to(device)
        token_obs = _subsample_tokens(
            sample["observations"],
            int(config["model"]["max_condition_tokens"]),
            np.random.default_rng(int(args.seed) + scene_index),
        )
        tokens = _token_tensor(token_obs, sample, config, device)
        guidance = (
            float(args.guidance_strength)
            if ablation_id == "A4"
            else 0.0
        )
        decoded, diagnostics = sample_mixed_ablation_ensemble(
            model=model,
            autoencoder=autoencoder,
            anchor=anchor,
            raster=raster,
            tokens=tokens,
            sample=sample,
            n_members=int(args.posterior_members),
            sampling_steps=int(args.sampling_steps),
            guidance_strength=guidance,
            guidance_batch_size=int(args.guidance_batch_size),
            seed=int(args.seed) + scene_index,
        )
        posterior = mixed_ensemble_to_posterior(decoded)
        truth = sample["fields"]
        probabilistic = posterior_diagnostics(
            posterior["eic_samples"], truth["eic"]
        )
        support = support_misfit_by_type(posterior, sample)
        rows.append(
            {
                "scene_id": record["scene_id"],
                "split": record["split"],
                "generator_family": record["generator_family"],
                "seed": int(saved["seed"]),
                "ablation_id": ablation_id,
                "cryofacies_miou": categorical_iou(
                    posterior["cryofacies_mode"], truth["facies"], 7
                ),
                "eic_rmse": _rmse(posterior["eic_mean"], truth["eic"]),
                "anchor_eic_rmse": _rmse(prior["eic"], truth["eic"]),
                "temperature_rmse": _rmse(
                    posterior["temperature_mean"], truth["temperature"]
                ),
                "unfrozen_water_rmse": _rmse(
                    posterior["unfrozen_water_mean"], truth["unfrozen_water"]
                ),
                "log_resistivity_rmse": _rmse(
                    posterior["log_resistivity_mean"],
                    np.log(np.maximum(truth["resistivity"], 1.0)),
                ),
                **{f"eic_{key}": value for key, value in probabilistic.items()},
                **support,
                **diagnostics,
            }
        )
        pd.DataFrame(rows).to_csv(detail_path, index=False)
        print(
            f"{remaining_index + 1}/{len(records)} {ablation_id} "
            f"{record['scene_id']} eic_rmse={rows[-1]['eic_rmse']:.4f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(detail_path, index=False)
    numeric = [
        column
        for column in frame.columns
        if column not in {"seed"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary_rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(numeric):
        mean, lower, upper = _bootstrap_mean_ci(
            frame[metric].astype(float).tolist(),
            int(args.seed) + 33 + metric_index,
        )
        summary_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_scenes": int(len(frame)),
            }
        )
    summary_path = detail_path.with_name(detail_path.name.replace("_detail", "_summary"))
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metadata = {
        "ablation_id": ablation_id,
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": int(saved["seed"]),
        "checkpoint_training_ablation": str(saved["ablation_id"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "scenes": int(len(frame)),
        "posterior_members": int(args.posterior_members),
        "sampling_steps": int(args.sampling_steps),
        "guidance_strength": (
            float(args.guidance_strength) if ablation_id == "A4" else 0.0
        ),
        "detail": str(detail_path),
        "summary": str(summary_path),
    }
    detail_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ablation-id", default="", choices=["", "A2", "A3", "A4", "A5", "A6"])
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--posterior-members", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--guidance-strength", type=float, default=2.0)
    parser.add_argument("--guidance-batch-size", type=int, default=16)
    parser.add_argument("--rf-trees", type=int, default=24)
    parser.add_argument("--seed", type=int, default=9200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default="outputs/m1_support_guided/formal_ablation",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
