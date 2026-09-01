from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from cold_recon.evaluation.block_conformal import (
    BlockConformalCalibrator,
    interval_diagnostics,
    posterior_diagnostics,
)
from cold_recon.evaluation.engineering_response import (
    DEFAULT_THAW_DEPTHS_M,
    engineering_response_metrics,
)
from cold_recon.evaluation.rare_structure_metrics import (
    binary_event_metrics,
    high_eic_object_metrics,
)
from cold_recon.evaluation.ood_control import (
    MahalanobisOODController,
    MaxScoreOODController,
    observation_ood_features,
    scene_ood_features,
)
from cold_recon.data.data_schema import OBS_TYPES
from cold_recon.operators.support import (
    apply_surface_crossing,
    build_error_covariance,
    build_observation_operator,
    normalized_misfit,
)
from cold_recon.data.support_raster import collapse_to_nearest_voxel_observations
from cold_recon.models.m1_sampling import sample_support_guided_ensemble
from cold_recon.training.factorized_volume_codec import (
    bounded_recenter_samples,
    factorized_ensemble_to_posterior,
    tensor_to_factorized_fields,
)

from scripts.train_m1_support_guided_flow import (
    _build_models,
    _context_raster,
    _load_autoencoder,
    _load_or_build_prior,
    _manifest_records,
    _prior_tensor,
    _subsample_tokens,
    _token_tensor,
)
from cold_recon.data.data_schema import load_sample_npz


def categorical_iou(predicted: np.ndarray, truth: np.ndarray, n_classes: int) -> float:
    values: list[float] = []
    for class_id in range(int(n_classes)):
        predicted_class = predicted == class_id
        truth_class = truth == class_id
        union = np.sum(predicted_class | truth_class)
        if union:
            values.append(float(np.sum(predicted_class & truth_class) / union))
    return float(np.mean(values)) if values else float("nan")


def _rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(truth)) ** 2)))


def apply_observation_mode(
    sample: dict[str, Any],
    mode: str,
    *,
    seed: int,
) -> dict[str, Any]:
    mode = str(mode).strip().lower()
    observations = sample["observations"].subset(
        np.arange(sample["observations"].n_obs)
    )
    if mode == "all":
        local = dict(sample)
        local["observations"] = observations
        return local
    if mode == "no_ert":
        observations.mask[observations.type_ids == OBS_TYPES["ert_log_resistivity"]] = False
    elif mode == "no_nmr":
        observations.mask[observations.type_ids == OBS_TYPES["nmr_unfrozen_water"]] = False
    elif mode == "no_temperature":
        observations.mask[
            observations.type_ids == OBS_TYPES["borehole_temperature"]
        ] = False
    elif mode == "boreholes_only":
        keep_types = {
            OBS_TYPES["borehole_facies"],
            OBS_TYPES["borehole_eic"],
            OBS_TYPES["borehole_temperature"],
        }
        observations.mask[~np.isin(observations.type_ids, list(keep_types))] = False
    elif mode in {"half_boreholes", "sparse_boreholes"}:
        borehole_types = np.isin(
            observations.type_ids,
            [
                OBS_TYPES["borehole_facies"],
                OBS_TYPES["borehole_eic"],
                OBS_TYPES["borehole_temperature"],
            ],
        )
        groups = np.unique(
            observations.group_ids[
                observations.mask & borehole_types & (observations.group_ids >= 0)
            ]
        )
        rng = np.random.default_rng(int(seed))
        retain_count = (
            max(2, int(np.ceil(len(groups) / 2)))
            if mode == "half_boreholes"
            else min(2, len(groups))
        )
        retained = (
            rng.choice(groups, size=retain_count, replace=False)
            if retain_count < len(groups)
            else groups
        )
        observations.mask[
            borehole_types & ~np.isin(observations.group_ids, retained)
        ] = False
    else:
        raise ValueError(f"unknown observation mode: {mode}")
    local = dict(sample)
    local["observations"] = observations
    return local


def support_misfit_by_type(
    posterior: dict[str, np.ndarray],
    sample: dict[str, Any],
    *,
    prefix: str = "support",
) -> dict[str, float]:
    observations = sample["observations"]
    specs = {
        OBS_TYPES["borehole_eic"]: ("borehole_eic", posterior["eic_mean"]),
        OBS_TYPES["borehole_temperature"]: ("borehole_temperature", posterior["temperature_mean"]),
        OBS_TYPES["nmr_unfrozen_water"]: ("nmr_unfrozen_water", posterior["unfrozen_water_mean"]),
        OBS_TYPES["ert_log_resistivity"]: ("ert_log_resistivity", posterior["log_resistivity_mean"]),
        OBS_TYPES["alt"]: (
            "alt",
            np.repeat(
                apply_surface_crossing(
                    posterior["temperature_mean"], sample["grid"]["z"]
                )[:, :, None],
                len(sample["grid"]["z"]),
                axis=2,
            ),
        ),
    }
    out: dict[str, float] = {}
    for type_id, (name, field) in specs.items():
        indices = np.flatnonzero(observations.mask & (observations.type_ids == type_id))
        if len(indices) == 0:
            continue
        groups = [indices]
        if type_id == OBS_TYPES["ert_log_resistivity"]:
            groups = [
                indices[observations.group_ids[indices] == group]
                for group in np.unique(observations.group_ids[indices])
            ]
        group_scores: list[tuple[int, float]] = []
        predicted_values: list[np.ndarray] = []
        observed_values: list[np.ndarray] = []
        sigma_values: list[np.ndarray] = []
        for group_indices in groups:
            operator = build_observation_operator(
                observations, sample["grid"], indices=group_indices
            )
            covariance = build_error_covariance(
                observations,
                group_indices,
                correlated=type_id == OBS_TYPES["ert_log_resistivity"],
            )
            predicted = np.asarray(operator.apply(field), dtype=np.float64)
            observed = np.asarray(
                observations.values[group_indices], dtype=np.float64
            )
            score = normalized_misfit(predicted, observed, covariance)
            group_scores.append((len(group_indices), score))
            predicted_values.append(predicted)
            observed_values.append(observed)
            sigma_values.append(
                np.maximum(
                    observations.sigma[group_indices].astype(np.float64), 1.0e-6
                )
            )
        out[f"{prefix}_nrmse_{name}"] = float(
            np.sqrt(
                sum(count * score**2 for count, score in group_scores)
                / max(sum(count for count, _ in group_scores), 1)
            )
        )
        out[f"{prefix}_bias_{name}"] = float(
            np.mean(np.concatenate(predicted_values) - np.concatenate(observed_values))
        )
        predicted_all = np.concatenate(predicted_values)
        observed_all = np.concatenate(observed_values)
        sigma_all = np.concatenate(sigma_values)
        out[f"{prefix}_standardized_bias_{name}"] = float(
            np.mean((predicted_all - observed_all) / sigma_all)
        )
    return out


def _bootstrap_mean_ci(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def spatial_block_ids(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int] = (8, 8, 6),
) -> np.ndarray:
    """Return deterministic 3-D block labels for spatial conformal calibration."""

    nx, ny, nz = (int(value) for value in shape)
    bx, by, bz = (max(int(value), 1) for value in block_shape)
    ix = np.arange(nx, dtype=np.int32)[:, None, None] // bx
    iy = np.arange(ny, dtype=np.int32)[None, :, None] // by
    iz = np.arange(nz, dtype=np.int32)[None, None, :] // bz
    nby = int(np.ceil(ny / by))
    nbz = int(np.ceil(nz / bz))
    return ((ix * nby + iy) * nbz + iz).astype(np.int32)


def _load_conformal_quantile(
    path: str,
    *,
    seed: int,
    manifest_sha256: str,
) -> float | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload["checkpoint_seed"]) != int(seed):
        raise ValueError("conformal calibration seed does not match checkpoint seed")
    if str(payload["manifest_sha256"]) != str(manifest_sha256):
        raise ValueError("conformal calibration manifest does not match evaluation manifest")
    return float(payload["global_quantile"])


def _fit_cached_spatial_conformal(
    cache_dir: Path,
    records: list[dict[str, Any]],
    *,
    level: float,
    within_block_quantile: float,
    std_floor: float,
) -> tuple[BlockConformalCalibrator, int, int]:
    all_scores: list[np.ndarray] = []
    all_blocks: list[np.ndarray] = []
    block_offset = 0
    used_scenes = 0
    for record in records:
        path = cache_dir / f"{record['scene_id']}_eic_score.npz"
        if not path.exists():
            continue
        saved = np.load(path, allow_pickle=False)
        scores = np.asarray(saved["score"], dtype=np.float32)
        block_shape = tuple(int(value) for value in saved["block_shape"])
        blocks = spatial_block_ids(scores.shape, block_shape)
        blocks = blocks + int(block_offset)
        block_offset = int(blocks.max()) + 1
        all_scores.append(scores.reshape(-1))
        all_blocks.append(blocks.reshape(-1))
        used_scenes += 1
    if not all_scores:
        raise ValueError("no cached validation scores are available for conformal fitting")
    scores = np.concatenate(all_scores)
    blocks = np.concatenate(all_blocks)
    calibrator = BlockConformalCalibrator(
        level=float(level),
        std_floor=float(std_floor),
        within_block_quantile=float(within_block_quantile),
    ).fit(
        truth=scores,
        mean=np.zeros_like(scores),
        std=np.ones_like(scores),
        block_ids=blocks,
    )
    return calibrator, used_scenes, int(block_offset)


def inflate_continuous_posterior(
    posterior: dict[str, np.ndarray],
    inflation: float,
) -> dict[str, np.ndarray]:
    """Inflate epistemic spread and refresh every dependent posterior product."""

    factor = max(float(inflation), 1.0)
    bounds = {
        "eic": (0.0, 0.90),
        "temperature": (-12.0, 4.0),
        "unfrozen_water": (0.0, 0.85),
        "log_resistivity": (0.0, 15.0),
    }
    for field_name in ("eic", "temperature", "unfrozen_water", "log_resistivity"):
        samples_key = f"{field_name}_samples"
        mean_key = f"{field_name}_mean"
        std_key = f"{field_name}_std"
        samples = np.asarray(posterior[samples_key], dtype=np.float32)
        mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
        proposed = mean[None, ...] + factor * (samples - mean[None, ...])
        lower, upper = bounds[field_name]
        posterior[samples_key] = bounded_recenter_samples(
            proposed, mean, lower, upper
        )
        posterior[mean_key] = posterior[samples_key].mean(axis=0).astype(np.float32)
        posterior[std_key] = posterior[samples_key].std(axis=0).astype(np.float32)

    posterior["resistivity_samples"] = np.exp(
        np.clip(posterior["log_resistivity_samples"], 0.0, 15.0)
    ).astype(np.float32)
    posterior["resistivity_mean"] = posterior["resistivity_samples"].mean(axis=0).astype(
        np.float32
    )
    posterior["ice_rich_probability"] = np.mean(
        posterior["eic_samples"] >= 0.30, axis=0
    ).astype(np.float32)
    return posterior


def enforce_exact_anchor_fallback(
    posterior: dict[str, np.ndarray],
    anchor_fields: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Set every reported posterior mean/mode to the conventional tree anchor."""

    bounds = {
        "eic": (0.0, 0.90),
        "temperature": (-12.0, 4.0),
        "unfrozen_water": (0.0, 0.85),
        "log_resistivity": (0.0, 15.0),
    }
    for field_name, (lower, upper) in bounds.items():
        samples_key = f"{field_name}_samples"
        target = np.asarray(anchor_fields[field_name], dtype=np.float32)
        posterior[samples_key] = bounded_recenter_samples(
            np.asarray(posterior[samples_key], dtype=np.float32),
            target,
            lower,
            upper,
        )
        posterior[f"{field_name}_mean"] = posterior[samples_key].mean(
            axis=0
        ).astype(np.float32)
        posterior[f"{field_name}_std"] = posterior[samples_key].std(
            axis=0
        ).astype(np.float32)

    for field_name, classes in (
        ("lithology", 4),
        ("thermal_state", 3),
        ("ice_structure", 3),
    ):
        mode = np.asarray(anchor_fields[field_name], dtype=np.int64)
        probability = np.eye(int(classes), dtype=np.float32)[mode]
        posterior[f"{field_name}_mode"] = mode.astype(np.int16)
        posterior[f"{field_name}_probability"] = probability
        posterior[f"{field_name}_entropy"] = np.zeros(
            mode.shape, dtype=np.float32
        )

    posterior["resistivity_samples"] = np.exp(
        np.clip(posterior["log_resistivity_samples"], 0.0, 15.0)
    ).astype(np.float32)
    posterior["resistivity_mean"] = posterior["resistivity_samples"].mean(
        axis=0
    ).astype(np.float32)
    posterior["ice_rich_probability"] = np.mean(
        posterior["eic_samples"] >= 0.30, axis=0
    ).astype(np.float32)
    return posterior


def _load_dual_ood_controller(
    path: str,
    *,
    manifest_sha256: str,
) -> MaxScoreOODController | None:
    if not path:
        return None
    saved = np.load(path, allow_pickle=False)
    metadata = json.loads(str(saved["metadata"].item()))
    if str(metadata["manifest_sha256"]) != str(manifest_sha256):
        raise ValueError("OOD controller manifest does not match evaluation manifest")
    controllers: list[MahalanobisOODController] = []
    for prefix in ("observation", "context"):
        controller = MahalanobisOODController(
            abstention_quantile=float(metadata["abstention_quantile"])
        )
        controller.mean = np.asarray(saved[f"{prefix}_mean"], dtype=np.float64)
        controller.precision = np.asarray(
            saved[f"{prefix}_precision"], dtype=np.float64
        )
        controller.reference_distances = np.asarray(
            saved[f"{prefix}_reference_distances"], dtype=np.float64
        )
        controllers.append(controller)
    return MaxScoreOODController(
        tuple(controllers),
        abstention_quantile=float(metadata["abstention_quantile"]),
    )


def load_bundle(checkpoint_path: Path, device: torch.device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    autoencoder = _load_autoencoder(config, device)
    model, bias_head, event_head = _build_models(config, int(config["model"]["latent_channels"]), device)
    model.load_state_dict(saved["model_state"])
    bias_head.load_state_dict(saved["bias_head_state"])
    event_head.load_state_dict(saved["event_head_state"], strict=False)
    event_head.calibration_temperature = torch.as_tensor(
        saved.get("event_temperature", [1.0] * len(config["evaluation"]["high_eic_thresholds"])),
        device=device,
        dtype=torch.float32,
    ).view(1, -1, 1, 1, 1)
    event_head.decision_thresholds = tuple(
        float(value)
        for value in saved.get(
            "event_decision_thresholds",
            [0.50] * len(config["evaluation"]["high_eic_thresholds"]),
        )
    )
    return saved, config, autoencoder, model, bias_head, event_head


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    saved, config, autoencoder, model, bias_head, event_head = load_bundle(Path(args.checkpoint), device)
    support_mode = str(
        args.support_mode
        if str(args.support_mode)
        else saved.get("support_mode", "support-aware")
    )
    manifest_path = Path(args.manifest or config["m1_training"]["manifest"])
    root, records, manifest = _manifest_records(manifest_path, args.split)
    record_positions = {
        str(record["scene_id"]): index for index, record in enumerate(records)
    }
    if str(args.example_scene_id).strip():
        requested_scene_id = str(args.example_scene_id).strip()
        records = [
            record
            for record in records
            if str(record["scene_id"]) == requested_scene_id
        ]
        if not records:
            raise ValueError(
                f"scene {requested_scene_id!r} is not present in split {args.split!r}"
            )
    records = records[: int(args.max_scenes)] if int(args.max_scenes) > 0 else records
    requested_records = list(records)
    conformal_quantile = _load_conformal_quantile(
        args.conformal_file,
        seed=int(saved["seed"]),
        manifest_sha256=str(manifest["manifest_sha256"]),
    )
    dual_ood_controller = (
        None
        if bool(args.disable_ood)
        else _load_dual_ood_controller(
            args.ood_controller_file,
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
    )
    ood_controller: MahalanobisOODController | None = None
    if dual_ood_controller is None and not bool(args.disable_ood):
        reference_records = [
            record for record in manifest["records"] if record["split"] == "train"
        ][: int(args.ood_reference_scenes)]
        reference_feature_rows: list[np.ndarray] = []
        for record in reference_records:
            reference_sample = load_sample_npz(root / record["relative_path"])
            reference_feature_rows.append(
                observation_ood_features(
                    reference_sample["observations"], reference_sample["grid"]
                )
            )
        reference_features = np.stack(reference_feature_rows, axis=0)
        ood_controller = MahalanobisOODController(
            abstention_quantile=float(config["evaluation"]["ood_abstention_quantile"])
        ).fit(reference_features)
    output_dir = Path(args.output_dir or config["paths"]["tables_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_suffix = "" if args.observation_mode == "all" else f"_{args.observation_mode}"
    detail_path = output_dir / f"m1_{args.split}{mode_suffix}_seed{saved['seed']}_detail.csv"
    response_path = output_dir / (
        f"m1_{args.split}{mode_suffix}_seed{saved['seed']}_engineering_response.csv"
    )
    conformal_cache_dir = (
        output_dir
        / "conformal_cache"
        / f"{args.split}{mode_suffix}_seed{saved['seed']}"
    )
    if bool(args.fit_conformal):
        if args.split != "validation" or args.observation_mode != "all":
            raise ValueError("spatial conformal fitting is restricted to the complete validation split")
        conformal_cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    completed_scene_ids: set[str] = set()
    if bool(args.resume) and detail_path.exists():
        existing = pd.read_csv(detail_path)
        completed_scene_ids = set(existing["scene_id"].astype(str))
        if bool(args.fit_conformal):
            completed_scene_ids = {
                scene_id
                for scene_id in completed_scene_ids
                if (conformal_cache_dir / f"{scene_id}_eic_score.npz").exists()
            }
            existing = existing[
                existing["scene_id"].astype(str).isin(completed_scene_ids)
            ]
        rows = existing.to_dict(orient="records")
        records = [
            record
            for record in records
            if str(record["scene_id"]) not in completed_scene_ids
        ]
    cache_dir = root / "prior_cache" / (
        args.split if args.observation_mode == "all" else f"{args.split}_{args.observation_mode}"
    )
    for remaining_index, record in enumerate(records):
        scene_started = time.perf_counter()
        index = int(record_positions[str(record["scene_id"])])
        full_sample = load_sample_npz(root / record["relative_path"])
        sample = apply_observation_mode(
            full_sample,
            args.observation_mode,
            seed=int(args.seed) + index,
        )
        conditioning_sample = sample
        if support_mode == "nearest-voxel":
            conditioning_sample = dict(sample)
            conditioning_sample["observations"] = collapse_to_nearest_voxel_observations(
                sample["observations"], sample["grid"]
            )
        prior = _load_or_build_prior(
            sample,
            record,
            cache_dir,
            int(config["model"]["n_facies"]),
            int(args.rf_trees or config["m1_training"]["rf_trees"]),
        )
        observation_features = observation_ood_features(
            sample["observations"], sample["grid"]
        )[None, :]
        if bool(args.disable_ood):
            ood_control = {
                "ood_score": np.asarray([0.0], dtype=np.float64),
                "ood_risk": np.asarray([0.0], dtype=np.float64),
                "bias_gate_multiplier": np.asarray([1.0], dtype=np.float64),
                "interval_inflation": np.asarray([1.0], dtype=np.float64),
                "abstain": np.asarray([False], dtype=bool),
            }
        elif dual_ood_controller is not None:
            context_features = scene_ood_features(
                sample["observations"], sample["grid"], prior
            )[None, :]
            ood_control = dual_ood_controller.control(
                (observation_features, context_features)
            )
        else:
            if ood_controller is None:
                raise RuntimeError("OOD controller was not initialized")
            ood_control = ood_controller.control(observation_features)
        prior_tensor = _prior_tensor(prior).to(device)
        with torch.no_grad():
            anchor = autoencoder.encode(prior_tensor)
        rng = np.random.default_rng(int(args.seed) + index)
        token_obs = _subsample_tokens(
            conditioning_sample["observations"],
            int(config["model"]["max_condition_tokens"]),
            rng,
        )
        raster = _context_raster(
            conditioning_sample,
            prior_tensor.cpu(),
            conditioning_sample["observations"],
            support_mode=support_mode,
        ).to(device)
        tokens = _token_tensor(token_obs, sample, config, device)
        decoded, diagnostics = sample_support_guided_ensemble(
            model=model,
            bias_head=bias_head,
            event_head=event_head,
            autoencoder=autoencoder,
            anchor=anchor,
            raster=raster,
            tokens=tokens,
            sample=conditioning_sample,
            n_members=int(args.posterior_members),
            sampling_steps=int(args.sampling_steps),
            guidance_strength=float(args.guidance_strength),
            guidance_batch_size=int(args.guidance_batch_size),
            ood_gate_multiplier=float(ood_control["bias_gate_multiplier"][0]),
            use_bias_decomposition=bool(
                saved.get("variant", {}).get(
                    "bias_anomaly_decomposition", True
                )
            ),
            seed=int(args.seed) + index,
        )
        posterior = factorized_ensemble_to_posterior(decoded)
        anchor_fields = tensor_to_factorized_fields(prior_tensor)
        fallback_applied = bool(ood_control["abstain"][0])
        if fallback_applied:
            posterior = enforce_exact_anchor_fallback(posterior, anchor_fields)
        inflation = float(ood_control["interval_inflation"][0])
        posterior = inflate_continuous_posterior(posterior, inflation)
        truth = sample["fields"]
        if bool(args.engineering_response_audit):
            for thaw_depth_m in args.engineering_response_depths:
                response_rows.append(
                    engineering_response_metrics(
                        scene_id=str(record["scene_id"]),
                        method="Tree anchor",
                        seed=int(saved["seed"]),
                        truth_eic=np.asarray(truth["eic"], dtype=np.float32),
                        candidate_eic_mean=np.asarray(
                            anchor_fields["eic"], dtype=np.float32
                        ),
                        grid=sample["grid"],
                        thaw_depth_m=float(thaw_depth_m),
                        screening_threshold_m=float(
                            args.engineering_response_threshold
                        ),
                        decision_probability=float(
                            args.engineering_response_probability
                        ),
                    )
                )
                response_rows.append(
                    engineering_response_metrics(
                        scene_id=str(record["scene_id"]),
                        method="Conditional residual flow",
                        seed=int(saved["seed"]),
                        truth_eic=np.asarray(truth["eic"], dtype=np.float32),
                        candidate_eic_mean=np.asarray(
                            posterior["eic_mean"], dtype=np.float32
                        ),
                        candidate_eic_samples=np.asarray(
                            posterior["eic_samples"], dtype=np.float32
                        ),
                        candidate_eic_std=np.asarray(
                            posterior["eic_std"], dtype=np.float32
                        ),
                        conformal_quantile=conformal_quantile,
                        grid=sample["grid"],
                        thaw_depth_m=float(thaw_depth_m),
                        screening_threshold_m=float(
                            args.engineering_response_threshold
                        ),
                        decision_probability=float(
                            args.engineering_response_probability
                        ),
                    )
                )
            pd.DataFrame(response_rows).to_csv(response_path, index=False)
        if bool(args.event_source_ensemble):
            event_probabilities = np.stack(
                [
                    np.mean(posterior["eic_samples"] >= float(threshold), axis=0)
                    for threshold in config["evaluation"]["high_eic_thresholds"]
                ],
                axis=0,
            ).astype(np.float32)
            event_decision_thresholds = (0.50,) * len(
                config["evaluation"]["high_eic_thresholds"]
            )
        elif fallback_applied:
            event_probabilities = np.stack(
                [
                    np.mean(posterior["eic_samples"] >= float(threshold), axis=0)
                    for threshold in config["evaluation"]["high_eic_thresholds"]
                ],
                axis=0,
            ).astype(np.float32)
        else:
            event_probabilities = np.asarray(diagnostics["event_probability"])[0]
        event_metrics: dict[str, float] = {}
        if not bool(args.event_source_ensemble):
            event_decision_thresholds = tuple(
                float(value)
                for value in getattr(
                    event_head,
                    "decision_thresholds",
                    [0.50] * len(config["evaluation"]["high_eic_thresholds"]),
                )
            )
        if not bool(args.skip_event_metrics):
            for threshold_index, threshold in enumerate(
                config["evaluation"]["high_eic_thresholds"]
            ):
                probability = event_probabilities[threshold_index]
                prefix = f"high_eic_t{int(round(100 * float(threshold))):02d}"
                voxel_metrics = binary_event_metrics(
                    probability,
                    truth["eic"] >= float(threshold),
                    probability_threshold=event_decision_thresholds[threshold_index],
                )
                object_metrics = high_eic_object_metrics(
                    probability,
                    truth["eic"],
                    eic_threshold=float(threshold),
                    probability_threshold=event_decision_thresholds[threshold_index],
                    dz=float(sample["grid"]["dz"]),
                )
                event_metrics.update(
                    {f"{prefix}_{key}": value for key, value in voxel_metrics.items()}
                )
                event_metrics.update(
                    {f"{prefix}_{key}": value for key, value in object_metrics.items()}
                )
        eic_probabilistic = posterior_diagnostics(posterior["eic_samples"], truth["eic"])
        calibrated_metrics: dict[str, float] = {}
        if conformal_quantile is not None:
            half_width = float(conformal_quantile) * np.maximum(
                posterior["eic_std"], float(args.conformal_std_floor)
            )
            lower = np.clip(posterior["eic_mean"] - half_width, 0.0, 0.90)
            upper = np.clip(posterior["eic_mean"] + half_width, 0.0, 0.90)
            calibrated_metrics = {
                f"eic_calibrated_{key}": value
                for key, value in interval_diagnostics(
                    truth["eic"], lower, upper
                ).items()
            }
        if bool(args.fit_conformal):
            score = np.abs(truth["eic"] - posterior["eic_mean"]) / np.maximum(
                posterior["eic_std"], float(args.conformal_std_floor)
            )
            np.savez_compressed(
                conformal_cache_dir / f"{record['scene_id']}_eic_score.npz",
                score=np.asarray(score, dtype=np.float32),
                block_shape=np.asarray(args.conformal_block_shape, dtype=np.int32),
            )
        support_misfit = support_misfit_by_type(
            posterior, sample, prefix="support"
        )
        nearest_sample = dict(sample)
        nearest_sample["observations"] = collapse_to_nearest_voxel_observations(
            sample["observations"], sample["grid"]
        )
        support_misfit.update(
            support_misfit_by_type(posterior, nearest_sample, prefix="voxel")
        )
        if str(args.example_scene_id).strip() == str(record["scene_id"]):
            example_path = Path(
                args.example_output
                or (
                    output_dir
                    / f"m1_{record['scene_id']}_seed{saved['seed']}_posterior.npz"
                )
            )
            example_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                example_path,
                scene_id=np.asarray(str(record["scene_id"])),
                manifest_sha256=np.asarray(str(manifest["manifest_sha256"])),
                checkpoint_seed=np.asarray(int(saved["seed"]), dtype=np.int32),
                x=np.asarray(sample["grid"]["x"], dtype=np.float32),
                y=np.asarray(sample["grid"]["y"], dtype=np.float32),
                z=np.asarray(sample["grid"]["z"], dtype=np.float32),
                truth_lithology=np.asarray(truth["lithology"], dtype=np.int16),
                truth_thermal_state=np.asarray(
                    truth["thermal_state"], dtype=np.int16
                ),
                truth_ice_structure=np.asarray(
                    truth["ice_structure"], dtype=np.int16
                ),
                truth_eic=np.asarray(truth["eic"], dtype=np.float32),
                truth_temperature=np.asarray(
                    truth["temperature"], dtype=np.float32
                ),
                anchor_eic=np.asarray(anchor_fields["eic"], dtype=np.float32),
                posterior_eic_mean=np.asarray(
                    posterior["eic_mean"], dtype=np.float32
                ),
                posterior_eic_std=np.asarray(
                    posterior["eic_std"], dtype=np.float32
                ),
                posterior_eic_samples=np.asarray(
                    posterior["eic_samples"], dtype=np.float32
                ),
                posterior_lithology_mode=np.asarray(
                    posterior["lithology_mode"], dtype=np.int16
                ),
                posterior_thermal_state_mode=np.asarray(
                    posterior["thermal_state_mode"], dtype=np.int16
                ),
                posterior_ice_structure_mode=np.asarray(
                    posterior["ice_structure_mode"], dtype=np.int16
                ),
                event_probability=np.asarray(
                    event_probabilities, dtype=np.float32
                ),
                event_thresholds=np.asarray(
                    config["evaluation"]["high_eic_thresholds"],
                    dtype=np.float32,
                ),
                ood_score=np.asarray(float(ood_control["ood_score"][0])),
                exact_anchor_fallback_applied=np.asarray(fallback_applied),
            )
            example_metadata = {
                "scene_id": str(record["scene_id"]),
                "selection_status": "locked before reconstruction inspection",
                "checkpoint": str(args.checkpoint),
                "checkpoint_seed": int(saved["seed"]),
                "manifest_sha256": str(manifest["manifest_sha256"]),
                "posterior_members": int(args.posterior_members),
                "sampling_steps": int(args.sampling_steps),
                "guidance_strength": float(args.guidance_strength),
                "posterior_source": str(example_path),
            }
            example_path.with_suffix(".json").write_text(
                json.dumps(example_metadata, indent=2), encoding="utf-8"
            )
        rows.append(
            {
                "scene_id": record["scene_id"],
                "split": record["split"],
                "generator_family": record["generator_family"],
                "seed": int(saved["seed"]),
                "lithology_miou": categorical_iou(
                    posterior["lithology_mode"], truth["lithology"], 4
                ),
                "thermal_state_miou": categorical_iou(
                    posterior["thermal_state_mode"], truth["thermal_state"], 3
                ),
                "ice_structure_miou": categorical_iou(
                    posterior["ice_structure_mode"], truth["ice_structure"], 3
                ),
                "eic_rmse": _rmse(posterior["eic_mean"], truth["eic"]),
                "anchor_eic_rmse": _rmse(anchor_fields["eic"], truth["eic"]),
                "temperature_rmse": _rmse(posterior["temperature_mean"], truth["temperature"]),
                "unfrozen_water_rmse": _rmse(
                    posterior["unfrozen_water_mean"], truth["unfrozen_water"]
                ),
                "log_resistivity_rmse": _rmse(
                    posterior["log_resistivity_mean"], np.log(np.maximum(truth["resistivity"], 1.0))
                ),
                **{f"eic_{key}": value for key, value in eic_probabilistic.items()},
                **calibrated_metrics,
                **event_metrics,
                **({
                    f"high_eic_t{int(round(100 * float(threshold))):02d}_decision_threshold": event_decision_thresholds[threshold_index]
                    for threshold_index, threshold in enumerate(
                        config["evaluation"]["high_eic_thresholds"]
                    )
                } if not bool(args.skip_event_metrics) else {}),
                **support_misfit,
                "bias_gate_mean": diagnostics["bias_gate_mean"],
                "local_scale_mean": diagnostics["local_scale_mean"],
                "anomaly_mean_abs": diagnostics["anomaly_mean_abs"],
                "support_likelihood_initial": diagnostics[
                    "support_likelihood_initial"
                ],
                "support_likelihood_final": diagnostics["support_likelihood_final"],
                "support_likelihood_reduction": (
                    diagnostics["support_likelihood_initial"]
                    - diagnostics["support_likelihood_final"]
                ),
                "guidance_anchor_shift_mean_abs": diagnostics[
                    "guidance_anchor_shift_mean_abs"
                ],
                "ood_score": float(ood_control["ood_score"][0]),
                "ood_risk": float(ood_control["ood_risk"][0]),
                "ood_bias_gate_multiplier": float(
                    ood_control["bias_gate_multiplier"][0]
                ),
                "ood_interval_inflation": inflation,
                "ood_abstain": bool(ood_control["abstain"][0]),
                "exact_anchor_fallback_applied": fallback_applied,
                "end_to_end_wall_seconds": time.perf_counter() - scene_started,
            }
        )
        pd.DataFrame(rows).to_csv(detail_path, index=False)
        print(
            f"{remaining_index + 1}/{len(records)} {record['scene_id']} "
            f"eic_rmse={rows[-1]['eic_rmse']:.4f} anchor={rows[-1]['anchor_eic_rmse']:.4f}"
        )

    pd.DataFrame(rows).to_csv(detail_path, index=False)
    if bool(args.engineering_response_audit):
        pd.DataFrame(response_rows).to_csv(response_path, index=False)
    numeric = (
        [
            key
            for key, value in rows[0].items()
            if key != "seed"
            and isinstance(value, (float, int))
            and all(key in row and isinstance(row[key], (float, int)) for row in rows)
        ]
        if rows
        else []
    )
    summary_rows: list[dict[str, Any]] = []
    for metric in numeric:
        mean, lower, upper = _bootstrap_mean_ci(
            [float(row[metric]) for row in rows], int(args.seed) + 33
        )
        summary_rows.append(
            {"metric": metric, "mean": mean, "ci95_lower": lower, "ci95_upper": upper, "n_scenes": len(rows)}
        )
    summary_path = output_dir / f"m1_{args.split}{mode_suffix}_seed{saved['seed']}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metadata_path = output_dir / f"m1_{args.split}{mode_suffix}_seed{saved['seed']}_metadata.json"
    metadata = {
        "checkpoint": str(args.checkpoint),
        "manifest_sha256": manifest["manifest_sha256"],
        "split": args.split,
        "observation_mode": args.observation_mode,
        "support_mode": support_mode,
        "scenes": len(rows),
        "posterior_members": int(args.posterior_members),
        "sampling_steps": int(args.sampling_steps),
        "guidance_strength": float(args.guidance_strength),
        "ood_controller_file": str(args.ood_controller_file),
        "ood_disabled_for_matched_ablation": bool(args.disable_ood),
        "ood_controller_method": (
            "disabled-for-matched-support-ablation"
            if bool(args.disable_ood)
            else "validation-calibrated-dual-max"
            if dual_ood_controller is not None
            else "training-observation-mahalanobis"
        ),
        "skip_event_metrics": bool(args.skip_event_metrics),
        "bias_anomaly_decomposition": bool(
            saved.get("variant", {}).get("bias_anomaly_decomposition", True)
        ),
        "event_decision_thresholds": [
            float(value)
            for value in getattr(
                event_head,
                "decision_thresholds",
                [0.50] * len(config["evaluation"]["high_eic_thresholds"]),
            )
        ],
        "conformal_file": str(args.conformal_file),
        "conformal_quantile": conformal_quantile,
        "detail": str(detail_path),
        "summary": str(summary_path),
        "engineering_response_audit": bool(args.engineering_response_audit),
        "engineering_response_file": (
            str(response_path) if bool(args.engineering_response_audit) else ""
        ),
    }
    if bool(args.fit_conformal):
        calibrator, calibration_scenes, calibration_blocks = _fit_cached_spatial_conformal(
            conformal_cache_dir,
            requested_records,
            level=float(args.conformal_level),
            within_block_quantile=float(args.conformal_within_block_quantile),
            std_floor=float(args.conformal_std_floor),
        )
        conformal_path = output_dir / f"m1_validation_seed{saved['seed']}_spatial_conformal.json"
        conformal_payload = {
            "method": "validation-only-spatial-block-conformal",
            "checkpoint": str(args.checkpoint),
            "checkpoint_seed": int(saved["seed"]),
            "manifest_sha256": manifest["manifest_sha256"],
            "fit_split": "validation",
            "fit_observation_mode": "all",
            "level": float(args.conformal_level),
            "within_block_quantile": float(args.conformal_within_block_quantile),
            "std_floor": float(args.conformal_std_floor),
            "block_shape_voxels": [int(value) for value in args.conformal_block_shape],
            "calibration_scenes": int(calibration_scenes),
            "calibration_blocks": int(calibration_blocks),
            "global_quantile": float(calibrator.global_quantile),
            "cache_dir": str(conformal_cache_dir),
        }
        conformal_path.write_text(
            json.dumps(conformal_payload, indent=2), encoding="utf-8"
        )
        metadata["fitted_conformal_file"] = str(conformal_path)
        metadata["fitted_conformal_quantile"] = float(calibrator.global_quantile)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--posterior-members", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--guidance-strength", type=float, default=2.0)
    parser.add_argument("--guidance-batch-size", type=int, default=8)
    parser.add_argument("--rf-trees", type=int, default=None)
    parser.add_argument("--seed", type=int, default=510)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--support-mode",
        choices=("", "support-aware", "nearest-voxel"),
        default="",
        help="Defaults to the support mode recorded in the checkpoint.",
    )
    parser.add_argument("--example-scene-id", default="")
    parser.add_argument("--example-output", default="")
    parser.add_argument("--ood-reference-scenes", type=int, default=100)
    parser.add_argument("--ood-controller-file", default="")
    parser.add_argument(
        "--disable-ood",
        action="store_true",
        help="Disable attenuation/fallback only for a matched component ablation.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conformal-file", default="")
    parser.add_argument("--fit-conformal", action="store_true")
    parser.add_argument("--skip-event-metrics", action="store_true")
    parser.add_argument(
        "--event-source-ensemble",
        action="store_true",
        help="Derive high-EIC probabilities from ensemble exceedance for matched ablations.",
    )
    parser.add_argument("--engineering-response-audit", action="store_true")
    parser.add_argument(
        "--engineering-response-depths",
        type=float,
        nargs="+",
        default=DEFAULT_THAW_DEPTHS_M,
    )
    parser.add_argument(
        "--engineering-response-threshold", type=float, default=0.30
    )
    parser.add_argument(
        "--engineering-response-probability", type=float, default=0.50
    )
    parser.add_argument("--conformal-level", type=float, default=0.90)
    parser.add_argument("--conformal-within-block-quantile", type=float, default=0.90)
    parser.add_argument("--conformal-std-floor", type=float, default=0.001)
    parser.add_argument(
        "--conformal-block-shape",
        type=int,
        nargs=3,
        default=(8, 8, 6),
        metavar=("BX", "BY", "BZ"),
    )
    parser.add_argument(
        "--observation-mode",
        default="all",
        choices=[
            "all",
            "no_ert",
            "no_nmr",
            "no_temperature",
            "boreholes_only",
            "half_boreholes",
            "sparse_boreholes",
        ],
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
