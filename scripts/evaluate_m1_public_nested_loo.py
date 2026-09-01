from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cold_recon.data.data_schema import OBS_TYPES, ObservationTable, load_sample_npz
from cold_recon.data.field_sample_builder import build_public_field_sample
from cold_recon.data.public_support_adapter import load_public_support_observations
from cold_recon.evaluation.block_conformal import (
    BlockConformalCalibrator,
    posterior_diagnostics,
)
from cold_recon.evaluation.ood_control import (
    MahalanobisOODController,
    observation_ood_features,
)
from cold_recon.evaluation.public_m1_adapter import (
    MaskedBoreholeAdapterCase,
    fit_masked_borehole_site_adapter,
)
from cold_recon.models.likelihood_guidance import _torch_sparse
from cold_recon.models.m1_sampling import sample_support_guided_ensemble
from cold_recon.training.factorized_volume_codec import (
    bounded_recenter_samples,
    factorized_ensemble_to_posterior,
    tensor_to_factorized_fields,
)
from cold_recon.operators.support import build_observation_operator
from scripts.build_tree_prior_residual_posterior import tree_prior_fields
from scripts.evaluate_m1_controlled import load_bundle
from scripts.train_m1_support_guided_flow import (
    _context_raster,
    _prior_tensor,
    _manifest_records,
    _subsample_tokens,
    _token_tensor,
)


MODEL_EIC_CEILING = 0.90


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_rf_trees(config: dict, override: int | None) -> int:
    value = int(
        override if override is not None else config["m1_training"]["rf_trees"]
    )
    if value <= 0:
        raise ValueError("rf_trees must be positive")
    return value


def _copy_with_values(observations: ObservationTable, fill: float) -> ObservationTable:
    out = observations.subset(np.arange(observations.n_obs))
    out.values[:] = float(fill)
    return out


def _evenly_spaced_groups(groups: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(groups)
    if int(maximum) <= 0 or len(values) <= int(maximum):
        return values
    positions = np.linspace(0, len(values) - 1, int(maximum)).round().astype(np.int64)
    return values[np.unique(positions)]


def _build_public_case(
    *,
    full_geometry: ObservationTable,
    conditioning: ObservationTable,
    held: ObservationTable,
    config: dict,
    autoencoder,
    model,
    device: torch.device,
    site_id: str,
    rf_trees: int,
    seed: int,
) -> tuple[MaskedBoreholeAdapterCase, dict]:
    fill = float(np.mean(conditioning.values)) if conditioning.n_obs else 0.05
    sample = build_public_field_sample(
        _copy_with_values(full_geometry, fill), config, site_id=site_id
    )
    sample["observations"] = conditioning
    prior = tree_prior_fields(
        sample,
        n_facies=int(config["model"]["n_facies"]),
        seed=int(seed),
        rf_trees=int(rf_trees),
    )
    prior_tensor = _prior_tensor(prior).to(device)
    with torch.no_grad():
        anchor = autoencoder.encode(prior_tensor)
    raster = _context_raster(sample, prior_tensor.cpu(), conditioning).to(device)
    token_observations = _subsample_tokens(
        conditioning,
        int(config["model"]["max_condition_tokens"]),
        np.random.default_rng(int(seed)),
    )
    tokens = _token_tensor(token_observations, sample, config, device)
    with torch.no_grad():
        encoded = model.encode_context(raster, tokens, target_shape=anchor.shape[-3:])
    operator = build_observation_operator(held, sample["grid"])
    support = _torch_sparse(operator.matrix, device, torch.float32)
    case = MaskedBoreholeAdapterCase(
        anchor=anchor,
        encoded_context=encoded.detach(),
        support_operator=support,
        observed_eic=torch.as_tensor(held.values, device=device, dtype=torch.float32),
        sigma=torch.as_tensor(held.sigma, device=device, dtype=torch.float32),
        held_group_id=int(held.group_ids[0]),
    )
    return case, {
        "sample": sample,
        "prior": prior,
        "prior_tensor": prior_tensor,
        "anchor": anchor,
        "raster": raster,
        "tokens": tokens,
        "held_operator": operator,
    }


@torch.no_grad()
def _case_mean_std(
    case,
    autoencoder,
    bias_head,
    adapter=None,
    gate_multiplier: float = 1.0,
):
    encoded = adapter(case.encoded_context) if adapter is not None else case.encoded_context
    bias, gate, scale = bias_head(encoded)
    mean_latent = case.anchor + float(gate_multiplier) * gate * bias
    mean_decoded = autoencoder.decode(mean_latent)
    channel = 10
    mean_decoded[:, channel] = mean_decoded[:, channel].clamp(0.0, 0.90)
    mean = torch.sparse.mm(case.support_operator, mean_decoded[:, channel].reshape(1, -1).T).T[0]
    plus = autoencoder.decode(mean_latent + scale)
    minus = autoencoder.decode(mean_latent - scale)
    plus[:, channel] = plus[:, channel].clamp(0.0, 0.90)
    minus[:, channel] = minus[:, channel].clamp(0.0, 0.90)
    voxel_std = 0.5 * torch.abs(plus[:, channel] - minus[:, channel])
    squared_operator = torch.sparse_coo_tensor(
        case.support_operator.indices(),
        case.support_operator.values().square(),
        case.support_operator.shape,
        device=case.support_operator.device,
        check_invariants=False,
    ).coalesce()
    variance = torch.sparse.mm(squared_operator, voxel_std.reshape(1, -1).square().T).T[0]
    return mean.float().cpu().numpy(), torch.sqrt(variance.clamp_min(1.0e-6)).float().cpu().numpy()


def _paired_noninferiority(
    candidate_rmse: list[float],
    anchor_rmse: list[float],
    margin: float,
    seed: int,
    bootstrap_samples: int = 5000,
) -> dict[str, float | bool]:
    candidate = np.asarray(candidate_rmse, dtype=np.float64)
    anchor = np.asarray(anchor_rmse, dtype=np.float64)
    if candidate.shape != anchor.shape:
        raise ValueError("paired non-inferiority arrays must have identical shape")
    differences = candidate - anchor
    if candidate.size < 2:
        return {
            "mean_difference": float(differences.mean()) if candidate.size else float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "margin": float(margin),
            "pass": False,
        }
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, len(differences), size=(int(bootstrap_samples), len(differences))
    )
    boot = differences[indices].mean(axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "margin": float(margin),
        "pass": bool(upper <= float(margin)),
    }


@torch.no_grad()
def _deployment_mean_fields(
    case,
    autoencoder,
    bias_head,
    *,
    adapter=None,
    gate_multiplier: float = 1.0,
) -> dict[str, np.ndarray]:
    encoded = adapter(case.encoded_context) if adapter is not None else case.encoded_context
    bias, gate, _ = bias_head(encoded)
    decoded = autoencoder.decode(
        case.anchor + float(gate_multiplier) * gate * bias
    )
    offset = 10
    decoded[:, offset] = decoded[:, offset].clamp(0.0, 0.90)
    decoded[:, offset + 1] = decoded[:, offset + 1].clamp(-1.2, 0.4)
    decoded[:, offset + 2] = decoded[:, offset + 2].clamp(0.0, 0.85)
    decoded[:, offset + 3] = decoded[:, offset + 3].clamp(0.0, 1.5)
    return tensor_to_factorized_fields(decoded)


def _recenter_continuous_posterior(
    posterior: dict[str, np.ndarray],
    target_means: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    bounds = {
        "eic": (0.0, 0.90),
        "temperature": (-12.0, 4.0),
        "unfrozen_water": (0.0, 0.85),
        "log_resistivity": (0.0, 15.0),
    }
    for name in ("eic", "temperature", "unfrozen_water", "log_resistivity"):
        samples = np.asarray(posterior[f"{name}_samples"], dtype=np.float32)
        target = np.asarray(target_means[name], dtype=np.float32)
        lower, upper = bounds[name]
        posterior[f"{name}_samples"] = bounded_recenter_samples(
            samples, target, lower, upper
        )
        posterior[f"{name}_mean"] = posterior[f"{name}_samples"].mean(axis=0).astype(
            np.float32
        )
        posterior[f"{name}_std"] = posterior[f"{name}_samples"].std(axis=0).astype(np.float32)
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


def _point_strata(
    held: ObservationTable,
    conditioning: ObservationTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(held.coords[:, 2], dtype=np.float64)
    depth_label = np.select(
        [depth < 2.0, depth < 4.0, depth < 8.0],
        ["depth_0_2", "depth_2_4", "depth_4_8"],
        default="depth_8_plus",
    )
    conditioning_xy = np.asarray(conditioning.coords[conditioning.mask, :2], dtype=np.float64)
    held_xy = np.asarray(held.coords[:, :2], dtype=np.float64)
    if len(conditioning_xy):
        distance = np.sqrt(
            np.sum((held_xy[:, None, :] - conditioning_xy[None, :, :]) ** 2, axis=2)
        ).min(axis=1)
    else:
        distance = np.full(len(held_xy), np.inf, dtype=np.float64)
    distance_label = np.select(
        [distance < 10.0, distance < 25.0],
        ["distance_0_10", "distance_10_25"],
        default="distance_25_plus",
    )
    strata = np.asarray(
        [f"{depth_name}|{distance_name}" for depth_name, distance_name in zip(depth_label, distance_label)],
        dtype=object,
    )
    return strata, depth, distance


def _fit_public_ood_controller(
    config: dict,
    reference_scenes: int,
) -> MahalanobisOODController:
    manifest_path = Path(config["m1_training"]["manifest"])
    root, train_records, _ = _manifest_records(manifest_path, "train")
    _, validation_records, _ = _manifest_records(manifest_path, "validation")

    def extract(records: list[dict]) -> np.ndarray:
        features: list[np.ndarray] = []
        for record in records:
            sample = load_sample_npz(root / record["relative_path"])
            observations = sample["observations"]
            selected = np.flatnonzero(
                observations.mask
                & (observations.type_ids == OBS_TYPES["borehole_eic"])
            )
            field_like = observations.subset(selected)
            features.append(
                observation_ood_features(field_like, sample["grid"])
            )
        return np.stack(features, axis=0)

    controller = MahalanobisOODController(
        abstention_quantile=float(config["evaluation"]["ood_abstention_quantile"])
    ).fit(extract(train_records[: int(reference_scenes)]))
    return controller.calibrate_reference_distances(extract(validation_records))


def _bootstrap_ci(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    boot = array[rng.integers(0, len(array), size=(2000, len(array)))].mean(axis=1)
    return float(array.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", default="data/processed/usgs_eic_observations.npz")
    parser.add_argument("--site-id", default="usgs_eic")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument(
        "--rf-trees",
        type=int,
        default=None,
        help="Tree count for the public anchor; defaults to the checkpoint training config.",
    )
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--adapter-boreholes", type=int, default=16)
    parser.add_argument("--calibration-boreholes", type=int, default=6)
    parser.add_argument("--posterior-members", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=5)
    parser.add_argument("--guidance-strength", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=730)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/m1_support_guided/tables/public_nested_loo.csv")
    parser.add_argument("--ood-reference-scenes", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    saved, config, autoencoder, model, bias_head, event_head = load_bundle(Path(args.checkpoint), device)
    rf_trees = _resolve_rf_trees(config, args.rf_trees)
    public, metadata = load_public_support_observations(args.observations)
    selected = np.flatnonzero(
        public.mask
        & (public.type_ids == OBS_TYPES["borehole_eic"])
        & (public.group_ids >= 0)
    )
    public = public.subset(selected)
    ood_controller = _fit_public_ood_controller(config, int(args.ood_reference_scenes))
    groups = np.unique(public.group_ids)
    groups_available = len(groups)
    if int(args.max_folds) > 0:
        groups = _evenly_spaced_groups(groups, int(args.max_folds))
    selected_groups = np.asarray(groups)
    group_positions = {
        int(group): index for index, group in enumerate(selected_groups)
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if bool(args.resume) and output.exists():
        existing = pd.read_csv(output)
        if "rf_trees" not in existing or not np.all(
            existing["rf_trees"].astype(int) == rf_trees
        ):
            raise ValueError(
                "Cannot resume public evaluation with missing or different rf_trees provenance"
            )
        rows = existing.to_dict(orient="records")
        completed_groups = set(existing["held_group_id"].astype(int))
        groups = np.asarray(
            [group for group in selected_groups if int(group) not in completed_groups]
        )
    margin = float(config["evaluation"]["noninferiority_margin_eic_rmse"])
    point_output = output.with_name(output.stem + "_points.csv")
    for remaining_index, outer_group in enumerate(groups):
        fold_index = int(group_positions[int(outer_group)])
        development = public.subset(np.flatnonzero(public.group_ids != outer_group))
        held = public.subset(np.flatnonzero(public.group_ids == outer_group))
        development_groups = np.unique(development.group_ids)
        calibration_groups = _evenly_spaced_groups(
            development_groups,
            max(2, int(args.calibration_boreholes)),
        )
        adapter_groups = _evenly_spaced_groups(
            np.setdiff1d(development_groups, calibration_groups),
            int(args.adapter_boreholes),
        )
        neutral = build_public_field_sample(
            _copy_with_values(public, float(np.mean(development.values))),
            config,
            site_id=args.site_id,
        )
        fold_ood_control = ood_controller.control(
            observation_ood_features(development, neutral["grid"])[None, :]
        )
        fold_gate_multiplier = float(fold_ood_control["bias_gate_multiplier"][0])
        adapter_cases: list[MaskedBoreholeAdapterCase] = []
        for inner_group in adapter_groups:
            condition = development.subset(
                np.flatnonzero(
                    np.isin(development.group_ids, adapter_groups)
                    & (development.group_ids != inner_group)
                )
            )
            inner_held = development.subset(np.flatnonzero(development.group_ids == inner_group))
            case, _ = _build_public_case(
                full_geometry=public,
                conditioning=condition,
                held=inner_held,
                config=config,
                autoencoder=autoencoder,
                model=model,
                device=device,
                site_id=args.site_id,
                rf_trees=rf_trees,
                seed=int(args.seed) + fold_index * 100 + int(inner_group),
            )
            adapter_cases.append(case)
        adapter, adapter_history = fit_masked_borehole_site_adapter(
            adapter_cases,
            autoencoder=autoencoder,
            bias_head=bias_head,
            context_channels=int(config["model"]["context_channels"]),
            steps=int(args.adapter_steps),
            seed=int(args.seed) + fold_index,
        )

        calibration_truth = []
        calibration_candidate = []
        calibration_anchor = []
        calibration_std = []
        calibration_blocks = []
        calibration_strata = []
        calibration_payloads: list[tuple[MaskedBoreholeAdapterCase, dict, ObservationTable]] = []
        calibration_candidate_group_rmse: list[float] = []
        calibration_anchor_group_rmse: list[float] = []
        for calibration_group in calibration_groups:
            condition = development.subset(
                np.flatnonzero(development.group_ids != calibration_group)
            )
            calibration_held = development.subset(
                np.flatnonzero(development.group_ids == calibration_group)
            )
            case, calibration_bundle = _build_public_case(
                full_geometry=public,
                conditioning=condition,
                held=calibration_held,
                config=config,
                autoencoder=autoencoder,
                model=model,
                device=device,
                site_id=args.site_id,
                rf_trees=rf_trees,
                seed=int(args.seed) + fold_index * 1000 + int(calibration_group),
            )
            candidate_mean, _ = _case_mean_std(
                case,
                autoencoder,
                bias_head,
                adapter=adapter,
                gate_multiplier=fold_gate_multiplier,
            )
            anchor_mean = calibration_bundle["held_operator"].apply(
                calibration_bundle["prior"]["eic"]
            )
            calibration_candidate_group_rmse.append(
                float(np.sqrt(np.mean((candidate_mean - calibration_held.values) ** 2)))
            )
            calibration_anchor_group_rmse.append(
                float(np.sqrt(np.mean((anchor_mean - calibration_held.values) ** 2)))
            )
            calibration_truth.append(calibration_held.values)
            calibration_candidate.append(candidate_mean)
            calibration_anchor.append(anchor_mean)
            calibration_blocks.append(
                np.full(calibration_held.n_obs, int(calibration_group), dtype=np.int64)
            )
            calibration_strata.append(
                _point_strata(calibration_held, condition)[0]
            )
            calibration_payloads.append(
                (case, calibration_bundle, calibration_held)
            )
        cal_truth = np.concatenate(calibration_truth)
        cal_candidate = np.concatenate(calibration_candidate)
        cal_anchor = np.concatenate(calibration_anchor)
        anchor_rmse = float(np.sqrt(np.mean((cal_anchor - cal_truth) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((cal_candidate - cal_truth) ** 2)))
        noninferiority = _paired_noninferiority(
            calibration_candidate_group_rmse,
            calibration_anchor_group_rmse,
            margin,
            seed=int(args.seed) + 50_000 + fold_index,
        )
        allow_bias = bool(noninferiority["pass"]) and not bool(
            fold_ood_control["abstain"][0]
        )
        for calibration_index, (
            calibration_case,
            calibration_bundle,
            calibration_held,
        ) in enumerate(calibration_payloads):
            calibration_decoded, _ = sample_support_guided_ensemble(
                model=model,
                bias_head=bias_head,
                event_head=event_head,
                autoencoder=autoencoder,
                anchor=calibration_bundle["anchor"],
                raster=calibration_bundle["raster"],
                tokens=calibration_bundle["tokens"],
                sample=calibration_bundle["sample"],
                site_adapter=adapter if allow_bias else None,
                ood_gate_multiplier=(
                    fold_gate_multiplier if allow_bias else 0.0
                ),
                n_members=int(args.posterior_members),
                sampling_steps=int(args.sampling_steps),
                guidance_strength=float(args.guidance_strength),
                seed=int(args.seed)
                + 200_000
                + fold_index * 100
                + calibration_index,
            )
            calibration_posterior = factorized_ensemble_to_posterior(
                calibration_decoded
            )
            calibration_members = np.stack(
                [
                    calibration_bundle["held_operator"].apply(member)
                    for member in calibration_posterior["eic_samples"]
                ],
                axis=0,
            )
            if calibration_members.shape[1] != calibration_held.n_obs:
                raise RuntimeError(
                    "calibration support ensemble does not match held observations"
                )
            calibration_std.append(
                calibration_members.std(axis=0).clip(min=1.0e-3)
            )
        cal_std = np.concatenate(calibration_std)
        calibration_mean = cal_candidate if allow_bias else cal_anchor
        calibrator = BlockConformalCalibrator(level=0.90, min_blocks_per_stratum=2).fit(
            cal_truth,
            calibration_mean,
            cal_std,
            np.concatenate(calibration_blocks),
            strata=np.concatenate(calibration_strata),
        )

        outer_case, bundle = _build_public_case(
            full_geometry=public,
            conditioning=development,
            held=held,
            config=config,
            autoencoder=autoencoder,
            model=model,
            device=device,
            site_id=args.site_id,
            rf_trees=rf_trees,
            seed=int(args.seed) + 10_000 + fold_index,
        )
        # Retain an outer audit of the learned correction before either the
        # non-inferiority decision or the OOD gate is applied. This diagnostic
        # never feeds back into model selection or deployment.
        ungated_candidate_mean, _ = _case_mean_std(
            outer_case,
            autoencoder,
            bias_head,
            adapter=adapter,
            gate_multiplier=1.0,
        )
        decoded, _ = sample_support_guided_ensemble(
            model=model,
            bias_head=bias_head,
            event_head=event_head,
            autoencoder=autoencoder,
            anchor=bundle["anchor"],
            raster=bundle["raster"],
            tokens=bundle["tokens"],
            sample=bundle["sample"],
            site_adapter=adapter if allow_bias else None,
            ood_gate_multiplier=fold_gate_multiplier if allow_bias else 0.0,
            n_members=int(args.posterior_members),
            sampling_steps=int(args.sampling_steps),
            guidance_strength=float(args.guidance_strength),
            seed=int(args.seed) + fold_index,
        )
        posterior = factorized_ensemble_to_posterior(decoded)
        if allow_bias:
            deployment_means = _deployment_mean_fields(
                outer_case,
                autoencoder,
                bias_head,
                adapter=adapter,
                gate_multiplier=fold_gate_multiplier,
            )
        else:
            deployment_means = {
                "eic": np.asarray(bundle["prior"]["eic"], dtype=np.float32),
                "temperature": np.asarray(bundle["prior"]["temperature"], dtype=np.float32),
                "unfrozen_water": np.asarray(bundle["prior"]["unfrozen_water"], dtype=np.float32),
                "log_resistivity": np.asarray(bundle["prior"]["log_resistivity"], dtype=np.float32),
            }
        posterior = _recenter_continuous_posterior(posterior, deployment_means)
        operator = bundle["held_operator"]
        held_mean = operator.apply(posterior["eic_mean"])
        held_members = np.stack(
            [operator.apply(member) for member in posterior["eic_samples"]], axis=0
        )
        held_std = held_members.std(axis=0)
        raw_lower = np.quantile(held_members, 0.05, axis=0)
        raw_upper = np.quantile(held_members, 0.95, axis=0)
        outer_strata, held_depth, held_distance = _point_strata(held, development)
        interval_inflation = float(fold_ood_control["interval_inflation"][0])
        calibrated_lower_unbounded, calibrated_upper_unbounded = calibrator.interval(
            held_mean,
            held_std,
            strata=outer_strata,
            inflation=interval_inflation,
        )
        # Public truth is retained on [0, 1]. The trained model has a narrower
        # declared output ceiling, so projected and unbounded interval coverage
        # are both retained for an explicit range-contract audit.
        calibrated_lower = np.clip(
            calibrated_lower_unbounded, 0.0, MODEL_EIC_CEILING
        )
        calibrated_upper = np.clip(
            calibrated_upper_unbounded, 0.0, MODEL_EIC_CEILING
        )
        truth = held.values
        raw_diagnostics = posterior_diagnostics(held_members, truth)
        calibrated_scale = (
            calibrator.quantile_for(outer_strata, held_mean.shape) * interval_inflation
        )
        calibrated_members = np.clip(
            held_mean[None, :]
            + calibrated_scale[None, :] * (held_members - held_mean[None, :]),
            0.0,
            MODEL_EIC_CEILING,
        )
        calibrated_diagnostics = posterior_diagnostics(calibrated_members, truth)
        outer_anchor_mean = operator.apply(bundle["prior"]["eic"])
        rows.append(
            {
                "held_group_id": int(outer_group),
                "model_seed": int(saved["seed"]),
                "rf_trees": rf_trees,
                "n_held": held.n_obs,
                "allow_bias": bool(allow_bias),
                "inner_anchor_rmse": anchor_rmse,
                "inner_candidate_rmse": candidate_rmse,
                "noninferiority_difference": noninferiority["mean_difference"],
                "noninferiority_ci95_lower": noninferiority["ci95_lower"],
                "noninferiority_ci95_upper": noninferiority["ci95_upper"],
                "inner_noninferiority_pass": bool(noninferiority["pass"]),
                "fallback_due_to_noninferiority": not bool(
                    noninferiority["pass"]
                ),
                "fallback_due_to_ood": bool(fold_ood_control["abstain"][0]),
                "exact_anchor_fallback_applied": not bool(allow_bias),
                "outer_anchor_rmse": float(
                    np.sqrt(np.mean((outer_anchor_mean - truth) ** 2))
                ),
                "outer_ungated_candidate_rmse": float(
                    np.sqrt(np.mean((ungated_candidate_mean - truth) ** 2))
                ),
                "outer_ungated_candidate_mae": float(
                    np.mean(np.abs(ungated_candidate_mean - truth))
                ),
                "outer_ungated_candidate_difference_vs_anchor": float(
                    np.sqrt(np.mean((ungated_candidate_mean - truth) ** 2))
                    - np.sqrt(np.mean((outer_anchor_mean - truth) ** 2))
                ),
                "outer_rmse": float(np.sqrt(np.mean((held_mean - truth) ** 2))),
                "outer_mae": float(np.mean(np.abs(held_mean - truth))),
                "raw_coverage_90": float(np.mean((truth >= raw_lower) & (truth <= raw_upper))),
                "raw_width_90": float(np.mean(raw_upper - raw_lower)),
                "calibrated_coverage_90": float(
                    np.mean((truth >= calibrated_lower) & (truth <= calibrated_upper))
                ),
                "calibrated_coverage_90_unbounded_audit": float(
                    np.mean(
                        (truth >= calibrated_lower_unbounded)
                        & (truth <= calibrated_upper_unbounded)
                    )
                ),
                "calibrated_width_90": float(
                    np.mean(calibrated_upper - calibrated_lower)
                ),
                "calibrated_width_90_unbounded_audit": float(
                    np.mean(calibrated_upper_unbounded - calibrated_lower_unbounded)
                ),
                "raw_crps": raw_diagnostics["crps"],
                "raw_pit_mean": raw_diagnostics["pit_mean"],
                "raw_pit_variance": raw_diagnostics["pit_variance"],
                "calibrated_crps": calibrated_diagnostics["crps"],
                "calibrated_pit_mean": calibrated_diagnostics["pit_mean"],
                "calibrated_pit_variance": calibrated_diagnostics["pit_variance"],
                "conformal_quantile": float(calibrator.global_quantile),
                "ood_score": float(fold_ood_control["ood_score"][0]),
                "ood_risk": float(fold_ood_control["ood_risk"][0]),
                "ood_gate_multiplier": fold_gate_multiplier,
                "ood_interval_inflation": interval_inflation,
                "ood_abstain": bool(fold_ood_control["abstain"][0]),
                "adapter_final_loss": float(adapter_history[-1]["loss"]),
                "model_eic_ceiling": MODEL_EIC_CEILING,
                "truth_above_model_eic_ceiling_count": int(
                    np.sum(truth > MODEL_EIC_CEILING)
                ),
            }
        )
        point_frame = pd.DataFrame(
            {
                "held_group_id": int(outer_group),
                "depth_m": held_depth,
                "distance_to_conditioning_m": held_distance,
                "stratum": outer_strata,
                "truth_eic": truth,
                "posterior_mean_eic": held_mean,
                "anchor_mean_eic": outer_anchor_mean,
                "ungated_candidate_mean_eic": ungated_candidate_mean,
                "raw_lower_90": raw_lower,
                "raw_upper_90": raw_upper,
                "calibrated_lower_90": calibrated_lower,
                "calibrated_upper_90": calibrated_upper,
                "calibrated_lower_90_unbounded_audit": calibrated_lower_unbounded,
                "calibrated_upper_90_unbounded_audit": calibrated_upper_unbounded,
                "ood_score": float(fold_ood_control["ood_score"][0]),
                "allow_bias": bool(allow_bias),
            }
        )
        point_output.parent.mkdir(parents=True, exist_ok=True)
        point_frame.to_csv(
            point_output,
            mode="a" if point_output.exists() else "w",
            header=not point_output.exists(),
            index=False,
        )
        pd.DataFrame(rows).to_csv(output, index=False)
        print(
            f"{remaining_index + 1}/{len(groups)} held={outer_group} rmse={rows[-1]['outer_rmse']:.4f} "
            f"allow_bias={allow_bias} coverage={rows[-1]['calibrated_coverage_90']:.3f}"
        )

    pd.DataFrame(rows).to_csv(output, index=False)
    deployment_noninferiority = _paired_noninferiority(
        [row["outer_rmse"] for row in rows],
        [row["outer_anchor_rmse"] for row in rows],
        margin,
        seed=int(args.seed) + 90_000,
    )
    ungated_candidate_noninferiority = _paired_noninferiority(
        [row["outer_ungated_candidate_rmse"] for row in rows],
        [row["outer_anchor_rmse"] for row in rows],
        margin,
        seed=int(args.seed) + 91_000,
    )
    summary = {
        "dataset": args.site_id,
        "source": args.observations,
        "source_sha256": _sha256(args.observations),
        "model_seed": int(saved["seed"]),
        "rf_trees": rf_trees,
        "model_eic_ceiling": MODEL_EIC_CEILING,
        "folds": len(rows),
        "boreholes_eligible": groups_available,
        "boreholes_available": metadata["n_boreholes"],
        "bare_borehole_labels_available": metadata.get(
            "n_bare_borehole_labels", metadata["n_boreholes"]
        ),
        "group_key_definition": metadata.get("group_key_definition", "unknown"),
        "observations_evaluated": int(public.n_obs),
        "truth_above_model_eic_ceiling": int(
            np.sum(public.values > MODEL_EIC_CEILING)
        ),
        "rmse_mean_ci": _bootstrap_ci([row["outer_rmse"] for row in rows], int(args.seed)),
        "raw_coverage_mean_ci": _bootstrap_ci(
            [row["raw_coverage_90"] for row in rows], int(args.seed) + 1
        ),
        "calibrated_coverage_mean_ci": _bootstrap_ci(
            [row["calibrated_coverage_90"] for row in rows], int(args.seed) + 2
        ),
        "bias_allowed_fraction": float(np.mean([row["allow_bias"] for row in rows])),
        "exact_anchor_fallback_fraction": float(
            np.mean([row["exact_anchor_fallback_applied"] for row in rows])
        ),
        "ood_abstention_fraction": float(
            np.mean([row["ood_abstain"] for row in rows])
        ),
        "noninferiority_margin": margin,
        "deployment_noninferiority": deployment_noninferiority,
        "ungated_candidate_noninferiority_audit": ungated_candidate_noninferiority,
        "ungated_candidate_improvement_fraction": float(
            np.mean(
                [
                    row["outer_ungated_candidate_rmse"]
                    < row["outer_anchor_rmse"]
                    for row in rows
                ]
            )
        ),
        "noninferiority_pass_fraction": float(
            np.mean([row["inner_noninferiority_pass"] for row in rows])
        ),
        "ood_controller_method": (
            "borehole-EIC observation features; fit on controlled train and "
            "score-calibrated on controlled validation"
        ),
        "ood_reference_scenes": int(args.ood_reference_scenes),
        "ood_calibration_scenes": 100,
        "output": str(output),
        "point_output": str(output.with_name(output.stem + "_points.csv")),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
