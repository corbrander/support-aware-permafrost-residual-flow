from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from cold_recon.baselines.idw import reconstruct_idw
from cold_recon.data.data_schema import OBS_TYPES, ObservationTable, load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.evaluation.observation_consistency import nearest_grid_indices
from cold_recon.evaluation.uncertainty import facies_entropy
from cold_recon.models.denoiser3d_unet import Denoiser3DUNet
from cold_recon.models.diffusion import GaussianDiffusion3D
from cold_recon.models.observation_tokenizer import ObservationTokenizer, build_observation_attention_mask
from cold_recon.models.observation_transformer import ObsTransformerEncoder
from cold_recon.synthetic.active_boreholes import active_borehole_config_from_dict, augment_sample_with_active_boreholes
from cold_recon.training.train_diffusion import _load_autoencoder, _posterior_arrays
from cold_recon.training.volume_codec import fields_to_volume_tensor, sample_to_volume_tensor


@dataclass
class DiffusionCase:
    path: Path
    sample: dict[str, Any]
    latent: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor
    attention_mask: torch.Tensor | None


def synthetic_sample_paths(config: dict, n_samples: int | None = None) -> list[Path]:
    data_dir = Path(config["paths"]["synthetic_dir"])
    paths = sorted(data_dir.glob("sample_*.npz"))
    if n_samples is not None:
        paths = paths[: int(n_samples)]
    if len(paths) < 2:
        raise ValueError("Multi-sample diffusion requires at least two synthetic samples")
    return paths


def subsample_observations_by_type(observations: ObservationTable, max_tokens: int | None, seed: int = 42) -> ObservationTable:
    if max_tokens is None or max_tokens <= 0 or observations.n_obs <= max_tokens:
        return observations
    rng = np.random.default_rng(seed)
    type_ids = np.unique(observations.type_ids)
    selected: list[int] = []
    leftovers: list[int] = []
    quota = max(1, int(max_tokens) // max(len(type_ids), 1))
    for type_id in type_ids:
        idx = np.where(observations.mask & (observations.type_ids == type_id))[0]
        rng.shuffle(idx)
        take = min(len(idx), quota)
        selected.extend(idx[:take].tolist())
        leftovers.extend(idx[take:].tolist())
    remaining = int(max_tokens) - len(selected)
    if remaining > 0 and leftovers:
        rest = np.asarray(leftovers, dtype=int)
        rng.shuffle(rest)
        selected.extend(rest[:remaining].tolist())
    selected_arr = np.sort(np.asarray(selected[: int(max_tokens)], dtype=int))
    return observations.subset(selected_arr)


def _prepare_case(
    config: dict,
    path: Path,
    ae,
    device: torch.device,
    n_facies: int,
    max_condition_tokens: int | None,
    seed: int,
) -> DiffusionCase:
    sample = load_sample_npz(path)
    sample = augment_sample_with_active_boreholes(
        sample,
        active_borehole_config_from_dict(config),
        n_facies=n_facies,
        seed=seed,
    )
    target = sample_to_volume_tensor(sample, n_facies=n_facies).to(device)
    with torch.no_grad():
        latent = ae.encode(target).detach()
    observations = subsample_observations_by_type(sample["observations"], max_condition_tokens, seed=seed)
    tokenizer = ObservationTokenizer(n_types=9).fit_from_grid(sample["grid"])
    tokens = tokenizer.encode_torch(observations, device=device).unsqueeze(0)
    token_mask = torch.zeros((1, tokens.shape[1]), dtype=torch.bool, device=device)
    attention_mask = build_observation_attention_mask(config, sample["grid"], observations, device=device)
    return DiffusionCase(
        path=path,
        sample=sample,
        latent=latent,
        tokens=tokens,
        token_mask=token_mask,
        attention_mask=attention_mask,
    )


def _proxy_from_observations(sample: dict, n_facies: int, use_base_observations: bool = False) -> dict[str, np.ndarray]:
    observations = sample.get("base_observations", sample["observations"]) if use_base_observations else sample["observations"]
    proxy = reconstruct_idw(observations, sample["grid"], n_facies=n_facies)
    shape = sample["fields"]["eic"].shape
    defaults = {
        "facies": np.zeros(shape, dtype=np.int16),
        "eic": np.full(
            shape,
            float(np.mean(observations.values[observations.mask & (observations.type_ids == 1)]))
            if np.any(observations.mask & (observations.type_ids == 1))
            else 0.05,
            dtype=np.float32,
        ),
        "temperature": np.zeros(shape, dtype=np.float32),
        "unfrozen_water": np.full(shape, 0.08, dtype=np.float32),
        "log_resistivity": np.full(shape, 5.0, dtype=np.float32),
    }
    for key, value in defaults.items():
        if key not in proxy:
            proxy[key] = value
    return proxy


def _posterior_mean_fields(posterior: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "facies": np.asarray(posterior["facies_mode"], dtype=np.int16),
        "eic": np.asarray(posterior["eic_mean"], dtype=np.float32),
        "temperature": np.asarray(posterior["temperature_mean"], dtype=np.float32),
        "unfrozen_water": np.asarray(posterior["unfrozen_water_mean"], dtype=np.float32),
        "log_resistivity": np.asarray(posterior["log_resistivity_mean"], dtype=np.float32),
    }


def _blend_continuous_with_proxy(
    posterior: dict[str, np.ndarray],
    proxy_fields: dict[str, np.ndarray],
    weight: float,
    field_weights: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    default_weight = float(np.clip(weight, 0.0, 1.0))
    field_weights = field_weights or {}
    if default_weight <= 0.0 and not field_weights:
        return posterior
    out = dict(posterior)
    for field in ["eic", "temperature", "unfrozen_water", "log_resistivity"]:
        if field not in proxy_fields:
            continue
        local_weight = float(np.clip(field_weights.get(field, default_weight), 0.0, 1.0))
        if local_weight <= 0.0:
            continue
        sample_key = f"{field}_samples"
        mean_key = f"{field}_mean"
        std_key = f"{field}_std"
        proxy = np.asarray(proxy_fields[field], dtype=np.float32)
        if sample_key in out:
            samples = (local_weight * proxy[None, ...] + (1.0 - local_weight) * out[sample_key]).astype(np.float32)
            out[sample_key] = samples
            out[mean_key] = samples.mean(axis=0).astype(np.float32)
            out[std_key] = samples.std(axis=0).astype(np.float32)
        elif mean_key in out:
            out[mean_key] = (local_weight * proxy + (1.0 - local_weight) * out[mean_key]).astype(np.float32)
    if "log_resistivity_samples" in out:
        res = np.exp(np.clip(out["log_resistivity_samples"], 0.0, 12.0)).astype(np.float32)
        out["resistivity_samples"] = res
        out["resistivity_mean"] = res.mean(axis=0).astype(np.float32)
        out["resistivity_std"] = res.std(axis=0).astype(np.float32)
    elif "log_resistivity_mean" in out:
        out["resistivity_mean"] = np.exp(np.clip(out["log_resistivity_mean"], 0.0, 12.0)).astype(np.float32)
    if "eic_samples" in out:
        out["ice_rich_probability"] = np.mean(out["eic_samples"] > 0.30, axis=0).astype(np.float32)
    out["continuous_proxy_blend_weight"] = np.asarray(default_weight, dtype=np.float32)
    for field, local_weight in field_weights.items():
        out[f"{field}_proxy_blend_weight"] = np.asarray(float(np.clip(local_weight, 0.0, 1.0)), dtype=np.float32)
    return out


def calibrate_ice_rich_eic(
    posterior: dict[str, np.ndarray],
    event_classes: tuple[int, ...] = (3, 6),
    event_threshold: float = 0.50,
    eic_floor: float = 0.31,
) -> dict[str, np.ndarray]:
    if "facies_probability" not in posterior:
        return posterior
    threshold = float(np.clip(event_threshold, 0.0, 1.0))
    floor = float(max(eic_floor, 0.0))
    probs = np.asarray(posterior["facies_probability"], dtype=np.float32)
    valid_classes = [cls for cls in event_classes if 0 <= int(cls) < probs.shape[-1]]
    if not valid_classes:
        return posterior
    event_probability = probs[..., valid_classes].sum(axis=-1)
    confidence = np.clip((event_probability - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    target = 0.30 + (floor - 0.30) * confidence
    event_mask = event_probability >= threshold
    out = dict(posterior)
    if "eic_samples" in out:
        samples = np.asarray(out["eic_samples"], dtype=np.float32).copy()
        samples = np.where(event_mask[None, ...], np.maximum(samples, target[None, ...]), samples)
        out["eic_samples"] = samples.astype(np.float32)
        out["eic_mean"] = samples.mean(axis=0).astype(np.float32)
        out["eic_std"] = samples.std(axis=0).astype(np.float32)
    elif "eic_mean" in out:
        out["eic_mean"] = np.where(event_mask, np.maximum(out["eic_mean"], target), out["eic_mean"]).astype(np.float32)
    if "eic_samples" in out:
        out["ice_rich_probability"] = np.mean(out["eic_samples"] > 0.30, axis=0).astype(np.float32)
    elif "eic_mean" in out:
        out["ice_rich_probability"] = (out["eic_mean"] > 0.30).astype(np.float32)
    out["ice_rich_event_probability"] = event_probability.astype(np.float32)
    out["ice_rich_event_threshold"] = np.asarray(threshold, dtype=np.float32)
    out["ice_rich_eic_floor"] = np.asarray(floor, dtype=np.float32)
    return out


def assimilate_facies_observations(
    posterior: dict[str, np.ndarray],
    observations: ObservationTable,
    grid: dict[str, Any],
    n_facies: int,
) -> dict[str, np.ndarray]:
    facies_mask = observations.type_ids == OBS_TYPES["borehole_facies"]
    if not np.any(facies_mask):
        out = dict(posterior)
        out["facies_observation_assimilation_count"] = np.asarray(0.0, dtype=np.float32)
        out["facies_observation_rare_count"] = np.asarray(0.0, dtype=np.float32)
        return out
    coords = observations.coords[facies_mask]
    values = np.rint(observations.values[facies_mask]).astype(np.int64)
    values = np.clip(values, 0, int(n_facies) - 1)
    ix, iy, iz = nearest_grid_indices(coords, grid)
    out = dict(posterior)
    if "facies_mode" in out:
        mode = np.asarray(out["facies_mode"], dtype=np.int16).copy()
    elif "facies_probability" in out:
        mode = np.argmax(np.asarray(out["facies_probability"], dtype=np.float32), axis=-1).astype(np.int16)
    else:
        return out
    if "facies_probability" in out:
        probs = np.asarray(out["facies_probability"], dtype=np.float32).copy()
    else:
        probs = np.zeros((*mode.shape, int(n_facies)), dtype=np.float32)
        for cls in range(int(n_facies)):
            probs[..., cls] = mode == cls
    samples = np.asarray(out["facies_samples"], dtype=np.int16).copy() if "facies_samples" in out else None
    rare_count = 0
    for x_idx, y_idx, z_idx, cls in zip(ix, iy, iz, values):
        mode[x_idx, y_idx, z_idx] = np.int16(cls)
        probs[x_idx, y_idx, z_idx, :] = 0.0
        probs[x_idx, y_idx, z_idx, cls] = 1.0
        if samples is not None:
            samples[:, x_idx, y_idx, z_idx] = np.int16(cls)
        if cls in (3, 6):
            rare_count += 1
    out["facies_mode"] = mode.astype(np.int16)
    out["facies_probability"] = probs.astype(np.float32)
    out["facies_entropy"] = facies_entropy(probs).astype(np.float32)
    if samples is not None:
        out["facies_samples"] = samples.astype(np.int16)
    out["facies_observation_assimilation_count"] = np.asarray(float(len(values)), dtype=np.float32)
    out["facies_observation_rare_count"] = np.asarray(float(rare_count), dtype=np.float32)
    return out


def assimilate_wedge_ice_morphology(
    posterior: dict[str, np.ndarray],
    observations: ObservationTable,
    grid: dict[str, Any],
    n_facies: int,
    xy_radius_cells: int = 1,
    vertical_buffer_cells: int = 3,
    connect_observed_columns: bool = True,
    max_connection_cells: int = 0,
) -> dict[str, np.ndarray]:
    facies_mask = observations.type_ids == OBS_TYPES["borehole_facies"]
    if not np.any(facies_mask):
        return posterior
    coords = observations.coords[facies_mask]
    values = np.rint(observations.values[facies_mask]).astype(np.int64)
    wedge_coords = coords[values == 6]
    if wedge_coords.size == 0:
        out = dict(posterior)
        out["wedge_morphology_voxels"] = np.asarray(0.0, dtype=np.float32)
        return out
    ix, iy, iz = nearest_grid_indices(wedge_coords, grid)
    shape = np.asarray(posterior["facies_mode"]).shape
    nx, ny, nz = shape
    radius = max(int(xy_radius_cells), 0)
    buffer = max(int(vertical_buffer_cells), 0)
    max_connect = int(max_connection_cells)
    mask = np.zeros(shape, dtype=bool)
    groups: dict[tuple[int, int], list[int]] = {}
    for x_idx, y_idx, z_idx in zip(ix, iy, iz):
        groups.setdefault((int(x_idx), int(y_idx)), []).append(int(z_idx))

    def fill_column(x_idx: int, y_idx: int, lo: int, hi: int) -> None:
        for xx in range(max(0, x_idx - radius), min(nx, x_idx + radius + 1)):
            for yy in range(max(0, y_idx - radius), min(ny, y_idx + radius + 1)):
                if (xx - x_idx) ** 2 + (yy - y_idx) ** 2 <= radius**2:
                    mask[xx, yy, lo : hi + 1] = True

    for (x_idx, y_idx), depths in groups.items():
        lo = max(0, min(depths) - buffer)
        hi = min(nz - 1, max(depths) + buffer)
        fill_column(x_idx, y_idx, lo, hi)
    if connect_observed_columns and len(groups) >= 2:
        group_items = list(groups.items())
        for left in range(len(group_items)):
            for right in range(left + 1, len(group_items)):
                (x0, y0), z0 = group_items[left]
                (x1, y1), z1 = group_items[right]
                distance = max(abs(x1 - x0), abs(y1 - y0))
                if max_connect > 0 and distance > max_connect:
                    continue
                steps = max(distance, 1)
                lo = max(0, min(min(z0), min(z1)) - buffer)
                hi = min(nz - 1, max(max(z0), max(z1)) + buffer)
                for step in range(steps + 1):
                    x_idx = int(round(x0 + (x1 - x0) * step / steps))
                    y_idx = int(round(y0 + (y1 - y0) * step / steps))
                    fill_column(x_idx, y_idx, lo, hi)

    if not np.any(mask):
        return posterior
    out = dict(posterior)
    mode = np.asarray(out["facies_mode"], dtype=np.int16).copy()
    mode[mask] = np.int16(6)
    out["facies_mode"] = mode
    if "facies_probability" in out:
        probs = np.asarray(out["facies_probability"], dtype=np.float32).copy()
    else:
        probs = np.zeros((*shape, int(n_facies)), dtype=np.float32)
        for cls in range(int(n_facies)):
            probs[..., cls] = mode == cls
    probs[mask, :] = 0.0
    probs[mask, 6] = 1.0
    out["facies_probability"] = probs.astype(np.float32)
    out["facies_entropy"] = facies_entropy(probs).astype(np.float32)
    if "facies_samples" in out:
        samples = np.asarray(out["facies_samples"], dtype=np.int16).copy()
        samples[:, mask] = np.int16(6)
        out["facies_samples"] = samples.astype(np.int16)
    out["wedge_morphology_voxels"] = np.asarray(float(np.sum(mask)), dtype=np.float32)
    out["wedge_morphology_columns"] = np.asarray(float(len(groups)), dtype=np.float32)
    return out


def facies_observation_accuracy(pred_facies: np.ndarray, sample: dict[str, Any]) -> float:
    observations = sample["observations"]
    facies_mask = observations.type_ids == OBS_TYPES["borehole_facies"]
    if not np.any(facies_mask):
        return float("nan")
    ix, iy, iz = nearest_grid_indices(observations.coords[facies_mask], sample["grid"])
    target = np.rint(observations.values[facies_mask]).astype(np.int64)
    pred = np.asarray(pred_facies, dtype=np.int64)[ix, iy, iz]
    return float(np.mean(pred == target))


def rare_facies_observation_diagnostics(sample: dict[str, Any]) -> dict[str, float]:
    observations = sample["observations"]
    facies_mask = observations.type_ids == OBS_TYPES["borehole_facies"]
    values = np.rint(observations.values[facies_mask]).astype(np.int64)
    truth = np.asarray(sample["fields"]["facies"], dtype=np.int64)
    active = sample.get("metadata", {}).get("active_borehole_sampling", {})
    return {
        "borehole_facies_n": float(values.size),
        "borehole_ice_rich_facies_n": float(np.sum(np.isin(values, [3, 6]))),
        "borehole_wedge_ice_n": float(np.sum(values == 6)),
        "truth_wedge_ice_fraction": float(np.mean(truth == 6)),
        "active_boreholes_n": float(active.get("n_boreholes_added", 0.0)),
        "active_borehole_observations_n": float(active.get("n_observations_added", 0.0)),
        "active_ice_rich_observations_n": float(active.get("active_ice_rich_observations_n", 0.0)),
        "active_wedge_ice_observations_n": float(active.get("active_wedge_ice_observations_n", 0.0)),
    }


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_multisample_diffusion(
    config: dict,
    sample_paths: list[Path] | None = None,
    holdout_index: int = -1,
    epochs: int | None = None,
    samples: int | None = None,
    max_condition_tokens: int | None = None,
    device: str | None = None,
) -> dict[str, Path | list[dict[str, float | str]]]:
    cfg = config.get("multisample_diffusion", {})
    paths = sample_paths or synthetic_sample_paths(config, n_samples=cfg.get("n_samples"))
    n_facies = int(config["model"]["n_facies"])
    holdout_index = int(holdout_index if holdout_index >= 0 else cfg.get("holdout_index", len(paths) - 1))
    if holdout_index < 0 or holdout_index >= len(paths):
        raise IndexError(f"holdout_index={holdout_index} outside {len(paths)} sample paths")
    seed = int(config.get("project", {}).get("seed", 42))
    max_tokens = int(max_condition_tokens if max_condition_tokens is not None else cfg.get("max_condition_tokens", 512))
    device_name = device or cfg.get("device", config["diffusion"].get("device", config["training"].get("device", "cuda")))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    dev = torch.device(device_name)

    ae = _load_autoencoder(config, dev)
    cases = [
        _prepare_case(config, path, ae, dev, n_facies, max_condition_tokens=max_tokens, seed=seed + idx)
        for idx, path in enumerate(paths)
    ]
    holdout = cases[holdout_index]
    train_cases = [case for idx, case in enumerate(cases) if idx != holdout_index]
    obs_hidden = int(config["model"].get("obs_hidden_dim", 96))
    obs_encoder = ObsTransformerEncoder(
        token_dim=int(config["model"]["token_dim"]),
        hidden_dim=obs_hidden,
        num_layers=int(config["model"].get("obs_layers", 2)),
        num_heads=int(config["model"].get("obs_heads", 4)),
    ).to(dev)
    denoiser = Denoiser3DUNet(
        channels=int(train_cases[0].latent.shape[1]),
        cond_dim=obs_hidden,
        base=int(cfg.get("denoiser_base_channels", config["diffusion"].get("denoiser_base_channels", 32))),
    ).to(dev)
    diffusion = GaussianDiffusion3D(denoiser, timesteps=int(cfg.get("timesteps", config["diffusion"].get("timesteps", 80))))
    opt = AdamW(
        list(obs_encoder.parameters()) + list(denoiser.parameters()),
        lr=float(cfg.get("lr", config["diffusion"].get("lr", 5e-4))),
        weight_decay=float(cfg.get("weight_decay", 1e-6)),
    )
    n_steps = int(epochs if epochs is not None else cfg.get("epochs", 48))
    history: list[dict[str, float | str]] = []
    obs_encoder.train()
    denoiser.train()
    for step in range(n_steps):
        case = train_cases[step % len(train_cases)]
        opt.zero_grad(set_to_none=True)
        cond = obs_encoder(case.tokens, case.token_mask, case.attention_mask)
        loss = diffusion.training_loss(case.latent, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(obs_encoder.parameters()) + list(denoiser.parameters()), float(cfg.get("grad_clip", 2.0)))
        opt.step()
        row: dict[str, float | str] = {
            "step": float(step),
            "loss": float(loss.detach().cpu()),
            "sample": case.path.stem,
        }
        history.append(row)
        if step == 0 or (step + 1) % max(1, n_steps // 5) == 0:
            print(f"multi-diff step {step + 1:04d}/{n_steps} sample={case.path.stem} loss={row['loss']:.4f}")

    ckpt_path = Path(cfg.get("checkpoint", Path(config["paths"]["checkpoints_dir"]) / "multisample_latent_diffusion.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "denoiser_state": denoiser.state_dict(),
            "obs_encoder_state": obs_encoder.state_dict(),
            "config": config,
            "train_paths": [str(case.path) for case in train_cases],
            "holdout_path": str(holdout.path),
            "latent_shape": tuple(train_cases[0].latent.shape[1:]),
            "n_facies": n_facies,
            "history": history,
            "max_condition_tokens": max_tokens,
        },
        ckpt_path,
    )

    obs_encoder.eval()
    denoiser.eval()
    n_posterior = int(samples if samples is not None else cfg.get("posterior_samples", config["diffusion"].get("posterior_samples", 8)))
    use_base_proxy = bool(cfg.get("active_borehole_sampling", {}).get("enabled", False)) and not bool(cfg.get("active_borehole_sampling", {}).get("use_active_observations_in_proxy", False))
    proxy_fields = _proxy_from_observations(holdout.sample, n_facies=n_facies, use_base_observations=use_base_proxy)
    with torch.no_grad():
        proxy_volume = fields_to_volume_tensor(proxy_fields, n_facies=n_facies).to(dev)
        anchor = ae.encode(proxy_volume).detach()
        cond = obs_encoder(holdout.tokens, holdout.token_mask, holdout.attention_mask).repeat(n_posterior, 1)
        scale = float(cfg.get("posterior_noise_scale", config["diffusion"].get("posterior_noise_scale", 0.08)))
        correction = float(cfg.get("posterior_correction_scale", config["diffusion"].get("posterior_correction_scale", 0.15)))
        latents = anchor.repeat(n_posterior, 1, 1, 1, 1) + scale * torch.randn((n_posterior, *anchor.shape[1:]), device=dev)
        t_mid = max(int(cfg.get("timesteps", config["diffusion"].get("timesteps", 80))) // 2, 1)
        t = torch.full((n_posterior,), t_mid, device=dev, dtype=torch.long)
        latents = latents - correction * denoiser(latents, t, cond)
        decoded = ae.decode(latents)
    posterior = _posterior_arrays(decoded, n_facies=n_facies)
    posterior = _blend_continuous_with_proxy(
        posterior,
        proxy_fields,
        weight=float(cfg.get("continuous_proxy_blend", 0.70)),
        field_weights={
            "eic": float(cfg.get("eic_proxy_blend", cfg.get("continuous_proxy_blend", 0.70))),
            "temperature": float(cfg.get("temperature_proxy_blend", cfg.get("continuous_proxy_blend", 0.70))),
            "unfrozen_water": float(cfg.get("unfrozen_water_proxy_blend", cfg.get("continuous_proxy_blend", 0.70))),
            "log_resistivity": float(cfg.get("log_resistivity_proxy_blend", cfg.get("continuous_proxy_blend", 0.70))),
        },
    )
    if bool(cfg.get("ice_rich_event_calibration", True)):
        posterior = calibrate_ice_rich_eic(
            posterior,
            event_classes=tuple(int(v) for v in cfg.get("ice_rich_event_classes", (3, 6))),
            event_threshold=float(cfg.get("ice_rich_event_threshold", 0.50)),
            eic_floor=float(cfg.get("ice_rich_eic_floor", 0.31)),
        )
    if bool(cfg.get("facies_observation_assimilation", True)):
        posterior = assimilate_facies_observations(posterior, holdout.sample["observations"], holdout.sample["grid"], n_facies=n_facies)
    wedge_cfg = cfg.get("wedge_ice_morphology", {})
    if bool(wedge_cfg.get("enabled", True)):
        posterior = assimilate_wedge_ice_morphology(
            posterior,
            holdout.sample["observations"],
            holdout.sample["grid"],
            n_facies=n_facies,
            xy_radius_cells=int(wedge_cfg.get("xy_radius_cells", 1)),
            vertical_buffer_cells=int(wedge_cfg.get("vertical_buffer_cells", 3)),
            connect_observed_columns=bool(wedge_cfg.get("connect_observed_columns", True)),
            max_connection_cells=int(wedge_cfg.get("max_connection_cells", 0)),
        )
        if bool(cfg.get("facies_observation_assimilation", True)):
            posterior = assimilate_facies_observations(posterior, holdout.sample["observations"], holdout.sample["grid"], n_facies=n_facies)
    posterior["grid_x"] = holdout.sample["grid"]["x"]
    posterior["grid_y"] = holdout.sample["grid"]["y"]
    posterior["grid_z"] = holdout.sample["grid"]["z"]
    posterior["holdout_sample_id"] = np.asarray(holdout.path.stem)
    posterior["anchor_strategy"] = np.asarray("idw_observation_proxy")
    posterior["active_observations_in_proxy"] = np.asarray(not use_base_proxy, dtype=np.uint8)

    pred_path = Path(cfg.get("posterior_path", Path(config["paths"]["predictions_dir"]) / "multisample_diffusion_holdout.npz"))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **posterior)
    metrics = synthetic_metrics(
        _posterior_mean_fields(posterior),
        holdout.sample["fields"],
        holdout.sample["grid"]["z"],
        n_facies=n_facies,
        ice_threshold=float(config["evaluation"].get("ice_rich_threshold", 0.30)),
    )
    metrics["borehole_facies_accuracy"] = facies_observation_accuracy(_posterior_mean_fields(posterior)["facies"], holdout.sample)
    metrics["wedge_morphology_voxels"] = float(np.asarray(posterior.get("wedge_morphology_voxels", 0.0)))
    metrics["wedge_morphology_columns"] = float(np.asarray(posterior.get("wedge_morphology_columns", 0.0)))
    proxy_metrics = synthetic_metrics(
        proxy_fields,
        holdout.sample["fields"],
        holdout.sample["grid"]["z"],
        n_facies=n_facies,
        ice_threshold=float(config["evaluation"].get("ice_rich_threshold", 0.30)),
    )
    proxy_metrics["borehole_facies_accuracy"] = facies_observation_accuracy(proxy_fields["facies"], holdout.sample)
    common = {
        "holdout_sample": holdout.path.stem,
        "n_train_samples": float(len(train_cases)),
        "optimization_steps": float(n_steps),
        "condition_tokens": float(holdout.tokens.shape[1]),
        **rare_facies_observation_diagnostics(holdout.sample),
    }
    proxy_row: dict[str, float | str] = {
        "model": "IDWObservationProxy",
        **common,
        **proxy_metrics,
    }
    metric_row: dict[str, float | str] = {
        "model": "COLDReconMultiSampleDiffusion",
        **common,
        **metrics,
    }
    metrics_path = Path(cfg.get("metrics_path", Path(config["paths"]["tables_dir"]) / "multisample_diffusion_holdout_metrics.csv"))
    history_path = Path(cfg.get("history_path", Path(config["paths"]["tables_dir"]) / "multisample_diffusion_history.csv"))
    _write_csv(metrics_path, [proxy_row, metric_row])
    _write_csv(history_path, history)
    return {
        "checkpoint": ckpt_path,
        "posterior_path": pred_path,
        "metrics_path": metrics_path,
        "history_path": history_path,
        "history": history,
    }
