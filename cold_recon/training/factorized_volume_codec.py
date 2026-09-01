from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


N_LITHOLOGY = 4
N_THERMAL_STATE = 3
N_ICE_STRUCTURE = 3
N_CONTINUOUS = 4
FACTORIZED_CHANNELS = N_LITHOLOGY + N_THERMAL_STATE + N_ICE_STRUCTURE + N_CONTINUOUS

CONTINUOUS_NAMES = ("eic", "temperature", "unfrozen_water", "log_resistivity")
CONTINUOUS_SCALES = (1.0, 10.0, 1.0, 10.0)


def _one_hot(volume: np.ndarray, classes: int) -> np.ndarray:
    values = np.asarray(volume, dtype=np.int64)
    return np.eye(int(classes), dtype=np.float32)[values].transpose(3, 0, 1, 2)


def sample_to_factorized_tensor(sample: dict) -> torch.Tensor:
    fields = sample["fields"]
    categorical = [
        _one_hot(fields["lithology"], N_LITHOLOGY),
        _one_hot(fields["thermal_state"], N_THERMAL_STATE),
        _one_hot(fields["ice_structure"], N_ICE_STRUCTURE),
    ]
    continuous = [
        np.asarray(fields["eic"], dtype=np.float32)[None, ...],
        np.asarray(fields["temperature"], dtype=np.float32)[None, ...] / 10.0,
        np.asarray(fields["unfrozen_water"], dtype=np.float32)[None, ...],
        np.log(np.maximum(np.asarray(fields["resistivity"], dtype=np.float32), 1.0))[None, ...] / 10.0,
    ]
    return torch.from_numpy(np.concatenate(categorical + continuous, axis=0)).unsqueeze(0)


def factorized_label_masks(sample: dict) -> dict[str, torch.Tensor]:
    fields = sample["fields"]
    shape = fields["lithology"].shape
    return {
        "lithology": torch.as_tensor(fields.get("label_mask_lithology", np.ones(shape)), dtype=torch.bool).unsqueeze(0),
        "thermal_state": torch.as_tensor(fields.get("label_mask_thermal_state", np.ones(shape)), dtype=torch.bool).unsqueeze(0),
        "ice_structure": torch.as_tensor(fields.get("label_mask_ice_structure", np.ones(shape)), dtype=torch.bool).unsqueeze(0),
    }


def factorized_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks: dict[str, torch.Tensor] | None = None,
    continuous_weights: tuple[float, float, float, float] = (4.0, 0.2, 1.0, 0.1),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.shape[1] != FACTORIZED_CHANNELS:
        raise ValueError("prediction and target must have the factorized channel layout")
    masks = masks or {}
    offsets = (0, N_LITHOLOGY, N_LITHOLOGY + N_THERMAL_STATE)
    names = ("lithology", "thermal_state", "ice_structure")
    classes = (N_LITHOLOGY, N_THERMAL_STATE, N_ICE_STRUCTURE)
    parts: dict[str, torch.Tensor] = {}
    total = prediction.new_zeros(())
    for name, offset, n_classes in zip(names, offsets, classes, strict=True):
        truth = target[:, offset : offset + n_classes].argmax(dim=1)
        loss_map = F.cross_entropy(prediction[:, offset : offset + n_classes], truth, reduction="none")
        mask = masks.get(name)
        if mask is not None:
            mask = mask.to(prediction.device)
            loss = torch.sum(loss_map * mask) / mask.sum().clamp_min(1)
        else:
            loss = loss_map.mean()
        parts[name] = loss
        total = total + loss
    start = sum(classes)
    for index, (name, weight) in enumerate(zip(CONTINUOUS_NAMES, continuous_weights, strict=True)):
        loss = F.smooth_l1_loss(prediction[:, start + index], target[:, start + index])
        parts[name] = loss
        total = total + float(weight) * loss
    parts["total"] = total
    return total, parts


def tensor_to_factorized_fields(volume: torch.Tensor) -> dict[str, np.ndarray]:
    values = volume.detach().cpu()
    if values.ndim == 5 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 4 or values.shape[0] != FACTORIZED_CHANNELS:
        raise ValueError("volume must have shape [14, x, y, z]")
    offset_s = N_LITHOLOGY
    offset_i = offset_s + N_THERMAL_STATE
    offset_c = offset_i + N_ICE_STRUCTURE
    return {
        "lithology": values[:offset_s].argmax(dim=0).numpy().astype(np.int16),
        "thermal_state": values[offset_s:offset_i].argmax(dim=0).numpy().astype(np.int16),
        "ice_structure": values[offset_i:offset_c].argmax(dim=0).numpy().astype(np.int16),
        "eic": values[offset_c].numpy().astype(np.float32),
        "temperature": (10.0 * values[offset_c + 1]).numpy().astype(np.float32),
        "unfrozen_water": values[offset_c + 2].numpy().astype(np.float32),
        "log_resistivity": (10.0 * values[offset_c + 3]).numpy().astype(np.float32),
    }


def factorized_ensemble_to_posterior(decoded: torch.Tensor) -> dict[str, np.ndarray]:
    values = decoded.detach().float().cpu()
    if values.ndim != 5 or values.shape[1] != FACTORIZED_CHANNELS:
        raise ValueError("decoded must have shape [ensemble, 14, x, y, z]")
    offset_s = N_LITHOLOGY
    offset_i = offset_s + N_THERMAL_STATE
    offset_c = offset_i + N_ICE_STRUCTURE
    outputs: dict[str, np.ndarray] = {}
    for name, lower, upper in (
        ("lithology", 0, offset_s),
        ("thermal_state", offset_s, offset_i),
        ("ice_structure", offset_i, offset_c),
    ):
        probabilities = torch.softmax(values[:, lower:upper], dim=1)
        mean_probability = probabilities.mean(dim=0).permute(1, 2, 3, 0).numpy().astype(np.float32)
        outputs[f"{name}_probability"] = mean_probability
        outputs[f"{name}_mode"] = np.argmax(mean_probability, axis=-1).astype(np.int16)
        outputs[f"{name}_entropy"] = (
            -np.sum(mean_probability * np.log(np.clip(mean_probability, 1.0e-8, 1.0)), axis=-1)
        ).astype(np.float32)
    continuous = {
        "eic": values[:, offset_c].numpy(),
        "temperature": 10.0 * values[:, offset_c + 1].numpy(),
        "unfrozen_water": values[:, offset_c + 2].numpy(),
        "log_resistivity": 10.0 * values[:, offset_c + 3].numpy(),
    }
    for name, samples in continuous.items():
        samples = samples.astype(np.float32)
        outputs[f"{name}_samples"] = samples
        outputs[f"{name}_mean"] = samples.mean(axis=0).astype(np.float32)
        outputs[f"{name}_std"] = samples.std(axis=0).astype(np.float32)
    outputs["resistivity_samples"] = np.exp(np.clip(outputs["log_resistivity_samples"], 0.0, 15.0)).astype(np.float32)
    outputs["resistivity_mean"] = outputs["resistivity_samples"].mean(axis=0).astype(np.float32)
    outputs["ice_rich_probability"] = np.mean(outputs["eic_samples"] >= 0.30, axis=0).astype(np.float32)
    return outputs


def bounded_recenter_samples(
    samples: np.ndarray,
    target_mean: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Recenter an ensemble exactly while preserving physical pointwise bounds."""

    values = np.asarray(samples, dtype=np.float32)
    target = np.clip(np.asarray(target_mean, dtype=np.float32), float(lower), float(upper))
    deviations = values - values.mean(axis=0, keepdims=True)
    positive = np.maximum(deviations.max(axis=0), 0.0)
    negative = np.maximum(-deviations.min(axis=0), 0.0)
    positive_scale = np.divide(
        float(upper) - target,
        positive,
        out=np.full_like(target, np.inf),
        where=positive > 0.0,
    )
    negative_scale = np.divide(
        target - float(lower),
        negative,
        out=np.full_like(target, np.inf),
        where=negative > 0.0,
    )
    scale = np.minimum(1.0, np.minimum(positive_scale, negative_scale))
    scale = np.clip(scale, 0.0, 1.0)
    return (target[None, ...] + scale[None, ...] * deviations).astype(np.float32)
