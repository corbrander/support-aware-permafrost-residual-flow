from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


N_CRYOFACIES = 7
N_CONTINUOUS = 4
MIXED_CHANNELS = N_CRYOFACIES + N_CONTINUOUS
CONTINUOUS_NAMES = ("eic", "temperature", "unfrozen_water", "log_resistivity")


def _one_hot(volume: np.ndarray, classes: int) -> np.ndarray:
    values = np.asarray(volume, dtype=np.int64)
    return np.eye(int(classes), dtype=np.float32)[values].transpose(3, 0, 1, 2)


def sample_to_mixed_tensor(sample: dict) -> torch.Tensor:
    """Encode the legacy seven-class state plus four continuous fields."""

    fields = sample["fields"]
    facies_source = fields.get("facies", fields.get("cryofacies"))
    if facies_source is None:
        raise KeyError("sample fields contain no facies or cryofacies volume")
    facies = np.asarray(facies_source)
    channels = [
        _one_hot(facies, N_CRYOFACIES),
        np.asarray(fields["eic"], dtype=np.float32)[None, ...],
        np.asarray(fields["temperature"], dtype=np.float32)[None, ...] / 10.0,
        np.asarray(fields["unfrozen_water"], dtype=np.float32)[None, ...],
        np.log(
            np.maximum(np.asarray(fields["resistivity"], dtype=np.float32), 1.0)
        )[None, ...]
        / 10.0,
    ]
    return torch.from_numpy(np.concatenate(channels, axis=0)).unsqueeze(0)


def prior_to_mixed_tensor(prior: dict[str, np.ndarray]) -> torch.Tensor:
    fields = {
        "facies": np.asarray(prior["facies"]),
        "eic": np.asarray(prior["eic"]),
        "temperature": np.asarray(prior["temperature"]),
        "unfrozen_water": np.asarray(prior["unfrozen_water"]),
        "resistivity": np.exp(np.asarray(prior["log_resistivity"])),
    }
    return sample_to_mixed_tensor({"fields": fields})


def mixed_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    continuous_weights: tuple[float, float, float, float] = (4.0, 0.2, 1.0, 0.1),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.shape[1] != MIXED_CHANNELS:
        raise ValueError("prediction and target must have the 11-channel mixed layout")
    truth_facies = target[:, :N_CRYOFACIES].argmax(dim=1)
    parts: dict[str, torch.Tensor] = {
        "cryofacies": F.cross_entropy(prediction[:, :N_CRYOFACIES], truth_facies)
    }
    total = parts["cryofacies"]
    for index, (name, weight) in enumerate(
        zip(CONTINUOUS_NAMES, continuous_weights, strict=True)
    ):
        loss = F.smooth_l1_loss(
            prediction[:, N_CRYOFACIES + index],
            target[:, N_CRYOFACIES + index],
        )
        parts[name] = loss
        total = total + float(weight) * loss
    parts["total"] = total
    return total, parts


def mixed_ensemble_to_posterior(decoded: torch.Tensor) -> dict[str, np.ndarray]:
    values = decoded.detach().float().cpu()
    if values.ndim != 5 or values.shape[1] != MIXED_CHANNELS:
        raise ValueError("decoded must have shape [ensemble, 11, x, y, z]")
    probability = torch.softmax(values[:, :N_CRYOFACIES], dim=1).mean(dim=0)
    probability = probability.permute(1, 2, 3, 0).numpy().astype(np.float32)
    outputs: dict[str, np.ndarray] = {
        "cryofacies_probability": probability,
        "cryofacies_mode": np.argmax(probability, axis=-1).astype(np.int16),
        "cryofacies_entropy": (
            -np.sum(probability * np.log(np.clip(probability, 1.0e-8, 1.0)), axis=-1)
        ).astype(np.float32),
    }
    continuous = {
        "eic": values[:, N_CRYOFACIES].numpy(),
        "temperature": 10.0 * values[:, N_CRYOFACIES + 1].numpy(),
        "unfrozen_water": values[:, N_CRYOFACIES + 2].numpy(),
        "log_resistivity": 10.0 * values[:, N_CRYOFACIES + 3].numpy(),
    }
    bounds = {
        "eic": (0.0, 0.90),
        "temperature": (-12.0, 4.0),
        "unfrozen_water": (0.0, 0.85),
        "log_resistivity": (0.0, 15.0),
    }
    for name, samples in continuous.items():
        lower, upper = bounds[name]
        samples = np.clip(samples, lower, upper).astype(np.float32)
        outputs[f"{name}_samples"] = samples
        outputs[f"{name}_mean"] = samples.mean(axis=0).astype(np.float32)
        outputs[f"{name}_std"] = samples.std(axis=0).astype(np.float32)
    outputs["resistivity_samples"] = np.exp(
        outputs["log_resistivity_samples"]
    ).astype(np.float32)
    outputs["resistivity_mean"] = outputs["resistivity_samples"].mean(axis=0).astype(
        np.float32
    )
    outputs["ice_rich_probability"] = np.mean(
        outputs["eic_samples"] >= 0.30, axis=0
    ).astype(np.float32)
    return outputs
