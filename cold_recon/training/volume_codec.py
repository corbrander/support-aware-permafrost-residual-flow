from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


VOLUME_CHANNELS = (
    "facies_0",
    "facies_1",
    "facies_2",
    "facies_3",
    "facies_4",
    "facies_5",
    "facies_6",
    "eic",
    "temperature_scaled",
    "unfrozen_water",
    "log_resistivity_scaled",
)


def sample_to_volume_tensor(sample: dict[str, Any], n_facies: int = 7) -> torch.Tensor:
    fields = sample["fields"]
    facies = torch.as_tensor(fields["facies"].astype(np.int64))
    one_hot = F.one_hot(facies, num_classes=n_facies).permute(3, 0, 1, 2).float()
    eic = torch.as_tensor(fields["eic"], dtype=torch.float32).unsqueeze(0)
    temperature = (torch.as_tensor(fields["temperature"], dtype=torch.float32) / 10.0).unsqueeze(0)
    unfrozen = torch.as_tensor(fields["unfrozen_water"], dtype=torch.float32).unsqueeze(0)
    log_rho = (torch.log(torch.as_tensor(fields["resistivity"], dtype=torch.float32).clamp_min(1.0)) / 10.0).unsqueeze(0)
    return torch.cat([one_hot, eic, temperature, unfrozen, log_rho], dim=0).unsqueeze(0)


def fields_to_volume_tensor(fields: dict[str, np.ndarray], n_facies: int = 7) -> torch.Tensor:
    facies_key = "facies" if "facies" in fields else "facies_mode"
    facies = torch.as_tensor(fields[facies_key].astype(np.int64))
    one_hot = F.one_hot(facies.clamp(min=0), num_classes=n_facies).permute(3, 0, 1, 2).float()
    eic_key = "eic" if "eic" in fields else "eic_mean"
    temp_key = "temperature" if "temperature" in fields else "temperature_mean"
    theta_key = "unfrozen_water" if "unfrozen_water" in fields else "unfrozen_water_mean"
    rho_key = "log_resistivity" if "log_resistivity" in fields else "log_resistivity_mean"
    eic = torch.as_tensor(fields[eic_key], dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(0)
    temperature = (torch.as_tensor(fields[temp_key], dtype=torch.float32) / 10.0).unsqueeze(0)
    unfrozen = torch.as_tensor(fields[theta_key], dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(0)
    log_rho = (torch.as_tensor(fields[rho_key], dtype=torch.float32) / 10.0).unsqueeze(0)
    return torch.cat([one_hot, eic, temperature, unfrozen, log_rho], dim=0).unsqueeze(0)


def volume_tensor_to_fields(volume: torch.Tensor, n_facies: int = 7) -> dict[str, np.ndarray]:
    if volume.dim() == 5:
        volume = volume[0]
    vol = volume.detach().cpu()
    facies_logits = vol[:n_facies]
    facies = torch.argmax(facies_logits, dim=0).short().numpy()
    eic = vol[n_facies].clamp(0.0, 1.0).numpy().astype(np.float32)
    temperature = (vol[n_facies + 1] * 10.0).numpy().astype(np.float32)
    unfrozen = vol[n_facies + 2].clamp(0.0, 1.0).numpy().astype(np.float32)
    log_resistivity = (vol[n_facies + 3] * 10.0).numpy().astype(np.float32)
    return {
        "facies": facies,
        "eic": eic,
        "temperature": temperature,
        "unfrozen_water": unfrozen,
        "log_resistivity": log_resistivity,
        "resistivity": np.exp(np.clip(log_resistivity, 0.0, 12.0)).astype(np.float32),
    }


def batch_volume_to_field_ensemble(volumes: torch.Tensor, n_facies: int = 7) -> dict[str, np.ndarray]:
    fields = [volume_tensor_to_fields(volumes[i], n_facies=n_facies) for i in range(volumes.shape[0])]
    keys = fields[0].keys()
    return {key: np.stack([f[key] for f in fields], axis=0) for key in keys}
