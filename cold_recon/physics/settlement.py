from __future__ import annotations

import numpy as np
import torch


def settlement_potential_numpy(eic: np.ndarray, future_temperature: np.ndarray, dz: float, thaw_threshold: float = 0.0) -> np.ndarray:
    thaw_mask = future_temperature > thaw_threshold
    return np.sum(eic * thaw_mask, axis=2).astype(np.float32) * float(dz)


def settlement_potential_torch(eic: torch.Tensor, future_temperature: torch.Tensor, dz: float, thaw_threshold: float = 0.0) -> torch.Tensor:
    thaw_mask = (future_temperature > thaw_threshold).float()
    return torch.sum(eic * thaw_mask, dim=-1) * float(dz)


def differential_settlement_risk(settlement: np.ndarray, critical_gradient: float = 0.08) -> np.ndarray:
    gx, gy = np.gradient(settlement)
    return (np.hypot(gx, gy) > critical_gradient).astype(np.float32)

