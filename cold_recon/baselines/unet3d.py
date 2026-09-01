from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cold_recon.data.data_schema import OBS_TYPES, SURFACE_FEATURE_NAMES


SPARSE_UNET_INPUT_CHANNELS = (
    "x_norm",
    "y_norm",
    "z_norm",
    *[f"surface_{name}" for name in SURFACE_FEATURE_NAMES],
    "obs_facies_norm",
    "mask_facies",
    "obs_eic",
    "mask_eic",
    "obs_temperature_scaled",
    "mask_temperature",
    "obs_unfrozen_water",
    "mask_unfrozen_water",
    "obs_log_resistivity_scaled",
    "mask_log_resistivity",
    "obs_alt_scaled",
    "mask_alt",
)


class SmallUNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base: int = 16) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(in_channels, base, 3, padding=1), nn.GELU(), nn.Conv3d(base, base, 3, padding=1), nn.GELU())
        self.down = nn.Conv3d(base, base * 2, 3, stride=2, padding=1)
        self.mid = nn.Sequential(nn.GELU(), nn.Conv3d(base * 2, base * 2, 3, padding=1), nn.GELU())
        self.up = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.out = nn.Conv3d(base * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.enc1(x)
        m = self.mid(self.down(e))
        u = self.up(m)
        u = u[..., : e.shape[-3], : e.shape[-2], : e.shape[-1]]
        return self.out(torch.cat([u, e], dim=1))


def _empty(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)


def _indices_from_coords(coords: np.ndarray, grid: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.clip(np.round(coords[:, 0] / float(grid["dx"])).astype(int), 0, len(grid["x"]) - 1)
    iy = np.clip(np.round(coords[:, 1] / float(grid["dy"])).astype(int), 0, len(grid["y"]) - 1)
    iz = np.clip(np.round(coords[:, 2] / float(grid["dz"])).astype(int), 0, len(grid["z"]) - 1)
    return ix, iy, iz


def _scatter_mean(
    value_grid: np.ndarray,
    mask_grid: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    values: np.ndarray,
) -> None:
    count = np.zeros_like(value_grid, dtype=np.float32)
    np.add.at(value_grid, (ix, iy, iz), values.astype(np.float32))
    np.add.at(count, (ix, iy, iz), 1.0)
    observed = count > 0
    value_grid[observed] /= count[observed]
    mask_grid[observed] = 1.0


def build_sparse_observation_volume(sample: dict, n_facies: int = 7) -> torch.Tensor:
    """Rasterize irregular sparse observations into a 3D U-Net conditioning volume."""
    grid = sample["grid"]
    obs = sample["observations"]
    x = np.asarray(grid["x"], dtype=np.float32)
    y = np.asarray(grid["y"], dtype=np.float32)
    z = np.asarray(grid["z"], dtype=np.float32)
    shape = (len(x), len(y), len(z))
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    channels: list[np.ndarray] = [
        (xx / max(float(x[-1]), 1.0)).astype(np.float32),
        (yy / max(float(y[-1]), 1.0)).astype(np.float32),
        (zz / max(float(z[-1]), 1.0)).astype(np.float32),
    ]
    for name in SURFACE_FEATURE_NAMES:
        surface = np.asarray(sample["surface_features"][name], dtype=np.float32)
        scale = float(np.nanstd(surface))
        normalized = (surface - float(np.nanmean(surface))) / (scale if scale > 1e-6 else 1.0)
        channels.append(np.repeat(normalized[:, :, None], len(z), axis=2).astype(np.float32))

    obs_specs = [
        ("borehole_facies", "facies", lambda v: np.clip(v / max(n_facies - 1, 1), 0.0, 1.0)),
        ("borehole_eic", "eic", lambda v: np.clip(v, 0.0, 1.0)),
        ("borehole_temperature", "temperature", lambda v: v / 10.0),
        ("nmr_unfrozen_water", "unfrozen_water", lambda v: np.clip(v, 0.0, 1.0)),
        ("ert_log_resistivity", "log_resistivity", lambda v: v / 10.0),
    ]
    for type_name, _, transform in obs_specs:
        values, mask = _empty(shape)
        obs_mask = obs.type_ids == OBS_TYPES[type_name]
        if np.any(obs_mask):
            ix, iy, iz = _indices_from_coords(obs.coords[obs_mask], grid)
            _scatter_mean(values, mask, ix, iy, iz, transform(obs.values[obs_mask]))
        channels.extend([values, mask])

    alt_values, alt_mask = _empty(shape)
    obs_mask = obs.type_ids == OBS_TYPES["alt"]
    if np.any(obs_mask):
        ix, iy, _ = _indices_from_coords(obs.coords[obs_mask], grid)
        zmax = max(float(z[-1]), 1.0)
        for x_idx, y_idx, value in zip(ix, iy, obs.values[obs_mask]):
            alt_values[x_idx, y_idx, :] = float(value) / zmax
            alt_mask[x_idx, y_idx, :] = 1.0
    channels.extend([alt_values, alt_mask])
    volume = np.stack(channels, axis=0).astype(np.float32)
    return torch.from_numpy(volume).unsqueeze(0)
