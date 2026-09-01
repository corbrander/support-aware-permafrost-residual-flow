from __future__ import annotations

import numpy as np


def generate_ice_fields(
    facies: np.ndarray,
    grid: dict,
    interfaces: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = grid["x"]
    y = grid["y"]
    z = grid["z"]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    lx = max(float(x[-1] - x[0]), 1.0)
    ly = max(float(y[-1] - y[0]), 1.0)

    base = np.zeros_like(facies, dtype=np.float32)
    base[facies == 0] = 0.05
    base[facies == 1] = 0.22
    base[facies == 2] = 0.18
    base[facies == 3] = 0.45
    base[facies == 4] = 0.08
    base[facies == 5] = 0.02
    base[facies == 6] = 0.72

    eic = np.maximum(base - 0.18, 0.0)
    ice_content = base.copy()

    for _ in range(18):
        cx = rng.uniform(0.05 * lx, 0.95 * lx)
        cy = rng.uniform(0.05 * ly, 0.95 * ly)
        cz = rng.uniform(1.2, min(float(z[-1]), 5.2))
        rx = rng.uniform(4.0, 12.0)
        ry = rng.uniform(3.0, 10.0)
        # Resolve lenses over multiple vertical cells. The previous 0.06-0.22 m
        # radii were sub-voxel for dz=0.25 m and produced unstable targets.
        dz = max(float(grid.get("dz", np.median(np.diff(z)) if len(z) > 1 else 0.25)), 1.0e-3)
        rz = rng.uniform(2.0 * dz, 4.0 * dz)
        lens = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2))
        mask = lens > 0.32
        mask &= facies != 5
        boost = rng.uniform(0.12, 0.35) * lens.astype(np.float32)
        ice_content = np.where(mask, np.maximum(ice_content, base + boost), ice_content)
        eic = np.where(mask, np.maximum(eic, boost), eic)

    rich_layer = facies == 3
    banding = 0.08 * (1.0 + np.sin(14.0 * zz + 0.06 * xx))
    eic = np.where(rich_layer, np.maximum(eic, 0.18 + banding), eic)
    ice_content = np.maximum(ice_content, eic + 0.16)

    ice_content = np.clip(ice_content, 0.0, 0.95).astype(np.float32)
    eic = np.clip(eic, 0.0, 0.75).astype(np.float32)
    return ice_content, eic
