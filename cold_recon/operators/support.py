from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter

from cold_recon.data.data_schema import SUPPORT_TYPES, ObservationTable


def _grid_axes(grid: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = tuple(np.asarray(grid[name], dtype=np.float64) for name in ("x", "y", "z"))
    if any(axis.ndim != 1 or len(axis) == 0 for axis in axes):
        raise ValueError("grid x, y, and z must be non-empty one-dimensional arrays")
    if any(len(axis) > 1 and np.any(np.diff(axis) <= 0) for axis in axes):
        raise ValueError("grid axes must be strictly increasing")
    return axes  # type: ignore[return-value]


def _flat_index(ix: int, iy: int, iz: int, shape: tuple[int, int, int]) -> int:
    return int(np.ravel_multi_index((ix, iy, iz), shape, order="C"))


def _axis_linear_weights(axis: np.ndarray, value: float) -> list[tuple[int, float]]:
    if len(axis) == 1 or value <= float(axis[0]):
        return [(0, 1.0)]
    if value >= float(axis[-1]):
        return [(len(axis) - 1, 1.0)]
    upper = int(np.searchsorted(axis, value, side="right"))
    lower = upper - 1
    fraction = float((value - axis[lower]) / (axis[upper] - axis[lower]))
    return [(lower, 1.0 - fraction), (upper, fraction)]


def _cell_bounds(axis: np.ndarray) -> np.ndarray:
    if len(axis) == 1:
        return np.asarray([axis[0] - 0.5, axis[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (axis[1:] + axis[:-1])
    first = axis[0] - 0.5 * (axis[1] - axis[0])
    last = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return np.concatenate([[first], mids, [last]])


def _axis_overlap_weights(axis: np.ndarray, lower: float, upper: float) -> list[tuple[int, float]]:
    if upper < lower:
        lower, upper = upper, lower
    if np.isclose(upper, lower):
        return _axis_linear_weights(axis, 0.5 * (lower + upper))
    bounds = _cell_bounds(axis)
    overlap = np.maximum(
        0.0,
        np.minimum(bounds[1:], float(upper)) - np.maximum(bounds[:-1], float(lower)),
    )
    total = float(overlap.sum())
    if total <= 0.0:
        return _axis_linear_weights(axis, 0.5 * (lower + upper))
    indices = np.flatnonzero(overlap > 0.0)
    return [(int(index), float(overlap[index] / total)) for index in indices]


@dataclass(frozen=True)
class SupportOperator:
    matrix: sparse.csr_matrix
    grid_shape: tuple[int, int, int]
    observation_indices: np.ndarray
    support_names: tuple[str, ...]

    def apply(self, field: np.ndarray) -> np.ndarray:
        values = np.asarray(field)
        if values.shape != self.grid_shape:
            raise ValueError(f"field shape {values.shape} does not match grid {self.grid_shape}")
        return np.asarray(self.matrix @ values.reshape(-1, order="C"), dtype=np.float64)

    def apply_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities)
        if values.shape[:3] != self.grid_shape or values.ndim != 4:
            raise ValueError("probabilities must have shape [nx, ny, nz, n_classes]")
        return np.asarray(self.matrix @ values.reshape(-1, values.shape[-1], order="C"))

    def adjoint(self, residual: np.ndarray) -> np.ndarray:
        values = np.asarray(residual, dtype=np.float64)
        if values.shape != (self.matrix.shape[0],):
            raise ValueError("residual length does not match operator rows")
        return np.asarray(self.matrix.T @ values).reshape(self.grid_shape, order="C")


def _assemble_rows(
    rows: Iterable[list[tuple[int, float]]],
    grid_shape: tuple[int, int, int],
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    rows_list = list(rows)
    for row_index, entries in enumerate(rows_list):
        merged: dict[int, float] = {}
        for column, weight in entries:
            merged[int(column)] = merged.get(int(column), 0.0) + float(weight)
        total = float(sum(merged.values()))
        if total <= 0.0:
            raise ValueError(f"support row {row_index} has no positive weight")
        for column, weight in merged.items():
            if weight <= 0.0:
                continue
            row_indices.append(row_index)
            col_indices.append(column)
            values.append(weight / total)
    return sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(rows_list), int(np.prod(grid_shape))),
        dtype=np.float64,
    )


def point_trilinear_operator(coords: np.ndarray, grid: dict) -> sparse.csr_matrix:
    x, y, z = _grid_axes(grid)
    shape = (len(x), len(y), len(z))
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [n, 3]")
    rows: list[list[tuple[int, float]]] = []
    for coord in coords:
        entries: list[tuple[int, float]] = []
        for ix, wx in _axis_linear_weights(x, float(coord[0])):
            for iy, wy in _axis_linear_weights(y, float(coord[1])):
                for iz, wz in _axis_linear_weights(z, float(coord[2])):
                    entries.append((_flat_index(ix, iy, iz, shape), wx * wy * wz))
        rows.append(entries)
    return _assemble_rows(rows, shape)


def interval_operator(
    coords: np.ndarray,
    vertical_extent: np.ndarray,
    grid: dict,
) -> sparse.csr_matrix:
    """Borehole interval averages with trilinear horizontal interpolation."""

    x, y, z = _grid_axes(grid)
    shape = (len(x), len(y), len(z))
    coords = np.asarray(coords, dtype=np.float64)
    extents = np.asarray(vertical_extent, dtype=np.float64).reshape(-1)
    if coords.shape != (len(extents), 3):
        raise ValueError("coords and vertical_extent lengths differ")
    rows: list[list[tuple[int, float]]] = []
    for coord, extent in zip(coords, extents, strict=True):
        z_weights = _axis_overlap_weights(
            z, float(coord[2] - 0.5 * extent), float(coord[2] + 0.5 * extent)
        )
        entries: list[tuple[int, float]] = []
        for ix, wx in _axis_linear_weights(x, float(coord[0])):
            for iy, wy in _axis_linear_weights(y, float(coord[1])):
                for iz, wz in z_weights:
                    entries.append((_flat_index(ix, iy, iz, shape), wx * wy * wz))
        rows.append(entries)
    return _assemble_rows(rows, shape)


def box_volume_operator(
    coords: np.ndarray,
    extents: np.ndarray,
    grid: dict,
) -> sparse.csr_matrix:
    """Volume-overlap average for ERT inversion cells or rectangular supports."""

    x, y, z = _grid_axes(grid)
    shape = (len(x), len(y), len(z))
    coords = np.asarray(coords, dtype=np.float64)
    extents = np.asarray(extents, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or extents.shape != coords.shape:
        raise ValueError("coords and extents must both have shape [n, 3]")
    rows: list[list[tuple[int, float]]] = []
    for coord, extent in zip(coords, extents, strict=True):
        axis_weights = [
            _axis_overlap_weights(axis, float(center - 0.5 * width), float(center + 0.5 * width))
            for axis, center, width in zip((x, y, z), coord, np.maximum(extent, 0.0), strict=True)
        ]
        entries: list[tuple[int, float]] = []
        for ix, wx in axis_weights[0]:
            for iy, wy in axis_weights[1]:
                for iz, wz in axis_weights[2]:
                    entries.append((_flat_index(ix, iy, iz, shape), wx * wy * wz))
        rows.append(entries)
    return _assemble_rows(rows, shape)


def gaussian_kernel_operator(
    coords: np.ndarray,
    scales: np.ndarray,
    grid: dict,
    truncate: float = 3.0,
) -> sparse.csr_matrix:
    """Anisotropic Gaussian support for NMR and probe-volume measurements."""

    x, y, z = _grid_axes(grid)
    shape = (len(x), len(y), len(z))
    coords = np.asarray(coords, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or scales.shape != coords.shape:
        raise ValueError("coords and scales must both have shape [n, 3]")
    rows: list[list[tuple[int, float]]] = []
    for coord, scale in zip(coords, scales, strict=True):
        scale = np.maximum(scale, 1.0e-6)
        candidates = [
            np.flatnonzero(np.abs(axis - center) <= float(truncate) * sigma)
            for axis, center, sigma in zip((x, y, z), coord, scale, strict=True)
        ]
        candidates = [
            indices if len(indices) else np.asarray([int(np.argmin(np.abs(axis - center)))])
            for indices, axis, center in zip(candidates, (x, y, z), coord, strict=True)
        ]
        entries: list[tuple[int, float]] = []
        for ix in candidates[0]:
            for iy in candidates[1]:
                for iz in candidates[2]:
                    delta = np.asarray([x[ix], y[iy], z[iz]]) - coord
                    weight = float(np.exp(-0.5 * np.sum((delta / scale) ** 2)))
                    entries.append((_flat_index(int(ix), int(iy), int(iz), shape), weight))
        rows.append(entries)
    return _assemble_rows(rows, shape)


def build_observation_operator(
    observations: ObservationTable,
    grid: dict,
    indices: np.ndarray | None = None,
) -> SupportOperator:
    """Build a single sparse operator while preserving observation row order."""

    selected = np.arange(observations.n_obs, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    x, y, z = _grid_axes(grid)
    shape = (len(x), len(y), len(z))
    rows: list[sparse.csr_matrix] = []
    names: list[str] = []
    for index in selected:
        coord = observations.coords[index : index + 1]
        extent = observations.support_extent[index : index + 1]
        support_type = int(observations.support_type_ids[index])
        if support_type == SUPPORT_TYPES["borehole_interval"]:
            matrix = interval_operator(coord, extent[:, 2], grid)
            name = "borehole_interval"
        elif support_type == SUPPORT_TYPES["ert_volume"]:
            matrix = box_volume_operator(coord, extent, grid)
            name = "ert_volume"
        elif support_type == SUPPORT_TYPES["nmr_kernel"]:
            matrix = gaussian_kernel_operator(coord, np.maximum(0.5 * extent, 1.0e-6), grid)
            name = "nmr_kernel"
        else:
            matrix = point_trilinear_operator(coord, grid)
            name = "surface_crossing" if support_type == SUPPORT_TYPES["surface_crossing"] else "point"
        rows.append(matrix)
        names.append(name)
    matrix = sparse.vstack(rows, format="csr") if rows else sparse.csr_matrix((0, int(np.prod(shape))))
    return SupportOperator(matrix, shape, selected, tuple(names))


def apply_surface_crossing(
    temperature: np.ndarray,
    z: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Return the first thaw-to-frozen crossing with linear depth interpolation."""

    values = np.asarray(temperature, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != len(z):
        raise ValueError("temperature must have shape [nx, ny, nz]")
    out = np.zeros(values.shape[:2], dtype=np.float64)
    for ix in range(values.shape[0]):
        for iy in range(values.shape[1]):
            profile = values[ix, iy] - float(threshold)
            crossings = np.flatnonzero((profile[:-1] > 0.0) & (profile[1:] <= 0.0))
            if len(crossings) == 0:
                thawed = np.flatnonzero(profile > 0.0)
                out[ix, iy] = float(z[thawed[-1]]) if len(thawed) else 0.0
                continue
            k = int(crossings[0])
            fraction = float(profile[k] / max(profile[k] - profile[k + 1], 1.0e-12))
            out[ix, iy] = float(z[k] + fraction * (z[k + 1] - z[k]))
    return out.astype(np.float32)


def build_error_covariance(
    observations: ObservationTable,
    indices: np.ndarray | None = None,
    *,
    correlated: bool = True,
    length_scale: float = 4.0,
    min_sigma: float = 1.0e-6,
    jitter: float = 1.0e-8,
) -> np.ndarray:
    selected = np.arange(observations.n_obs, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    sigma = np.maximum(observations.sigma[selected].astype(np.float64), float(min_sigma))
    covariance = np.diag(sigma**2)
    if correlated and len(selected) > 1:
        coords = observations.coords[selected].astype(np.float64)
        groups = observations.group_ids[selected]
        source = observations.source_ids[selected]
        delta = coords[:, None, :] - coords[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        same_profile = (groups[:, None] == groups[None, :]) & (groups[:, None] >= 0)
        same_source = source[:, None] == source[None, :]
        correlation = np.exp(-distance / max(float(length_scale), 1.0e-6))
        add = sigma[:, None] * sigma[None, :] * correlation
        mask = same_profile & same_source & ~np.eye(len(selected), dtype=bool)
        covariance[mask] = add[mask]
    covariance.flat[:: len(selected) + 1] += float(jitter) if len(selected) else 0.0
    return covariance


def sample_profile_correlated_noise(
    observations: ObservationTable,
    indices: np.ndarray,
    rng: np.random.Generator,
    *,
    length_scale: float = 4.0,
    min_sigma: float = 1.0e-6,
) -> np.ndarray:
    """Sample a fast correlated error field on a regular/near-regular profile.

    The likelihood still uses the explicit exponential covariance returned by
    :func:`build_error_covariance`. This sampler avoids an O(n^3) dense
    decomposition for large synthetic ERT profiles by drawing a two-dimensional
    correlated field along profile distance and depth.
    """

    selected = np.asarray(indices, dtype=np.int64)
    if len(selected) == 0:
        return np.empty((0,), dtype=np.float64)
    coords = observations.coords[selected].astype(np.float64)
    orientation = np.nanmean(observations.orientation[selected, :2], axis=0)
    norm = float(np.linalg.norm(orientation))
    if norm <= 1.0e-8:
        spread = np.ptp(coords[:, :2], axis=0)
        orientation = np.asarray([1.0, 0.0]) if spread[0] >= spread[1] else np.asarray([0.0, 1.0])
    else:
        orientation = orientation / norm
    along = coords[:, :2] @ orientation
    along_axis, along_inverse = np.unique(np.round(along, 6), return_inverse=True)
    depth_axis, depth_inverse = np.unique(np.round(coords[:, 2], 6), return_inverse=True)

    def spacing(axis: np.ndarray) -> float:
        differences = np.diff(axis)
        differences = differences[differences > 1.0e-8]
        return float(np.median(differences)) if len(differences) else 1.0

    white = rng.normal(size=(len(along_axis), len(depth_axis)))
    correlated = gaussian_filter(
        white,
        sigma=(
            max(float(length_scale) / spacing(along_axis), 0.25),
            max(float(length_scale) / spacing(depth_axis), 0.25),
        ),
        mode="reflect",
    )
    values = correlated[along_inverse, depth_inverse]
    values -= float(np.mean(values))
    values /= max(float(np.std(values)), 1.0e-8)
    sigma = np.maximum(observations.sigma[selected].astype(np.float64), float(min_sigma))
    return values * sigma


def normalized_misfit(
    predicted: np.ndarray,
    observed: np.ndarray,
    covariance: np.ndarray,
) -> float:
    residual = np.asarray(predicted, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (len(residual), len(residual)):
        raise ValueError("covariance shape does not match residual")
    if len(residual) == 0:
        return float("nan")
    chol = np.linalg.cholesky(covariance)
    whitened = solve_triangular(chol, residual, lower=True, check_finite=False)
    return float(np.sqrt(np.mean(whitened**2)))
