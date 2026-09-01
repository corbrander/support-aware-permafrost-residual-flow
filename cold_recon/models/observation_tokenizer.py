from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cold_recon.data.data_schema import OBS_TYPES, SUPPORT_TYPES, ObservationTable


@dataclass
class TokenizerStats:
    xyz_max: tuple[float, float, float]
    value_scale: float = 10.0
    sigma_scale: float = 1.0


@dataclass
class ObservationGraph:
    """Sparse observation-neighborhood graph and dense Transformer mask view."""

    edge_index: np.ndarray
    edge_weight: np.ndarray
    attention_mask: np.ndarray
    normalized_coords: np.ndarray


class ObservationTokenizer:
    """Convert heterogeneous sparse observations into fixed-width tokens."""

    def __init__(
        self,
        n_types: int | None = None,
        stats: TokenizerStats | None = None,
        support_aware: bool = False,
        n_support_types: int | None = None,
        n_sites: int = 8,
        n_sources: int = 16,
    ):
        self.n_types = int(n_types if n_types is not None else max(OBS_TYPES.values()) + 1)
        self.stats = stats
        self.support_aware = bool(support_aware)
        self.n_support_types = int(
            n_support_types if n_support_types is not None else max(SUPPORT_TYPES.values()) + 1
        )
        self.n_sites = int(n_sites)
        self.n_sources = int(n_sources)

    @property
    def token_dim(self) -> int:
        legacy = 4 + self.n_types + 3
        if not self.support_aware:
            return legacy
        return legacy + self.n_support_types + 7 + self.n_sites + self.n_sources

    @staticmethod
    def _one_hot(ids: np.ndarray, size: int) -> np.ndarray:
        out = np.zeros((len(ids), int(size)), dtype=np.float32)
        valid = (ids >= 0) & (ids < int(size))
        out[np.arange(len(ids))[valid], ids[valid]] = 1.0
        return out

    def fit_from_grid(self, grid: dict) -> "ObservationTokenizer":
        self.stats = TokenizerStats(
            xyz_max=(float(np.max(grid["x"])), float(np.max(grid["y"])), float(np.max(grid["z"]))),
        )
        return self

    def encode_numpy(self, observations: ObservationTable) -> np.ndarray:
        if self.stats is None:
            xyz_max = tuple(np.maximum(np.nanmax(observations.coords, axis=0), 1.0).tolist())
            self.stats = TokenizerStats(xyz_max=xyz_max)
        xyz_max_arr = np.asarray(self.stats.xyz_max, dtype=np.float32)
        coords = observations.coords / np.maximum(xyz_max_arr[None, :], 1e-6)
        times = np.nan_to_num(observations.times.astype(np.float32), nan=0.0)[:, None]
        one_hot = self._one_hot(observations.type_ids, self.n_types)
        values = observations.values[:, None] / self.stats.value_scale
        if self.support_aware:
            sigma = np.log(np.maximum(observations.sigma, 1.0e-6))[:, None]
        else:
            sigma = observations.sigma[:, None] / max(self.stats.sigma_scale, 1e-6)
        mask = observations.mask.astype(np.float32)[:, None]
        base = [coords, times, one_hot, values, sigma, mask]
        if not self.support_aware:
            return np.concatenate(base, axis=1).astype(np.float32)
        extent = observations.support_extent / np.maximum(xyz_max_arr[None, :], 1e-6)
        orientation = observations.orientation.astype(np.float32)
        norms = np.linalg.norm(orientation, axis=1, keepdims=True)
        orientation = np.divide(
            orientation,
            np.maximum(norms, 1.0e-6),
            out=np.zeros_like(orientation),
            where=norms > 0,
        )
        support_type = self._one_hot(observations.support_type_ids, self.n_support_types)
        quality = np.clip(observations.quality, 0.0, 1.0)[:, None]
        site = self._one_hot(observations.site_ids, self.n_sites)
        source = self._one_hot(observations.source_ids, self.n_sources)
        return np.concatenate(
            base + [support_type, extent, orientation, quality, site, source], axis=1
        ).astype(np.float32)

    def encode_torch(self, observations: ObservationTable, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.as_tensor(self.encode_numpy(observations), dtype=torch.float32, device=device)


def collate_observation_tokens(batch: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.shape[0] for x in batch)
    dim = batch[0].shape[1]
    tokens = np.zeros((len(batch), max_len, dim), dtype=np.float32)
    mask = np.ones((len(batch), max_len), dtype=bool)
    for i, item in enumerate(batch):
        tokens[i, : item.shape[0]] = item
        mask[i, : item.shape[0]] = False
    return torch.from_numpy(tokens), torch.from_numpy(mask)


class ObservationGraphBuilder:
    """Build local spatial neighborhoods for sparse multi-source observations.

    The dense ``attention_mask`` follows PyTorch Transformer semantics: ``True`` blocks
    attention between a query-key pair, while ``False`` leaves the pair visible.
    """

    def __init__(
        self,
        k_neighbors: int = 12,
        length_scale_xyz: tuple[float, float, float] = (0.15, 0.15, 0.25),
        same_type_bonus: float = 0.20,
        symmetric: bool = True,
        stats: TokenizerStats | None = None,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.length_scale_xyz = tuple(float(v) for v in length_scale_xyz)
        self.same_type_bonus = float(same_type_bonus)
        self.symmetric = bool(symmetric)
        self.stats = stats

    def fit_from_grid(self, grid: dict) -> "ObservationGraphBuilder":
        self.stats = TokenizerStats(
            xyz_max=(float(np.max(grid["x"])), float(np.max(grid["y"])), float(np.max(grid["z"]))),
        )
        return self

    def normalize_coords(self, observations: ObservationTable) -> np.ndarray:
        if self.stats is None:
            xyz_max = tuple(np.maximum(np.nanmax(observations.coords, axis=0), 1.0).tolist())
            self.stats = TokenizerStats(xyz_max=xyz_max)
        xyz_max = np.asarray(self.stats.xyz_max, dtype=np.float32)
        return (observations.coords / np.maximum(xyz_max[None, :], 1e-6)).astype(np.float32)

    def build(self, observations: ObservationTable) -> ObservationGraph:
        n_obs = observations.n_obs
        coords = self.normalize_coords(observations)
        if n_obs == 0:
            return ObservationGraph(
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_weight=np.zeros((0,), dtype=np.float32),
                attention_mask=np.zeros((0, 0), dtype=bool),
                normalized_coords=coords,
            )

        scales = np.asarray(self.length_scale_xyz, dtype=np.float32)
        scales = np.maximum(scales, 1e-6)
        delta = (coords[:, None, :] - coords[None, :, :]) / scales[None, None, :]
        distances = np.sqrt(np.sum(delta * delta, axis=-1, dtype=np.float32))
        np.fill_diagonal(distances, np.inf)

        k = max(0, min(self.k_neighbors, n_obs - 1))
        allowed = np.eye(n_obs, dtype=bool)
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        if k > 0:
            for src in range(n_obs):
                nearest = np.argpartition(distances[src], kth=k - 1)[:k]
                nearest = nearest[np.argsort(distances[src, nearest])]
                for dst in nearest:
                    weight = float(np.exp(-distances[src, dst]))
                    if observations.type_ids[src] == observations.type_ids[dst]:
                        weight *= 1.0 + self.same_type_bonus
                    edges.append((src, int(dst)))
                    weights.append(weight)
                    allowed[src, dst] = True
                    if self.symmetric:
                        allowed[dst, src] = True

        if self.symmetric and edges:
            edge_set = {(src, dst) for src, dst in edges}
            for src, dst in list(edge_set):
                if (dst, src) not in edge_set:
                    edges.append((dst, src))
                    weights.append(weights[edges.index((src, dst))])

        edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.asarray(weights, dtype=np.float32)
        return ObservationGraph(
            edge_index=edge_index,
            edge_weight=edge_weight,
            attention_mask=~allowed,
            normalized_coords=coords,
        )


def build_observation_attention_mask(
    config: dict,
    grid: dict,
    observations: ObservationTable,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    graph_cfg = config.get("observation_graph", {})
    if not bool(graph_cfg.get("enabled", False)):
        return None
    builder = ObservationGraphBuilder(
        k_neighbors=int(graph_cfg.get("k_neighbors", 12)),
        length_scale_xyz=tuple(float(v) for v in graph_cfg.get("length_scale_xyz", (0.15, 0.15, 0.25))),
        same_type_bonus=float(graph_cfg.get("same_type_bonus", 0.20)),
        symmetric=bool(graph_cfg.get("symmetric", True)),
    ).fit_from_grid(grid)
    graph = builder.build(observations)
    return torch.as_tensor(graph.attention_mask, dtype=torch.bool, device=device)
