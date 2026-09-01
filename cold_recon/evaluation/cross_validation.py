from __future__ import annotations

import numpy as np

from cold_recon.data.data_schema import ObservationTable


def split_observations(observations: ObservationTable, holdout_fraction: float = 0.2, seed: int = 0) -> tuple[ObservationTable, ObservationTable]:
    rng = np.random.default_rng(seed)
    idx = np.arange(observations.n_obs)
    rng.shuffle(idx)
    n_hold = int(round(len(idx) * holdout_fraction))
    hold_idx = idx[:n_hold]
    train_idx = idx[n_hold:]
    return observations.subset(train_idx), observations.subset(hold_idx)

