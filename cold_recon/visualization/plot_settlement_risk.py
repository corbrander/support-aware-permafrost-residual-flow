from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def plot_settlement_map(settlement: np.ndarray, out_path: str | Path, title: str = "Settlement potential") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(settlement.T, origin="lower", cmap="inferno")
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
