from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def plot_uncertainty_section(std_field: np.ndarray, out_path: str | Path, y_index: int, title: str = "Posterior std") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    im = ax.imshow(std_field[:, y_index, :].T, origin="upper", aspect="auto", cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
