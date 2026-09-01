from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def plot_truth_prediction_sections(
    truth: dict[str, np.ndarray],
    pred: dict[str, np.ndarray] | None,
    out_path: str | Path,
    y_index: int,
    title: str = "COLD-Recon synthetic section",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, np.ndarray, str]] = [
        ("Truth facies", truth["facies"][:, y_index, :].T, "tab20"),
        ("Truth EIC", truth["eic"][:, y_index, :].T, "viridis"),
        ("Truth T", truth["temperature"][:, y_index, :].T, "coolwarm"),
    ]
    if pred is not None:
        if "facies" in pred:
            panels.append(("Pred facies", pred["facies"][:, y_index, :].T, "tab20"))
        if "eic" in pred:
            panels.append(("Pred EIC", pred["eic"][:, y_index, :].T, "viridis"))
            panels.append(("EIC abs error", np.abs(pred["eic"] - truth["eic"])[:, y_index, :].T, "magma"))
        if "temperature" in pred:
            panels.append(("Pred T", pred["temperature"][:, y_index, :].T, "coolwarm"))
            panels.append(("T abs error", np.abs(pred["temperature"] - truth["temperature"])[:, y_index, :].T, "magma"))
    n = len(panels)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False, constrained_layout=True)
    fig.suptitle(title)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, arr, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="upper", aspect="auto", cmap=cmap)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        ax.axis("on")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
