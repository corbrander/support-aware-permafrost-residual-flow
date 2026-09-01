from __future__ import annotations

from pathlib import Path

from cold_recon.visualization.plot_sections import plot_truth_prediction_sections


def make_synthetic_summary_figure(sample: dict, pred: dict | None, out_dir: str | Path, y_index: int) -> Path:
    out = Path(out_dir) / "figure_synthetic_summary.png"
    plot_truth_prediction_sections(sample["fields"], pred, out, y_index=y_index, title="Synthetic cryostratigraphy reconstruction")
    return out

