from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
import torch


@dataclass(frozen=True)
class CheckpointSpec:
    model: str
    role: str
    path: str
    component_key: str
    obs_key: str | None = None
    notes: str = ""


DEFAULT_CHECKPOINT_SPECS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        "COLDReconImplicit",
        "coordinate implicit field",
        "outputs/checkpoints/implicit_mlp.pt",
        "model_state",
        notes="observation Transformer + surface encoder + Fourier-coordinate MLP",
    ),
    CheckpointSpec(
        "Autoencoder3D",
        "voxel field autoencoder",
        "outputs/checkpoints/autoencoder3d.pt",
        "model_state",
        notes="compresses 11-channel cryostratigraphic volumes to latent grids",
    ),
    CheckpointSpec(
        "COLDReconLatentDiffusion",
        "conditional latent diffusion denoiser",
        "outputs/checkpoints/latent_diffusion.pt",
        "denoiser_state",
        obs_key="obs_encoder_state",
        notes="U-Net denoiser conditioned by sparse observation tokens",
    ),
    CheckpointSpec(
        "COLDReconFNOOperatorDiffusion",
        "conditional neural-operator denoiser",
        "outputs/checkpoints/fno_operator_diffusion.pt",
        "denoiser_state",
        obs_key="obs_encoder_state",
        notes="FNO-Transformer hybrid denoiser with low-mode 3D spectral convolutions",
    ),
    CheckpointSpec(
        "COLDReconRectifiedFlow",
        "conditional rectified-flow velocity model",
        "outputs/checkpoints/rectified_flow.pt",
        "velocity_state",
        obs_key="obs_encoder_state",
        notes="flow-matching generative alternative using the same latent conditioning",
    ),
    CheckpointSpec(
        "SparseUNet3D",
        "deterministic sparse-observation baseline",
        "outputs/checkpoints/baseline_unet3d.pt",
        "model_state",
        notes="rasterized observation volume + surface covariates",
    ),
)


def count_state_dict_parameters(state_dict: dict[str, Any]) -> int:
    return int(sum(int(value.numel()) for value in state_dict.values() if hasattr(value, "numel")))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def summarize_architecture(
    root: str | Path = ".",
    specs: tuple[CheckpointSpec, ...] = DEFAULT_CHECKPOINT_SPECS,
) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = root / spec.path
        if not path.exists():
            rows.append(
                {
                    "model": spec.model,
                    "role": spec.role,
                    "checkpoint": spec.path,
                    "component_params": 0,
                    "obs_encoder_params": 0,
                    "total_params": 0,
                    "latent_shape": "",
                    "n_facies": "",
                    "status": "missing",
                    "notes": spec.notes,
                }
            )
            continue
        ckpt = load_checkpoint(path)
        component_params = count_state_dict_parameters(ckpt.get(spec.component_key, {}))
        obs_params = count_state_dict_parameters(ckpt.get(spec.obs_key, {})) if spec.obs_key else 0
        rows.append(
            {
                "model": spec.model,
                "role": spec.role,
                "checkpoint": spec.path,
                "component_params": component_params,
                "obs_encoder_params": obs_params,
                "total_params": component_params + obs_params,
                "latent_shape": "x".join(str(v) for v in ckpt.get("latent_shape", ())),
                "n_facies": ckpt.get("n_facies", ""),
                "status": "ok",
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def _box(ax, xy: tuple[float, float], text: str, width: float = 2.55, height: float = 0.82, color: str = "#f7f7f7") -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.1,
        edgecolor="#333333",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2.0, y + height / 2.0, text, ha="center", va="center", fontsize=9, wrap=True)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#333333") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color=color))


def plot_algorithm_schematic(out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 6.8), constrained_layout=True)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    ax.set_title("COLD-Recon physics-guided conditional diffusion neural operator", fontsize=14, pad=12)

    _box(ax, (0.35, 5.35), "Sparse multi-source observations\nborehole / ERT / NMR / ALT", color="#e8f1fb")
    _box(ax, (0.35, 3.95), "Surface and environmental priors\nterrain / soil / climate / imagery", color="#e8f1fb")
    _box(ax, (3.15, 5.35), "Observation tokenizer + kNN graph\n(x,y,z,t,type,value,sigma,mask)", color="#fdf4df")
    _box(ax, (3.15, 3.95), "Surface encoder\nlocal covariates to features", color="#fdf4df")
    _box(ax, (5.95, 5.35), "Obs Transformer context\ncondition vector c", color="#fdf4df")
    _box(ax, (5.95, 3.95), "3D autoencoder latent grid\nz = Enc(M)", color="#eef7e8")
    _box(ax, (8.75, 4.65), "Conditional generator\nlatent diffusion / FNO denoiser / rectified flow", width=3.0, color="#eef7e8")
    _box(ax, (8.75, 2.75), "Physics guidance\nunfrozen water + resistivity + heat residual", width=3.0, color="#fde9e7")
    _box(ax, (8.75, 0.95), "Posterior ensemble\nfacies, EIC, T, theta_u, rho, uncertainty", width=3.0, color="#e9e6f7")
    _box(ax, (0.35, 1.1), "Validation\nsynthetic truth + USGS holdout + calibration", width=5.35, color="#f0f0f0")

    _arrow(ax, (2.9, 5.76), (3.15, 5.76))
    _arrow(ax, (2.9, 4.36), (3.15, 4.36))
    _arrow(ax, (5.7, 5.76), (5.95, 5.76))
    _arrow(ax, (5.7, 4.36), (5.95, 4.36))
    _arrow(ax, (8.5, 5.76), (8.75, 5.25))
    _arrow(ax, (8.5, 4.36), (8.75, 4.95))
    _arrow(ax, (10.25, 4.65), (10.25, 3.57), color="#9b2f2f")
    _arrow(ax, (10.25, 2.75), (10.25, 1.77))
    _arrow(ax, (5.7, 1.5), (8.75, 1.35))
    ax.text(7.2, 6.35, "conditioning", fontsize=8, color="#333333")
    ax.text(12.05, 3.7, "loss / sampling guidance", fontsize=8, color="#9b2f2f", rotation=90, va="center")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return out_path
