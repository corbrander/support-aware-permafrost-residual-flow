from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cold_recon.baselines.unet3d import SPARSE_UNET_INPUT_CHANNELS, SmallUNet3D, build_sparse_observation_volume
from cold_recon.data.data_schema import load_sample_npz
from cold_recon.evaluation.metrics import synthetic_metrics
from cold_recon.training.volume_codec import sample_to_volume_tensor, volume_tensor_to_fields
from cold_recon.utils.config import ensure_dirs, load_config
from cold_recon.visualization.plot_sections import plot_truth_prediction_sections


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _loss(pred: torch.Tensor, target: torch.Tensor, facies: torch.Tensor, n_facies: int) -> tuple[torch.Tensor, dict[str, float]]:
    facies_loss = F.cross_entropy(pred[:, :n_facies], facies)
    eic_loss = F.mse_loss(torch.sigmoid(pred[:, n_facies]), target[:, n_facies])
    temp_loss = F.mse_loss(pred[:, n_facies + 1], target[:, n_facies + 1])
    unfrozen_loss = F.mse_loss(torch.sigmoid(pred[:, n_facies + 2]), target[:, n_facies + 2])
    rho_loss = F.mse_loss(pred[:, n_facies + 3], target[:, n_facies + 3])
    total = facies_loss + 4.0 * eic_loss + 0.5 * temp_loss + 2.0 * unfrozen_loss + 0.2 * rho_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "facies_loss": float(facies_loss.detach().cpu()),
        "eic_loss": float(eic_loss.detach().cpu()),
        "temperature_loss": float(temp_loss.detach().cpu()),
        "unfrozen_water_loss": float(unfrozen_loss.detach().cpu()),
        "log_resistivity_loss": float(rho_loss.detach().cpu()),
    }


def _decode_output(pred: torch.Tensor, n_facies: int) -> dict[str, np.ndarray]:
    decoded = pred.detach().cpu().clone()
    decoded[:, n_facies] = torch.sigmoid(decoded[:, n_facies])
    decoded[:, n_facies + 2] = torch.sigmoid(decoded[:, n_facies + 2])
    return volume_tensor_to_fields(decoded, n_facies=n_facies)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    unet_cfg = config.get("baseline_unet3d", {})
    n_facies = int(config["model"]["n_facies"])
    sample = load_sample_npz(args.sample or unet_cfg.get("sample_path", config["training"]["sample_path"]))
    device = _device(str(args.device or unet_cfg.get("device", config["training"].get("device", "cpu"))))
    epochs = int(args.epochs or unet_cfg.get("epochs", 32))
    base_channels = int(args.base_channels or unet_cfg.get("base_channels", 12))
    lr = float(args.lr or unet_cfg.get("lr", 0.001))

    condition = build_sparse_observation_volume(sample, n_facies=n_facies).to(device)
    target = sample_to_volume_tensor(sample, n_facies=n_facies).to(device)
    facies = torch.as_tensor(sample["fields"]["facies"].astype(np.int64), device=device).unsqueeze(0)
    model = SmallUNet3D(condition.shape[1], n_facies + 4, base=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(unet_cfg.get("weight_decay", 1e-6)))
    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = model(condition)
        loss, parts = _loss(pred, target, facies, n_facies=n_facies)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(unet_cfg.get("grad_clip", 1.0)))
        optimizer.step()
        history.append({"epoch": epoch, **parts})

    model.eval()
    with torch.no_grad():
        pred_tensor = model(condition)
    pred = _decode_output(pred_tensor, n_facies=n_facies)
    pred["input_channel_names"] = np.asarray(SPARSE_UNET_INPUT_CHANNELS)
    pred_dir = Path(config["paths"]["predictions_dir"])
    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    table_dir = Path(config["paths"]["tables_dir"])
    fig_dir = Path(config["paths"]["figures_dir"])
    pred_path = Path(unet_cfg.get("prediction_path", pred_dir / "baseline_unet3d.npz"))
    ckpt_path = Path(unet_cfg.get("checkpoint", ckpt_dir / "baseline_unet3d.pt"))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **pred)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_channels": list(SPARSE_UNET_INPUT_CHANNELS),
            "n_facies": n_facies,
            "base_channels": base_channels,
            "epochs": epochs,
            "lr": lr,
        },
        ckpt_path,
    )
    metrics = synthetic_metrics(
        pred,
        sample["fields"],
        sample["grid"]["z"],
        n_facies=n_facies,
        ice_threshold=float(config["evaluation"]["ice_rich_threshold"]),
    )
    metrics_path = table_dir / "baseline_unet3d_metrics.csv"
    history_path = table_dir / "baseline_unet3d_history.csv"
    _write_rows(metrics_path, [{"model": "SparseUNet3D", **metrics}])
    _write_rows(history_path, history)
    fig_path = fig_dir / "baseline_unet3d_sections.png"
    plot_truth_prediction_sections(
        sample["fields"],
        pred,
        fig_path,
        int(config["evaluation"]["section_y_index"]),
        "Sparse observation 3D U-Net baseline",
    )
    print(f"checkpoint={ckpt_path}")
    print(f"prediction={pred_path}")
    print(f"metrics={metrics_path}")
    print(f"history={history_path}")
    print(f"figure={fig_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
