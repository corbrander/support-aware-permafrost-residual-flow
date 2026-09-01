from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cold_recon.training.train_autoencoder import train_autoencoder
from cold_recon.training.train_diffusion import train_diffusion
from cold_recon.training.train_implicit import train_implicit_model
from cold_recon.utils.config import ensure_dirs, load_config


VALID_STAGES = ("implicit", "autoencoder", "diffusion")


@dataclass(frozen=True)
class JointTrainingStage:
    name: str
    status: str
    checkpoint: str = ""
    prediction: str = ""
    epochs: int | None = None


def parse_stages(stages: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if stages is None:
        parsed = VALID_STAGES
    elif isinstance(stages, str):
        parsed = tuple(part.strip() for part in stages.split(",") if part.strip())
    else:
        parsed = tuple(str(part).strip() for part in stages if str(part).strip())
    unknown = [stage for stage in parsed if stage not in VALID_STAGES]
    if unknown:
        raise ValueError(f"Unknown joint training stage(s): {', '.join(unknown)}. Valid stages: {', '.join(VALID_STAGES)}")
    return parsed


def _stage_epochs(config: dict[str, Any], stage: str, overrides: dict[str, int | None]) -> int | None:
    if overrides.get(stage) is not None:
        return int(overrides[stage])
    if stage == "implicit":
        return int(config.get("training", {}).get("epochs", 100))
    if stage == "autoencoder":
        return int(config.get("autoencoder", {}).get("epochs", 80))
    if stage == "diffusion":
        return int(config.get("diffusion", {}).get("epochs", 200))
    return None


def _write_summary(path: Path, rows: list[JointTrainingStage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "status", "checkpoint", "prediction", "epochs"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.name,
                    "status": row.status,
                    "checkpoint": row.checkpoint,
                    "prediction": row.prediction,
                    "epochs": "" if row.epochs is None else row.epochs,
                }
            )


def train_joint_pipeline(
    config: dict[str, Any],
    stages: str | list[str] | tuple[str, ...] | None = None,
    *,
    dry_run: bool = False,
    device: str | None = None,
    implicit_epochs: int | None = None,
    autoencoder_epochs: int | None = None,
    diffusion_epochs: int | None = None,
    diffusion_samples: int | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run or plan the staged COLD-Recon training pipeline.

    This is a practical joint-training orchestrator: each stage uses the
    validated local trainer for that model family and records the artifacts it
    produced. Full end-to-end co-optimization can be layered on top of this
    stable stage contract later.
    """
    selected = parse_stages(stages)
    overrides = {
        "implicit": implicit_epochs,
        "autoencoder": autoencoder_epochs,
        "diffusion": diffusion_epochs,
    }
    rows: list[JointTrainingStage] = []
    if dry_run:
        for stage in selected:
            rows.append(JointTrainingStage(stage, "planned", epochs=_stage_epochs(config, stage, overrides)))
    else:
        for stage in selected:
            epochs = _stage_epochs(config, stage, overrides)
            if stage == "implicit":
                result = train_implicit_model(config, epochs=epochs, device=device)
                rows.append(
                    JointTrainingStage(
                        stage,
                        "completed",
                        checkpoint=str(result.get("checkpoint", "")),
                        prediction=str(result.get("prediction_path", "")),
                        epochs=epochs,
                    )
                )
            elif stage == "autoencoder":
                result = train_autoencoder(config, epochs=epochs, device=device)
                rows.append(
                    JointTrainingStage(
                        stage,
                        "completed",
                        checkpoint=str(result.get("checkpoint", "")),
                        prediction=str(result.get("reconstruction_path", "")),
                        epochs=epochs,
                    )
                )
            elif stage == "diffusion":
                result = train_diffusion(config, epochs=epochs, samples=diffusion_samples, device=device)
                rows.append(
                    JointTrainingStage(
                        stage,
                        "completed",
                        checkpoint=str(result.get("checkpoint", "")),
                        prediction=str(result.get("posterior_path", "")),
                        epochs=epochs,
                    )
                )
    out_path = Path(summary_path or Path(config["paths"]["tables_dir"]) / "joint_training_summary.csv")
    _write_summary(out_path, rows)
    return {"summary_path": out_path, "stages": rows, "dry_run": dry_run}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synth_default.yaml")
    parser.add_argument("--stages", default="implicit,autoencoder,diffusion")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--implicit-epochs", type=int, default=None)
    parser.add_argument("--autoencoder-epochs", type=int, default=None)
    parser.add_argument("--diffusion-epochs", type=int, default=None)
    parser.add_argument("--diffusion-samples", type=int, default=None)
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    result = train_joint_pipeline(
        config,
        args.stages,
        dry_run=args.dry_run,
        device=args.device,
        implicit_epochs=args.implicit_epochs,
        autoencoder_epochs=args.autoencoder_epochs,
        diffusion_epochs=args.diffusion_epochs,
        diffusion_samples=args.diffusion_samples,
        summary_path=args.summary_path,
    )
    print(f"joint_summary={result['summary_path']}")
    for stage in result["stages"]:
        print(f"{stage.name}: {stage.status} epochs={stage.epochs} checkpoint={stage.checkpoint} prediction={stage.prediction}")


if __name__ == "__main__":
    main()
