from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any]) -> None:
    for value in config.get("paths", {}).values():
        Path(value).mkdir(parents=True, exist_ok=True)


def get_path(config: dict[str, Any], key: str) -> Path:
    return Path(config["paths"][key])

