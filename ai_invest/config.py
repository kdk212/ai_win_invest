from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config" / "default.yml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for path in [
        DATA_DIR / "cache",
        DATA_DIR / "recommendations",
        DATA_DIR / "backtests",
        DATA_DIR / "performance",
    ]:
        path.mkdir(parents=True, exist_ok=True)
