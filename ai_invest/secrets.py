from __future__ import annotations

import os
from pathlib import Path

from .config import ROOT


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}
    return values


def load_openai_api_key() -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")

    return _read_env_file(ROOT / "openai.env").get("OPENAI_API_KEY")


def load_telegram_credentials() -> tuple[str | None, str | None]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return token, chat_id

    env_values = _read_env_file(ROOT / "telegram.env")
    return token or env_values.get("TELEGRAM_BOT_TOKEN"), chat_id or env_values.get("TELEGRAM_CHAT_ID")
