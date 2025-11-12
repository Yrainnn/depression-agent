from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def get_prompt(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.txt"
    if not path.exists():  # pragma: no cover - developer error guard
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
