"""Minimal stub for python-dotenv to satisfy tests in offline environments."""

from pathlib import Path
from typing import Optional


def load_dotenv(dotenv_path: Optional[str | Path] = None, override: bool = False) -> bool:  # type: ignore[override]
    """Pretend to load environment variables from a .env file."""

    # The real implementation reads key/value pairs and injects them into os.environ.
    # For our test environment we simply return False to indicate no overrides occurred.
    return False


__all__ = ["load_dotenv"]
