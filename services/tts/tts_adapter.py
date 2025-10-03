import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


def synthesize(text: str, *, voice: Optional[str] = None) -> None:
    """Placeholder for future CoSyVoice integration."""

    LOGGER.info("[TTS:stub] Synthesize request received", extra={"voice": voice, "text": text})
