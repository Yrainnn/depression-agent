from __future__ import annotations

from typing import List, Optional


def transcribe(text: Optional[str] = None, audio_ref: Optional[str] = None) -> List[dict]:
    """Return ASR segments for provided text or audio reference.

    This is a lightweight stub that directly maps input text into a single
    transcript segment. Audio support is reserved for future TingWu
    integration.
    """

    if text:
        return [
            {
                "utt_id": "u1",
                "text": text,
                "speaker": "patient",
                "ts": [0, 3],
                "conf": 0.95,
            }
        ]

    # Placeholder for upcoming TingWu ASR support.
    if audio_ref:
        return []

    return []
