from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.digital_human.service import generate_digital_human_video
from services.oss.client import oss_client
from services.tts.tts_adapter import TTSAdapter

LOGGER = logging.getLogger(__name__)


def build_media_payload(
    tts_adapter: TTSAdapter,
    sid: str,
    text: Optional[str],
    *,
    digital_human_enabled: bool = False,
) -> Dict[str, Any]:
    """Synthesize audio/video assets for the given text."""

    media: Dict[str, Any] = {}
    normalized = (text or "").strip()
    if not normalized:
        return media

    media["tts_text"] = normalized

    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    try:
        audio_path = tts_adapter.synthesize(sid, normalized)
    except Exception:  # pragma: no cover - defensive log guard
        LOGGER.exception("TTS synthesis failed for session %s", sid)
    else:
        if audio_path:
            media["tts_local_path"] = audio_path

            oss_instance = oss_client
            if oss_instance is not None and getattr(oss_instance, "enabled", False):
                try:
                    audio_url = oss_instance.store_artifact(
                        sid,
                        "tts/audio",
                        audio_path,
                        metadata={"source": "cosyvoice", "text_preview": normalized[:120]},
                    )
                except Exception:  # pragma: no cover - defensive log guard
                    LOGGER.exception(
                        "Failed to upload TTS audio for session %s", sid
                    )
            if not audio_url:
                audio_url = audio_path

            media["tts_url"] = audio_url
            media["media_type"] = "audio"

    if digital_human_enabled and audio_path:
        try:
            video_url = generate_digital_human_video(sid, audio_path)
        except Exception:  # pragma: no cover - defensive log guard
            LOGGER.exception(
                "Digital human generation failed for session %s", sid
            )
        else:
            if video_url:
                media["video_url"] = video_url
                media["media_type"] = "video"

    if "media_type" not in media:
        media["media_type"] = "text"

    return media
