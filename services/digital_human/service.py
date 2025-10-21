"""High level API for generating digital human videos."""

from __future__ import annotations

from pathlib import Path

from .generator import run_pipeline
from .merger import merge_audio_video
from .uploader import upload_to_oss


def generate_digital_human_video(sid: str, audio_path: str) -> str:
    """Generate a digital human video from audio and upload it to OSS."""
    audio_path = Path(audio_path).resolve()
    print(f"[DigitalHuman] 输入音频: {audio_path}")

    silent_video = run_pipeline(str(audio_path))
    print(f"[DigitalHuman] 推理输出: {silent_video}")

    merged_video = merge_audio_video(silent_video, audio_path)
    print(f"[DigitalHuman] 合成输出: {merged_video}")

    url = upload_to_oss(merged_video, sid)
    print(f"[DigitalHuman] 上传完成: {url}")
    return url
