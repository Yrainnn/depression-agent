"""Combine generated video with the original audio."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path


def merge_audio_video(video_path: Path, audio_path: Path) -> Path:
    """Merge the silent video with audio and return the merged video path."""
    merged_path = video_path.parent / f"{uuid.uuid4().hex}_merged.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            str(merged_path),
        ],
        check=True,
    )
    return merged_path
