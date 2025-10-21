import subprocess
from pathlib import Path
import uuid

def merge_audio_video(video_path: Path, audio_path: Path) -> Path:
    merged_path = video_path.parent / f"{uuid.uuid4().hex}_merged.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        str(merged_path)
    ], check=True)
    return merged_path
