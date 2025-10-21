"""Utilities for generating digital human videos from audio."""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

MODEL_ROOT = Path("/home/yitaoWang/Ultralight-Digital-Human")
CHECKPOINT_PATH = MODEL_ROOT / "checkpoint/185.pth"
DATASET_DIR = MODEL_ROOT / "data_dir"
OUTPUT_DIR = MODEL_ROOT / "output"
ASSETS_DIR = MODEL_ROOT / "assets/train"


def ensure_16k_mono(input_path: Path) -> Path:
    """Ensure the audio file is converted to 16 kHz mono WAV."""
    fixed_path = input_path.parent / f"{input_path.stem}_16k.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(fixed_path),
        ],
        check=True,
    )
    return fixed_path


def run_pipeline(audio_path: str) -> Path:
    """Run the digital human inference pipeline and return the output video path."""
    audio_path = Path(audio_path).resolve()
    audio_16k = ensure_16k_mono(audio_path)

    subprocess.run(
        ["python", str(MODEL_ROOT / "wenet_infer.py"), str(audio_16k)],
        cwd=MODEL_ROOT,
        check=True,
    )

    audio_feat = ASSETS_DIR / f"{audio_16k.stem}_wenet.npy"
    output_video = OUTPUT_DIR / f"{uuid.uuid4().hex}_silent.mp4"

    subprocess.run(
        [
            "python",
            str(MODEL_ROOT / "inference.py"),
            "--asr",
            "wenet",
            "--dataset",
            str(DATASET_DIR),
            "--audio_feat",
            str(audio_feat),
            "--save_path",
            str(output_video),
            "--checkpoint",
            str(CHECKPOINT_PATH),
        ],
        cwd=MODEL_ROOT,
        check=True,
    )

    return output_video
