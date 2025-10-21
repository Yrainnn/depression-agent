import subprocess
from pathlib import Path
import uuid

MODEL_ROOT = Path("/home/yitaoWang/Ultralight-Digital-Human")
CHECKPOINT_PATH = MODEL_ROOT / "checkpoint/185.pth"
DATASET_DIR = MODEL_ROOT / "data_dir"
OUTPUT_DIR = MODEL_ROOT / "output"
ASSETS_DIR = MODEL_ROOT / "assets/train"

def ensure_16k_mono(input_path: Path) -> Path:
    """保证音频为16kHz单声道"""
    fixed_path = input_path.parent / f"{input_path.stem}_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", str(fixed_path)],
        check=True
    )
    return fixed_path

def run_pipeline(audio_path: str) -> Path:
    """执行完整推理流程并输出视频路径"""
    audio_path = Path(audio_path).resolve()
    audio_16k = ensure_16k_mono(audio_path)

    # 🧹 Step0: 清理同名旧特征文件（避免 wenet 输出冲突）
    old_feat = MODEL_ROOT / f"assets/train/{audio_16k.stem}_wenet.npy"
    if old_feat.exists():
        old_feat.unlink()
    
    # Step1: 声学特征提取
    subprocess.run(["python", str(MODEL_ROOT / "wenet_infer.py"), str(audio_16k)],
                   cwd=MODEL_ROOT, check=True)

    # Step2: 数字人视频推理
    audio_feat = MODEL_ROOT / f"assets/train/{audio_16k.stem}_wenet.npy"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_video = OUTPUT_DIR / f"{uuid.uuid4().hex}_silent.mp4"

    subprocess.run([
        "python", str(MODEL_ROOT / "inference.py"),
        "--asr", "wenet",
        "--dataset", str(DATASET_DIR),
        "--audio_feat", str(audio_feat),
        "--save_path", str(output_video),
        "--checkpoint", str(CHECKPOINT_PATH)
    ], cwd=MODEL_ROOT, check=True)

    return output_video
