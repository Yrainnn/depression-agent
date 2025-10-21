from pathlib import Path

from .generator import run_pipeline
from .merger import merge_audio_video
from .uploader import upload_to_oss

def generate_digital_human_video(sid: str, audio_path: str) -> str:
    """
    主入口：音频→16kHz转换→推理→合成→上传OSS→返回URL
    """
    audio_path = Path(audio_path).resolve()
    print(f"[DigitalHuman] 输入音频: {audio_path}")

    # 推理生成无声视频
    silent_video = run_pipeline(str(audio_path))
    print(f"[DigitalHuman] 推理输出: {silent_video}")

    # 合成音视频
    merged_video = merge_audio_video(silent_video, audio_path)
    print(f"[DigitalHuman] 合成输出: {merged_video}")

    # 上传到OSS
    url = upload_to_oss(merged_video, sid)
    print(f"[DigitalHuman] 上传完成: {url}")
    return url
