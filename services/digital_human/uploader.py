from pathlib import Path
from typing import Optional

from services.oss.client import OSSClient, oss_client as default_oss_client

def upload_to_oss(
    file_path: Path,
    sid: str,
    oss_dir: str = "digital-human/videos",
    oss_client: OSSClient = default_oss_client,
) -> str:
    """上传视频到OSS并返回公网URL"""
    if oss_client is None or not getattr(oss_client, "enabled", False):
        raise RuntimeError("OSS client 未启用")

    url: Optional[str] = oss_client.store_artifact(
        sid,
        oss_dir,
        file_path,
        metadata={"type": "digital_human"},
    )
    if not url:
        raise RuntimeError("视频上传失败")
    return url
