"""Upload digital human artifacts to OSS."""

from __future__ import annotations

from pathlib import Path

from services.oss.client import OSSClient, oss_client as default_oss_client


def upload_to_oss(
    file_path: Path,
    sid: str,
    oss_dir: str = "digital-human/videos",
    oss_client: OSSClient = default_oss_client,
) -> str:
    """Upload the video to OSS and return the public URL."""
    if oss_client is None or not getattr(oss_client, "enabled", False):
        raise RuntimeError("OSS client 未启用")
    return oss_client.store_artifact(
        sid,
        oss_dir,
        file_path,
        metadata={"type": "digital_human"},
    )
