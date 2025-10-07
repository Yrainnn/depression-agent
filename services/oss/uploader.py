"""Alibaba Cloud OSS uploader helpers."""

from __future__ import annotations

import datetime
import importlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

from packages.common.config import settings

LOGGER = logging.getLogger(__name__)


class OSSUploaderError(RuntimeError):
    """Raised when OSS uploads cannot be performed."""


@dataclass
class _OSSResources:
    module: object
    client: object
    put_request_cls: type
    get_request_cls: type
    bucket: str
    key_prefix: str


def _load_sdk() -> Optional[object]:
    """Try to import the Alibaba Cloud OSS SDK module."""

    try:
        return importlib.import_module("alibabacloud_oss_v2")
    except ModuleNotFoundError:
        LOGGER.warning("alibabacloud_oss_v2 is not installed; OSS uploads disabled")
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.exception("Failed to import alibabacloud_oss_v2")
    return None


def _normalise_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else f"{prefix}/"


class OSSUploader:
    """Lightweight helper for uploading artefacts to Alibaba Cloud OSS."""

    def __init__(
        self,
        *,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
        key_prefix: Optional[str] = None,
    ) -> None:
        self._resources: Optional[_OSSResources] = None
        module = _load_sdk()
        if module is None:
            return

        resolved_bucket = bucket or settings.OSS_BUCKET
        resolved_region = region or settings.OSS_REGION
        resolved_endpoint = endpoint or settings.OSS_ENDPOINT
        resolved_prefix = _normalise_prefix(
            key_prefix if key_prefix is not None else settings.OSS_KEY_PREFIX
        )

        if not resolved_bucket or not resolved_region:
            LOGGER.warning(
                "OSS bucket/region not configured; uploads disabled (bucket=%s, region=%s)",
                resolved_bucket,
                resolved_region,
            )
            return

        cfg = module.config.load_default()
        cfg.credentials_provider = module.credentials.EnvironmentVariableCredentialsProvider()
        cfg.region = resolved_region
        if resolved_endpoint:
            cfg.endpoint = resolved_endpoint

        try:
            client = module.Client(cfg)
            put_cls = getattr(module, "PutObjectRequest")
            get_cls = getattr(module, "GetObjectRequest")
        except AttributeError as exc:  # pragma: no cover - SDK contract guard
            LOGGER.exception("Unexpected OSS SDK layout: %s", exc)
            return

        self._resources = _OSSResources(
            module=module,
            client=client,
            put_request_cls=put_cls,
            get_request_cls=get_cls,
            bucket=resolved_bucket,
            key_prefix=resolved_prefix,
        )

    @property
    def enabled(self) -> bool:
        """Whether the uploader is ready to interact with OSS."""

        return self._resources is not None

    def upload_file(self, local_file_path: str, *, oss_key_prefix: Optional[str] = None) -> str:
        """Upload ``local_file_path`` to OSS and return the object key."""

        resources = self._require_resources()
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file does not exist: {local_file_path}")

        prefix = _normalise_prefix(oss_key_prefix or resources.key_prefix)
        file_name = os.path.basename(local_file_path)
        oss_key = f"{prefix}{file_name}" if prefix else file_name

        with open(local_file_path, "rb") as handle:
            request = resources.put_request_cls(
                bucket=resources.bucket,
                key=oss_key,
                body=handle,
            )
            resources.client.put_object(request)

        LOGGER.info("Uploaded %s to OSS as %s", local_file_path, oss_key)
        return oss_key

    def get_presigned_url(self, oss_key: str, *, expires_minutes: int = 60) -> str:
        """Return a presigned URL for ``oss_key``."""

        resources = self._require_resources()
        request = resources.get_request_cls(
            bucket=resources.bucket,
            key=oss_key,
        )
        result = resources.client.presign(request, expires=datetime.timedelta(minutes=expires_minutes))
        return result.url

    # ------------------------------------------------------------------
    def _require_resources(self) -> _OSSResources:
        if self._resources is None:
            raise OSSUploaderError("OSS uploader is not configured or the SDK is unavailable")
        return self._resources

