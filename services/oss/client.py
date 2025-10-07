"""Aliyun OSS upload helper."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from packages.common.config import settings

try:  # pragma: no cover - optional dependency
    import oss2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    oss2 = None  # type: ignore

try:  # pragma: no cover - optional import during bootstrap
    from services.store.repository import repository
except Exception:  # pragma: no cover - runtime guard
    repository = None  # type: ignore


LOGGER = logging.getLogger(__name__)


class OSSClient:
    """Thin wrapper around Aliyun OSS uploads."""

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        bucket: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        prefix: Optional[str] = None,
        public_base_url: Optional[str] = None,
        repo: Any = None,
        bucket_factory: Optional[Any] = None,
    ) -> None:
        self.endpoint = endpoint or settings.OSS_ENDPOINT
        self.bucket_name = bucket or settings.OSS_BUCKET
        self.access_key_id = access_key_id or settings.OSS_ACCESS_KEY_ID
        self.access_key_secret = access_key_secret or settings.OSS_ACCESS_KEY_SECRET
        self.prefix = (prefix or settings.OSS_PREFIX or "").strip("/")
        self.public_base_url = public_base_url or settings.OSS_BASE_URL
        self.repository = repo or repository

        self._bucket_factory = bucket_factory
        self._bucket = None
        self.enabled = (
            bool(oss2)
            and bool(self.endpoint)
            and bool(self.bucket_name)
            and bool(self.access_key_id)
            and bool(self.access_key_secret)
        )

        if self.enabled:
            try:
                if self._bucket_factory is not None:
                    self._bucket = self._bucket_factory(
                        self.endpoint,
                        self.bucket_name,
                        self.access_key_id,
                        self.access_key_secret,
                    )
                else:
                    auth = oss2.Auth(self.access_key_id, self.access_key_secret)  # type: ignore[arg-type]
                    self._bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)  # type: ignore[misc]
            except Exception:
                LOGGER.exception("Failed to initialise OSS bucket")
                self._bucket = None
                self.enabled = False
        else:
            LOGGER.debug(
                "OSS disabled (module loaded: %s, endpoint: %s, bucket: %s)",
                bool(oss2),
                bool(self.endpoint),
                bool(self.bucket_name),
            )

    # ------------------------------------------------------------------
    def store_artifact(
        self,
        sid: str,
        category: str,
        path: Union[str, Path],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Upload *path* to OSS and persist metadata. Return public URL or None."""

        if not self.enabled or self._bucket is None:
            return None

        file_path = Path(path)
        if not file_path.exists():
            LOGGER.warning("OSS upload skipped for missing file: %s", file_path)
            return None

        object_name = self._make_object_name(sid, category, file_path.name)
        try:
            self._bucket.put_object_from_file(object_name, str(file_path))  # type: ignore[operator]
        except Exception:
            LOGGER.exception("Failed to upload %s to OSS", object_name)
            return None

        url = self._build_public_url(object_name)
        if self.repository is not None and url:
            reference: Dict[str, Any] = {
                "category": category,
                "object": object_name,
                "url": url,
                "local_path": str(file_path),
                "ts": int(time.time()),
            }
            if metadata:
                reference["meta"] = metadata
            try:
                self.repository.save_oss_reference(sid, reference)
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.exception("Failed to persist OSS reference for %s", sid)
        return url

    # ------------------------------------------------------------------
    def _make_object_name(self, sid: str, category: str, filename: str) -> str:
        parts = [self.prefix] if self.prefix else []
        if category:
            parts.append(category.strip("/"))
        parts.append(sid)
        parts.append(filename)
        return "/".join(part for part in parts if part)

    def _build_public_url(self, object_name: str) -> Optional[str]:
        if not object_name:
            return None
        if self.public_base_url:
            base = self.public_base_url.rstrip("/")
            return f"{base}/{object_name}"
        endpoint = (self.endpoint or "").rstrip("/")
        if not endpoint:
            return None
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            base = endpoint
        else:
            base = f"https://{self.bucket_name}.{endpoint}"
        return f"{base.rstrip('/')}/{object_name}"


oss_client = OSSClient()

