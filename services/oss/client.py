"""Aliyun OSS upload helper."""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    requests = None  # type: ignore

from packages.common.config import settings

try:  # pragma: no cover - optional dependency (new OSS SDK)
    import alibabacloud_oss_v2 as oss_v2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    oss_v2 = None  # type: ignore

try:  # pragma: no cover - optional dependency (legacy SDK)
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
        region: Optional[str] = None,
        presign_ttl: int = 3600,
    ) -> None:
        self.endpoint = (
            endpoint
            or getattr(settings, "OSS_ENDPOINT", None)
            or getattr(settings, "oss_endpoint", None)
        )
        self.bucket_name = (
            bucket
            or getattr(settings, "OSS_BUCKET", None)
            or getattr(settings, "oss_bucket", None)
        )
        self.region = region or getattr(settings, "OSS_REGION", None) or getattr(
            settings, "oss_region", None
        )
        self.access_key_id = (
            access_key_id
            or getattr(settings, "OSS_ACCESS_KEY_ID", None)
            or getattr(settings, "oss_access_key_id", None)
        )
        self.access_key_secret = (
            access_key_secret
            or getattr(settings, "OSS_ACCESS_KEY_SECRET", None)
            or getattr(settings, "oss_access_key_secret", None)
        )
        prefix_value = prefix
        if prefix_value is None:
            prefix_value = getattr(settings, "OSS_PREFIX", None)
        if prefix_value is None:
            prefix_value = getattr(settings, "oss_prefix", "")
        self.prefix = (prefix_value or "").strip("/")
        self.public_base_url = (
            public_base_url
            or getattr(settings, "OSS_BASE_URL", None)
            or getattr(settings, "oss_public_base_url", None)
        )
        self.repository = repo or repository
        self.presign_ttl = max(int(presign_ttl), 1)

        self._bucket_factory = bucket_factory
        self._bucket = None
        self._client_v2 = None
        self._mode: Optional[str] = None

        self.enabled = False

        if oss_v2 and self.endpoint and self.bucket_name:
            self._initialise_v2_client()

        if not self.enabled and oss2:
            self._initialise_legacy_client()

        if not self.enabled:
            LOGGER.debug(
                "OSS disabled (v2 available: %s, legacy available: %s, endpoint: %s, bucket: %s)",
                bool(oss_v2),
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

        if not self.enabled:
            return None

        file_path = Path(path)
        if not file_path.exists():
            LOGGER.warning("OSS upload skipped for missing file: %s", file_path)
            return None

        object_name = self._make_object_name(sid, category, file_path.name)
        uploaded = False

        if self._mode == "v2" and self._client_v2 is not None:
            uploaded = self._upload_v2(object_name, file_path)
        elif self._mode == "oss2" and self._bucket is not None:
            uploaded = self._upload_legacy(object_name, file_path)

        if not uploaded:
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

    # ------------------------------------------------------------------
    def _initialise_v2_client(self) -> None:
        if not oss_v2:
            return

        try:
            config_module = getattr(oss_v2, "config", None)
            credentials_module = getattr(oss_v2, "credentials", None)
            if config_module is None or credentials_module is None:
                return

            cfg = config_module.load_default()
            if self.region and hasattr(cfg, "region"):
                cfg.region = self.region
            if self.endpoint and hasattr(cfg, "endpoint"):
                cfg.endpoint = self.endpoint

            provider = None
            static_provider_cls = getattr(credentials_module, "StaticCredentialsProvider", None)
            if static_provider_cls and self.access_key_id and self.access_key_secret:
                provider = static_provider_cls(
                    access_key_id=self.access_key_id,
                    access_key_secret=self.access_key_secret,
                )

            if provider is None:
                env_provider_cls = getattr(
                    credentials_module, "EnvironmentVariableCredentialsProvider", None
                )
                if env_provider_cls:
                    provider = env_provider_cls()

            if provider is not None:
                cfg.credentials_provider = provider

            client_cls = getattr(oss_v2, "Client", None)
            if client_cls is None:
                return

            self._client_v2 = client_cls(cfg)
            self.enabled = True
            self._mode = "v2"
        except Exception:
            LOGGER.exception("Failed to initialise Alibaba Cloud OSS v2 client")
            self._client_v2 = None
            self._mode = None

    def _initialise_legacy_client(self) -> None:
        if not oss2:
            return

        if not (self.endpoint and self.bucket_name and self.access_key_id and self.access_key_secret):
            return

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
            self.enabled = True
            self._mode = "oss2"
        except Exception:
            LOGGER.exception("Failed to initialise legacy OSS client")
            self._bucket = None
            self._mode = None

    def _upload_v2(self, object_name: str, file_path: Path) -> bool:
        if not (oss_v2 and self._client_v2):
            return False

        put_request_cls = getattr(oss_v2, "PutObjectRequest", None)
        if put_request_cls is None:
            LOGGER.warning("Alibaba Cloud OSS v2 SDK missing PutObjectRequest")
            return False

        try:
            request = put_request_cls(bucket=self.bucket_name, key=object_name)
            presign_kwargs = {"expires": timedelta(seconds=self.presign_ttl)}
            presign_result = self._client_v2.presign(request, **presign_kwargs)
        except Exception:
            LOGGER.exception("Failed to presign OSS upload for %s", object_name)
            return False

        url = getattr(presign_result, "url", None)
        method = getattr(presign_result, "method", "PUT")
        signed_headers = getattr(presign_result, "signed_headers", {}) or {}

        if not url:
            LOGGER.error("Presign result missing URL for %s", object_name)
            return False

        if requests is None:
            LOGGER.warning("requests library unavailable; skipping OSS upload")
            return False

        try:
            with file_path.open("rb") as handle:
                response = requests.request(
                    method or "PUT",
                    url,
                    data=handle,
                    headers=dict(signed_headers),
                    timeout=30,
                )
            response.raise_for_status()
        except Exception:
            LOGGER.exception("Failed to upload %s via presigned request", object_name)
            return False

        return True

    def _upload_legacy(self, object_name: str, file_path: Path) -> bool:
        if not self._bucket:
            return False

        try:
            self._bucket.put_object_from_file(object_name, str(file_path))  # type: ignore[operator]
        except Exception:
            LOGGER.exception("Failed to upload %s to OSS", object_name)
            return False

        return True


oss_client = OSSClient()

