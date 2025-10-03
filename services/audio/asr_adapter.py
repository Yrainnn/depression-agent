from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from hashlib import sha1
from hmac import new as hmac_new
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote

import httpx
import websockets
from websockets.exceptions import WebSocketException

from packages.common.config import settings


LOGGER = logging.getLogger(__name__)


class AsrError(RuntimeError):
    """Raised when the primary ASR provider cannot fulfil a request."""


class StubASR:
    """Fallback ASR implementation that echoes supplied text."""

    def transcribe(
        self,
        text: Optional[str] = None,
        audio_ref: Optional[str] = None,
    ) -> List[dict]:
        if text:
            return [
                {
                    "utt_id": "u1",
                    "text": text,
                    "speaker": "patient",
                    "ts": [0, 3],
                    "conf": 0.95,
                }
            ]

        LOGGER.debug("StubASR invoked for audio_ref=%s", audio_ref)
        return []


class TingWuASR:
    """Adapter for Alibaba TingWu real-time ASR."""

    API_VERSION = "2018-05-23"

    def __init__(self, app_settings):
        self.settings = app_settings
        self.appkey = app_settings.tingwu_appkey
        self.ak_id = app_settings.tingwu_ak_id
        self.ak_secret = app_settings.tingwu_ak_secret
        self.region = app_settings.tingwu_region
        self.base = app_settings.tingwu_base.rstrip("/")
        self.ws_base = app_settings.tingwu_ws_base.rstrip("/")
        self.sample_rate = app_settings.tingwu_sr
        self.audio_format = app_settings.tingwu_format
        self.language = app_settings.tingwu_lang

        if not all([self.appkey, self.ak_id, self.ak_secret]):
            raise ValueError("TingWuASR requires APPKEY, AK_ID and AK_SECRET")

    # ------------------------------------------------------------------
    def _percent_encode(self, value: str) -> str:
        return quote(str(value), safe="-_.~")

    def _sign(self, method: str, params: Dict[str, str]) -> str:
        sorted_params = sorted((k, v) for k, v in params.items())
        canonicalized = "&".join(
            f"{self._percent_encode(k)}={self._percent_encode(v)}"
            for k, v in sorted_params
        )
        string_to_sign = (
            f"{method}&%2F&{self._percent_encode(canonicalized)}"
        )
        key = f"{self.ak_secret}&".encode("utf-8")
        signature = hmac_new(
            key,
            string_to_sign.encode("utf-8"),
            sha1,
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _build_params(self, action: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        params: Dict[str, str] = {
            "Action": action,
            "Version": self.API_VERSION,
            "Format": "JSON",
            "AccessKeyId": self.ak_id,
            "SignatureMethod": "HMAC-SHA1",
            "SignatureVersion": "1.0",
            "SignatureNonce": uuid.uuid4().hex,
            "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if extra:
            params.update(extra)
        return params

    def _request(
        self,
        method: str,
        action: str,
        *,
        params: Optional[Dict[str, str]] = None,
        json_body: Optional[dict] = None,
        timeout: float = 10.0,
    ) -> httpx.Response:
        query = self._build_params(action, params)
        query["Signature"] = self._sign(method, query)
        url = f"{self.base}/"
        try:
            response = httpx.request(
                method,
                url,
                params=query,
                json=json_body,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:  # pragma: no cover - network failures
            raise AsrError(f"TingWu request failed: {exc}") from exc

        if response.status_code >= 400:
            raise AsrError(
                f"TingWu request error {response.status_code}: {response.text}"
            )
        return response

    # ------------------------------------------------------------------
    def create_task(self) -> dict:
        """Create a TingWu transcription task via OpenAPI."""

        body = {
            "AppKey": self.appkey,
            "Input": {
                "Format": self.audio_format,
                "SampleRate": self.sample_rate,
                "SourceLanguage": self.language,
            },
            "Parameters": {
                "Transcription": {
                    "OutputLevel": 2,
                    "DiarizationEnabled": False,
                },
                "Translation": {"TranslationEnabled": False},
            },
        }

        response = self._request("POST", "CreateTask", json_body=body)
        payload = response.json()
        data = payload.get("Data") or {}

        record_id = data.get("RecordId") or payload.get("RecordId")
        token = data.get("Token") or data.get("TaskToken")
        stream_url = data.get("StreamUrl") or data.get("WsUrl")

        if not record_id:
            raise AsrError(f"TingWu CreateTask missing record_id: {payload}")

        if not stream_url:
            stream_url = self._build_ws_url(record_id, token)

        return {
            "record_id": record_id,
            "token": token,
            "stream_url": stream_url,
        }

    # ------------------------------------------------------------------
    async def stream_transcribe(self, audio_iter: Iterable[bytes]) -> List[dict]:
        """Stream audio frames to TingWu's WebSocket endpoint."""

        task_info = self.create_task()
        record_id = task_info["record_id"]
        token = task_info.get("token")
        stream_url = task_info.get("stream_url") or self._build_ws_url(
            record_id, token
        )

        if not stream_url:
            raise AsrError("TingWu stream URL unavailable")

        segments: List[dict] = []
        utt_inc = 0

        try:
            async with websockets.connect(stream_url) as ws:
                await ws.send(json.dumps({"type": "start"}))
                async for chunk in _async_iter(audio_iter):
                    if not chunk:
                        continue
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "stop"}))

                async for message in ws:
                    if isinstance(message, bytes):
                        continue
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:  # pragma: no cover - guard
                        continue

                    for result in self._extract_final_results(data):
                        utt_inc += 1
                        segments.append(
                            {
                                "utt_id": f"tw_{utt_inc}",
                                "text": result.get("text", ""),
                                "speaker": "patient",
                                "ts": [
                                    result.get("start", 0.0),
                                    result.get("end", 0.0),
                                ],
                                "conf": result.get("confidence", 0.95),
                            }
                        )
        except (WebSocketException, OSError) as exc:
            raise AsrError(f"TingWu streaming error: {exc}") from exc

        return segments

    # ------------------------------------------------------------------
    def transcribe(
        self,
        text: Optional[str] = None,
        audio_ref: Optional[str] = None,
    ) -> List[dict]:
        if text:
            return DEFAULT_ASR.transcribe(text=text)

        if not audio_ref:
            return []

        audio_path = Path(audio_ref)
        if not audio_path.exists():
            raise AsrError(f"audio_ref not found: {audio_ref}")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.stream_transcribe(self._iter_file_chunks(audio_path))
            )
        except AsrError as exc:
            LOGGER.warning("TingWu streaming failed: %s", exc)
            try:
                return self._transcribe_offline(audio_path)
            except Exception as offline_exc:  # pragma: no cover - defensive
                raise AsrError("TingWu offline transcription failed") from offline_exc
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("TingWu streaming unexpected error: %s", exc)
            try:
                return self._transcribe_offline(audio_path)
            except Exception as offline_exc:  # pragma: no cover
                raise AsrError("TingWu offline transcription failed") from offline_exc

    # ------------------------------------------------------------------
    def _iter_file_chunks(self, path: Path, chunk_size: int = 6400) -> Iterable[bytes]:
        with path.open("rb") as source:
            while True:
                data = source.read(chunk_size)
                if not data:
                    break
                yield data

    # ------------------------------------------------------------------
    def _build_ws_url(self, record_id: str, token: Optional[str]) -> Optional[str]:
        if not record_id:
            return None
        query = f"appkey={self._percent_encode(self.appkey)}&record_id={self._percent_encode(record_id)}"
        if token:
            query += f"&token={self._percent_encode(token)}"
        return f"{self.ws_base}?{query}"

    def _extract_final_results(self, payload: dict) -> List[dict]:
        results: List[dict] = []
        if not isinstance(payload, dict):
            return results

        final_flags = {"final", "is_final", "IsFinal", "Final"}
        text_fields = ["text", "Text", "sentence", "Sentence"]

        # Attempt to locate payload that signals final results
        if "payload" in payload:
            return self._extract_final_results(payload["payload"])

        if "Result" in payload and isinstance(payload["Result"], dict):
            sub = payload["Result"]
            if any(payload.get(flag) for flag in final_flags) or any(
                sub.get(flag) for flag in final_flags
            ):
                results.append(
                    {
                        "text": self._first_non_empty(sub, text_fields),
                        "start": self._safe_time(sub.get("BeginTime")),
                        "end": self._safe_time(sub.get("EndTime")),
                        "confidence": sub.get("Confidence", 0.95),
                    }
                )
                return results

        if "results" in payload and isinstance(payload["results"], list):
            for entry in payload["results"]:
                if not isinstance(entry, dict):
                    continue
                if any(entry.get(flag) for flag in final_flags):
                    results.append(
                        {
                            "text": self._first_non_empty(entry, text_fields),
                            "start": self._safe_time(entry.get("begin_time")),
                            "end": self._safe_time(entry.get("end_time")),
                            "confidence": entry.get("confidence", 0.95),
                        }
                    )
        elif any(payload.get(flag) for flag in final_flags):
            results.append(
                {
                    "text": self._first_non_empty(payload, text_fields),
                    "start": self._safe_time(payload.get("BeginTime")),
                    "end": self._safe_time(payload.get("EndTime")),
                    "confidence": payload.get("Confidence", 0.95),
                }
            )

        return results

    def _safe_time(self, value: Optional[float]) -> float:
        if value is None:
            return 0.0
        try:
            return float(value) / (1000 if float(value) > 32 else 1)
        except (TypeError, ValueError):  # pragma: no cover - guard
            return 0.0

    def _first_non_empty(self, data: dict, keys: List[str]) -> str:
        for key in keys:
            value = data.get(key)
            if value:
                return str(value)
        return ""

    def _transcribe_offline(self, audio_path: Path) -> List[dict]:
        task = self.create_task()
        record_id = task["record_id"]

        # Upload audio if upload URL available
        upload_url = task.get("upload_url") or task.get("UploadUrl")
        if upload_url:
            with audio_path.open("rb") as source:
                try:
                    httpx.put(upload_url, content=source.read(), timeout=30.0)
                except httpx.HTTPError as exc:  # pragma: no cover
                    raise AsrError(f"TingWu upload failed: {exc}") from exc

        deadline = time.time() + 60
        segments: List[dict] = []
        poll_interval = 2.0
        utt_inc = 0

        while time.time() < deadline:
            info = self._request(
                "GET",
                "GetTaskInfo",
                params={"RecordId": record_id},
                timeout=10.0,
            ).json()

            data = info.get("Data") or {}
            status = (data.get("Status") or data.get("TaskStatus") or "").lower()
            if status in {"succeeded", "success", "completed", "complete"}:
                sentences = self._extract_sentences(data)
                for sentence in sentences:
                    utt_inc += 1
                    segments.append(
                        {
                            "utt_id": f"tw_off_{utt_inc}",
                            "text": sentence.get("text", ""),
                            "speaker": "patient",
                            "ts": [
                                sentence.get("start", 0.0),
                                sentence.get("end", 0.0),
                            ],
                            "conf": sentence.get("confidence", 0.95),
                        }
                    )
                break

            if status in {"failed", "error"}:
                raise AsrError(f"TingWu task failed: {info}")

            time.sleep(poll_interval)

        return segments

    def _extract_sentences(self, data: dict) -> List[dict]:
        sentences: List[dict] = []
        transcription = data.get("TranscriptionResult") or data.get("Result") or {}
        if isinstance(transcription, dict):
            if "Sentences" in transcription and isinstance(
                transcription["Sentences"], list
            ):
                for sent in transcription["Sentences"]:
                    if not isinstance(sent, dict):
                        continue
                    sentences.append(
                        {
                            "text": sent.get("Text", ""),
                            "start": self._safe_time(sent.get("BeginTime")),
                            "end": self._safe_time(sent.get("EndTime")),
                            "confidence": sent.get("Confidence", 0.95),
                        }
                    )
            elif "Text" in transcription:
                sentences.append(
                    {
                        "text": transcription.get("Text", ""),
                        "start": self._safe_time(transcription.get("BeginTime")),
                        "end": self._safe_time(transcription.get("EndTime")),
                        "confidence": transcription.get("Confidence", 0.95),
                    }
                )
        return sentences


async def _async_iter(source: Iterable[bytes]):
    for chunk in source:
        yield chunk


DEFAULT_ASR = StubASR()
_TINGWU_ASR: Optional[TingWuASR] = None


def _provider() -> StubASR | TingWuASR:
    global _TINGWU_ASR
    if (
        settings.tingwu_appkey
        and settings.tingwu_ak_id
        and settings.tingwu_ak_secret
    ):
        if _TINGWU_ASR is None:
            try:
                _TINGWU_ASR = TingWuASR(settings)
            except Exception as exc:  # pragma: no cover - configuration guard
                LOGGER.warning("Failed to initialise TingWuASR: %s", exc)
                return DEFAULT_ASR
        return _TINGWU_ASR
    return DEFAULT_ASR


def transcribe(
    text: Optional[str] = None,
    audio_ref: Optional[str] = None,
) -> List[dict]:
    provider = _provider()
    try:
        return provider.transcribe(text=text, audio_ref=audio_ref)
    except AsrError as exc:
        LOGGER.warning("Primary ASR provider failed, falling back to stub: %s", exc)
        if provider is DEFAULT_ASR:
            return []
        return DEFAULT_ASR.transcribe(text=text, audio_ref=audio_ref)


__all__ = [
    "AsrError",
    "StubASR",
    "TingWuASR",
    "transcribe",
]

