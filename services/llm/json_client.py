from __future__ import annotations

import json
import logging
import os
from time import monotonic
from typing import Any, Dict, List, Optional, Union

import httpx

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from packages.common.config import settings
LOGGER = logging.getLogger(__name__)


class DeepSeekTemporarilyUnavailableError(RuntimeError):
    """Raised when the DeepSeek client is temporarily unavailable."""
    pass


class HAMDItem(BaseModel):
    item_id: int
    symptom_summary: Optional[str] = None
    dialogue_evidence: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)
    score: int
    score_type: Optional[str] = None
    score_reason: Optional[str] = None
    clarify_need: Optional[Union[str, bool]] = None
    clarify_prompt: Optional[str] = None


class HAMDTotal(BaseModel):
    得分序列: str
    pre_correction_total: int
    corrected_total: int
    correction_basis: str


class HAMDResult(BaseModel):
    items: List[HAMDItem]
    total_score: HAMDTotal
    summary: Optional[str] = None


class ClarifyTarget(BaseModel):
    item_id: int
    clarify_need: Optional[str] = None


class HAMDPartial(BaseModel):
    items: List[dict] = Field(default_factory=list)
    total_score: Dict[str, Any] = Field(default_factory=dict)


class ControllerDecision(BaseModel):
    action: str
    current_item_id: int
    next_utterance: str
    clarify_target: Optional[ClarifyTarget] = None
    hamd_partial: Optional[HAMDPartial] = None


class DeepSeekJSONClient:
    """Minimal OpenAI-compatible client targeting DeepSeek JSON responses."""

    def __init__(
        self,
        base: Optional[str] = None,
        key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.base = base or settings.deepseek_api_base
        self.key = key or settings.deepseek_api_key
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.chat_timeout = float(settings.deepseek_chat_timeout)
        self._warned_bad_base = False
        self._circuit_open_until: Optional[float] = None
        trimmed_base = (self.base or "").rstrip("/")
        if trimmed_base and not trimmed_base.endswith("/v1"):
            LOGGER.warning(
                "DEEPSEEK_API_BASE should target the OpenAI-compatible /v1 host; "
                "currently=%s",
                self.base,
            )
            self._warned_bad_base = True
        if not self.key:
            LOGGER.warning("DeepSeek client initialised without an API key")

    def enabled(self) -> bool:
        return bool(self.base and self.key)

    def usable(self) -> bool:
        return self.enabled() and not self._is_circuit_open()

    def _is_circuit_open(self) -> bool:
        if self._circuit_open_until is None:
            return False
        if monotonic() >= self._circuit_open_until:
            self._circuit_open_until = None
            return False
        return True

    def _trip_circuit(self, duration: Optional[float] = None) -> None:
        effective = self.chat_timeout if duration is None else duration
        if effective <= 0:
            self._circuit_open_until = None
            return
        self._circuit_open_until = monotonic() + effective

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    def _post_chat(
        self,
        *,
        messages: List[dict],
        response_format: Optional[dict] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        if not self.enabled():  # pragma: no cover - guard rail
            raise RuntimeError("DeepSeek client not configured")

        if self._is_circuit_open():
            remaining = max(self._circuit_open_until - monotonic(), 0) if self._circuit_open_until else 0
            raise DeepSeekTemporarilyUnavailableError(
                f"DeepSeek client temporarily disabled (retry in {remaining:.1f}s)"
            )

        url_base = (self.base or "").strip()
        trimmed_base = url_base.rstrip("/")
        if (
            not self._warned_bad_base
            and trimmed_base
            and not trimmed_base.endswith("/v1")
        ):
            LOGGER.warning(
                "DeepSeek API base %s should include the /v1 suffix; requests will "
                "append it automatically",
                url_base,
            )
            self._warned_bad_base = True
        trimmed_base = url_base.rstrip("/")
        if trimmed_base.endswith("/v1"):
            url = trimmed_base + "/chat/completions"
        else:
            url = trimmed_base + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if stream:
            payload["stream"] = True

        effective_timeout = self.chat_timeout if timeout is None else timeout
        with httpx.Client(timeout=effective_timeout) as client:
            try:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response else "?"
                text_preview = exc.response.text if exc.response else ""
                if len(text_preview) > 500:
                    text_preview = text_preview[:500] + "…"
                LOGGER.error(
                    "DeepSeek chat request failed with status %s: %s", status, text_preview
                )
                raise
            except httpx.RequestError as exc:
                LOGGER.error("DeepSeek chat request errored: %s", exc)
                read_timeout_cls = getattr(httpx, "ReadTimeout", None)
                if read_timeout_cls and isinstance(exc, read_timeout_cls):
                    self._trip_circuit()
                    raise DeepSeekTemporarilyUnavailableError(
                        "DeepSeek timed out and is temporarily unavailable"
                    ) from exc
                raise

            data = response.json()
            self._circuit_open_until = None
            return data["choices"][0]["message"]["content"]

    def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """Send a chat completion request and return the assistant content."""

        return self._post_chat(
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
        )

    def call(
        self,
        prompt: str,
        *,
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """Convenience wrapper for single-message prompts."""

        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
        )

    def call_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Invoke DeepSeek for a strict JSON response."""

        content = self.call(
            prompt,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
        )
        return json.loads(content)


# Convenience singleton -------------------------------------------------
client = DeepSeekJSONClient()

__all__ = [
    "DeepSeekJSONClient",
    "DeepSeekTemporarilyUnavailableError",
    "HAMDItem",
    "HAMDResult",
    "HAMDTotal",
    "client",
]
