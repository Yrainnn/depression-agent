from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
from time import monotonic
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from packages.common.config import settings
from services.llm.prompts import (
    get_prompt_clarify_cn,
    get_prompt_diagnosis,
    get_prompt_hamd17,
    get_prompt_hamd17_controller,
    get_prompt_mdd_judgment,
)
from services.orchestrator.gap_utils import GAP_LABELS, detect_information_gaps
from services.orchestrator.questions_hamd17 import HAMD17_QUESTION_BANK

LOGGER = logging.getLogger(__name__)


class DeepSeekTemporarilyUnavailableError(RuntimeError):
    """Raised when the DeepSeek client is temporarily unavailable."""
    pass


class HAMDItem(BaseModel):
    item_id: int
    symptom_summary: str
    dialogue_evidence: str
    evidence_refs: List[str] = Field(default_factory=list)
    score: int
    score_type: str
    score_reason: str
    clarify_need: Optional[str] = None


class HAMDTotal(BaseModel):
    得分序列: str
    pre_correction_total: int
    corrected_total: int
    correction_basis: str


class HAMDResult(BaseModel):
    items: List[HAMDItem]
    total_score: HAMDTotal


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
        self.clarify_timeout = float(settings.deepseek_clarify_timeout)
        self.controller_timeout = float(settings.deepseek_controller_timeout)
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
        url = url_base + "/v1/chat/completions"
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

    def analyze(
        self,
        dialogue_json: List[dict],
        system_prompt: Optional[str] = None,
    ) -> HAMDResult:
        if not self.usable():
            raise DeepSeekTemporarilyUnavailableError(
                "DeepSeek analyze skipped because the client is temporarily unavailable"
            )
        prompt = system_prompt or get_prompt_hamd17()
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(dialogue_json, ensure_ascii=False)},
        ]
        try:
            content = self._post_chat(
                messages=messages,
                response_format={"type": "json_object"},
                timeout=self.chat_timeout,
            )
            parsed = json.loads(content)
            return HAMDResult.model_validate(parsed)
        except DeepSeekTemporarilyUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("DeepSeek analyze failed: %s", exc)

            latest_user_text = ""
            latest_item_id = 1
            for entry in reversed(dialogue_json):
                role = entry.get("role")
                if role == "user" and not latest_user_text:
                    latest_user_text = entry.get("text") or ""
                if role == "assistant" and entry.get("text"):
                    text = str(entry.get("text"))
                    for item_id, payload in HAMD17_QUESTION_BANK.items():
                        questions = payload.get("primary", [])
                        clarifies = []
                        clarify_map = payload.get("clarify", {})
                        for value in clarify_map.values():
                            if isinstance(value, list):
                                clarifies.extend(value)
                        if any(q in text or text in q for q in questions + clarifies):
                            latest_item_id = item_id
                            break
                    if latest_item_id != 1:
                        break

            gaps = detect_information_gaps(latest_user_text, item_id=latest_item_id)
            gap_key = gaps[0] if gaps else None
            clarify_need = GAP_LABELS.get(gap_key, None)

            mock = {
                "items": [
                    {
                        "item_id": latest_item_id,
                        "symptom_summary": "信息有限",
                        "dialogue_evidence": latest_user_text or "信息缺失",
                        "evidence_refs": [],
                        "score": 0,
                        "score_type": "类型4",
                        "score_reason": "信息不足",
                        "clarify_need": clarify_need,
                    }
                ]
                + [
                    {
                        "item_id": idx,
                        "symptom_summary": "未提及",
                        "dialogue_evidence": "未提及",
                        "evidence_refs": [],
                        "score": 0,
                        "score_type": "类型3",
                        "score_reason": "未涉及",
                        "clarify_need": None,
                    }
                    for idx in range(1, 18)
                    if idx != latest_item_id
                ],
                "total_score": {
                    "得分序列": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
                    "pre_correction_total": 0,
                    "corrected_total": 0,
                    "correction_basis": "类型4条目数量0，平均分0.00，修正总分=A+0.00×0≈0",
                },
            }
            return HAMDResult.model_validate(mock)

    def gen_clarify_question(
        self,
        item_id: int,
        item_name: str,
        clarify_need: str,
        evidence_text: str,
    ) -> Optional[str]:
        prompt = get_prompt_clarify_cn().format(
            item_id=item_id,
            item_name=item_name,
            clarify_need=clarify_need or "（未标注）",
            evidence_text=(evidence_text or "（无明确证据片段）")[:200],
        )
        try:
            content = self._post_chat(
                messages=[{"role": "user", "content": prompt}],
                response_format=None,
                max_tokens=64,
                temperature=0.2,
                timeout=self.clarify_timeout,
            )
            text = (content or "").strip()
            for end in ["？", "。", "!", "！", "?"]:
                if end in text:
                    text = text.split(end)[0] + end
                    break
            return text[:30] if text else None
        except DeepSeekTemporarilyUnavailableError:
            return None
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("DeepSeek clarify generation failed: %s", exc)
            return None

    def plan_turn(
        self,
        dialogue_json: List[dict],
        progress: dict,
    ) -> ControllerDecision:
        if not self.usable():
            raise DeepSeekTemporarilyUnavailableError(
                "DeepSeek controller planning skipped because the client is temporarily unavailable"
            )
        prompt = get_prompt_hamd17_controller()
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "dialogue_json": dialogue_json,
                        "progress": progress,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        content = self._post_chat(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=2048,
            temperature=0.2,
            timeout=self.controller_timeout,
        )
        data = json.loads(content)
        return ControllerDecision.model_validate(data)


# Convenience singleton -------------------------------------------------
client = DeepSeekJSONClient()

__all__ = [
    "DeepSeekJSONClient",
    "DeepSeekTemporarilyUnavailableError",
    "HAMDItem",
    "HAMDResult",
    "HAMDTotal",
    "client",
    "get_prompt_hamd17",
    "get_prompt_diagnosis",
    "get_prompt_mdd_judgment",
    "get_prompt_clarify_cn",
]
