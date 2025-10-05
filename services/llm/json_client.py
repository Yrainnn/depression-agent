from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
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

LOGGER = logging.getLogger(__name__)


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

    def enabled(self) -> bool:
        return bool(self.base and self.key)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    def _post_chat(
        self,
        *,
        messages: List[dict],
        response_format: Optional[dict] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout: float = 20.0,
    ) -> str:
        if not self.enabled():  # pragma: no cover - guard rail
            raise RuntimeError("DeepSeek client not configured")

        url = self.base.rstrip("/") + "/v1/chat/completions"
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

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def analyze(
        self,
        dialogue_json: List[dict],
        system_prompt: Optional[str] = None,
    ) -> HAMDResult:
        prompt = system_prompt or get_prompt_hamd17()
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(dialogue_json, ensure_ascii=False)},
        ]
        try:
            content = self._post_chat(
                messages=messages,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(content)
            return HAMDResult.model_validate(parsed)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("DeepSeek analyze failed: %s", exc)
            latest_user_text = ""
            for entry in reversed(dialogue_json):
                if entry.get("role") == "user":
                    latest_user_text = entry.get("text") or ""
                    break

            lowered = (latest_user_text or "").lower()
            needs_frequency = not any(keyword in lowered for keyword in ("次", "天", "每周"))
            needs_duration = not any(
                keyword in lowered for keyword in ("整天", "小时", "多久")
            )
            needs_severity = not any(
                keyword in lowered for keyword in ("严重", "很难", "影响")
            )
            needs_negation = any(keyword in lowered for keyword in ("没有", "不"))

            clarify_need: Optional[str] = None
            if needs_frequency:
                clarify_need = "频次"
            elif needs_duration:
                clarify_need = "持续时间"
            elif needs_severity:
                clarify_need = "严重程度"
            elif needs_negation:
                clarify_need = "是否否定"

            mock = {
                "items": [
                    {
                        "item_id": 1,
                        "symptom_summary": "信息有限",
                        "dialogue_evidence": "信息缺失",
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
                    for idx in range(2, 18)
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
                timeout=15,
            )
            text = (content or "").strip()
            for end in ["？", "。", "!", "！", "?"]:
                if end in text:
                    text = text.split(end)[0] + end
                    break
            return text[:30] if text else None
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("DeepSeek clarify generation failed: %s", exc)
            return None

    def plan_turn(
        self,
        dialogue_json: List[dict],
        progress: dict,
    ) -> ControllerDecision:
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
            timeout=25,
        )
        data = json.loads(content)
        return ControllerDecision.model_validate(data)


# Convenience singleton -------------------------------------------------
client = DeepSeekJSONClient()

__all__ = [
    "DeepSeekJSONClient",
    "HAMDItem",
    "HAMDResult",
    "HAMDTotal",
    "client",
    "get_prompt_hamd17",
    "get_prompt_diagnosis",
    "get_prompt_mdd_judgment",
    "get_prompt_clarify_cn",
]
