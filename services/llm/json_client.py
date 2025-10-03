import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, ValidationError, root_validator

from packages.common.config import settings

LOGGER = logging.getLogger(__name__)

VAGUE_PHRASES = [
    "还好",
    "一般",
    "差不多",
    "说不清",
    "不好说",
    "看情况",
    "可能吧",
    "偶尔吧",
    "有点吧",
    "还行",
    "凑合",
]


class SymptomScore(BaseModel):
    name: str
    score: int
    evidence_refs: List[str]


class AnalysisResult(BaseModel):
    summary: str
    scores: List[SymptomScore]
    follow_up_questions: List[str]

    @root_validator(pre=True)
    def ensure_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: N805
        values.setdefault("scores", [])
        values.setdefault("follow_up_questions", [])
        return values


@dataclass
class LLMJSONClient:
    base_url: Optional[str] = settings.deepseek_api_base
    api_key: Optional[str] = settings.deepseek_api_key
    model: str = "gpt-4o-mini"
    max_retries: int = 2

    keyword_map: Dict[str, Dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.keyword_map is None:
            self.keyword_map = {
                "早醒": {"name": "sleep_disturbance", "score": 2},
                "没兴趣": {"name": "anhedonia", "score": 2},
                "情绪低落": {"name": "low_mood", "score": 3},
                "焦虑": {"name": "anxiety", "score": 2},
            }

    # ------------------------------------------------------------------
    def analyze_transcript(self, segments: List[Dict[str, Any]]) -> AnalysisResult:
        """Run structured LLM analysis for the given transcript segments."""

        if self.base_url and self.api_key:
            try:
                return self._call_remote_llm(segments)
            except Exception as exc:  # pragma: no cover - runtime guard
                LOGGER.warning("Remote LLM failed (%s), falling back to mock", exc)

        return self._mock_analysis(segments)

    # ------------------------------------------------------------------
    def _call_remote_llm(self, segments: List[Dict[str, Any]]) -> AnalysisResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {
                "role": "system",
                "content": "You are a clinician assistant returning JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "请阅读患者访谈文本，评估相关抑郁症状的量表分项，"
                    "并返回JSON：{summary: str, scores: [{name, score, evidence_refs: [utt_id]}],"
                    "follow_up_questions: [str]}"
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    f"{seg.get('utt_id')}: {seg.get('text')}" for seg in segments
                ),
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = AnalysisResult.parse_raw(content)
                return parsed
            except (requests.RequestException, KeyError, ValidationError, ValueError) as exc:
                last_error = exc
                LOGGER.warning("LLM attempt %s failed: %s", attempt, exc)
        raise RuntimeError(f"LLM call failed: {last_error}")

    # ------------------------------------------------------------------
    def _mock_analysis(self, segments: List[Dict[str, Any]]) -> Any:
        last_patient_text = ""
        for segment in reversed(segments):
            speaker = str(segment.get("speaker", "patient"))
            if speaker and speaker != "patient":
                continue
            candidate = str(segment.get("text", "")).strip()
            if candidate:
                last_patient_text = candidate
                break

        if last_patient_text and self._is_vague(last_patient_text):
            return {
                "intent": "score_item",
                "schema_id": "PHQ9.v1",
                "item_tag": None,
                "per_item_scores": [],
                "evidence_spans": [],
                "opinion": {"summary": "", "rationale": ""},
                "total_score": 0,
                "clarify_request": "请具体点：每周大概几天？",
            }

        text = "\n".join(seg.get("text", "") for seg in segments)
        scores: List[SymptomScore] = []
        follow_up_questions: List[str] = []

        for keyword, meta in self.keyword_map.items():
            evidence_refs = [seg.get("utt_id", "") for seg in segments if keyword in seg.get("text", "")]
            if evidence_refs:
                scores.append(
                    SymptomScore(
                        name=meta["name"],
                        score=meta["score"],
                        evidence_refs=evidence_refs,
                    )
                )

        if "睡" in text and not any(score.name == "sleep_disturbance" for score in scores):
            follow_up_questions.append("最近的睡眠情况如何？")
        if "食欲" not in text:
            follow_up_questions.append("最近食欲有变化吗？")

        summary = "患者分享了情绪与生活状态，建议继续跟进。"
        return AnalysisResult(summary=summary, scores=scores, follow_up_questions=follow_up_questions)

    @staticmethod
    def _is_vague(text: str) -> bool:
        if not text:
            return False
        normalized = re.sub(r"[\s\W]+", "", text, flags=re.UNICODE).lower()
        for phrase in VAGUE_PHRASES:
            phrase_norm = re.sub(r"[\s\W]+", "", phrase, flags=re.UNICODE).lower()
            if phrase_norm and phrase_norm in normalized:
                return True
        return False


client = LLMJSONClient()
