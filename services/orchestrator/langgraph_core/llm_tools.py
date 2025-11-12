from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.orchestrator.prompts import get_prompt

try:  # pragma: no cover - runtime wiring best effort
    from services.llm.json_client import client as _deepseek_client
except Exception:  # pragma: no cover - fallback when DeepSeek is unavailable
    _deepseek_client = None


class _FallbackLLM:
    """Minimal heuristics used when the real client is unavailable."""

    _RISK_TOKENS = ("想死", "结束生命", "自杀", "活着没意思")

    def generate(self, prompt: str, **_: Any) -> str:
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.endswith("？") or line.endswith("?"):
                return line
        return "最近两周，您的心情怎么样？"

    def extract_facts(self, answer: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        match = re.search(r"(\d+)分", answer)
        if match:
            data["self_rating"] = int(match.group(1))
        if any(token in answer for token in ("凌晨", "早醒", "夜里")):
            data["sleep_pattern"] = "early_awake"
        return data

    def risk_detect(self, answer: str) -> Optional[str]:
        if any(token in answer for token in self._RISK_TOKENS):
            return "suicide_high"
        return None

    def identify_themes(self, text: str) -> List[str]:
        vocab = ["失落", "罪恶", "绝望", "睡眠异常", "食欲改变", "焦虑", "兴趣缺失", "精力下降", "自责", "无价值感"]
        _ = get_prompt("theme_identification").format(vocab=",".join(vocab), text=text)
        themes: List[str] = []
        if any(token in text for token in ("没意思", "绝望", "无望", "活不下去")):
            themes.append("绝望")
        if any(token in text for token in ("凌晨", "早醒", "三四点", "五点醒")):
            themes.append("睡眠异常")
        return list(dict.fromkeys(themes))

    def summarize_context(self, previous: str, new: str, limit: int = 500) -> str:
        _ = get_prompt("rolling_summary").format(prev=previous, new=new, limit=limit)
        merged = (previous + " " + new).strip()
        return merged[:limit]

    def score_item(self, item_ctx: Dict[str, Any]) -> Dict[str, Any]:
        facts = item_ctx.get("facts", {})
        themes = item_ctx.get("themes", [])
        score = 0
        if facts.get("self_rating", 0) >= 7:
            score = 3
        if "绝望" in themes:
            score = max(score, 2)
        return {"item_id": item_ctx.get("item_id"), "score": score}


class _DeepSeekLLM(_FallbackLLM):
    """Light wrapper around the DeepSeek JSON client used by the project."""

    def __init__(self) -> None:
        self._client = _deepseek_client

    def generate(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - network path
        if not self._client or not getattr(self._client, "usable", lambda: False)():
            return super().generate(prompt, **kwargs)
        try:
            payload = self._client.plan_turn(
                dialogue=[],
                progress={},
                prompt=prompt,
            )
        except Exception:
            return super().generate(prompt, **kwargs)
        question = payload.get("question") or payload.get("next_utterance")
        if isinstance(question, str) and question.strip():
            return question.strip()
        return super().generate(prompt, **kwargs)

    def extract_facts(self, answer: str) -> Dict[str, Any]:  # pragma: no cover - rely on fallback heuristics
        return super().extract_facts(answer)

    def risk_detect(self, answer: str) -> Optional[str]:  # pragma: no cover - rely on fallback heuristics
        return super().risk_detect(answer)

    def identify_themes(self, text: str) -> List[str]:  # pragma: no cover - rely on fallback heuristics
        return super().identify_themes(text)

    def summarize_context(self, previous: str, new: str, limit: int = 500) -> str:  # pragma: no cover
        return super().summarize_context(previous, new, limit)

    def score_item(self, item_ctx: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        return super().score_item(item_ctx)


LLM = _DeepSeekLLM() if _deepseek_client else _FallbackLLM()
