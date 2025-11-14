from __future__ import annotations

import json
from typing import Any, Dict

from services.orchestrator.prompts import get_prompt

try:  # pragma: no cover - runtime wiring best effort
    from services.llm.json_client import client as _deepseek_client
except Exception:  # pragma: no cover - fallback when DeepSeek is unavailable
    _deepseek_client = None


class _BaseBackend:
    """Abstract backend definition for tool invocations."""

    def call(self, func: str, payload: Dict[str, Any]) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class _FallbackBackend(_BaseBackend):
    """Offline fallback used during tests or when the LLM is unavailable."""

    def call(self, func: str, payload: Dict[str, Any]) -> Any:
        text = payload.get("text", "")
        if func == "generate":
            template = payload.get("template") or "最近两周您的心情是否低落？"
            question = template.strip()
            if not question.endswith("？") and not question.endswith("?"):
                question += "？"
            return {"text": question}
        if func == "risk_detect":
            if any(token in text for token in ("想死", "自杀", "活着没意思", "结束生命")):
                return {"risk_level": "high"}
            return {"risk_level": "none"}
        if func == "extract_facts":
            facts: Dict[str, Any] = {}
            if "分" in text:
                import re

                if match := re.search(r"(\d+)分", text):
                    facts["self_rating"] = int(match.group(1))
            if any(token in text for token in ("凌晨", "早醒")):
                facts["sleep_pattern"] = "early_awake"
            return {"facts": facts}
        if func == "identify_themes":
            themes = []
            if any(token in text for token in ("绝望", "没意思", "无望")):
                themes.append("绝望")
            if "凌晨" in text or "早醒" in text:
                themes.append("睡眠异常")
            return {"themes": themes}
        if func == "summarize_context":
            prev = payload.get("prev", "")
            new = payload.get("new", "")
            limit = int(payload.get("limit", 500))
            summary = (prev + " " + new).strip()
            return {"summary": summary[:limit]}
        if func == "score_item":
            return {"score": 2}
        if func == "clarify_branch":
            answer = payload.get("answer", "")
            branches = payload.get("branches", []) or []
            if any(token in answer for token in ("没有", "不", "偶尔", "说不清", "否认", "没觉得")):
                for branch in branches:
                    if "否定" in str(branch.get("condition", "")):
                        return {
                            "matched": branch.get("condition"),
                            "next": branch.get("next"),
                            "reason": "fallback_negative_match",
                            "clarify": False,
                        }
            if branches:
                branch = branches[0]
                return {
                    "matched": branch.get("condition"),
                    "next": branch.get("next"),
                    "reason": "fallback_default_branch",
                    "clarify": False,
                }
            return {
                "matched": None,
                "next": None,
                "clarify": True,
                "clarify_question": "能再具体描述一下你的感受吗？",
                "reason": "fallback_no_branch",
            }
        if func == "clarify_question":
            template = (payload.get("template") or "可以再详细描述一下吗？").strip()
            for prefix in ("请简单描述", "请描述", "请简要描述", "请", "请问"):
                if template.startswith(prefix):
                    template = template[len(prefix) :].strip()
                    break
            template = template.strip("？?。:：") or "情况"
            return {"question": f"是否可以具体说明{template}？"}
        raise ValueError(f"Unknown tool function: {func}")


class _DeepSeekBackend(_BaseBackend):
    """Backend delegating to the DeepSeek JSON client."""

    def __init__(self) -> None:
        self.client = _deepseek_client

    def _chat_json(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> Dict[str, Any]:
        if not self.client or not getattr(self.client, "usable", lambda: False)():
            return {}
        try:
            content = self.client._post_chat(  # type: ignore[attr-defined]
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            return {}
        try:
            return json.loads(content)
        except Exception:
            return {}

    def _handle_clarify_branch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = get_prompt("branch_clarification").format(
            answer=payload.get("answer", ""),
            branches=json.dumps(payload.get("branches", []), ensure_ascii=False),
        )
        result = self._chat_json(prompt)
        if result and not result.get("clarify"):
            return result
        if result and result.get("clarify") and result.get("clarify_question"):
            return result
        question = self._handle_clarify_question(payload)
        if isinstance(question, dict) and question.get("question"):
            result = result or {}
            result["clarify"] = True
            result["clarify_question"] = question["question"]
        return result

    def _handle_clarify_question(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = get_prompt("clarify_question").format(
            context=payload.get("context", ""),
            template=payload.get("template", ""),
            answer=payload.get("answer", ""),
        )
        result = self._chat_json(prompt)
        if isinstance(result, dict) and result.get("question"):
            return result
        return {"question": payload.get("template", "能再详细说明一下吗？")}

    def call(self, func: str, payload: Dict[str, Any]) -> Any:
        if not self.client or not getattr(self.client, "usable", lambda: False)():
            return _FallbackBackend().call(func, payload)

        try:
            if func == "generate":
                prompt = get_prompt("strategy_generation").format(
                    context=payload.get("context", ""),
                    template=payload.get("template", ""),
                )
                dialogue = payload.get("dialogue") or []
                progress = payload.get("progress") or {}
                result = self.client.plan_turn(
                    dialogue=dialogue,
                    progress=progress,
                    prompt=prompt,
                )
                question = None
                if isinstance(result, dict):
                    question = (
                        result.get("question")
                        or result.get("next")
                        or result.get("next_utterance")
                        or result.get("text")
                    )
                if isinstance(question, str) and question.strip():
                    return {"text": question.strip(), "raw": result}
                return _FallbackBackend().call(func, payload)
            if func == "risk_detect":
                prompt = get_prompt("risk_detection").format(text=payload.get("text", ""))
                return self._chat_json(prompt)
            elif func == "extract_facts":
                prompt = get_prompt("fact_extraction").format(text=payload.get("text", ""))
                return self._chat_json(prompt)
            elif func == "identify_themes":
                vocab = payload.get(
                    "vocab",
                    "失落,罪恶,绝望,焦虑,兴趣缺失,睡眠异常,食欲改变,精力下降,自责,无价值感",
                )
                prompt = get_prompt("theme_identification").format(vocab=vocab, text=payload.get("text", ""))
                return self._chat_json(prompt)
            elif func == "summarize_context":
                prompt = get_prompt("rolling_summary").format(
                    prev=payload.get("prev", ""),
                    new=payload.get("new", ""),
                )
                result = self._chat_json(prompt)
                if result:
                    limit = int(payload.get("limit", 500))
                    summary = result.get("summary")
                    if isinstance(summary, str):
                        result["summary"] = summary[:limit]
                return result
            elif func == "score_item":
                prompt = get_prompt("single_item_scoring").format(
                    item_name=payload.get("item_name", ""),
                    facts=json.dumps(payload.get("facts", {}), ensure_ascii=False),
                    themes=json.dumps(payload.get("themes", []), ensure_ascii=False),
                    summary=payload.get("summary", ""),
                    risks=json.dumps(payload.get("risks", []), ensure_ascii=False),
                    patient_snapshot=json.dumps(payload.get("patient_snapshot", {}), ensure_ascii=False),
                    dialogue=payload.get("dialogue", ""),
                )
                return self._chat_json(prompt, max_tokens=256)
            elif func == "clarify_branch":
                return self._handle_clarify_branch(payload)
            elif func == "clarify_question":
                return self._handle_clarify_question(payload)
        except Exception:
            return _FallbackBackend().call(func, payload)

        return _FallbackBackend().call(func, payload)


class LLMToolBox:
    """Unified entry point for LangGraph LLM utilities."""

    def __init__(self) -> None:
        self.backend: _BaseBackend
        if _deepseek_client:
            self.backend = _DeepSeekBackend()
        else:
            self.backend = _FallbackBackend()

    def call(self, tool_name: str, payload: Dict[str, Any]) -> Any:
        if isinstance(self.backend, _DeepSeekBackend) and tool_name in {"clarify_branch", "clarify_question"}:
            handler = getattr(self.backend, f"_handle_{tool_name}")
            return handler(payload)
        return self.backend.call(tool_name, payload)


LLM = LLMToolBox()
