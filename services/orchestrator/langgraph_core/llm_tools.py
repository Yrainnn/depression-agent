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
        if func == "template_builder":
            stub = {
                "yaml": "project_id: 0\nstrategies: {}\n",
                "strategy_descriptions": {},
            }
            return {"text": json.dumps(stub, ensure_ascii=False)}
        if func == "generate":
            template = payload.get("template") or "最近两周您的心情是否低落？"
            question = template.strip()
            if not question.endswith("？") and not question.endswith("?"):
                question += "？"
            return {"text": question}
        if func == "risk_detect":
            high_keywords = ["自杀", "suicide", "结束生命", "想死", "割腕", "跳楼", "寻死", "自残", "了结自己"]
            medium_keywords = ["不想活", "活着没意思", "想消失", "想离开这个世界", "活得好累", "毫无希望", "绝望", "轻生"]
            low_keywords = ["没意思", "难受", "情绪低落", "很沮丧", "不开心", "消极"]

            triggers: list[str] = []
            risk_level = "none"

            for token in high_keywords:
                if token and token in text:
                    triggers.append(token)
                    risk_level = "high"

            if risk_level != "high":
                for token in medium_keywords:
                    if token and token in text:
                        triggers.append(token)
                        risk_level = "medium"

            if risk_level == "none":
                for token in low_keywords:
                    if token and token in text:
                        triggers.append(token)
                        risk_level = "low"

            reason = "未检测到明显危险信号"
            if risk_level == "high":
                reason = "出现明确的自伤或自杀意图"
            elif risk_level == "medium":
                reason = "表达强烈的求死或消失念头"
            elif risk_level == "low":
                reason = "存在轻度的消极或绝望情绪"

            result: Dict[str, Any] = {"risk_level": risk_level}
            if triggers:
                result["triggers"] = triggers
            result["reason"] = reason
            if risk_level in {"medium", "high"}:
                result["advice"] = (
                    "请立即转接人工或联系当地紧急服务。"
                    if risk_level == "high"
                    else "请加以安抚并鼓励其寻求专业帮助。"
                )
            return result
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
            for branch in branches:
                condition = branch.get("condition", "")
                if any(token in answer for token in ("没有", "不", "偶尔", "说不清")) and "否定" in condition:
                    return {
                        "matched": condition,
                        "next": branch.get("next"),
                        "reason": "回答包含否定语气",
                    }
            if branches:
                branch = branches[0]
                return {
                    "matched": branch.get("condition"),
                    "next": branch.get("next"),
                    "reason": "默认选择首分支",
                }
            return {"matched": None, "next": None, "reason": "无可用分支"}
        if func == "match_condition":
            condition = str(payload.get("condition") or "")
            answer = str(payload.get("answer") or "")
            lowered = condition.lower()
            if not condition.strip() or not answer.strip():
                return {"match": False}
            if any(keyword in lowered for keyword in ("肯定", "存在", "明确", "抑郁", "阳性")):
                positive_tokens = ["抑郁", "低落", "难过", "情绪不好", "沮丧", "不好"]
                positive_tokens.extend(["有", "经常", "总是", "大部分", "明显", "非常"])
                return {
                    "match": any(token in answer for token in positive_tokens)
                    and not any(token in answer for token in ("没有", "不", "未", "无"))
                }
            if any(keyword in lowered for keyword in ("否", "含糊", "模糊", "不清", "拒绝")):
                negative_tokens = ["没有", "不", "未", "无", "偶尔", "说不清", "一般", "否认", "否定", "含糊"]
                return {"match": any(token in answer for token in negative_tokens)}
            if any(keyword in lowered for keyword in ("昼夜", "波动", "早醒", "晚", "早上", "晚上")):
                oscillation_tokens = ["早上", "晚上", "白天", "夜里", "早醒", "凌晨"]
                return {"match": any(token in answer for token in oscillation_tokens)}
            tokens = [token for token in condition.replace("、", " ").replace("或", " ").split() if len(token) > 1]
            if not tokens:
                tokens = [condition]
            return {"match": any(token in answer for token in tokens)}
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

    def call(self, func: str, payload: Dict[str, Any]) -> Any:
        if not self.client or not getattr(self.client, "usable", lambda: False)():
            return _FallbackBackend().call(func, payload)

        try:
            if func == "template_builder":
                prompt = payload.get("prompt", "")
                result = self._chat_json(prompt, max_tokens=1024)
                if result:
                    return {"text": json.dumps(result, ensure_ascii=False)}
                return _FallbackBackend().call(func, payload)
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
            if func == "extract_facts":
                prompt = get_prompt("fact_extraction").format(text=payload.get("text", ""))
                return self._chat_json(prompt)
            if func == "identify_themes":
                vocab = payload.get(
                    "vocab",
                    "失落,罪恶,绝望,焦虑,兴趣缺失,睡眠异常,食欲改变,精力下降,自责,无价值感",
                )
                prompt = get_prompt("theme_identification").format(vocab=vocab, text=payload.get("text", ""))
                return self._chat_json(prompt)
            if func == "summarize_context":
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
            if func == "score_item":
                prompt = get_prompt("single_item_scoring").format(
                    item_name=payload.get("item_name", ""),
                    facts=json.dumps(payload.get("facts", {}), ensure_ascii=False),
                    themes=json.dumps(payload.get("themes", []), ensure_ascii=False),
                    summary=payload.get("summary", ""),
                    risks=json.dumps(payload.get("risks", []), ensure_ascii=False),
                    dialogue=payload.get("dialogue", ""),
                )
                return self._chat_json(prompt, max_tokens=256)
            if func == "clarify_branch":
                prompt = get_prompt("branch_clarification").format(
                    answer=payload.get("answer", ""),
                    branches=json.dumps(payload.get("branches", []), ensure_ascii=False),
                )
                return self._chat_json(prompt)
            if func == "match_condition":
                prompt = get_prompt("condition_match").format(
                    answer=payload.get("answer", ""),
                    condition=payload.get("condition", ""),
                )
                return self._chat_json(prompt)
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
        return self.backend.call(tool_name, payload)


LLM = LLMToolBox()
