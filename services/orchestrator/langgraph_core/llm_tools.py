from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from services.orchestrator.prompts import get_prompt

try:  # pragma: no cover - runtime wiring best effort
    from services.llm.json_client import client as _deepseek_client
except Exception:  # pragma: no cover - fallback when DeepSeek is unavailable
    _deepseek_client = None


class _ToolKey:
    """Marker base class for LLM toolbox tool identifiers."""


class TemplateBuilderTool(_ToolKey):
    """Generate YAML templates and strategy descriptions from natural language."""


class GenerateTool(_ToolKey):
    """Produce the next strategy question."""


class RiskDetectTool(_ToolKey):
    """Detect acute risk level from user responses."""


class ExtractFactsTool(_ToolKey):
    """Extract structured facts from the latest utterance."""


class IdentifyThemesTool(_ToolKey):
    """Identify therapeutic themes in user responses."""


class SummarizeContextTool(_ToolKey):
    """Summarize rolling context for strategy planning."""


class ScoreItemTool(_ToolKey):
    """Score a single assessment item."""


class ClarifyBranchTool(_ToolKey):
    """Select follow-up branches during clarification."""


class MatchConditionTool(_ToolKey):
    """Match user answers against branch conditions."""


class _BaseBackend(ABC):
    """Abstract backend definition for tool invocations."""

    @abstractmethod
    def call(self, tool: Type[_ToolKey], payload: Dict[str, Any]) -> Any:
        raise NotImplementedError


class _FallbackToolStrategy(ABC):
    """Strategy definition for the fallback backend."""

    tool: Type[_ToolKey]

    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> Any:
        raise NotImplementedError


_FALLBACK_STRATEGIES: Dict[Type[_ToolKey], Type[_FallbackToolStrategy]] = {}


def _register_fallback_strategy(tool: Type[_ToolKey]):
    def decorator(cls: Type[_FallbackToolStrategy]) -> Type[_FallbackToolStrategy]:
        cls.tool = tool
        _FALLBACK_STRATEGIES[tool] = cls
        return cls

    return decorator


@_register_fallback_strategy(TemplateBuilderTool)
class _TemplateBuilderFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
        stub = {
            "yaml": "project_id: 0\nstrategies: {}\n",
            "strategy_descriptions": {},
        }
        return {"text": json.dumps(stub, ensure_ascii=False)}


@_register_fallback_strategy(GenerateTool)
class _GenerateFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
        template = payload.get("template") or "最近两周您的心情是否低落？"
        question = template.strip()
        if not question.endswith("？") and not question.endswith("?"):
            question += "？"
        return {"text": question}


@_register_fallback_strategy(RiskDetectTool)
class _RiskDetectFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
        text = payload.get("text", "")
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


@_register_fallback_strategy(ExtractFactsTool)
class _ExtractFactsFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
        text = payload.get("text", "")
        facts: Dict[str, Any] = {}
        if "分" in text:
            import re

            if match := re.search(r"(\d+)分", text):
                facts["self_rating"] = int(match.group(1))
        if any(token in text for token in ("凌晨", "早醒")):
            facts["sleep_pattern"] = "early_awake"
        return {"facts": facts}


@_register_fallback_strategy(IdentifyThemesTool)
class _IdentifyThemesFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
        text = payload.get("text", "")
        themes = []
        if any(token in text for token in ("绝望", "没意思", "无望")):
            themes.append("绝望")
        if "凌晨" in text or "早醒" in text:
            themes.append("睡眠异常")
        return {"themes": themes}


@_register_fallback_strategy(SummarizeContextTool)
class _SummarizeContextFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
        prev = payload.get("prev", "")
        new = payload.get("new", "")
        limit = int(payload.get("limit", 500))
        summary = (prev + " " + new).strip()
        return {"summary": summary[:limit]}


@_register_fallback_strategy(ScoreItemTool)
class _ScoreItemFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
        return {"score": 2}


@_register_fallback_strategy(ClarifyBranchTool)
class _ClarifyBranchFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
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


@_register_fallback_strategy(MatchConditionTool)
class _MatchConditionFallbackStrategy(_FallbackToolStrategy):

    def run(self, payload: Dict[str, Any]) -> Any:
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


class _FallbackBackend(_BaseBackend):
    """Offline fallback used during tests or when the LLM is unavailable."""

    def __init__(self) -> None:
        self._strategies = {
            tool: strategy_cls() for tool, strategy_cls in _FALLBACK_STRATEGIES.items()
        }

    def call(self, tool: Type[_ToolKey], payload: Dict[str, Any]) -> Any:
        strategy = self._strategies.get(tool)
        if not strategy:
            raise ValueError(f"Unknown tool function: {tool}")
        return strategy.run(payload)


class _DeepSeekToolStrategy(ABC):
    """Strategy definition for DeepSeek backed tools."""

    tool: Type[_ToolKey]

    @abstractmethod
    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        raise NotImplementedError


_DEEPSEEK_STRATEGIES: Dict[Type[_ToolKey], Type[_DeepSeekToolStrategy]] = {}


def _register_deepseek_strategy(tool: Type[_ToolKey]):
    def decorator(cls: Type[_DeepSeekToolStrategy]) -> Type[_DeepSeekToolStrategy]:
        cls.tool = tool
        _DEEPSEEK_STRATEGIES[tool] = cls
        return cls

    return decorator


@_register_deepseek_strategy(TemplateBuilderTool)
class _TemplateBuilderDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        prompt = payload.get("prompt", "")
        result = backend._chat_json(prompt, max_tokens=1024)
        if result:
            return {"text": json.dumps(result, ensure_ascii=False)}
        return backend._fallback.call(self.tool, payload)


@_register_deepseek_strategy(GenerateTool)
class _GenerateDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        prompt = get_prompt("strategy_generation").format(
            context=payload.get("context", ""),
            template=payload.get("template", ""),
        )
        try:
            text = backend.client.call(  # type: ignore[attr-defined]
                prompt,
                max_tokens=256,
                temperature=0.2,
            )
        except Exception:
            return backend._fallback.call(self.tool, payload)
        question = (text or "").strip()
        if question:
            return {"text": question}
        return backend._fallback.call(self.tool, payload)


@_register_deepseek_strategy(RiskDetectTool)
class _RiskDetectDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        prompt = get_prompt("risk_detection").format(text=payload.get("text", ""))
        result = backend._chat_json(prompt)
        return result or backend._fallback.call(self.tool, payload)


@_register_deepseek_strategy(ExtractFactsTool)
class _ExtractFactsDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        prompt = get_prompt("fact_extraction").format(text=payload.get("text", ""))
        result = backend._chat_json(prompt)
        return result or backend._fallback.call(self.tool, payload)


@_register_deepseek_strategy(IdentifyThemesTool)
class _IdentifyThemesDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        vocab = payload.get(
            "vocab",
            "失落,罪恶,绝望,焦虑,兴趣缺失,睡眠异常,食欲改变,精力下降,自责,无价值感",
        )
        prompt = get_prompt("theme_identification").format(vocab=vocab, text=payload.get("text", ""))
        result = backend._chat_json(prompt)
        return result or backend._fallback.call(self.tool, payload)


@_register_deepseek_strategy(SummarizeContextTool)
class _SummarizeContextDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        prompt = get_prompt("rolling_summary").format(
            prev=payload.get("prev", ""),
            new=payload.get("new", ""),
        )
        result = backend._chat_json(prompt)
        if result:
            limit = int(payload.get("limit", 500))
            summary = result.get("summary")
            if isinstance(summary, str):
                result["summary"] = summary[:limit]
            return result
        return backend._fallback.call(self.tool, payload)


@_register_deepseek_strategy(ScoreItemTool)
class _ScoreItemDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        prompt = get_prompt("single_item_scoring").format(
            item_name=payload.get("item_name", ""),
            facts=json.dumps(payload.get("facts", {}), ensure_ascii=False),
            themes=json.dumps(payload.get("themes", []), ensure_ascii=False),
            summary=payload.get("summary", ""),
            risks=json.dumps(payload.get("risks", []), ensure_ascii=False),
            dialogue=payload.get("dialogue", ""),
        )
        result = backend._chat_json(prompt, max_tokens=256)
        return result or backend._fallback.call(self.tool, payload)


@_register_deepseek_strategy(ClarifyBranchTool)
class _ClarifyBranchDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        prompt = get_prompt("branch_clarification").format(
            answer=payload.get("answer", ""),
            branches=json.dumps(payload.get("branches", []), ensure_ascii=False),
        )
        result = backend._chat_json(prompt)
        return result or backend._fallback.call(self.tool, payload)


@_register_deepseek_strategy(MatchConditionTool)
class _MatchConditionDeepSeekStrategy(_DeepSeekToolStrategy):

    def run(self, backend: "_DeepSeekBackend", payload: Dict[str, Any]) -> Any:
        prompt = get_prompt("condition_match").format(
            answer=payload.get("answer", ""),
            condition=payload.get("condition", ""),
        )
        result = backend._chat_json(prompt)
        return result or backend._fallback.call(self.tool, payload)


class _DeepSeekBackend(_BaseBackend):
    """Backend delegating to the DeepSeek JSON client."""

    def __init__(self) -> None:
        self.client = _deepseek_client
        self._fallback = _FallbackBackend()
        self._strategies = {
            tool: strategy_cls() for tool, strategy_cls in _DEEPSEEK_STRATEGIES.items()
        }

    def _chat_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.client or not getattr(self.client, "usable", lambda: False)():
            return {}
        try:
            return self.client.call_json(  # type: ignore[attr-defined]
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception:
            return {}

    def call(self, tool: Type[_ToolKey], payload: Dict[str, Any]) -> Any:
        if not self.client or not getattr(self.client, "usable", lambda: False)():
            return self._fallback.call(tool, payload)

        strategy = self._strategies.get(tool)
        if not strategy:
            return self._fallback.call(tool, payload)

        try:
            result = strategy.run(self, payload)
        except Exception:
            return self._fallback.call(tool, payload)

        if result is None:
            return self._fallback.call(tool, payload)
        return result


class LLMToolBox:
    """Unified entry point for LangGraph LLM utilities."""

    def __init__(self) -> None:
        self.backend: _BaseBackend
        if _deepseek_client:
            self.backend = _DeepSeekBackend()
        else:
            self.backend = _FallbackBackend()

    def call(self, tool: Type[_ToolKey], payload: Dict[str, Any]) -> Any:
        return self.backend.call(tool, payload)


LLM = LLMToolBox()

__all__ = [
    "LLM",
    "LLMToolBox",
    "TemplateBuilderTool",
    "GenerateTool",
    "RiskDetectTool",
    "ExtractFactsTool",
    "IdentifyThemesTool",
    "SummarizeContextTool",
    "ScoreItemTool",
    "ClarifyBranchTool",
    "MatchConditionTool",
]
