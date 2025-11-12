from __future__ import annotations

"""Central registry for orchestrator prompt templates."""

from typing import Dict

_PROMPTS: Dict[str, str] = {
    "theme_identification": (
        "你是精神科临床记录助手。请从患者这段话中识别叙事主题，\n"
        "严格从以下词表中选择（可多选，按出现重要性排序）：\n"
        "词表：{vocab}\n"
        "文本：{text}\n"
        "仅输出 JSON：{\"themes\": [\"...\"]}\n"
    ),
    "rolling_summary": (
        "你是精神科‘滚动摘要’助手。请把‘新增陈述’融合进‘既有摘要’，删除冗余与无关信息，\n"
        "保留风险相关表述；限制在 {limit} 字以内。\n"
        "既有摘要：{prev}\n"
        "新增陈述：{new}\n"
        "仅输出 JSON：{\"summary\": \"...\"}\n"
    ),
    "question_generation": (
        "你是一名精神科医生。根据策略模板与短期上下文，生成一句简洁、自然的中文问题。\n"
        "【短期上下文】\n"
        "{context}\n\n"
        "【策略】{strategy_id} {strategy_name}\n"
        "【问句模板】{template}\n"
        "仅输出一句问题，不要解释。\n"
    ),
}


def get_prompt(key: str) -> str:
    """Return the template string registered under ``key``."""

    return _PROMPTS.get(key, "")
