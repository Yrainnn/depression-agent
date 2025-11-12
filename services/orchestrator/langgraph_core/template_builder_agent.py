from __future__ import annotations

import os
from typing import Optional

try:  # pragma: no cover - prefer PyYAML when available
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback to internal dumper
    yaml = None

from .llm_tools import LLM
from .utils import _safe_load_fallback


def _format_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in [":", "\"", " "]):
        return f'"{text}"'
    return text


def _dump_yaml(data: object, indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_dump_yaml(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_format_scalar(value)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_dump_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
    else:
        lines.append(f"{pad}{_format_scalar(data)}")
    return "\n".join(line for line in lines if line)


class TemplateBuilderAgent:
    """Convert natural language specifications into YAML item templates."""

    def __init__(self, output_root: str) -> None:
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def build_from_text(self, text: str, item_id: int, *, project_name: Optional[str] = None) -> str:
        prompt = "请将以下策略稿转为结构化YAML配置：\n" + text
        response = LLM.call("generate", {"template": prompt})
        yaml_text = response.get("text", "") if isinstance(response, dict) else ""
        if not yaml_text.strip():
            raise ValueError("LLM did not return YAML content")
        if yaml:
            data = yaml.safe_load(yaml_text)
        else:
            data = _safe_load_fallback(yaml_text)
        data.setdefault("project_id", item_id)
        if project_name:
            data["project_name"] = project_name
        path = os.path.join(self.output_root, f"item_{item_id:02d}.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            if yaml:
                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
            else:
                fh.write(_dump_yaml(data) + "\n")
        return path
