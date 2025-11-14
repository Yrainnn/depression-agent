from __future__ import annotations

import json
import os
import runpy
from typing import Dict, Mapping, Optional


StrategyDescriptions = Mapping[str, Mapping[str, str]]

try:  # pragma: no cover - prefer PyYAML when available
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback to internal dumper
    yaml = None

from .llm_tools import LLM, TemplateBuilderTool
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

    def __init__(self, output_root: str, *, descriptions_path: Optional[str] = None) -> None:
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        default_descriptions_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "prompts", "strategy_descriptions.py")
        )
        self.descriptions_path = descriptions_path or default_descriptions_path
        os.makedirs(os.path.dirname(self.descriptions_path), exist_ok=True)

    def build_from_text(
        self, text: str, item_id: int, *, item_name: Optional[str] = None
    ) -> str:
        prompt = (
            "请将以下策略稿转为结构化YAML配置，并额外输出每个策略的描述信息。\n"
            "生成JSON对象，包含字段：\n"
            "- \"yaml\": YAML文本，描述题目的策略模板；\n"
            "- \"strategy_descriptions\": 一个字典，键为策略ID，值为包含\"name\"、\"description\"、\"tone\"的对象；\n"
            "请确保策略描述清晰说明策略的临床目的和执行方式，并指明对话语气要求。\n\n"
            "原始策略稿如下：\n"
            f"{text}"
        )
        response = LLM.call(TemplateBuilderTool, {"prompt": prompt})
        payload_text = response.get("text", "") if isinstance(response, dict) else ""
        if not payload_text.strip():
            raise ValueError("LLM did not return JSON content for template building")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("LLM returned invalid JSON for template building") from exc

        yaml_text = str(payload.get("yaml", "")).strip()
        if not yaml_text:
            raise ValueError("LLM response missing YAML content")

        if yaml:
            data = yaml.safe_load(yaml_text)
        else:
            data = _safe_load_fallback(yaml_text)
        data.setdefault("item_id", item_id)
        if item_name:
            data["name"] = item_name

        descriptions = payload.get("strategy_descriptions", {})
        normalized_descriptions = self._normalize_descriptions(descriptions)
        if normalized_descriptions:
            self._update_descriptions(normalized_descriptions)

        path = os.path.join(self.output_root, f"item_{item_id:02d}.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            if yaml:
                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
            else:
                fh.write(_dump_yaml(data) + "\n")
        return path

    def _normalize_descriptions(self, descriptions: object) -> Dict[str, Dict[str, str]]:
        if descriptions is None:
            return {}
        if not isinstance(descriptions, Mapping):
            raise ValueError("LLM response missing strategy descriptions map")
        normalized: Dict[str, Dict[str, str]] = {}
        for key, value in descriptions.items():
            if not isinstance(value, Mapping):
                raise ValueError("Strategy description must be an object with fields")
            entry: Dict[str, str] = {}
            for field, text in value.items():
                if text is None:
                    continue
                entry[str(field)] = str(text).strip()
            normalized[str(key)] = entry
        return normalized

    def _load_existing_descriptions(self) -> Dict[str, Dict[str, str]]:
        if not os.path.exists(self.descriptions_path):
            return {}
        try:
            module_data = runpy.run_path(self.descriptions_path)
            loaded = module_data.get("STRATEGY_DESCRIPTIONS", {})
            if isinstance(loaded, Mapping):
                return {str(k): dict(v) for k, v in loaded.items() if isinstance(v, Mapping)}
        except Exception:  # pragma: no cover - fallback for invalid file
            return {}
        return {}

    def _update_descriptions(self, additions: StrategyDescriptions) -> None:
        existing = self._load_existing_descriptions()
        updated = False
        for key, value in additions.items():
            current = existing.get(key)
            if current != dict(value):
                existing[key] = dict(value)
                updated = True
        if not updated:
            return
        rendered = self._render_descriptions(existing)
        with open(self.descriptions_path, "w", encoding="utf-8") as fh:
            fh.write(rendered)

    def _render_descriptions(self, descriptions: Mapping[str, Mapping[str, str]]) -> str:
        lines = [
            '"""Strategy description templates used to enrich LLM prompts."""',
            "",
            "from __future__ import annotations",
            "",
            "from typing import Dict, Mapping",
            "",
            "StrategyDescription = Mapping[str, str]",
            "",
            "STRATEGY_DESCRIPTIONS: Dict[str, StrategyDescription] = {",
        ]
        for strategy_id in sorted(descriptions):
            entry = descriptions[strategy_id]
            lines.append(f'    "{strategy_id}": {{')
            preferred_order = ["name", "description", "tone"]
            seen = set()
            for field in preferred_order:
                if field in entry:
                    value = entry[field]
                    seen.add(field)
                    lines.append(self._format_field(field, value))
            for field in sorted(k for k in entry if k not in seen):
                value = entry[field]
                lines.append(self._format_field(field, value))
            lines.append("    },")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _format_field(self, field: str, value: str) -> str:
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'        "{field}": "{escaped}",' 
