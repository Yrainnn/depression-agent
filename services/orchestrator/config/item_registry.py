from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - prefer PyYAML when available
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback parser
    yaml = None

def _parse_scalar(value: str):
    if not value:
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _tokenize_yaml(text: str):
    tokens = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        tokens.append((indent, raw.strip()))
    return tokens


def _parse_structure(tokens, index, indent):
    if index >= len(tokens):
        return None, index
    cur_indent, content = tokens[index]
    if cur_indent < indent:
        return None, index
    if content.startswith("- "):
        return _parse_list(tokens, index, indent)
    return _parse_dict(tokens, index, indent)


def _parse_dict(tokens, index, indent):
    result = {}
    i = index
    while i < len(tokens):
        cur_indent, content = tokens[i]
        if cur_indent < indent or content.startswith("- "):
            break
        key, _, value_part = content.partition(":")
        key = key.strip()
        value_part = value_part.strip()
        i += 1
        if not value_part:
            if i < len(tokens) and tokens[i][0] > cur_indent:
                value, i = _parse_structure(tokens, i, tokens[i][0])
            else:
                value = None
        else:
            value = _parse_scalar(value_part)
        result[key] = value
    return result, i


def _parse_list(tokens, index, indent):
    items = []
    i = index
    while i < len(tokens):
        cur_indent, content = tokens[i]
        if cur_indent < indent or not content.startswith("- "):
            break
        item_content = content[2:].strip()
        i += 1
        item_value = None
        if item_content:
            if ":" in item_content:
                key, _, value_part = item_content.partition(":")
                key = key.strip()
                value_part = value_part.strip()
                item_value = {key: _parse_scalar(value_part) if value_part else None}
            else:
                item_value = _parse_scalar(item_content)
        if i < len(tokens) and tokens[i][0] > cur_indent:
            nested, i = _parse_structure(tokens, i, tokens[i][0])
            if isinstance(item_value, dict) and isinstance(nested, dict):
                item_value.update(nested)
            elif isinstance(item_value, dict) and isinstance(nested, list):
                last_key = next(reversed(item_value))
                item_value[last_key] = nested
            elif item_value is None:
                item_value = nested
        items.append(item_value)
    return items, i


def _safe_load_fallback(text: str):
    tokens = _tokenize_yaml(text)
    value, _ = _parse_structure(tokens, 0, 0)
    if isinstance(value, dict):
        return value
    raise ValueError("Unable to parse YAML content")


CONFIG_DIR = Path(__file__).resolve().parent


# 官方 HAMD-17 每个条目的最高得分
DEFAULT_MAX_SCORES: Dict[int, int] = {
    1: 4,
    2: 4,
    3: 4,
    4: 2,
    5: 2,
    6: 2,
    7: 4,
    8: 4,
    9: 4,
    10: 4,
    11: 4,
    12: 2,
    13: 2,
    14: 2,
    15: 4,
    16: 2,
    17: 4,
}


@dataclass(frozen=True)
class ItemConfig:
    item_id: int
    name: str
    max_score: Optional[int]


def _parse_item_id_from_filename(path: Path) -> Optional[int]:
    match = re.search(r"item_(\d+)", path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - defensive guard
        return None


def _load_yaml(path: Path) -> Dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:  # pragma: no cover - defensive guard
        return {}

    if yaml:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:  # pragma: no cover - invalid yaml should not break import
            data = None
    else:
        try:
            data = _safe_load_fallback(text)
        except ValueError:  # pragma: no cover - invalid fallback parse
            data = None

    return data or {}


def _discover_item_configs() -> Dict[int, ItemConfig]:
    registry: Dict[int, ItemConfig] = {
        item_id: ItemConfig(
            item_id=item_id,
            name=f"条目 {item_id:02d}",
            max_score=DEFAULT_MAX_SCORES.get(item_id),
        )
        for item_id in sorted(DEFAULT_MAX_SCORES)
    }

    for path in sorted(CONFIG_DIR.glob("item_*.yaml")):
        data = _load_yaml(path)

        raw_id = data.get("item_id")
        item_id: Optional[int]
        if isinstance(raw_id, int):
            item_id = raw_id
        elif isinstance(raw_id, str) and raw_id.strip().isdigit():
            item_id = int(raw_id.strip())
        else:
            item_id = _parse_item_id_from_filename(path)

        if item_id is None:
            continue

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            name = registry.get(item_id, ItemConfig(item_id, f"条目 {item_id:02d}", None)).name

        max_score_raw = data.get("max_score")
        if isinstance(max_score_raw, (int, float)):
            max_score = int(max_score_raw)
        else:
            max_score = DEFAULT_MAX_SCORES.get(item_id)

        registry[item_id] = ItemConfig(item_id=item_id, name=name.strip(), max_score=max_score)

    return registry


ITEM_CONFIGS: Dict[int, ItemConfig] = _discover_item_configs()
ITEM_IDS: List[int] = sorted(ITEM_CONFIGS)
ITEM_MAX_SCORES: Dict[int, int] = {
    item_id: cfg.max_score
    for item_id, cfg in ITEM_CONFIGS.items()
    if cfg.max_score is not None
}


def get_item_config(item_id: int) -> ItemConfig:
    return ITEM_CONFIGS.get(
        item_id,
        ItemConfig(
            item_id=item_id,
            name=f"条目 {item_id:02d}",
            max_score=DEFAULT_MAX_SCORES.get(item_id),
        ),
    )


def get_item_name(item_id: int) -> str:
    return get_item_config(item_id).name


def get_item_max_score(item_id: int) -> Optional[int]:
    return get_item_config(item_id).max_score


def iter_item_ids() -> Iterable[int]:
    return list(ITEM_IDS)


MAX_TOTAL_SCORE: int = sum(score for score in ITEM_MAX_SCORES.values() if score is not None) or sum(
    DEFAULT_MAX_SCORES.values()
)


__all__ = [
    "ItemConfig",
    "ITEM_CONFIGS",
    "ITEM_IDS",
    "ITEM_MAX_SCORES",
    "MAX_TOTAL_SCORE",
    "DEFAULT_MAX_SCORES",
    "get_item_config",
    "get_item_name",
    "get_item_max_score",
    "iter_item_ids",
]

