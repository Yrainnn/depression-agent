from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover - prefer installed PyYAML
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback parser
    yaml = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    import datetime

    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_jsonl(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_snapshot(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _parse_scalar(value: str) -> Any:
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


def _tokenize_yaml(text: str) -> List[Tuple[int, str]]:
    tokens: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        tokens.append((indent, raw.strip()))
    return tokens


def _parse_structure(tokens: List[Tuple[int, str]], index: int, indent: int) -> Tuple[Any, int]:
    if index >= len(tokens):
        return None, index
    cur_indent, content = tokens[index]
    if cur_indent < indent:
        return None, index
    if content.startswith("- "):
        return _parse_list(tokens, index, indent)
    return _parse_dict(tokens, index, indent)


def _parse_dict(tokens: List[Tuple[int, str]], index: int, indent: int) -> Tuple[Dict[str, Any], int]:
    result: Dict[str, Any] = {}
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


def _parse_list(tokens: List[Tuple[int, str]], index: int, indent: int) -> Tuple[List[Any], int]:
    items: List[Any] = []
    i = index
    while i < len(tokens):
        cur_indent, content = tokens[i]
        if cur_indent < indent or not content.startswith("- "):
            break
        item_content = content[2:].strip()
        i += 1
        item_value: Any = None
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
                # attach list to last key when nested returns list
                last_key = next(reversed(item_value))
                item_value[last_key] = nested
            elif item_value is None:
                item_value = nested
        items.append(item_value)
    return items, i


def _safe_load_fallback(text: str) -> Dict[str, Any]:
    tokens = _tokenize_yaml(text)
    value, _ = _parse_structure(tokens, 0, 0)
    if isinstance(value, dict):
        return value
    raise ValueError("Unable to parse YAML content")


def load_template(root: str, index: int) -> Dict[str, Any]:
    path = os.path.join(root, f"item_{index:02d}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if yaml:
        return yaml.safe_load(text)
    return _safe_load_fallback(text)
