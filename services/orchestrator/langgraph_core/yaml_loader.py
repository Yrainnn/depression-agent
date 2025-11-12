from __future__ import annotations

from typing import Any, List, Sequence, Tuple


def _tokenise(text: str) -> List[Tuple[int, str]]:
    tokens: List[Tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        tokens.append((indent, line.strip()))
    return tokens


def _convert(value: str) -> Any:
    if value.startswith(("'", '"')) and value.endswith(value[0]):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if value.isdigit():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _parse_block(tokens: Sequence[Tuple[int, str]], index: int, indent: int) -> Tuple[Any, int]:
    if index >= len(tokens):
        return None, index
    if tokens[index][0] < indent:
        return None, index
    if tokens[index][1].startswith("- "):
        return _parse_list(tokens, index, indent)
    return _parse_dict(tokens, index, indent)


def _parse_dict(tokens: Sequence[Tuple[int, str]], index: int, indent: int) -> Tuple[dict, int]:
    mapping: dict = {}
    while index < len(tokens):
        current_indent, content = tokens[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation for mapping at line {index + 1}")
        if content.startswith("- "):
            raise ValueError("List item not allowed at mapping level")
        if ":" not in content:
            key = content.rstrip(":")
            value = ""
        else:
            key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        index += 1
        if not value:
            if index >= len(tokens) or tokens[index][0] <= current_indent:
                mapping[key] = None
                continue
            child, index = _parse_block(tokens, index, tokens[index][0])
            mapping[key] = child
        else:
            mapping[key] = _convert(value)
    return mapping, index


def _parse_list(tokens: Sequence[Tuple[int, str]], index: int, indent: int) -> Tuple[list, int]:
    items: list = []
    while index < len(tokens):
        current_indent, content = tokens[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation for list at line {index + 1}")
        if not content.startswith("- "):
            break
        item_content = content[2:].strip()
        index += 1
        if not item_content:
            child, index = _parse_block(tokens, index, indent + 2)
            items.append(child)
            continue
        if ":" not in item_content:
            items.append(_convert(item_content))
            continue
        key, value = item_content.split(":", 1)
        key = key.strip()
        value = value.strip()
        node: dict = {}
        if value:
            node[key] = _convert(value)
        else:
            child, index = _parse_block(tokens, index, indent + 2)
            node[key] = child
        if index < len(tokens) and tokens[index][0] >= indent + 2:
            extra, index = _parse_dict(tokens, index, indent + 2)
            node.update(extra)
        items.append(node)
    return items, index


def load_yaml(text: str) -> Any:
    tokens = _tokenise(text)
    data, index = _parse_dict(tokens, 0, tokens[0][0] if tokens else 0)
    if index != len(tokens):
        raise ValueError("Unexpected trailing tokens in YAML stream")
    return data


def load_yaml_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return load_yaml(text)
