#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML is absent
    yaml = None

_DEFAULT_PIPELINE = [
    {
        "id": "S2",
        "name": "问题簇验证",
        "exec_node": "初筛",
        "template": "最近两周，大部分时间您的心情是不是都很低落、忧郁？",
        "branches": [
            {"condition": "明确存在抑郁情绪", "next": "S4"},
            {
                "condition": "否定或含糊",
                "next": "S10",
                "next_prompt": "情绪没受影响，那有没有觉得每天过得很空洞、乏味？",
            },
        ],
    },
    {
        "id": "S4",
        "name": "比较框架",
        "exec_node": "情绪确认后",
        "template": "您觉得现在的难过跟亲人去世那种悲伤比起来，是一样严重还是不太一样？",
    },
    {
        "id": "S9",
        "name": "时间弹性",
        "exec_node": "补充追问",
        "template": "这种情绪一天当中有变化吗？比如早上是不是特别重，晚上会不会好一些？",
        "branches": [{"condition": "提及昼夜波动", "next": "S8"}],
    },
    {
        "id": "S8",
        "name": "回声确认",
        "exec_node": "关键信息后",
        "template": "听您说{复述患者表述}，我理解对吗？",
    },
    {
        "id": "S10",
        "name": "非对立追问",
        "exec_node": "备选",
        "template": "那有没有觉得每天过得很空洞、乏味？",
    },
]


def _parse_spec(text: str, item_id: int, fallback_name: str) -> Dict[str, Any]:
    match = re.search(r"项目名称[:：]\s*([^\n]+)", text)
    name = match.group(1).strip() if match else fallback_name
    return {
        "project_id": item_id,
        "project_name": name or f"item_{item_id}",
        "risk_level": "中等",
        "response_time": "<=30s",
        "strategies": _DEFAULT_PIPELINE,
    }


def build_template(root: str, item_id: int, spec_path: str, name: str = "") -> str:
    with open(spec_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    data = _parse_spec(text, item_id, name or f"item_{item_id}")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"item_{item_id:02d}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        _dump_yaml(data, handle)
    return path


def build_all(root: str, total: int = 17) -> None:
    os.makedirs(root, exist_ok=True)
    for item_id in range(1, total + 1):
        path = os.path.join(root, f"item_{item_id:02d}.yaml")
        if os.path.exists(path):
            continue
        data = {
            "project_id": item_id,
            "project_name": f"item_{item_id}",
            "risk_level": "中等",
            "response_time": "<=30s",
            "strategies": _DEFAULT_PIPELINE,
        }
          with open(path, "w", encoding="utf-8") as handle:
              _dump_yaml(data, handle)
 

def _dump_yaml(data: Dict[str, Any], handle) -> None:
    if yaml:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    else:  # pragma: no cover - fallback path
        import json

        handle.write(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LangGraph YAML templates from natural language specs.")
    parser.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "config"))
    parser.add_argument("--spec", help="Path to the natural-language specification file")
    parser.add_argument("--id", type=int, default=1, help="Item identifier")
    parser.add_argument("--name", default="", help="Override project name")
    parser.add_argument("--all", action="store_true", help="Generate default templates for 1..17 if missing")
    args = parser.parse_args()

    if args.all:
        build_all(args.root)
        print(f"Generated default templates into {args.root}")
        return

    if not args.spec:
        raise SystemExit("--spec is required unless --all is provided")

    path = build_template(args.root, args.id, args.spec, args.name)
    print(f"Generated {path}")


if __name__ == "__main__":
    main()
