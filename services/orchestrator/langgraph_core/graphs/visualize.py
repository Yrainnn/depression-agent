from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .main_graph import create_state_graph


def _edge_target(edge: Any) -> str:
    if hasattr(edge, "target"):
        return getattr(edge, "target")
    if isinstance(edge, dict):
        return edge.get("target", "")
    return str(edge)


def export_mermaid(template_root: str, output: Path) -> Path:
    """Generate a Mermaid flowchart describing the compiled LangGraph."""

    state_graph = create_state_graph(template_root)

    if not hasattr(state_graph, "_nodes"):
        raise RuntimeError(
            "StateGraph implementation does not expose internal edges; "
            "use the official langgraph visualizer instead."
        )

    nodes = getattr(state_graph, "_nodes")
    edges = getattr(state_graph, "_edges")
    conditionals = getattr(state_graph, "_conditionals")

    lines: list[str] = ["flowchart TD"]

    for name in nodes:
        lines.append(f"    {name}[{name}]")

    for start, targets in edges.items():
        for edge in targets:
            target = _edge_target(edge)
            if not target:
                continue
            lines.append(f"    {start} --> {target}")

    for node, (_selector, mapping) in conditionals.items():
        for label, target in mapping.items():
            lines.append(f"    {node} -->|{label}| {target}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_structure(template_root: str, output: Path) -> Path:
    """Dump the raw adjacency structure to JSON for inspection."""

    state_graph = create_state_graph(template_root)

    if not hasattr(state_graph, "_nodes"):
        raise RuntimeError(
            "StateGraph implementation does not expose internal edges; "
            "use the official langgraph visualizer instead."
        )

    payload = {
        "nodes": sorted(getattr(state_graph, "_nodes").keys()),
        "edges": {
            start: [
                _edge_target(edge)
                for edge in getattr(state_graph, "_edges").get(start, [])
            ]
            for start in getattr(state_graph, "_nodes").keys()
        },
        "conditionals": {
            node: mapping for node, (_selector, mapping) in getattr(state_graph, "_conditionals").items()
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:  # pragma: no cover - CLI utility
    import argparse

    parser = argparse.ArgumentParser(description="Export LangGraph topology visualizations.")
    parser.add_argument("--template-root", default=str(Path(__file__).resolve().parents[2] / "config"))
    parser.add_argument("--mermaid", type=Path, help="Path to output Mermaid diagram", default=None)
    parser.add_argument("--json", type=Path, help="Path to output JSON structure", default=None)
    args = parser.parse_args()

    if args.mermaid is None and args.json is None:
        raise SystemExit("Specify --mermaid and/or --json output paths")

    if args.mermaid is not None:
        path = export_mermaid(args.template_root, args.mermaid)
        print(f"[mermaid] wrote {path}")

    if args.json is not None:
        path = export_structure(args.template_root, args.json)
        print(f"[json] wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
