from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

END = "__end__"

NodeFn = Callable[[Dict[str, Any]], Dict[str, Any]]
SelectorFn = Callable[[Dict[str, Any]], str]


@dataclass
class _Edge:
    target: str


class _CompiledGraph:
    def __init__(
        self,
        entry: str,
        nodes: Dict[str, NodeFn],
        edges: Dict[str, List[_Edge]],
        conditionals: Dict[str, Tuple[SelectorFn, Dict[str, str]]],
    ) -> None:
        self._entry = entry
        self._nodes = nodes
        self._edges = edges
        self._conditionals = conditionals

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current = self._entry
        while current and current != END:
            node_fn = self._nodes[current]
            state = node_fn(state)

            if current in self._conditionals:
                selector, mapping = self._conditionals[current]
                key = selector(state)
                target = mapping.get(key, END)
                current = target
                continue

            next_edges = self._edges.get(current, [])
            if not next_edges:
                break
            current = next_edges[0].target
        return state


class StateGraph:
    """Compatibility implementation of :class:`langgraph.graph.StateGraph`."""

    def __init__(self, _state_type: Any) -> None:  # pragma: no cover
        self._nodes: Dict[str, NodeFn] = {}
        self._edges: Dict[str, List[_Edge]] = {}
        self._conditionals: Dict[str, Tuple[SelectorFn, Dict[str, str]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> None:
        self._nodes[name] = fn

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def add_edge(self, start: str, end: str) -> None:
        self._edges.setdefault(start, []).append(_Edge(end))

    def add_conditional_edges(
        self,
        node: str,
        selector: SelectorFn,
        mapping: Dict[str, str],
    ) -> None:
        self._conditionals[node] = (selector, mapping)

    def compile(self) -> _CompiledGraph:
        if self._entry is None:
            raise ValueError("entry point not set")
        return _CompiledGraph(self._entry, self._nodes, self._edges, self._conditionals)
