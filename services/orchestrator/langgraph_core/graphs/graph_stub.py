from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

END = "__end__"


class _CompiledGraph:
    def __init__(
        self,
        entry: str,
        nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        edges: Dict[str, List[str]],
        conditionals: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]],
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
                if key not in mapping:
                    raise RuntimeError(f"conditional edge from {current} missing key {key}")
                current = mapping[key]
                continue
            next_nodes = self._edges.get(current, [])
            if not next_nodes:
                current = END
            else:
                current = next_nodes[0]
        return state


class StateGraph:
    def __init__(self, _: Any):
        self._nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._edges: Dict[str, List[str]] = {}
        self._conditionals: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self._nodes[name] = fn

    def add_edge(self, start: str, end: str) -> None:
        self._edges.setdefault(start, []).append(end)

    def add_conditional_edges(
        self,
        start: str,
        selector: Callable[[Dict[str, Any]], str],
        mapping: Dict[str, str],
    ) -> None:
        self._conditionals[start] = (selector, mapping)

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def compile(self) -> _CompiledGraph:
        if self._entry is None:
            raise RuntimeError("entry point not set")
        return _CompiledGraph(self._entry, self._nodes, self._edges, self._conditionals)
