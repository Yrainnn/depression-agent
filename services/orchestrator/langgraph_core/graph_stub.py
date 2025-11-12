"""Lightweight fallback implementation mimicking langgraph's StateGraph.

This stub is only used when the real `langgraph` package is unavailable.
It supports the minimal subset required by LangGraphCoordinator: adding
nodes, conditional edges, simple linear edges, and compiling into an
object exposing an `invoke` method.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

END = object()

NodeCallable = Callable[[Dict[str, Any]], Dict[str, Any]]
ConditionalCallable = Callable[[Dict[str, Any]], str]


class _CompiledGraph:
    def __init__(self, graph: "StateGraph") -> None:
        self._graph = graph

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._graph._run(dict(state))


class StateGraph:
    """Very small subset of the langgraph StateGraph API."""

    def __init__(self, _state_type) -> None:  # pragma: no cover - signature compatibility
        self._nodes: Dict[str, NodeCallable] = {}
        self._edges: Dict[str, List[str]] = {}
        self._conditional: Dict[str, Tuple[ConditionalCallable, Dict[str, str]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, func: NodeCallable) -> None:
        self._nodes[name] = func

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def add_edge(self, start: str, end: str | object) -> None:
        self._edges.setdefault(start, []).append(end)  # type: ignore[arg-type]

    def add_conditional_edges(
        self,
        node: str,
        func: ConditionalCallable,
        mapping: Dict[str, str | object],
    ) -> None:
        # store mapping but keep only string targets; END handled at runtime
        self._conditional[node] = (func, mapping)  # type: ignore[assignment]

    def compile(self) -> _CompiledGraph:
        if self._entry is None:
            raise ValueError("entry point not set for StateGraph stub")
        return _CompiledGraph(self)

    # ------------------------------------------------------------------
    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self._entry is None:
            raise ValueError("entry point not configured")
        current = self._entry
        while current is not END:
            node_fn = self._nodes[current]
            update = node_fn(state)
            if update:
                state.update(update)

            if current in self._conditional:
                func, mapping = self._conditional[current]
                key = func(state)
                target = mapping[key]
                if target is END:
                    break
                current = target  # type: ignore[assignment]
                continue

            next_edges = self._edges.get(current, [])
            if not next_edges:
                break
            target = next_edges[0]
            if target is END:
                break
            current = target  # type: ignore[assignment]
        return state
