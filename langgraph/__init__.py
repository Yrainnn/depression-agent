"""Local compatibility shim for the external ``langgraph`` package."""

from .graph import END, StateGraph  # noqa: F401

__all__ = ["END", "StateGraph"]
