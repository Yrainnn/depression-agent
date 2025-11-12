from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..state_types import SessionState


class Node(ABC):
    """所有节点的基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, state: SessionState, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
