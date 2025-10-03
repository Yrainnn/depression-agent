import json
import logging
from typing import Any, Dict, List, Optional

from packages.common.config import settings

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore


LOGGER = logging.getLogger(__name__)


class ConversationRepository:
    """Persistence layer for conversation state stored in Redis."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis_url = redis_url or settings.redis_url
        self._client = self._create_client()
        self._memory_store: Dict[str, str] = {}

    def _create_client(self) -> Optional["redis.Redis"]:
        if not redis:
            LOGGER.warning("redis package not available, falling back to in-memory store")
            return None
        try:
            client = redis.from_url(self._redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Failed to connect to Redis at %s: %s", self._redis_url, exc)
            return None

    # Generic helpers -----------------------------------------------------
    def _get(self, key: str) -> Optional[Dict[str, Any]]:
        raw: Optional[str]
        if self._client is not None:
            raw = self._client.get(key)
        else:
            raw = self._memory_store.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.error("Malformed JSON for key %s", key)
            return None

    def _set(self, key: str, value: Dict[str, Any]) -> None:
        serialized = json.dumps(value, ensure_ascii=False)
        if self._client is not None:
            self._client.set(key, serialized)
        else:
            self._memory_store[key] = serialized

    def _append_list(self, key: str, item: Any) -> None:
        data = self._get(key) or {"items": []}
        items: List[Any] = data.get("items", [])
        items.append(item)
        data["items"] = items
        self._set(key, data)

    def _get_list(self, key: str) -> List[Any]:
        data = self._get(key) or {"items": []}
        return data.get("items", [])

    # Session state -------------------------------------------------------
    def load_session_state(self, session_id: str) -> Dict[str, Any]:
        return self._get(f"session:{session_id}") or {}

    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        self._set(f"session:{session_id}", state)

    # Scores --------------------------------------------------------------
    def save_scores(self, session_id: str, scores: List[Dict[str, Any]]) -> None:
        self._set(f"score:{session_id}", {"items": scores})

    def load_scores(self, session_id: str) -> List[Dict[str, Any]]:
        return self._get_list(f"score:{session_id}")

    # Transcripts ---------------------------------------------------------
    def append_transcript(self, session_id: str, segment: Dict[str, Any]) -> None:
        self._append_list(f"transcripts:{session_id}", segment)

    def load_transcripts(self, session_id: str) -> List[Dict[str, Any]]:
        return self._get_list(f"transcripts:{session_id}")

    # Risk events ---------------------------------------------------------
    def append_risk_event(self, session_id: str, event: Dict[str, Any]) -> None:
        self._append_list(f"risk:events:{session_id}", event)

    def load_risk_events(self, session_id: str) -> List[Dict[str, Any]]:
        return self._get_list(f"risk:events:{session_id}")

    # OSS placeholders ----------------------------------------------------
    def save_oss_reference(self, session_id: str, reference: Dict[str, Any]) -> None:
        self._append_list(f"oss:{session_id}", reference)

    def load_oss_references(self, session_id: str) -> List[Dict[str, Any]]:
        return self._get_list(f"oss:{session_id}")


repository = ConversationRepository()
