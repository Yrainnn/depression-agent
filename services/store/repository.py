import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from packages.common.config import settings

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore


LOGGER = logging.getLogger(__name__)


class ConversationRepository:
    """Persistence layer for conversation state stored in Redis."""

    STATE_TTL_SECONDS = 7 * 24 * 60 * 60
    TRANSCRIPTS_TTL_SECONDS = 30 * 24 * 60 * 60
    SCORE_TTL_SECONDS = 180 * 24 * 60 * 60
    RISK_TTL_SECONDS = 180 * 24 * 60 * 60

    def __init__(self, redis_url: Optional[str] = None) -> None:
        base_url = redis_url or settings.redis_url
        self._redis_url = self._inject_password(base_url)
        self._client = self._create_client()
        self._memory_store: Dict[str, str] = {}

    def _create_client(self) -> Optional["redis.Redis"]:
        if not redis:
            LOGGER.warning("Repo: memory fallback (redis package not available)")
            LOGGER.warning("Repo: memory fallback")
            return None
        try:
            client = redis.from_url(
                self._redis_url,
                decode_responses=True,
            )
            client.ping()
            return client
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Failed to connect to Redis at %s: %s", self._redis_url, exc)
            LOGGER.warning("Repo: memory fallback")
            return None

    def _inject_password(self, redis_url: str) -> str:
        if not settings.redis_password:
            return redis_url

        parsed = urlparse(redis_url)
        if parsed.password:
            return redis_url

        hostinfo = parsed.netloc
        userinfo = f":{settings.redis_password}"
        if "@" in parsed.netloc:
            userinfo_part, hostinfo = parsed.netloc.split("@", 1)
            if ":" in userinfo_part:
                username, _ = userinfo_part.split(":", 1)
                userinfo = f"{username}:{settings.redis_password}"
            elif userinfo_part:
                userinfo = f"{userinfo_part}:{settings.redis_password}"

        new_netloc = f"{userinfo}@{hostinfo}" if hostinfo else userinfo
        return urlunparse(parsed._replace(netloc=new_netloc))

    def _key(self, *parts: str) -> str:
        return ":".join([settings.redis_prefix, *parts])

    # Generic helpers -----------------------------------------------------
    def _raw_get(self, key: str) -> Optional[str]:
        if self._client is not None:
            return self._client.get(key)
        return self._memory_store.get(key)

    def _get(
        self, key: str, *, legacy_keys: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        raw = self._raw_get(key)
        if not raw and legacy_keys:
            for legacy_key in legacy_keys:
                raw = self._raw_get(legacy_key)
                if raw:
                    break
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.error("Malformed JSON for key %s", key)
            return None

    def _set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        serialized = json.dumps(value, ensure_ascii=False)
        if self._client is not None:
            self._client.set(key, serialized)
            if ttl is not None:
                try:
                    self._client.expire(key, ttl)
                except Exception:  # pragma: no cover - runtime guard
                    LOGGER.exception("Failed to set TTL for %s", key)
        else:
            self._memory_store[key] = serialized

    def _append_list(
        self,
        key: str,
        item: Any,
        ttl: Optional[int] = None,
        *,
        legacy_keys: Optional[List[str]] = None,
    ) -> None:
        data = self._get(key, legacy_keys=legacy_keys) or {"items": []}
        items: List[Any] = data.get("items", [])
        items.append(item)
        data["items"] = items
        self._set(key, data, ttl=ttl)

    def _get_list(
        self, key: str, *, legacy_keys: Optional[List[str]] = None
    ) -> List[Any]:
        data = self._get(key, legacy_keys=legacy_keys) or {"items": []}
        return data.get("items", [])

    # Session state -------------------------------------------------------
    def load_session_state(self, session_id: str) -> Dict[str, Any]:
        return self._get(
            self._key("session", session_id, "state"),
            legacy_keys=[f"session:{session_id}:state"],
        ) or {}

    def save_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        self._set(
            self._key("session", session_id, "state"),
            state,
            ttl=self.STATE_TTL_SECONDS,
        )

    # Scores --------------------------------------------------------------
    def save_scores(self, session_id: str, scores: Any) -> None:
        if isinstance(scores, dict):
            payload = dict(scores)
            payload.setdefault("per_item_scores", payload.get("items", []))
        else:
            payload = {"per_item_scores": list(scores or [])}
        payload.setdefault("items", payload.get("per_item_scores", []))
        self._set(
            self._key("score", session_id),
            payload,
            ttl=self.SCORE_TTL_SECONDS,
        )

    def load_scores(self, session_id: str) -> Any:
        data = self._get(
            self._key("score", session_id),
            legacy_keys=[f"score:{session_id}", f"session:{session_id}:score"],
        )
        if not data:
            return []
        if isinstance(data, dict) and (
            "per_item_scores" in data or "total_score" in data or "opinion" in data
        ):
            data.setdefault("per_item_scores", data.get("items", []))
            return data
        return data.get("items", [])

    # Transcripts ---------------------------------------------------------
    def append_transcript(self, session_id: str, segment: Dict[str, Any]) -> None:
        self._append_list(
            self._key("session", session_id, "transcripts"),
            segment,
            ttl=self.TRANSCRIPTS_TTL_SECONDS,
            legacy_keys=[f"session:{session_id}:transcripts"],
        )

    def load_transcripts(self, session_id: str) -> List[Dict[str, Any]]:
        return self._get_list(
            self._key("session", session_id, "transcripts"),
            legacy_keys=[f"session:{session_id}:transcripts"],
        )

    # Alias maintained for orchestrator compatibility -------------------
    def get_transcripts(self, session_id: str) -> List[Dict[str, Any]]:
        return self.load_transcripts(session_id)

    # Risk events ---------------------------------------------------------
    def append_risk_event(self, session_id: str, event: Dict[str, Any]) -> None:
        risk_key = self._key("risk", "events", session_id)
        self._append_list(
            risk_key,
            event,
            ttl=self.RISK_TTL_SECONDS,
            legacy_keys=[f"session:{session_id}:risk:events"],
        )
        self.push_risk_event_stream(session_id, event)

    def load_risk_events(self, session_id: str) -> List[Dict[str, Any]]:
        return self._get_list(
            self._key("risk", "events", session_id),
            legacy_keys=[f"session:{session_id}:risk:events"],
        )

    def push_risk_event_stream(self, session_id: str, payload: Dict[str, Any]) -> str:
        """Append a risk event to the Redis stream for the session."""

        if self._client is None:
            return ""

        stream_key = f"{self._key('risk', 'events', session_id)}:stream"
        try:
            entry_id = self._client.xadd(
                stream_key,
                {"event": json.dumps(payload, ensure_ascii=False)},
                maxlen=1000,
                approximate=True,
            )
            try:
                self._client.expire(stream_key, self.RISK_TTL_SECONDS)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to set TTL for %s", stream_key)
            return entry_id
        except Exception:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to append to risk stream %s", stream_key)
            return ""

    def get_risk_recent(self, session_id: str, count: int = 20) -> List[Dict[str, Any]]:
        """Return recent risk events, preferring Redis Streams when available."""

        risk_key = self._key("risk", "events", session_id)
        stream_key = f"{risk_key}:stream"
        if self._client is not None and count > 0:
            try:
                entries = self._client.xrange(stream_key, "-", "+")
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to read risk stream %s", stream_key)
            else:
                if entries:
                    sliced = entries[-count:]
                    results: List[Dict[str, Any]] = []
                    for entry_id, payload in sliced:
                        raw_event = payload.get("event") if isinstance(payload, dict) else None
                        if not raw_event:
                            continue
                        try:
                            event = json.loads(raw_event)
                        except json.JSONDecodeError:
                            LOGGER.error(
                                "Malformed risk event in stream %s: %s", stream_key, raw_event
                            )
                            continue
                        if isinstance(event, dict):
                            results.append(
                                {
                                    "id": entry_id,
                                    "ts": event.get("ts"),
                                    "reason": event.get("reason"),
                                    "match_text": event.get("match_text"),
                                    "raw": event,
                                }
                            )
                    if results:
                        return results

        legacy_events = self.load_risk_events(session_id)
        if count > 0:
            legacy_events = legacy_events[-count:]
        parsed: List[Dict[str, Any]] = []
        for event in legacy_events:
            if isinstance(event, dict):
                parsed.append(
                    {
                        "id": event.get("id"),
                        "ts": event.get("ts"),
                        "reason": event.get("reason"),
                        "match_text": event.get("match_text"),
                        "raw": event,
                    }
                )
        return parsed

    # OSS placeholders ----------------------------------------------------
    def save_oss_reference(self, session_id: str, reference: Dict[str, Any]) -> None:
        self._append_list(
            self._key("oss", session_id),
            reference,
            legacy_keys=[f"session:{session_id}:oss"],
        )

    def load_oss_references(self, session_id: str) -> List[Dict[str, Any]]:
        return self._get_list(
            self._key("oss", session_id),
            legacy_keys=[f"session:{session_id}:oss"],
        )

    def ping(self) -> bool:
        """Return True when the Redis client responds to PING."""

        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            LOGGER.exception("Redis ping failed")
            return False


repository = ConversationRepository()
