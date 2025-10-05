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

    def get_progress(self, session_id: str) -> Optional[Dict[str, int]]:
        state = self.load_session_state(session_id)
        if not state:
            return None

        index = state.get("index")
        total = state.get("total")
        try:
            index_val = int(index)
        except (TypeError, ValueError):
            index_val = 1
        try:
            total_val = int(total)
        except (TypeError, ValueError):
            total_val = 17

        index_val = max(1, index_val)
        total_val = max(index_val, total_val)
        return {"index": index_val, "total": total_val}

    def set_progress(self, session_id: str, progress: Dict[str, Any]) -> None:
        state = self.load_session_state(session_id) or {}
        if isinstance(progress, dict):
            index = progress.get("index")
            total = progress.get("total")
            if index is not None:
                try:
                    state["index"] = max(1, int(index))
                except (TypeError, ValueError):
                    state.setdefault("index", 1)
            if total is not None:
                try:
                    state["total"] = max(state.get("index", 1), int(total))
                except (TypeError, ValueError):
                    state.setdefault("total", state.get("index", 1))
        self.save_session_state(session_id, state)

    def set_last_clarify_need(
        self, session_id: str, item_id: int, clarify_need: str
    ) -> None:
        state = self.load_session_state(session_id) or {}
        targets = state.get("clarify_targets")
        if not isinstance(targets, dict):
            targets = {}
        targets[str(item_id)] = clarify_need
        state["clarify_targets"] = targets
        state["last_clarify"] = {"item_id": item_id, "need": clarify_need}
        self.save_session_state(session_id, state)

    def get_last_clarify_need(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self.load_session_state(session_id)
        if not state:
            return None
        last = state.get("last_clarify")
        if isinstance(last, dict) and "item_id" in last:
            return last
        return None

    def clear_last_clarify_need(self, session_id: str) -> None:
        state = self.load_session_state(session_id) or {}
        if "last_clarify" in state:
            state.pop("last_clarify", None)
            self.save_session_state(session_id, state)

    def mark_finished(self, session_id: str) -> None:
        state = self.load_session_state(session_id) or {}
        state["completed"] = True
        total = state.get("total")
        try:
            index_value = int(total) if total is not None else 17
        except (TypeError, ValueError):
            index_value = 17
        state["index"] = max(index_value, state.get("index", index_value))
        state.setdefault("total", index_value)
        self.save_session_state(session_id, state)

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

    def get_score(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = self._get(
            self._key("score", session_id),
            legacy_keys=[f"score:{session_id}", f"session:{session_id}:score"],
        )
        if data is None:
            return None
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"items": data}
        return None

    def set_score(self, session_id: str, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self._set(
            self._key("score", session_id),
            payload,
            ttl=self.SCORE_TTL_SECONDS,
        )

    def merge_scores(self, session_id: str, partial: Dict[str, Any]) -> None:
        if not partial:
            return

        current = self.get_score(session_id) or {"items": [], "total_score": {}}
        index: Dict[int, Dict[str, Any]] = {}
        for item in current.get("items", []):
            if isinstance(item, dict) and "item_id" in item:
                try:
                    key = int(item["item_id"])
                except (TypeError, ValueError):
                    continue
                index[key] = item

        for item in partial.get("items", []):
            if isinstance(item, dict) and "item_id" in item:
                try:
                    key = int(item["item_id"])
                except (TypeError, ValueError):
                    continue
                index[key] = item

        current["items"] = [index[k] for k in sorted(index)]
        current["per_item_scores"] = current["items"]
        if "total_score" in partial:
            current["total_score"] = partial["total_score"]

        self.set_score(session_id, current)

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
            legacy_keys=[
                f"session:{session_id}:risk:events",
                f"risk:events:{session_id}",
            ],
        )
        self.push_risk_event_stream(session_id, event)

    def push_risk_event(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Alias for backward compatibility."""

        self.append_risk_event(session_id, payload)

    def load_risk_events(self, session_id: str) -> List[Dict[str, Any]]:
        return self._get_list(
            self._key("risk", "events", session_id),
            legacy_keys=[
                f"session:{session_id}:risk:events",
                f"risk:events:{session_id}",
            ],
        )

    def push_risk_event_stream(
        self,
        session_id: str,
        payload: Dict[str, Any],
        maxlen: int = 1000,
    ) -> str:
        """Append a risk event to the Redis stream for the session."""

        if self._client is None:
            return ""

        stream_key = f"{self._key('risk', 'events', session_id)}:stream"
        flat_payload: Dict[str, str] = {}
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                flat_payload[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat_payload[key] = str(value)
        if not flat_payload:
            flat_payload["raw"] = json.dumps(payload, ensure_ascii=False)

        try:
            entry_id = self._client.xadd(
                stream_key,
                flat_payload,
                maxlen=maxlen,
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
                entries = self._client.xrevrange(stream_key, "+", "-", count=count)
            except Exception:  # pragma: no cover - runtime guard
                LOGGER.exception("Failed to read risk stream %s", stream_key)
            else:
                if entries:
                    results: List[Dict[str, Any]] = []
                    for entry_id, payload in entries:
                        event: Dict[str, Any] = {}
                        raw_blob: Optional[str] = None
                        if isinstance(payload, dict):
                            for key, value in payload.items():
                                event[key] = value
                                if key == "raw":
                                    raw_blob = value
                        if raw_blob:
                            try:
                                decoded = json.loads(raw_blob)
                                if isinstance(decoded, dict):
                                    event = {**decoded, **event}
                            except json.JSONDecodeError:
                                LOGGER.error(
                                    "Malformed raw risk event in stream %s: %s",
                                    stream_key,
                                    raw_blob,
                                )
                        results.append(
                            {
                                "id": entry_id,
                                "ts": event.get("ts"),
                                "reason": event.get("reason"),
                                "match_text": event.get("match_text"),
                                "raw": event or payload,
                            }
                        )
                    if results:
                        return list(reversed(results))

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
