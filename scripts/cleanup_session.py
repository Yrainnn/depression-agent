#!/usr/bin/env python3
"""Utility for deleting conversation artifacts from Redis."""

import argparse
import logging
import sys
from typing import Iterable, List

from packages.common.config import settings
from services.store.repository import ConversationRepository

LOGGER = logging.getLogger(__name__)


def _ensure_client(repo: ConversationRepository):
    """Return the Redis client or exit when unavailable."""

    client = getattr(repo, "_client", None)
    if client is None:
        LOGGER.error("Redis client unavailable. Ensure Redis is running and REDIS_URL is configured.")
        sys.exit(1)
    return client


def _prefixed_key(*parts: str) -> str:
    return ":".join([settings.redis_prefix, *parts])


def _legacy_keys(session_id: str) -> List[str]:
    return [
        f"session:{session_id}:state",
        f"session:{session_id}:transcripts",
        f"session:{session_id}:score",
        f"session:{session_id}:risk:events",
        f"session:{session_id}:risk:events:stream",
        f"session:{session_id}:oss",
    ]


def _collect_keys(session_id: str) -> List[str]:
    prefixed = [
        _prefixed_key("session", session_id, "state"),
        _prefixed_key("session", session_id, "transcripts"),
        _prefixed_key("score", session_id),
        _prefixed_key("risk", "events", session_id),
        f"{_prefixed_key('risk', 'events', session_id)}:stream",
        _prefixed_key("oss", session_id),
    ]
    return prefixed + _legacy_keys(session_id)


def _delete_keys(client, keys: Iterable[str]) -> int:
    unique_keys = list(dict.fromkeys(keys))
    if not unique_keys:
        return 0
    return int(client.delete(*unique_keys))


def cleanup_session(session_id: str) -> None:
    repo = ConversationRepository()
    client = _ensure_client(repo)
    deleted = _delete_keys(client, _collect_keys(session_id))
    print(f"Deleted {deleted} keys for session {session_id}")


def flush_database() -> None:
    repo = ConversationRepository()
    client = _ensure_client(repo)
    warning = (
        "WARNING: This will FLUSHDB on the configured Redis instance. "
        "Type 'FLUSH' to continue: "
    )
    confirmation = input(warning)
    if confirmation.strip().upper() != "FLUSH":
        print("Aborted.")
        return
    client.flushdb()
    print("Redis database flushed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup session artifacts in Redis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sid", help="Session ID to remove from Redis")
    group.add_argument(
        "--all",
        action="store_true",
        help="Flush the entire Redis database (development use only)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.all:
        flush_database()
    else:
        assert args.sid  # for type checkers
        cleanup_session(args.sid)


if __name__ == "__main__":
    main()
