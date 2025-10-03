#!/usr/bin/env python3
"""Cleanup helper for Redis conversation artifacts."""

import argparse
import sys
from typing import List
from urllib.parse import urlparse, urlunparse

from packages.common.config import settings

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore


KEY_TEMPLATES = [
    "{prefix}:session:{sid}:state",
    "{prefix}:session:{sid}:transcripts",
    "{prefix}:score:{sid}",
    "{prefix}:risk:events:{sid}",
    "{prefix}:risk:events:{sid}:stream",
    "{prefix}:oss:{sid}",
]


def _inject_password(redis_url: str) -> str:
    password = getattr(settings, "redis_password", None)
    if not password:
        return redis_url

    parsed = urlparse(redis_url)
    if parsed.password:
        return redis_url

    netloc = parsed.netloc or ""
    if "@" in netloc:
        userinfo, hostinfo = netloc.split("@", 1)
        if ":" in userinfo:
            username, _ = userinfo.split(":", 1)
            userinfo = f"{username}:{password}"
        elif userinfo:
            userinfo = f"{userinfo}:{password}"
        else:
            userinfo = f":{password}"
        netloc = f"{userinfo}@{hostinfo}"
    else:
        if netloc:
            netloc = f":{password}@{netloc}"
        else:
            netloc = f":{password}@"

    return urlunparse(parsed._replace(netloc=netloc))


def _get_client() -> "redis.Redis":
    if redis is None:
        print("redis package not installed", file=sys.stderr)
        sys.exit(1)

    redis_url = _inject_password(settings.redis_url)
    try:
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"failed to connect to redis at {redis_url}: {exc}", file=sys.stderr)
        sys.exit(1)


def _prefixed_keys(prefix: str, sid: str) -> List[str]:
    return [template.format(prefix=prefix, sid=sid) for template in KEY_TEMPLATES]


def cleanup_session(sid: str) -> None:
    client = _get_client()
    keys = _prefixed_keys(settings.redis_prefix, sid)
    deleted = client.delete(*keys)
    print(f"deleted: {deleted} keys")


def flush_database() -> None:
    client = _get_client()
    confirmation = input("WARNING: This will FLUSHDB. Type 'FLUSH' to continue: ")
    if confirmation.strip().upper() != "FLUSH":
        print("aborted.")
        return
    client.flushdb()
    print("Redis database flushed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup Redis session data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sid", help="Session identifier to delete")
    group.add_argument(
        "--all",
        action="store_true",
        help="Flush the entire Redis database (development use only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        flush_database()
    else:
        cleanup_session(args.sid)


if __name__ == "__main__":
    main()
