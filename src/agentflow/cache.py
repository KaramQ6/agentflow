"""LLM response caching for agentflow."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from .exceptions import AgentFlowError


class ResponseCache(ABC):
    """Abstract base for LLM response caches.

    Implement this interface to plug in any cache backend
    (in-memory, Redis, DynamoDB, etc.).
    """

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Return cached response or None if missing/expired."""
        ...

    @abstractmethod
    async def set(self, key: str, value: dict[str, Any], ttl: int = 3600) -> None:
        """Store a response with a time-to-live in seconds."""
        ...

    @staticmethod
    def make_key(messages: list[dict[str, Any]], model: str) -> str:
        """Deterministic SHA-256 cache key from messages + model."""
        payload = json.dumps({"messages": messages, "model": model}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


class InMemoryCache(ResponseCache):
    """In-process async-safe FIFO cache backed by a plain dict.

    Suitable for single-threaded async applications and testing.
    TTL is enforced lazily on ``get()`` — expired entries are evicted
    when first accessed rather than on a background timer.

    Args:
        default_ttl: Default time-to-live in seconds (default 3600 = 1 hour).
        max_size: Maximum number of entries before oldest are evicted (default 1024).
    """

    def __init__(self, default_ttl: int = 3600, max_size: int = 1024):
        self._default_ttl = default_ttl
        self._max_size = max_size
        # key -> (value, expiry_timestamp)
        self._store: dict[str, tuple[dict[str, Any], float]] = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.monotonic() > expiry:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        if len(self._store) >= self._max_size:
            # Evict the oldest inserted entry
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        expiry = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        self._store[key] = (value, expiry)

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()


try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RedisCache(ResponseCache):
    """Redis-backed LLM response cache.

    Requires the ``redis`` extra: ``pip install agentflowkit[redis]``

    Args:
        url: Redis connection URL (default ``redis://localhost:6379/0``).
        prefix: Key prefix to namespace cache entries (default ``"agentflow:"``)
        default_ttl: Default TTL in seconds (default 3600).

    Raises:
        ImportError: If the ``redis`` package is not installed.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "agentflow:",
        default_ttl: int = 3600,
    ):
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "Redis cache requires the redis package. "
                "Install it with: pip install agentflowkit[redis]"
            )
        self._client = aioredis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._default_ttl = default_ttl

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            raw = await self._client.get(self._full_key(key))
        except Exception as e:
            raise AgentFlowError(
                f"Redis cache get failed for key {key}: {e}"
            ) from e
        if raw is None:
            return None
        data: dict[str, Any] = json.loads(raw)
        return data

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        try:
            await self._client.setex(
                self._full_key(key),
                ttl if ttl is not None else self._default_ttl,
                json.dumps(value),
            )
        except Exception as e:
            raise AgentFlowError(
                f"Redis cache set failed for key {key}: {e}"
            ) from e
