"""Cross-session persistent memory for agentflow agents.

Agents use memory to retain and share context across multiple pipeline
executions. The abstract ``BaseMemory`` defines the interface;
``InMemoryContext`` is a lightweight dict-based implementation with
TTL expiration and LRU eviction. ``RedisContext`` provides Redis-backed
persistent memory, and ``VectorContext`` enables semantic search over
stored context using a vector database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

_log = logging.getLogger("agentflow.memory")

DEFAULT_TTL = 3600
DEFAULT_MAX_ENTRIES = 1000


class BaseMemory(ABC):
    """Abstract interface for agent memory backends."""

    @abstractmethod
    async def save_context(self, session_id: str, key: str, value: Any) -> None:
        """Persist a key-value pair under *session_id*."""

    @abstractmethod
    async def load_context(self, session_id: str) -> dict[str, Any]:
        """Return all stored key-value pairs for *session_id* as a dict."""

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Remove all entries for *session_id*."""

    @abstractmethod
    async def delete_key(self, session_id: str, key: str) -> None:
        """Remove a single key from *session_id*."""


class InMemoryContext(BaseMemory):
    """Thread-safe in-process memory store with per-entry TTL and LRU eviction.

    Args:
        default_ttl: Seconds before a stored entry expires (default 3600).
        max_entries: Maximum entries per session before LRU eviction kicks in
                     (default 1000). Set to 0 for unlimited.
    """

    def __init__(self, default_ttl: float = DEFAULT_TTL, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._store: dict[str, OrderedDict[str, tuple[Any, float]]] = {}
        self._lock = asyncio.Lock()

    async def save_context(self, session_id: str, key: str, value: Any) -> None:
        expiry = time.monotonic() + self._default_ttl
        async with self._lock:
            session = self._store.setdefault(session_id, OrderedDict())
            # Move to end (most-recently-used) or insert
            session[key] = (value, expiry)
            session.move_to_end(key)
            # LRU eviction: remove oldest entries when over limit
            if self._max_entries > 0 and len(session) > self._max_entries:
                excess = len(session) - self._max_entries
                for _ in range(excess):
                    oldest = next(iter(session))
                    del session[oldest]
                    _log.debug("LRU evicted %s:%s", session_id, oldest)

    async def load_context(self, session_id: str) -> dict[str, Any]:
        now = time.monotonic()
        async with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return {}
            # Expire stale entries in-place
            expired = [k for k, (_, exp) in session.items() if now >= exp]
            for k in expired:
                del session[k]
            if not session:
                del self._store[session_id]
                return {}
            return {k: v for k, (v, _) in session.items()}

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._store.pop(session_id, None)

    async def delete_key(self, session_id: str, key: str) -> None:
        async with self._lock:
            session = self._store.get(session_id)
            if session is not None:
                session.pop(key, None)
                if not session:
                    del self._store[session_id]


# ── Optional dependency guards ────────────────────────────────────────────────

try:
    import redis.asyncio as aioredis  # noqa: I001, F811

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    aioredis = None

try:
    import chromadb  # noqa: I001

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    chromadb = None


def _require_redis() -> None:
    """Re-import redis lazily so tests can inject mock modules at runtime."""
    global aioredis
    try:
        import redis.asyncio as _aioredis
    except ImportError as exc:
        raise ImportError(
            "RedisContext requires the redis package. "
            "Install it with: pip install agentflowkit[redis]"
        ) from exc
    aioredis = _aioredis


def _require_chromadb() -> None:
    """Re-import chromadb lazily so tests can inject mock modules at runtime."""
    global chromadb
    try:
        import chromadb as _chromadb
    except ImportError as exc:
        raise ImportError(
            "VectorContext requires the chromadb package. "
            "Install it with: pip install agentflowkit[redis]"
        ) from exc
    chromadb = _chromadb


# ── Redis-backed persistent memory ────────────────────────────────────────────


class RedisContext(BaseMemory):
    """Redis-backed persistent context memory.

    Stores session data as a Redis hash with optional global TTL.
    All values are JSON-serialised.

    Requires the ``redis`` extra: ``pip install agentflowkit[redis]``

    Args:
        url: Redis connection URL (default ``redis://localhost:6379/0``).
        prefix: Key prefix namespace (default ``"agentflow:mem:"``).
        ttl: Optional TTL in seconds applied to the entire session hash.
            When set, the hash key is expired *ttl* seconds after every write.

    Raises:
        ImportError: If the ``redis`` package is not installed.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "agentflow:mem:",
        ttl: int | None = None,
    ):
        _require_redis()
        self._client = aioredis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._ttl = ttl

    def _session_key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def save_context(self, session_id: str, key: str, value: Any) -> None:
        sk = self._session_key(session_id)
        await self._client.hset(sk, key, json.dumps(value))
        if self._ttl is not None:
            await self._client.expire(sk, self._ttl)

    async def load_context(self, session_id: str) -> dict[str, Any]:
        sk = self._session_key(session_id)
        raw: dict[str, str] = await self._client.hgetall(sk)
        return {k: json.loads(v) for k, v in raw.items()}

    async def clear(self, session_id: str) -> None:
        await self._client.delete(self._session_key(session_id))

    async def delete_key(self, session_id: str, key: str) -> None:
        await self._client.hdel(self._session_key(session_id), key)


# ── Vector DB-backed semantic memory ──────────────────────────────────────────


_EmbeddingFn = Callable[[list[str]], list[list[float]]]


class VectorContext(BaseMemory):
    """Vector database-backed semantic context memory.

    Stores agent context as documents in a vector database, enabling
    semantic search over historical context. Supports any ChromaDB-compatible
    collection with a user-provided embedding function.

    Requires the ``redis`` extra: ``pip install agentflowkit[redis]``

    Args:
        collection_name: ChromaDB collection name (default ``"agentflow_memory"``).
        embedding_fn: A callable that takes a list of strings and returns a list
            of embedding vectors. Required for ``save_context`` and ``search_context``.
        persist_dir: Directory for persistent storage. If provided, uses
            ``chromadb.PersistentClient``; otherwise uses an in-memory
            ``EphemeralClient``.
        client: An existing ``chromadb.ClientAPI`` instance. Takes precedence
            over ``persist_dir``.

    Raises:
        ImportError: If the ``chromadb`` package is not installed.
    """

    def __init__(
        self,
        collection_name: str = "agentflow_memory",
        embedding_fn: _EmbeddingFn | None = None,
        persist_dir: str | None = None,
        client: Any = None,
    ):
        if client is not None:
            self._client: Any = client
        else:
            _require_chromadb()
            if persist_dir is not None:
                self._client = chromadb.PersistentClient(path=persist_dir)
            else:
                self._client = chromadb.EphemeralClient()
        self._embedding_fn = embedding_fn
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

    @staticmethod
    def _doc_id(session_id: str, key: str) -> str:
        return f"{session_id}::{key}"

    async def save_context(self, session_id: str, key: str, value: Any) -> None:
        doc = value if isinstance(value, str) else json.dumps(value)
        self._collection.upsert(
            ids=[self._doc_id(session_id, key)],
            documents=[doc],
            metadatas=[{"session_id": session_id, "key": key}],
        )

    async def load_context(self, session_id: str) -> dict[str, Any]:
        try:
            result = self._collection.get(where={"session_id": session_id})
        except Exception:
            _log.debug("load_context: no matching documents for session %s", session_id)
            return {}
        ctx: dict[str, Any] = {}
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for doc_id, doc, meta in zip(ids, documents, metadatas, strict=False):
            key = meta.get("key") if meta else doc_id
            ctx[key] = doc
        return ctx

    async def search_context(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Semantic search over stored context documents.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return (default 5).

        Returns:
            List of result dicts with keys: ``id``, ``document``, ``metadata``,
            ``distance``.
        """
        result = self._collection.query(query_texts=[query], n_results=top_k)
        items: list[dict[str, Any]] = []
        ids_batch = result.get("ids")
        docs_batch = result.get("documents")
        metas_batch = result.get("metadatas")
        dists_batch = result.get("distances")
        if ids_batch and ids_batch[0]:
            for i, doc_id in enumerate(ids_batch[0]):
                items.append(
                    {
                        "id": doc_id,
                        "document": docs_batch[0][i] if docs_batch else None,
                        "metadata": metas_batch[0][i] if metas_batch else None,
                        "distance": dists_batch[0][i] if dists_batch else None,
                    }
                )
        return items

    async def clear(self, session_id: str) -> None:
        try:
            result = self._collection.get(where={"session_id": session_id})
            ids = result.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
        except Exception:
            _log.debug("clear: no documents to remove for session %s", session_id)

    async def delete_key(self, session_id: str, key: str) -> None:
        self._collection.delete(ids=[self._doc_id(session_id, key)])
