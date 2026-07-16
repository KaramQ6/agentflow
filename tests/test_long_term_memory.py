"""Tests for RedisContext and VectorContext with mocked external clients."""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentflow.memory import BaseMemory, InMemoryContext, RedisContext, VectorContext

# ── Mock module helpers ──────────────────────────────────────────────────────


def _inject_redis_mock() -> MagicMock:
    mock_aioredis = MagicMock()
    mock_redis = MagicMock()
    mock_redis.asyncio = mock_aioredis
    sys.modules["redis"] = mock_redis
    sys.modules["redis.asyncio"] = mock_aioredis
    return mock_aioredis


def _remove_redis_mock() -> None:
    sys.modules.pop("redis", None)
    sys.modules.pop("redis.asyncio", None)


def _inject_chroma_mock() -> MagicMock:
    mock_chromadb = MagicMock()
    sys.modules["chromadb"] = mock_chromadb
    return mock_chromadb


def _remove_chroma_mock() -> None:
    sys.modules.pop("chromadb", None)


def _make_redis_client() -> MagicMock:
    return MagicMock(
        hset=AsyncMock(),
        hgetall=AsyncMock(),
        expire=AsyncMock(),
        delete=AsyncMock(),
        hdel=AsyncMock(),
    )


def _make_chroma_collection(
    ids: list[str] | None = None,
    documents: list[str] | None = None,
    metadatas: list[dict[str, str]] | None = None,
) -> MagicMock:
    return MagicMock(
        upsert=MagicMock(),
        get=MagicMock(
            return_value={
                "ids": ids or [],
                "documents": documents or [],
                "metadatas": metadatas or [],
            }
        ),
        query=MagicMock(
            return_value={
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
        ),
        delete=MagicMock(),
    )


def _make_chroma_client(collection: MagicMock) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


# ── RedisContext tests ──────────────────────────────────────────────────────


class TestRedisContext:

    @pytest.fixture(autouse=True)
    def _inject(self) -> Any:
        self._mock_aioredis = _inject_redis_mock()
        yield
        _remove_redis_mock()

    def test_import_error_when_redis_not_installed(self) -> None:
        _remove_redis_mock()
        try:
            with pytest.raises(ImportError, match="agentflowkit\\[redis\\]"):
                RedisContext()
        finally:
            _inject_redis_mock()

    def test_save_context_hset_and_expire_no_ttl(self) -> None:
        client = _make_redis_client()
        self._mock_aioredis.from_url.return_value = client
        ctx = RedisContext()

        async def _run() -> None:
            await ctx.save_context("s1", "agent", "output")
            client.hset.assert_awaited_once_with(
                "agentflow:mem:s1", "agent", '"output"'
            )
            client.expire.assert_not_awaited()

        asyncio.run(_run())

    def test_save_context_with_ttl(self) -> None:
        client = _make_redis_client()
        self._mock_aioredis.from_url.return_value = client
        ctx = RedisContext(ttl=7200)

        async def _run() -> None:
            await ctx.save_context("s1", "k", "v")
            client.expire.assert_awaited_once_with("agentflow:mem:s1", 7200)

        asyncio.run(_run())

    def test_load_context_returns_parsed_dict(self) -> None:
        client = _make_redis_client()
        client.hgetall = AsyncMock(
            return_value={"agent_a": '"result A"', "agent_b": '"result B"'}
        )
        self._mock_aioredis.from_url.return_value = client
        ctx = RedisContext()

        async def _run() -> None:
            result = await ctx.load_context("s1")
            assert result == {"agent_a": "result A", "agent_b": "result B"}

        asyncio.run(_run())

    def test_load_context_empty_session(self) -> None:
        client = _make_redis_client()
        client.hgetall = AsyncMock(return_value={})
        self._mock_aioredis.from_url.return_value = client
        ctx = RedisContext()

        async def _run() -> None:
            result = await ctx.load_context("s1")
            assert result == {}

        asyncio.run(_run())

    def test_clear_deletes_session_key(self) -> None:
        client = _make_redis_client()
        self._mock_aioredis.from_url.return_value = client
        ctx = RedisContext()

        async def _run() -> None:
            await ctx.clear("s1")
            client.delete.assert_awaited_once_with("agentflow:mem:s1")

        asyncio.run(_run())

    def test_delete_key(self) -> None:
        client = _make_redis_client()
        self._mock_aioredis.from_url.return_value = client
        ctx = RedisContext()

        async def _run() -> None:
            await ctx.delete_key("s1", "agent_a")
            client.hdel.assert_awaited_once_with("agentflow:mem:s1", "agent_a")

        asyncio.run(_run())

    def test_session_isolation_via_prefix(self) -> None:
        client = _make_redis_client()
        client.hgetall = AsyncMock(
            side_effect=[
                {"key": '"v1"'},
                {"key": '"v2"'},
            ]
        )
        self._mock_aioredis.from_url.return_value = client
        ctx = RedisContext()

        async def _run() -> None:
            r1 = await ctx.load_context("alpha")
            r2 = await ctx.load_context("beta")
            assert r1 == {"key": "v1"}
            assert r2 == {"key": "v2"}

        asyncio.run(_run())

    def test_custom_prefix(self) -> None:
        client = _make_redis_client()
        self._mock_aioredis.from_url.return_value = client
        ctx = RedisContext(prefix="custom:")

        async def _run() -> None:
            await ctx.save_context("s99", "k", "v")
            client.hset.assert_awaited_once_with("custom:s99", "k", '"v"')

        asyncio.run(_run())

    def test_constructor_calls_from_url(self) -> None:
        self._mock_aioredis.from_url.return_value = _make_redis_client()
        RedisContext(url="redis://other:6380")
        self._mock_aioredis.from_url.assert_called_with(
            "redis://other:6380", decode_responses=True
        )


# ── VectorContext tests ──────────────────────────────────────────────────────


class TestVectorContext:

    @pytest.fixture(autouse=True)
    def _inject(self) -> Any:
        self._mock_chromadb = _inject_chroma_mock()
        yield
        _remove_chroma_mock()

    def test_import_error_when_chromadb_not_installed(self) -> None:
        _remove_chroma_mock()
        try:
            with pytest.raises(ImportError, match="agentflowkit\\[redis\\]"):
                VectorContext()
        finally:
            _inject_chroma_mock()

    def test_save_context_upserts_document(self) -> None:
        coll = _make_chroma_collection()
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            await ctx.save_context("s1", "agent_a", "long output text")
            coll.upsert.assert_called_once_with(
                ids=["s1::agent_a"],
                documents=["long output text"],
                metadatas=[{"session_id": "s1", "key": "agent_a"}],
            )

        asyncio.run(_run())

    def test_save_context_serializes_non_strings(self) -> None:
        coll = _make_chroma_collection()
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            await ctx.save_context("s1", "k", {"nested": 42, "list": [1, 2]})
            call_args = coll.upsert.call_args
            assert call_args[1]["documents"][0] == '{"nested": 42, "list": [1, 2]}'

        asyncio.run(_run())

    def test_load_context_returns_db_documents(self) -> None:
        coll = _make_chroma_collection(
            ids=["s1::agent_a", "s1::agent_b"],
            documents=["result A", "result B"],
            metadatas=[
                {"session_id": "s1", "key": "agent_a"},
                {"session_id": "s1", "key": "agent_b"},
            ],
        )
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            result = await ctx.load_context("s1")
            assert result == {"agent_a": "result A", "agent_b": "result B"}

        asyncio.run(_run())

    def test_load_context_empty_collection(self) -> None:
        coll = _make_chroma_collection()
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            result = await ctx.load_context("nonexistent")
            assert result == {}

        asyncio.run(_run())

    def test_load_context_graceful_on_error(self) -> None:
        coll = _make_chroma_collection()
        coll.get.side_effect = RuntimeError("db down")
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            result = await ctx.load_context("s1")
            assert result == {}

        asyncio.run(_run())

    def test_search_context_returns_scored_results(self) -> None:
        coll = _make_chroma_collection()
        coll.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc A", "doc B"]],
            "metadatas": [[{"key": "a"}, {"key": "b"}]],
            "distances": [[0.12, 0.34]],
        }
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            results = await ctx.search_context("memory query", top_k=3)
            assert len(results) == 2
            assert results[0]["id"] == "id1"
            assert results[0]["document"] == "doc A"
            assert results[0]["distance"] == 0.12
            assert results[1]["metadata"]["key"] == "b"

        asyncio.run(_run())

    def test_search_context_empty_results(self) -> None:
        coll = _make_chroma_collection()
        coll.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            results = await ctx.search_context("nothing matches", top_k=5)
            assert results == []

        asyncio.run(_run())

    def test_search_context_respects_top_k(self) -> None:
        coll = _make_chroma_collection()
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            await ctx.search_context("q", top_k=10)
            coll.query.assert_called_once_with(query_texts=["q"], n_results=10)

        asyncio.run(_run())

    def test_clear_deletes_session_documents(self) -> None:
        coll = _make_chroma_collection(ids=["s1::a", "s1::b"])
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            await ctx.clear("s1")
            coll.delete.assert_called_once_with(ids=["s1::a", "s1::b"])

        asyncio.run(_run())

    def test_clear_no_documents(self) -> None:
        coll = _make_chroma_collection()
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            await ctx.clear("empty_session")
            coll.delete.assert_not_called()

        asyncio.run(_run())

    def test_clear_graceful_on_error(self) -> None:
        coll = _make_chroma_collection()
        coll.get.side_effect = RuntimeError("gone")
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            await ctx.clear("s1")  # should not raise

        asyncio.run(_run())

    def test_delete_key(self) -> None:
        coll = _make_chroma_collection()
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        ctx = VectorContext()

        async def _run() -> None:
            await ctx.delete_key("s1", "agent_x")
            coll.delete.assert_called_once_with(ids=["s1::agent_x"])

        asyncio.run(_run())

    def test_persistent_client_used_when_dir_provided(self) -> None:
        coll = _make_chroma_collection()
        client = _make_chroma_client(coll)
        self._mock_chromadb.PersistentClient.return_value = client
        VectorContext(persist_dir="/tmp/test_db")
        self._mock_chromadb.PersistentClient.assert_called_once_with(path="/tmp/test_db")

    def test_custom_client_passed_through(self) -> None:
        coll = _make_chroma_collection()
        custom = MagicMock()
        custom.get_or_create_collection.return_value = coll
        ctx = VectorContext(client=custom)
        assert ctx._client is custom

    def test_custom_embedding_fn_passed_to_collection(self) -> None:
        def dummy_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 10 for _ in texts]

        coll = _make_chroma_collection()
        client = _make_chroma_client(coll)
        self._mock_chromadb.EphemeralClient.return_value = client
        VectorContext(embedding_fn=dummy_embed)
        kwargs = client.get_or_create_collection.call_args[1]
        assert kwargs["embedding_function"] is dummy_embed


# ── Interface compliance ────────────────────────────────────────────────────


class TestInterfaceCompliance:
    def test_redis_context_is_base_memory(self) -> None:
        assert issubclass(RedisContext, BaseMemory)

    def test_vector_context_is_base_memory(self) -> None:
        assert issubclass(VectorContext, BaseMemory)

    def test_search_context_only_on_vector(self) -> None:
        assert hasattr(VectorContext, "search_context")
        assert not hasattr(RedisContext, "search_context")
        assert not hasattr(InMemoryContext, "search_context")
