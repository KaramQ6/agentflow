"""Tests for LLM response caching."""

import asyncio
import time

import pytest

from agentflow.cache import InMemoryCache, ResponseCache


# ─── InMemoryCache ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_inmemory_cache_set_and_get():
    cache = InMemoryCache()
    await cache.set("key1", {"content": "hello", "tokens": 10})
    result = await cache.get("key1")
    assert result == {"content": "hello", "tokens": 10}


@pytest.mark.asyncio
async def test_inmemory_cache_miss_returns_none():
    cache = InMemoryCache()
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_inmemory_cache_ttl_expiry():
    cache = InMemoryCache(default_ttl=1)
    await cache.set("k", {"v": 1}, ttl=1)
    # Manually expire by manipulating the stored timestamp
    key = list(cache._store.keys())[0]
    value, _ = cache._store[key]
    cache._store[key] = (value, time.monotonic() - 1)  # already expired

    result = await cache.get("k")
    assert result is None
    assert len(cache._store) == 0  # lazy eviction happened


@pytest.mark.asyncio
async def test_inmemory_cache_max_size_eviction():
    cache = InMemoryCache(max_size=3)
    for i in range(4):
        await cache.set(f"key{i}", {"i": i})

    assert len(cache) == 3
    # Oldest key should be evicted
    assert await cache.get("key0") is None


@pytest.mark.asyncio
async def test_inmemory_cache_clear():
    cache = InMemoryCache()
    await cache.set("a", {"x": 1})
    await cache.set("b", {"x": 2})
    cache.clear()
    assert len(cache) == 0


# ─── Cache Key Consistency ──────────────────────────────────────────────────────

def test_cache_key_is_deterministic():
    messages = [{"role": "user", "content": "hello"}]
    key1 = ResponseCache.make_key(messages, "gpt-4o")
    key2 = ResponseCache.make_key(messages, "gpt-4o")
    assert key1 == key2


def test_cache_key_differs_for_different_messages():
    key1 = ResponseCache.make_key([{"role": "user", "content": "hello"}], "gpt-4o")
    key2 = ResponseCache.make_key([{"role": "user", "content": "world"}], "gpt-4o")
    assert key1 != key2


def test_cache_key_differs_for_different_models():
    messages = [{"role": "user", "content": "hello"}]
    key1 = ResponseCache.make_key(messages, "gpt-4o")
    key2 = ResponseCache.make_key(messages, "gpt-4o-mini")
    assert key1 != key2


def test_cache_key_is_order_stable():
    """Message list order must be preserved (not sorted)."""
    msgs_a = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msgs_b = [{"role": "user", "content": "u"}, {"role": "system", "content": "s"}]
    assert ResponseCache.make_key(msgs_a, "m") != ResponseCache.make_key(msgs_b, "m")


# ─── LLM Cache Integration ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_cache_hit_skips_api_call():
    """When cache has a hit, the OpenAI client must NOT be called."""
    from unittest.mock import AsyncMock, MagicMock
    from agentflow.llm import LLM

    cache = InMemoryCache()
    messages = [{"role": "user", "content": "what is 2+2"}]
    key = ResponseCache.make_key(messages, "gpt-4o-mini")
    await cache.set(key, {"content": "4", "tokens": 5, "duration": 0.1, "model": "gpt-4o-mini"})

    llm = LLM(model="gpt-4o-mini", api_key="fake", cache=cache)
    # Replace the underlying client so any call to it raises
    llm._client = MagicMock()
    llm._client.chat.completions.create = AsyncMock(side_effect=AssertionError("API was called!"))

    result = await llm.generate(messages)
    assert result["cached"] is True
    assert result["content"] == "4"


@pytest.mark.asyncio
async def test_llm_cache_miss_stores_result():
    """After a successful API call, the result must be stored in the cache."""
    from unittest.mock import AsyncMock, MagicMock
    from agentflow.llm import LLM

    cache = InMemoryCache()
    messages = [{"role": "user", "content": "hello"}]

    llm = LLM(model="gpt-4o-mini", api_key="fake", cache=cache)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "world"
    mock_response.usage.total_tokens = 20
    mock_response.model = "gpt-4o-mini"
    # Replace the entire client so we control the call
    llm._client = MagicMock()
    llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await llm.generate(messages)
    assert result["cached"] is False
    assert result["content"] == "world"

    # Second call must hit cache
    result2 = await llm.generate(messages)
    assert result2["cached"] is True
    assert result2["content"] == "world"
