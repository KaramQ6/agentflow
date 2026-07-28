"""Tests for the LLM provider."""

import pytest

from agentflow import LLM, LLMResponse


def test_llm_init_defaults():
    llm = LLM(api_key="test-key")
    assert llm.model == "gpt-4o-mini"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 4096
    assert llm.max_retries == 2


def test_llm_init_custom():
    llm = LLM(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        temperature=0.5,
        max_tokens=2048,
        max_retries=1,
    )
    assert llm.model == "llama-3.3-70b-versatile"
    assert llm.temperature == 0.5
    assert llm.max_tokens == 2048
    assert llm.max_retries == 1


def test_llm_has_client():
    llm = LLM(api_key="test")
    assert llm._client is not None


def test_llmresponse_dict_shim_warns():
    """Dict-style access still works but warns; attribute access is silent."""
    response = LLMResponse(content="hello", tokens=3)

    with pytest.warns(DeprecationWarning, match="attribute access"):
        assert response["content"] == "hello"
    with pytest.warns(DeprecationWarning, match="attribute access"):
        assert response.get("cost") == 0.0
    with pytest.warns(DeprecationWarning), pytest.raises(KeyError):
        response["nonexistent"]

    # Attribute access must not warn.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", DeprecationWarning)
        assert response.content == "hello"
        assert response.tokens == 3


# ── Provider passthrough ──────────────────────────────────────────────────────


class _RecordingCompletions:
    """Captures the kwargs the OpenAI client was called with."""

    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs

        class _Message:
            content = "ok"
            tool_calls = None

        class _Choice:
            message = _Message()
            finish_reason = "stop"

        class _Usage:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        class _Response:
            choices = [_Choice()]
            usage = _Usage()
            model = "gpt-4o-mini"

        return _Response()


def _patch_client(llm):
    completions = _RecordingCompletions()

    class _Chat:
        pass

    chat = _Chat()
    chat.completions = completions
    llm._client.chat = chat
    return completions


@pytest.mark.asyncio
async def test_extra_kwargs_reach_the_provider():
    """The escape hatch for anything agentflow does not model itself."""
    llm = LLM(api_key="test")
    completions = _patch_client(llm)

    await llm.generate(
        [{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
        seed=7,
    )

    assert completions.kwargs["response_format"] == {"type": "json_object"}
    assert completions.kwargs["seed"] == 7


@pytest.mark.asyncio
async def test_generate_without_extras_sends_no_stray_kwargs():
    llm = LLM(api_key="test")
    completions = _patch_client(llm)

    await llm.generate([{"role": "user", "content": "hi"}])

    assert set(completions.kwargs) == {"model", "messages", "temperature", "max_tokens"}


@pytest.mark.asyncio
async def test_extra_kwargs_are_part_of_the_cache_key():
    """A request asking for JSON must not be served the cached prose answer."""
    from agentflow import InMemoryCache

    cache = InMemoryCache()
    llm = LLM(api_key="test", cache=cache)
    completions = _patch_client(llm)
    messages = [{"role": "user", "content": "hi"}]

    await llm.generate(messages)
    first = await llm.generate(messages)
    assert first.cached, "identical request should hit the cache"

    completions.kwargs = None
    second = await llm.generate(messages, response_format={"type": "json_object"})
    assert not second.cached, "a different request must not reuse the cached answer"
    assert completions.kwargs is not None
