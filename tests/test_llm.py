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
