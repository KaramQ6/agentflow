"""Tests for LLM.astream token streaming."""

from types import SimpleNamespace

import pytest
from agentflow import LLM
from agentflow.exceptions import LLMError
from openai import APIError


def _chunk(content: str | None):
    """Build a fake streaming chunk shaped like an OpenAI delta chunk."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


@pytest.mark.asyncio
async def test_astream_yields_deltas(monkeypatch):
    llm = LLM(api_key="test")

    async def fake_create(**kwargs):
        assert kwargs["stream"] is True
        return _FakeStream([_chunk("Hel"), _chunk("lo"), _chunk(None), _chunk("!")])

    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)

    out = [tok async for tok in llm.astream([{"role": "user", "content": "hi"}])]
    assert out == ["Hel", "lo", "!"]  # None delta skipped
    assert "".join(out) == "Hello!"


@pytest.mark.asyncio
async def test_astream_wraps_api_error(monkeypatch):
    llm = LLM(api_key="test")

    async def boom(**kwargs):
        raise APIError("down", request=None, body=None)  # type: ignore[arg-type]

    monkeypatch.setattr(llm._client.chat.completions, "create", boom)

    with pytest.raises(LLMError):
        async for _ in llm.astream([{"role": "user", "content": "hi"}]):
            pass
