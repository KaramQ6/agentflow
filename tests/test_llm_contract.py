"""The LLM client (0.8.0 spec 5.7): one retry source, honest parameters, visible truncation,
and a replacement seam for tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import openai

try:  # openai 3.x switched its HTTP client to httpx2; 1.x and 2.x use httpx
    import httpx2 as httpx
except ImportError:
    import httpx

import pytest
from pydantic import BaseModel

from agentflow import LLM, Agent, AgentError, LLMError, Pipeline, register_price, tool
from agentflow.testing import ScriptedLLM, call, reply

_REQUEST = httpx.Request("POST", "http://provider.test/v1/chat/completions")


def _status_error(cls, status: int):
    return cls("boom", response=httpx.Response(status, request=_REQUEST), body=None)


def _ok(model: str = "gpt-4o-mini", content: str = "ok", finish: str = "stop", choices: bool = True):
    response = MagicMock()
    if choices:
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = finish
    else:
        response.choices = []
    response.usage.prompt_tokens = 1
    response.usage.completion_tokens = 1
    response.usage.total_tokens = 2
    response.model = model
    return response


def _llm(**kwargs) -> LLM:
    llm = LLM(model="gpt-4o-mini", api_key="fake", retry_base_delay=0.0, retry_jitter=False, **kwargs)
    llm._client = MagicMock()
    return llm


# --- one retry source (LLM-1) --------------------------------------------------


def test_the_sdk_does_not_retry_and_waits_at_most_120_seconds():
    llm = LLM(model="gpt-4o-mini", api_key="fake")

    assert llm._client.max_retries == 0, "the SDK retried on its own, multiplying LLM.max_retries (LLM-1)"
    timeout = llm._client.timeout
    seconds = timeout if isinstance(timeout, (int, float)) else timeout.read
    assert seconds == 120, f"the 600 s SDK default hangs a run for ten minutes (decision 5), got {timeout}"


@pytest.mark.asyncio
async def test_a_bad_api_key_is_not_retried():
    llm = _llm()
    llm._client.chat.completions.create = AsyncMock(side_effect=_status_error(openai.AuthenticationError, 401))

    with pytest.raises(LLMError):
        await llm.generate([{"role": "user", "content": "hi"}])

    assert llm._client.chat.completions.create.await_count == 1, "a 401 cannot succeed on retry (it made 18 calls)"


@pytest.mark.asyncio
async def test_a_server_error_is_retried_exactly_max_retries_times():
    llm = _llm(max_retries=2)
    llm._client.chat.completions.create = AsyncMock(side_effect=_status_error(openai.InternalServerError, 500))

    with pytest.raises(LLMError):
        await llm.generate([{"role": "user", "content": "hi"}])

    assert llm._client.chat.completions.create.await_count == 3, "max_retries=2 means 3 calls, not 9"


@pytest.mark.asyncio
async def test_a_connection_error_is_retried():
    llm = _llm(max_retries=1)
    llm._client.chat.completions.create = AsyncMock(
        side_effect=[openai.APIConnectionError(request=_REQUEST), _ok()]
    )

    response = await llm.generate([{"role": "user", "content": "hi"}])

    assert response.content == "ok"


@pytest.mark.asyncio
async def test_an_agent_adds_no_retries_of_its_own():
    @tool(effect="read")
    async def ping() -> str:
        return "pong"

    @Agent(name="a", role="r", tools=[ping])
    async def a(task, context):
        return task

    llm = _llm(max_retries=2)
    llm._client.chat.completions.create = AsyncMock(side_effect=_status_error(openai.InternalServerError, 500))
    pipe = Pipeline(llm=llm)
    pipe.add(a)

    with pytest.raises(AgentError):
        await pipe.run("go")

    assert llm._client.chat.completions.create.await_count == 3, "the agent loop used to double every retry"


# --- honest parameters (LLM-3, LLM-4) ------------------------------------------


@pytest.mark.asyncio
async def test_none_leaves_temperature_and_max_tokens_out_of_the_request():
    llm = _llm(temperature=None, max_tokens=None)
    llm._client.chat.completions.create = AsyncMock(return_value=_ok())

    await llm.generate([{"role": "user", "content": "hi"}])

    sent = llm._client.chat.completions.create.await_args.kwargs
    assert "temperature" not in sent and "max_tokens" not in sent, (
        f"reasoning models reject these parameters; None must omit them (LLM-3): {sorted(sent)}"
    )


@pytest.mark.asyncio
async def test_an_empty_choices_list_is_an_llm_error():
    llm = _llm(max_retries=0)
    llm._client.chat.completions.create = AsyncMock(return_value=_ok(choices=False))

    with pytest.raises(LLMError, match="choices"):
        await llm.generate([{"role": "user", "content": "hi"}])


# --- visible truncation (NEW-02) -------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_reply_cut_off_by_max_tokens_is_an_error():
    @Agent(name="a", role="r")
    async def a(task, context):
        return task

    pipe = Pipeline(llm=ScriptedLLM([reply("", finish_reason="length")]))
    pipe.add(a)

    with pytest.raises(AgentError, match="max_tokens"):
        await pipe.run("go")


@pytest.mark.asyncio
async def test_a_partial_reply_cut_off_by_max_tokens_is_marked():
    @Agent(name="a", role="r")
    async def a(task, context):
        return task

    pipe = Pipeline(llm=ScriptedLLM([reply("half an ans", finish_reason="length")]))
    pipe.add(a)

    result = await pipe.run("go")

    metadata = result.results["a"].metadata
    assert metadata["finish_reason"] == "length" and metadata["truncated"] is True


@pytest.mark.asyncio
async def test_finish_reason_is_recorded_after_a_tool_loop():
    @tool(effect="read")
    async def ping() -> str:
        return "pong"

    @Agent(name="a", role="r", tools=[ping])
    async def a(task, context):
        return task

    pipe = Pipeline(llm=ScriptedLLM([reply(calls=[call("1", "ping")]), reply("done")]))
    pipe.add(a)

    result = await pipe.run("go")

    assert result.results["a"].metadata["finish_reason"] == "stop"


# --- the replacement seam -----------------------------------------------------------


@pytest.mark.asyncio
async def test_scripted_llm_keeps_the_real_accounting():
    register_price("scripted-priced", prompt_per_1m=1_000_000.0, completion_per_1m=1_000_000.0)

    @Agent(name="a", role="r")
    async def a(task, context):
        return task

    llm = ScriptedLLM([reply("hi", prompt_tokens=3, completion_tokens=2)], model="scripted-priced")
    pipe = Pipeline(llm=llm)
    pipe.add(a)

    result = await pipe.run("go")

    assert result.total_cost == pytest.approx(5.0), "generate() runs unchanged: price, ledger, budget"
    assert llm.requests[0]["messages"][-1]["content"] == "go", "requests are recorded for assertions"


@pytest.mark.asyncio
async def test_scripted_llm_says_when_the_script_runs_out():
    llm = ScriptedLLM([])

    with pytest.raises(LLMError, match="script"):
        await llm.generate([{"role": "user", "content": "hi"}])


# --- schema repair (NEW-10) ---------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_repair_does_not_replay_tool_messages_without_tools():
    class Answer(BaseModel):
        value: int

    @tool(effect="read")
    async def lookup() -> str:
        return "42"

    @Agent(name="a", role="r", tools=[lookup], output_schema=Answer)
    async def a(task, context):
        return task

    llm = ScriptedLLM([reply(calls=[call("1", "lookup")]), reply("not json"), reply('{"value": 42}')])
    pipe = Pipeline(llm=llm)
    pipe.add(a)

    result = await pipe.run("go")

    repair = llm.requests[-1]
    roles = [m["role"] for m in repair["messages"]]
    assert "tool" not in roles or repair.get("tools"), f"tool messages need tools declared (NEW-10): {roles}"
    assert result.results["a"].data == {"value": 42}
    json.dumps(repair, default=str)
