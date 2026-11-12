"""Tool loop correctness (0.8.0 spec 5.6): cancellation, the iteration cap, strict arguments,
results by position, whole-turn trimming, the context guard and the output cap."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

import agentflow.agent as agent_module
from agentflow import LLM, Agent, AgentError, AgentTimeoutError, LLMResponse, Pipeline, tool


def _call(call_id: str, name: str, arguments: dict | str | None = None) -> dict:
    raw = arguments if isinstance(arguments, str) or arguments is None else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def _reply(content: str = "", calls: list | None = None) -> LLMResponse:
    return LLMResponse(content=content, tokens=10, prompt_tokens=6, completion_tokens=4, duration=0.0,
                       model="fake-model", cached=False, tool_calls=calls,
                       finish_reason="tool_calls" if calls else "stop")


class Script:
    model = "fake-model"

    def __init__(self, replies: list, context_window: int | None = None):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.context_window = context_window

    async def generate(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        reply = self.replies.pop(0) if self.replies else _reply("fallback")
        if isinstance(reply, Exception):
            raise reply
        return reply


def _orphans(messages: list[dict]) -> list[str]:
    asked: set[str] = set()
    bad = []
    for m in messages:
        if m["role"] == "assistant":
            asked |= {c["id"] for c in (m.get("tool_calls") or [])}
        elif m["role"] == "tool" and m["tool_call_id"] not in asked:
            bad.append(m["tool_call_id"])
    return bad


# --- cancellation (LOOP-3, B2b, B4b) -----------------------------------------


@pytest.mark.asyncio
async def test_a_timeout_cancels_the_tools_still_running():
    finished = []

    @tool(effect="read")
    async def slow(n: int) -> str:
        await asyncio.sleep(1.0)
        finished.append(n)
        return "late"

    @Agent(name="a", role="r", tools=[slow])
    async def a(task, context):
        return task

    pipe = Pipeline(llm=Script([_reply(calls=[_call("1", "slow", {"n": 1}), _call("2", "slow", {"n": 2})])]))
    pipe.add(a, timeout=0.2)

    with pytest.raises(AgentTimeoutError):
        await pipe.run("go")
    await asyncio.sleep(1.2)

    assert finished == [], "tools must stop with their agent, not finish unobserved (LOOP-3)"


@pytest.mark.asyncio
async def test_a_sync_tool_that_outlives_its_agent_is_named_in_a_warning(caplog):
    import time

    @tool
    def blocking() -> str:
        time.sleep(0.6)
        return "done"

    @Agent(name="a", role="r", tools=[blocking])
    async def a(task, context):
        return task

    pipe = Pipeline(llm=Script([_reply(calls=[_call("1", "blocking")])]))
    pipe.add(a, timeout=0.1)
    with caplog.at_level(logging.WARNING, logger="agentflow.agent"), pytest.raises(AgentTimeoutError):
        await pipe.run("go")

    assert any("blocking" in r.getMessage() and "still running" in r.getMessage() for r in caplog.records), (
        "a thread cannot be cancelled; the user must at least be told (NEW-07)"
    )


# --- iteration cap (NEW-01) ----------------------------------------------------


@pytest.mark.asyncio
async def test_at_the_iteration_cap_the_model_gets_one_last_call_without_tools():
    @tool(effect="read")
    async def read(n: int) -> str:
        return f"page {n}"

    @Agent(name="a", role="r", tools=[read], max_tool_iterations=2)
    async def a(task, context):
        return task

    llm = Script([_reply(calls=[_call("1", "read", {"n": 1})]), _reply(calls=[_call("2", "read", {"n": 2})]),
                  _reply("summary from two pages")])
    pipe = Pipeline(llm=llm)
    pipe.add(a)

    result = await pipe.run("go")

    assert result.output == "summary from two pages", "work done before the cap must not be thrown away"
    assert llm.requests[-1]["tools"] is None, "the last call must not offer tools"


@pytest.mark.asyncio
async def test_if_the_last_call_fails_the_error_carries_usage_and_trace():
    @tool(effect="read")
    async def read(n: int) -> str:
        return "page"

    @Agent(name="a", role="r", tools=[read], max_tool_iterations=2)
    async def a(task, context):
        return task

    llm = Script([_reply(calls=[_call("1", "read", {"n": 1})]), _reply(calls=[_call("2", "read", {"n": 2})]),
                  _reply("")])
    pipe = Pipeline(llm=llm)
    pipe.add(a)

    with pytest.raises(AgentError) as caught:
        await pipe.run("go")

    assert caught.value.tokens == 30, "three calls of 10 tokens were spent and must be reported"
    assert [t["tool"] for t in caught.value.trace] == ["read", "read"], "the tool trace survives the error"


# --- strict arguments (NEW-04, NEW-05, NEW-06) ---------------------------------


async def _run_one_call(target, raw_arguments):
    @Agent(name="a", role="r", tools=[target])
    async def a(task, context):
        return task

    pipe = Pipeline(llm=Script([_reply(calls=[_call("1", target.name, raw_arguments)]), _reply("done")]))
    pipe.add(a)
    result = await pipe.run("go")
    return result.results["a"].metadata["tool_calls"][0]["result"]


@pytest.mark.asyncio
async def test_nan_and_infinity_never_reach_a_tool():
    received = []

    @tool(effect="write")
    async def move(speed: float) -> str:
        received.append(speed)
        return "moved"

    for raw in ('{"speed": NaN}', '{"speed": Infinity}', '{"speed": -Infinity}'):
        output = await _run_one_call(move, raw)
        assert output.startswith("Error"), f"{raw} must be refused, got {output!r}"

    assert received == [], "a non-finite speed must never reach an actuator (NEW-05)"


@pytest.mark.asyncio
async def test_an_unknown_argument_is_refused_instead_of_dropped():
    received = []

    @tool(effect="write")
    async def move(speed: float) -> str:
        received.append(speed)
        return "moved"

    output = await _run_one_call(move, {"speed": 1.0, "speed_limit": 0.2})

    assert output.startswith("Error") and "speed_limit" in output, (
        "silently dropping a limit the model asked for is unsafe (NEW-06)"
    )
    assert received == []


@pytest.mark.asyncio
async def test_null_arguments_mean_no_arguments():
    @tool(effect="read")
    async def ping() -> str:
        return "pong"

    assert await _run_one_call(ping, None) == "pong", "arguments=null must not crash the loop (NEW-04)"


@pytest.mark.asyncio
async def test_colliding_call_ids_keep_each_result_with_its_call():
    @tool(effect="read")
    async def echo(value: str) -> str:
        return f"echo {value}"

    @Agent(name="a", role="r", tools=[echo])
    async def a(task, context):
        return task

    llm = Script([_reply(calls=[_call("same", "echo", {"value": "one"}), _call("same", "echo", {"value": "two"})]),
                  _reply("done")])
    pipe = Pipeline(llm=llm)
    pipe.add(a)
    await pipe.run("go")

    tool_messages = [m["content"] for m in llm.requests[-1]["messages"] if m["role"] == "tool"]
    assert tool_messages == ["echo one", "echo two"], f"each call keeps its own result (NEW-03): {tool_messages}"


# --- trimming and context (LOOP-2, NEW-09, NEW-08) -----------------------------


@pytest.mark.asyncio
async def test_trimming_never_separates_a_tool_result_from_its_call(monkeypatch):
    monkeypatch.setattr(agent_module, "MESSAGES_MAX_LENGTH", 6)

    @tool(effect="read")
    async def read(n: int) -> str:
        return f"page {n}"

    @Agent(name="a", role="r", tools=[read], max_tool_iterations=8)
    async def a(task, context):
        return task

    replies = [_reply(calls=[_call(f"{i}a", "read", {"n": i}), _call(f"{i}b", "read", {"n": i + 100})])
               for i in range(6)] + [_reply("done")]
    llm = Script(replies)
    pipe = Pipeline(llm=llm)
    pipe.add(a)
    await pipe.run("go")

    broken = [r for r in llm.requests if _orphans(r["messages"])]
    assert broken == [], "a batch must be trimmed as one turn (LOOP-2: 6 of 30 real requests were broken)"


@pytest.mark.asyncio
async def test_the_context_guard_trims_whole_turns_to_fit_the_window():
    @tool(effect="read")
    async def read(n: int) -> str:
        return "x" * 300

    @Agent(name="a", role="r", tools=[read], max_tool_iterations=8)
    async def a(task, context):
        return task

    llm = Script([_reply(calls=[_call(str(i), "read", {"n": i})]) for i in range(5)] + [_reply("done")],
                 context_window=400)
    pipe = Pipeline(llm=llm)
    pipe.add(a)
    await pipe.run("go")

    from agentflow.llm import _estimate_prompt_tokens

    sizes = [_estimate_prompt_tokens(r["messages"], r["tools"]) for r in llm.requests]
    assert max(sizes) <= 400, f"every request must fit the declared window (NEW-09), sizes={sizes}"
    assert all(not _orphans(r["messages"]) for r in llm.requests)


@pytest.mark.asyncio
async def test_a_request_that_cannot_fit_the_window_is_refused():
    @Agent(name="a", role="r")
    async def a(task, context):
        return "y" * 3000

    pipe = Pipeline(llm=Script([_reply("never")], context_window=100))
    pipe.add(a)

    with pytest.raises(AgentError, match="context_window"):
        await pipe.run("go")


def test_a_local_endpoint_without_a_context_window_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="agentflow.llm"):
        LLM(model="m", base_url="http://localhost:11434/v1", api_key="x")
        LLM(model="m", base_url="http://127.0.0.1:11434/v1", api_key="x", context_window=8192)

    hits = [r for r in caplog.records if "context_window" in r.getMessage()]
    assert len(hits) == 1, "a local server may truncate silently; say so unless the window is declared"


@pytest.mark.asyncio
async def test_the_tool_output_cap_is_a_setting():
    @tool(effect="read")
    async def big() -> str:
        return "z" * 100

    @Agent(name="a", role="r", tools=[big], tool_output_max_chars=10)
    async def a(task, context):
        return task

    pipe = Pipeline(llm=Script([_reply(calls=[_call("1", "big")]), _reply("done")]))
    pipe.add(a)
    result = await pipe.run("go")

    assert result.results["a"].metadata["tool_calls"][0]["result"].startswith("z" * 10 + "...[Truncated"), (
        "the cap was a hidden constant (NEW-08)"
    )
