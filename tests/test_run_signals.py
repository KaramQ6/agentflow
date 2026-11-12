"""Run-level signals and serve() (0.8.0 spec 5.8, 5.9): hooks that always close, tool failures
that show up on the result, an honest error for an agent-internal cancel, and a bounded serve()."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agentflow import Agent, AgentError, BaseAgent, Hooks, Pipeline, tool
from agentflow.testing import ScriptedLLM, call, reply
from agentflow.triggers import BaseTrigger
from agentflow.types import AgentResult


class Recorder(Hooks):
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_pipeline_start(self, task, run_id, agent_count):
        self.events.append(("start", run_id))

    def on_pipeline_end(self, result):
        self.events.append(("end", result.run_id))

    def on_pipeline_error(self, error, run_id):
        self.events.append(("error", type(error).__name__, run_id))

    def on_tool_error(self, agent, tool_name, error):
        self.events.append(("tool_error", agent, tool_name))


class Echo(BaseAgent):
    def __init__(self, name: str = "echo"):
        super().__init__(name, "echo")

    async def execute(self, task, context, llm):
        return AgentResult(agent=self.name, output=f"echo: {task}")


class Boom(BaseAgent):
    def __init__(self) -> None:
        super().__init__("boom", "boom")

    async def execute(self, task, context, llm):
        raise AgentError(self.name, "boom")


# --- hooks (HOOK-1, HOOK-2) -----------------------------------------------------


@pytest.mark.asyncio
async def test_stream_fires_the_same_pipeline_hooks_as_run():
    hooks = Recorder()
    pipe = Pipeline(llm=ScriptedLLM([]), hooks=hooks)
    pipe.add(Echo())

    events = [e async for e in pipe.stream("go")]

    assert [e.type for e in events][-1] == "pipeline_complete"
    assert [name for name, *_ in hooks.events] == ["start", "end"], (
        f"stream() must open and close the pipeline span like run() (HOOK-1): {hooks.events}"
    )
    assert hooks.events[0][1] == hooks.events[1][1], "start and end belong to one run"


@pytest.mark.asyncio
async def test_a_failed_run_closes_its_span_with_an_error_hook():
    hooks = Recorder()
    pipe = Pipeline(llm=ScriptedLLM([]), hooks=hooks)
    pipe.add(Boom())

    with pytest.raises(AgentError):
        await pipe.run("go")

    assert [e[0] for e in hooks.events] == ["start", "error"], f"HOOK-2: {hooks.events}"
    assert hooks.events[1][1] == "AgentError"
    assert hooks.events[0][1] == hooks.events[1][2], "the error closes the run that started"


@pytest.mark.asyncio
async def test_a_failed_stream_fires_the_error_hook_too():
    hooks = Recorder()
    pipe = Pipeline(llm=ScriptedLLM([]), hooks=hooks)
    pipe.add(Boom())

    events = [e async for e in pipe.stream("go")]

    assert events[-1].type == "pipeline_error"
    assert [e[0] for e in hooks.events] == ["start", "error"]


# --- PIPE-1 -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_agent_that_cancels_itself_raises_a_named_agent_error():
    class SelfCancel(BaseAgent):
        def __init__(self) -> None:
            super().__init__("flaky", "flaky")

        async def execute(self, task, context, llm):
            raise asyncio.CancelledError  # e.g. a library inside the agent cancelled its own task

    pipe = Pipeline(llm=ScriptedLLM([]))
    pipe.add(SelfCancel())

    with pytest.raises(AgentError, match="flaky") as caught:
        await pipe.run("go")

    assert "cancel" in str(caught.value).lower(), "a bare AssertionError told nobody anything (PIPE-1)"


# --- tool errors on the result (NEW-11, NEW-13) ------------------------------------


def _agent_with(tools: list) -> Any:
    @Agent(name="pilot", role="r", tools=tools)
    async def pilot(task, context):
        return task

    return pilot


@pytest.mark.asyncio
async def test_a_failing_write_tool_is_listed_on_the_result_and_reported_to_hooks():
    @tool(effect="write")
    async def move() -> str:
        raise ConnectionError("link lost mid-move")

    hooks = Recorder()
    llm = ScriptedLLM([reply(calls=[call("1", "move")]), reply("All done, the move succeeded.")])
    pipe = Pipeline(llm=llm, hooks=hooks)
    pipe.add(_agent_with([move]))

    result = await pipe.run("go")

    assert result.status == "completed", "status keeps its two values; the failure is additive"
    (record,) = result.tool_errors
    assert (record["agent"], record["tool"], record["effect"], record["kind"]) == ("pilot", "move", "write", "failed")
    assert "link lost" in record["error"], "NEW-13: the robot's state is unknown and the result must say so"
    assert ("tool_error", "pilot", "move") in hooks.events


@pytest.mark.asyncio
async def test_a_call_to_a_tool_that_does_not_exist_is_reported():
    @tool(effect="read")
    async def ping() -> str:
        return "pong"

    llm = ScriptedLLM([reply(calls=[call("1", "send_report", {"data": "secrets"})]), reply("done")])
    pipe = Pipeline(llm=llm)
    pipe.add(_agent_with([ping]))

    result = await pipe.run("go")

    (record,) = result.tool_errors
    assert (record["tool"], record["kind"]) == ("send_report", "unknown_tool"), (
        "a model asking for a tool it was never given is a security event (NEW-11)"
    )


@pytest.mark.asyncio
async def test_invalid_arguments_are_listed_as_such():
    @tool(effect="read")
    async def lookup(q: str) -> str:
        return "r"

    llm = ScriptedLLM([reply(calls=[call("1", "lookup", {"wrong": 1})]), reply("done")])
    pipe = Pipeline(llm=llm)
    pipe.add(_agent_with([lookup]))

    result = await pipe.run("go")

    assert [r["kind"] for r in result.tool_errors] == ["invalid_arguments"]


@pytest.mark.asyncio
async def test_a_clean_run_has_no_tool_errors():
    @tool(effect="read")
    async def ping() -> str:
        return "pong"

    pipe = Pipeline(llm=ScriptedLLM([reply(calls=[call("1", "ping")]), reply("done")]))
    pipe.add(_agent_with([ping]))

    assert (await pipe.run("go")).tool_errors == []


# --- serve() (MQTT-3, SERVE-1) -------------------------------------------------------


class Items(BaseTrigger):
    def __init__(self, items: list[tuple[str, dict[str, Any]]], *, gap: float = 0.0, late: tuple | None = None):
        self._items = items
        self._gap = gap
        self._late = late

    async def listen(self):
        for item in self._items:
            yield item
            await asyncio.sleep(self._gap)  # sleep(0) lets a worker take the item
        if self._late is not None:
            await asyncio.sleep(self._late[0])
            yield self._late[1]


class Gated(BaseAgent):
    """Records every task and blocks on a gate, so a backlog can build up."""

    def __init__(self, gate: asyncio.Event | None = None):
        super().__init__("gated", "gated")
        self.gate = gate
        self.tasks: list[str] = []

    async def execute(self, task, context, llm):
        self.tasks.append(task.splitlines()[0])
        if self.gate is not None:
            await self.gate.wait()
        return AgentResult(agent=self.name, output="ok")


@pytest.mark.asyncio
async def test_the_payload_reaches_the_prompt_once():
    agent = Gated()
    pipe = Pipeline(llm=ScriptedLLM([]))
    pipe.add(agent)
    raw = json.dumps({"marker_value": 1})
    context = {"topic": "t", "qos": 0, "payload_raw": raw, "payload": {"marker_value": 1}}
    seen: list[str] = []

    class Spy(Gated):
        async def execute(self, task, context, llm):
            seen.append(task)
            return AgentResult(agent=self.name, output="ok")

    pipe = Pipeline(llm=ScriptedLLM([]))
    pipe.add(Spy())

    await pipe.serve(Items([(f"Analyze this data: {raw}", context)]), max_concurrent=1)

    assert seen[0].count("marker_value") == 1, f"the payload was sent three times, all paid for (MQTT-3): {seen[0]}"
    assert '"topic": "t"' in seen[0], "the rest of the context still reaches the agents"


@pytest.mark.asyncio
async def test_by_default_a_full_queue_makes_the_trigger_wait_and_nothing_is_lost():
    agent = Gated()
    pipe = Pipeline(llm=ScriptedLLM([]))
    pipe.add(agent)
    items = [(f"t{i}", {}) for i in range(8)]

    await pipe.serve(Items(items), max_concurrent=1, max_pending=2)

    assert agent.tasks == [f"t{i}" for i in range(8)], "waiting means no data is dropped"


@pytest.mark.asyncio
async def test_drop_oldest_keeps_the_newest_readings_when_the_queue_is_full():
    gate = asyncio.Event()
    agent = Gated(gate)
    pipe = Pipeline(llm=ScriptedLLM([]))
    pipe.add(agent)
    asyncio.get_running_loop().call_later(0.15, gate.set)
    items = [(f"t{i}", {}) for i in range(1, 7)]

    await pipe.serve(Items(items), max_concurrent=1, max_pending=2, overflow="drop_oldest")

    assert agent.tasks == ["t1", "t5", "t6"], (
        f"stale readings must not queue up behind a slow model (SERVE-1): {agent.tasks}"
    )


@pytest.mark.asyncio
async def test_a_reading_older_than_max_age_is_dropped_unread():
    gate = asyncio.Event()
    agent = Gated(gate)
    pipe = Pipeline(llm=ScriptedLLM([]))
    pipe.add(agent)
    asyncio.get_running_loop().call_later(0.2, gate.set)
    trigger = Items([("t1", {}), ("t2", {}), ("t3", {})], gap=0.0, late=(0.3, ("t4", {})))

    await pipe.serve(trigger, max_concurrent=1, max_pending=10, max_age_s=0.05)

    assert agent.tasks == ["t1", "t4"], f"t2 and t3 were stale when the model came free: {agent.tasks}"


@pytest.mark.asyncio
async def test_serve_rejects_a_nonsense_concurrency():
    pipe = Pipeline(llm=ScriptedLLM([]))
    pipe.add(Echo())

    with pytest.raises(Exception, match="max_concurrent"):
        await pipe.serve(Items([]), max_concurrent=0)
