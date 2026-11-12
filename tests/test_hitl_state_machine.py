"""HITL as a state machine (0.8.0 spec 5.5): per-call approval, claims, several pauses."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import pytest
from pydantic import BaseModel

import agentflow.agent as agent_module
import agentflow.pipeline as pipeline_module
from agentflow import (
    Agent,
    AgentError,
    ApprovalPolicy,
    EffectUncertainError,
    InMemoryContext,
    LLMResponse,
    Pipeline,
    PipelineError,
    SupervisorAgent,
    tool,
)

DELEGATE = "delegate_task"


def _call(call_id: str, name: str, arguments: dict | None = None) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments or {})}}


def _reply(content: str = "", calls: list | None = None) -> LLMResponse:
    return LLMResponse(content=content, tokens=10, prompt_tokens=6, completion_tokens=4, duration=0.0,
                       model="fake-model", cached=False, tool_calls=calls,
                       finish_reason="tool_calls" if calls else "stop")


class Script:
    model = "fake-model"

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    async def generate(self, messages, tools=None, **kwargs):
        self.requests.append(json.loads(json.dumps(messages, default=str)))
        reply = self.replies.pop(0) if self.replies else _reply("fallback")
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def instant_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def instant(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(agent_module.asyncio, "sleep", instant)
    monkeypatch.setattr(pipeline_module.asyncio, "sleep", instant)


def _pipeline(llm, *agents, blocked=("arm", "gripper"), **kwargs) -> Pipeline:
    pipe = Pipeline(llm=llm, memory=InMemoryContext(), approval_policy=ApprovalPolicy(blocked_tools=list(blocked)),
                    **kwargs)
    for spec, deps in agents:
        pipe.add(spec, depends_on=deps)
    return pipe


def _robot(executed: list):
    @tool(effect="write")
    async def arm(angle: int) -> str:
        executed.append(("arm", angle))
        return f"arm at {angle}"

    @tool(effect="write")
    async def gripper(state: str) -> str:
        executed.append(("gripper", state))
        return f"gripper {state}"

    @Agent(name="robot", role="r", tools=[arm, gripper], max_tool_iterations=6)
    async def robot(task, context):
        return task

    return robot


# --- HITL-1: each call is decided on its own; a rejection halts the batch ---


@pytest.mark.asyncio
async def test_rejecting_one_call_halts_the_rest_of_its_batch():
    executed: list = []
    llm = Script([_reply(calls=[_call("1", "arm", {"angle": 90}), _call("2", "gripper", {"state": "open"})]),
                  _reply("stopped")])
    pipe = _pipeline(llm, (_robot(executed), None))

    paused = await pipe.run("go")
    result = await pipe.resume(paused.pause_info["session_id"], "no", approved=False)

    assert executed == [], "the second blocked call must not run after the first was rejected (HITL-1, B5b)"
    trace = result.results["robot"].metadata["tool_calls"]
    assert "[NOT RUN]" in trace[-1]["result"], f"the halted call is reported to the model: {trace}"


@pytest.mark.asyncio
async def test_approving_one_call_does_not_approve_the_next_blocked_call():
    executed: list = []
    llm = Script([_reply(calls=[_call("1", "arm", {"angle": 90}), _call("2", "gripper", {"state": "open"})]),
                  _reply("done")])
    pipe = _pipeline(llm, (_robot(executed), None))

    paused = await pipe.run("go")
    session = paused.pause_info["session_id"]
    second = await pipe.resume(session, "ok", approved=True)

    assert second.status == "paused", "the second blocked call needs its own approval"
    assert second.pause_info["tool_name"] == "gripper"
    assert executed == [("arm", 90)]

    final = await pipe.resume(session, "ok", approved=True)

    assert final.status == "completed"
    assert executed == [("arm", 90), ("gripper", "open")]


# --- HITL-2, HITL-3: an approved call runs once ------------------------------


@pytest.mark.asyncio
async def test_concurrent_resumes_of_one_session_run_the_approved_call_once():
    executed: list = []
    gate = asyncio.Event()

    @tool(effect="write")
    async def arm(angle: int) -> str:
        await gate.wait()  # held open until both resumes have started
        executed.append(("arm", angle))
        return "moved"

    @Agent(name="robot", role="r", tools=[arm])
    async def robot(task, context):
        return task

    llm = Script([_reply(calls=[_call("1", "arm", {"angle": 90})]), _reply("done"), _reply("done")])
    pipe = _pipeline(llm, (robot, None))
    paused = await pipe.run("go")
    session = paused.pause_info["session_id"]

    both = asyncio.gather(pipe.resume(session, "ok", approved=True),
                          pipe.resume(session, "ok", approved=True), return_exceptions=True)
    for _ in range(20):
        await asyncio.sleep(0)
    gate.set()
    outcomes = await both

    assert executed == [("arm", 90)], "one approval, one motion (HITL-2)"
    assert any(isinstance(o, PipelineError) for o in outcomes), f"the second resume must be refused: {outcomes}"


@pytest.mark.asyncio
async def test_retrying_a_failed_resume_does_not_repeat_the_approved_call():
    executed: list = []
    llm = Script([_reply(calls=[_call("1", "arm", {"angle": 90})]),
                  RuntimeError("down"),  # the resumed agent fails after the call ran (one retry layer: one error)
                  _reply("done")])
    pipe = _pipeline(llm, (_robot(executed), None))
    paused = await pipe.run("go")
    session = paused.pause_info["session_id"]

    with pytest.raises(AgentError):
        await pipe.resume(session, "ok", approved=True)
    result = await pipe.resume(session, "ok", approved=True)

    assert executed == [("arm", 90)], "the retried resume must reuse the recorded result (HITL-3)"
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_an_approved_call_that_never_finished_is_in_doubt():
    executed: list = []

    @tool(effect="write")
    async def arm(angle: int) -> str:
        executed.append(angle)
        raise asyncio.CancelledError  # the process died mid-call

    @Agent(name="robot", role="r", tools=[arm])
    async def robot(task, context):
        return task

    llm = Script([_reply(calls=[_call("1", "arm", {"angle": 90})]), _reply("done"), _reply("done")])
    pipe = _pipeline(llm, (robot, None))
    paused = await pipe.run("go")
    session = paused.pause_info["session_id"]
    with pytest.raises(asyncio.CancelledError):
        await pipe.resume(session, "ok", approved=True)

    with pytest.raises(EffectUncertainError):
        await pipe.resume(session, "ok", approved=True)
    assert executed == [90], "an in-doubt call is never repeated by default"

    skipped = await pipe.resume(session, "ok", approved=True, on_in_doubt="skip")
    assert skipped.status == "completed", "the operator checked the robot and chose to continue"
    assert executed == [90]


# --- HITL-4: several pauses in one level -------------------------------------


@pytest.mark.asyncio
async def test_two_paused_siblings_are_both_resumed_before_dependents_run():
    executed: list = []

    def arm_agent(name: str, angle: int):
        @tool(name=f"{name}_arm", effect="write")
        async def move(a: int) -> str:
            executed.append((name, a))
            return "moved"

        @Agent(name=name, role="r", tools=[move])
        async def spec(task, context):
            return task

        return spec, move.name

    left, left_tool = arm_agent("left", 1)
    right, right_tool = arm_agent("right", 2)

    @Agent(name="report", role="r")
    async def report(task, context):
        return f"left={context.get('left')} right={context.get('right')}"

    replies = {"left": [_reply(calls=[_call("l", left_tool, {"a": 1})]), _reply("left done")],
               "right": [_reply(calls=[_call("r", right_tool, {"a": 2})]), _reply("right done")]}

    class PerAgent(Script):
        async def generate(self, messages, tools=None, **kwargs):
            for name, queue in replies.items():
                if tools and any(t["function"]["name"] == f"{name}_arm" for t in tools):
                    return queue.pop(0)
            return _reply(f"report saw: {messages[-1]['content'][:200]}")

    pipe = _pipeline(PerAgent([]), (left, None), (right, None), (report, ["left", "right"]),
                     blocked=(left_tool, right_tool))
    first = await pipe.run("go")
    session = first.pause_info["session_id"]
    second = await pipe.resume(session, "ok", approved=True)

    assert second.status == "paused", "the second paused sibling must not be lost (HITL-4)"
    final = await pipe.resume(session, "ok", approved=True)

    assert final.status == "completed"
    assert sorted(executed) == [("left", 1), ("right", 2)]
    assert set(final.results) == {"left", "right", "report"}, "the dependent runs only after both"


# --- HITL-5, HITL-6, HITL-7, HITL-8 ------------------------------------------


@pytest.mark.asyncio
async def test_pause_state_does_not_leak_into_an_agent_prompt():
    executed: list = []
    memory = InMemoryContext()

    @Agent(name="writer", role="r", memory=memory)
    async def writer(task, context):
        return "summarize"

    llm = Script([_reply(calls=[_call("1", "arm", {"angle": 90})]), _reply("done"), _reply("summary")])
    pipe = Pipeline(llm=llm, memory=memory, approval_policy=ApprovalPolicy(blocked_tools=["arm"]))
    pipe.add(_robot(executed))
    pipe.add(writer, depends_on=["robot"])

    paused = await pipe.run("go")
    await pipe.resume(paused.pause_info["session_id"], "ok", approved=True)

    prompts = json.dumps(llm.requests)
    assert "__hitl" not in prompts and "paused_agent_name" not in prompts, "HITL-5"


@pytest.mark.asyncio
async def test_a_datetime_in_a_completed_siblings_output_survives_the_pause():
    class Reading(BaseModel):
        at: datetime
        value: float

    @Agent(name="sensor", role="r", output_schema=Reading)
    async def sensor(task, context):
        return "read"

    executed: list = []

    class Router(Script):
        async def generate(self, messages, tools=None, **kwargs):
            if tools:
                return self.replies.pop(0)
            return _reply('{"at": "2026-11-12T10:00:00", "value": 1.5}')

    llm = Router([_reply(calls=[_call("1", "arm", {"angle": 90})]), _reply("done")])
    pipe = _pipeline(llm, (sensor, None), (_robot(executed), None))

    paused = await pipe.run("go")

    assert paused.status == "paused", "a datetime in a sibling's output must not break persistence (HITL-6)"
    result = await pipe.resume(paused.pause_info["session_id"], "ok", approved=True)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_a_policy_without_memory_says_the_pause_cannot_be_resumed(caplog):
    executed: list = []
    llm = Script([_reply(calls=[_call("1", "arm", {"angle": 90})])])
    with caplog.at_level(logging.WARNING, logger="agentflow.pipeline"):
        pipe = Pipeline(llm=llm, approval_policy=ApprovalPolicy(blocked_tools=["arm"]))
    pipe.add(_robot(executed))

    paused = await pipe.run("go")

    assert paused.pause_info["resumable"] is False, "HITL-7"
    assert any("memory" in r.getMessage() for r in caplog.records), "the misconfiguration is reported once"


@pytest.mark.asyncio
async def test_every_tool_call_is_answered_after_a_resume():
    @tool
    async def lookup(q: str) -> str:
        return "r"

    executed: list = []

    @tool(effect="write")
    async def arm(angle: int) -> str:
        executed.append(angle)
        return "ok"

    @Agent(name="robot", role="r", tools=[lookup, arm])
    async def robot(task, context):
        return task

    llm = Script([
        _reply(calls=[_call("0", "lookup", {"q": "a"})]),
        _reply(calls=[_call("1", "arm", {"angle": 1}), _call("2", "lookup", {"q": "a"})]),
        _reply("done"),
    ])
    pipe = _pipeline(llm, (robot, None), blocked=("arm",))
    paused = await pipe.run("go")
    await pipe.resume(paused.pause_info["session_id"], "ok", approved=True)

    last = llm.requests[-1]
    asked = {c["id"] for m in last if m["role"] == "assistant" for c in (m.get("tool_calls") or [])}
    answered = {m["tool_call_id"] for m in last if m["role"] == "tool"}
    assert asked <= answered, f"unanswered tool calls make the request invalid (HITL-8): {asked - answered}"


# --- HITL-9 ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_supervisor_worker_cannot_bypass_the_approval_policy():
    executed: list = []

    @tool(effect="write")
    async def actuate() -> str:
        executed.append(1)
        return "done"

    @Agent(name="worker", role="w", tools=[actuate])
    async def worker(task, context):
        return task

    class Router(Script):
        async def generate(self, messages, tools=None, **kwargs):
            names = {t["function"]["name"] for t in (tools or [])}
            if "actuate" in names:
                return _reply(calls=[_call("w1", "actuate")]) if len(messages) < 4 else _reply("worker done")
            if len(messages) < 4:
                return _reply(calls=[_call("s1", DELEGATE, {"worker_name": "worker", "sub_task": "go"})])
            return _reply("supervisor done")

    pipe = Pipeline(llm=Router([]), approval_policy=ApprovalPolicy(blocked_tools=["actuate"]))
    pipe.add(SupervisorAgent(name="boss", role="b", workers=[worker]))

    await pipe.run("go")

    assert executed == [], "a blocked tool must not run just because a worker called it (HITL-9)"
