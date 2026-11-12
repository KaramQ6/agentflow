"""Tool effects (0.8.0 spec 5.2, ADR 0006) and the effect ledger (5.3, ADR 0004 amended)."""

from __future__ import annotations

import asyncio
import json

import pytest

import agentflow.agent as agent_module
import agentflow.pipeline as pipeline_module
from agentflow import (
    Agent,
    AgentError,
    EffectUncertainError,
    InMemoryContext,
    LLMResponse,
    Pipeline,
    Tool,
    tool,
)
from agentflow.effects import EffectRecord, EffectStatus, InMemoryLedger, SqliteLedger


def _call(call_id: str, name: str, arguments: dict | None = None) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments or {})}}


def _reply(content: str = "", calls: list | None = None) -> LLMResponse:
    return LLMResponse(content=content, tokens=10, prompt_tokens=6, completion_tokens=4, duration=0.0,
                       model="fake-model", cached=False, tool_calls=calls,
                       finish_reason="tool_calls" if calls else "stop")


class Script:
    """An LLM stand-in that plays a list of replies; an Exception entry is raised."""

    model = "fake-model"

    def __init__(self, replies: list):
        self.replies = list(replies)

    async def generate(self, messages, tools=None, **kwargs):
        reply = self.replies.pop(0)
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


def _agent(tools, name="a"):
    @Agent(name=name, role="r", tools=tools, max_tool_iterations=8)
    async def a(task, context):
        return task

    return a


# --- the effect field -------------------------------------------------------


def test_a_write_tool_must_be_async_so_it_can_be_cancelled():
    def move(x: float) -> str:
        return "moved"

    with pytest.raises(TypeError, match="async"):
        Tool(move, effect="write")


def test_an_unknown_effect_is_rejected_at_decoration():
    async def f() -> str:
        return ""

    with pytest.raises(ValueError, match="effect"):
        Tool(f, effect="delete")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_read_tool_can_be_called_again_with_the_same_arguments():
    reads = []

    @tool(effect="read")
    async def get_pose() -> str:
        reads.append(1)
        return "x=1"

    llm = Script([_reply(calls=[_call("1", "get_pose")]), _reply(calls=[_call("2", "get_pose")]), _reply("done")])
    pipe = Pipeline(llm=llm)
    pipe.add(_agent([get_pose]))

    await pipe.run("go")

    assert len(reads) == 2, "re-reading a sensor is a normal request (LOOP-1)"


@pytest.mark.asyncio
async def test_a_write_tool_can_repeat_the_same_command_in_one_run():
    moves = []

    @tool(effect="write")
    async def move(linear: float) -> str:
        moves.append(linear)
        return "moved"

    llm = Script([_reply(calls=[_call("1", "move", {"linear": 1.0})]),
                  _reply(calls=[_call("2", "move", {"linear": 1.0})]), _reply("done")])
    pipe = Pipeline(llm=llm)
    pipe.add(_agent([move]))

    await pipe.run("go")

    assert moves == [1.0, 1.0], "forward 1 m, then again, is two moves (NEW-14)"


@pytest.mark.asyncio
async def test_an_unclassified_tool_keeps_the_duplicate_guard():
    calls = []

    @tool
    async def lookup(q: str) -> str:
        calls.append(q)
        return "r"

    llm = Script([_reply(calls=[_call("1", "lookup", {"q": "a"})]),
                  _reply(calls=[_call("2", "lookup", {"q": "a"})]), _reply("done")])
    pipe = Pipeline(llm=llm)
    pipe.add(_agent([lookup]))

    await pipe.run("go")

    assert calls == ["a"], "tools without an effect keep 0.7 behaviour"


@pytest.mark.asyncio
async def test_write_calls_in_one_reply_run_one_at_a_time_in_order():
    events = []

    @tool(effect="write")
    async def step(n: int) -> str:
        events.append(f"{n}:start")
        await asyncio.sleep(0)
        events.append(f"{n}:end")
        return "ok"

    llm = Script([_reply(calls=[_call("1", "step", {"n": 1}), _call("2", "step", {"n": 2})]), _reply("done")])
    pipe = Pipeline(llm=llm)
    pipe.add(_agent([step]))

    await pipe.run("go")

    assert events == ["1:start", "1:end", "2:start", "2:end"], (
        "side effects in one reply must not overlap (NEW-12, B2b)"
    )


# --- exactly once across retries --------------------------------------------


@pytest.mark.asyncio
async def test_a_retried_agent_replays_a_completed_write_instead_of_repeating_it():
    charges = []

    @tool(effect="write")
    async def charge(customer: str, amount: int) -> str:
        charges.append((customer, amount))
        return "charged"

    args = {"customer": "ali", "amount": 50}
    llm = Script([
        _reply(calls=[_call("1", "charge", args)]), RuntimeError("provider down"), RuntimeError("provider down"),
        _reply(calls=[_call("1", "charge", args)]), _reply("done"),
    ])
    pipe = Pipeline(llm=llm, retry_failed_agents=1)
    pipe.add(_agent([charge]))

    result = await pipe.run("go")

    assert charges == [("ali", 50)], "the retry must not charge the card a second time"
    assert result.results["a"].metadata["tool_calls"][0]["result"] == "charged", "the stored result is replayed"


@pytest.mark.asyncio
async def test_after_a_retry_a_second_identical_write_still_runs():
    charges = []

    @tool(effect="write")
    async def charge(amount: int) -> str:
        charges.append(amount)
        return f"charged #{len(charges)}"

    args = {"amount": 5}
    llm = Script([
        _reply(calls=[_call("1", "charge", args)]), RuntimeError("down"), RuntimeError("down"),
        _reply(calls=[_call("1", "charge", args)]), _reply(calls=[_call("2", "charge", args)]), _reply("done"),
    ])
    pipe = Pipeline(llm=llm, retry_failed_agents=1)
    pipe.add(_agent([charge]))

    await pipe.run("go")

    assert charges == [5, 5], "the first repeat replays (occurrence 0); a new occurrence (1) runs"


@pytest.mark.asyncio
async def test_an_effect_left_pending_raises_and_is_never_retried():
    class StuckLedger(InMemoryLedger):
        async def claim(self, key):
            return EffectRecord(key, EffectStatus.PENDING, "")

    ran = []

    @tool(effect="write")
    async def actuate() -> str:
        ran.append(1)
        return "done"

    llm = Script([_reply(calls=[_call("1", "actuate")]), _reply(calls=[_call("1", "actuate")])])
    pipe = Pipeline(llm=llm, retry_failed_agents=3, effect_ledger=StuckLedger())
    pipe.add(_agent([actuate]))

    with pytest.raises(EffectUncertainError):
        await pipe.run("go")

    assert ran == [], "an effect that may already have happened must not run again"
    assert len(llm.replies) == 1, "an uncertain effect is not retried"


@pytest.mark.asyncio
async def test_a_failing_ledger_fails_closed():
    class BrokenLedger(InMemoryLedger):
        async def claim(self, key):
            raise OSError("disk full")

    ran = []

    @tool(effect="write")
    async def actuate() -> str:
        ran.append(1)
        return "done"

    llm = Script([_reply(calls=[_call("1", "actuate")]), _reply("gave up")])
    pipe = Pipeline(llm=llm, effect_ledger=BrokenLedger())
    pipe.add(_agent([actuate]))

    result = await pipe.run("go")

    assert ran == [], "if the ledger cannot record the effect, the effect must not happen"
    assert "disk full" in result.results["a"].metadata["tool_calls"][0]["result"]


# --- decision 4: retries after unclassified tools ---------------------------


@pytest.mark.asyncio
async def test_an_agent_that_ran_an_unclassified_tool_is_not_retried():
    @tool
    async def send_email(to: str) -> str:
        return "sent"

    llm = Script([_reply(calls=[_call("1", "send_email", {"to": "x"})]), RuntimeError("down"), RuntimeError("down"),
                  _reply("should never be reached")])
    pipe = Pipeline(llm=llm, retry_failed_agents=2)
    pipe.add(_agent([send_email]))

    with pytest.raises(AgentError, match="effect="):
        await pipe.run("go")

    assert len(llm.replies) == 1, "retrying could send the email twice; the error says how to opt in"


@pytest.mark.asyncio
async def test_an_agent_that_ran_only_read_tools_is_retried():
    @tool(effect="read")
    async def get_pose() -> str:
        return "x=1"

    llm = Script([_reply(calls=[_call("1", "get_pose")]), RuntimeError("down"), RuntimeError("down"),
                  _reply("done")])
    pipe = Pipeline(llm=llm, retry_failed_agents=1)
    pipe.add(_agent([get_pose]))

    result = await pipe.run("go")

    assert result.output == "done", "reads are safe to repeat, so the retry runs"


# --- durable ledger ---------------------------------------------------------


@pytest.mark.asyncio
async def test_the_sqlite_ledger_survives_a_new_instance(tmp_path):
    path = tmp_path / "effects.db"
    first = SqliteLedger(path)
    assert await first.claim("k") is None, "a new key is owned by the caller"
    await first.settle("k", EffectStatus.DONE, "result")

    record = await SqliteLedger(path).claim("k")

    assert record is not None and record.status is EffectStatus.DONE and record.result == "result"


@pytest.mark.asyncio
async def test_a_failed_record_can_be_claimed_again(tmp_path):
    ledger = SqliteLedger(tmp_path / "effects.db")
    await ledger.claim("k")
    await ledger.settle("k", EffectStatus.FAILED, "boom")

    assert await ledger.claim("k") is None, "the tool reported failure, so running it again is safe"


@pytest.mark.asyncio
async def test_resumable_runs_with_write_tools_warn_when_the_ledger_is_in_memory(caplog):
    @tool(effect="write")
    async def actuate() -> str:
        return "done"

    pipe = Pipeline(llm=Script([_reply("done"), _reply("done")]), memory=InMemoryContext())
    pipe.add(_agent([actuate]))

    with caplog.at_level("WARNING", logger="agentflow.pipeline"):
        await pipe.run("go")
        await pipe.run("go")

    warnings = [r for r in caplog.records if "effect_ledger" in r.getMessage()]
    assert len(warnings) == 1, "a resume in a new process needs a durable ledger; say so once"
