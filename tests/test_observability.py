"""Tests for observability hooks wired into Pipeline.run()."""

import pytest

from agentflow import AgentResult, Hooks, Pipeline
from agentflow.agent import BaseAgent


class MockLLM:
    async def generate(self, messages, **kwargs):
        return {"content": "ok", "tokens": 10, "cost": 0.001, "duration": 0.0, "model": "m"}


class SimpleAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name=name, role="role")

    async def execute(self, task, context, llm):
        r = await llm.generate([{"role": "user", "content": task}])
        return AgentResult(agent=self.name, output=r["content"], tokens_used=r["tokens"], cost=r["cost"])


class FailingAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name=name, role="role")

    async def execute(self, task, context, llm):
        raise RuntimeError("boom")


class RecordingHooks(Hooks):
    def __init__(self):
        self.events: list[tuple] = []

    def on_pipeline_start(self, task, run_id, agent_count):
        self.events.append(("pipeline_start", task, run_id, agent_count))

    def on_agent_start(self, agent, level):
        self.events.append(("agent_start", agent, level))

    def on_agent_end(self, result):
        self.events.append(("agent_end", result.agent))

    def on_agent_error(self, agent, error):
        self.events.append(("agent_error", agent, str(error)))

    def on_pipeline_end(self, result):
        self.events.append(("pipeline_end", result.run_id))


@pytest.mark.asyncio
async def test_hooks_fire_in_order():
    hooks = RecordingHooks()
    pipe = Pipeline(llm=MockLLM(), hooks=hooks)
    pipe.add(SimpleAgent("a"))
    pipe.add(SimpleAgent("b"), depends_on=["a"])

    result = await pipe.run("task")

    kinds = [e[0] for e in hooks.events]
    assert kinds[0] == "pipeline_start"
    assert kinds[-1] == "pipeline_end"
    assert kinds.count("agent_start") == 2
    assert kinds.count("agent_end") == 2
    # run_id is consistent start → end → result
    assert hooks.events[0][2] == result.run_id
    assert hooks.events[-1][1] == result.run_id


@pytest.mark.asyncio
async def test_hooks_error_callback_then_raises():
    hooks = RecordingHooks()
    pipe = Pipeline(llm=MockLLM(), hooks=hooks)
    pipe.add(FailingAgent("boom"))

    with pytest.raises(RuntimeError):
        await pipe.run("task")

    assert ("agent_error", "boom", "boom") in hooks.events


@pytest.mark.asyncio
async def test_broken_hook_does_not_break_pipeline():
    class BrokenHooks(Hooks):
        def on_agent_end(self, result):
            raise ValueError("hook exploded")

    pipe = Pipeline(llm=MockLLM(), hooks=BrokenHooks())
    pipe.add(SimpleAgent("a"))

    result = await pipe.run("task")  # must not raise
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_no_hooks_still_works():
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SimpleAgent("a"))
    result = await pipe.run("task")
    assert result.output == "ok"
