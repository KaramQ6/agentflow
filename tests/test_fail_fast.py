"""A failing DAG level must stop paying for its siblings.

`asyncio.gather(..., return_exceptions=True)` waits for every sibling even
after one has raised, and the level's results are discarded anyway — so those
siblings only ever burned tokens. These tests pin the cancellation, and pin the
one case that must NOT cancel: a HITL pause, whose siblings' results are
persisted with the pause state.
"""

from __future__ import annotations

import asyncio

import pytest

from agentflow import AgentResult, Pipeline
from agentflow.agent import BaseAgent
from agentflow.exceptions import AgentError, BudgetExceededError
from agentflow.hitl import PauseExecution


class MockLLM:
    async def generate(self, messages, **kwargs):
        return {"content": "ok", "tokens": 1, "duration": 0.0, "model": "mock-model"}


class SlowAgent(BaseAgent):
    """Records whether it ran to completion or was cancelled mid-flight."""

    def __init__(self, name: str, delay: float = 5.0):
        super().__init__(name=name, role="slow")
        self._delay = delay
        self.completed = False
        self.cancelled = False

    async def execute(self, task, context, llm):
        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.completed = True
        return AgentResult(agent=self.name, output="slow done")


class FailingAgent(BaseAgent):
    def __init__(self, name: str, delay: float = 0.0):
        super().__init__(name=name, role="failing")
        self._delay = delay

    async def execute(self, task, context, llm):
        if self._delay:
            await asyncio.sleep(self._delay)
        raise AgentError(self.name, "boom")


class PausingAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name=name, role="pausing")

    async def execute(self, task, context, llm):
        raise PauseExecution(
            agent_name=self.name,
            tool_name="dangerous_tool",
            tool_arguments="{}",
            tool_call_id="call_1",
            messages=[],
            total_tokens=0,
            total_cost=0.0,
            model_name="mock-model",
            trace=[],
            pending_calls=[],
            seen_calls=[],
            iterations_used=0,
        )


class CostlyAgent(BaseAgent):
    def __init__(self, name: str, cost: float):
        super().__init__(name=name, role="costly")
        self._cost = cost

    async def execute(self, task, context, llm):
        return AgentResult(
            agent=self.name, output=f"{self.name} output", cost=self._cost, tokens_used=10
        )


@pytest.mark.asyncio
async def test_failing_agent_cancels_its_siblings():
    slow = SlowAgent("slow", delay=5.0)
    pipe = Pipeline(llm=MockLLM())
    pipe.add(slow)
    pipe.add(FailingAgent("failer"))

    with pytest.raises(AgentError, match="boom"):
        await asyncio.wait_for(pipe.run("task"), timeout=2.0)

    assert slow.cancelled, "sibling kept running after the level had already failed"
    assert not slow.completed


@pytest.mark.asyncio
async def test_the_reported_error_is_the_real_failure_not_a_cancellation():
    """The error surfaced must name the agent that failed, whatever the order."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("slow_a", delay=5.0))
    pipe.add(SlowAgent("slow_b", delay=5.0))
    pipe.add(FailingAgent("failer", delay=0.05))

    with pytest.raises(AgentError) as exc_info:
        await asyncio.wait_for(pipe.run("task"), timeout=2.0)

    assert exc_info.value.agent_name == "failer"


@pytest.mark.asyncio
async def test_a_pause_does_not_cancel_its_siblings():
    """PauseExecution is control flow: sibling results are persisted with it."""
    sibling = SlowAgent("sibling", delay=0.05)
    pipe = Pipeline(llm=MockLLM())
    pipe.add(PausingAgent("pauser"))
    pipe.add(sibling)

    result = await asyncio.wait_for(pipe.run("task"), timeout=2.0)

    assert result.status == "paused"
    assert sibling.completed, "a pause must let siblings finish"
    assert not sibling.cancelled
    assert "sibling" in result.results


@pytest.mark.asyncio
async def test_successful_level_still_returns_every_result():
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("a", delay=0.01))
    pipe.add(SlowAgent("b", delay=0.01))
    pipe.add(SlowAgent("c", delay=0.01))

    result = await pipe.run("task")

    assert sorted(result.results) == ["a", "b", "c"]
    assert result.levels_executed == 1


@pytest.mark.asyncio
async def test_abandoning_a_stream_cancels_in_flight_agents():
    slow = SlowAgent("slow", delay=5.0)
    pipe = Pipeline(llm=MockLLM())
    pipe.add(slow)

    stream = pipe.stream("task")
    await stream.__anext__()  # agent_start
    await stream.aclose()
    await asyncio.sleep(0)  # let the cancellation propagate

    assert not slow.completed


# ── Budget ceilings must not destroy work already paid for ────────────────────


@pytest.mark.asyncio
async def test_budget_error_carries_the_results_already_paid_for():
    pipe = Pipeline(llm=MockLLM(), budget_usd=0.10)
    pipe.add(CostlyAgent("first", cost=0.08))
    pipe.add(CostlyAgent("second", cost=0.50), depends_on=["first"])
    pipe.add(CostlyAgent("third", cost=0.01), depends_on=["second"])

    with pytest.raises(BudgetExceededError) as exc_info:
        await pipe.run("task")

    partial = exc_info.value.partial_result
    assert partial is not None
    assert set(partial.results) == {"first", "second"}
    assert partial.total_cost == pytest.approx(0.58)
    assert partial.levels_executed == 2
    assert "third" not in partial.results, "the level after the breach must not run"


@pytest.mark.asyncio
async def test_budget_error_reports_budget_and_spend():
    pipe = Pipeline(llm=MockLLM(), budget_usd=0.05)
    pipe.add(CostlyAgent("only", cost=0.30))

    with pytest.raises(BudgetExceededError) as exc_info:
        await pipe.run("task")

    assert exc_info.value.budget_usd == 0.05
    assert exc_info.value.spent_usd == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_run_within_budget_does_not_raise():
    pipe = Pipeline(llm=MockLLM(), budget_usd=1.00)
    pipe.add(CostlyAgent("only", cost=0.30))

    result = await pipe.run("task")

    assert result.total_cost == pytest.approx(0.30)
