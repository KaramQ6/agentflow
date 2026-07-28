"""Pipeline introspection (`explain`) and the in-flight agent cap."""

from __future__ import annotations

import asyncio

import pytest

from agentflow import AgentResult, Pipeline
from agentflow.agent import BaseAgent
from agentflow.exceptions import PipelineError


class MockLLM:
    async def generate(self, messages, **kwargs):
        return {"content": "ok", "tokens": 1, "duration": 0.0, "model": "mock-model"}


class SimpleAgent(BaseAgent):
    def __init__(self, name: str, role: str = "worker"):
        super().__init__(name=name, role=role)

    async def execute(self, task, context, llm):
        return AgentResult(agent=self.name, output=f"{self.name} done")


class TrackingAgent(BaseAgent):
    """Records the peak number of agents running at the same moment."""

    peak = 0
    live = 0

    def __init__(self, name: str):
        super().__init__(name=name, role="tracked")

    @classmethod
    def reset(cls):
        cls.peak = 0
        cls.live = 0

    async def execute(self, task, context, llm):
        TrackingAgent.live += 1
        TrackingAgent.peak = max(TrackingAgent.peak, TrackingAgent.live)
        try:
            await asyncio.sleep(0.02)
        finally:
            TrackingAgent.live -= 1
        return AgentResult(agent=self.name, output="done")


# ── explain() ─────────────────────────────────────────────────────────────────


def _diamond() -> Pipeline:
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SimpleAgent("fetcher", "Fetcher"))
    pipe.add(SimpleAgent("analyst_a", "Analyst A"), depends_on=["fetcher"])
    pipe.add(SimpleAgent("analyst_b", "Analyst B"), depends_on=["fetcher"])
    pipe.add(
        SimpleAgent("writer", "Writer"),
        depends_on=["analyst_a", "analyst_b"],
        timeout=30,
    )
    return pipe


def test_explain_reports_levels_and_dependencies():
    text = _diamond().explain()

    assert "4 agents, 3 levels" in text
    assert "Level 0 (1 agent):" in text
    assert "Level 1 (2 agents, run in parallel):" in text
    assert "after=[analyst_a, analyst_b]" in text
    assert "role=Writer" in text
    assert "timeout=30s" in text


def test_explain_flags_conditional_agents():
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SimpleAgent("classifier"))
    pipe.add(
        SimpleAgent("handler"),
        depends_on=["classifier"],
        condition=lambda ctx: True,
    )

    assert "conditional" in pipe.explain()


def test_explain_reports_the_concurrency_cap():
    pipe = _diamond()
    assert "max 2 concurrent" in pipe.explain()

    capped = Pipeline(llm=MockLLM(), max_concurrency=1)
    capped.add(SimpleAgent("a"))
    capped.add(SimpleAgent("b"))
    assert "max 1 concurrent" in capped.explain()


def test_explain_rejects_a_cycle_like_run_would():
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SimpleAgent("alpha"))
    pipe.add(SimpleAgent("beta"), depends_on=["alpha"])
    pipe._nodes[0].depends_on = ["beta"]

    with pytest.raises(PipelineError, match="Cycle"):
        pipe.explain()


def test_explain_rejects_an_unknown_dependency():
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SimpleAgent("writer"), depends_on=["ghost"])

    with pytest.raises(PipelineError, match="ghost"):
        pipe.explain()


def test_explain_on_an_empty_pipeline():
    assert "0 agents, 0 levels" in Pipeline(llm=MockLLM()).explain()


# ── max_concurrency ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrency_cap_limits_agents_in_flight():
    TrackingAgent.reset()
    pipe = Pipeline(llm=MockLLM(), max_concurrency=2)
    for i in range(6):
        pipe.add(TrackingAgent(f"agent{i}"))

    result = await pipe.run("task")

    assert len(result.results) == 6
    assert TrackingAgent.peak <= 2, f"{TrackingAgent.peak} agents ran at once, cap was 2"


@pytest.mark.asyncio
async def test_without_a_cap_the_whole_level_runs_at_once():
    TrackingAgent.reset()
    pipe = Pipeline(llm=MockLLM())
    for i in range(6):
        pipe.add(TrackingAgent(f"agent{i}"))

    await pipe.run("task")

    assert TrackingAgent.peak == 6


@pytest.mark.asyncio
async def test_a_cap_wider_than_the_level_changes_nothing():
    TrackingAgent.reset()
    pipe = Pipeline(llm=MockLLM(), max_concurrency=50)
    for i in range(4):
        pipe.add(TrackingAgent(f"agent{i}"))

    await pipe.run("task")

    assert TrackingAgent.peak == 4


@pytest.mark.asyncio
async def test_concurrent_runs_of_one_pipeline_each_get_their_own_budget():
    """The cap is per run, so serve()'s parallel runs do not contend."""
    TrackingAgent.reset()
    pipe = Pipeline(llm=MockLLM(), max_concurrency=1)
    pipe.add(TrackingAgent("a"))
    pipe.add(TrackingAgent("b"))

    await asyncio.gather(pipe.run("one"), pipe.run("two"))

    assert TrackingAgent.peak <= 2


def test_a_meaningless_cap_is_rejected_at_construction():
    with pytest.raises(PipelineError, match="at least 1"):
        Pipeline(llm=MockLLM(), max_concurrency=0)
