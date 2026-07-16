"""Tests for conditional branching in the pipeline."""

import pytest

from agentflow import AgentResult, Pipeline
from agentflow.agent import BaseAgent


class MockLLM:
    async def generate(self, messages, **kwargs):
        return {"content": "ok", "tokens": 5, "duration": 0.01, "model": "mock", "cached": False}


class OutputAgent(BaseAgent):
    """Returns a fixed output string."""

    def __init__(self, name: str, output: str):
        super().__init__(name=name, role="output")
        self._output = output

    async def execute(self, task, context, llm):
        return AgentResult(agent=self.name, output=self._output, tokens_used=0, duration=0.0)


# ─── Conditional Execution ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_conditional_agent_skipped_when_false():
    """An agent whose condition returns False must not appear in results."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(OutputAgent("classifier", "normal"))
    pipe.add(
        OutputAgent("urgent_handler", "URGENT"),
        depends_on=["classifier"],
        condition=lambda ctx: "urgent" in ctx.get("classifier", "").lower(),
    )
    pipe.add(
        OutputAgent("normal_handler", "NORMAL"),
        depends_on=["classifier"],
        condition=lambda ctx: "urgent" not in ctx.get("classifier", "").lower(),
    )

    result = await pipe.run("task")

    assert "classifier" in result.results
    assert "urgent_handler" not in result.results
    assert "normal_handler" in result.results
    assert result.results["normal_handler"].output == "NORMAL"


@pytest.mark.asyncio
async def test_conditional_agent_runs_when_true():
    """An agent whose condition returns True must run normally."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(OutputAgent("classifier", "this is urgent!"))
    pipe.add(
        OutputAgent("urgent_handler", "ESCALATED"),
        depends_on=["classifier"],
        condition=lambda ctx: "urgent" in ctx.get("classifier", "").lower(),
    )

    result = await pipe.run("task")
    assert "urgent_handler" in result.results
    assert result.results["urgent_handler"].output == "ESCALATED"


@pytest.mark.asyncio
async def test_no_condition_means_always_run():
    """An agent with no condition must always execute."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(OutputAgent("always", "yes"))

    result = await pipe.run("task")
    assert "always" in result.results


@pytest.mark.asyncio
async def test_conditional_skipped_agent_not_in_context():
    """Skipped agents must not contribute keys to downstream context."""

    received: list[dict] = []

    class ContextSpy(BaseAgent):
        async def execute(self, task, context, llm):
            received.append(dict(context))
            return AgentResult(agent=self.name, output="spy", tokens_used=0, duration=0.0)

    pipe = Pipeline(llm=MockLLM())
    pipe.add(OutputAgent("root", "root_output"))
    pipe.add(
        OutputAgent("skipped", "never"),
        depends_on=["root"],
        condition=lambda ctx: False,
    )
    pipe.add(ContextSpy("spy", "Spy"), depends_on=["root"])

    await pipe.run("task")

    assert "skipped" not in received[0]


# ─── Conditional Streaming Events ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_emits_agent_skipped_event():
    """Streaming must emit agent_skipped for conditionally skipped agents."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(OutputAgent("root", "output"))
    pipe.add(
        OutputAgent("skipped", "never"),
        depends_on=["root"],
        condition=lambda ctx: False,
    )

    event_types = []
    skipped_agents = []

    async for event in pipe.stream("task"):
        event_types.append(event.type)
        if event.type == "agent_skipped":
            skipped_agents.append(event.agent)

    assert "agent_skipped" in event_types
    assert "skipped" in skipped_agents
    assert "pipeline_complete" in event_types
