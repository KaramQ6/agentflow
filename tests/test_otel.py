"""The OpenTelemetry bridge, exercised against a real in-memory exporter.

README and docs advertise OTel integration, so the adapter is a documented
promise — it shipped with no test at all. These assert on spans the SDK
actually produced, not on calls to a mock.
"""

from __future__ import annotations

import pytest

from agentflow import AgentResult, Pipeline
from agentflow.agent import BaseAgent
from agentflow.exceptions import AgentError

pytest.importorskip("opentelemetry.sdk", reason="requires the 'otel' extra")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from agentflow.contrib.otel import OTelHooks  # noqa: E402


class MockLLM:
    async def generate(self, messages, **kwargs):
        return {"content": "ok", "tokens": 1, "duration": 0.0, "model": "mock-model"}


class SimpleAgent(BaseAgent):
    def __init__(self, name: str, tokens: int = 42, cost: float = 0.5):
        super().__init__(name=name, role="worker")
        self._tokens = tokens
        self._cost = cost

    async def execute(self, task, context, llm):
        return AgentResult(
            agent=self.name, output="done", tokens_used=self._tokens, cost=self._cost
        )


class ExplodingAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name=name, role="broken")

    async def execute(self, task, context, llm):
        raise AgentError(self.name, "kaboom")


@pytest.fixture
def exporter():
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    return memory, provider.get_tracer("test")


def _by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


@pytest.mark.asyncio
async def test_a_run_produces_a_pipeline_span_and_one_span_per_agent(exporter):
    memory, tracer = exporter
    pipe = Pipeline(llm=MockLLM(), hooks=OTelHooks(tracer=tracer))
    pipe.add(SimpleAgent("first"))
    pipe.add(SimpleAgent("second"), depends_on=["first"])

    await pipe.run("task")

    assert set(_by_name(memory)) == {"pipeline.run", "agent.first", "agent.second"}


@pytest.mark.asyncio
async def test_agent_spans_carry_cost_and_token_attributes(exporter):
    memory, tracer = exporter
    pipe = Pipeline(llm=MockLLM(), hooks=OTelHooks(tracer=tracer))
    pipe.add(SimpleAgent("solo", tokens=123, cost=0.25))

    await pipe.run("task")

    attributes = _by_name(memory)["agent.solo"].attributes
    assert attributes["agentflow.tokens"] == 123
    assert attributes["agentflow.cost_usd"] == 0.25
    assert attributes["agentflow.cached"] is False
    assert attributes["agentflow.level"] == 0


@pytest.mark.asyncio
async def test_pipeline_span_carries_run_totals(exporter):
    memory, tracer = exporter
    pipe = Pipeline(llm=MockLLM(), hooks=OTelHooks(tracer=tracer))
    pipe.add(SimpleAgent("a", tokens=10, cost=0.1))
    pipe.add(SimpleAgent("b", tokens=20, cost=0.2))

    await pipe.run("summarise the quarterly report")

    attributes = _by_name(memory)["pipeline.run"].attributes
    assert attributes["agentflow.total_tokens"] == 30
    assert attributes["agentflow.total_cost_usd"] == pytest.approx(0.3)
    assert attributes["agentflow.agent_count"] == 2
    assert attributes["agentflow.status"] == "completed"
    assert attributes["agentflow.task_preview"] == "summarise the quarterly report"


@pytest.mark.asyncio
async def test_agent_spans_are_children_of_the_pipeline_span(exporter):
    memory, tracer = exporter
    pipe = Pipeline(llm=MockLLM(), hooks=OTelHooks(tracer=tracer))
    pipe.add(SimpleAgent("child"))

    await pipe.run("task")

    spans = _by_name(memory)
    assert spans["agent.child"].parent.span_id == spans["pipeline.run"].context.span_id


@pytest.mark.asyncio
async def test_a_failing_agent_marks_its_span_as_errored(exporter):
    memory, tracer = exporter
    pipe = Pipeline(llm=MockLLM(), hooks=OTelHooks(tracer=tracer))
    pipe.add(ExplodingAgent("boom"))

    with pytest.raises(AgentError):
        await pipe.run("task")

    span = _by_name(memory)["agent.boom"]
    assert span.status.status_code is trace.StatusCode.ERROR
    assert "kaboom" in span.status.description
    assert span.events, "the exception should be recorded on the span"


@pytest.mark.asyncio
async def test_a_failed_run_does_not_leave_a_dangling_pipeline_span(exporter):
    """A run that raises never calls on_pipeline_end; the span must not appear."""
    memory, tracer = exporter
    pipe = Pipeline(llm=MockLLM(), hooks=OTelHooks(tracer=tracer))
    pipe.add(ExplodingAgent("boom"))

    with pytest.raises(AgentError):
        await pipe.run("task")

    assert "pipeline.run" not in _by_name(memory)


def test_hooks_default_to_the_global_tracer():
    hooks = OTelHooks()
    assert hooks._tracer is not None


def test_end_callbacks_ignore_agents_that_never_started():
    """safe_invoke swallows hook errors, so the hook must not rely on that."""
    hooks = OTelHooks()

    hooks.on_agent_end(AgentResult(agent="never-started", output=""))
    hooks.on_agent_error("never-started", RuntimeError("x"))
    hooks.on_pipeline_end(
        Pipeline(llm=MockLLM())._build_result(
            last_output="", results={}, run_id="r", levels_executed=0, wall_start=0.0
        )
    )
