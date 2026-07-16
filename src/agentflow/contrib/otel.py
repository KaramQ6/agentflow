"""OpenTelemetry bridge for pipeline observability.

Requires the ``opentelemetry-api`` package (and an SDK/exporter of your
choice)::

    pip install opentelemetry-api opentelemetry-sdk

Usage::

    from agentflow import Pipeline
    from agentflow.contrib.otel import OTelHooks

    pipe = Pipeline(llm=llm, hooks=OTelHooks())

Every pipeline run becomes a root span; every agent execution becomes a
child span carrying ``agentflow.tokens``, ``agentflow.cost_usd``,
``agentflow.cached``, and ``agentflow.level`` attributes. Errors are
recorded on the agent span and mark it as failed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError as _exc:  # pragma: no cover - import guard
    raise ImportError(
        "OTelHooks requires the 'opentelemetry-api' package: "
        "pip install opentelemetry-api opentelemetry-sdk"
    ) from _exc

from ..observability import Hooks

if TYPE_CHECKING:
    from ..types import AgentResult, PipelineResult


class OTelHooks(Hooks):
    """Hooks implementation that emits OpenTelemetry spans.

    Args:
        tracer: An existing tracer to use. Defaults to
                ``trace.get_tracer("agentflow")`` (configure your provider
                and exporter globally as usual).
    """

    def __init__(self, tracer: Any = None):
        self._tracer = tracer or trace.get_tracer("agentflow")
        self._pipeline_span: Any = None
        self._agent_spans: dict[str, Any] = {}

    def on_pipeline_start(self, task: str, run_id: str, agent_count: int) -> None:
        self._pipeline_span = self._tracer.start_span(
            "pipeline.run",
            attributes={
                "agentflow.run_id": run_id,
                "agentflow.agent_count": agent_count,
                "agentflow.task_preview": task[:200],
            },
        )

    def on_agent_start(self, agent: str, level: int) -> None:
        parent_ctx = (
            trace.set_span_in_context(self._pipeline_span)
            if self._pipeline_span is not None
            else None
        )
        self._agent_spans[agent] = self._tracer.start_span(
            f"agent.{agent}",
            context=parent_ctx,
            attributes={"agentflow.agent": agent, "agentflow.level": level},
        )

    def on_agent_end(self, result: AgentResult) -> None:
        span = self._agent_spans.pop(result.agent, None)
        if span is None:
            return
        span.set_attribute("agentflow.tokens", result.tokens_used)
        span.set_attribute("agentflow.cost_usd", result.cost)
        span.set_attribute("agentflow.cached", result.cached)
        span.set_attribute("agentflow.duration_s", result.duration)
        span.end()

    def on_agent_error(self, agent: str, error: Exception) -> None:
        span = self._agent_spans.pop(agent, None)
        if span is None:
            return
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()

    def on_pipeline_end(self, result: PipelineResult) -> None:
        if self._pipeline_span is None:
            return
        self._pipeline_span.set_attribute("agentflow.total_tokens", result.total_tokens)
        self._pipeline_span.set_attribute("agentflow.total_cost_usd", result.total_cost)
        self._pipeline_span.set_attribute("agentflow.wall_time_s", result.wall_time)
        self._pipeline_span.set_attribute("agentflow.status", result.status)
        self._pipeline_span.end()
        self._pipeline_span = None
