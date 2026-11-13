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

from .._scope import current_scope
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
        # Keyed by run: one hooks object serves concurrent runs (serve()).
        self._pipeline_spans: dict[str, Any] = {}
        self._agent_spans: dict[tuple[str, str], Any] = {}

    @staticmethod
    def _run_id() -> str:
        scope = current_scope()
        return scope.run_id if scope is not None else ""

    def on_pipeline_start(self, task: str, run_id: str, agent_count: int) -> None:
        self._pipeline_spans[run_id] = self._tracer.start_span(
            "pipeline.run",
            attributes={
                "agentflow.run_id": run_id,
                "agentflow.agent_count": agent_count,
                "agentflow.task_preview": task[:200],
            },
        )

    def on_agent_start(self, agent: str, level: int) -> None:
        run_id = self._run_id()
        pipeline_span = self._pipeline_spans.get(run_id)
        parent_ctx = trace.set_span_in_context(pipeline_span) if pipeline_span is not None else None
        self._agent_spans[(run_id, agent)] = self._tracer.start_span(
            f"agent.{agent}",
            context=parent_ctx,
            attributes={"agentflow.agent": agent, "agentflow.level": level},
        )

    def on_agent_end(self, result: AgentResult) -> None:
        span = self._agent_spans.pop((self._run_id(), result.agent), None)
        if span is None:
            return
        span.set_attribute("agentflow.tokens", result.tokens_used)
        span.set_attribute("agentflow.cost_usd", result.cost)
        span.set_attribute("agentflow.cached", result.cached)
        span.set_attribute("agentflow.duration_s", result.duration)
        span.end()

    def on_agent_error(self, agent: str, error: Exception) -> None:
        span = self._agent_spans.pop((self._run_id(), agent), None)
        if span is None:
            return
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()

    def on_pipeline_end(self, result: PipelineResult) -> None:
        span = self._pipeline_spans.pop(result.run_id, None)
        if span is None:
            return
        span.set_attribute("agentflow.total_tokens", result.total_tokens)
        span.set_attribute("agentflow.total_cost_usd", result.total_cost)
        span.set_attribute("agentflow.wall_time_s", result.wall_time)
        span.set_attribute("agentflow.status", result.status)
        span.set_attribute("agentflow.tool_errors", len(result.tool_errors))
        span.end()

    def on_pipeline_error(self, error: BaseException, run_id: str) -> None:
        span = self._pipeline_spans.pop(run_id, None)
        if span is None:
            return
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error) or type(error).__name__))
        span.end()
