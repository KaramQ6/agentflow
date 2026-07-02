"""Observability hooks for pipeline execution.

``Pipeline.run()`` is otherwise silent (only ``stream()`` emits events). Pass a
:class:`Hooks` instance to observe the full lifecycle — wire it to logging,
metrics, OpenTelemetry, Langfuse, or anything else. All hook methods default to
no-ops, so you only override what you need, and a hook that raises is caught and
warned rather than crashing the pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .logging import PipelineLogger

if TYPE_CHECKING:
    from .types import AgentResult, PipelineResult

_internal_log = logging.getLogger("agentflow.hooks")


class Hooks:
    """Lifecycle callbacks for a pipeline run. Subclass and override any subset."""

    def on_pipeline_start(self, task: str, run_id: str, agent_count: int) -> None:
        """Called once before any agent runs."""

    def on_agent_start(self, agent: str, level: int) -> None:
        """Called just before an agent executes."""

    def on_agent_end(self, result: AgentResult) -> None:
        """Called after an agent completes successfully."""

    def on_agent_error(self, agent: str, error: Exception) -> None:
        """Called when an agent raises (before the error propagates)."""

    def on_pipeline_end(self, result: PipelineResult) -> None:
        """Called once after the pipeline finishes successfully."""


class LoggingHooks(Hooks):
    """Hooks that emit structured JSON logs via :class:`PipelineLogger`.

    Args:
        pipeline_name: Identifier included in every log record.
        level: Logging level (default INFO).
    """

    def __init__(self, pipeline_name: str = "pipeline", level: int = logging.INFO):
        self._pipeline_name = pipeline_name
        self._level = level
        self._log: PipelineLogger | None = None

    def on_pipeline_start(self, task: str, run_id: str, agent_count: int) -> None:
        self._log = PipelineLogger(self._pipeline_name, run_id=run_id, level=self._level)
        self._log.log_pipeline_start(task, agent_count=agent_count, level_count=0)

    def on_agent_start(self, agent: str, level: int) -> None:
        if self._log is not None:
            self._log.log_agent_start(agent, level)

    def on_agent_end(self, result: AgentResult) -> None:
        if self._log is not None:
            self._log.log_agent_complete(
                result.agent, result.tokens_used, result.duration, result.cached
            )

    def on_agent_error(self, agent: str, error: Exception) -> None:
        if self._log is not None:
            self._log.log_agent_error(agent, str(error))

    def on_pipeline_end(self, result: PipelineResult) -> None:
        if self._log is not None:
            self._log.log_pipeline_complete(
                result.run_id, result.total_tokens, result.total_duration
            )


def safe_invoke(hook: object, method: str, *args: object) -> None:
    """Call ``hook.method(*args)``, swallowing and warning on any exception.

    Observability must never take down a pipeline.
    """
    fn = getattr(hook, method, None)
    if fn is None:
        return
    try:
        fn(*args)
    except Exception as e:  # pragma: no cover - defensive
        _internal_log.warning("hook %s raised: %s", method, e)
