"""Core data models for agentflow."""

from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "agent_start",
    "agent_complete",
    "agent_error",
    "agent_skipped",
    "pipeline_complete",
    "pipeline_error",
    "pipeline_paused",
]
"""Every event type the pipeline emits (see :class:`Event`)."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


class LLMResponse(BaseModel):
    """Typed result of :meth:`agentflow.LLM.generate`.

    Supports dict-style access (``response["content"]``, ``response.get(...)``)
    as a migration shim for code written against the pre-0.6 dict return type.
    Prefer attribute access; the dict-style shim is deprecated.
    """

    content: str = ""
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    duration: float = 0.0
    model: str = ""
    cached: bool = False
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None

    def __getitem__(self, key: str) -> Any:
        warnings.warn(
            "Dict-style access on LLMResponse is deprecated and will be "
            "removed in 1.0; use attribute access (response.content) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        warnings.warn(
            "Dict-style access on LLMResponse is deprecated and will be "
            "removed in 1.0; use attribute access (response.content) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key, default)


class AgentResult(BaseModel):
    """Result from a single agent execution."""

    agent: str
    output: str
    tokens_used: int = 0
    cost: float = 0.0
    duration: float = 0.0
    cached: bool = False
    level: int = 0
    timestamp: str = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] | None = None
    """Validated structured output (``output_schema.model_dump()``), if any.

    When set, downstream agents receive this dict as the context value
    instead of the raw ``output`` string.
    """


class PipelineResult(BaseModel):
    """Result from a full pipeline execution.

    When the pipeline is paused by the HITL mechanism, ``status`` is set to
    ``"paused"`` and ``pause_info`` contains the details of the action
    awaiting human review.
    """

    output: str  # Last agent's output
    results: dict[str, AgentResult] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration: float = 0.0
    """Sum of per-agent durations. Deprecated name: parallel agents overlap,
    so this is CPU-style "agent seconds", not elapsed time. Prefer
    :attr:`agent_seconds` for this value and :attr:`wall_time` for elapsed."""
    wall_time: float = 0.0
    run_id: str = Field(default_factory=_short_uuid)
    levels_executed: int = 0
    agents_with_cache_hits: int = 0
    status: Literal["completed", "paused"] = "completed"
    pause_info: dict[str, Any] | None = None

    @property
    def agent_seconds(self) -> float:
        """Sum of per-agent execution durations (parallel agents overlap)."""
        return self.total_duration

    def get(self, agent_name: str) -> AgentResult | None:
        """Get a specific agent's result."""
        return self.results.get(agent_name)


class Event(BaseModel):
    """Pipeline event for streaming.

    Event types: agent_start, agent_complete, agent_error, agent_skipped,
    pipeline_complete, pipeline_error, pipeline_paused (see :data:`EventType`).
    """

    type: EventType
    agent: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utc_now)
