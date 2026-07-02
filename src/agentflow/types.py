"""Core data models for agentflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


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


class PipelineResult(BaseModel):
    """Result from a full pipeline execution."""

    output: str  # Last agent's output
    results: dict[str, AgentResult] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration: float = 0.0
    run_id: str = Field(default_factory=_short_uuid)
    levels_executed: int = 0
    agents_with_cache_hits: int = 0

    def get(self, agent_name: str) -> AgentResult | None:
        """Get a specific agent's result."""
        return self.results.get(agent_name)


class Event(BaseModel):
    """Pipeline event for streaming.

    Event types: agent_start, agent_complete, agent_error,
                 agent_skipped, pipeline_complete, pipeline_error.
    """

    type: str
    agent: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utc_now)
