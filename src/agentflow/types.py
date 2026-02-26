"""Core data models for agentflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Result from a single agent execution."""

    agent: str
    output: str
    tokens_used: int = 0
    duration: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Result from a full pipeline execution."""

    output: str  # Last agent's output
    results: dict[str, AgentResult] = Field(default_factory=dict)
    total_tokens: int = 0
    total_duration: float = 0.0

    def get(self, agent_name: str) -> AgentResult | None:
        """Get a specific agent's result."""
        return self.results.get(agent_name)


class Event(BaseModel):
    """Pipeline event for streaming."""

    type: str  # agent_start, agent_complete, agent_error, pipeline_complete
    agent: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
