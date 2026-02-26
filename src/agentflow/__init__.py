"""agentflow - Lightweight multi-agent AI pipeline framework."""

__version__ = "0.1.0"

from .agent import Agent, BaseAgent
from .llm import LLM
from .pipeline import Pipeline
from .types import AgentResult, PipelineResult, Event
from .events import EventEmitter
from .exceptions import AgentFlowError, AgentError, PipelineError, LLMError

__all__ = [
    "Agent",
    "BaseAgent",
    "LLM",
    "Pipeline",
    "AgentResult",
    "PipelineResult",
    "Event",
    "EventEmitter",
    "AgentFlowError",
    "AgentError",
    "PipelineError",
    "LLMError",
]
