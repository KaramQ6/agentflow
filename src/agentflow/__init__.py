"""agentflow - Lightweight multi-agent AI pipeline framework."""

__version__ = "0.3.0"

from .agent import Agent, BaseAgent
from .cache import InMemoryCache, RedisCache, ResponseCache
from .events import EventEmitter
from .exceptions import (
    AgentError,
    AgentFlowError,
    AgentOutputValidationError,
    AgentTimeoutError,
    LLMError,
    PipelineError,
    ToolError,
)
from .llm import LLM
from .logging import PipelineLogger, get_logger
from .memory import BaseMemory, InMemoryContext, RedisContext, VectorContext
from .observability import Hooks, LoggingHooks
from .pipeline import Pipeline
from .pricing import estimate_cost, register_price
from .rate_limiter import RateLimiter
from .tools import Tool, tool
from .types import AgentResult, Event, PipelineResult

__all__ = [
    # Core
    "Agent",
    "BaseAgent",
    "LLM",
    "Pipeline",
    # Tools
    "Tool",
    "tool",
    # Cost
    "estimate_cost",
    "register_price",
    # Data models
    "AgentResult",
    "PipelineResult",
    "Event",
    "EventEmitter",
    # Memory
    "BaseMemory",
    "InMemoryContext",
    "RedisContext",
    "VectorContext",
    # Rate limiting
    "RateLimiter",
    # Caching
    "ResponseCache",
    "InMemoryCache",
    "RedisCache",
    # Logging & observability
    "PipelineLogger",
    "get_logger",
    "Hooks",
    "LoggingHooks",
    # Exceptions
    "AgentFlowError",
    "AgentError",
    "AgentTimeoutError",
    "AgentOutputValidationError",
    "PipelineError",
    "LLMError",
    "ToolError",
]
