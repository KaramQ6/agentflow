"""agentflow - Lightweight multi-agent AI pipeline framework.

The core public surface is deliberately narrow: two decorators (``@Agent``,
``@tool``), a ``Pipeline``, and an ``LLM``. Peripheral capabilities live in
opt-in submodules and are NOT re-exported here:

- ``agentflow.sandbox``   — Docker/subprocess code-execution sandboxes
- ``agentflow.triggers``  — event-driven daemon triggers (MQTT)
- ``agentflow.distillation`` — background memory compression

See PUBLIC_API.md for the stability contract.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("agentflowkit")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0.dev0"

from .agent import Agent, AgentSpec, BaseAgent
from .cache import InMemoryCache, RedisCache, ResponseCache
from .events import EventEmitter
from .exceptions import (
    AgentError,
    AgentFlowError,
    AgentOutputValidationError,
    AgentTimeoutError,
    BudgetExceededError,
    LLMError,
    PipelineError,
    ToolError,
)
from .hitl import ApprovalPolicy, PauseExecution
from .llm import LLM
from .logging import PipelineLogger, get_logger
from .memory import BaseMemory, InMemoryContext, RedisContext, VectorContext
from .observability import Hooks, LoggingHooks
from .pipeline import Pipeline
from .pricing import estimate_cost, register_price
from .rate_limiter import RateLimiter
from .swarm import SupervisorAgent
from .tools import Tool, tool
from .types import AgentResult, Event, LLMResponse, PipelineResult

__all__ = [
    # Core
    "Agent",
    "AgentSpec",
    "BaseAgent",
    "LLM",
    "Pipeline",
    "SupervisorAgent",
    # Tools
    "Tool",
    "tool",
    # Cost
    "estimate_cost",
    "register_price",
    # Data models
    "AgentResult",
    "PipelineResult",
    "LLMResponse",
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
    "BudgetExceededError",
    "PipelineError",
    "LLMError",
    "ToolError",
    # HITL
    "ApprovalPolicy",
    "PauseExecution",
]
