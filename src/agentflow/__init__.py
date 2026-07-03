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
from .hitl import ApprovalPolicy, PauseExecution
from .llm import LLM
from .logging import PipelineLogger, get_logger
from .memory import BaseMemory, InMemoryContext, RedisContext, VectorContext
from .observability import Hooks, LoggingHooks
from .pipeline import Pipeline
from .pricing import estimate_cost, register_price
from .rate_limiter import RateLimiter
from .sandbox import (
    DockerSandbox,
    SandboxError,
    SandboxTimeoutError,
    SubprocessSandbox,
    create_sandbox,
    execute_code,
    sandboxed_tool,
)
from .swarm import SupervisorAgent
from .tools import Tool, tool
from .triggers import BaseTrigger, MQTTTrigger
from .types import AgentResult, Event, PipelineResult

__all__ = [
    # Core
    "Agent",
    "BaseAgent",
    "LLM",
    "Pipeline",
    "SupervisorAgent",
    # Tools
    "Tool",
    "tool",
    # Sandbox
    "DockerSandbox",
    "SubprocessSandbox",
    "SandboxError",
    "SandboxTimeoutError",
    "create_sandbox",
    "execute_code",
    "sandboxed_tool",
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
    # Triggers
    "BaseTrigger",
    "MQTTTrigger",
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
    # HITL
    "ApprovalPolicy",
    "PauseExecution",
]
