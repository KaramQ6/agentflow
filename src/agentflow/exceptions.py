"""agentflow - Lightweight multi-agent AI pipeline framework."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import PipelineResult


class AgentFlowError(Exception):
    """Base exception for agentflow."""


class LLMError(AgentFlowError):
    """Raised when an LLM call fails."""


class AgentError(AgentFlowError):
    """Raised when an agent execution fails."""

    def __init__(
        self,
        agent_name: str,
        message: str,
        *,
        tokens: int = 0,
        cost: float = 0.0,
        trace: list[dict[str, Any]] | None = None,
    ):
        self.agent_name = agent_name
        self.tokens = tokens  # usage of the failed agent, when known
        self.cost = cost
        self.trace = trace or []  # tool calls it made before failing
        super().__init__(f"Agent '{agent_name}' failed: {message}")


class EffectUncertainError(AgentError):
    """An earlier attempt started a ``write`` tool call and never finished it.

    The library cannot tell whether the effect happened, so it refuses to guess:
    the pipeline does not retry this error, and the call does not run again.
    Check the outside world, then resume or re-run on purpose.
    """

    def __init__(self, agent_name: str, tool_name: str, key: str):
        self.tool_name = tool_name
        self.key = key
        super().__init__(
            agent_name,
            f"tool '{tool_name}' was started earlier in this run and never finished; the "
            f"effect may or may not have happened. Check it before running it again "
            f"(effect key {key}).",
        )


class AgentTimeoutError(AgentError):
    """Raised when an agent exceeds its configured timeout."""

    def __init__(self, agent_name: str, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(agent_name, f"timed out after {timeout_seconds}s")


class AgentOutputValidationError(AgentError):
    """Raised when an agent's output fails Pydantic schema validation."""

    def __init__(self, agent_name: str, validation_errors: str):
        self.validation_errors = validation_errors
        super().__init__(agent_name, f"output validation failed: {validation_errors}")


class ToolError(AgentFlowError):
    """Raised when a tool's arguments are invalid or its execution fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class PipelineError(AgentFlowError):
    """Raised when pipeline orchestration fails."""


class BudgetExceededError(PipelineError):
    """Raised when a pipeline run's accumulated cost exceeds ``budget_usd``.

    Every LLM call reserves its worst-case cost before it is sent; this is
    raised when a reservation cannot fit even after in-flight calls settle.
    ``spent_usd`` includes attempts whose result was discarded. See ADR 0005.

    Attributes:
        partial_result: The ``PipelineResult`` for the levels that did
            complete, or ``None`` when the error was raised outside a run. The
            work it describes has already been paid for — abandoning it is the
            one thing a cost ceiling should not force you to do. Read
            ``.results`` for the agents that finished, or resume from them.
    """

    def __init__(
        self,
        budget_usd: float,
        spent_usd: float,
        partial_result: PipelineResult | None = None,
    ):
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self.partial_result = partial_result
        super().__init__(
            f"Pipeline budget exceeded: spent ${spent_usd:.6f} of ${budget_usd:.6f} budget"
        )
