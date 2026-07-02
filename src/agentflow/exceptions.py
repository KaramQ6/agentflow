"""agentflow - Lightweight multi-agent AI pipeline framework."""


class AgentFlowError(Exception):
    """Base exception for agentflow."""


class LLMError(AgentFlowError):
    """Raised when an LLM call fails."""


class AgentError(AgentFlowError):
    """Raised when an agent execution fails."""

    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}' failed: {message}")


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
