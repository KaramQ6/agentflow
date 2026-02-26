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


class PipelineError(AgentFlowError):
    """Raised when pipeline orchestration fails."""
