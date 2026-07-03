"""Human-in-the-Loop (HITL) approval mechanism for agentflow pipelines.

An ``ApprovalPolicy`` is attached to a :class:`~agentflow.pipeline.Pipeline`
to intercept tool calls that require human sign-off. When the policy blocks a
tool, a :class:`PauseExecution` exception carries the full agent loop state so
the pipeline can serialize it to memory and return control to the caller.

On resume, the pipeline restores the saved state, either executes the pending
tool (approved) or injects human feedback (rejected), then continues the ReAct
loop and the rest of the DAG.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .exceptions import AgentFlowError


class PauseExecution(AgentFlowError):
    """Raised inside the ReAct tool loop when a tool call needs human approval.

    Carries the full conversation state so that execution can be paused at
    the pipeline level, serialized to memory, and resumed later with an
    approve/reject decision.

    Attributes:
        agent_name: The agent whose tool invocation triggered the pause.
        tool_name: Name of the tool that requires approval.
        tool_arguments: JSON-encoded arguments for the pending tool call.
        tool_call_id: OpenAI ``tool_call.id`` for the pending call.
        messages: Complete message list (system, user, assistant with tool_calls).
        total_tokens: Tokens accumulated so far in this ReAct loop.
        total_cost: Cost accumulated so far in this ReAct loop.
        model_name: Model used for the LLM calls.
        trace: Tool call trace built up before the pause.
        pending_calls: All remaining tool calls from the current LLM response
                       (including the paused one) that have not been executed.
        seen_calls: Set of ``(tool_name, arguments)`` already executed this loop,
                    serialized as a list of ``[name, args]`` pairs.
        iterations_used: How many ReAct loop iterations have been consumed.
    """

    def __init__(
        self,
        agent_name: str,
        tool_name: str,
        tool_arguments: str,
        tool_call_id: str,
        messages: list[dict[str, Any]],
        total_tokens: int,
        total_cost: float,
        model_name: str,
        trace: list[dict[str, Any]],
        pending_calls: list[dict[str, Any]],
        seen_calls: list[list[str]],
        iterations_used: int,
    ):
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        self.tool_call_id = tool_call_id
        self.messages = messages
        self.total_tokens = total_tokens
        self.total_cost = total_cost
        self.model_name = model_name
        self.trace = trace
        self.pending_calls = pending_calls
        self.seen_calls = seen_calls
        self.iterations_used = iterations_used
        super().__init__(
            f"Agent '{agent_name}' requires human approval to call "
            f"'{tool_name}' with arguments: {tool_arguments}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the pause state to a JSON-compatible dict."""
        return {
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_call_id": self.tool_call_id,
            "messages": self.messages,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "model_name": self.model_name,
            "trace": self.trace,
            "pending_calls": self.pending_calls,
            "seen_calls": self.seen_calls,
            "iterations_used": self.iterations_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PauseExecution:
        """Reconstruct a PauseExecution from its ``as_dict()`` representation."""
        return cls(
            agent_name=data["agent_name"],
            tool_name=data["tool_name"],
            tool_arguments=data["tool_arguments"],
            tool_call_id=data["tool_call_id"],
            messages=data["messages"],
            total_tokens=data["total_tokens"],
            total_cost=data["total_cost"],
            model_name=data["model_name"],
            trace=data["trace"],
            pending_calls=data["pending_calls"],
            seen_calls=data["seen_calls"],
            iterations_used=data["iterations_used"],
        )


class ApprovalPolicy:
    """Configurable policy that determines which tool calls require a human in the loop.

    Three matching strategies, evaluated in order (first match wins):

    1. **blocked_tools** – exact tool-name match.
    2. **custom_rule** – a ``(tool_name, arguments) -> bool`` callable.
    3. **allowed_tools** – if set, any tool NOT in the allow-list is blocked.

    Args:
        blocked_tools: Tool names that always require human approval.
        allowed_tools: Exclusive allow-list.  When provided, only tools in this
                       list may execute without approval.
        custom_rule: Optional callable that receives the tool name and its
                     JSON arguments string and returns ``True`` if the call
                     should be paused for review.
    """

    def __init__(
        self,
        blocked_tools: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        custom_rule: Callable[[str, str], bool] | None = None,
    ):
        self._blocked: set[str] = set(blocked_tools or [])
        self._allowed: set[str] | None = set(allowed_tools) if allowed_tools is not None else None
        self._custom: Callable[[str, str], bool] | None = custom_rule

    def requires_approval(self, tool_name: str, arguments: str) -> bool:
        """Return ``True`` if calling *tool_name* with *arguments* must be reviewed by a human.

        Args:
            tool_name: The name of the tool the LLM wants to invoke.
            arguments: The JSON-encoded arguments string for the invocation.

        Returns:
            ``True`` if the pipeline should pause for human approval.
        """
        if self._blocked and tool_name in self._blocked:
            return True
        if self._custom is not None and self._custom(tool_name, arguments):
            return True
        return self._allowed is not None and tool_name not in self._allowed
