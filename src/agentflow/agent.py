"""Agent definition via decorators and base class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable

from .llm import LLM
from .types import AgentResult
from .exceptions import AgentError


class BaseAgent(ABC):
    """Base class for agents that need full control.

    Subclass this for complex agents with custom logic.
    """

    name: str
    role: str

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @abstractmethod
    async def execute(self, task: str, context: dict[str, str], llm: LLM) -> AgentResult:
        """Execute the agent's task.

        Args:
            task: The task/topic string.
            context: Dict mapping agent_name -> output from previous agents.
            llm: The LLM provider to use.

        Returns:
            AgentResult with the agent's output.
        """
        ...


class _DecoratorAgent:
    """Agent created via the @Agent decorator."""

    def __init__(self, name: str, role: str, prompt_fn: Callable[..., Awaitable[str]]):
        self.name = name
        self.role = role
        self._prompt_fn = prompt_fn

    async def execute(self, task: str, context: dict[str, str], llm: LLM) -> AgentResult:
        start = time.perf_counter()
        try:
            user_message = await self._prompt_fn(task, context)
        except Exception as e:
            raise AgentError(self.name, f"Prompt function failed: {e}") from e

        system_prompt = f"You are a {self.role}. Provide clear, thorough, well-structured responses."

        try:
            response = await llm.generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ])
        except Exception as e:
            raise AgentError(self.name, str(e)) from e

        duration = time.perf_counter() - start
        return AgentResult(
            agent=self.name,
            output=response["content"],
            tokens_used=response["tokens"],
            duration=round(duration, 3),
            metadata={"model": response["model"]},
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r})"


class Agent:
    """Decorator to define an agent from an async function.

    The decorated function receives (task, context) and returns
    the user message to send to the LLM.

    Usage:
        @Agent(name="researcher", role="Research Analyst")
        async def researcher(task: str, context: dict) -> str:
            return f"Research this topic: {task}"
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def __call__(self, fn: Callable[..., Awaitable[str]]) -> _DecoratorAgent:
        return _DecoratorAgent(self.name, self.role, fn)
