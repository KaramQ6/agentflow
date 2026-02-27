"""Agent definition via decorators and base class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from .exceptions import AgentError, AgentOutputValidationError
from .llm import LLM
from .types import AgentResult


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

    def __init__(
        self,
        name: str,
        role: str,
        prompt_fn: Callable[..., Awaitable[str]],
        output_schema: type[BaseModel] | None = None,
    ):
        self.name = name
        self.role = role
        self._prompt_fn = prompt_fn
        self._output_schema = output_schema

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

        content = response["content"]
        metadata: dict[str, Any] = {"model": response["model"]}

        if self._output_schema is not None:
            try:
                validated = self._output_schema.model_validate_json(content)
                metadata["validated_output"] = validated.model_dump()
            except ValidationError as e:
                raise AgentOutputValidationError(self.name, str(e)) from e

        duration = time.perf_counter() - start
        return AgentResult(
            agent=self.name,
            output=content,
            tokens_used=response["tokens"],
            duration=round(duration, 3),
            cached=response.get("cached", False),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r})"


class Agent:
    """Decorator to define an agent from an async function.

    The decorated function receives (task, context) and returns
    the user message to send to the LLM.

    Args:
        name: Unique identifier for the agent within a pipeline.
        role: Describes the agent's persona (used as system prompt context).
        output_schema: Optional Pydantic BaseModel subclass. If provided, the
                       LLM response must be valid JSON matching the schema, or
                       AgentOutputValidationError is raised.

    Usage:
        @Agent(name="researcher", role="Research Analyst")
        async def researcher(task: str, context: dict) -> str:
            return f"Research this topic: {task}"

        # With structured output:
        class Summary(BaseModel):
            title: str
            points: list[str]

        @Agent(name="summarizer", role="Summarizer", output_schema=Summary)
        async def summarizer(task: str, context: dict) -> str:
            return f"Summarize as JSON: {task}"
    """

    def __init__(self, name: str, role: str, output_schema: type[BaseModel] | None = None):
        self.name = name
        self.role = role
        self._output_schema = output_schema

    def __call__(self, fn: Callable[..., Awaitable[str]]) -> _DecoratorAgent:
        return _DecoratorAgent(self.name, self.role, fn, output_schema=self._output_schema)
