"""Agent definition via decorators and base class."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from .exceptions import AgentError, AgentOutputValidationError, ToolError
from .llm import LLM
from .memory import BaseMemory
from .tools import Tool
from .types import AgentResult

DEFAULT_MAX_TOOL_ITERATIONS = 6
TOOL_OUTPUT_MAX_CHARS = 5000
MESSAGES_MAX_LENGTH = 20
LLM_RETRIES_PER_ITERATION = 1

_log = logging.getLogger("agentflow.agent")


def _truncate_output(output: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    """Truncate tool output if it exceeds *max_chars*, appending a truncation marker."""
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"...[Truncated: {len(output) - max_chars} chars]"


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
        tools: list[Tool] | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        memory: BaseMemory | None = None,
    ):
        self.name = name
        self.role = role
        self._prompt_fn = prompt_fn
        self._output_schema = output_schema
        self._tools = tools or []
        self._max_tool_iterations = max_tool_iterations
        self._memory = memory
        self._session_id = "default"
        # B5/B8: Pre-compute tool schemas once at construction time.
        self._openai_tools = [t.openai_schema for t in self._tools]

    def set_session(self, session_id: str) -> None:
        """Set the session ID used for memory load/save operations."""
        self._session_id = session_id

    async def execute(self, task: str, context: dict[str, str], llm: LLM) -> AgentResult:
        start = time.perf_counter()
        try:
            user_message = await self._prompt_fn(task, context)
        except Exception as e:
            raise AgentError(self.name, f"Prompt function failed: {e}") from e

        system_prompt = f"You are a {self.role}. Provide clear, thorough, well-structured responses."

        # M3: Inject prior session context from memory into the system prompt.
        if self._memory is not None:
            prev = await self._memory.load_context(self._session_id)
            if prev:
                parts = [f"{name}: {output[:300]}" for name, output in prev.items()]
                system_prompt += (
                    "\n\n[Memory — previous outputs from this session:\n"
                    + "\n".join(parts)
                    + "\n]"
                )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        metadata: dict[str, Any] = {}
        if self._tools:
            content, tokens_used, cost, model_name, trace = await self._run_tool_loop(messages, llm)
            metadata["model"] = model_name
            metadata["tool_calls"] = trace
            cached = False
        else:
            try:
                response = await llm.generate(messages)
            except Exception as e:
                raise AgentError(self.name, str(e)) from e
            content = response["content"]
            tokens_used = response["tokens"]
            cost = response.get("cost", 0.0)
            model_name = response["model"]
            cached = response.get("cached", False)
            metadata["model"] = model_name

        # M3: Persist this agent's output back to memory for downstream agents.
        if self._memory is not None and content:
            await self._memory.save_context(self._session_id, self.name, content)

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
            tokens_used=tokens_used,
            cost=round(cost, 6),
            duration=round(duration, 3),
            cached=cached,
            metadata=metadata,
        )

    async def _run_tool_loop(
        self, messages: list[dict[str, Any]], llm: LLM
    ) -> tuple[str, int, float, str, list[dict[str, Any]]]:
        """Drive a ReAct-style loop: call the LLM, run any requested tools, repeat.

        Features:
        - Multiple tool calls run concurrently via ``asyncio.gather`` (B1).
        - Duplicate tool calls are detected, skipped, and reported as errors (B3).
        - Tool outputs > 5000 chars are truncated (B4).
        - Message list is trimmed to prevent context overflow (B2).
        - Transient LLM errors are retried once per iteration (B7).

        Returns:
            (final_content, total_tokens, total_cost, model_name, tool_call_trace)
        """
        tool_map = {t.name: t for t in self._tools}
        total_tokens = 0
        total_cost = 0.0
        model_name = llm.model
        trace: list[dict[str, Any]] = []
        seen_calls: set[tuple[str, str]] = set()

        for iteration in range(self._max_tool_iterations):
            # B7: Per-iteration LLM retry for transient failures.
            for retry in range(LLM_RETRIES_PER_ITERATION + 1):
                try:
                    response = await llm.generate(messages, tools=self._openai_tools)
                    break
                except Exception as exc:
                    if retry == LLM_RETRIES_PER_ITERATION:
                        raise AgentError(
                            self.name,
                            f"LLM call failed after {LLM_RETRIES_PER_ITERATION + 1} iteration-level attempts",
                        ) from exc
                    _log.warning(
                        "LLM generation retry in agent %s (iteration %d, attempt %d)",
                        self.name, iteration + 1, retry + 1,
                    )
                    await asyncio.sleep(0.5 * (2 ** retry))

            total_tokens += response["tokens"]
            total_cost += response.get("cost", 0.0)
            model_name = response["model"]
            tool_calls = response["tool_calls"]

            if not tool_calls:
                return response["content"], total_tokens, total_cost, model_name, trace

            messages.append(
                {
                    "role": "assistant",
                    "content": response["content"] or None,
                    "tool_calls": tool_calls,
                }
            )

            # B3: Separate duplicate calls from new ones; duplicates get an
            # immediate error observation without being re-executed.
            results_map: dict[str, tuple[str, str, str, str]] = {}
            coros: list[tuple[str, asyncio.Task[tuple[str, str, str, str]]]] = []
            call_order: list[str] = []

            for call in tool_calls:
                fn = call["function"]
                name = fn["name"]
                arguments = fn["arguments"]
                key = (name, arguments)

                if key in seen_calls:
                    dup_msg = (
                        f"Error: Duplicate tool call detected. You already called "
                        f"'{name}' with these arguments. Try a different approach."
                    )
                    results_map[call["id"]] = (call["id"], name, arguments, dup_msg)
                    _log.warning(
                        "Duplicate tool call blocked in agent %s: %s(%s)",
                        self.name, name, arguments[:120],
                    )
                else:
                    seen_calls.add(key)
                    coros.append(
                        (call["id"], asyncio.create_task(self._execute_single_tool(call, tool_map)))
                    )
                call_order.append(call["id"])

            # B1: Await all unique tool calls concurrently.
            if coros:
                for cid, task in coros:
                    results_map[cid] = await task

            # Append results in the original tool_calls order.
            for call_id in call_order:
                _, name, arguments, output = results_map[call_id]
                # B4: Truncate oversized tool results.
                output = _truncate_output(output)
                trace.append({"tool": name, "arguments": arguments, "result": output})
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": output}
                )

            # B2: Sliding window — keep system + user, drop oldest tool pairs
            # when the message list grows too large.
            if len(messages) > MESSAGES_MAX_LENGTH:
                overflow = len(messages) - MESSAGES_MAX_LENGTH
                messages[2:2 + overflow] = []

        raise AgentError(
            self.name,
            f"exceeded max_tool_iterations={self._max_tool_iterations} without a final answer",
        )

    async def _execute_single_tool(
        self, call: dict[str, Any], tool_map: dict[str, Any]
    ) -> tuple[str, str, str, str]:
        """Execute a single tool call and return (call_id, name, arguments, output).

        B6: Execution is logged for observability.
        """
        fn = call["function"]
        name = fn["name"]
        arguments = fn["arguments"]
        target = tool_map.get(name)

        if target is None:
            output = f"Error: unknown tool '{name}'"
            _log.error("Unknown tool called in agent %s: %s", self.name, name)
        else:
            _log.info("Tool executing in agent %s: %s(%s)", self.name, name, arguments[:120])
            try:
                output = await target.acall(arguments)
                _log.info(
                    "Tool completed in agent %s: %s (output %d chars)",
                    self.name, name, len(output),
                )
            except ToolError as e:
                output = f"Error: {e}"
                _log.error("Tool error in agent %s: %s — %s", self.name, name, e)
            except Exception as e:
                output = f"Error: unexpected exception - {str(e)}"
                _log.error(
                    "Unexpected exception in agent %s tool %s: %s", self.name, name, e,
                )

        return call["id"], name, arguments, output

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
        tools: Optional list of ``Tool`` objects (see :func:`agentflow.tool`).
               When present the agent runs a ReAct loop, letting the model call
               tools and observe results until it produces a final answer.
        max_tool_iterations: Safety cap on tool-calling rounds (default 6).

    Usage:
        @Agent(name="researcher", role="Research Analyst")
        async def researcher(task: str, context: dict) -> str:
            return f"Research this topic: {task}"

        # With tools:
        @tool
        async def search(query: str) -> str:
            \"\"\"Search the web.\"\"\"
            ...

        @Agent(name="assistant", role="Assistant", tools=[search])
        async def assistant(task: str, context: dict) -> str:
            return task

        # With structured output:
        class Summary(BaseModel):
            title: str
            points: list[str]

        @Agent(name="summarizer", role="Summarizer", output_schema=Summary)
        async def summarizer(task: str, context: dict) -> str:
            return f"Summarize as JSON: {task}"
    """

    def __init__(
        self,
        name: str,
        role: str,
        output_schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        memory: BaseMemory | None = None,
    ):
        self.name = name
        self.role = role
        self._output_schema = output_schema
        self._tools = tools
        self._max_tool_iterations = max_tool_iterations
        self._memory = memory

    def __call__(self, fn: Callable[..., Awaitable[str]]) -> _DecoratorAgent:
        return _DecoratorAgent(
            self.name,
            self.role,
            fn,
            output_schema=self._output_schema,
            tools=self._tools,
            max_tool_iterations=self._max_tool_iterations,
            memory=self._memory,
        )
