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
from .hitl import ApprovalPolicy, PauseExecution
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
    async def execute(self, task: str, context: dict[str, Any], llm: LLM) -> AgentResult:
        """Execute the agent's task.

        Args:
            task: The task/topic string.
            context: Dict mapping agent_name -> output from previous agents.
                     Values are strings, or dicts when the upstream agent
                     declared an ``output_schema`` (its validated output).
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
        self._approval_policy: ApprovalPolicy | None = None
        # B5/B8: Pre-compute tool schemas once at construction time.
        self._openai_tools = [t.openai_schema for t in self._tools]

    def set_session(self, session_id: str) -> None:
        """Set the default session ID used for memory load/save operations.

        Deprecated: prefer passing ``session_id`` to :meth:`execute`. Mutating
        a shared agent instance is unsafe under concurrent pipeline runs.
        """
        self._session_id = session_id

    def set_approval_policy(self, policy: ApprovalPolicy | None) -> None:
        """Attach a default HITL approval policy.

        Deprecated: prefer passing ``approval_policy`` to :meth:`execute` —
        same shared-instance concern as :meth:`set_session`.
        """
        self._approval_policy = policy

    async def execute(
        self,
        task: str,
        context: dict[str, Any],
        llm: LLM,
        *,
        session_id: str | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> AgentResult:
        # Run-scoped state: never mutate the (potentially shared) agent
        # instance for per-run values — concurrent pipelines share it.
        effective_session = session_id if session_id is not None else self._session_id
        effective_policy = approval_policy if approval_policy is not None else self._approval_policy
        start = time.perf_counter()
        try:
            user_message = await self._prompt_fn(task, context)
        except Exception as e:
            raise AgentError(self.name, f"Prompt function failed: {e}") from e

        system_prompt = f"You are a {self.role}. Provide clear, thorough, well-structured responses."

        # M3: Inject prior session context from memory into the system prompt.
        if self._memory is not None:
            prev = await self._memory.load_context(effective_session)
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
            content, tokens_used, cost, model_name, trace = await self._run_tool_loop(
                messages, llm, approval_policy=effective_policy
            )
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
            await self._memory.save_context(effective_session, self.name, content)

        data: dict[str, Any] | None = None
        if self._output_schema is not None:
            try:
                validated = self._output_schema.model_validate_json(content)
                data = validated.model_dump()
                metadata["validated_output"] = data
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
            data=data,
        )

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        llm: LLM,
        start_iteration: int = 0,
        initial_total_tokens: int = 0,
        initial_total_cost: float = 0.0,
        initial_trace: list[dict[str, Any]] | None = None,
        initial_seen_calls: set[tuple[str, str]] | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> tuple[str, int, float, str, list[dict[str, Any]]]:
        """Drive a ReAct-style loop: call the LLM, run any requested tools, repeat.

        Features:
        - Multiple tool calls run concurrently via ``asyncio.gather`` (B1).
        - Duplicate tool calls are detected, skipped, and reported as errors (B3).
        - Tool outputs > 5000 chars are truncated (B4).
        - Message list is trimmed to prevent context overflow (B2).
        - Transient LLM errors are retried once per iteration (B7).

        When *start_iteration* > 0 the loop resumes from a prior
        :class:`~agentflow.hitl.PauseExecution`, preserving accumulated tokens,
        cost, trace, and seen-calls state.

        Returns:
            (final_content, total_tokens, total_cost, model_name, tool_call_trace)
        """
        tool_map = {t.name: t for t in self._tools}
        total_tokens = initial_total_tokens
        total_cost = initial_total_cost
        model_name = llm.model
        trace: list[dict[str, Any]] = initial_trace or []
        seen_calls: set[tuple[str, str]] = initial_seen_calls or set()

        for iteration in range(start_iteration, self._max_tool_iterations):
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
                    # HITL: check approval policy before dispatching.
                    if approval_policy is not None and approval_policy.requires_approval(
                        name, arguments
                    ):
                        # Drain results for calls handled before the pause —
                        # dispatched tasks would otherwise leak, and the saved
                        # conversation must contain a tool message for every
                        # already-processed call to stay API-valid on resume.
                        for done_id, done_task in coros:
                            results_map[done_id] = await done_task
                        for done_id in call_order:
                            _, dname, dargs, doutput = results_map[done_id]
                            doutput = _truncate_output(doutput)
                            trace.append(
                                {"tool": dname, "arguments": dargs, "result": doutput}
                            )
                            messages.append(
                                {"role": "tool", "tool_call_id": done_id, "content": doutput}
                            )

                        pending: list[dict[str, Any]] = []
                        pause_idx = tool_calls.index(call)
                        for rc in tool_calls[pause_idx:]:
                            rfn = rc["function"]
                            rkey = (rfn["name"], rfn["arguments"])
                            if rkey not in seen_calls:
                                pending.append(rc)
                        raise PauseExecution(
                            agent_name=self.name,
                            tool_name=name,
                            tool_arguments=arguments,
                            tool_call_id=call["id"],
                            messages=messages,
                            total_tokens=total_tokens,
                            total_cost=total_cost,
                            model_name=model_name,
                            trace=trace,
                            pending_calls=pending,
                            seen_calls=[[n, a] for n, a in seen_calls],
                            iterations_used=iteration,
                        )
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

    async def resume_execution(
        self,
        pause_data: dict[str, Any],
        llm: LLM,
        approved: bool,
        human_feedback: str = "",
    ) -> AgentResult:
        """Resume execution after a :class:`~agentflow.hitl.PauseExecution`.

        Applies the human decision to the paused tool call, processes any
        remaining tool calls from the same LLM response batch, then re-enters
        the ReAct loop from where it left off.

        Args:
            pause_data: Serialized state from ``PauseExecution.as_dict()``.
            llm: The LLM provider for continued generation.
            approved: ``True`` to execute the pending tool; ``False`` to inject
                      *human_feedback* as an error observation instead.
            human_feedback: Contextual message injected when *approved* is
                            ``False`` so the agent can self-correct.

        Returns:
            An ``AgentResult`` with the final agent output.
        """
        start = time.perf_counter()
        tool_map = {t.name: t for t in self._tools}
        messages: list[dict[str, Any]] = pause_data["messages"]
        pending_calls: list[dict[str, Any]] = pause_data.get("pending_calls", [])
        seen_calls: set[tuple[str, str]] = set(
            tuple(p) for p in pause_data.get("seen_calls", [])
        )

        if pending_calls:
            paused_call = pending_calls[0]
            if approved:
                call_id, name, args, output = await self._execute_single_tool(
                    paused_call, tool_map
                )
                output = _truncate_output(output)
                pause_data["trace"].append(
                    {"tool": name, "arguments": args, "result": output}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": output}
                )
            else:
                name = paused_call["function"]["name"]
                args = paused_call["function"]["arguments"]
                messages.append(
                    {"role": "tool", "tool_call_id": paused_call["id"], "content": human_feedback}
                )
                pause_data["trace"].append(
                    {
                        "tool": name,
                        "arguments": args,
                        "result": f"[HUMAN REJECTED] {human_feedback}",
                    }
                )

            for call in pending_calls[1:]:
                fn = call["function"]
                key = (fn["name"], fn["arguments"])
                if key in seen_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": (
                                f"Error: Duplicate tool call detected. You already called "
                                f"'{fn['name']}' with these arguments."
                            ),
                        }
                    )
                else:
                    seen_calls.add(key)
                    cid, tname, targs, output = await self._execute_single_tool(
                        call, tool_map
                    )
                    output = _truncate_output(output)
                    pause_data["trace"].append(
                        {"tool": tname, "arguments": targs, "result": output}
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": cid, "content": output}
                    )

        content, tokens_used, cost, model_name, final_trace = await self._run_tool_loop(
            messages,
            llm,
            start_iteration=pause_data["iterations_used"] + 1,
            initial_total_tokens=pause_data["total_tokens"],
            initial_total_cost=pause_data["total_cost"],
            initial_trace=pause_data["trace"],
            initial_seen_calls=seen_calls,
            approval_policy=self._approval_policy,
        )

        # _run_tool_loop was seeded with the pre-pause totals, so its return
        # values already include them.
        total_tokens = tokens_used
        total_cost = cost

        if self._memory is not None and content:
            await self._memory.save_context(self._session_id, self.name, content)

        if self._output_schema is not None:
            try:
                self._output_schema.model_validate_json(content)
            except Exception as e:
                raise AgentOutputValidationError(self.name, str(e)) from e

        duration = time.perf_counter() - start
        return AgentResult(
            agent=self.name,
            output=content,
            tokens_used=total_tokens,
            cost=round(total_cost, 6),
            duration=round(duration, 3),
            cached=False,
            metadata={
                "model": model_name,
                "tool_calls": final_trace,
            },
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
