"""Agent definition via decorators and base class."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from ._scope import run_policy
from .effects import current_effects, run_tool
from .exceptions import AgentError, AgentOutputValidationError, EffectUncertainError, ToolError
from .hitl import ApprovalPolicy, PauseExecution
from .llm import LLM, _estimate_prompt_tokens
from .memory import BaseMemory
from .tools import Tool
from .types import AgentResult

DEFAULT_MAX_TOOL_ITERATIONS = 6
TOOL_OUTPUT_MAX_CHARS = 5000
MESSAGES_MAX_LENGTH = 20
LLM_RETRIES_PER_ITERATION = 1
MEMORY_SNIPPET_MAX_CHARS = 300
DEFAULT_OUTPUT_RETRIES = 1

_log = logging.getLogger("agentflow.agent")


TOOL_CANCEL_GRACE_S = 2.0

_FINAL_ANSWER_PROMPT = (
    "You have used every tool call allowed for this task. Do not call any tool. "
    "Answer now from what you have found so far, and say what is still missing."
)


def _turn_end(messages: list[dict[str, Any]], start: int) -> int:
    """Index just past the turn at *start*: one message and the tool results after it."""
    end = start + 1
    while end < len(messages) and messages[end].get("role") == "tool":
        end += 1
    return end


def _drop_oldest_turn(messages: list[dict[str, Any]]) -> bool:
    """Drop the oldest whole turn after the system and user messages (LOOP-2).

    An assistant message and all of its tool results go together, so no tool result is
    ever left without its call. The latest turn is never dropped. Returns False when
    nothing more can go.
    """
    start = 2
    end = _turn_end(messages, start)
    if start >= len(messages) or end >= len(messages):
        return False
    del messages[start:end]
    return True


def _normalized(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of *tool_calls* with ``arguments: null`` read as ``{}`` (NEW-04)."""
    out = []
    for call in tool_calls:
        fn = call["function"]
        if fn.get("arguments") is None:
            call = {**call, "function": {**fn, "arguments": "{}"}}
        out.append(call)
    return out


_HALTED = "[NOT RUN] batch halted after a rejected call"


def _guarded(target: Any) -> bool:
    """Whether the duplicate-call guard applies. Classified tools opt out (ADR 0006):
    re-reading is normal, and repeating a command is a new effect."""
    return target is None or target.effect is None


def _truncate_output(output: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    """Truncate tool output if it exceeds *max_chars*, appending a truncation marker."""
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"...[Truncated: {len(output) - max_chars} chars]"


def _strip_json_fence(content: str) -> str:
    """Drop a surrounding markdown code fence from *content*.

    Wrapping JSON in ```json fences is the most common way a model breaks a
    "JSON only" instruction. Fixing it locally is free; letting it through
    would spend a whole repair round-trip on a formatting artefact. Anything
    beyond a fence (JSON buried in prose) is left alone — guessing where the
    object starts is how you silently validate the wrong thing.
    """
    text = content.strip()
    if not text.startswith("```"):
        return content
    text = text[3:]
    if text[:4].lower() == "json":
        text = text[4:]
    closing = text.rfind("```")
    if closing != -1:
        text = text[:closing]
    return text.strip()


def _schema_instruction(schema: type[BaseModel]) -> str:
    """The system-prompt block telling the model what shape to answer in."""
    return (
        "\n\nRespond with a single JSON object and nothing else — no prose, no "
        "markdown fences. It must validate against this JSON Schema:\n"
        + json.dumps(schema.model_json_schema())
    )


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


class AgentSpec:
    """Agent created via the :class:`Agent` decorator.

    This is the object users hold after decorating a function; ``Pipeline.add``
    accepts it directly. (Named ``_DecoratorAgent`` before 0.6.)
    """

    def __init__(
        self,
        name: str,
        role: str,
        prompt_fn: Callable[..., Awaitable[str]],
        output_schema: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        memory: BaseMemory | None = None,
        system_prompt: str | None = None,
        output_retries: int = DEFAULT_OUTPUT_RETRIES,
        tool_output_max_chars: int = TOOL_OUTPUT_MAX_CHARS,
    ):
        self.name = name
        self.role = role
        self._prompt_fn = prompt_fn
        self._tool_output_max_chars = tool_output_max_chars
        self._output_schema = output_schema
        self._tools = tools or []
        self._max_tool_iterations = max_tool_iterations
        self._memory = memory
        self._system_prompt = system_prompt
        self._output_retries = output_retries
        self._session_id = "default"
        self._approval_policy: ApprovalPolicy | None = None
        # Tool schemas are static per agent — compute them once, not per call.
        self._openai_tools = [t.openai_schema for t in self._tools]

    def set_session(self, session_id: str) -> None:
        """Set the default session ID used for memory load/save operations.

        Deprecated: prefer passing ``session_id`` to :meth:`execute`. Mutating
        a shared agent instance is unsafe under concurrent pipeline runs.
        """
        warnings.warn(
            "set_session() is deprecated and will be removed in 1.0; pass "
            "session_id= to execute() instead — mutating a shared agent "
            "instance is unsafe under concurrent pipeline runs.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._session_id = session_id

    def set_approval_policy(self, policy: ApprovalPolicy | None) -> None:
        """Attach a default HITL approval policy.

        Deprecated: prefer passing ``approval_policy`` to :meth:`execute` —
        same shared-instance concern as :meth:`set_session`.
        """
        warnings.warn(
            "set_approval_policy() is deprecated and will be removed in 1.0; "
            "pass approval_policy= to execute() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        effective_policy = (
            approval_policy if approval_policy is not None else self._approval_policy or run_policy()
        )
        start = time.perf_counter()
        try:
            user_message = await self._prompt_fn(task, context)
        except Exception as e:
            raise AgentError(self.name, f"Prompt function failed: {e}") from e

        system_prompt = self._system_prompt or (
            f"You are a {self.role}. Provide clear, thorough, well-structured responses."
        )

        # Inject prior session context from memory into the system prompt.
        if self._memory is not None:
            prev = await self._memory.load_context(effective_session)
            if prev:
                parts: list[str] = []
                for name, output in prev.items():
                    if len(str(output)) > MEMORY_SNIPPET_MAX_CHARS:
                        _log.debug(
                            "Memory snippet '%s' truncated to %d chars for agent %s",
                            name, MEMORY_SNIPPET_MAX_CHARS, self.name,
                        )
                    parts.append(f"{name}: {str(output)[:MEMORY_SNIPPET_MAX_CHARS]}")
                system_prompt += (
                    "\n\n[Memory — previous outputs from this session:\n"
                    + "\n".join(parts)
                    + "\n]"
                )

        # Last, so the format contract is the final thing the model reads.
        if self._output_schema is not None:
            system_prompt += _schema_instruction(self._output_schema)

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
            self._fit_context(messages, llm, None)
            try:
                response = await llm.generate(messages)
            except Exception as e:
                raise AgentError(self.name, str(e)) from e
            content = response.content
            tokens_used = response.tokens
            cost = response.cost
            model_name = response.model
            cached = response.cached
            metadata["model"] = model_name

        # M3: Persist this agent's output back to memory for downstream agents.
        if self._memory is not None and content:
            await self._memory.save_context(effective_session, self.name, content)

        data: dict[str, Any] | None = None
        if self._output_schema is not None:
            data, content, repair_tokens, repair_cost = await self._validate_output(
                content, messages, llm
            )
            metadata["validated_output"] = data
            tokens_used += repair_tokens
            cost += repair_cost
            if repair_tokens:
                cached = False  # a repair round-trip means this was not a pure cache hit

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

    async def _validate_output(
        self, content: str, messages: list[dict[str, Any]], llm: LLM
    ) -> tuple[dict[str, Any], str, int, float]:
        """Validate *content* against ``output_schema``, re-asking on failure.

        A single malformed response used to kill the whole run. The model is
        instead shown its own output and the validation errors and asked to
        correct itself, up to ``output_retries`` times — the same self-repair
        loop instructor and PydanticAI make standard. Repairs cost tokens, so
        they are counted into the agent's usage and the run's budget.

        Returns:
            ``(validated_data, final_content, repair_tokens, repair_cost)``.

        Raises:
            AgentOutputValidationError: The last attempt still failed to validate.
        """
        assert self._output_schema is not None
        repair_tokens = 0
        repair_cost = 0.0
        attempts = max(self._output_retries, 0) + 1
        conversation = messages

        for attempt in range(attempts):
            try:
                validated = self._output_schema.model_validate_json(
                    _strip_json_fence(content)
                )
                return validated.model_dump(), content, repair_tokens, repair_cost
            except ValidationError as error:
                if attempt == attempts - 1:
                    raise AgentOutputValidationError(self.name, str(error)) from error

                _log.warning(
                    "Output schema validation failed in agent %s (attempt %d/%d); "
                    "asking the model to correct it",
                    self.name, attempt + 1, attempts,
                )
                # A fresh list per attempt: the caller's messages must not grow
                # a repair transcript as a side effect.
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That response did not validate against the required "
                            f"JSON Schema. Errors:\n{error}\n\n"
                            "Reply with corrected JSON only — no prose, no markdown "
                            "fences."
                        ),
                    },
                ]
                try:
                    response = await llm.generate(conversation)
                except Exception as exc:
                    raise AgentError(self.name, str(exc)) from exc
                content = response.content
                repair_tokens += response.tokens
                repair_cost += response.cost

        raise AgentError(self.name, "output validation loop exited without a result")

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
        - Multiple tool calls run concurrently.
        - Duplicate tool calls are detected, skipped, and reported as errors.
        - Tool outputs beyond ``TOOL_OUTPUT_MAX_CHARS`` are truncated.
        - Message list is trimmed to prevent context overflow.
        - Transient LLM errors are retried once per iteration.

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
            self._fit_context(messages, llm, self._openai_tools)
            # Per-iteration LLM retry for transient failures.
            for retry in range(LLM_RETRIES_PER_ITERATION + 1):
                try:
                    response = await llm.generate(messages, tools=self._openai_tools)
                    break
                except Exception as exc:
                    if retry == LLM_RETRIES_PER_ITERATION:
                        raise AgentError(
                            self.name,
                            f"LLM call failed after {LLM_RETRIES_PER_ITERATION + 1} iteration-level attempts",
                            tokens=total_tokens, cost=total_cost, trace=trace,
                        ) from exc
                    _log.warning(
                        "LLM generation retry in agent %s (iteration %d, attempt %d)",
                        self.name, iteration + 1, retry + 1,
                    )
                    await asyncio.sleep(0.5 * (2 ** retry))

            total_tokens += response.tokens
            total_cost += response.cost
            model_name = response.model

            if not response.tool_calls:
                return response.content, total_tokens, total_cost, model_name, trace

            batch = _normalized(response.tool_calls)
            messages.append({"role": "assistant", "content": response.content or None, "tool_calls": batch})

            # Results are kept by position in the batch, never by call id, so two
            # calls that share an id each keep their own result (NEW-03).
            results: list[tuple[str, str, str, str] | None] = [None] * len(batch)
            tasks: dict[int, asyncio.Task[tuple[str, str, str, str]]] = {}
            last_write: asyncio.Task[tuple[str, str, str, str]] | None = None
            try:
                for index, call in enumerate(batch):
                    fn = call["function"]
                    name = fn["name"]
                    arguments = fn["arguments"]
                    key = (name, arguments)
                    target = tool_map.get(name)
                    guarded = _guarded(target)

                    if guarded and key in seen_calls:
                        results[index] = (call["id"], name, arguments, (
                            f"Error: Duplicate tool call detected. You already called "
                            f"'{name}' with these arguments. Try a different approach."
                        ))
                        _log.warning(
                            "Duplicate tool call blocked in agent %s: %s(%s)",
                            self.name, name, arguments[:120],
                        )
                        continue
                    if approval_policy is not None and approval_policy.requires_approval(name, arguments):
                        # Finish what was dispatched before the pause, so the saved
                        # conversation answers every call already made.
                        for done_index, done_task in tasks.items():
                            results[done_index] = await done_task
                        for done in results[:index]:
                            assert done is not None
                            self._record(done, trace, messages)
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
                            # Every remaining call is pending, so each is answered on resume (HITL-8).
                            pending_calls=list(batch[index:]),
                            seen_calls=[[n, a] for n, a in seen_calls],
                            iterations_used=iteration,
                        )
                    if guarded:
                        seen_calls.add(key)
                    if target is not None and target.effect == "write":
                        # Side effects run one at a time, in the model's order (NEW-12).
                        last_write = asyncio.create_task(self._after(last_write, call, tool_map))
                        tasks[index] = last_write
                    else:
                        tasks[index] = asyncio.create_task(self._execute_single_tool(call, tool_map))

                for index, task in tasks.items():
                    results[index] = await task
            finally:
                # On a timeout, cancel, error or pause, no tool keeps running unobserved (LOOP-3).
                await self._stop_tools(tasks, batch, tool_map)

            for result in results:
                assert result is not None
                self._record(result, trace, messages)

            while len(messages) > MESSAGES_MAX_LENGTH and _drop_oldest_turn(messages):
                pass

        return await self._final_answer(messages, llm, total_tokens, total_cost, trace)

    def _truncate(self, output: str) -> str:
        return _truncate_output(output, self._tool_output_max_chars)

    def _record(
        self, result: tuple[str, str, str, str], trace: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> None:
        call_id, name, arguments, output = result
        output = self._truncate(output)
        trace.append({"tool": name, "arguments": arguments, "result": output})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

    def _fit_context(self, messages: list[dict[str, Any]], llm: LLM, tools: list[dict[str, Any]] | None) -> None:
        """Trim whole turns until the request fits ``llm.context_window`` (NEW-09)."""
        window = getattr(llm, "context_window", None)
        if not window:
            return
        while (needed := _estimate_prompt_tokens(messages, tools)) > window:
            if not _drop_oldest_turn(messages):
                raise AgentError(
                    self.name,
                    f"the request needs about {needed} tokens, more than context_window={window}, "
                    "even with every earlier turn dropped",
                )

    async def _stop_tools(
        self,
        tasks: dict[int, asyncio.Task[tuple[str, str, str, str]]],
        batch: list[dict[str, Any]],
        tool_map: dict[str, Any],
    ) -> None:
        running = [task for task in tasks.values() if not task.done()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.wait(running, timeout=TOOL_CANCEL_GRACE_S)
        # Awaiting a task propagates the agent's own cancellation into it, so a tool may
        # already be cancelled by the time we get here: check every task, not only these.
        for index, task in tasks.items():
            if not task.cancelled():
                continue
            name = batch[index]["function"]["name"]
            target = tool_map.get(name)
            if target is not None and not target._is_async:
                _log.warning(
                    "Tool %s in agent %s is still running after the agent stopped: a sync tool "
                    "runs in a thread that cannot be cancelled. Make it async so it can stop.",
                    name, self.name,
                )

    async def _final_answer(
        self,
        messages: list[dict[str, Any]],
        llm: LLM,
        total_tokens: int,
        total_cost: float,
        trace: list[dict[str, Any]],
    ) -> tuple[str, int, float, str, list[dict[str, Any]]]:
        """At the iteration cap, ask once more without tools instead of discarding the work (NEW-01)."""
        final = [*messages, {"role": "user", "content": _FINAL_ANSWER_PROMPT}]
        limit = f"exceeded max_tool_iterations={self._max_tool_iterations} without a final answer"
        try:
            self._fit_context(final, llm, None)
            response = await llm.generate(final)
        except Exception as exc:
            raise AgentError(self.name, limit, tokens=total_tokens, cost=total_cost, trace=trace) from exc
        total_tokens += response.tokens
        total_cost += response.cost
        if not (response.content or "").strip():
            raise AgentError(self.name, limit, tokens=total_tokens, cost=total_cost, trace=trace)
        return response.content, total_tokens, total_cost, response.model, trace

    async def _after(
        self,
        previous: asyncio.Task[tuple[str, str, str, str]] | None,
        call: dict[str, Any],
        tool_map: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        """Run *call* once the previous write call in the batch has finished."""
        if previous is not None:
            await previous  # its failure propagates, so later writes never run after it
        return await self._execute_single_tool(call, tool_map)

    async def _execute_single_tool(
        self, call: dict[str, Any], tool_map: dict[str, Any]
    ) -> tuple[str, str, str, str]:
        """Execute a single tool call and return (call_id, name, arguments, output).

        Execution is logged for observability.
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
                output = await run_tool(target, arguments, current_effects())
                _log.info(
                    "Tool completed in agent %s: %s (output %d chars)",
                    self.name, name, len(output),
                )
            except EffectUncertainError:
                raise  # the model cannot resolve this; stop the agent and ask a human
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
        *,
        session_id: str | None = None,
        approval_policy: ApprovalPolicy | None = None,
        prior_output: str | None = None,
        on_executed: Callable[[str], Awaitable[None]] | None = None,
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
            session_id: Run-scoped session for memory operations. Falls back to
                        the instance default when omitted.
            approval_policy: Run-scoped HITL policy for the continued loop.
                             Falls back to the instance default when omitted.

        Returns:
            An ``AgentResult`` with the final agent output (including
            ``data`` when an ``output_schema`` is declared, so downstream
            agents see the same context contract as the normal path).
        """
        # Run-scoped state: never mutate the (potentially shared) agent
        # instance for per-run values — concurrent pipelines share it.
        effective_session = session_id if session_id is not None else self._session_id
        effective_policy = (
            approval_policy if approval_policy is not None else self._approval_policy or run_policy()
        )
        start = time.perf_counter()
        tool_map = {t.name: t for t in self._tools}
        messages: list[dict[str, Any]] = pause_data["messages"]
        pending_calls: list[dict[str, Any]] = pause_data.get("pending_calls", [])
        seen_calls: set[tuple[str, str]] = set(
            tuple(p) for p in pause_data.get("seen_calls", [])
        )
        effects = current_effects()
        if effects is not None:
            effects.replay_history(tool_map, pause_data["trace"])

        trace: list[dict[str, Any]] = pause_data["trace"]

        def answer(call_id: str, name: str, args: str, output: str) -> None:
            trace.append({"tool": name, "arguments": args, "result": output})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})

        if pending_calls:
            paused_call = pending_calls[0]
            pfn = paused_call["function"]
            if not approved:
                messages.append({"role": "tool", "tool_call_id": paused_call["id"], "content": human_feedback})
                trace.append({"tool": pfn["name"], "arguments": pfn["arguments"],
                              "result": f"[HUMAN REJECTED] {human_feedback}"})
                # A rejection stops the whole batch: the rest was planned around it (HITL-1).
                for call in pending_calls[1:]:
                    fn = call["function"]
                    answer(call["id"], fn["name"], fn["arguments"], _HALTED)
            else:
                if prior_output is not None:
                    # Recorded by an earlier resume, or skipped by the operator: never re-run.
                    output = prior_output
                else:
                    _, _, _, output = await self._execute_single_tool(paused_call, tool_map)
                    if on_executed is not None:
                        await on_executed(output)
                answer(paused_call["id"], pfn["name"], pfn["arguments"], self._truncate(output))
                for index, call in enumerate(pending_calls[1:], start=1):
                    fn = call["function"]
                    key = (fn["name"], fn["arguments"])
                    guarded = _guarded(tool_map.get(fn["name"]))
                    if guarded and key in seen_calls:
                        answer(call["id"], fn["name"], fn["arguments"],
                               f"Error: Duplicate tool call detected. You already called "
                               f"'{fn['name']}' with these arguments.")
                        continue
                    if effective_policy is not None and effective_policy.requires_approval(
                        fn["name"], fn["arguments"]
                    ):
                        # Approving one call never approves the next (HITL-1).
                        raise PauseExecution(
                            agent_name=self.name,
                            tool_name=fn["name"],
                            tool_arguments=fn["arguments"],
                            tool_call_id=call["id"],
                            messages=messages,
                            total_tokens=pause_data["total_tokens"],
                            total_cost=pause_data["total_cost"],
                            model_name=pause_data["model_name"],
                            trace=trace,
                            pending_calls=pending_calls[index:],
                            seen_calls=[[n, a] for n, a in seen_calls],
                            iterations_used=pause_data["iterations_used"],
                        )
                    if guarded:
                        seen_calls.add(key)
                    cid, tname, targs, output = await self._execute_single_tool(call, tool_map)
                    answer(cid, tname, targs, self._truncate(output))

        content, tokens_used, cost, model_name, final_trace = await self._run_tool_loop(
            messages,
            llm,
            start_iteration=pause_data["iterations_used"] + 1,
            initial_total_tokens=pause_data["total_tokens"],
            initial_total_cost=pause_data["total_cost"],
            initial_trace=pause_data["trace"],
            initial_seen_calls=seen_calls,
            approval_policy=effective_policy,
        )

        # _run_tool_loop was seeded with the pre-pause totals, so its return
        # values already include them.
        total_tokens = tokens_used
        total_cost = cost

        if self._memory is not None and content:
            await self._memory.save_context(effective_session, self.name, content)

        metadata: dict[str, Any] = {
            "model": model_name,
            "tool_calls": final_trace,
        }
        data: dict[str, Any] | None = None
        if self._output_schema is not None:
            data, content, repair_tokens, repair_cost = await self._validate_output(
                content, messages, llm
            )
            metadata["validated_output"] = data
            total_tokens += repair_tokens
            total_cost += repair_cost

        duration = time.perf_counter() - start
        return AgentResult(
            agent=self.name,
            output=content,
            tokens_used=total_tokens,
            cost=round(total_cost, 6),
            duration=round(duration, 3),
            cached=False,
            metadata=metadata,
            data=data,
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
        output_schema: Optional Pydantic BaseModel subclass. When provided, the
                       schema is appended to the system prompt so the model
                       knows the shape to produce, its reply is validated
                       against it, and the validated dict is what downstream
                       agents receive in ``context``. You do not need to embed
                       the schema in your prompt yourself.
        output_retries: How many times to show the model its validation errors
                        and ask it to correct itself before raising
                        AgentOutputValidationError (default 1; 0 disables).
                        Each repair is a real LLM call and is billed to the
                        agent's tokens, cost, and the pipeline budget.
        tools: Optional list of ``Tool`` objects (see :func:`agentflow.tool`).
               When present the agent runs a ReAct loop, letting the model call
               tools and observe results until it produces a final answer.
        max_tool_iterations: Safety cap on tool-calling rounds (default 6).
        system_prompt: Full replacement for the default system prompt
                       (``"You are a {role}. Provide clear, thorough,
                       well-structured responses."``). Memory context, when
                       configured, is appended to whichever prompt is active.

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
        system_prompt: str | None = None,
        output_retries: int = DEFAULT_OUTPUT_RETRIES,
        tool_output_max_chars: int = TOOL_OUTPUT_MAX_CHARS,
    ):
        self.name = name
        self.role = role
        self._output_schema = output_schema
        self._tool_output_max_chars = tool_output_max_chars
        self._tools = tools
        self._max_tool_iterations = max_tool_iterations
        self._memory = memory
        self._system_prompt = system_prompt
        self._output_retries = output_retries

    def __call__(self, fn: Callable[..., Awaitable[str]]) -> AgentSpec:
        return AgentSpec(
            self.name,
            self.role,
            fn,
            output_schema=self._output_schema,
            tools=self._tools,
            max_tool_iterations=self._max_tool_iterations,
            memory=self._memory,
            system_prompt=self._system_prompt,
            output_retries=self._output_retries,
            tool_output_max_chars=self._tool_output_max_chars,
        )


# Backwards-compatible aliases. ``_DecoratorAgent`` was the pre-0.6 name of
# ``AgentSpec``; remove at 1.0.
_DecoratorAgent = AgentSpec

AgentLike = AgentSpec | BaseAgent
"""Anything ``Pipeline.add`` accepts: a decorated agent or a BaseAgent subclass."""
