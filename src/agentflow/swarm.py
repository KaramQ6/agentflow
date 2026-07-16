"""Swarm routing via SupervisorAgent that delegates to worker agents."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from .agent import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MESSAGES_MAX_LENGTH,
    BaseAgent,
    _DecoratorAgent,
    _truncate_output,
)
from .exceptions import AgentError
from .llm import LLM
from .tools import Tool
from .types import AgentResult

_log = logging.getLogger("agentflow.swarm")

AgentLike = _DecoratorAgent | BaseAgent


class _WorkerLedger:
    """Run-scoped accumulator for delegated-worker tokens, cost, and trace.

    One ledger is created per ``execute()`` call so that concurrent runs
    sharing a ``SupervisorAgent`` instance never see each other's billing.
    """

    __slots__ = ("tokens", "cost", "trace")

    def __init__(self) -> None:
        self.tokens = 0
        self.cost = 0.0
        self.trace: list[dict[str, Any]] = []


class SupervisorAgent(BaseAgent):
    """An agent that orchestrates a swarm of specialized worker agents.

    Instead of running a fixed pipeline DAG, the Supervisor uses an LLM to
    dynamically decide which worker(s) to delegate sub-tasks to. Workers are
    owned exclusively by the Supervisor — they are not added to the pipeline.

    Token usage and cost incurred by delegated workers are bubbled up and
    added to the Supervisor's final ``AgentResult`` to maintain accurate
    billing.

    Args:
        name: Unique identifier for the supervisor.
        role: Describes the supervisor's persona.
        workers: List of agent instances this supervisor can delegate to.
        max_tool_iterations: Safety cap on delegation rounds (default 6).
    """

    def __init__(
        self,
        name: str,
        role: str,
        workers: list[AgentLike],
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ):
        super().__init__(name=name, role=role)
        self._workers: dict[str, AgentLike] = {w.name: w for w in workers}
        self._max_tool_iterations = max_tool_iterations

    async def execute(self, task: str, context: dict[str, Any], llm: LLM) -> AgentResult:
        """Execute the supervisor ReAct loop.

        Builds a dynamic system prompt listing available workers, generates
        a ``delegate_task`` tool, then runs a tool-calling loop. Worker
        results are accumulated and their token/cost is merged into the
        supervisor's final result.
        """
        start = time.perf_counter()
        # Run-scoped ledger: concurrent runs sharing this instance must not
        # see each other's delegation costs.
        ledger = _WorkerLedger()

        supervisor_trace: list[dict[str, Any]] = []

        delegate_tool = Tool(
            self._make_delegate_fn(llm, ledger),
            name="delegate_task",
            description=(
                "Delegate a sub-task to one of the specialized worker agents. "
                "Provide the worker_name (must match an available worker) and "
                "a detailed sub_task description. Returns the worker's output."
            ),
        )

        system_prompt = self._build_system_prompt()
        user_message = task
        if context:
            # Context values may be dicts (validated output_schema results).
            ctx_parts = [f"{k}: {str(v)[:200]}" for k, v in context.items()]
            user_message = (
                "Context from upstream:\n" + "\n".join(ctx_parts) + f"\n\nTask:\n{task}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        openai_tools = [delegate_tool.openai_schema]
        tool_map = {delegate_tool.name: delegate_tool}
        total_tokens = 0
        total_cost = 0.0
        model_name = llm.model

        for _ in range(self._max_tool_iterations):
            try:
                response = await llm.generate(messages, tools=openai_tools)
            except Exception as e:
                raise AgentError(self.name, str(e)) from e

            total_tokens += response.tokens
            total_cost += response.cost
            model_name = response.model
            tool_calls = response.tool_calls

            if not tool_calls:
                content = response.content
                duration = time.perf_counter() - start
                return AgentResult(
                    agent=self.name,
                    output=content,
                    tokens_used=total_tokens + ledger.tokens,
                    cost=round(total_cost + ledger.cost, 6),
                    duration=round(duration, 3),
                    metadata={
                        "model": model_name,
                        "tool_calls": supervisor_trace,
                        "worker_delegations": ledger.trace,
                        "accumulated_worker_tokens": ledger.tokens,
                        "accumulated_worker_cost": round(ledger.cost, 6),
                    },
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": tool_calls,
                }
            )

            results_map: dict[str, tuple[str, str, str]] = {}
            coros: list[tuple[str, asyncio.Task[tuple[str, str, str]]]] = []
            call_order: list[str] = []

            for call in tool_calls:
                fn = call["function"]
                name = fn["name"]
                if name in tool_map:
                    task_obj = asyncio.create_task(
                        self._execute_delegate(call, tool_map)
                    )
                    coros.append((call["id"], task_obj))
                else:
                    results_map[call["id"]] = (
                        call["id"],
                        name,
                        f"Error: unknown tool '{name}'",
                    )
                call_order.append(call["id"])

            for cid, task_obj in coros:
                results_map[cid] = await task_obj

            for call_id in call_order:
                cid, tname, output = results_map[call_id]
                output = _truncate_output(output)
                supervisor_trace.append(
                    {"tool": tname, "arguments": "...", "result": output[:500]}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": cid, "content": output}
                )

            if len(messages) > MESSAGES_MAX_LENGTH:
                overflow = len(messages) - MESSAGES_MAX_LENGTH
                messages[2 : 2 + overflow] = []

        raise AgentError(
            self.name,
            f"exceeded max_tool_iterations={self._max_tool_iterations} without a final answer",
        )

    def _make_delegate_fn(self, llm: LLM, ledger: _WorkerLedger) -> Callable[..., Any]:
        """Create the ``delegate_task`` tool function for one run.

        Worker usage is recorded on *ledger* (run-scoped), never on ``self``.
        """

        supervisor_name = self.name

        async def delegate(worker_name: str, sub_task: str) -> str:
            """Delegate a sub-task to a specialized worker agent.

            Args:
                worker_name: Name of the worker agent to delegate to.
                sub_task: The sub-task description for the worker.
            """
            if worker_name == supervisor_name:
                return "Error: Cannot delegate to yourself (infinite loop prevention)."
            worker = self._workers.get(worker_name)
            if worker is None:
                return (
                    f"Error: No worker named '{worker_name}'. "
                    f"Available workers: {list(self._workers.keys())}"
                )
            try:
                result = await worker.execute(sub_task, {}, llm)
            except Exception as e:
                return f"Error: Worker '{worker_name}' failed — {e}"
            ledger.tokens += result.tokens_used
            ledger.cost += result.cost
            ledger.trace.append(
                {
                    "worker": worker_name,
                    "sub_task": sub_task,
                    "output": result.output,
                    "tokens_used": result.tokens_used,
                    "cost": result.cost,
                }
            )
            return result.output

        return delegate

    async def _execute_delegate(
        self, call: dict[str, Any], tool_map: dict[str, Tool]
    ) -> tuple[str, str, str]:
        """Execute a single tool call and return (call_id, tool_name, output)."""
        fn = call["function"]
        name = fn["name"]
        target = tool_map.get(name)
        if target is None:
            return call["id"], name, f"Error: unknown tool '{name}'"
        try:
            output = await target.acall(call["function"]["arguments"])
            _log.info(
                "Supervisor %s delegating to tool: %s", self.name, name,
            )
        except Exception as e:
            output = f"Error: {e}"
            _log.error(
                "Delegation error in supervisor %s: %s — %s", self.name, name, e,
            )
        return call["id"], name, output

    def _build_system_prompt(self) -> str:
        """Build the supervisor's system prompt with worker descriptions."""
        lines: list[str] = []
        for name, agent in self._workers.items():
            lines.append(f"- **{name}**: {agent.role}")
        workers_text = "\n".join(lines) if lines else "(none)"

        return (
            f"You are a {self.role}. You manage a team of specialized workers. "
            f"Break down the user's task and delegate sub-tasks using the "
            f"`delegate_task` tool. After collecting all results, synthesize "
            f"a comprehensive final answer.\n\n"
            f"## Available Workers\n\n{workers_text}\n\n"
            f"## Guidelines\n\n"
            f"- Match each sub-task to the most suitable worker based on its role.\n"
            f"- Call `delegate_task` one or more times as needed.\n"
            f"- Once all delegations return, compile the findings into one cohesive response.\n"
            f"- Do not ask the user for clarification — use your best judgment."
        )
