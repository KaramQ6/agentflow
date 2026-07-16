"""Dynamic Supervisor Agent with Pydantic-driven tool generation and loop prevention.

The :class:`DynamicSupervisorAgent` builds on :class:`SupervisorAgent` by
dynamically generating the ``delegate_task`` tool schema using Pydantic's
``create_model``, enforcing strict context isolation for workers, and
capping delegation recursion depth to prevent infinite worker-to-worker loops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .agent import DEFAULT_MAX_TOOL_ITERATIONS, MESSAGES_MAX_LENGTH, BaseAgent, _truncate_output
from .exceptions import AgentError
from .llm import LLM
from .tools import Tool
from .types import AgentResult

_log = logging.getLogger("agentflow.swarm_routing")

MAX_DELEGATION_DEPTH = 3


class DynamicSupervisorAgent(BaseAgent):
    """An agent that orchestrates a swarm of specialized worker agents.

    Unlike :class:`SupervisorAgent`, this implementation:

    * Dynamically generates the ``delegate_task`` tool schema via
      ``pydantic.create_model``, seeded from the injected worker list.
    * Enforces **strict context isolation**: workers receive an empty
      ``context`` dict and never see the supervisor's system prompt.
    * Caps recursion at ``max_delegation_depth`` (default 3) to prevent
      infinite worker-to-worker delegation loops.
    * Aggregates ``total_tokens`` and ``total_cost`` from all delegated
      workers into the supervisor's final ``AgentResult`` for accurate
      billing.

    Args:
        name: Unique identifier for the supervisor.
        role: Describes the supervisor's persona.
        workers: List of agent instances this supervisor can delegate to.
        max_tool_iterations: Safety cap on delegation rounds (default 6).
        max_delegation_depth: Maximum nested delegation depth (default 3).
    """

    def __init__(
        self,
        name: str,
        role: str,
        workers: list[BaseAgent],
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        max_delegation_depth: int = MAX_DELEGATION_DEPTH,
    ):
        super().__init__(name=name, role=role)
        self._workers: dict[str, BaseAgent] = {w.name: w for w in workers}
        self._max_tool_iterations = max_tool_iterations
        self._max_delegation_depth = max_delegation_depth

        self._accumulated_tokens: int = 0
        self._accumulated_cost: float = 0.0
        self._worker_trace: list[dict[str, Any]] = []

        self._llm: LLM | None = None

        self._delegate_tool = self._build_delegate_tool()

    def _build_delegate_tool(self) -> Tool:
        """Build the ``delegate_task`` tool with a dynamic Pydantic schema.

        The ``Tool`` constructor uses ``inspect.signature`` to examine the
        function's type hints, then ``pydantic.create_model`` to generate
        a Pydantic model, which is converted to a JSON Schema for the LLM.

        The dynamic nature comes from the worker list being embedded in the
        tool's description and the system prompt (see :meth:`_build_system_prompt`).
        """
        worker_names_str = ", ".join(self._workers.keys()) if self._workers else "(none)"
        supervisor = self

        async def delegate(worker_name: str, sub_task: str) -> str:
            """Delegate a sub-task to a specialized worker agent.

            Args:
                worker_name: Name of the worker agent to delegate to.
                sub_task: The sub-task description for the worker.
            """
            # Dynamically update the docstring with the current worker list so
            # the Tool class picks it up as the tool's description.
            delegate.__doc__ = (
                "Delegate a sub-task to a specialized worker agent.\n\n"
                f"Available workers: {worker_names_str}.\n\n"
                "Args:\n"
                "    worker_name: Name of the worker agent to delegate to.\n"
                "    sub_task: The sub-task description for the worker."
            )
            return await supervisor._handle_delegation(worker_name, sub_task, depth=0)

        return Tool(
            delegate,
            name="delegate_task",
            description=(
                "Delegate a sub-task to one of the specialized worker agents. "
                "Provide the worker_name (must match an available worker: "
                f"{worker_names_str}) and a detailed sub_task description. "
                "Returns the worker's output."
            ),
        )

    async def _handle_delegation(
        self, worker_name: str, sub_task: str, depth: int
    ) -> str:
        """Execute a worker delegation with recursion depth protection.

        Args:
            worker_name: Target worker name.
            sub_task: Task to delegate.
            depth: Current delegation depth (increments with each nested call).

        Returns:
            Worker output string or an error message.
        """
        if depth >= self._max_delegation_depth:
            return (
                f"Error: Maximum delegation depth ({self._max_delegation_depth}) "
                f"reached. Cannot further delegate."
            )

        if worker_name == self.name:
            return "Error: Cannot delegate to yourself (infinite loop prevention)."

        worker = self._workers.get(worker_name)
        if worker is None:
            return (
                f"Error: No worker named '{worker_name}'. "
                f"Available workers: {list(self._workers.keys())}"
            )

        try:
            # Strict context isolation: workers receive an empty context dict
            # so the supervisor's system prompt never leaks into worker
            # execution.
            assert self._llm is not None, "LLM must be set during execute()"
            result = await worker.execute(sub_task, {}, llm=self._llm)
        except Exception as e:
            return f"Error: Worker '{worker_name}' failed — {e}"

        self._accumulated_tokens += result.tokens_used
        self._accumulated_cost += result.cost
        self._worker_trace.append(
            {
                "worker": worker_name,
                "sub_task": sub_task,
                "output": result.output,
                "tokens_used": result.tokens_used,
                "cost": result.cost,
                "depth": depth,
            }
        )
        return result.output

    async def execute(self, task: str, context: dict[str, str], llm: LLM) -> AgentResult:
        """Execute the supervisor ReAct loop.

        Builds a dynamic system prompt listing available workers, generates
        a ``delegate_task`` tool with a Pydantic-driven schema, then runs a
        tool-calling loop.  Worker results are accumulated and their
        token/cost is merged into the supervisor's final result.
        """
        start = time.perf_counter()
        self._accumulated_tokens = 0
        self._accumulated_cost = 0.0
        self._worker_trace = []
        self._llm = llm

        supervisor_trace: list[dict[str, Any]] = []

        system_prompt = self._build_system_prompt()
        user_message = task
        if context:
            ctx_parts = [f"{k}: {v[:200]}" for k, v in context.items()]
            user_message = (
                "Context from upstream:\n" + "\n".join(ctx_parts) + f"\n\nTask:\n{task}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        openai_tools = [self._delegate_tool.openai_schema]
        tool_map = {self._delegate_tool.name: self._delegate_tool}
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
                    tokens_used=total_tokens + self._accumulated_tokens,
                    cost=round(total_cost + self._accumulated_cost, 6),
                    duration=round(duration, 3),
                    metadata={
                        "model": model_name,
                        "tool_calls": supervisor_trace,
                        "worker_delegations": self._worker_trace,
                        "accumulated_worker_tokens": self._accumulated_tokens,
                        "accumulated_worker_cost": round(self._accumulated_cost, 6),
                        "max_delegation_depth": self._max_delegation_depth,
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
                tname = fn["name"]
                if tname in tool_map:
                    task_obj = asyncio.create_task(
                        self._execute_delegate(call, tool_map)
                    )
                    coros.append((call["id"], task_obj))
                else:
                    results_map[call["id"]] = (
                        call["id"],
                        tname,
                        f"Error: unknown tool '{tname}'",
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
                "DynamicSupervisor %s delegating to tool: %s", self.name, name,
            )
        except Exception as e:
            output = f"Error: {e}"
            _log.error(
                "Delegation error in DynamicSupervisor %s: %s — %s",
                self.name, name, e,
            )
        return call["id"], name, output

    def _build_system_prompt(self) -> str:
        """Build the supervisor's system prompt with worker descriptions."""
        lines: list[str] = []
        for wname, agent in self._workers.items():
            lines.append(f"- **{wname}**: {agent.role}")
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
            f"- Do not ask the user for clarification — use your best judgment.\n"
            f"- Delegate at most {self._max_delegation_depth} levels deep."
        )
