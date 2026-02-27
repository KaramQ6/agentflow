"""Pipeline orchestrator for multi-agent execution."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, AsyncGenerator, Callable

from .agent import _DecoratorAgent, BaseAgent
from .events import EventEmitter
from .exceptions import AgentError, AgentTimeoutError, PipelineError
from .llm import LLM
from .types import AgentResult, Event, PipelineResult


AgentLike = _DecoratorAgent | BaseAgent


class _PipelineNode:
    """Internal node in the pipeline graph."""

    def __init__(
        self,
        agent: AgentLike,
        depends_on: list[str],
        timeout: float | None = None,
        condition: Callable[[dict[str, str]], bool] | None = None,
    ):
        self.agent = agent
        self.depends_on = depends_on
        self.timeout = timeout
        self.condition = condition


class Pipeline:
    """Multi-agent pipeline with parallel DAG execution.

    Agents at the same dependency level run concurrently via asyncio.gather().

    Args:
        llm: The LLM provider for all agents.
        retry_failed_agents: How many times to retry a failed agent (default 0).

    Usage:
        pipe = Pipeline(llm=llm)
        pipe.add(researcher)
        pipe.add(fact_checker)                          # runs in parallel with researcher
        pipe.add(writer, depends_on=["researcher", "fact_checker"])
        result = await pipe.run("AI in Healthcare")
    """

    def __init__(self, llm: LLM, retry_failed_agents: int = 0):
        self._llm = llm
        self._retry_failed_agents = retry_failed_agents
        self._nodes: list[_PipelineNode] = []
        self._agent_names: set[str] = set()

    def add(
        self,
        agent: AgentLike,
        depends_on: list[str] | None = None,
        timeout: float | None = None,
        condition: Callable[[dict[str, str]], bool] | None = None,
    ) -> "Pipeline":
        """Add an agent to the pipeline.

        Args:
            agent: An @Agent-decorated function or BaseAgent subclass instance.
            depends_on: List of agent names this agent depends on.
            timeout: Max seconds this agent may run before AgentTimeoutError is raised.
            condition: Callable receiving current context dict; if it returns False
                       the agent is skipped (emits agent_skipped event).

        Returns:
            self (for chaining).
        """
        name = agent.name
        if name in self._agent_names:
            raise PipelineError(f"Duplicate agent name: '{name}'")

        deps = depends_on or []
        for dep in deps:
            if dep not in self._agent_names:
                raise PipelineError(
                    f"Agent '{name}' depends on '{dep}', but '{dep}' hasn't been added yet"
                )

        self._nodes.append(_PipelineNode(agent, deps, timeout=timeout, condition=condition))
        self._agent_names.add(name)
        return self

    def _resolve_levels(self) -> list[list[_PipelineNode]]:
        """Group agents into parallel execution levels using Kahn's algorithm.

        Returns a list of levels; agents within the same level have no
        inter-dependencies and can run concurrently.

        Example:
            researcher, fact_checker (no deps) -> Level 0  [parallel]
            writer (depends on both)           -> Level 1
        """
        node_map = {n.agent.name: n for n in self._nodes}
        # Build in-degree count
        in_degree: dict[str, int] = {n.agent.name: 0 for n in self._nodes}
        dependents: dict[str, list[str]] = {n.agent.name: [] for n in self._nodes}

        for node in self._nodes:
            for dep in node.depends_on:
                in_degree[node.agent.name] += 1
                dependents[dep].append(node.agent.name)

        # Kahn's BFS for level assignment
        queue: deque[str] = deque(
            name for name, degree in in_degree.items() if degree == 0
        )
        levels: list[list[_PipelineNode]] = []

        while queue:
            level_size = len(queue)
            level: list[_PipelineNode] = []
            for _ in range(level_size):
                name = queue.popleft()
                level.append(node_map[name])
                for child in dependents[name]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            levels.append(level)

        if sum(len(lvl) for lvl in levels) != len(self._nodes):
            raise PipelineError("Cycle detected in pipeline dependency graph")

        return levels

    async def _execute_node(
        self,
        node: _PipelineNode,
        task: str,
        context: dict[str, str],
        level_index: int,
    ) -> AgentResult:
        """Execute a single node with timeout and retry support."""
        agent = node.agent
        attempts = self._retry_failed_agents + 1

        for attempt in range(attempts):
            try:
                coro = agent.execute(task, context, self._llm)
                if node.timeout is not None:
                    try:
                        result = await asyncio.wait_for(coro, timeout=node.timeout)
                    except asyncio.TimeoutError:
                        raise AgentTimeoutError(agent.name, node.timeout)
                else:
                    result = await coro

                result.level = level_index
                return result

            except AgentTimeoutError:
                raise  # timeouts are not retriable
            except AgentError:
                if attempt < attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        # unreachable, but makes type-checkers happy
        raise AgentError(agent.name, "all retry attempts exhausted")

    async def run(self, task: str) -> PipelineResult:
        """Execute the pipeline with parallel level execution.

        Args:
            task: The task string passed to each agent.

        Returns:
            PipelineResult with all agent results and pipeline metadata.
        """
        levels = self._resolve_levels()
        results: dict[str, AgentResult] = {}
        context: dict[str, str] = {}
        last_output = ""

        for level_index, level in enumerate(levels):
            # Filter out agents whose condition is not met
            to_run: list[_PipelineNode] = []
            for node in level:
                if node.condition is not None and not node.condition(context):
                    continue
                to_run.append(node)

            if not to_run:
                continue

            # Build context scoped to each agent's declared dependencies
            level_coros = [
                self._execute_node(node, task, {k: v for k, v in context.items() if k in node.depends_on}, level_index)
                for node in to_run
            ]

            level_results = await asyncio.gather(*level_coros, return_exceptions=True)

            for node, result in zip(to_run, level_results):
                if isinstance(result, BaseException):
                    raise result
                results[node.agent.name] = result
                context[node.agent.name] = result.output
                last_output = result.output

        total_tokens = sum(r.tokens_used for r in results.values())
        total_duration = sum(r.duration for r in results.values())
        cache_hits = sum(1 for r in results.values() if r.cached)

        return PipelineResult(
            output=last_output,
            results=results,
            total_tokens=total_tokens,
            total_duration=round(total_duration, 3),
            levels_executed=len(levels),
            agents_with_cache_hits=cache_hits,
        )

    async def stream(self, task: str) -> AsyncGenerator[Event, None]:
        """Execute the pipeline and yield real-time events.

        Yields Event objects as agents start, complete, skip, or error.
        The final event has type "pipeline_complete" or "pipeline_error".
        """
        emitter = EventEmitter()
        levels = self._resolve_levels()
        results: dict[str, AgentResult] = {}
        context: dict[str, str] = {}

        async def _run() -> None:
            try:
                for level_index, level in enumerate(levels):
                    to_run: list[_PipelineNode] = []
                    for node in level:
                        if node.condition is not None and not node.condition(context):
                            emitter.emit("agent_skipped", agent=node.agent.name, level=level_index)
                            continue
                        to_run.append(node)

                    if not to_run:
                        continue

                    # Emit start events for all agents in this level
                    for node in to_run:
                        emitter.emit("agent_start", agent=node.agent.name, role=node.agent.role, level=level_index)

                    level_coros = [
                        self._execute_node(
                            node,
                            task,
                            {k: v for k, v in context.items() if k in node.depends_on},
                            level_index,
                        )
                        for node in to_run
                    ]

                    level_results = await asyncio.gather(*level_coros, return_exceptions=True)

                    for node, result in zip(to_run, level_results):
                        if isinstance(result, BaseException):
                            emitter.emit("agent_error", agent=node.agent.name, error=str(result))
                            raise result
                        results[node.agent.name] = result
                        context[node.agent.name] = result.output
                        emitter.emit(
                            "agent_complete",
                            agent=node.agent.name,
                            tokens=result.tokens_used,
                            duration=result.duration,
                            cached=result.cached,
                            level=level_index,
                            output_preview=result.output[:200],
                        )

                total_tokens = sum(r.tokens_used for r in results.values())
                total_duration = sum(r.duration for r in results.values())
                emitter.emit(
                    "pipeline_complete",
                    total_tokens=total_tokens,
                    total_duration=round(total_duration, 3),
                    agents_completed=len(results),
                    levels_executed=len(levels),
                )
            except Exception as e:
                emitter.emit("pipeline_error", error=str(e))
            finally:
                emitter.done()

        run_task = asyncio.create_task(_run())

        async for event in emitter.stream():
            yield event

        await run_task
