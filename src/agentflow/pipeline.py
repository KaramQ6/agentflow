"""Pipeline orchestrator for multi-agent execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from .agent import BaseAgent, _DecoratorAgent
from .events import EventEmitter
from .exceptions import AgentError, AgentTimeoutError, BudgetExceededError, PipelineError
from .hitl import ApprovalPolicy, PauseExecution
from .llm import LLM
from .memory import BaseMemory
from .observability import Hooks, safe_invoke
from .types import AgentResult, Event, PipelineResult

if TYPE_CHECKING:
    from .triggers import BaseTrigger

_logger = logging.getLogger(__name__)

AgentLike = _DecoratorAgent | BaseAgent


class _PipelineNode:
    """Internal node in the pipeline graph."""

    def __init__(
        self,
        agent: AgentLike,
        depends_on: list[str],
        timeout: float | None = None,
        condition: Callable[[dict[str, Any]], bool] | None = None,
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
        budget_usd: Optional hard cost ceiling for a single run. Checked after
            each DAG level; exceeding it raises BudgetExceededError.

    Usage:
        pipe = Pipeline(llm=llm)
        pipe.add(researcher)
        pipe.add(fact_checker)                          # runs in parallel with researcher
        pipe.add(writer, depends_on=["researcher", "fact_checker"])
        result = await pipe.run("AI in Healthcare")
    """

    def __init__(
        self,
        llm: LLM,
        retry_failed_agents: int = 0,
        hooks: Hooks | None = None,
        memory: BaseMemory | None = None,
        session_id: str | None = None,
        approval_policy: ApprovalPolicy | None = None,
        budget_usd: float | None = None,
    ):
        self._llm = llm
        self._retry_failed_agents = retry_failed_agents
        self._hooks = hooks
        self._memory = memory
        self._session_id = session_id
        self._approval_policy = approval_policy
        self._budget_usd = budget_usd
        self._nodes: list[_PipelineNode] = []
        self._agent_names: set[str] = set()

    def add(
        self,
        agent: AgentLike,
        depends_on: list[str] | None = None,
        timeout: float | None = None,
        condition: Callable[[dict[str, Any]], bool] | None = None,
    ) -> Pipeline:
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
        context: dict[str, Any],
        level_index: int,
        session_id: str,
    ) -> AgentResult:
        """Execute a single node with timeout and retry support."""
        agent = node.agent
        attempts = self._retry_failed_agents + 1

        for attempt in range(attempts):
            try:
                if isinstance(agent, _DecoratorAgent):
                    # Run-scoped state travels with the call — mutating a
                    # shared agent instance would cross-contaminate
                    # concurrent pipeline runs.
                    coro = agent.execute(
                        task,
                        context,
                        self._llm,
                        session_id=session_id,
                        approval_policy=self._approval_policy,
                    )
                else:
                    coro = agent.execute(task, context, self._llm)
                if node.timeout is not None:
                    try:
                        result = await asyncio.wait_for(coro, timeout=node.timeout)
                    except asyncio.TimeoutError:
                        raise AgentTimeoutError(agent.name, node.timeout) from None
                else:
                    result = await coro

                result.level = level_index
                return result

            except PauseExecution:
                raise  # let the pipeline-level handler deal with it
            except AgentTimeoutError:
                raise  # timeouts are not retriable
            except AgentError:
                if attempt < attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        # unreachable, but makes type-checkers happy
        raise AgentError(agent.name, "all retry attempts exhausted")

    async def _persist_pause_state(
        self,
        session_id: str,
        run_id: str,
        task: str,
        level_index: int,
        pause_exc: PauseExecution,
        to_run: list[_PipelineNode],
        level_results: list[Any],
        results: dict[str, AgentResult],
        context: dict[str, Any],
    ) -> str:
        """Collect completed agent results and persist HITL pause state to memory.

        Returns the ``last_output`` from the most-recently-completed agent.
        Results and context dicts are mutated in-place.
        """
        last_output = ""
        for n, r in zip(to_run, level_results, strict=False):
            if isinstance(r, AgentResult):
                results[n.agent.name] = r
                context[n.agent.name] = r.data if r.data is not None else r.output
                last_output = r.output

        if self._memory is not None:
            pipeline_state = json.dumps(
                {
                    "task": task,
                    "run_id": run_id,
                    "session_id": session_id,
                    "results": {k: v.model_dump() for k, v in results.items()},
                    "context": context,
                    "current_level": level_index,
                    "paused_agent_name": pause_exc.agent_name,
                }
            )
            await self._memory.save_context(
                session_id, "__hitl_pipeline", pipeline_state
            )
            await self._memory.save_context(
                session_id, "__hitl_agent", json.dumps(pause_exc.as_dict())
            )

        return last_output

    def _check_budget(self, results: dict[str, AgentResult]) -> None:
        """Raise BudgetExceededError if accumulated cost passed ``budget_usd``."""
        if self._budget_usd is None:
            return
        spent = sum(r.cost for r in results.values())
        if spent > self._budget_usd:
            raise BudgetExceededError(self._budget_usd, round(spent, 6))

    async def run(self, task: str) -> PipelineResult:
        """Execute the pipeline with parallel level execution.

        Args:
            task: The task string passed to each agent.

        Returns:
            PipelineResult with all agent results and pipeline metadata.
            When an HITL approval policy blocks a tool call the returned
            result has ``status="paused"`` and ``pause_info`` containing the
            details needed to later call :meth:`resume`.
        """
        wall_start = time.perf_counter()
        levels = self._resolve_levels()
        results: dict[str, AgentResult] = {}
        context: dict[str, Any] = {}
        last_output = ""
        run_id = uuid.uuid4().hex[:8]
        session_id = self._session_id or run_id

        if self._hooks is not None:
            safe_invoke(self._hooks, "on_pipeline_start", task, run_id, len(self._nodes))

        for level_index, level in enumerate(levels):
            # Filter out agents whose condition is not met
            to_run: list[_PipelineNode] = []
            for node in level:
                if node.condition is not None and not node.condition(context):
                    continue
                to_run.append(node)

            if not to_run:
                continue

            if self._hooks is not None:
                for node in to_run:
                    safe_invoke(self._hooks, "on_agent_start", node.agent.name, level_index)

            # Build context scoped to each agent's declared dependencies
            level_coros = [
                self._execute_node(node, task, {k: v for k, v in context.items() if k in node.depends_on}, level_index, session_id)
                for node in to_run
            ]

            level_results = await asyncio.gather(*level_coros, return_exceptions=True)

            # ── HITL: check for paused agents first ──────────────────────────
            for _node, result in zip(to_run, level_results, strict=False):
                if isinstance(result, PauseExecution):
                    last_output = await self._persist_pause_state(
                        session_id, run_id, task, level_index, result,
                        to_run, level_results, results, context,
                    )
                    return PipelineResult(
                        output=last_output,
                        results=results,
                        total_tokens=sum(r.tokens_used for r in results.values()),
                        total_cost=round(sum(r.cost for r in results.values()), 6),
                        total_duration=round(sum(r.duration for r in results.values()), 3),
                        wall_time=round(time.perf_counter() - wall_start, 3),
                        run_id=run_id,
                        levels_executed=level_index,
                        agents_with_cache_hits=sum(1 for r in results.values() if r.cached),
                        status="paused",
                        pause_info={
                            "agent_name": result.agent_name,
                            "tool_name": result.tool_name,
                            "tool_arguments": result.tool_arguments,
                            "session_id": session_id,
                        },
                    )

            # ── Other errors: fail the pipeline ──────────────────────────────
            for node, result in zip(to_run, level_results, strict=False):
                if isinstance(result, BaseException):
                    if self._hooks is not None:
                        err = result if isinstance(result, Exception) else Exception(str(result))
                        safe_invoke(self._hooks, "on_agent_error", node.agent.name, err)
                    raise result
                results[node.agent.name] = result
                context[node.agent.name] = result.data if result.data is not None else result.output
                last_output = result.output
                if self._hooks is not None:
                    safe_invoke(self._hooks, "on_agent_end", result)

            self._check_budget(results)

        total_tokens = sum(r.tokens_used for r in results.values())
        total_cost = sum(r.cost for r in results.values())
        total_duration = sum(r.duration for r in results.values())
        cache_hits = sum(1 for r in results.values() if r.cached)

        pipeline_result = PipelineResult(
            output=last_output,
            results=results,
            total_tokens=total_tokens,
            total_cost=round(total_cost, 6),
            total_duration=round(total_duration, 3),
            wall_time=round(time.perf_counter() - wall_start, 3),
            run_id=run_id,
            levels_executed=len(levels),
            agents_with_cache_hits=cache_hits,
        )
        if self._hooks is not None:
            safe_invoke(self._hooks, "on_pipeline_end", pipeline_result)
        return pipeline_result

    async def resume(
        self, session_id: str, human_feedback: str, approved: bool
    ) -> PipelineResult:
        """Resume a pipeline paused by the HITL approval mechanism.

        Loads the saved pipeline and agent state from memory, applies the
        human decision (execute the pending tool or inject feedback), then
        continues executing the remaining DAG levels.

        Args:
            session_id: The session identifier returned in
                        ``PipelineResult.pause_info["session_id"]``.
            human_feedback: A message injected as a tool observation when
                            *approved* is ``False``, allowing the agent to
                            self-correct.
            approved: ``True`` to execute the pending tool call; ``False`` to
                      reject it and feed *human_feedback* to the agent.

        Returns:
            A ``PipelineResult`` with ``status="completed"`` and all agent
            outputs including the resumed agent and any downstream agents.

        Raises:
            PipelineError: If no memory backend is configured, no paused state
                           exists for the session, or the paused agent is
                           no longer part of the pipeline.
        """
        wall_start = time.perf_counter()
        if self._memory is None:
            raise PipelineError("Cannot resume without a memory backend configured")

        ctx = await self._memory.load_context(session_id)
        pipeline_raw = ctx.get("__hitl_pipeline")
        agent_raw = ctx.get("__hitl_agent")

        if not pipeline_raw or not agent_raw:
            raise PipelineError(
                f"No paused state found for session '{session_id}'"
            )

        pipeline_state: dict[str, Any] = (
            json.loads(pipeline_raw) if isinstance(pipeline_raw, str) else pipeline_raw
        )
        pause_data: dict[str, Any] = (
            json.loads(agent_raw) if isinstance(agent_raw, str) else agent_raw
        )

        task: str = pipeline_state["task"]
        run_id: str = pipeline_state["run_id"]
        current_level: int = pipeline_state["current_level"]
        context: dict[str, Any] = pipeline_state["context"]
        paused_name: str = pipeline_state["paused_agent_name"]

        results: dict[str, AgentResult] = {}
        for name, data in pipeline_state["results"].items():
            results[name] = AgentResult(**data)
        last_output = next(reversed(context.values()), "") if context else ""

        # Locate the paused agent in the pipeline.
        agent: _DecoratorAgent | None = None
        for node in self._nodes:
            if node.agent.name == paused_name and hasattr(node.agent, "resume_execution"):
                agent = node.agent  # type: ignore[assignment]
                break

        if agent is None:
            raise PipelineError(
                f"Paused agent '{paused_name}' not found in pipeline or "
                "does not support resume_execution"
            )

        if hasattr(agent, 'set_session'):
            agent.set_session(session_id)

        # Resume the agent.
        agent_result = await agent.resume_execution(
            pause_data, self._llm, approved, human_feedback
        )
        results[paused_name] = agent_result
        context[paused_name] = (
            agent_result.data if agent_result.data is not None else agent_result.output
        )
        last_output = agent_result.output

        # Continue with remaining pipeline levels.
        levels = self._resolve_levels()
        for level_index in range(current_level + 1, len(levels)):
            level = levels[level_index]
            to_run: list[_PipelineNode] = []
            for node in level:
                if node.condition is not None and not node.condition(context):
                    continue
                to_run.append(node)

            if not to_run:
                continue

            level_coros = [
                self._execute_node(
                    node,
                    task,
                    {k: v for k, v in context.items() if k in node.depends_on},
                    level_index,
                    session_id,
                )
                for node in to_run
            ]

            level_results = await asyncio.gather(*level_coros, return_exceptions=True)

            # Support nested pauses during resume.
            for _node, result in zip(to_run, level_results, strict=False):
                if isinstance(result, PauseExecution):
                    last_output = await self._persist_pause_state(
                        session_id, run_id, task, level_index, result,
                        to_run, level_results, results, context,
                    )
                    return PipelineResult(
                        output=last_output,
                        results=results,
                        total_tokens=sum(r.tokens_used for r in results.values()),
                        total_cost=round(sum(r.cost for r in results.values()), 6),
                        total_duration=round(sum(r.duration for r in results.values()), 3),
                        wall_time=round(time.perf_counter() - wall_start, 3),
                        run_id=run_id,
                        levels_executed=level_index,
                        agents_with_cache_hits=sum(1 for r in results.values() if r.cached),
                        status="paused",
                        pause_info={
                            "agent_name": result.agent_name,
                            "tool_name": result.tool_name,
                            "tool_arguments": result.tool_arguments,
                            "session_id": session_id,
                        },
                    )

            for node, result in zip(to_run, level_results, strict=False):
                if isinstance(result, BaseException):
                    if self._hooks is not None:
                        err = result if isinstance(result, Exception) else Exception(str(result))
                        safe_invoke(self._hooks, "on_agent_error", node.agent.name, err)
                    raise result
                results[node.agent.name] = result
                context[node.agent.name] = result.data if result.data is not None else result.output
                last_output = result.output

            self._check_budget(results)

        # Clean up HITL keys from memory.
        await self._memory.delete_key(session_id, "__hitl_pipeline")
        await self._memory.delete_key(session_id, "__hitl_agent")

        total_tokens = sum(r.tokens_used for r in results.values())
        total_cost = sum(r.cost for r in results.values())
        total_duration = sum(r.duration for r in results.values())
        cache_hits = sum(1 for r in results.values() if r.cached)

        return PipelineResult(
            output=last_output,
            results=results,
            total_tokens=total_tokens,
            total_cost=round(total_cost, 6),
            total_duration=round(total_duration, 3),
            wall_time=round(time.perf_counter() - wall_start, 3),
            run_id=run_id,
            levels_executed=len(levels),
            agents_with_cache_hits=cache_hits,
        )

    async def stream(self, task: str) -> AsyncGenerator[Event, None]:
        """Execute the pipeline and yield real-time events.

        Yields Event objects as agents start, complete, skip, or error.
        The final event has type "pipeline_complete", "pipeline_error",
        or "pipeline_paused" (when the HITL policy blocks a tool call).
        """
        wall_start = time.perf_counter()
        emitter = EventEmitter()
        levels = self._resolve_levels()
        results: dict[str, AgentResult] = {}
        context: dict[str, Any] = {}

        async def _run() -> None:
            session_id = self._session_id or uuid.uuid4().hex[:8]
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
                            session_id,
                        )
                        for node in to_run
                    ]

                    level_results = await asyncio.gather(*level_coros, return_exceptions=True)

                    # ── HITL: check for paused agents first ──────────────────
                    for _node, result in zip(to_run, level_results, strict=False):
                        if isinstance(result, PauseExecution):
                            await self._persist_pause_state(
                                session_id, session_id, task, level_index, result,
                                to_run, level_results, results, context,
                            )
                            emitter.emit(
                                "pipeline_paused",
                                agent=result.agent_name,
                                tool_name=result.tool_name,
                                tool_arguments=result.tool_arguments,
                                session_id=session_id,
                            )
                            emitter.done()
                            return

                    for node, result in zip(to_run, level_results, strict=False):
                        if isinstance(result, BaseException):
                            emitter.emit("agent_error", agent=node.agent.name, error=str(result))
                            raise result
                        results[node.agent.name] = result
                        context[node.agent.name] = (
                            result.data if result.data is not None else result.output
                        )
                        emitter.emit(
                            "agent_complete",
                            agent=node.agent.name,
                            tokens=result.tokens_used,
                            duration=result.duration,
                            cached=result.cached,
                            level=level_index,
                            output_preview=result.output[:200],
                        )

                    self._check_budget(results)

                total_tokens = sum(r.tokens_used for r in results.values())
                total_cost = sum(r.cost for r in results.values())
                total_duration = sum(r.duration for r in results.values())
                emitter.emit(
                    "pipeline_complete",
                    total_tokens=total_tokens,
                    total_cost=round(total_cost, 6),
                    total_duration=round(total_duration, 3),
                    wall_time=round(time.perf_counter() - wall_start, 3),
                    agents_completed=len(results),
                    levels_executed=len(levels),
                )
            except Exception as e:
                emitter.emit("pipeline_error", error=str(e))
            finally:
                emitter.done()

        run_task = asyncio.create_task(_run())

        try:
            async for event in emitter.stream():
                yield event
        finally:
            # Also reached when the consumer abandons the generator early —
            # don't leave the pipeline running in the background.
            if not run_task.done():
                run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run_task

    async def serve(
        self,
        trigger: BaseTrigger,
        max_concurrent: int = 5,
        on_result: Callable[[PipelineResult], Any] | None = None,
        on_error: Callable[[Exception, str], Any] | None = None,
    ) -> None:
        """Run the pipeline as a daemon consuming a continuous trigger stream.

        Continuously consumes ``(task_prompt, context_data)`` tuples from
        *trigger* and dispatches independent :meth:`run` invocations in the
        background via ``asyncio.create_task``.  The method runs forever
        until the trigger stream is exhausted or a ``CancelledError`` is
        received.

        A semaphore-based backpressure mechanism caps the number of
        concurrently executing pipeline runs to *max_concurrent*, preventing
        a spike of incoming messages from overwhelming the host.

        Args:
            trigger: A :class:`~agentflow.triggers.BaseTrigger` that yields
                     ``(task_prompt, context_data)`` tuples.
            max_concurrent: Maximum number of pipeline runs allowed to execute
                            in parallel (default 5).
            on_result: Optional async/sync callback invoked with each
                       completed :class:`PipelineResult`.
            on_error: Optional async/sync callback invoked with
                      ``(exception, task_prompt)`` when a single pipeline
                      run fails.  If not provided errors are logged and
                      swallowed so the daemon keeps running.

        Example::

            pipe = Pipeline(llm=LLM(...))
            pipe.add(my_agent)
            await pipe.serve(
                MQTTTrigger(broker="localhost", topic="sensors/#"),
                max_concurrent=3,
                on_result=lambda r: print(f"Done: {r.run_id}"),
            )
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _run_one(task_prompt: str, context_data: dict[str, Any]) -> None:
            async with sem:
                try:
                    result = await self.run(task_prompt)
                    if on_result is not None:
                        maybe_coro = on_result(result)
                        if asyncio.iscoroutine(maybe_coro):
                            await maybe_coro
                except Exception as exc:
                    if on_error is not None:
                        maybe_coro = on_error(exc, task_prompt)
                        if asyncio.iscoroutine(maybe_coro):
                            await maybe_coro
                    else:
                        _logger.error(
                            "Pipeline daemon run failed for prompt %r: %s",
                            task_prompt[:200],
                            exc,
                        )

        tasks: set[asyncio.Task[None]] = set()

        try:
            async for task_prompt, context_data in trigger.listen():
                task = asyncio.create_task(_run_one(task_prompt, context_data))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except asyncio.CancelledError:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            raise

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
