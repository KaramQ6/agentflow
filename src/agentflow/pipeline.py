"""Pipeline orchestrator for multi-agent execution."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_core import to_jsonable_python

from ._scope import Budget, RunScope, Spend, current_scope, new_run_id, run_scope, tracking
from .agent import AgentLike, AgentSpec
from .effects import EffectLedger, EffectScope, InMemoryLedger, using_effects
from .events import EventEmitter
from .exceptions import (
    AgentError,
    AgentTimeoutError,
    BudgetExceededError,
    EffectUncertainError,
    PipelineError,
)
from .hitl import ApprovalPolicy, PauseExecution
from .llm import LLM, unpriced_budget_error
from .memory import BaseMemory
from .observability import Hooks, safe_invoke
from .pricing import is_priced
from .types import AgentResult, Event, PipelineResult

if TYPE_CHECKING:
    from .triggers import BaseTrigger

_logger = logging.getLogger(__name__)


def _unpriced_models_used(results: dict[str, AgentResult]) -> list[str]:
    """Models in *results* with no pricing entry, so their cost is a placeholder.

    Sorted for a stable, comparable value.
    """
    models = {
        model
        for r in results.values()
        if (model := r.metadata.get("model")) and not is_priced(model)
    }
    return sorted(models)


async def _gather_level(
    coros: list[Coroutine[Any, Any, AgentResult]],
) -> list[AgentResult | BaseException]:
    """Run *coros* concurrently, cancelling the rest once one really fails.

    ``asyncio.gather(..., return_exceptions=True)`` waits for every sibling
    even after one has raised. A failing level discards its results and
    propagates the error, so those siblings were only ever going to burn tokens
    and wall time — in a library whose pitch is per-run cost, that is money
    spent on output nobody reads.

    A :class:`~agentflow.hitl.PauseExecution` is control flow, not a failure,
    and does **not** cancel anything: the pause path persists every sibling
    result alongside the pause so the run can resume.

    Returns results positionally, mirroring ``gather``. A sibling cancelled by
    this function is reported as a :class:`asyncio.CancelledError`.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    pending: set[asyncio.Task[AgentResult]] = set(tasks)

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            if any(_is_real_failure(task) for task in done):
                break
    except asyncio.CancelledError:
        # The whole run was cancelled from outside (e.g. a stream consumer
        # abandoned the generator) — take the level down with it.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    if pending:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    results: list[AgentResult | BaseException] = []
    for task in tasks:
        if task.cancelled():
            results.append(asyncio.CancelledError())
            continue
        error = task.exception()
        results.append(error if error is not None else task.result())
    return results


def _run_totals(results: dict[str, AgentResult]) -> tuple[int, float]:
    """Tokens and cost of the run: kept results plus attempts whose result was discarded."""
    scope = current_scope()
    discarded = scope.discarded if scope is not None else Spend()
    tokens = sum(r.tokens_used for r in results.values()) + discarded.tokens
    cost = sum(r.cost for r in results.values()) + discarded.cost
    return tokens, round(cost, 6)


def _discard(spend: Spend) -> None:
    """Count the spend of an attempt whose result is thrown away."""
    scope = current_scope()
    if scope is not None:
        scope.discarded.add(spend.cost, spend.tokens)


def _budget_stop() -> Exception | None:
    """The run's budget refusal, if any. Agents may wrap or swallow it; the pipeline must not."""
    scope = current_scope()
    return scope.budget.blocked if scope is not None and scope.budget is not None else None


# Pause state lives in its own memory namespace, never in the one agents read
# their context from (HITL-5).
_HITL_SUFFIX = "::agentflow-hitl"
InDoubt = Literal["raise", "retry", "skip"]


def hitl_session(session_id: str) -> str:
    """The memory namespace that holds the pause state of *session_id*."""
    return session_id + _HITL_SUFFIX


def _dumps(value: Any) -> str:
    # to_jsonable_python: datetimes and other schema types survive (HITL-6).
    return json.dumps(value, default=to_jsonable_python)


@dataclass
class _PausedRun:
    """What resume() loads: the run so far, the pauses still open, and call claims."""

    state: dict[str, Any]
    pauses: list[dict[str, Any]]
    claims: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Decision:
    session_id: str
    feedback: str
    approved: bool
    on_in_doubt: InDoubt


def _discarded_state() -> tuple[float, int]:
    scope = current_scope()
    return (scope.discarded.cost, scope.discarded.tokens) if scope is not None else (0.0, 0)


def _last_output_of(results: dict[str, AgentResult]) -> str:
    """The most recently recorded agent output, or ``""`` when nothing ran."""
    return next(reversed(results.values())).output if results else ""


def _levels_completed(results: dict[str, AgentResult]) -> int:
    """How many DAG levels produced results (``AgentResult.level`` is 0-indexed)."""
    return max((r.level for r in results.values()), default=-1) + 1


def _is_real_failure(task: asyncio.Task[AgentResult]) -> bool:
    """Whether *task* finished with an error that should stop its level."""
    if task.cancelled():
        return False
    error = task.exception()
    return error is not None and not isinstance(error, PauseExecution)


async def _with_limit(
    semaphore: asyncio.Semaphore | None,
    start: Callable[[], Coroutine[Any, Any, AgentResult]],
) -> AgentResult:
    """Await the coroutine *start* produces, holding *semaphore* if configured.

    Takes a factory rather than a coroutine so that a task cancelled while
    queueing never creates one — an un-awaited coroutine is a RuntimeWarning
    and a confusing traceback.
    """
    if semaphore is None:
        return await start()
    async with semaphore:
        return await start()


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
        budget_usd: Optional hard cost ceiling for a single run. Each LLM call
            reserves its worst case first; a call that cannot fit raises
            BudgetExceededError. The model must have a price (see ADR 0005).
        effect_ledger: Where ``@tool(effect="write")`` calls are recorded so a
            retry or ``resume()`` never repeats one. Default: a new in-memory
            ledger per run, which covers retries in this process. Pass a durable
            one (``agentflow.effects.SqliteLedger``) to cover ``resume()`` in a new
            process. See ADR 0004.
        max_concurrency: Maximum agents in flight at once, across the whole
            run. Without it a level of 40 agents opens 40 concurrent LLM
            calls. ``None`` (default) means the level width is the only limit.

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
        max_concurrency: int | None = None,
        effect_ledger: EffectLedger | None = None,
    ):
        if max_concurrency is not None and max_concurrency < 1:
            raise PipelineError(
                f"max_concurrency must be at least 1, got {max_concurrency}"
            )
        self._max_concurrency = max_concurrency
        self._effect_ledger = effect_ledger
        self._warned_volatile_effects = False
        self._resuming: set[str] = set()
        if approval_policy is not None and memory is None:
            _logger.warning(
                "Pipeline has an approval_policy but no memory backend: a run that pauses "
                "for approval can never be resumed (pause_info['resumable'] is False)."
            )
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
            depends_on: List of agent names this agent depends on. Forward
                references are allowed — the graph is validated when it is
                resolved (``run``/``stream``/``explain``), not as you build it,
                so you are not forced to topologically sort your own source.
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

        # Dependencies are validated here rather than in add() so that a node
        # may name an agent added after it.
        for node in self._nodes:
            unknown = sorted(d for d in node.depends_on if d not in node_map)
            if unknown:
                raise PipelineError(
                    f"Agent '{node.agent.name}' depends on unknown agent(s): "
                    f"{', '.join(unknown)}. "
                    f"Agents in this pipeline: {', '.join(sorted(node_map)) or '(none)'}"
                )

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
            cycle_members = sorted(name for name, deg in in_degree.items() if deg > 0)
            raise PipelineError(
                "Cycle detected in pipeline dependency graph involving: "
                + ", ".join(cycle_members)
            )

        return levels

    def explain(self) -> str:
        """Describe the resolved DAG without running it or calling any LLM.

        Answers "why did my agents run in that order" — and, because it
        resolves the graph, fails on a cycle or an unknown dependency the same
        way :meth:`run` would. Cheap enough to assert on in a test.

        Returns:
            A multi-line description; print it or log it.

        Example::

            >>> print(pipe.explain())
            Pipeline: 3 agents, 2 levels, max 2 concurrent
            Level 0 (2 agents, run in parallel):
              researcher    role=Research Analyst
              fact_checker  role=Fact Checker
            Level 1 (1 agent):
              writer        role=Content Writer  after=[fact_checker, researcher]
        """
        levels = self._resolve_levels()
        widest = max((len(lvl) for lvl in levels), default=0)
        cap = self._max_concurrency or widest
        lines = [
            f"Pipeline: {len(self._nodes)} agents, {len(levels)} levels, "
            f"max {min(cap, widest) if widest else 0} concurrent"
        ]

        for index, level in enumerate(levels):
            count = len(level)
            suffix = ", run in parallel" if count > 1 else ""
            lines.append(f"Level {index} ({count} agent{'s' if count != 1 else ''}{suffix}):")
            width = max(len(n.agent.name) for n in level)
            for node in level:
                parts = [f"  {node.agent.name.ljust(width)}  role={node.agent.role}"]
                if node.depends_on:
                    parts.append(f"after=[{', '.join(sorted(node.depends_on))}]")
                if node.timeout is not None:
                    parts.append(f"timeout={node.timeout}s")
                if node.condition is not None:
                    parts.append("conditional (may be skipped at run time)")
                lines.append("  ".join(parts))

        return "\n".join(lines)

    def _new_scope(self, run_id: str, spent: float = 0.0) -> RunScope:
        """A run scope with this pipeline's budget and effect ledger.

        Refuses a budget that cannot be measured.
        """
        self._warn_if_effects_are_volatile()
        ledger = self._effect_ledger if self._effect_ledger is not None else InMemoryLedger()
        if self._budget_usd is None:
            return RunScope(run_id, effect_ledger=ledger, policy=self._approval_policy)
        # A duck-typed LLM without a model id is checked per call instead, where it bills.
        model = getattr(self._llm, "model", None)
        if model and not is_priced(model):
            raise unpriced_budget_error(model)
        return RunScope(run_id, budget=Budget(self._budget_usd, spent), effect_ledger=ledger,
                        policy=self._approval_policy)

    def _warn_if_effects_are_volatile(self) -> None:
        """A paused run can resume in another process; an in-memory ledger cannot follow it."""
        if self._warned_volatile_effects or self._memory is None or self._effect_ledger is not None:
            return
        writes = [t.name for n in self._nodes for t in getattr(n.agent, "_tools", []) if t.effect == "write"]
        if writes:
            self._warned_volatile_effects = True
            _logger.warning(
                "Pipeline has a memory backend and write tools %s but no durable effect_ledger. "
                "A resume() in a new process cannot tell which writes already ran. Pass "
                "effect_ledger=agentflow.effects.SqliteLedger(path).",
                writes,
            )

    def _effects_for(self, agent_name: str) -> EffectScope | None:
        """A fresh effect scope for one attempt (one conversation) of *agent_name*."""
        scope = current_scope()
        if scope is None or scope.effect_ledger is None:
            return None
        return EffectScope(scope.effect_ledger, scope.run_id, agent_name)

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
            spend = Spend()
            effects = self._effects_for(agent.name)
            try:
                with tracking(spend), using_effects(effects):
                    result = await self._attempt(node, task, context, session_id)
                result.level = level_index
                return result

            except PauseExecution:
                raise  # its spend travels in the pause state, on the resumed result
            except (AgentTimeoutError, EffectUncertainError):
                _discard(spend)
                raise  # not retriable: a timeout repeats, an uncertain effect needs a human
            except AgentError as exc:
                _discard(spend)
                stop = _budget_stop()
                if stop is not None:
                    raise stop from exc  # retrying cannot make a full budget fit
                if attempts > 1 and effects is not None and effects.ran_unclassified:
                    tools = sorted(effects.ran_unclassified)
                    raise AgentError(
                        agent.name,
                        f"not retried: it ran {tools}, which declare no effect, so a retry could "
                        "repeat them. Mark each tool effect=\"read\" (safe to repeat) or "
                        "effect=\"write\" (recorded, never repeated) to allow retries.",
                    ) from exc
                if attempt < attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except BaseException:
                _discard(spend)  # includes cancellation by a failing sibling
                raise

        # unreachable, but makes type-checkers happy
        raise AgentError(agent.name, "all retry attempts exhausted")

    async def _attempt(
        self, node: _PipelineNode, task: str, context: dict[str, Any], session_id: str
    ) -> AgentResult:
        """One execution of *node*, with its timeout."""
        agent = node.agent
        if isinstance(agent, AgentSpec):
            # Run-scoped state travels with the call — mutating a shared agent
            # instance would cross-contaminate concurrent pipeline runs.
            coro = agent.execute(
                task,
                context,
                self._llm,
                session_id=session_id,
                approval_policy=self._approval_policy,
            )
        else:
            coro = agent.execute(task, context, self._llm)
        if node.timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=node.timeout)
        except asyncio.TimeoutError:
            raise AgentTimeoutError(agent.name, node.timeout) from None

    async def _persist_pause_state(
        self,
        session_id: str,
        run_id: str,
        task: str,
        level_index: int,
        pauses: list[PauseExecution],
        to_run: list[_PipelineNode],
        level_results: list[Any],
        results: dict[str, AgentResult],
        context: dict[str, Any],
    ) -> str:
        """Collect completed agent results and persist every pause of the level.

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
            state = {
                "task": task,
                "run_id": run_id,
                "session_id": session_id,
                "current_level": level_index,
                "discarded_spend": _discarded_state(),
            }
            await self._save_paused(
                session_id,
                _PausedRun(state, [p.as_dict() for p in pauses]),
                results,
                context,
            )
        return last_output

    async def _save_paused(
        self, session_id: str, paused: _PausedRun, results: dict[str, AgentResult], context: dict[str, Any]
    ) -> None:
        assert self._memory is not None
        paused.state["results"] = {k: v.model_dump(mode="json") for k, v in results.items()}
        paused.state["context"] = context
        namespace = hitl_session(session_id)
        await self._memory.save_context(namespace, "pipeline", _dumps(paused.state))
        await self._memory.save_context(namespace, "pauses", _dumps(paused.pauses))
        await self._memory.save_context(namespace, "claims", _dumps(paused.claims))

    async def _load_paused(self, session_id: str) -> _PausedRun:
        assert self._memory is not None
        ctx = await self._memory.load_context(hitl_session(session_id))
        if not ctx.get("pipeline") or not ctx.get("pauses"):
            raise PipelineError(f"No paused state found for session '{session_id}'")

        def load(value: Any) -> Any:
            return json.loads(value) if isinstance(value, str) else value

        return _PausedRun(load(ctx["pipeline"]), load(ctx["pauses"]), load(ctx.get("claims") or "{}"))

    async def _clear_paused(self, session_id: str) -> None:
        assert self._memory is not None
        for key in ("pipeline", "pauses", "claims"):
            await self._memory.delete_key(hitl_session(session_id), key)

    async def _claim_approved_call(
        self, decision: _Decision, paused: _PausedRun
    ) -> tuple[str | None, Any]:
        """Run an approved call at most once across resumes (HITL-2, HITL-3).

        Returns ``(prior_output, on_executed)``: a recorded or skipped result to reuse
        instead of running the call, or a callback that records the result once it ran.
        """
        if not decision.approved:
            return None, None
        pause = paused.pauses[0]
        call_id = pause["tool_call_id"]
        claim = paused.claims.get(call_id)
        if claim is not None and claim["status"] == "done":
            return claim["result"], None
        if claim is not None and claim["status"] == "pending":
            if decision.on_in_doubt == "raise":
                raise EffectUncertainError(pause["agent_name"], pause["tool_name"], call_id)
            if decision.on_in_doubt == "skip":
                return "[IN DOUBT] this call may already have run; the operator chose to continue.", None
        namespace = hitl_session(decision.session_id)
        assert self._memory is not None
        paused.claims[call_id] = {"status": "pending", "result": ""}
        await self._memory.save_context(namespace, "claims", _dumps(paused.claims))

        async def record(output: str) -> None:
            assert self._memory is not None
            paused.claims[call_id] = {"status": "done", "result": output}
            await self._memory.save_context(namespace, "claims", _dumps(paused.claims))

        return None, record

    def _check_budget(self, results: dict[str, AgentResult]) -> None:
        """Raise BudgetExceededError if accumulated cost passed ``budget_usd``."""
        if self._budget_usd is None:
            return
        stop = _budget_stop()
        if stop is not None:
            raise stop
        _, spent = _run_totals(results)
        if spent > self._budget_usd:
            raise BudgetExceededError(self._budget_usd, round(spent, 6))

    def _build_result(
        self,
        *,
        last_output: str,
        results: dict[str, AgentResult],
        run_id: str,
        levels_executed: int,
        wall_start: float,
        pause: PauseExecution | None = None,
        session_id: str = "",
    ) -> PipelineResult:
        """Build the final PipelineResult (``paused`` when *pause* is given)."""
        total_tokens, total_cost = _run_totals(results)
        return PipelineResult(
            unpriced_models=_unpriced_models_used(results),
            output=last_output,
            results=results,
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_duration=round(sum(r.duration for r in results.values()), 3),
            wall_time=round(time.perf_counter() - wall_start, 3),
            run_id=run_id,
            levels_executed=levels_executed,
            agents_with_cache_hits=sum(1 for r in results.values() if r.cached),
            status="paused" if pause is not None else "completed",
            pause_info=(
                {
                    "agent_name": pause.agent_name,
                    "tool_name": pause.tool_name,
                    "tool_arguments": pause.tool_arguments,
                    "session_id": session_id,
                    "resumable": self._memory is not None,
                }
                if pause is not None
                else None
            ),
        )

    async def _run_levels(
        self,
        *,
        levels: list[list[_PipelineNode]],
        start_level: int,
        task: str,
        context: dict[str, Any],
        results: dict[str, AgentResult],
        session_id: str,
        run_id: str,
        emitter: EventEmitter | None = None,
    ) -> tuple[str, PauseExecution | None, int]:
        """Execute DAG levels sequentially; agents within a level run concurrently.

        The single driver behind :meth:`run`, :meth:`resume`, and
        :meth:`stream`: condition filtering, dep-scoped context slices,
        error/pause precedence, result recording, hooks/events, and budget
        enforcement live only here. *context* and *results* are mutated in
        place.

        Returns:
            ``(last_output, pause, levels_executed)`` — *pause* is the
            PauseExecution that stopped the run (its state already persisted),
            or None when every level completed. A real agent failure raises
            instead; an error in a level takes precedence over a sibling
            pause, because silently losing an exception is worse than losing
            a pause (the pause can simply trigger again on a re-run).
        """
        last_output = ""
        # Per-run, not per-pipeline: two concurrent runs of the same Pipeline
        # (as serve() dispatches) each get their own budget of in-flight
        # agents rather than contending for one shared pool.
        semaphore = (
            asyncio.Semaphore(self._max_concurrency)
            if self._max_concurrency is not None
            else None
        )

        for level_index in range(start_level, len(levels)):
            # Filter out agents whose condition is not met
            to_run: list[_PipelineNode] = []
            for node in levels[level_index]:
                if node.condition is not None and not node.condition(context):
                    if emitter is not None:
                        emitter.emit("agent_skipped", agent=node.agent.name, level=level_index)
                    continue
                to_run.append(node)

            if not to_run:
                continue

            for node in to_run:
                if self._hooks is not None:
                    safe_invoke(self._hooks, "on_agent_start", node.agent.name, level_index)
                if emitter is not None:
                    emitter.emit(
                        "agent_start",
                        agent=node.agent.name,
                        role=node.agent.role,
                        level=level_index,
                    )

            # Build context scoped to each agent's declared dependencies
            level_coros = [
                _with_limit(
                    semaphore,
                    functools.partial(
                        self._execute_node,
                        node,
                        task,
                        {k: v for k, v in context.items() if k in node.depends_on},
                        level_index,
                        session_id,
                    ),
                )
                for node in to_run
            ]

            level_results = await _gather_level(level_coros)

            first_error: tuple[_PipelineNode, BaseException] | None = None
            pauses: list[PauseExecution] = []
            for node, result in zip(to_run, level_results, strict=False):
                if isinstance(result, PauseExecution):
                    pauses.append(result)
                elif isinstance(result, asyncio.CancelledError):
                    # Cancelled by the fail-fast above; the agent that actually
                    # failed is another entry in this same list.
                    continue
                elif isinstance(result, BaseException) and first_error is None:
                    first_error = (node, result)

            if first_error is not None:
                node, error = first_error
                for discarded_pause in pauses:
                    _logger.warning(
                        "Discarding HITL pause from agent '%s': sibling agent "
                        "'%s' failed in the same level",
                        discarded_pause.agent_name,
                        node.agent.name,
                    )
                if self._hooks is not None:
                    err = error if isinstance(error, Exception) else Exception(str(error))
                    safe_invoke(self._hooks, "on_agent_error", node.agent.name, err)
                if emitter is not None:
                    emitter.emit("agent_error", agent=node.agent.name, error=str(error))
                raise error

            if pauses:
                persisted_output = await self._persist_pause_state(
                    session_id, run_id, task, level_index, pauses,
                    to_run, level_results, results, context,
                )
                return persisted_output or last_output, pauses[0], level_index

            for node, result in zip(to_run, level_results, strict=False):
                assert isinstance(result, AgentResult)
                results[node.agent.name] = result
                context[node.agent.name] = result.data if result.data is not None else result.output
                last_output = result.output
                if self._hooks is not None:
                    safe_invoke(self._hooks, "on_agent_end", result)
                if emitter is not None:
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

        return last_output, None, len(levels)

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
        run_id = new_run_id()
        with run_scope(self._new_scope(run_id)):
            return await self._run(task, run_id, levels, wall_start)

    async def _run(
        self, task: str, run_id: str, levels: list[list[_PipelineNode]], wall_start: float
    ) -> PipelineResult:
        results: dict[str, AgentResult] = {}
        context: dict[str, Any] = {}
        session_id = self._session_id or run_id

        if self._hooks is not None:
            safe_invoke(self._hooks, "on_pipeline_start", task, run_id, len(self._nodes))

        try:
            last_output, pause, levels_executed = await self._run_levels(
                levels=levels,
                start_level=0,
                task=task,
                context=context,
                results=results,
                session_id=session_id,
                run_id=run_id,
            )
        except BudgetExceededError as exc:
            # The completed levels have already been paid for; hand them back
            # rather than making the ceiling destroy what it was protecting.
            exc.partial_result = self._build_result(
                last_output=_last_output_of(results),
                results=results,
                run_id=run_id,
                levels_executed=_levels_completed(results),
                wall_start=wall_start,
            )
            raise

        pipeline_result = self._build_result(
            last_output=last_output,
            results=results,
            run_id=run_id,
            levels_executed=levels_executed,
            wall_start=wall_start,
            pause=pause,
            session_id=session_id,
        )
        if pause is None and self._hooks is not None:
            safe_invoke(self._hooks, "on_pipeline_end", pipeline_result)
        return pipeline_result

    async def resume(
        self,
        session_id: str,
        human_feedback: str,
        approved: bool,
        *,
        on_in_doubt: InDoubt = "raise",
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
                      reject it and feed *human_feedback* to the agent. A rejection
                      also stops the rest of that reply's tool calls. Approving one
                      call never approves the next: a later blocked call pauses again.
            on_in_doubt: What to do when an earlier resume started this approved
                      call and never recorded its result (a crash). ``"raise"``
                      (default) raises :class:`EffectUncertainError`; ``"skip"``
                      continues without running it; ``"retry"`` runs it again.

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

        if on_in_doubt not in ("raise", "retry", "skip"):
            raise ValueError(f"on_in_doubt must be 'raise', 'retry' or 'skip', got {on_in_doubt!r}")
        if session_id in self._resuming:
            raise PipelineError(f"resume() is already running for session '{session_id}'")
        self._resuming.add(session_id)
        try:
            paused = await self._load_paused(session_id)
            state = paused.state
            discarded = state.get("discarded_spend", (0.0, 0))
            spent_before = discarded[0] + sum(r["cost"] for r in state["results"].values())
            scope = self._new_scope(state["run_id"], spent=spent_before)
            scope.discarded.add(*discarded)
            with run_scope(scope):
                return await self._resume(
                    _Decision(session_id, human_feedback, approved, on_in_doubt), paused, wall_start
                )
        finally:
            self._resuming.discard(session_id)

    async def _resume(self, decision: _Decision, paused: _PausedRun, wall_start: float) -> PipelineResult:
        assert self._memory is not None  # checked by resume()
        session_id = decision.session_id
        state = paused.state
        task: str = state["task"]
        run_id: str = state["run_id"]
        current_level: int = state["current_level"]
        context: dict[str, Any] = state["context"]
        pause_data = paused.pauses[0]
        paused_name: str = pause_data["agent_name"]

        results: dict[str, AgentResult] = {}
        for name, data in state["results"].items():
            results[name] = AgentResult(**data)
        last_output = next(reversed(results.values())).output if results else ""

        # Locate the paused agent in the pipeline.
        agent: AgentSpec | None = None
        for node in self._nodes:
            if node.agent.name == paused_name and hasattr(node.agent, "resume_execution"):
                agent = node.agent  # type: ignore[assignment]
                break

        if agent is None:
            raise PipelineError(
                f"Paused agent '{paused_name}' not found in pipeline or "
                "does not support resume_execution"
            )

        prior_output, on_executed = await self._claim_approved_call(decision, paused)
        # Resume the agent with run-scoped state — never mutate the shared
        # instance (concurrent pipeline runs may share it).
        spend = Spend()
        try:
            with tracking(spend), using_effects(self._effects_for(paused_name)):
                agent_result = await agent.resume_execution(
                    pause_data,
                    self._llm,
                    decision.approved,
                    decision.feedback,
                    session_id=session_id,
                    approval_policy=self._approval_policy,
                    prior_output=prior_output,
                    on_executed=on_executed,
                )
        except PauseExecution as again:
            # The same agent needs another approval: the next blocked call of the
            # batch, or a later call. Its earlier calls are already recorded.
            paused.pauses[0] = again.as_dict()
            await self._save_paused(session_id, paused, results, context)
            return self._build_result(
                last_output=last_output, results=results, run_id=run_id,
                levels_executed=current_level, wall_start=wall_start, pause=again,
                session_id=session_id,
            )
        except BaseException:
            _discard(spend)
            raise
        results[paused_name] = agent_result
        context[paused_name] = (
            agent_result.data if agent_result.data is not None else agent_result.output
        )
        last_output = agent_result.output

        # A sibling paused in the same level is resolved before the level is done (HITL-4).
        if len(paused.pauses) > 1:
            paused.pauses = paused.pauses[1:]
            await self._save_paused(session_id, paused, results, context)
            return self._build_result(
                last_output=last_output, results=results, run_id=run_id,
                levels_executed=current_level, wall_start=wall_start,
                pause=PauseExecution.from_dict(paused.pauses[0]), session_id=session_id,
            )

        # Continue with remaining pipeline levels (supports nested pauses).
        levels = self._resolve_levels()
        try:
            tail_output, pause, levels_executed = await self._run_levels(
                levels=levels,
                start_level=current_level + 1,
                task=task,
                context=context,
                results=results,
                session_id=session_id,
                run_id=run_id,
            )
        except BudgetExceededError as exc:
            exc.partial_result = self._build_result(
                last_output=_last_output_of(results),
                results=results,
                run_id=run_id,
                levels_executed=_levels_completed(results),
                wall_start=wall_start,
            )
            raise
        last_output = tail_output or last_output

        if pause is not None:
            return self._build_result(
                last_output=last_output,
                results=results,
                run_id=run_id,
                levels_executed=levels_executed,
                wall_start=wall_start,
                pause=pause,
                session_id=session_id,
            )

        await self._clear_paused(session_id)

        return self._build_result(
            last_output=last_output,
            results=results,
            run_id=run_id,
            levels_executed=len(levels),
            wall_start=wall_start,
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
            run_id = new_run_id()
            with run_scope(self._new_scope(run_id)):
                await _run_scoped(run_id)

        async def _run_scoped(run_id: str) -> None:
            session_id = self._session_id or run_id
            try:
                _, pause, _ = await self._run_levels(
                    levels=levels,
                    start_level=0,
                    task=task,
                    context=context,
                    results=results,
                    session_id=session_id,
                    run_id=run_id,
                    emitter=emitter,
                )

                if pause is not None:
                    emitter.emit(
                        "pipeline_paused",
                        agent=pause.agent_name,
                        tool_name=pause.tool_name,
                        tool_arguments=pause.tool_arguments,
                        session_id=session_id,
                    )
                    return

                total_tokens, total_cost = _run_totals(results)
                emitter.emit(
                    "pipeline_complete",
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    total_duration=round(sum(r.duration for r in results.values()), 3),
                    wall_time=round(time.perf_counter() - wall_start, 3),
                    agents_completed=len(results),
                    levels_executed=len(levels),
                    unpriced_models=_unpriced_models_used(results),
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

        When a trigger yields a non-empty ``context_data`` dict, it is
        appended to the task prompt as a JSON block so the payload reaches
        the agents.

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
            # Trigger payloads travel with the task — otherwise they'd be lost.
            if context_data:
                task_prompt = (
                    f"{task_prompt}\n\nContext data (JSON):\n{json.dumps(context_data)}"
                )
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
