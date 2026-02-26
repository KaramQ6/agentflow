"""Pipeline orchestrator for multi-agent execution."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from .agent import _DecoratorAgent, BaseAgent
from .events import EventEmitter
from .exceptions import PipelineError, AgentError
from .llm import LLM
from .types import AgentResult, PipelineResult, Event


AgentLike = _DecoratorAgent | BaseAgent


class _PipelineNode:
    """Internal node in the pipeline graph."""

    def __init__(self, agent: AgentLike, depends_on: list[str]):
        self.agent = agent
        self.depends_on = depends_on


class Pipeline:
    """Multi-agent pipeline with dependency resolution.

    Args:
        llm: The LLM provider for all agents.

    Usage:
        pipe = Pipeline(llm=llm)
        pipe.add(researcher)
        pipe.add(writer, depends_on=["researcher"])
        result = await pipe.run("AI in Healthcare")
    """

    def __init__(self, llm: LLM):
        self._llm = llm
        self._nodes: list[_PipelineNode] = []
        self._agent_names: set[str] = set()

    def add(self, agent: AgentLike, depends_on: list[str] | None = None) -> "Pipeline":
        """Add an agent to the pipeline.

        Args:
            agent: An @Agent-decorated function or BaseAgent subclass instance.
            depends_on: List of agent names this agent depends on.

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

        self._nodes.append(_PipelineNode(agent, deps))
        self._agent_names.add(name)
        return self

    def _resolve_order(self) -> list[_PipelineNode]:
        """Topological sort of the pipeline graph."""
        resolved: list[_PipelineNode] = []
        seen: set[str] = set()
        node_map = {n.agent.name: n for n in self._nodes}

        def visit(name: str) -> None:
            if name in seen:
                return
            node = node_map[name]
            for dep in node.depends_on:
                if dep not in seen:
                    visit(dep)
            seen.add(name)
            resolved.append(node)

        for node in self._nodes:
            visit(node.agent.name)

        return resolved

    async def run(self, task: str) -> PipelineResult:
        """Execute the pipeline sequentially.

        Args:
            task: The task string passed to each agent.

        Returns:
            PipelineResult with all agent results.
        """
        ordered = self._resolve_order()
        results: dict[str, AgentResult] = {}
        context: dict[str, str] = {}

        for node in ordered:
            agent = node.agent
            result = await agent.execute(task, context, self._llm)
            results[agent.name] = result
            context[agent.name] = result.output

        last_output = results[ordered[-1].agent.name].output if ordered else ""
        total_tokens = sum(r.tokens_used for r in results.values())
        total_duration = sum(r.duration for r in results.values())

        return PipelineResult(
            output=last_output,
            results=results,
            total_tokens=total_tokens,
            total_duration=round(total_duration, 3),
        )

    async def stream(self, task: str) -> AsyncGenerator[Event, None]:
        """Execute the pipeline and yield events.

        Yields Event objects as agents start, complete, or error.
        The final event has type "pipeline_complete".
        """
        emitter = EventEmitter()
        ordered = self._resolve_order()
        results: dict[str, AgentResult] = {}
        context: dict[str, str] = {}

        async def _run() -> None:
            try:
                for node in ordered:
                    agent = node.agent
                    emitter.emit("agent_start", agent=agent.name, role=agent.role)

                    try:
                        result = await agent.execute(task, context, self._llm)
                        results[agent.name] = result
                        context[agent.name] = result.output
                        emitter.emit(
                            "agent_complete",
                            agent=agent.name,
                            tokens=result.tokens_used,
                            duration=result.duration,
                            output_preview=result.output[:200],
                        )
                    except AgentError as e:
                        emitter.emit("agent_error", agent=agent.name, error=str(e))
                        raise

                total_tokens = sum(r.tokens_used for r in results.values())
                total_duration = sum(r.duration for r in results.values())
                emitter.emit(
                    "pipeline_complete",
                    total_tokens=total_tokens,
                    total_duration=round(total_duration, 3),
                    agents_completed=len(results),
                )
            except Exception as e:
                emitter.emit("pipeline_error", error=str(e))
            finally:
                emitter.done()

        run_task = asyncio.create_task(_run())

        async for event in emitter.stream():
            yield event

        await run_task
