"""Tests for parallel pipeline execution and DAG level resolution."""

import asyncio
import time

import pytest
from agentflow import AgentResult, Pipeline
from agentflow.agent import BaseAgent
from agentflow.exceptions import AgentError, AgentTimeoutError


class MockLLM:
    async def generate(self, messages, **kwargs):
        user_msg = messages[-1]["content"] if messages else ""
        return {
            "content": f"LLM response to: {user_msg[:50]}",
            "tokens": 100,
            "duration": 0.05,
            "model": "mock-model",
            "cached": False,
        }


class SlowAgent(BaseAgent):
    """Agent that sleeps for a fixed duration — used to measure parallelism."""

    def __init__(self, name: str, sleep_s: float):
        super().__init__(name=name, role=f"{name} role")
        self._sleep_s = sleep_s

    async def execute(self, task, context, llm):
        await asyncio.sleep(self._sleep_s)
        return AgentResult(agent=self.name, output=f"output-{self.name}", tokens_used=10, duration=self._sleep_s)


class ContextCapturingAgent(BaseAgent):
    """Records contexts it receives at execute time."""

    def __init__(self, name: str, store: list):
        super().__init__(name=name, role="recorder")
        self._store = store

    async def execute(self, task, context, llm):
        self._store.append((self.name, dict(context)))
        return AgentResult(agent=self.name, output=f"out-{self.name}", tokens_used=0, duration=0.0)


# ─── Level Resolution ──────────────────────────────────────────────────────────

def test_resolve_levels_single_chain():
    """A → B → C should produce 3 levels each with one agent."""
    pipe = Pipeline(llm=MockLLM())
    for name, deps in [("a", []), ("b", ["a"]), ("c", ["b"])]:
        pipe.add(SlowAgent(name, 0.0), depends_on=deps)

    levels = pipe._resolve_levels()
    assert len(levels) == 3
    assert [n.agent.name for n in levels[0]] == ["a"]
    assert [n.agent.name for n in levels[1]] == ["b"]
    assert [n.agent.name for n in levels[2]] == ["c"]


def test_resolve_levels_diamond_dag():
    """Diamond: A → [B, C] → D should produce 3 levels, B+C in 1 parallel level."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("a", 0.0))
    pipe.add(SlowAgent("b", 0.0), depends_on=["a"])
    pipe.add(SlowAgent("c", 0.0), depends_on=["a"])
    pipe.add(SlowAgent("d", 0.0), depends_on=["b", "c"])

    levels = pipe._resolve_levels()
    assert len(levels) == 3
    assert {n.agent.name for n in levels[0]} == {"a"}
    assert {n.agent.name for n in levels[1]} == {"b", "c"}
    assert {n.agent.name for n in levels[2]} == {"d"}


def test_resolve_levels_fully_parallel():
    """Three independent agents should all land in level 0."""
    pipe = Pipeline(llm=MockLLM())
    for name in ["x", "y", "z"]:
        pipe.add(SlowAgent(name, 0.0))

    levels = pipe._resolve_levels()
    assert len(levels) == 1
    assert {n.agent.name for n in levels[0]} == {"x", "y", "z"}


# ─── Parallel Timing ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_agents_faster_than_sequential():
    """Two independent 0.1s agents should finish in ~0.1s, not ~0.2s."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("p1", 0.1))
    pipe.add(SlowAgent("p2", 0.1))

    start = time.perf_counter()
    result = await pipe.run("task")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.18, f"Expected parallel (~0.1s) but took {elapsed:.3f}s"
    assert "p1" in result.results
    assert "p2" in result.results


@pytest.mark.asyncio
async def test_sequential_requires_full_duration():
    """Two dependent 0.1s agents should take ~0.2s (sequential)."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("s1", 0.1))
    pipe.add(SlowAgent("s2", 0.1), depends_on=["s1"])

    start = time.perf_counter()
    await pipe.run("task")
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.18, f"Expected sequential (~0.2s) but took {elapsed:.3f}s"


# ─── Context Isolation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_agents_do_not_see_each_others_context():
    """Agents in the same level must not receive each other's outputs."""
    received: list[tuple[str, dict]] = []

    pipe = Pipeline(llm=MockLLM())
    pipe.add(ContextCapturingAgent("a", received))
    pipe.add(ContextCapturingAgent("b", received))
    pipe.add(ContextCapturingAgent("c", received), depends_on=["a", "b"])

    await pipe.run("task")

    ctx_a = next(ctx for name, ctx in received if name == "a")
    ctx_b = next(ctx for name, ctx in received if name == "b")
    assert ctx_a == {}
    assert ctx_b == {}


@pytest.mark.asyncio
async def test_agent_receives_only_declared_dependency_context():
    """Agent B that depends on A (not C) should only see A's output, not C's."""
    received: list[tuple[str, dict]] = []

    pipe = Pipeline(llm=MockLLM())
    pipe.add(ContextCapturingAgent("a", received))
    pipe.add(ContextCapturingAgent("c", received))
    # B is defined second-level but only depends on A
    pipe.add(ContextCapturingAgent("b", received), depends_on=["a"])
    pipe.add(ContextCapturingAgent("d", received), depends_on=["b", "c"])

    await pipe.run("task")

    ctx_b = next(ctx for name, ctx in received if name == "b")
    assert "a" in ctx_b
    assert "c" not in ctx_b


# ─── Timeout ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeout_raises_agent_timeout_error():
    """An agent exceeding its timeout must raise AgentTimeoutError."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("slow", sleep_s=0.5), timeout=0.05)

    with pytest.raises(AgentTimeoutError) as exc_info:
        await pipe.run("task")

    assert exc_info.value.agent_name == "slow"
    assert exc_info.value.timeout_seconds == 0.05


@pytest.mark.asyncio
async def test_timeout_not_raised_when_agent_finishes_in_time():
    """A generous timeout must not interfere with normal execution."""
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("fast", sleep_s=0.01), timeout=1.0)

    result = await pipe.run("task")
    assert "fast" in result.results


# ─── Pipeline Retry ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_failed_agents_succeeds_on_second_attempt():
    """An agent that fails once should succeed on retry."""
    attempts = {"count": 0}

    class FlakyAgent(BaseAgent):
        async def execute(self, task, context, llm):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise AgentError(self.name, "transient failure")
            return AgentResult(agent=self.name, output="recovered", tokens_used=5, duration=0.0)

    pipe = Pipeline(llm=MockLLM(), retry_failed_agents=1)
    pipe.add(FlakyAgent("flaky", "Flaky Agent"))

    result = await pipe.run("task")
    assert result.output == "recovered"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_retry_exhausted_raises_original_error():
    """An agent that always fails should raise after all retries are exhausted."""

    class AlwaysFailAgent(BaseAgent):
        async def execute(self, task, context, llm):
            raise AgentError(self.name, "permanent failure")

    pipe = Pipeline(llm=MockLLM(), retry_failed_agents=1)
    pipe.add(AlwaysFailAgent("broken", "Broken Agent"))

    with pytest.raises(AgentError):
        await pipe.run("task")


# ─── Level Metadata ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_result_contains_levels_executed():
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("a", 0.0))
    pipe.add(SlowAgent("b", 0.0), depends_on=["a"])

    result = await pipe.run("task")
    assert result.levels_executed == 2


@pytest.mark.asyncio
async def test_agent_result_has_correct_level():
    pipe = Pipeline(llm=MockLLM())
    pipe.add(SlowAgent("a", 0.0))
    pipe.add(SlowAgent("b", 0.0), depends_on=["a"])

    result = await pipe.run("task")
    assert result.results["a"].level == 0
    assert result.results["b"].level == 1
