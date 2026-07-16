"""Tests for the memory module and agent-memory integration."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentflow import Agent, BaseMemory, InMemoryContext, LLMResponse, Pipeline

# ─── Test utilities ────────────────────────────────────────────────────────────

def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _response(content: str, tool_calls=None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tokens=10,
        prompt_tokens=6,
        completion_tokens=4,
        duration=0.0,
        model="fake-model",
        cached=False,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
    )


class ScriptedLLM:
    """LLM stub that returns queued responses."""

    model = "fake-model"

    def __init__(self, responses: list[dict]):
        self._responses = responses

    async def generate(self, messages, tools=None, **kwargs):
        return self._responses.pop(0)


# ─── Unit tests: InMemoryContext ────────────────────────────────────────────────

class TestInMemoryContext:
    def test_save_and_load(self):
        mem = InMemoryContext()

        async def _run():
            await mem.save_context("s1", "agent_a", "output A")
            await mem.save_context("s1", "agent_b", "output B")
            ctx = await mem.load_context("s1")
            assert ctx == {"agent_a": "output A", "agent_b": "output B"}

        asyncio.run(_run())

    def test_load_missing_session_returns_empty(self):
        mem = InMemoryContext()

        async def _run():
            ctx = await mem.load_context("nonexistent")
            assert ctx == {}

        asyncio.run(_run())

    def test_session_isolation(self):
        mem = InMemoryContext()

        async def _run():
            await mem.save_context("s1", "key", "val1")
            await mem.save_context("s2", "key", "val2")
            assert await mem.load_context("s1") == {"key": "val1"}
            assert await mem.load_context("s2") == {"key": "val2"}

        asyncio.run(_run())

    def test_clear_removes_all_entries(self):
        mem = InMemoryContext()

        async def _run():
            await mem.save_context("s1", "a", "1")
            await mem.save_context("s1", "b", "2")
            await mem.clear("s1")
            assert await mem.load_context("s1") == {}

        asyncio.run(_run())

    def test_delete_key(self):
        mem = InMemoryContext()

        async def _run():
            await mem.save_context("s1", "a", "1")
            await mem.save_context("s1", "b", "2")
            await mem.delete_key("s1", "a")
            assert await mem.load_context("s1") == {"b": "2"}
            await mem.delete_key("s1", "b")
            assert await mem.load_context("s1") == {}

        asyncio.run(_run())

    def test_ttl_expiry(self):
        mem = InMemoryContext(default_ttl=0.05)

        async def _run():
            await mem.save_context("s1", "ephemeral", "soon gone")
            assert await mem.load_context("s1") == {"ephemeral": "soon gone"}
            await asyncio.sleep(0.1)
            assert await mem.load_context("s1") == {}

        asyncio.run(_run())

    def test_ttl_does_not_expire_fresh_entries(self):
        mem = InMemoryContext(default_ttl=60.0)

        async def _run():
            await mem.save_context("s1", "fresh", "value")
            await asyncio.sleep(0.01)
            assert await mem.load_context("s1") == {"fresh": "value"}

        asyncio.run(_run())

    def test_lru_eviction(self):
        mem = InMemoryContext(max_entries=3)

        async def _run():
            await mem.save_context("s1", "a", "1")
            await mem.save_context("s1", "b", "2")
            await mem.save_context("s1", "c", "3")
            await mem.save_context("s1", "d", "4")  # should evict "a"
            ctx = await mem.load_context("s1")
            assert "a" not in ctx
            assert ctx == {"b": "2", "c": "3", "d": "4"}

        asyncio.run(_run())

    def test_lru_touch_moves_to_end(self):
        mem = InMemoryContext(max_entries=3)

        async def _run():
            await mem.save_context("s1", "a", "1")
            await mem.save_context("s1", "b", "2")
            await mem.save_context("s1", "c", "3")
            # Re-save "a" to touch it (move to MRU end)
            await mem.save_context("s1", "a", "1")
            await mem.save_context("s1", "d", "4")  # should evict "b", not "a"
            ctx = await mem.load_context("s1")
            assert "b" not in ctx
            assert ctx == {"c": "3", "a": "1", "d": "4"}

        asyncio.run(_run())

    def test_abstract_base_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseMemory()  # type: ignore[abstract]


# ─── Integration tests: Agent + Memory ─────────────────────────────────────────

class TestAgentMemoryIntegration:
    def test_agent_saves_to_memory(self):
        mem = InMemoryContext()

        @Agent(name="test_agent", role="Tester", memory=mem)
        async def test_agent(task: str, context: dict) -> str:
            return f"processed: {task}"

        async def _run():
            agent = test_agent
            llm = ScriptedLLM([_response("processed: hello")])
            await agent.execute("hello", {}, llm, session_id="integration_test")
            ctx = await mem.load_context("integration_test")
            assert "test_agent" in ctx
            assert "processed: hello" in ctx["test_agent"]

        asyncio.run(_run())

    def test_agent_without_memory_does_not_save(self):
        @Agent(name="no_mem", role="Tester")
        async def no_mem(task: str, context: dict) -> str:
            return f"done: {task}"

        async def _run():
            agent = no_mem
            llm = ScriptedLLM([_response("done: test")])
            result = await agent.execute("test", {}, llm)
            assert result.output == "done: test"

        asyncio.run(_run())

    def test_downstream_agent_receives_memory_context(self):
        mem = InMemoryContext()

        @Agent(name="agent_a", role="First", memory=mem)
        async def agent_a(task: str, context: dict) -> str:
            return "data from A"

        @Agent(name="agent_b", role="Second", memory=mem)
        async def agent_b(task: str, context: dict) -> str:
            return "processed B with context"

        async def _run():
            a = agent_a
            b = agent_b
            with pytest.warns(DeprecationWarning):
                a.set_session("shared")
            with pytest.warns(DeprecationWarning):
                b.set_session("shared")
            await a.execute("task", {}, ScriptedLLM([_response("data from A")]))
            await b.execute("task", {}, ScriptedLLM([_response("processed B with context")]))

        asyncio.run(_run())

    def test_pipeline_with_memory_shares_context(self):
        mem = InMemoryContext()

        @Agent(name="first_agent", role="Producer", memory=mem)
        async def first_agent(task: str, context: dict) -> str:
            return f"step1: {task}"

        @Agent(name="second_agent", role="Consumer", memory=mem)
        async def second_agent(task: str, context: dict) -> str:
            return f"step2 got: {context.get('first_agent', 'missing')}"

        pipe = Pipeline(
            llm=ScriptedLLM([
                _response("step1: demo task"),
                _response("step2 got: step1: demo task"),
            ]),
            memory=mem,
        )
        pipe.add(first_agent)
        pipe.add(second_agent, depends_on=["first_agent"])

        async def _run():
            result = await pipe.run("demo task")
            assert "step1: demo task" in result.results["first_agent"].output
            assert "step2 got: step1: demo task" in result.results["second_agent"].output

        asyncio.run(_run())


# ─── Concurrency safety ────────────────────────────────────────────────────────

class TestConcurrency:
    def test_concurrent_saves_dont_corrupt(self):
        mem = InMemoryContext()

        async def _save(session: str, key: str, val: str):
            await mem.save_context(session, key, val)

        async def _run():
            tasks = [
                _save("s1", f"k{i}", f"v{i}") for i in range(50)
            ]
            await asyncio.gather(*tasks)
            ctx = await mem.load_context("s1")
            assert len(ctx) == 50
            for i in range(50):
                assert ctx[f"k{i}"] == f"v{i}"

        asyncio.run(_run())
