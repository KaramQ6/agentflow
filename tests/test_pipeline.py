"""Tests for the Pipeline orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentflow import Agent, Pipeline, AgentResult, PipelineResult
from agentflow.agent import _DecoratorAgent, BaseAgent
from agentflow.exceptions import PipelineError
from agentflow.llm import LLM


class MockLLM:
    """Mock LLM that returns predictable responses."""

    async def generate(self, messages, **kwargs):
        user_msg = messages[-1]["content"] if messages else ""
        return {
            "content": f"LLM response to: {user_msg[:50]}",
            "tokens": 100,
            "duration": 0.5,
            "model": "mock-model",
        }


class SimpleAgent(BaseAgent):
    """Simple agent for testing."""

    def __init__(self, name: str):
        super().__init__(name=name, role=f"{name} role")

    async def execute(self, task, context, llm):
        prev = ", ".join(f"{k}={v[:20]}" for k, v in context.items()) if context else "none"
        response = await llm.generate([
            {"role": "user", "content": f"{self.name}: {task} (context: {prev})"},
        ])
        return AgentResult(
            agent=self.name,
            output=response["content"],
            tokens_used=response["tokens"],
            duration=response["duration"],
        )


def test_pipeline_add_agents():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    a1 = SimpleAgent("agent1")
    a2 = SimpleAgent("agent2")

    pipe.add(a1)
    pipe.add(a2, depends_on=["agent1"])

    assert len(pipe._nodes) == 2


def test_pipeline_duplicate_name_raises():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    a1 = SimpleAgent("agent1")
    a2 = SimpleAgent("agent1")

    pipe.add(a1)
    with pytest.raises(PipelineError, match="Duplicate"):
        pipe.add(a2)


def test_pipeline_missing_dependency_raises():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    a1 = SimpleAgent("agent1")

    with pytest.raises(PipelineError, match="hasn't been added"):
        pipe.add(a1, depends_on=["nonexistent"])


def test_pipeline_chaining():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    a1 = SimpleAgent("a")
    a2 = SimpleAgent("b")

    result = pipe.add(a1).add(a2, depends_on=["a"])
    assert result is pipe


@pytest.mark.asyncio
async def test_pipeline_run():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)

    a1 = SimpleAgent("first")
    a2 = SimpleAgent("second")

    pipe.add(a1)
    pipe.add(a2, depends_on=["first"])

    result = await pipe.run("test task")

    assert isinstance(result, PipelineResult)
    assert "first" in result.results
    assert "second" in result.results
    assert result.total_tokens == 200  # 100 per agent
    assert result.output == result.results["second"].output


@pytest.mark.asyncio
async def test_pipeline_context_passing():
    llm = MockLLM()

    received_contexts = []

    class ContextAgent(BaseAgent):
        async def execute(self, task, context, llm):
            received_contexts.append(dict(context))
            return AgentResult(
                agent=self.name,
                output=f"output-from-{self.name}",
                tokens_used=10,
                duration=0.1,
            )

    pipe = Pipeline(llm=llm)
    pipe.add(ContextAgent("a", "role_a"))
    pipe.add(ContextAgent("b", "role_b"), depends_on=["a"])
    pipe.add(ContextAgent("c", "role_c"), depends_on=["a", "b"])

    await pipe.run("test")

    assert received_contexts[0] == {}  # 'a' has no deps
    assert received_contexts[1] == {"a": "output-from-a"}  # 'b' depends on 'a'
    assert received_contexts[2] == {"a": "output-from-a", "b": "output-from-b"}  # 'c' depends on both


@pytest.mark.asyncio
async def test_pipeline_stream():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    pipe.add(SimpleAgent("only"))

    events = []
    async for event in pipe.stream("test"):
        events.append(event)

    types = [e.type for e in events]
    assert "agent_start" in types
    assert "agent_complete" in types
    assert "pipeline_complete" in types


@pytest.mark.asyncio
async def test_pipeline_get_result():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    pipe.add(SimpleAgent("alpha"))

    result = await pipe.run("test")
    agent_result = result.get("alpha")
    assert agent_result is not None
    assert agent_result.agent == "alpha"

    assert result.get("nonexistent") is None
