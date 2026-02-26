"""Tests for the Agent decorator and BaseAgent."""

import pytest
from agentflow import Agent, BaseAgent, AgentResult
from agentflow.agent import _DecoratorAgent


def test_agent_decorator_creates_decorator_agent():
    @Agent(name="test", role="Tester")
    async def test_agent(task: str, context: dict) -> str:
        return f"testing: {task}"

    assert isinstance(test_agent, _DecoratorAgent)
    assert test_agent.name == "test"
    assert test_agent.role == "Tester"


def test_agent_repr():
    @Agent(name="myagent", role="My Role")
    async def my_agent(task: str, context: dict) -> str:
        return task

    assert "myagent" in repr(my_agent)
    assert "My Role" in repr(my_agent)


@pytest.mark.asyncio
async def test_decorator_agent_calls_prompt_fn():
    calls = []

    @Agent(name="tracker", role="Tracker")
    async def tracker(task: str, context: dict) -> str:
        calls.append((task, context))
        return f"prompt for: {task}"

    # We can't test execute() without a real LLM, but we can test the prompt fn
    prompt = await tracker._prompt_fn("hello", {"prev": "data"})
    assert prompt == "prompt for: hello"
    assert calls == [("hello", {"prev": "data"})]


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent("test", "role")  # Can't instantiate abstract class


def test_base_agent_subclass():
    class MyAgent(BaseAgent):
        async def execute(self, task, context, llm):
            return AgentResult(agent=self.name, output="done")

    agent = MyAgent("custom", "Custom Role")
    assert agent.name == "custom"
    assert agent.role == "Custom Role"
