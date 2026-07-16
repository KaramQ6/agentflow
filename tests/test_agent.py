"""Tests for the Agent decorator and BaseAgent."""

import pytest

from agentflow import Agent, AgentResult, AgentSpec, BaseAgent, LLMResponse


def test_agent_decorator_creates_agent_spec():
    @Agent(name="test", role="Tester")
    async def test_agent(task: str, context: dict) -> str:
        return f"testing: {task}"

    assert isinstance(test_agent, AgentSpec)
    assert test_agent.name == "test"
    assert test_agent.role == "Tester"


def test_decorator_agent_alias_still_works():
    """Pre-0.6 private name remains importable until 1.0."""
    from agentflow.agent import _DecoratorAgent

    assert _DecoratorAgent is AgentSpec


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


# ── 0.6 regressions ────────────────────────────────────────────────────────────


class RecordingLLM:
    """Stub LLM that records messages and returns a fixed LLMResponse."""

    model = "fake-model"

    def __init__(self):
        self.messages: list[list[dict]] = []

    async def generate(self, messages, tools=None, **kwargs):
        self.messages.append([dict(m) for m in messages])
        return LLMResponse(content="ok", tokens=5, model="fake-model")


def test_set_session_warns_deprecation():
    @Agent(name="warner", role="Warner")
    async def warner(task: str, context: dict) -> str:
        return task

    with pytest.warns(DeprecationWarning, match="set_session"):
        warner.set_session("s1")
    with pytest.warns(DeprecationWarning, match="set_approval_policy"):
        warner.set_approval_policy(None)


@pytest.mark.asyncio
async def test_custom_system_prompt_replaces_default():
    @Agent(name="custom_sys", role="Ignored Role", system_prompt="You are a pirate.")
    async def custom_sys(task: str, context: dict) -> str:
        return task

    llm = RecordingLLM()
    await custom_sys.execute("ahoy", {}, llm)

    system_msg = llm.messages[0][0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == "You are a pirate."


@pytest.mark.asyncio
async def test_default_system_prompt_uses_role():
    @Agent(name="default_sys", role="Data Analyst")
    async def default_sys(task: str, context: dict) -> str:
        return task

    llm = RecordingLLM()
    await default_sys.execute("go", {}, llm)

    assert "You are a Data Analyst." in llm.messages[0][0]["content"]
