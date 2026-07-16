"""Tests for SupervisorAgent swarm routing."""

import json

import pytest

from agentflow import Agent, AgentResult, BaseAgent, LLMResponse
from agentflow.exceptions import AgentError
from agentflow.llm import LLM
from agentflow.swarm import SupervisorAgent

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _response(content: str, tool_calls: list | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tokens=10,
        prompt_tokens=6,
        completion_tokens=4,
        duration=0.0,
        model="fake-model",
        cached=False,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
    )


class ScriptedLLM:
    """LLM stub that returns queued responses and records requests."""

    model = "fake-model"

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.requests: list[dict] = []

    async def generate(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self._responses.pop(0)


class DummyWorker(BaseAgent):
    """A minimal worker agent that returns a fixed output."""

    def __init__(self, name: str, role: str, output: str, tokens: int = 5, cost: float = 0.001):
        super().__init__(name=name, role=role)
        self._output = output
        self._tokens = tokens
        self._cost = cost
        self.last_task: str | None = None

    async def execute(self, task: str, context: dict[str, str], llm: LLM) -> AgentResult:
        self.last_task = task
        return AgentResult(
            agent=self.name,
            output=self._output,
            tokens_used=self._tokens,
            cost=self._cost,
            duration=0.01,
        )


# --------------------------------------------------------------------------- #
# Construction & basic properties
# --------------------------------------------------------------------------- #


def test_supervisor_construction():
    w1 = DummyWorker("researcher", "Researcher", "research output")
    w2 = DummyWorker("writer", "Writer", "written content")

    sup = SupervisorAgent("manager", "Project Manager", workers=[w1, w2])

    assert sup.name == "manager"
    assert sup.role == "Project Manager"
    assert set(sup._workers.keys()) == {"researcher", "writer"}


def test_supervisor_construction_empty_workers():
    sup = SupervisorAgent("solo", "Solo Agent", workers=[])
    assert sup._workers == {}


def test_supervisor_default_max_iterations():
    from agentflow.agent import DEFAULT_MAX_TOOL_ITERATIONS

    sup = SupervisorAgent("mgr", "Manager", workers=[])
    assert sup._max_tool_iterations == DEFAULT_MAX_TOOL_ITERATIONS


def test_supervisor_custom_max_iterations():
    sup = SupervisorAgent("mgr", "Manager", workers=[], max_tool_iterations=3)
    assert sup._max_tool_iterations == 3


# --------------------------------------------------------------------------- #
# Single delegation — supervisor delegates then answers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_single_delegation():
    worker = DummyWorker("researcher", "Research Analyst", "The answer is 42.", tokens=7, cost=0.002)
    sup = SupervisorAgent("manager", "Manager", workers=[worker])

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "researcher", "sub_task": "Find the answer"})]),
            _response("Based on research, the answer is 42."),
        ]
    )

    result = await sup.execute("What is the answer?", {}, llm)

    assert result.output == "Based on research, the answer is 42."
    assert result.agent == "manager"
    # Supervisor tokens (10+10) + worker tokens (7) = 27
    assert result.tokens_used == 27
    # Supervisor cost (0 + 0) + worker cost (0.002) = 0.002
    assert result.cost == 0.002

    # Worker trace
    worker_trace = result.metadata["worker_delegations"]
    assert len(worker_trace) == 1
    assert worker_trace[0]["worker"] == "researcher"
    assert worker_trace[0]["output"] == "The answer is 42."
    assert worker_trace[0]["tokens_used"] == 7
    assert worker_trace[0]["cost"] == 0.002

    # Worker received the sub-task
    assert worker.last_task == "Find the answer"


# --------------------------------------------------------------------------- #
# Multiple delegations — token/cost accumulation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_multiple_delegations_bubbles_cost():
    w1 = DummyWorker("researcher", "Researcher", "Fact A.", tokens=5, cost=0.001)
    w2 = DummyWorker("analyst", "Analyst", "Insight B.", tokens=8, cost=0.003)
    sup = SupervisorAgent("manager", "Manager", workers=[w1, w2])

    llm = ScriptedLLM(
        [
            _response("", [
                _tool_call("c1", "delegate_task", {"worker_name": "researcher", "sub_task": "Research"}),
                _tool_call("c2", "delegate_task", {"worker_name": "analyst", "sub_task": "Analyze"}),
            ]),
            _response("Combined: Fact A. Insight B."),
        ]
    )

    result = await sup.execute("Task", {}, llm)

    assert result.output == "Combined: Fact A. Insight B."
    # Supervisor: 2 rounds x 10 tokens = 20; workers: 5 + 8 = 13; total = 33
    assert result.tokens_used == 33
    # Supervisor cost: 0 (mock returns no cost); workers: 0.001 + 0.003 = 0.004
    assert result.cost == 0.004

    worker_trace = result.metadata["worker_delegations"]
    assert len(worker_trace) == 2
    assert result.metadata["accumulated_worker_tokens"] == 13
    assert result.metadata["accumulated_worker_cost"] == 0.004


# --------------------------------------------------------------------------- #
# No delegation — supervisor answers directly
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_no_delegation_direct_answer():
    sup = SupervisorAgent("manager", "Manager", workers=[DummyWorker("w", "Worker", "x")])

    llm = ScriptedLLM([_response("I can handle this directly.")])

    result = await sup.execute("Simple question", {}, llm)

    assert result.output == "I can handle this directly."
    assert result.tokens_used == 10
    assert result.metadata["worker_delegations"] == []


# --------------------------------------------------------------------------- #
# Self-delegation prevention
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_cannot_delegate_to_self():
    worker = DummyWorker("researcher", "Researcher", "output")
    sup = SupervisorAgent("manager", "Manager", workers=[worker])

    # LLM tries to delegate to the supervisor itself
    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "manager", "sub_task": "x"})]),
            _response("I'll handle it myself then. Here is the answer."),
        ]
    )

    result = await sup.execute("Task", {}, llm)

    assert result.output == "I'll handle it myself then. Here is the answer."
    # Self-delegation should have produced an error, captured in the trace
    supervisor_trace = result.metadata["tool_calls"]
    assert len(supervisor_trace) == 1
    assert "Cannot delegate to yourself" in supervisor_trace[0]["result"]


# --------------------------------------------------------------------------- #
# Unknown worker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_unknown_worker_error():
    sup = SupervisorAgent("manager", "Manager", workers=[DummyWorker("a", "A", "x")])

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "nonexistent", "sub_task": "x"})]),
            _response("Corrected answer."),
        ]
    )

    result = await sup.execute("Task", {}, llm)

    assert result.output == "Corrected answer."
    supervisor_trace = result.metadata["tool_calls"]
    assert "No worker named 'nonexistent'" in supervisor_trace[0]["result"]
    assert "Available workers" in supervisor_trace[0]["result"]


# --------------------------------------------------------------------------- #
# Context from upstream pipeline
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_receives_upstream_context():
    sup = SupervisorAgent("manager", "Manager", workers=[DummyWorker("w", "W", "ok")])

    llm = ScriptedLLM([_response("I acknowledge the context.")])
    await sup.execute("Do X", {"preprocessor": "preprocessed data here"}, llm)

    # Verify the context was injected into the user message
    user_msg = llm.requests[0]["messages"][1]["content"]
    assert "Context from upstream" in user_msg
    assert "preprocessor" in user_msg
    assert "preprocessed data here" in user_msg
    assert "Do X" in user_msg


# --------------------------------------------------------------------------- #
# Worker failure during delegation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_worker_failure():
    class FailingWorker(BaseAgent):
        def __init__(self):
            super().__init__(name="failer", role="Failing Worker")

        async def execute(self, task, context, llm):
            raise RuntimeError("worker crash")

    sup = SupervisorAgent("manager", "Manager", workers=[FailingWorker()])

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "failer", "sub_task": "x"})]),
            _response("The worker failed, but here is my answer."),
        ]
    )

    result = await sup.execute("Task", {}, llm)

    assert result.output == "The worker failed, but here is my answer."
    trace = result.metadata["tool_calls"]
    assert "Worker 'failer' failed" in trace[0]["result"]


# --------------------------------------------------------------------------- #
# Max iterations exceeded
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_exceeds_max_tool_iterations():
    sup = SupervisorAgent("manager", "Manager", workers=[], max_tool_iterations=2)

    # Always returns a tool call → never terminates
    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "w", "sub_task": "x"})]),
            _response("", [_tool_call("c2", "delegate_task", {"worker_name": "w", "sub_task": "y"})]),
        ]
    )

    with pytest.raises(AgentError) as exc:
        await sup.execute("Task", {}, llm)
    assert "max_tool_iterations" in str(exc.value)


# --------------------------------------------------------------------------- #
# Concurrency — multiple tool calls in same turn run concurrently
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_concurrent_delegations():
    w1 = DummyWorker("a", "A", "Result A", tokens=5, cost=0.001)
    w2 = DummyWorker("b", "B", "Result B", tokens=5, cost=0.001)
    sup = SupervisorAgent("manager", "Manager", workers=[w1, w2])

    llm = ScriptedLLM(
        [
            _response("", [
                _tool_call("c1", "delegate_task", {"worker_name": "a", "sub_task": "A"}),
                _tool_call("c2", "delegate_task", {"worker_name": "b", "sub_task": "B"}),
            ]),
            _response("Combined A and B."),
        ]
    )

    result = await sup.execute("Task", {}, llm)

    assert result.output == "Combined A and B."
    worker_trace = result.metadata["worker_delegations"]
    assert len(worker_trace) == 2


# --------------------------------------------------------------------------- #
# System prompt includes worker descriptions
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_system_prompt_lists_workers():
    w1 = DummyWorker("researcher", "Expert Researcher", "x")
    w2 = DummyWorker("coder", "Senior Developer", "y")
    sup = SupervisorAgent("lead", "Tech Lead", workers=[w1, w2])

    llm = ScriptedLLM([_response("Done.")])
    await sup.execute("Task", {}, llm)

    system_prompt = llm.requests[0]["messages"][0]["content"]
    assert "researcher" in system_prompt
    assert "Expert Researcher" in system_prompt
    assert "coder" in system_prompt
    assert "Senior Developer" in system_prompt
    assert "Available Workers" in system_prompt
    assert "delegate_task" in system_prompt
    assert "Tech Lead" in system_prompt


# --------------------------------------------------------------------------- #
# Works with both BaseAgent and @Agent decorated workers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_accepts_decorator_agent_workers():
    @Agent(name="helper", role="Helper Agent")
    async def helper(task: str, context: dict) -> str:
        return task

    sup = SupervisorAgent("mgr", "Manager", workers=[helper])

    # helper is a _DecoratorAgent — its own LLM call also goes through ScriptedLLM
    llm = ScriptedLLM(
        [
            # Turn 1: Supervisor decides to delegate
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "helper", "sub_task": "do thing"})]),
            # Turn 2: helper's internal LLM call (no tools → single generate)
            _response("I did the thing successfully."),
            # Turn 3: Supervisor synthesizes final answer
            _response("The helper says: I did the thing successfully."),
        ]
    )

    result = await sup.execute("Task", {}, llm)

    assert result.output == "The helper says: I did the thing successfully."
    worker_trace = result.metadata["worker_delegations"]
    assert len(worker_trace) == 1
    assert worker_trace[0]["worker"] == "helper"
    assert "I did the thing successfully" in worker_trace[0]["output"]


# --------------------------------------------------------------------------- #
# Duplicate worker names (last wins)
# --------------------------------------------------------------------------- #


def test_supervisor_duplicate_worker_names_last_wins():
    w1 = DummyWorker("a", "First", "first output")
    w2 = DummyWorker("a", "Second", "second output")
    sup = SupervisorAgent("mgr", "Manager", workers=[w1, w2])

    assert sup._workers["a"] is w2
