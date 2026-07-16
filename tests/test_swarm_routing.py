"""Tests for DynamicSupervisorAgent swarm routing."""

import json

import pytest

from agentflow import Agent, AgentResult, BaseAgent
from agentflow.exceptions import AgentError
from agentflow.llm import LLM
from agentflow.swarm_routing import MAX_DELEGATION_DEPTH, DynamicSupervisorAgent

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _response(content: str, tool_calls: list | None = None) -> dict:
    return {
        "content": content,
        "tokens": 10,
        "prompt_tokens": 6,
        "completion_tokens": 4,
        "duration": 0.0,
        "model": "fake-model",
        "cached": False,
        "tool_calls": tool_calls or [],
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }


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


def test_dynamic_supervisor_construction():
    w1 = DummyWorker("researcher", "Researcher", "research output")
    w2 = DummyWorker("writer", "Writer", "written content")

    sup = DynamicSupervisorAgent("manager", "Project Manager", workers=[w1, w2])

    assert sup.name == "manager"
    assert sup.role == "Project Manager"
    assert set(sup._workers.keys()) == {"researcher", "writer"}
    assert sup._max_delegation_depth == MAX_DELEGATION_DEPTH


def test_dynamic_supervisor_construction_empty_workers():
    sup = DynamicSupervisorAgent("solo", "Solo Agent", workers=[])
    assert sup._workers == {}


def test_dynamic_supervisor_custom_max_iterations():
    sup = DynamicSupervisorAgent("mgr", "Manager", workers=[], max_tool_iterations=3)
    assert sup._max_tool_iterations == 3


def test_dynamic_supervisor_custom_delegation_depth():
    sup = DynamicSupervisorAgent("mgr", "Manager", workers=[], max_delegation_depth=5)
    assert sup._max_delegation_depth == 5


def test_dynamic_supervisor_default_delegation_depth():
    sup = DynamicSupervisorAgent("mgr", "Manager", workers=[])
    assert sup._max_delegation_depth == MAX_DELEGATION_DEPTH


# --------------------------------------------------------------------------- #
# Single delegation — supervisor delegates then answers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_single_delegation():
    worker = DummyWorker("researcher", "Research Analyst", "The answer is 42.", tokens=7, cost=0.002)
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[worker])

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "researcher", "sub_task": "Find the answer"})]),
            _response("Based on research, the answer is 42."),
        ]
    )

    result = await sup.execute("What is the answer?", {}, llm)

    assert result.output == "Based on research, the answer is 42."
    assert result.agent == "manager"
    assert result.tokens_used == 27
    assert result.cost == 0.002

    worker_trace = result.metadata["worker_delegations"]
    assert len(worker_trace) == 1
    assert worker_trace[0]["worker"] == "researcher"
    assert worker_trace[0]["output"] == "The answer is 42."
    assert worker_trace[0]["tokens_used"] == 7
    assert worker_trace[0]["cost"] == 0.002
    assert worker_trace[0]["depth"] == 0

    assert worker.last_task == "Find the answer"


# --------------------------------------------------------------------------- #
# Multiple delegations — token/cost accumulation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_multiple_delegations_bubbles_cost():
    w1 = DummyWorker("researcher", "Researcher", "Fact A.", tokens=5, cost=0.001)
    w2 = DummyWorker("analyst", "Analyst", "Insight B.", tokens=8, cost=0.003)
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[w1, w2])

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
    assert result.tokens_used == 33
    assert result.cost == 0.004

    worker_trace = result.metadata["worker_delegations"]
    assert len(worker_trace) == 2
    assert result.metadata["accumulated_worker_tokens"] == 13
    assert result.metadata["accumulated_worker_cost"] == 0.004


# --------------------------------------------------------------------------- #
# No delegation — supervisor answers directly
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_no_delegation_direct_answer():
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[DummyWorker("w", "Worker", "x")])

    llm = ScriptedLLM([_response("I can handle this directly.")])

    result = await sup.execute("Simple question", {}, llm)

    assert result.output == "I can handle this directly."
    assert result.tokens_used == 10
    assert result.metadata["worker_delegations"] == []


# --------------------------------------------------------------------------- #
# Self-delegation prevention
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_cannot_delegate_to_self():
    worker = DummyWorker("researcher", "Researcher", "output")
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[worker])

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "manager", "sub_task": "x"})]),
            _response("I'll handle it myself then. Here is the answer."),
        ]
    )

    result = await sup.execute("Task", {}, llm)

    assert result.output == "I'll handle it myself then. Here is the answer."
    supervisor_trace = result.metadata["tool_calls"]
    assert len(supervisor_trace) == 1
    assert "Cannot delegate to yourself" in supervisor_trace[0]["result"]


# --------------------------------------------------------------------------- #
# Unknown worker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_unknown_worker_error():
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[DummyWorker("a", "A", "x")])

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
async def test_dynamic_supervisor_receives_upstream_context():
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[DummyWorker("w", "W", "ok")])

    llm = ScriptedLLM([_response("I acknowledge the context.")])
    await sup.execute("Do X", {"preprocessor": "preprocessed data here"}, llm)

    user_msg = llm.requests[0]["messages"][1]["content"]
    assert "Context from upstream" in user_msg
    assert "preprocessor" in user_msg
    assert "preprocessed data here" in user_msg
    assert "Do X" in user_msg


# --------------------------------------------------------------------------- #
# Worker failure during delegation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_worker_failure():
    class FailingWorker(BaseAgent):
        def __init__(self):
            super().__init__(name="failer", role="Failing Worker")

        async def execute(self, task, context, llm):
            raise RuntimeError("worker crash")

    sup = DynamicSupervisorAgent("manager", "Manager", workers=[FailingWorker()])

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
async def test_dynamic_supervisor_exceeds_max_tool_iterations():
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[], max_tool_iterations=2)

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
# Max delegation depth enforcement
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_enforces_max_delegation_depth():
    """When delegation depth is exceeded, the tool returns an error message."""
    sup = DynamicSupervisorAgent(
        "manager", "Manager",
        workers=[DummyWorker("w", "Worker", "output")],
        max_delegation_depth=0,
    )
    # Rebuild the delegate tool to use max_delegation_depth=0
    sup._max_delegation_depth = 0
    sup._delegate_tool = sup._build_delegate_tool()

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "w", "sub_task": "x"})]),
            _response("Depth limit hit, here is my answer anyway."),
        ]
    )

    result = await sup.execute("Task", {}, llm)

    assert result.output == "Depth limit hit, here is my answer anyway."
    supervisor_trace = result.metadata["tool_calls"]
    assert "Maximum delegation depth" in supervisor_trace[0]["result"]
    assert str(0) in supervisor_trace[0]["result"]


# --------------------------------------------------------------------------- #
# Concurrency — multiple tool calls in same turn run concurrently
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_concurrent_delegations():
    w1 = DummyWorker("a", "A", "Result A", tokens=5, cost=0.001)
    w2 = DummyWorker("b", "B", "Result B", tokens=5, cost=0.001)
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[w1, w2])

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
# Context isolation — worker receives empty context
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_context_isolation():
    """Workers must receive an empty context dict (no supervisor leakage)."""
    class ContextInspectingWorker(BaseAgent):
        def __init__(self):
            super().__init__(name="inspector", role="Inspector")
            self.received_context: dict | None = None

        async def execute(self, task, context, llm):
            self.received_context = context
            return AgentResult(agent=self.name, output="inspected", tokens_used=2, cost=0.0, duration=0.0)

    worker = ContextInspectingWorker()
    sup = DynamicSupervisorAgent("manager", "Manager", workers=[worker])

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "inspector", "sub_task": "check context"})]),
            _response("Context inspection complete."),
        ]
    )

    result = await sup.execute("Task", {"upstream_key": "secret_data"}, llm)

    assert result.output == "Context inspection complete."
    assert worker.received_context == {}


# --------------------------------------------------------------------------- #
# System prompt includes worker descriptions and depth limit
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_system_prompt_lists_workers():
    w1 = DummyWorker("researcher", "Expert Researcher", "x")
    w2 = DummyWorker("coder", "Senior Developer", "y")
    sup = DynamicSupervisorAgent("lead", "Tech Lead", workers=[w1, w2])

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
    assert "3 levels deep" in system_prompt


# --------------------------------------------------------------------------- #
# Works with @Agent decorated workers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_accepts_decorated_agent_workers():
    @Agent(name="helper", role="Helper Agent")
    async def helper(task: str, context: dict) -> str:
        return task

    sup = DynamicSupervisorAgent("mgr", "Manager", workers=[helper])

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "delegate_task", {"worker_name": "helper", "sub_task": "do thing"})]),
            _response("I did the thing successfully."),
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


def test_dynamic_supervisor_duplicate_worker_names_last_wins():
    w1 = DummyWorker("a", "First", "first output")
    w2 = DummyWorker("a", "Second", "second output")
    sup = DynamicSupervisorAgent("mgr", "Manager", workers=[w1, w2])

    assert sup._workers["a"] is w2


# --------------------------------------------------------------------------- #
# Dynamic tool schema reflects worker list
# --------------------------------------------------------------------------- #


def test_dynamic_tool_schema_reflects_workers():
    """The delegate_task tool's parameters schema should list available workers."""
    w1 = DummyWorker("researcher", "Researcher", "x")
    w2 = DummyWorker("coder", "Coder", "y")
    sup = DynamicSupervisorAgent("mgr", "Manager", workers=[w1, w2])

    schema = sup._delegate_tool.openai_schema
    assert schema["function"]["name"] == "delegate_task"
    props = schema["function"]["parameters"]["properties"]
    assert "worker_name" in props
    assert "sub_task" in props


def test_dynamic_tool_schema_empty_workers():
    """With no workers, the tool still has valid schema."""
    sup = DynamicSupervisorAgent("mgr", "Manager", workers=[])
    schema = sup._delegate_tool.openai_schema
    assert schema["function"]["name"] == "delegate_task"


# --------------------------------------------------------------------------- #
# Cost aggregation in metadata
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dynamic_supervisor_cost_aggregation_metadata():
    w1 = DummyWorker("a", "A", "out a", tokens=10, cost=0.005)
    w2 = DummyWorker("b", "B", "out b", tokens=15, cost=0.007)
    sup = DynamicSupervisorAgent("mgr", "Manager", workers=[w1, w2])

    llm = ScriptedLLM(
        [
            _response("", [
                _tool_call("c1", "delegate_task", {"worker_name": "a", "sub_task": "do a"}),
                _tool_call("c2", "delegate_task", {"worker_name": "b", "sub_task": "do b"}),
            ]),
            _response("All done."),
        ]
    )

    result = await sup.execute("Task", {}, llm)
    assert result.metadata["accumulated_worker_tokens"] == 25
    assert result.metadata["accumulated_worker_cost"] == 0.012
    assert result.metadata["max_delegation_depth"] == MAX_DELEGATION_DEPTH
