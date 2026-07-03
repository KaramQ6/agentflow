"""Tests for the Human-in-the-Loop (HITL) mechanism."""

from __future__ import annotations

import json

import pytest
from agentflow import (
    Agent,
    ApprovalPolicy,
    InMemoryContext,
    PauseExecution,
    Pipeline,
    tool,
)
from agentflow.exceptions import PipelineError

# ── Test utilities ────────────────────────────────────────────────────────────


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _response(content: str, tool_calls=None, tokens: int = 10) -> dict:
    return {
        "content": content,
        "tokens": tokens,
        "prompt_tokens": 6,
        "completion_tokens": 4,
        "duration": 0.0,
        "model": "fake-model",
        "cached": False,
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }


class ScriptedLLM:
    """LLM stub that returns queued responses."""

    model = "fake-model"

    def __init__(self, responses: list[dict]):
        self._responses = responses

    async def generate(self, messages, tools=None, **kwargs):
        if not self._responses:
            return _response("fallback: no more responses")
        return self._responses.pop(0)


def _make_pause_dict(overrides: dict | None = None) -> dict:
    data = {
        "agent_name": "test_agent",
        "tool_name": "send_email",
        "tool_arguments": '{"to": "a@b.com"}',
        "tool_call_id": "call_123",
        "messages": [{"role": "system", "content": "You are a tester."}],
        "total_tokens": 50,
        "total_cost": 0.001,
        "model_name": "fake-model",
        "trace": [],
        "pending_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "send_email", "arguments": '{"to": "a@b.com"}'},
            }
        ],
        "seen_calls": [["search", '{"q": "test"}']],
        "iterations_used": 0,
    }
    if overrides:
        data.update(overrides)
    return data


# ── ApprovalPolicy unit tests ─────────────────────────────────────────────────


class TestApprovalPolicy:
    def test_blocked_tools_match(self):
        policy = ApprovalPolicy(blocked_tools=["execute_sql", "send_email"])
        assert policy.requires_approval("execute_sql", '{"query": "DROP TABLE"}')
        assert policy.requires_approval("send_email", '{"to": "x@y.com"}')
        assert not policy.requires_approval("search", '{"q": "hello"}')

    def test_allowed_tools_block_others(self):
        policy = ApprovalPolicy(allowed_tools=["search", "calculator"])
        assert not policy.requires_approval("search", '{"q": "test"}')
        assert not policy.requires_approval("calculator", '{"expr": "2+2"}')
        assert policy.requires_approval("send_email", '{"to": "x@y.com"}')
        assert policy.requires_approval("execute_sql", '{"q": "SELECT 1"}')

    def test_custom_rule(self):
        policy = ApprovalPolicy(
            custom_rule=lambda name, args: "DROP" in args.upper()
        )
        assert policy.requires_approval("execute_sql", '{"query": "DROP TABLE users"}')
        assert not policy.requires_approval("execute_sql", '{"query": "SELECT 1"}')

    def test_no_rules_allows_all(self):
        policy = ApprovalPolicy()
        assert not policy.requires_approval("any_tool", "{}")
        assert not policy.requires_approval("send_email", '{"to": "x@y.com"}')

    def test_blocked_takes_priority_over_allowed(self):
        policy = ApprovalPolicy(
            blocked_tools=["dangerous"],
            allowed_tools=["dangerous", "safe"],
        )
        assert policy.requires_approval("dangerous", "{}")

    def test_custom_rule_before_allowed(self):
        policy = ApprovalPolicy(
            allowed_tools=["safe"],
            custom_rule=lambda name, _: name == "dangerous",
        )
        assert policy.requires_approval("dangerous", "{}")
        assert not policy.requires_approval("safe", "{}")
        assert policy.requires_approval("other", "{}")


# ── PauseExecution tests ──────────────────────────────────────────────────────


class TestPauseExecution:
    def test_as_dict_roundtrip(self):
        original = PauseExecution(**_make_pause_dict())
        data = original.as_dict()
        assert data["agent_name"] == "test_agent"
        assert data["tool_name"] == "send_email"

        recreated = PauseExecution.from_dict(data)
        assert recreated.agent_name == original.agent_name
        assert recreated.tool_name == original.tool_name
        assert recreated.tool_arguments == original.tool_arguments
        assert recreated.tool_call_id == original.tool_call_id
        assert recreated.total_tokens == original.total_tokens
        assert recreated.total_cost == original.total_cost
        assert recreated.seen_calls == original.seen_calls
        assert recreated.iterations_used == original.iterations_used

    def test_is_agentflow_error(self):
        exc = PauseExecution(**_make_pause_dict())
        assert isinstance(exc, Exception)

    async def test_error_message(self):
        exc = PauseExecution(**_make_pause_dict())
        assert "test_agent" in str(exc)
        assert "send_email" in str(exc)


# ── Pipeline HITL pause tests ─────────────────────────────────────────────────


class TestPipelinePause:
    async def test_pipeline_pauses_on_blocked_tool(self):
        """When an agent tries to call a blocked tool, the pipeline pauses."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["send_email"])

        @tool
        async def send_email(to: str, body: str) -> str:
            return f"Email sent to {to}"

        @tool
        async def search(q: str) -> str:
            return f"Results for {q}"

        @Agent(name="assistant", role="Assistant", tools=[send_email, search])
        async def assistant(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                _response(
                    "Let me send an email.",
                    tool_calls=[_tool_call("c1", "send_email", {"to": "boss@corp.com", "body": "Report"})],
                ),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(assistant)

        result = await pipe.run("Send a report")

        assert result.status == "paused"
        assert result.pause_info is not None
        assert result.pause_info["agent_name"] == "assistant"
        assert result.pause_info["tool_name"] == "send_email"

        # Verify state was saved to memory.
        ctx = await mem.load_context(result.pause_info["session_id"])
        assert "__hitl_pipeline" in ctx
        assert "__hitl_agent" in ctx

    async def test_pipeline_does_not_pause_on_allowed_tool(self):
        """Tools not blocked by the policy execute normally."""
        policy = ApprovalPolicy(blocked_tools=["dangerous"])

        @tool
        async def safe_tool(x: int) -> str:
            return str(x * 2)

        @Agent(name="bot", role="Bot", tools=[safe_tool])
        async def bot(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                _response(
                    "Calling safe tool.",
                    tool_calls=[_tool_call("c1", "safe_tool", {"x": 5})],
                ),
                _response("The answer is 10"),
            ]
        )

        pipe = Pipeline(llm=llm, approval_policy=policy)
        pipe.add(bot)

        result = await pipe.run("Calculate")

        assert result.status == "completed"
        assert "bot" in result.results
        assert "10" in result.output

    async def test_pipeline_pauses_on_custom_rule(self):
        """Custom rules in ApprovalPolicy trigger pause."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(
            custom_rule=lambda name, args: "admin" in args.lower()
        )

        @tool
        async def execute_command(cmd: str) -> str:
            return f"Executed: {cmd}"

        @Agent(name="admin_bot", role="Admin", tools=[execute_command])
        async def admin_bot(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                _response(
                    "Running admin command.",
                    tool_calls=[
                        _tool_call("c1", "execute_command", {"cmd": "rm -rf /admin"})
                    ],
                ),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(admin_bot)

        result = await pipe.run("Clean up")
        assert result.status == "paused"
        assert result.pause_info["tool_name"] == "execute_command"

    async def test_pipeline_without_memory_cannot_save_pause_state(self):
        """Pipeline still returns paused result even without memory."""
        policy = ApprovalPolicy(blocked_tools=["send_email"])

        @tool
        async def send_email(to: str) -> str:
            return f"Sent to {to}"

        @Agent(name="assistant", role="Assistant", tools=[send_email])
        async def assistant(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                _response(
                    "", tool_calls=[_tool_call("c1", "send_email", {"to": "x"})]
                ),
            ]
        )

        pipe = Pipeline(llm=llm, approval_policy=policy)
        pipe.add(assistant)

        result = await pipe.run("Send mail")
        assert result.status == "paused"


# ── Pipeline resume tests ─────────────────────────────────────────────────────


class TestPipelineResume:
    async def test_resume_approved_executes_tool(self):
        """Resume with approved=True executes the pending tool and continues."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["send_email"])

        @tool
        async def send_email(to: str, body: str) -> str:
            return f"Email sent to {to}: {body}"

        @Agent(name="assistant", role="Assistant", tools=[send_email])
        async def assistant(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                # First call: LLM emits tool call -> pause
                _response(
                    "Sending email now.",
                    tool_calls=[
                        _tool_call("c1", "send_email", {"to": "boss@corp.com", "body": "Report"})
                    ],
                ),
                # Resume: after tool executes, LLM returns final answer
                _response("The email was sent successfully. The report has been delivered."),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(assistant)

        paused_result = await pipe.run("Send report")
        assert paused_result.status == "paused"
        session_id = paused_result.pause_info["session_id"]

        # Resume with approval
        final_result = await pipe.resume(
            session_id=session_id,
            human_feedback="",
            approved=True,
        )

        assert final_result.status == "completed"
        assert "assistant" in final_result.results
        assert "delivered" in final_result.output.lower()

        # HITL keys should be cleaned up.
        ctx = await mem.load_context(session_id)
        assert "__hitl_pipeline" not in ctx
        assert "__hitl_agent" not in ctx

    async def test_resume_rejected_injects_feedback(self):
        """Resume with approved=False injects human feedback, agent self-corrects."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["send_email"])

        @tool
        async def send_email(to: str, body: str) -> str:
            return f"Email sent to {to}"

        @Agent(name="assistant", role="Assistant", tools=[send_email])
        async def assistant(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                # Pause: LLM wants to send email
                _response(
                    "I will email the CEO.",
                    tool_calls=[
                        _tool_call("c1", "send_email", {"to": "ceo@corp.com", "body": "Draft"})
                    ],
                ),
                # Resume rejected: agent sees feedback and corrects course
                _response(
                    "Understood. I should not send that email. The task is cancelled.",
                ),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(assistant)

        paused = await pipe.run("Draft email")
        assert paused.status == "paused"

        final = await pipe.resume(
            session_id=paused.pause_info["session_id"],
            human_feedback="Do NOT send emails to the CEO. Use the internal messaging system instead.",
            approved=False,
        )

        assert final.status == "completed"
        # Agent should have acknowledged the feedback.
        output = final.results["assistant"].output.lower()
        assert ("cancelled" in output or "should not" in output or "not send" in output)

    async def test_resume_rejected_feedback_in_tool_trace(self):
        """Rejected tool calls are recorded in the agent's trace."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["risky_op"])

        @tool
        async def risky_op(action: str) -> str:
            return f"Did: {action}"

        @Agent(name="bot", role="Bot", tools=[risky_op])
        async def bot(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                _response(
                    "", tool_calls=[_tool_call("c1", "risky_op", {"action": "delete_all"})]
                ),
                _response("Operation aborted by human review."),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(bot)

        paused = await pipe.run("Clean")
        final = await pipe.resume(
            session_id=paused.pause_info["session_id"],
            human_feedback="Denied: not enough privileges.",
            approved=False,
        )

        result = final.results["bot"]
        tool_trace = result.metadata.get("tool_calls", [])
        assert any(
            "[HUMAN REJECTED]" in str(t.get("result", "")) for t in tool_trace
        )

    async def test_resume_without_memory_raises(self):
        """Calling resume on a pipeline without memory raises PipelineError."""
        pipe = Pipeline(llm=ScriptedLLM([]))
        with pytest.raises(PipelineError, match="memory"):
            await pipe.resume("any_session", "", True)

    async def test_resume_invalid_session_raises(self):
        """Resuming with a session that has no paused state raises PipelineError."""
        pipe = Pipeline(llm=ScriptedLLM([]), memory=InMemoryContext())
        with pytest.raises(PipelineError, match="No paused state"):
            await pipe.resume("nonexistent", "", True)


# ── Streaming HITL tests ──────────────────────────────────────────────────────


class TestStreamPause:
    async def test_stream_emits_pipeline_paused(self):
        """When a tool is blocked, streaming emits 'pipeline_paused'."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["dangerous_tool"])

        @tool
        async def dangerous_tool(x: str) -> str:
            return f"Processed: {x}"

        @Agent(name="agent", role="Agent", tools=[dangerous_tool])
        async def agent(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                _response(
                    "", tool_calls=[_tool_call("c1", "dangerous_tool", {"x": "boom"})]
                ),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(agent)

        events = []
        async for event in pipe.stream("Do something dangerous"):
            events.append(event)

        types = [e.type for e in events]
        assert "agent_start" in types
        assert "pipeline_paused" in types
        assert "pipeline_error" not in types

        # Verify the paused event carries the expected data.
        paused_event = next(e for e in events if e.type == "pipeline_paused")
        assert paused_event.agent == "agent"
        assert paused_event.data["tool_name"] == "dangerous_tool"
        assert "session_id" in paused_event.data


# ── Multi-agent pipeline HITL tests ───────────────────────────────────────────


class TestMultiAgentPause:
    async def test_pause_preserves_completed_agent_results(self):
        """When an agent pauses, previously completed agents' results are kept."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["dangerous"])

        @tool
        async def dangerous(x: str) -> str:
            return f"danger: {x}"

        @Agent(name="safe_agent", role="Safe")
        async def safe_agent(task: str, context: dict) -> str:
            return f"Safe processed: {task}"

        @Agent(name="risky_agent", role="Risky", tools=[dangerous])
        async def risky_agent(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                _response("Safe agent response."),
                _response(
                    "", tool_calls=[_tool_call("c1", "dangerous", {"x": "boom"})]
                ),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(safe_agent)
        pipe.add(risky_agent)

        result = await pipe.run("Test")

        assert result.status == "paused"
        assert "safe_agent" in result.results
        assert "risky_agent" not in result.results  # paused, so not completed

    async def test_resume_continues_downstream(self):
        """After resuming a paused agent, downstream agents execute normally."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["sensitive"])

        @tool
        async def sensitive(data: str) -> str:
            return f"processed: {data}"

        @Agent(name="first", role="First", tools=[sensitive])
        async def first(task: str, context: dict) -> str:
            return task

        @Agent(name="second", role="Second")
        async def second(task: str, context: dict) -> str:
            prev = context.get("first", "no context")
            return f"Second processed after: {prev[:30]}"

        llm = ScriptedLLM(
            [
                _response(
                    "", tool_calls=[_tool_call("c1", "sensitive", {"data": "secret"})]
                ),
                _response("First agent final answer."),
                _response("Second agent final answer."),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(first)
        pipe.add(second, depends_on=["first"])

        paused = await pipe.run("Process secret")
        assert paused.status == "paused"

        final = await pipe.resume(
            session_id=paused.pause_info["session_id"],
            human_feedback="Approved",
            approved=True,
        )

        assert final.status == "completed"
        assert "first" in final.results
        assert "second" in final.results
        assert final.results["first"].output == "First agent final answer."

    async def test_nested_pause_during_resume(self):
        """If a tool is blocked AGAIN during resume (nested pause), it pauses again."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["sensitive"])

        @tool
        async def sensitive(data: str) -> str:
            return f"processed: {data}"

        @Agent(name="first", role="First", tools=[sensitive])
        async def first(task: str, context: dict) -> str:
            return task

        @Agent(name="second", role="Second", tools=[sensitive])
        async def second(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                # first agent pause
                _response(
                    "", tool_calls=[_tool_call("c1", "sensitive", {"data": "phase1"})]
                ),
                # resume first -> final answer
                _response("First agent done."),
                # second agent tries sensitive -> pause again
                _response(
                    "", tool_calls=[_tool_call("c2", "sensitive", {"data": "phase2"})]
                ),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(first)
        pipe.add(second, depends_on=["first"])

        paused1 = await pipe.run("Two-phase task")
        assert paused1.status == "paused"

        paused2 = await pipe.resume(
            session_id=paused1.pause_info["session_id"],
            human_feedback="Phase 1 approved.",
            approved=True,
        )
        assert paused2.status == "paused"
        assert paused2.pause_info["agent_name"] == "second"

        final = await pipe.resume(
            session_id=paused2.pause_info["session_id"],
            human_feedback="Phase 2 approved.",
            approved=True,
        )
        assert final.status == "completed"
        assert "first" in final.results
        assert "second" in final.results


# ── Concurrency: pause in parallel level ──────────────────────────────────────


class TestParallelPause:
    async def test_pause_in_parallel_level_saves_all_completed(self):
        """When one agent pauses in a parallel level, completed sibling results are saved."""
        mem = InMemoryContext()
        policy = ApprovalPolicy(blocked_tools=["blocked"])

        @tool
        async def blocked(x: str) -> str:
            return f"blocked: {x}"

        @tool
        async def safe(x: str) -> str:
            return f"safe: {x}"

        @Agent(name="agent_a", role="A", tools=[blocked])
        async def agent_a(task: str, context: dict) -> str:
            return task

        @Agent(name="agent_b", role="B", tools=[safe])
        async def agent_b(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                # agent_a: blocked tool -> pause
                _response(
                    "", tool_calls=[_tool_call("c1", "blocked", {"x": "y"})]
                ),
                # agent_b: safe tool + final
                _response(
                    "", tool_calls=[_tool_call("c2", "safe", {"x": "hello"})]
                ),
                _response("Agent B final answer."),
            ]
        )

        pipe = Pipeline(llm=llm, memory=mem, approval_policy=policy)
        pipe.add(agent_a)
        pipe.add(agent_b)

        result = await pipe.run("Parallel test")
        assert result.status == "paused"
        # agent_b should have completed since it used a safe tool.
        assert "agent_b" in result.results
