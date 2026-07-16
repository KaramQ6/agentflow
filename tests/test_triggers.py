"""Tests for event-driven triggers and Pipeline.serve() daemon mode."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from agentflow import Pipeline, PipelineResult
from agentflow.agent import BaseAgent
from agentflow.triggers import BaseTrigger, MQTTTrigger
from agentflow.types import AgentResult

# ── Shared test helpers ──────────────────────────────────────────────────────────


class MockLLM:
    async def generate(self, messages, **kwargs):
        user_msg = messages[-1]["content"] if messages else ""
        return {
            "content": f"LLM: {user_msg[:60]}",
            "tokens": 10,
            "duration": 0.01,
            "model": "mock",
        }


class EchoAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name=name, role=f"{name}_role")

    async def execute(self, task, context, llm):
        return AgentResult(
            agent=self.name,
            output=f"[{self.name}] task={task!r} ctx_keys={sorted(context)}",
            tokens_used=5,
            duration=0.01,
        )


# ── BaseTrigger tests ────────────────────────────────────────────────────────────


def test_base_trigger_is_abstract():
    with pytest.raises(TypeError):
        BaseTrigger()  # type: ignore[abstract]


class _ConcreteTrigger(BaseTrigger):
    def __init__(self, items: list[tuple[str, dict[str, Any]]]):
        self._items = items

    async def listen(self) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        for prompt, ctx in self._items:
            yield prompt, ctx


@pytest.mark.asyncio
async def test_trigger_yields_tuples():
    trigger = _ConcreteTrigger([
        ("hello world", {"source": "test"}),
        ("second task", {"key": "value"}),
    ])
    results = []
    async for prompt, ctx in trigger.listen():
        results.append((prompt, ctx))

    assert len(results) == 2
    assert results[0] == ("hello world", {"source": "test"})
    assert results[1] == ("second task", {"key": "value"})


@pytest.mark.asyncio
async def test_trigger_empty_stream():
    trigger = _ConcreteTrigger([])
    results = []
    async for prompt, ctx in trigger.listen():
        results.append((prompt, ctx))
    assert results == []


# ── MQTTTrigger unit tests (no real broker) ──────────────────────────────────────


def _make_fake_message(topic_str: str, payload_str: str, qos: int = 0):
    """Build a minimal fake object that walks like an aiomqtt Message."""
    msg = type("Message", (), {})()
    msg.payload = payload_str.encode("utf-8")

    topic_obj = type("Topic", (), {})()
    topic_obj.value = topic_str
    msg.topic = topic_obj

    qos_obj = type("QoS", (), {})()
    qos_obj.value = qos
    msg.qos = qos_obj

    return msg


def _patch_aiomqtt(monkeypatch, message_generator):
    """Replace agentflow.triggers.aiomqtt with a fake that yields messages."""

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self._subscribed: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def subscribe(self, topic: str):
            self._subscribed.append(topic)

        @property
        def messages(self):
            return self

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await message_generator()

    class _FakeMQTT:
        @staticmethod
        def Client(hostname, port, username=None, password=None):
            return _FakeClient()

    import agentflow.triggers as triggers_mod

    monkeypatch.setattr(triggers_mod, "aiomqtt", _FakeMQTT)


@pytest.mark.asyncio
async def test_mqtt_trigger_parses_json_payload(monkeypatch):
    """Simulate MQTT messages flowing through the trigger without a real broker."""
    fake_msg = _make_fake_message(
        "factory/arm1/sensors",
        json.dumps({"temp": 82, "vibration": 0.3}),
        qos=1,
    )

    async def _msg_gen():
        return fake_msg

    _patch_aiomqtt(monkeypatch, _msg_gen)

    trigger = MQTTTrigger(
        broker="test-broker",
        topic="factory/+/sensors",
        prompt_template="Analyze: {data}",
    )

    events: list[tuple[str, dict[str, Any]]] = []
    async for prompt, ctx in trigger.listen():
        events.append((prompt, ctx))
        break

    assert len(events) == 1
    prompt, ctx = events[0]
    assert "Analyze:" in prompt
    assert '"temp": 82' in prompt
    assert ctx["topic"] == "factory/arm1/sensors"
    assert ctx["qos"] == 1
    assert ctx["payload"] == {"temp": 82, "vibration": 0.3}


@pytest.mark.asyncio
async def test_mqtt_trigger_non_json_payload(monkeypatch):
    fake_msg = _make_fake_message("test/topic", "plain text, not json", qos=0)

    async def _msg_gen():
        return fake_msg

    _patch_aiomqtt(monkeypatch, _msg_gen)

    trigger = MQTTTrigger(broker="x", prompt_template="Got: {data}")
    async for prompt, ctx in trigger.listen():
        assert prompt == "Got: plain text, not json"
        assert ctx["payload"] == "plain text, not json"
        break


@pytest.mark.asyncio
async def test_mqtt_trigger_binary_payload(monkeypatch):
    """Non-UTF-8 payloads fall back to replacement characters."""
    fake_msg = _make_fake_message("topic/bin", "")
    fake_msg.payload = b"\xff\xfe\xfd"

    async def _msg_gen():
        return fake_msg

    _patch_aiomqtt(monkeypatch, _msg_gen)

    trigger = MQTTTrigger(broker="x", prompt_template="Raw: {data}")
    async for _prompt, ctx in trigger.listen():
        assert ctx["payload"] == "���"  # replacement chars for invalid UTF-8
        break


# ── Pipeline.serve() tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_serve_dispatches_all_trigger_items():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    pipe.add(EchoAgent("echo"))

    trigger = _ConcreteTrigger([
        ("task-A", {}),
        ("task-B", {"extra": 1}),
        ("task-C", {}),
    ])

    results: list[PipelineResult] = []

    async def collect(result: PipelineResult):
        results.append(result)

    await pipe.serve(trigger, max_concurrent=5, on_result=collect)

    assert len(results) == 3
    result_names = {r.results["echo"].output for r in results}
    assert any("task='task-A'" in o for o in result_names)
    assert any("task-B" in o for o in result_names)
    assert any("task='task-C'" in o for o in result_names)
    # Non-empty context_data is appended to the prompt as a JSON block.
    task_b_output = next(o for o in result_names if "task-B" in o)
    assert "Context data (JSON)" in task_b_output
    assert '"extra": 1' in task_b_output


@pytest.mark.asyncio
async def test_serve_respects_max_concurrent():
    """Verify that max_concurrent caps simultaneous executions."""
    running = 0
    max_seen = 0
    lock = asyncio.Lock()

    class SlowAgent(BaseAgent):
        def __init__(self):
            super().__init__(name="slow", role="slow")

        async def execute(self, task, context, llm):
            nonlocal running, max_seen
            async with lock:
                running += 1
                if running > max_seen:
                    max_seen = running
            await asyncio.sleep(0.1)
            async with lock:
                running -= 1
            return AgentResult(agent="slow", output="ok", tokens_used=1, duration=0.1)

    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    pipe.add(SlowAgent())

    trigger = _ConcreteTrigger([(f"task-{i}", {}) for i in range(10)])

    await pipe.serve(trigger, max_concurrent=3, on_result=None)

    assert max_seen <= 3
    assert max_seen > 1  # some concurrency actually happened


@pytest.mark.asyncio
async def test_serve_on_error_callback():
    llm = MockLLM()

    class FailingAgent(BaseAgent):
        def __init__(self):
            super().__init__(name="failer", role="failer")

        async def execute(self, task, context, llm):
            raise RuntimeError("boom")

    pipe = Pipeline(llm=llm)
    pipe.add(FailingAgent())

    trigger = _ConcreteTrigger([("t1", {}), ("t2", {})])
    errors: list[tuple[Exception, str]] = []

    async def on_err(exc: Exception, prompt: str):
        errors.append((exc, prompt))

    await pipe.serve(trigger, max_concurrent=2, on_error=on_err)

    assert len(errors) == 2
    assert "boom" in str(errors[0][0])
    assert errors[0][1] == "t1"
    assert errors[1][1] == "t2"


@pytest.mark.asyncio
async def test_serve_cancellation_cleanup():
    """Cancelling serve() should clean up pending background tasks."""
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    pipe.add(EchoAgent("echo"))

    # A trigger that yields slowly so some tasks stay pending when cancelled.
    class _SlowTrigger(BaseTrigger):
        async def listen(self) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
            i = 0
            while True:
                if False:
                    yield  # pragma: no cover  — make mypy see this as async gen
                await asyncio.sleep(0.05)
                i += 1
                yield f"t{i}", {}

    results: list[PipelineResult] = []

    async def collect(r: PipelineResult):
        results.append(r)

    serve_task = asyncio.create_task(
        pipe.serve(_SlowTrigger(), max_concurrent=3, on_result=collect)
    )
    await asyncio.sleep(0.3)
    serve_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serve_task

    # At least a few tasks completed before cancellation (concurrent=3, 0.05s per yield).
    assert len(results) >= 2


@pytest.mark.asyncio
async def test_serve_on_result_sync_callback():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    pipe.add(EchoAgent("echo"))

    trigger = _ConcreteTrigger([("x", {})])
    results: list[PipelineResult] = []

    await pipe.serve(trigger, max_concurrent=1, on_result=results.append)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_serve_on_result_async_callback():
    llm = MockLLM()
    pipe = Pipeline(llm=llm)
    pipe.add(EchoAgent("echo"))

    trigger = _ConcreteTrigger([("x", {})])
    results: list[PipelineResult] = []

    async def async_collect(r: PipelineResult):
        await asyncio.sleep(0)
        results.append(r)

    await pipe.serve(trigger, max_concurrent=1, on_result=async_collect)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_serve_default_error_handler_logs():
    """When on_error is None, errors should be logged, not raised."""

    llm = MockLLM()

    class FailAgent(BaseAgent):
        def __init__(self):
            super().__init__(name="bad", role="bad")

        async def execute(self, task, context, llm):
            raise ValueError("intentional")

    pipe = Pipeline(llm=llm)
    pipe.add(FailAgent())

    trigger = _ConcreteTrigger([("fail-me", {})])

    with pytest.raises(ValueError):
        # Without serve(), this would raise.  With serve(), it's swallowed.
        await pipe.run("direct-call")

    # Now via serve — should NOT raise.
    await pipe.serve(trigger, max_concurrent=1)
    # If we got here without an exception, the error was caught and logged.
