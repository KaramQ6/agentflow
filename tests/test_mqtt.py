"""MQTT triggers, policies, and the daemon — driven by a fake broker.

`aiomqtt` is bound at module scope in triggers.py, so a stub swapped in with
monkeypatch exercises the real subscribe/decode/dispatch/reconnect paths
without a broker and without the optional dependency installed. Before this,
none of it was covered by anything.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from agentflow import triggers as triggers_module
from agentflow.triggers import MQTTDaemon, MQTTTrigger, PydanticTriggerPolicy, TriggerPolicy

# ── Fake aiomqtt ──────────────────────────────────────────────────────────────


class FakeTopic:
    def __init__(self, value: str):
        self.value = value


class FakeQoS:
    def __init__(self, value: int = 0):
        self.value = value


class FakeMessage:
    def __init__(self, payload: bytes, topic: str = "test/topic", qos: int = 0):
        self.payload = payload
        self.topic = FakeTopic(topic)
        self.qos = FakeQoS(qos)


class FakeClient:
    """Yields a fixed list of messages, then stops the loop it is driving."""

    def __init__(self, messages, on_exhausted=None, fail_with=None):
        self._messages = messages
        self._on_exhausted = on_exhausted
        self._fail_with = fail_with
        self.subscribed_to = None

    async def __aenter__(self):
        if self._fail_with is not None:
            raise self._fail_with
        return self

    async def __aexit__(self, *exc):
        return False

    async def subscribe(self, topic):
        self.subscribed_to = topic

    @property
    def messages(self):
        return self._iterate()

    async def _iterate(self):
        for message in self._messages:
            yield message
        if self._on_exhausted is not None:
            self._on_exhausted()


class FakeAiomqtt:
    def __init__(self, client_factory):
        self._client_factory = client_factory
        self.connect_calls = 0

    def Client(self, **kwargs):  # noqa: N802 - mirrors aiomqtt's public name
        self.connect_calls += 1
        self.last_kwargs = kwargs
        return self._client_factory(self.connect_calls)


@pytest.fixture
def fake_mqtt(monkeypatch):
    def install(client_factory):
        fake = FakeAiomqtt(client_factory)
        monkeypatch.setattr(triggers_module, "aiomqtt", fake)
        return fake

    return install


# ── MQTTTrigger ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trigger_yields_prompt_and_parsed_payload(fake_mqtt):
    fake_mqtt(
        lambda _: FakeClient([FakeMessage(b'{"temp": 82}', topic="factory/arm1")])
    )
    trigger = MQTTTrigger(broker="localhost", topic="factory/#", prompt_template="Data: {data}")

    received = [item async for item in trigger.listen()]

    assert len(received) == 1
    prompt, ctx = received[0]
    assert prompt == 'Data: {"temp": 82}'
    assert ctx["payload"] == {"temp": 82}
    assert ctx["topic"] == "factory/arm1"
    assert ctx["qos"] == 0


@pytest.mark.asyncio
async def test_trigger_subscribes_to_the_configured_topic(fake_mqtt):
    client = FakeClient([])
    fake_mqtt(lambda _: client)
    trigger = MQTTTrigger(broker="localhost", topic="sensors/+/temp")

    [item async for item in trigger.listen()]

    assert client.subscribed_to == "sensors/+/temp"


@pytest.mark.asyncio
async def test_trigger_passes_non_json_payloads_through_as_text(fake_mqtt):
    fake_mqtt(lambda _: FakeClient([FakeMessage(b"not json at all")]))
    trigger = MQTTTrigger(broker="localhost")

    (_, ctx), = [item async for item in trigger.listen()]

    assert ctx["payload"] == "not json at all"
    assert ctx["payload_raw"] == "not json at all"


@pytest.mark.asyncio
async def test_trigger_survives_undecodable_bytes(fake_mqtt):
    fake_mqtt(lambda _: FakeClient([FakeMessage(b"\xff\xfe bad bytes")]))
    trigger = MQTTTrigger(broker="localhost")

    received = [item async for item in trigger.listen()]

    assert len(received) == 1, "a bad byte sequence must not kill the listener"


@pytest.mark.asyncio
async def test_trigger_without_aiomqtt_explains_the_extra(monkeypatch):
    monkeypatch.setattr(triggers_module, "aiomqtt", None)
    trigger = MQTTTrigger(broker="localhost")

    with pytest.raises(ImportError, match=r"agentflowkit\[mqtt\]"):
        [item async for item in trigger.listen()]


# ── TriggerPolicy ─────────────────────────────────────────────────────────────


class Telemetry(BaseModel):
    battery: float
    altitude: float


def _policy(**kwargs):
    return PydanticTriggerPolicy(
        model=Telemetry,
        condition=lambda data: data.battery < 15,
        **kwargs,
    )


def test_policy_is_abstract():
    with pytest.raises(TypeError):
        TriggerPolicy()  # type: ignore[abstract]


def test_policy_triggers_when_the_condition_holds():
    assert _policy().evaluate({"battery": 9, "altitude": 30}) is True


def test_policy_stays_quiet_when_the_condition_does_not_hold():
    assert _policy().evaluate({"battery": 90, "altitude": 30}) is False


def test_policy_rejects_payloads_that_fail_validation():
    """An invalid payload is not a trigger — and must not raise."""
    assert _policy().evaluate({"battery": "flat"}) is False
    assert _policy().evaluate({}) is False


def test_policy_builds_a_prompt_from_validated_fields():
    policy = _policy(prompt_template="battery={battery}% alt={altitude}m")

    prompt = policy.build_task_prompt({"battery": 9, "altitude": 30.5})

    assert prompt == "battery=9.0% alt=30.5m"


def test_policy_prompt_falls_back_to_the_raw_payload_when_invalid():
    policy = _policy(prompt_template="Alert: {data}")

    prompt = policy.build_task_prompt({"battery": "flat"})

    assert json.loads(prompt.removeprefix("Alert: ")) == {"battery": "flat"}


def test_policy_prompt_survives_a_template_typo():
    """A daemon processing live messages must not die on a bad placeholder."""
    policy = _policy(prompt_template="alt={altitude} bogus={nonexistent_field}")

    prompt = policy.build_task_prompt({"battery": 9, "altitude": 30})

    assert "alt=30.0" in prompt, "known fields still render"
    assert "{nonexistent_field}" in prompt, "the typo is left visible, not raised"


def test_policy_prompt_data_placeholder_wins_over_a_model_field():
    class Payload(BaseModel):
        data: str

    policy = PydanticTriggerPolicy(
        model=Payload, condition=lambda _: True, prompt_template="{data}"
    )

    assert json.loads(policy.build_task_prompt({"data": "inner"})) == {"data": "inner"}


# ── MQTTDaemon ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_daemon_invokes_the_handler_only_for_triggering_payloads(fake_mqtt):
    handled = []

    async def handler(prompt, payload, ctx):
        handled.append((prompt, payload, ctx))

    stop = asyncio.Event()
    fake_mqtt(
        lambda _: FakeClient(
            [
                FakeMessage(b'{"battery": 9, "altitude": 30}'),   # triggers
                FakeMessage(b'{"battery": 90, "altitude": 30}'),  # does not
                FakeMessage(b"not json"),                          # ignored
            ],
            on_exhausted=stop.set,
        )
    )

    daemon = MQTTDaemon(
        broker="localhost",
        topic="drones/#",
        policy=_policy(prompt_template="battery={battery}"),
        handler=handler,
    )
    task = asyncio.create_task(daemon.serve())
    await asyncio.wait_for(stop.wait(), timeout=2)
    await asyncio.sleep(0)  # let the spawned handler task run
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(handled) == 1
    prompt, payload, ctx = handled[0]
    assert prompt == "battery=9.0"
    assert payload == {"battery": 9, "altitude": 30}
    assert ctx["topic"] == "test/topic"


@pytest.mark.asyncio
async def test_daemon_reconnects_after_a_dropped_connection(fake_mqtt):
    stop = asyncio.Event()

    def client_for(attempt):
        if attempt == 1:
            return FakeClient([], fail_with=ConnectionError("broker went away"))
        return FakeClient([], on_exhausted=stop.set)

    fake = fake_mqtt(client_for)
    daemon = MQTTDaemon(
        broker="localhost",
        topic="#",
        policy=_policy(),
        handler=lambda *a: asyncio.sleep(0),
        backoff_base=0.01,
        backoff_jitter=0.0,
    )

    task = asyncio.create_task(daemon.serve())
    await asyncio.wait_for(stop.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake.connect_calls >= 2, "the daemon gave up instead of reconnecting"


@pytest.mark.asyncio
async def test_daemon_backs_off_when_the_stream_ends_cleanly(fake_mqtt):
    """A clean stream end used to reconnect in a tight loop with no await,
    starving the event loop. It must back off like a dropped connection."""
    fake = fake_mqtt(lambda _: FakeClient([]))
    daemon = MQTTDaemon(
        broker="localhost",
        topic="#",
        policy=_policy(),
        handler=None,
        backoff_base=0.05,
        backoff_jitter=0.0,
    )

    task = asyncio.create_task(daemon.serve())
    await asyncio.sleep(0.12)  # long enough for ~2 backed-off reconnects
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake.connect_calls <= 5, (
        f"reconnected {fake.connect_calls} times in 0.12s — the loop is spinning"
    )
    assert fake.connect_calls >= 2, "the daemon stopped reconnecting"


@pytest.mark.asyncio
async def test_daemon_cancellation_propagates(fake_mqtt):
    started = asyncio.Event()

    async def block_forever():
        started.set()
        await asyncio.sleep(3600)

    class BlockingClient(FakeClient):
        @property
        def messages(self):
            return self._blocking()

        async def _blocking(self):
            await block_forever()
            yield  # pragma: no cover

    fake_mqtt(lambda _: BlockingClient([]))
    daemon = MQTTDaemon(broker="localhost", topic="#", policy=_policy(), handler=None)

    task = asyncio.create_task(daemon.serve())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_daemon_without_aiomqtt_explains_the_extra(monkeypatch):
    monkeypatch.setattr(triggers_module, "aiomqtt", None)
    daemon = MQTTDaemon(broker="localhost", topic="#", policy=_policy(), handler=None)

    with pytest.raises(ImportError, match=r"agentflowkit\[mqtt\]"):
        await daemon.serve()
