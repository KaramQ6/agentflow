"""MQTT robustness (0.8.0): bad messages, alarm storms, reconnects, TLS, Windows loops.

Reuses the fake broker from test_mqtt.py, which now matches aiomqtt 2.x (``qos`` is an int).
"""

from __future__ import annotations

import asyncio
import json
import ssl

import pytest
from test_mqtt import FakeAiomqtt, FakeClient, FakeMessage, _policy

from agentflow import triggers as triggers_module
from agentflow.triggers import MQTTDaemon, MQTTTrigger, TriggerPolicy


@pytest.fixture
def fake_mqtt(monkeypatch):
    def install(client_factory):
        fake = FakeAiomqtt(client_factory)
        monkeypatch.setattr(triggers_module, "aiomqtt", fake)
        return fake

    return install


async def _run_daemon_until(daemon: MQTTDaemon, stop: asyncio.Event) -> None:
    task = asyncio.create_task(daemon.serve())
    await asyncio.wait_for(stop.wait(), timeout=2)
    for _ in range(3):
        await asyncio.sleep(0)  # let spawned handler tasks run
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _alarm(battery: float, topic: str = "drones/1") -> FakeMessage:
    return FakeMessage(json.dumps({"battery": battery, "altitude": 30}).encode(), topic=topic)


@pytest.mark.asyncio
async def test_one_non_object_payload_is_skipped_without_dropping_the_connection(fake_mqtt):
    handled = []
    stop = asyncio.Event()
    stream = [FakeMessage(b"42"), FakeMessage(b"[1, 2]"), _alarm(5)]
    fake = fake_mqtt(lambda _: FakeClient(stream, on_exhausted=stop.set))

    async def handler(prompt, payload, ctx):
        handled.append(payload)

    await _run_daemon_until(MQTTDaemon(broker="b", topic="#", policy=_policy(), handler=handler), stop)

    assert fake.connect_calls == 1, "a malformed payload must never tear down the connection (MQTT-1)"
    assert handled == [{"battery": 5, "altitude": 30}], "the valid message after it must still be handled"


@pytest.mark.asyncio
async def test_a_policy_that_raises_on_one_message_does_not_stop_the_daemon(fake_mqtt):
    class Exploding(TriggerPolicy):
        def evaluate(self, payload):
            if payload.get("boom"):
                raise RuntimeError("policy bug")
            return True

        def build_task_prompt(self, payload):
            return "p"

    handled = []
    stop = asyncio.Event()
    stream = [FakeMessage(b'{"boom": 1}'), FakeMessage(b'{"ok": 1}')]
    fake = fake_mqtt(lambda _: FakeClient(stream, on_exhausted=stop.set))

    async def handler(prompt, payload, ctx):
        handled.append(payload)

    await _run_daemon_until(MQTTDaemon(broker="b", topic="#", policy=Exploding(), handler=handler), stop)

    assert fake.connect_calls == 1, "a per-message error is isolated to that message"
    assert handled == [{"ok": 1}]


@pytest.mark.asyncio
async def test_a_sustained_alarm_fires_once_and_a_new_alarm_fires_again(fake_mqtt):
    handled = []
    stop = asyncio.Event()
    stream = [_alarm(5)] * 50 + [_alarm(90), _alarm(5)]
    fake_mqtt(lambda _: FakeClient(stream, on_exhausted=stop.set))

    async def handler(prompt, payload, ctx):
        handled.append(payload["battery"])

    await _run_daemon_until(MQTTDaemon(broker="b", topic="#", policy=_policy(), handler=handler), stop)

    assert handled == [5, 5], (
        "50 messages of the same low-battery state are one alarm (MQTT-2); "
        "recovering and dropping again is a new alarm"
    )


@pytest.mark.asyncio
async def test_alarms_on_different_topics_are_independent(fake_mqtt):
    handled = []
    stop = asyncio.Event()
    stream = [_alarm(5, "drones/1"), _alarm(5, "drones/2"), _alarm(5, "drones/1")]
    fake_mqtt(lambda _: FakeClient(stream, on_exhausted=stop.set))

    async def handler(prompt, payload, ctx):
        handled.append(ctx["topic"])

    await _run_daemon_until(MQTTDaemon(broker="b", topic="#", policy=_policy(), handler=handler), stop)

    assert sorted(handled) == ["drones/1", "drones/2"], "each device gets its own alarm state"


@pytest.mark.asyncio
async def test_cooldown_zero_fires_on_every_triggering_message(fake_mqtt):
    handled = []
    stop = asyncio.Event()
    fake_mqtt(lambda _: FakeClient([_alarm(5)] * 3, on_exhausted=stop.set))

    async def handler(prompt, payload, ctx):
        handled.append(1)

    daemon = MQTTDaemon(broker="b", topic="#", policy=_policy(), handler=handler, cooldown_s=0)
    await _run_daemon_until(daemon, stop)

    assert len(handled) == 3, "cooldown_s=0 restores the 0.7 behaviour for users who want it"


@pytest.mark.asyncio
async def test_handlers_beyond_max_concurrent_are_skipped(fake_mqtt):
    release = asyncio.Event()
    started = []
    stop = asyncio.Event()
    stream = [_alarm(5, f"drones/{i}") for i in range(4)]
    fake_mqtt(lambda _: FakeClient(stream, on_exhausted=stop.set))

    async def handler(prompt, payload, ctx):
        started.append(ctx["topic"])
        await release.wait()

    daemon = MQTTDaemon(broker="b", topic="#", policy=_policy(), handler=handler, max_concurrent=2)
    task = asyncio.create_task(daemon.serve())
    await asyncio.wait_for(stop.wait(), timeout=2)
    for _ in range(3):
        await asyncio.sleep(0)
    in_flight = list(started)
    release.set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert in_flight == ["drones/0", "drones/1"], "at most max_concurrent handlers run at once"


@pytest.mark.asyncio
async def test_a_skipped_alarm_is_retried_on_the_next_message(fake_mqtt):
    release = asyncio.Event()
    started = []
    stop = asyncio.Event()
    stream = [_alarm(5, "drones/0"), _alarm(5, "drones/1")]
    fake_mqtt(lambda _: FakeClient(stream, on_exhausted=stop.set))

    async def handler(prompt, payload, ctx):
        started.append(ctx["topic"])
        await release.wait()

    daemon = MQTTDaemon(broker="b", topic="#", policy=_policy(), handler=handler, max_concurrent=1)
    task = asyncio.create_task(daemon.serve())
    await asyncio.wait_for(stop.wait(), timeout=2)
    await asyncio.sleep(0)
    alarm_state = dict(daemon._alarms)
    release.set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert started == ["drones/0"]
    assert "drones/1" not in alarm_state, "a skipped alarm must not count as fired, so it fires next time"


@pytest.mark.asyncio
async def test_the_trigger_reconnects_after_a_dropped_connection(fake_mqtt):
    def client_for(attempt):
        if attempt == 1:
            return FakeClient([], fail_with=ConnectionError("broker hiccup"))
        return FakeClient([FakeMessage(b'{"v": 1}')])

    fake_mqtt(client_for)
    trigger = MQTTTrigger(broker="b", topic="#", backoff_base=0.01, backoff_jitter=0.0)

    _prompt, ctx = await asyncio.wait_for(trigger.listen().__anext__(), timeout=2)

    assert ctx["payload"] == {"v": 1}, "a broker hiccup must not end the stream that serve() reads (SERVE-2)"


@pytest.mark.asyncio
async def test_tls_and_qos_are_passed_to_the_client(fake_mqtt):
    context = ssl.create_default_context()
    clients = []

    def client_for(_attempt):
        clients.append(FakeClient([FakeMessage(b"{}")]))
        return clients[-1]

    fake = fake_mqtt(client_for)
    trigger = MQTTTrigger(broker="b", topic="t", qos=1, tls_context=context)
    await asyncio.wait_for(trigger.listen().__anext__(), timeout=2)

    assert fake.last_kwargs["tls_context"] is context
    assert clients[0].subscribed_qos == 1


def test_run_mqtt_runs_on_a_loop_that_supports_add_reader():
    async def main():
        return type(asyncio.get_running_loop()).__name__

    name = triggers_module.run_mqtt(main())

    assert "Proactor" not in name, f"aiomqtt needs add_reader, got {name}"
