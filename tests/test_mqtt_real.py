"""MQTT against a real broker (amqtt) and the real client (aiomqtt).

The fakes hid MQTT-4 for two releases: the real library gives ``qos`` as an int, the fake
gave an object. These tests skip when the optional packages are absent, so the suite still
runs without them (repo convention), and run in the dev environment, where they are installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import sys

import pytest

aiomqtt = pytest.importorskip("aiomqtt")
amqtt_broker = pytest.importorskip("amqtt.broker")

from agentflow.triggers import MQTTDaemon, MQTTTrigger, TriggerPolicy, run_mqtt  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.asynccontextmanager
async def _broker():
    port = _free_port()
    broker = amqtt_broker.Broker(
        {
            "listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{port}"}},
            "sys_interval": 0,
            "auth": {"allow-anonymous": True},
        }
    )
    await broker.start()
    try:
        yield port
    finally:
        await broker.shutdown()


async def _publish(port: int, *payloads: bytes) -> None:
    async with aiomqtt.Client(hostname="127.0.0.1", port=port) as client:
        for payload in payloads:
            await client.publish("drones/1", payload, qos=1)


class LowBattery(TriggerPolicy):
    def evaluate(self, payload):
        return payload.get("battery", 100) < 15

    def build_task_prompt(self, payload):
        return f"battery {payload['battery']}"


def test_the_trigger_decodes_a_real_message_from_a_real_broker():
    async def main():
        async with _broker() as port:
            trigger = MQTTTrigger(broker="127.0.0.1", port=port, topic="drones/#", qos=1)
            stream = trigger.listen()
            first = asyncio.ensure_future(stream.__anext__())
            await asyncio.sleep(0.5)  # let it subscribe
            await _publish(port, b'{"battery": 9}')
            prompt, ctx = await asyncio.wait_for(first, timeout=5)
            await stream.aclose()
            return prompt, ctx

    prompt, ctx = run_mqtt(main())

    assert ctx["payload"] == {"battery": 9}, "MQTT-4: the 0.7 code crashed here on message.qos.value"
    assert isinstance(ctx["qos"], int)


def test_the_daemon_survives_a_malformed_message_from_a_real_broker():
    handled: list[dict] = []

    async def handler(prompt, payload, ctx):
        handled.append(payload)

    async def main():
        async with _broker() as port:
            daemon = MQTTDaemon(broker="127.0.0.1", port=port, topic="drones/#", policy=LowBattery(),
                                handler=handler, qos=1)
            task = asyncio.create_task(daemon.serve())
            await asyncio.sleep(0.5)
            await _publish(port, b"42", json.dumps({"battery": 5}).encode())
            for _ in range(50):
                if handled:
                    break
                await asyncio.sleep(0.1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    run_mqtt(main())

    assert handled == [{"battery": 5}], "MQTT-1: b'42' used to tear down the connection in a reconnect loop"


@pytest.mark.skipif(sys.platform != "win32", reason="the Proactor loop exists only on Windows")
def test_the_default_windows_loop_fails_once_with_a_clear_message():
    async def main():
        async with _broker() as port:
            daemon = MQTTDaemon(broker="127.0.0.1", port=port, topic="#", policy=LowBattery(),
                                handler=lambda *a: None)
            return await asyncio.wait_for(daemon.serve(), timeout=10)

    # The broker itself runs fine on a Proactor loop; only aiomqtt cannot.
    loop = asyncio.ProactorEventLoop()
    try:
        with pytest.raises(RuntimeError, match="run_mqtt"):
            loop.run_until_complete(main())
    finally:
        loop.close()
