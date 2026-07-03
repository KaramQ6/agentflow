"""Event-driven triggers for autonomous pipeline execution.

Triggers listen to continuous data streams (MQTT, WebSockets, etc.) and yield
(task_prompt, context_data) tuples that the Pipeline.serve() method consumes
to dispatch autonomous pipeline runs.
"""

from __future__ import annotations

import json as _json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

try:
    import aiomqtt  # noqa: F401
except ImportError:  # pragma: no cover
    aiomqtt = None


class BaseTrigger(ABC):
    """Abstract base for event-driven triggers.

    Subclasses implement :meth:`listen` to connect to a data source and yield
    ``(task_prompt, context_data)`` tuples.  Each yielded tuple spawns one
    pipeline run via :meth:`Pipeline.serve`.

    Usage::

        class MyTrigger(BaseTrigger):
            async def listen(self) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
                while True:
                    data = await some_source.receive()
                    yield f"Process: {data}", {"raw": data}
    """

    @abstractmethod
    async def listen(self) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """Yield ``(task_prompt, context_data)`` tuples indefinitely.

        Each tuple describes a self-contained pipeline invocation:
        ``task_prompt`` is the string passed as the pipeline task and
        ``context_data`` is an arbitrary dict available to agents via
        the execution context.
        """
        if False:  # pragma: no cover
            yield


class MQTTTrigger(BaseTrigger):
    """Trigger that subscribes to an MQTT topic and yields sensor/data payloads.

    Requires the optional ``mqtt`` extra: ``pip install agentflowkit[mqtt]``.

    Args:
        broker: MQTT broker hostname or IP (e.g. ``"localhost"``).
        port: Broker port (default 1883). Use 8883 for TLS typically.
        topic: MQTT topic filter to subscribe to (supports wildcards ``+``, ``#``).
        prompt_template: Python format string used to build the task prompt.
            The incoming JSON payload is available as ``{data}``.  Example:
            ``"Analyze this sensor anomaly: {data}"``.
        username: Optional broker username.
        password: Optional broker password.

    Example::

        trigger = MQTTTrigger(
            broker="192.168.1.100",
            topic="factory/+/sensors",
            prompt_template="Robot arm anomaly detected: {data}",
        )
        async for prompt, ctx in trigger.listen():
            print(prompt, ctx)
    """

    def __init__(
        self,
        broker: str,
        topic: str = "#",
        port: int = 1883,
        prompt_template: str = "Analyze this data: {data}",
        username: str | None = None,
        password: str | None = None,
    ):
        self._broker = broker
        self._port = port
        self._topic = topic
        self._prompt_template = prompt_template
        self._username = username
        self._password = password

    async def listen(self) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """Connect to broker, subscribe to topic, and yield incoming payloads."""
        if aiomqtt is None:
            raise ImportError(
                "aiomqtt is required for MQTTTrigger.  Install with: "
                "pip install agentflowkit[mqtt]"
            )

        async with aiomqtt.Client(
            hostname=self._broker,
            port=self._port,
            username=self._username,
            password=self._password,
        ) as client:
            await client.subscribe(self._topic)
            async for message in client.messages:
                try:
                    payload_str = message.payload.decode("utf-8")
                except UnicodeDecodeError:
                    payload_str = message.payload.decode("utf-8", errors="replace")

                ctx: dict[str, Any] = {
                    "topic": message.topic.value,
                    "qos": message.qos.value,
                    "payload_raw": payload_str,
                }

                # Attempt JSON parse; fall back to raw string on failure.
                try:
                    parsed = _json.loads(payload_str)
                    ctx["payload"] = parsed
                except (ValueError, TypeError):
                    ctx["payload"] = payload_str  # non-JSON payload → raw string

                task_prompt = self._prompt_template.format(data=payload_str)
                yield task_prompt, ctx
