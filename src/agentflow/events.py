"""Event system for pipeline streaming, trigger policies, and MQTT daemon."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .types import Event, EventType

_logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class EventEmitter:
    """Async event emitter for pipeline progress."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()

    def emit(self, event_type: EventType, agent: str = "", **data: Any) -> None:
        """Emit an event (non-blocking)."""
        event = Event(type=event_type, agent=agent, data=data)
        self._queue.put_nowait(event)

    def done(self) -> None:
        """Signal that no more events will be emitted."""
        self._queue.put_nowait(None)

    async def stream(self) -> AsyncGenerator[Event, None]:
        """Async generator that yields events until done."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event


class TriggerPolicy(ABC):
    """Abstract base for trigger policies that evaluate payload conditions.

    Subclasses implement :meth:`evaluate` to inspect an incoming payload and
    decide whether a pipeline should be spawned.
    """

    @abstractmethod
    def evaluate(self, payload: dict[str, Any]) -> bool:
        """Return True if the payload should trigger a pipeline run."""

    @abstractmethod
    def build_task_prompt(self, payload: dict[str, Any]) -> str:
        """Build the task prompt string from the validated payload."""


class PydanticTriggerPolicy(TriggerPolicy):
    """Trigger policy that validates payloads with a Pydantic model and evaluates
    a user-supplied condition.

    Args:
        model: A Pydantic ``BaseModel`` subclass used to validate incoming
               JSON payloads.  Invalid payloads are rejected silently.
        condition: A callable receiving the validated model instance; returns
                   ``True`` when a pipeline should be triggered.
        prompt_template: Python format string for building the task prompt.
                         The validated model fields are available as format
                         variables (e.g. ``"battery: {battery}% at {altitude}m"``).

    Example::

        class DroneTelemetry(BaseModel):
            battery: float
            altitude: float
            altitude_drop_rate: float

        def low_battery_or_rapid_descent(data: DroneTelemetry) -> bool:
            return data.battery < 15 or data.altitude_drop_rate > 5

        policy = PydanticTriggerPolicy(
            model=DroneTelemetry,
            condition=low_battery_or_rapid_descent,
            prompt_template="Drone alert: battery={battery}%, altitude={altitude}m",
        )
    """

    def __init__(
        self,
        model: type[BaseModel],
        condition: Callable[[Any], bool],
        prompt_template: str = "Analyze this data: {data}",
    ):
        self._model = model
        self._condition = condition
        self._prompt_template = prompt_template

    def evaluate(self, payload: dict[str, Any]) -> bool:
        try:
            validated = self._model(**payload)
        except ValidationError:
            _logger.debug("Payload failed Pydantic validation: %s", payload)
            return False
        return self._condition(validated)

    def build_task_prompt(self, payload: dict[str, Any]) -> str:
        try:
            validated = self._model(**payload)
            fields = validated.model_dump()
        except ValidationError:
            fields = payload
        try:
            return self._prompt_template.format(**fields, data=json.dumps(payload))
        except KeyError:
            return self._prompt_template.format(data=json.dumps(payload))


class MQTTDaemon:
    """Standalone MQTT event-driven daemon with exponential backoff reconnection.

    Subscribes to an MQTT topic, validates incoming JSON payloads against
    a :class:`TriggerPolicy`, and spawns a user-supplied async handler
    via ``asyncio.create_task`` — never blocking the MQTT listener loop.

    Connection drops are handled with exponential backoff (base delay with
    jitter) up to a configurable maximum delay.

    Requires the ``mqtt`` extra: ``pip install agentflowkit[mqtt]``

    Args:
        broker: MQTT broker hostname or IP.
        topic: MQTT topic filter to subscribe to.
        policy: A :class:`TriggerPolicy` for payload validation and condition
                evaluation.
        handler: Async callable ``(task_prompt, payload, context) -> None``
                 invoked in a background task when the policy triggers.
        port: Broker port (default 1883).
        username: Optional broker username.
        password: Optional broker password.
        backoff_base: Base delay in seconds for exponential backoff (default 1.0).
        backoff_max: Maximum backoff delay in seconds (default 60.0).
        backoff_jitter: Add random jitter up to this many seconds (default 0.5).

    Example::

        daemon = MQTTDaemon(
            broker="localhost",
            topic="drones/+/telemetry",
            policy=my_policy,
            handler=my_handler,
        )
        await daemon.serve()
    """

    def __init__(
        self,
        broker: str,
        topic: str,
        policy: TriggerPolicy,
        handler: Callable[[str, dict[str, Any], dict[str, Any]], Any],
        port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        backoff_jitter: float = 0.5,
    ):
        self._broker = broker
        self._port = port
        self._topic = topic
        self._policy = policy
        self._handler = handler
        self._username = username
        self._password = password
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._backoff_jitter = backoff_jitter
        # Strong references to in-flight handler tasks; asyncio only keeps
        # weak references, so unanchored tasks can be garbage-collected mid-run.
        self._handler_tasks: set[asyncio.Task[Any]] = set()

    async def serve(self) -> None:
        """Run the MQTT daemon forever with automatic reconnection.

        Spawns ``asyncio.create_task`` for each triggered pipeline so the
        MQTT listener is never blocked.
        """
        try:
            import aiomqtt  # noqa: F811, F401
        except ImportError as exc:
            raise ImportError(
                "aiomqtt is required for MQTTDaemon.  Install with: "
                "pip install agentflowkit[mqtt]"
            ) from exc

        delay = self._backoff_base

        while True:
            try:
                async with aiomqtt.Client(
                    hostname=self._broker,
                    port=self._port,
                    username=self._username,
                    password=self._password,
                ) as client:
                    await client.subscribe(self._topic)
                    _logger.info(
                        "MQTT daemon connected to %s:%d [%s]",
                        self._broker,
                        self._port,
                        self._topic,
                    )
                    delay = self._backoff_base  # reset backoff on success

                    async for message in client.messages:
                        try:
                            payload_str = message.payload.decode("utf-8")
                        except UnicodeDecodeError:
                            payload_str = message.payload.decode("utf-8", errors="replace")

                        try:
                            parsed: dict[str, Any] = json.loads(payload_str)
                        except (json.JSONDecodeError, TypeError):
                            _logger.debug("Non-JSON MQTT payload on %s", message.topic.value)
                            continue

                        ctx: dict[str, Any] = {
                            "topic": message.topic.value,
                            "qos": message.qos.value,
                        }

                        if self._policy.evaluate(parsed):
                            task_prompt = self._policy.build_task_prompt(parsed)
                            # Non-blocking: spawn handler in background task
                            task = asyncio.create_task(
                                self._handler(task_prompt, parsed, ctx)
                            )
                            self._handler_tasks.add(task)
                            task.add_done_callback(self._handler_tasks.discard)

            except asyncio.CancelledError:
                _logger.info("MQTT daemon cancelled")
                raise

            except Exception as exc:
                _logger.warning(
                    "MQTT daemon connection lost (%s). "
                    "Reconnecting in %.1fs...",
                    exc,
                    delay,
                )
                # Exponential backoff with jitter
                jitter = time.monotonic() % 1 * self._backoff_jitter
                await asyncio.sleep(delay + jitter)
                delay = min(delay * 2, self._backoff_max)
