"""Function/tool calling for agentflow agents.

A ``Tool`` wraps a plain Python callable and exposes it to an LLM in the
OpenAI function-calling format. The JSON schema for the tool's arguments is
generated automatically from the function's type hints via Pydantic, so you
never hand-write a schema:

    @tool
    async def get_weather(city: str, unit: str = "celsius") -> str:
        \"\"\"Look up the current weather for a city.\"\"\"
        ...

    @Agent(name="assistant", role="Helpful Assistant", tools=[get_weather])
    async def assistant(task: str, context: dict) -> str:
        return task
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections.abc import Callable
from typing import Any, Literal

from pydantic import ConfigDict, create_model

from .exceptions import ToolError

Effect = Literal["read", "write"]


class InvalidArguments(ToolError):
    """The arguments never reached the function: bad JSON, an unknown or missing field."""


class Tool:
    """An LLM-callable wrapper around a Python function.

    Sync and async callables are both supported; sync tools are executed in a
    thread so they never block the event loop.

    Attributes:
        name: Tool name exposed to the model (defaults to the function name).
        description: Human/LLM-readable description (defaults to the docstring).
        parameters: JSON schema for the arguments, in OpenAI function format.
        effect: What calling the tool does to the world (ADR 0006).
            ``"read"`` only observes: it may be repeated, retried and run
            concurrently. ``"write"`` changes something: it runs at most once per
            occurrence in a run (the effect ledger) and one at a time, in the
            model's order, and must be async so it can be cancelled. ``None``
            (default) keeps 0.7 behaviour, but an agent that ran such a tool is
            not retried, because a retry could repeat it.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        *,
        effect: Effect | None = None,
    ):
        if effect not in (None, "read", "write"):
            raise ValueError(f"effect must be 'read', 'write' or None, got {effect!r}")
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""
        self.effect = effect
        self._is_async = asyncio.iscoroutinefunction(func)
        if effect == "write" and not self._is_async:
            raise TypeError(
                f"tool {self.name!r}: a write tool must be async. A sync function runs in a "
                "thread that cannot be cancelled, so it would keep acting after a timeout."
            )
        self._model = _build_args_model(func, self.name)
        self.parameters = _clean_schema(self._model.model_json_schema())

    @property
    def openai_schema(self) -> dict[str, Any]:
        """This tool as an OpenAI ``tools=[...]`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def acall(self, arguments: dict[str, Any] | str) -> str:
        """Validate ``arguments`` against the schema and invoke the function.

        Args:
            arguments: Either a kwargs dict or a JSON string (as sent by the LLM).

        Returns:
            The function's return value coerced to ``str`` (JSON-encoded if it is
            a mapping or sequence) so it can be fed back to the model.

        Raises:
            ToolError: If the arguments are malformed or the function raises.
        """
        return await self._invoke(self._validate(arguments))

    def _validate(self, arguments: dict[str, Any] | str) -> dict[str, Any]:
        """Parse and validate *arguments*; return the coerced kwargs. Never calls the function."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                raise InvalidArguments(self.name, f"invalid JSON arguments: {e}") from e
        if not isinstance(arguments, dict):
            raise InvalidArguments(self.name, "arguments must be a JSON object")

        try:
            validated = self._model(**arguments)
        except Exception as e:  # pydantic ValidationError or TypeError
            raise InvalidArguments(self.name, f"argument validation failed: {e}") from e
        return {k: getattr(validated, k) for k in type(validated).model_fields}

    async def _invoke(self, kwargs: dict[str, Any]) -> str:
        """Call the function with already-validated kwargs."""
        try:
            if self._is_async:
                result = await self.func(**kwargs)
            else:
                result = await asyncio.to_thread(self.func, **kwargs)
        except Exception as e:
            raise ToolError(self.name, str(e)) from e

        return _stringify(result)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    effect: Effect | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator that turns a function into a :class:`Tool`.

    Usable bare (``@tool``) or with arguments (``@tool(name="lookup")``).
    Declare what the tool does with ``effect="read"`` or ``effect="write"``
    (see :class:`Tool`).
    """

    def wrap(fn: Callable[..., Any]) -> Tool:
        return Tool(fn, name=name, description=description, effect=effect)

    return wrap(func) if func is not None else wrap


def _build_args_model(func: Callable[..., Any], tool_name: str) -> Any:
    """Build a Pydantic model describing ``func``'s call signature."""
    sig = inspect.signature(func)
    # Under ``from __future__ import annotations`` every annotation is a string. Resolve them in
    # the function's own module, or a Literal or an alias cannot be found (TOOL-1).
    try:
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001 - a name only visible inside a function stays unresolved, as before
        hints = {}
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, param.annotation if param.annotation is not inspect.Parameter.empty else str)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, default)
    # Unknown arguments and NaN/Infinity are refused, never dropped or passed on:
    # a speed limit silently ignored, or an infinite speed, reaches an actuator.
    strict = ConfigDict(extra="forbid", allow_inf_nan=False)
    return create_model(f"{tool_name}_Args", __config__=strict, **fields)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip Pydantic-only keys the model doesn't need, keep it OpenAI-friendly."""
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema.setdefault("type", "object")
    return schema


def _stringify(value: Any) -> str:
    """Coerce a tool return value into a string the model can consume."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, default=str)
    return str(value)
