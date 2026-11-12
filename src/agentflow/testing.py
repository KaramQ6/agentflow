"""Test helpers: run pipelines against a script instead of a provider, at zero cost.

Opt-in module, outside the semver contract (see ``PUBLIC_API.md``)::

    from agentflow import Agent, Pipeline
    from agentflow.testing import ScriptedLLM, call, reply

    llm = ScriptedLLM([
        reply(calls=[call("1", "get_pose")]),
        reply("The turtle is at (5, 5)."),
    ])
    pipe = Pipeline(llm=llm)

``ScriptedLLM`` replaces only the provider round trip (``LLM._complete``), so
``generate()`` runs unchanged around it: caching, budget reservation, pricing and
the run's cost ledger behave exactly as they do against a real provider.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from collections.abc import Iterable
from typing import Any

from .exceptions import LLMError
from .llm import LLM
from .types import LLMResponse

__all__ = ["ScriptedLLM", "call", "reply"]


def call(call_id: str, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """One tool call, in the shape a provider returns it."""
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments or {})}}


def reply(
    content: str = "",
    calls: list[dict[str, Any]] | None = None,
    *,
    prompt_tokens: int = 6,
    completion_tokens: int = 4,
    finish_reason: str | None = None,
) -> LLMResponse:
    """One model reply. ``finish_reason`` defaults to what a provider would send."""
    return LLMResponse(
        content=content,
        tokens=prompt_tokens + completion_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_calls=calls,
        finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
    )


class ScriptedLLM(LLM):
    """An :class:`~agentflow.LLM` that plays back a list of replies.

    Each entry is an :class:`~agentflow.LLMResponse` (see :func:`reply`), a string
    (a plain text reply), or an exception to raise. Every request is recorded in
    :attr:`requests` for assertions.
    """

    def __init__(self, replies: Iterable[LLMResponse | str | Exception], *, model: str = "scripted",
                 **kwargs: Any):
        kwargs.setdefault("max_retries", 0)
        super().__init__(model=model, api_key="scripted", **kwargs)
        self._script: deque[LLMResponse | str | Exception] = deque(replies)
        self.requests: list[dict[str, Any]] = []

    async def _complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> LLMResponse:
        self.requests.append({"model": model, "messages": copy.deepcopy(messages), **copy.deepcopy(params)})
        if not self._script:
            raise LLMError(f"ScriptedLLM: the script ran out after {len(self.requests) - 1} replies")
        entry = self._script.popleft()
        if isinstance(entry, Exception):
            raise entry
        if isinstance(entry, str):
            entry = reply(entry)
        return entry.model_copy(update={"model": entry.model or model})
