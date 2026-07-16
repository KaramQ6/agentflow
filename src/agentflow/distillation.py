"""Background memory compression via LLM distillation.

Provides a lightweight token counter and a :class:`ContextDistiller` that
triggers asynchronous distillation when a session's context exceeds a
configurable token threshold.  The distiller compresses conversation history
while retaining critical information (code, URLs, JSON, numerics) verbatim.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM
    from .memory import InMemoryContext

_log = logging.getLogger("agentflow.distillation")

DISTILLATION_PROMPT = (
    "Compress the following conversation context. "
    "Retain all code snippets, URLs, JSON payloads, and numerical data VERBATIM. "
    "Discard conversational pleasantries. "
    "Output only the compressed context, with no additional commentary."
)

try:
    import tiktoken  # noqa: F811

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses ``tiktoken`` with the ``cl100k_base`` encoding when available;
    falls back to ``len(text) // 4`` otherwise.

    Args:
        text: The text to count tokens for.

    Returns:
        Estimated token count as an integer.
    """
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            _log.debug("tiktoken token counting failed, falling back to char/4")
    return len(text) // 4


class ContextDistiller:
    """Manages background distillation of oversized context sessions.

    When the total token count for a session stored in an
    :class:`~agentflow.memory.InMemoryContext` exceeds *distill_threshold_tokens*,
    the distiller spawns a background :func:`asyncio.create_task` to compress the
    content via the LLM.  The ``save_context`` return path is never blocked.

    Concurrency safety is achieved by taking a snapshot of the session under
    the memory lock, releasing the lock during the LLM call, then atomically
    writing back the distilled content only if the session has not been
    modified by another concurrent operation.

    Args:
        llm: The LLM provider used for generating the compressed summary.
        memory: The :class:`InMemoryContext` instance to monitor and compress.
        distill_threshold_tokens: Token count above which distillation is
            triggered (default 4000).
    """

    def __init__(
        self,
        llm: LLM,
        memory: InMemoryContext,
        distill_threshold_tokens: int = 4000,
    ):
        self._llm = llm
        self._memory = memory
        self._threshold = distill_threshold_tokens
        self._inflight: set[str] = set()

    async def maybe_distill(self, session_id: str) -> None:
        """Check session token count and trigger background distillation if needed.

        If a distillation is already in-flight for *session_id*, this call
        is a no-op.

        Args:
            session_id: The session identifier to check.
        """
        if session_id in self._inflight:
            return

        ctx = await self._memory.load_context(session_id)
        if not ctx:
            return

        raw_parts: list[str] = []
        for k, v in ctx.items():
            raw_parts.append(f"{k}: {v}")
        raw = "\n".join(raw_parts)

        if count_tokens(raw) <= self._threshold:
            return

        self._inflight.add(session_id)
        asyncio.create_task(self._distill(session_id))

    async def _distill(self, session_id: str) -> None:
        """Run the distillation workflow for *session_id* in the background.

        Workflow:
        1. Snapshot the session content and version under the lock.
        2. Release the lock and call the LLM to compress.
        3. Re-acquire the lock; if the version matches the snapshot,
           atomically replace the session content with the distilled text.
        """
        try:
            # Snapshot under the memory lock (using its public API).
            snapshot = await self._memory._get_session_snapshot(session_id)
            if snapshot is None:
                return
            content_snapshot, version = snapshot
            if not content_snapshot:
                return

            raw_parts: list[str] = []
            for k, v in content_snapshot.items():
                raw_parts.append(f"{k}: {v}")
            raw = "\n".join(raw_parts)

            # Call the LLM outside any lock so other operations proceed.
            messages: list[dict[str, str]] = [
                {"role": "system", "content": DISTILLATION_PROMPT},
                {"role": "user", "content": raw},
            ]
            response = await self._llm.generate(messages)
            distilled = response.content

            # Atomically write back only if unchanged.
            replaced = await self._memory._replace_distilled(
                session_id, content_snapshot, version, distilled
            )
            if replaced:
                _log.info("Distillation completed for session %s", session_id)
            else:
                _log.debug(
                    "Distillation skipped for session %s (modified concurrently)",
                    session_id,
                )
        except Exception:
            _log.exception("Distillation failed for session %s", session_id)
        finally:
            self._inflight.discard(session_id)
