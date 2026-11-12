"""Effect ledger: a ``write`` tool runs at most once per occurrence in a run.

Opt-in module, outside the semver contract (see ``PUBLIC_API.md``). The pipeline
always records ``@tool(effect="write")`` calls in a per-run :class:`InMemoryLedger`,
which covers retries inside one process. A durable ledger also covers ``resume()``
in a new process::

    from agentflow import Pipeline, tool
    from agentflow.effects import SqliteLedger

    @tool(effect="write")
    async def charge_card(customer: str, amount: int) -> str:
        ...

    pipeline = Pipeline(llm=llm, retry_failed_agents=2, effect_ledger=SqliteLedger("effects.db"))

See ADR 0004 (as amended) and ADR 0006.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .exceptions import EffectUncertainError
from .tools import Tool

__all__ = [
    "EffectLedger",
    "EffectRecord",
    "EffectStatus",
    "EffectUncertainError",
    "InMemoryLedger",
    "SqliteLedger",
]

_log = logging.getLogger("agentflow.effects")


class EffectStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class EffectRecord:
    key: str
    status: EffectStatus
    result: str


class EffectLedger(Protocol):
    """Where effect records live. Both methods must be atomic per key."""

    async def claim(self, key: str) -> EffectRecord | None:
        """Take ownership of ``key``.

        With no record, or a ``FAILED`` one, write ``PENDING`` and return ``None``:
        the caller owns the effect and must run it. Otherwise return the existing
        record unchanged.
        """
        ...

    async def settle(self, key: str, status: EffectStatus, result: str) -> None: ...


class InMemoryLedger:
    """One-process ledger. Covers retries inside a run; lost on restart."""

    def __init__(self) -> None:
        self._records: dict[str, EffectRecord] = {}
        self._lock = asyncio.Lock()

    async def claim(self, key: str) -> EffectRecord | None:
        async with self._lock:
            record = self._records.get(key)
            if record is None or record.status is EffectStatus.FAILED:
                self._records[key] = EffectRecord(key, EffectStatus.PENDING, "")
                return None
            return record

    async def settle(self, key: str, status: EffectStatus, result: str) -> None:
        async with self._lock:
            self._records[key] = EffectRecord(key, status, result)


class SqliteLedger:
    """Durable ledger in one SQLite file (stdlib ``sqlite3``).

    Survives restarts, so it also covers ``resume()`` in a new process. Several
    processes may share the file: ``claim`` takes the write lock before reading.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = os.fspath(path)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, "
                "status TEXT NOT NULL, result TEXT NOT NULL, updated_at REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        # timeout: a concurrent writer waits for the lock instead of failing at once.
        # isolation_level=None: transactions are opened explicitly below.
        return sqlite3.connect(self._path, timeout=5.0, isolation_level=None)

    async def claim(self, key: str) -> EffectRecord | None:
        return await asyncio.to_thread(self._claim, key)

    async def settle(self, key: str, status: EffectStatus, result: str) -> None:
        await asyncio.to_thread(self._write, key, status, result)

    def _claim(self, key: str) -> EffectRecord | None:
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT status, result FROM effects WHERE key = ?", (key,)).fetchone()
                if row is not None and row[0] != EffectStatus.FAILED.value:
                    record: EffectRecord | None = EffectRecord(key, EffectStatus(row[0]), row[1])
                else:
                    record = None
                    self._upsert(conn, key, EffectStatus.PENDING, "")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return record

    def _write(self, key: str, status: EffectStatus, result: str) -> None:
        with closing(self._connect()) as conn:
            self._upsert(conn, key, status, result)

    @staticmethod
    def _upsert(conn: sqlite3.Connection, key: str, status: EffectStatus, result: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO effects (key, status, result, updated_at) VALUES (?, ?, ?, ?)",
            (key, status.value, result, time.time()),
        )


@dataclass
class EffectScope:
    """One agent conversation in one run: what makes two identical calls the same effect.

    ``occurrences`` counts write calls per (tool, validated arguments) in this
    conversation. It is part of the key, so repeating a command is a new effect
    while a retry, which replays the conversation from the start, maps each call
    back to the effect it already made. Write calls are serialized, so the count
    follows the model's order.
    """

    ledger: EffectLedger
    run_id: str
    agent: str
    occurrences: dict[str, int] = field(default_factory=dict)
    ran_unclassified: set[str] = field(default_factory=set)

    def next_key(self, tool_name: str, kwargs: Mapping[str, Any]) -> str:
        call = _canonical(tool_name, kwargs)
        n = self.occurrences.get(call, 0)
        self.occurrences[call] = n + 1
        payload = json.dumps([self.run_id, self.agent, call, n], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def replay_history(self, tools: Mapping[str, Tool], trace: Iterable[Mapping[str, Any]]) -> None:
        """Rebuild the counts from a restored conversation, so a resume continues them."""
        for entry in trace:
            target = tools.get(entry.get("tool", ""))
            if target is None or target.effect != "write":
                continue
            try:
                kwargs = target._validate(entry.get("arguments", "{}"))
            except Exception:  # noqa: BLE001 - an invalid call never reached the ledger
                continue
            call = _canonical(target.name, kwargs)
            self.occurrences[call] = self.occurrences.get(call, 0) + 1


def _canonical(tool_name: str, kwargs: Mapping[str, Any]) -> str:
    return json.dumps([tool_name, kwargs], sort_keys=True, default=str, separators=(",", ":"))


async def run_tool(target: Tool, arguments: dict[str, Any] | str, scope: EffectScope | None) -> str:
    """Call *target*. A ``write`` tool goes through the ledger, at most once per occurrence.

    Raises:
        ToolError: Invalid arguments (nothing recorded) or the tool failed
            (recorded ``FAILED``, so a later attempt runs it again).
        EffectUncertainError: An earlier attempt started this effect and never finished.
        Exception: Whatever the ledger raises. The tool does not run (fail closed).
    """
    if scope is None or target.effect != "write":
        if scope is not None and target.effect is None:
            scope.ran_unclassified.add(target.name)
        return await target.acall(arguments)
    kwargs = target._validate(arguments)
    key = scope.next_key(target.name, kwargs)
    record = await scope.ledger.claim(key)
    if record is not None:
        if record.status is EffectStatus.DONE:
            _log.info("tool %r replayed from the effect ledger (key %s)", target.name, key)
            return record.result
        raise EffectUncertainError(scope.agent, target.name, key)
    try:
        output = await target._invoke(kwargs)
    except Exception as e:
        await scope.ledger.settle(key, EffectStatus.FAILED, str(e))
        raise
    # CancelledError is not an Exception: the record stays PENDING on purpose.
    await scope.ledger.settle(key, EffectStatus.DONE, output)
    return output


_current: ContextVar[EffectScope | None] = ContextVar("agentflow_effect_scope", default=None)


def current_effects() -> EffectScope | None:
    return _current.get()


@contextmanager
def using_effects(scope: EffectScope | None) -> Iterator[EffectScope | None]:
    """Route every tool call made inside this block, in this task and its children."""
    token = _current.set(scope)
    try:
        yield scope
    finally:
        _current.reset(token)
