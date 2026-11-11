"""Run-scoped state that reaches every LLM call of a run without changing a signature.

Both variables are ``ContextVar``s: asyncio copies the context into every task, so
concurrent runs on one ``Pipeline`` never see each other's state, while retries,
supervisor workers and resumed agents inside one run all see the same scope.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class Spend:
    """Billed usage. ``add`` never awaits, so it cannot interleave under asyncio."""

    cost: float = 0.0
    tokens: int = 0

    def add(self, cost: float, tokens: int) -> None:
        self.cost += cost
        self.tokens += tokens


@dataclass
class RunScope:
    """State of one pipeline run.

    ``discarded`` is spend whose agent result was thrown away: failed or cancelled
    attempts. Successful results carry their own cost, so a run's total is the sum
    of its results plus ``discarded``.
    """

    run_id: str
    discarded: Spend = field(default_factory=Spend)


_scope: ContextVar[RunScope | None] = ContextVar("agentflow_run_scope", default=None)
_trackers: ContextVar[tuple[Spend, ...]] = ContextVar("agentflow_spend_trackers", default=())


def new_run_id() -> str:
    """32 hex chars. 8 collide at about 65k runs, and durable ledgers key on this."""
    return uuid.uuid4().hex


def current_scope() -> RunScope | None:
    return _scope.get()


@contextmanager
def run_scope(scope: RunScope) -> Iterator[RunScope]:
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


@contextmanager
def tracking(spend: Spend) -> Iterator[Spend]:
    """Add every call billed inside this block, in this task and its children, to *spend*."""
    token = _trackers.set((*_trackers.get(), spend))
    try:
        yield spend
    finally:
        _trackers.reset(token)


def record_spend(cost: float, tokens: int) -> None:
    """Called by ``LLM.generate`` for every billed provider response."""
    for spend in _trackers.get():
        spend.add(cost, tokens)
