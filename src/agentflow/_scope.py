"""Run-scoped state that reaches every LLM call of a run without changing a signature.

Both variables are ``ContextVar``s: asyncio copies the context into every task, so
concurrent runs on one ``Pipeline`` never see each other's state, while retries,
supervisor workers and resumed agents inside one run all see the same scope.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .exceptions import BudgetExceededError

if TYPE_CHECKING:
    from .effects import EffectLedger


@dataclass
class Spend:
    """Billed usage. ``add`` never awaits, so it cannot interleave under asyncio."""

    cost: float = 0.0
    tokens: int = 0

    def add(self, cost: float, tokens: int) -> None:
        self.cost += cost
        self.tokens += tokens


class Budget:
    """A hard spending ceiling for one run, enforced by reservation.

    Each LLM call reserves its worst-case cost before it is sent and settles to the
    billed amount afterwards. A reservation that does not fit waits while other calls
    are in flight (they may settle below their reservation), and fails once nothing is.
    The failure is sticky: every later call fails the same way, so an agent that
    swallows the error cannot keep spending, and the pipeline can re-raise it.

    Single event loop only; no method awaits between a check and its update.
    """

    def __init__(self, limit: float, spent: float = 0.0):
        self.limit = limit
        self.spent = spent
        self.reserved = 0.0
        self.blocked: Exception | None = None
        self._settled = asyncio.Event()

    def block(self, error: Exception) -> Exception:
        if self.blocked is None:
            self.blocked = error
        return self.blocked

    async def reserve(self, amount: float) -> None:
        while True:
            if self.blocked is not None:
                raise self.blocked
            if self.spent + self.reserved + amount <= self.limit:
                self.reserved += amount
                return
            if self.reserved <= 0:
                raise self.block(BudgetExceededError(self.limit, round(self.spent, 6)))
            self._settled.clear()
            await self._settled.wait()

    def settle(self, reserved: float, billed: float) -> None:
        self.reserved -= reserved
        self.spent += billed
        self._settled.set()


@dataclass
class RunScope:
    """State of one pipeline run.

    ``discarded`` is spend whose agent result was thrown away: failed or cancelled
    attempts. Successful results carry their own cost, so a run's total is the sum
    of its results plus ``discarded``.
    """

    run_id: str
    discarded: Spend = field(default_factory=Spend)
    budget: Budget | None = None
    effect_ledger: EffectLedger | None = None


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
