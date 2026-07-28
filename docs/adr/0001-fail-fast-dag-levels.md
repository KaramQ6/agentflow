# 0001 — Cancel sibling agents when a DAG level fails

Date: 2026-07-29 · Status: accepted

## Context

Agents at the same DAG level run through `asyncio.gather(..., return_exceptions=True)`,
which waits for every coroutine even after one has raised. When any agent in a
level fails, `Pipeline.run()` discards the whole level's results and propagates
the error — so the siblings that kept running produced output that is thrown
away, having spent real tokens and real wall-clock time to make it.

This is worse here than in a generic task runner. The library's stated wedge is
that every run tells you what it cost; spending money on discarded output is
the specific failure the pitch says it prevents. A five-agent level where one
agent fails immediately could bill for four full completions.

The complication is `PauseExecution`. Human-in-the-loop approval is implemented
as an exception, and the pause path deliberately collects and persists the
results of every sibling in the level so the run can resume from them. Treating
it as a failure would cancel exactly the work the resume depends on.

## Decision

A level cancels its outstanding agents as soon as one finishes with a real
error. `PauseExecution` is not a real error: it is control flow, and a level
containing a pause runs to completion as before.

Cancelled siblings are reported positionally as `asyncio.CancelledError`, and
the error-selection pass skips them, so the exception that surfaces always
names the agent that actually failed rather than whichever cancellation
happened to sort first.

An outer cancellation — a `stream()` consumer abandoning the generator — also
cancels the level's in-flight tasks rather than leaving them running.

## Consequences

Easier: a failing run stops costing money at the point of failure. Abandoning a
stream no longer leaks agents. The error a user sees is the causal one.

Harder: an agent's side effects may now be interrupted part-way. Agents that
write to external systems must already tolerate this — a timeout or a process
death could always cut them off — but the window is now wider in practice. This
is documented rather than mitigated: making agent side effects transactional is
the agent author's problem, not the orchestrator's.

Also harder: a caller who wanted a sibling's result *despite* the level failing
can no longer get it. Nothing exposed it before either (the results were
discarded), so no API regresses, but the option is now foreclosed.

Exit path: the cancellation is one function, `_gather_level`. Restoring the old
behaviour is deleting it and calling `asyncio.gather` again. If a real use case
appears for "let siblings finish", it becomes a `Pipeline(fail_fast=False)`
flag rather than a rewrite.
