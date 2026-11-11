# 0005 - The budget is enforced by reservation, and must be measurable

Date: 2026-11-11 · Status: accepted · Narrows: [0003](0003-unpriced-models-report-zero-loudly.md)

## Context

`budget_usd` was documented as a hard ceiling, but it was checked once after each DAG
level, from the costs on the surviving agent results. Measured on 0.7.1:

- A level of 10 parallel agents on a $0.10 budget spent $1.00 before the check ran (COST-2).
- A ReAct loop is one agent, so the check never ran inside it: 41 calls and $4.10 on a $0.10
  budget (COST-3).
- An agent attempt that failed and was retried billed real money that no result carried,
  so neither the total nor the check saw it (COST-1).
- A model with no price costs `0.0` by ADR 0003, so the budget silently never fired: 10M
  tokens on a $0.000001 budget, `status=completed` (COST-4).

A ceiling that is checked after the money is spent, against numbers that omit some of the
money, is not a ceiling.

## Decision

1. **Reserve before every call.** `LLM.generate` reserves the worst case before it sends a
   request: the estimated prompt (3 characters per token, deliberately pessimistic) plus
   `max_tokens` of completion, at the model's price. After the call it settles to the billed
   amount. A cache hit reserves nothing.
2. **Wait, then refuse.** A reservation that does not fit waits while other calls are in
   flight, because they usually settle below their reservation. When nothing is in flight
   and it still does not fit, the call raises `BudgetExceededError`.
3. **The refusal is sticky for the run.** Every later call in the run raises the same error,
   so an agent that catches or wraps it (a supervisor turning a worker failure into an
   observation, the tool loop wrapping it in `AgentError`) cannot keep spending. The pipeline
   re-raises the original error and does not retry the agent: retrying cannot make a full
   budget fit.
4. **One measurement per run.** The run scope (`_scope.py`) counts every billed call,
   including attempts whose result was discarded, across a HITL pause and `resume()`.
   `BudgetExceededError.spent_usd` and `partial_result.total_cost` use it.
5. **A budget must be measurable.** With `budget_usd` set, a model without a registered price
   raises `PipelineError` before any request, naming the model and the fix. A local or free
   model is declared with `register_price(model, prompt_per_1m=0, completion_per_1m=0)`.
   Without a budget, ADR 0003 holds unchanged: unpriced models cost `0.0`, loudly.
6. The check after each level stays as a backstop for agents that report cost without going
   through `LLM.generate`.

## Consequences

- The ceiling holds against parallel siblings and inside tool loops. The overshoot is bounded
  by how far a reply exceeds its own reservation, which only happens if the 3-characters-per-
  token estimate is wrong for a prompt.
- Runs near their budget stop earlier than before, by up to one reservation.
- A large `max_tokens` makes reservations large. A run whose budget is smaller than one
  reservation stops before its first call. That is correct, and the error says how much was
  spent (nothing).
- Every Ollama user who sets a budget needs one `register_price` line. The quickstart shows it.
- An `LLM` subclass that overrides `generate` bypasses both reservation and measurement. The
  replacement seam for tests and custom providers is `_complete` (0.8.0, step 6).
- The provider may bill a more specific model id than the one requested. Reservation uses the
  requested id; billing uses the reported one. If only the requested id is priced, the call is
  measured at `0.0`. Not observed with the built-in price table, which matches by prefix.
