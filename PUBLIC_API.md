# Public API & Stability Contract

This document defines exactly what agentflow promises to keep stable, how
deprecations happen, and what 1.0 requires. If a name is not listed here, it
is internal — it may change or disappear in any release without notice.

## Covered surface (semver applies)

Everything importable from the top-level `agentflow` package, i.e. the names
in `agentflow.__all__`:

| Area | Names |
|---|---|
| Core | `Agent`, `AgentSpec`, `BaseAgent`, `LLM`, `Pipeline`, `SupervisorAgent` |
| Tools | `Tool`, `tool` |
| Cost | `estimate_cost`, `register_price` |
| Data models | `AgentResult`, `PipelineResult`, `LLMResponse`, `Event`, `EventEmitter` |
| Memory | `BaseMemory`, `InMemoryContext`, `RedisContext`, `VectorContext` |
| Ops | `RateLimiter`, `ResponseCache`, `InMemoryCache`, `RedisCache` |
| Observability | `Hooks`, `LoggingHooks`, `PipelineLogger`, `get_logger` |
| Exceptions | `AgentFlowError`, `AgentError`, `AgentTimeoutError`, `AgentOutputValidationError`, `BudgetExceededError`, `EffectUncertainError`, `PipelineError`, `LLMError`, `ToolError` |
| HITL | `ApprovalPolicy`, `PauseExecution` |

Covered semantics (not just names):

- `Pipeline.add/run/resume/stream/explain` signatures and event types.
- The context contract: each agent receives only its declared dependencies;
  values are `str`, or `dict` when the upstream agent declared an
  `output_schema` (its validated output). This holds on the resume path too.
- `LLMResponse` attribute names, and cost/token accounting fields on
  `AgentResult` / `PipelineResult`.
- The `EventType` Literal values (`agent_start`, `agent_complete`,
  `agent_error`, `agent_skipped`, `pipeline_complete`, `pipeline_error`,
  `pipeline_paused`) and `PipelineResult.status`
  (`"completed"` / `"paused"`).
- Error precedence within a DAG level: a real agent failure raises even when
  a sibling paused for HITL approval (the pause is not persisted).
- Failure cancels the rest of the level; a `PauseExecution` does not
  (see [ADR 0001](docs/adr/0001-fail-fast-dag-levels.md)).
- `BudgetExceededError.partial_result` carries the completed levels.
- `output_schema` puts the schema in the prompt and repairs an invalid reply
  `output_retries` times before raising
  (see [ADR 0002](docs/adr/0002-structured-output-via-prompt-and-repair.md)).
- An unpriced model costs `0.0` and is reported in
  `PipelineResult.unpriced_models`
  (see [ADR 0003](docs/adr/0003-unpriced-models-report-zero-loudly.md)). It raises
  only when `budget_usd` is set, because a budget must be measurable
  (see [ADR 0005](docs/adr/0005-hard-budget-by-reservation.md)). *Changed in 0.8.0.*
- `Hooks.on_pipeline_error` and `Hooks.on_tool_error` (no-op defaults) and
  `PipelineResult.tool_errors` report failures that `status` does not. A started run
  ends with `on_pipeline_end` or `on_pipeline_error`; `stream()` fires the same hooks as
  `run()`. *Added in 0.8.0.*
- `LLM.generate` is the one retry layer, for transient errors only. `AgentResult.metadata`
  carries `finish_reason`, and `truncated` when a reply was cut off; an empty cut-off reply
  raises. *Changed in 0.8.0.*
- Tool arguments are validated strictly: unknown arguments and non-finite numbers are
  refused. At `max_tool_iterations` the agent makes one last call without tools; if that
  fails, `AgentError` carries `tokens`, `cost` and `trace`. *Changed in 0.8.0.*
- HITL: each blocked call is approved on its own; a rejection stops the rest of that
  reply's tool calls; an approved call runs at most once across `resume()` calls, and
  `resume(on_in_doubt=...)` decides what happens to one that never recorded a result.
  *Changed in 0.8.0.*
- A dependent of a skipped agent still runs, and the skipped agent's output is simply absent from its
  context: read it with `context.get(name)`. Optional fan-in needs this, so it is the contract and
  not a bug (PIPE-2). *Documented in 0.8.0.*
- `tool(effect=...)`: `"read"` tools may repeat and run concurrently; `"write"` tools run
  one at a time in the model's order and at most once per occurrence in a run; an agent
  that ran a tool without `effect` is not retried. See ADR 0006. *Added in 0.8.0.*
- `budget_usd` is enforced by reserving each LLM call's worst-case cost before it
  is sent, and its total includes discarded attempts. *Changed in 0.8.0.*
- `LLM.generate(**extra)` forwards unknown keywords to the provider unchanged,
  and they participate in the cache key.
- `depends_on` accepts forward references; unknown names raise when the graph
  is resolved, not when `add()` is called.

`PipelineResult.output` is the output of the last agent recorded in the final
executed level, in `add()` order. For a DAG whose last level holds several
agents this is well defined but rarely what you want — read
`result.results[name]` instead.

## Opt-in modules (best-effort, NOT covered by semver)

Importable by full path, excluded from the stability contract:

- `agentflow.sandbox` — code-execution sandboxes (this is a security
  surface; review it before use)
- `agentflow.triggers` — event-driven triggers and the MQTT daemon
  (`BaseTrigger`, `MQTTTrigger`, `TriggerPolicy`, `PydanticTriggerPolicy`,
  `MQTTDaemon`)
- `agentflow.swarm_routing` — `DynamicSupervisorAgent`. **Deprecated in 0.8.0,
  removed in 0.9.0** (see below)
- `agentflow.effects` — effect ledgers (`InMemoryLedger`, `SqliteLedger`)
- `agentflow.testing` — `ScriptedLLM`, `reply`, `call` for tests
- `agentflow.contrib.*` — third-party bridges (e.g. `contrib.otel.OTelHooks`)

## Current deprecations

Deprecated names emit `DeprecationWarning` as of 0.6.

| Deprecated | Use instead | Removal |
|---|---|---|
| `agentflow.events.MQTTDaemon`, `TriggerPolicy`, `PydanticTriggerPolicy` | Same names from `agentflow.triggers` (moved in 0.7; these were never exported, so no shim) | done in 0.7 |
| `agentflow.distillation` | Removed in 0.7 — never exported, never tested, never documented | done in 0.7 |
| Dict-style access on `LLMResponse` (`response["content"]`) | Attribute access (`response.content`) | 1.0 |
| `PipelineResult.total_duration` | `agent_seconds` (summed agent time) or `wall_time` (elapsed) | 1.0 |
| `set_session` / `set_approval_policy` on agents | `execute(..., session_id=, approval_policy=)` — mutating shared instances is unsafe under concurrency | 1.0 |
| The `_DecoratorAgent` name | `AgentSpec` (same class; renamed in 0.6) | 1.0 |
| Top-level imports of sandbox/trigger names | Full-path imports (`agentflow.sandbox`, `agentflow.triggers`) | done in 0.6 |
| `agentflow.swarm_routing` (`DynamicSupervisorAgent`) | `SupervisorAgent`. Its depth cap never fired (`depth=0` is hard-coded), it crashed on structured context, and it shared cost across runs. Importing it warns as of 0.8.0 | 0.9.0 |

## Deprecation policy

- A covered name or behavior is deprecated in a minor release (documented
  here + docstring note) and kept working for at least one further minor
  release before removal in the next major.
- Pre-1.0 caveat: breaking changes may land in minor versions, but only
  with an entry in CHANGELOG.md and a migration note in this file.

## Migrating to 0.8.0

0.8.0 changes behaviour in the places below. The pre-1.0 caveat above requires this note.
The full list, with the reasons, is in [CHANGELOG.md](CHANGELOG.md).

| If you... | Then 0.8.0... | Do |
|---|---|---|
| Set `budget_usd` with a model that has no registered price | Raises `PipelineError` before the first call | `register_price(model, 0, 0)` for a local model, or its real price |
| Have tools that take fewer arguments than the model sends, or pass NaN or Infinity | Refuses the call with an error observation | Declare the argument, or let the model correct itself |
| Use `retry_failed_agents` with tools that have no `effect` | Does not retry an agent that ran one | Set `effect="read"` or `effect="write"` on each tool |
| Rely on a 600 s request timeout | Times out at 120 s | `LLM(timeout=600)` |
| Have runs paused by 0.7.x | Cannot resume them | Resume or drop them before upgrading |
| Run `MQTTDaemon` on a noisy topic | Fires once per alarm, then at most every 30 s, 5 at a time | `cooldown_s=0, max_concurrent=None` |
| Import `agentflow.swarm_routing` | Warns, and removes it in 0.9.0 | `SupervisorAgent` |
| Subclass `LLM` and override `generate` | Skips retries, pricing and the budget | Override `_complete` |
| Parse or truncate `run_id` | Gets 32 hex characters, was 8 | Do not assume a length |

## What 1.0 requires

1. The typed data plane finalized (`LLMResponse`, typed context contract).
   *Done in 0.6.*
2. Three consecutive minor releases with zero breaking changes to the
   covered surface. **Not started** — 0.7 changes `add()`'s validation timing
   and level-failure semantics, so the count begins after it.
3. Provider compatibility matrix (OpenAI, Groq, Ollama, OpenRouter, vLLM)
   exercised in CI with recorded responses. **Not started.**
4. All current deprecations removed.
5. Version/tag/changelog discipline: every release tagged, changelog entry,
   `agentflow.__version__` sourced from package metadata. *Done in 0.6.*
6. Green CI on a tagged commit. **Blocked** — GitHub Actions has not executed
   for this repository since 0.5.0 (account billing), so no release since then
   has been verified by CI. Every gate is runnable locally in the meantime:
   `ruff check`, `mypy src/agentflow/`, `pytest` (90% coverage gate),
   `mkdocs build --strict`, `python scripts/check_sdist.py`.
