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
| Exceptions | `AgentFlowError`, `AgentError`, `AgentTimeoutError`, `AgentOutputValidationError`, `BudgetExceededError`, `PipelineError`, `LLMError`, `ToolError` |
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
  `PipelineResult.unpriced_models`; it never raises
  (see [ADR 0003](docs/adr/0003-unpriced-models-report-zero-loudly.md)).
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
- `agentflow.swarm_routing` — `DynamicSupervisorAgent`, a supervisor with
  Pydantic-generated delegation schemas, strict worker context isolation and
  depth-capped delegation
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

## Deprecation policy

- A covered name or behavior is deprecated in a minor release (documented
  here + docstring note) and kept working for at least one further minor
  release before removal in the next major.
- Pre-1.0 caveat: breaking changes may land in minor versions, but only
  with an entry in CHANGELOG.md and a migration note in this file.

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
