# Public API & Stability Contract

This document defines exactly what agentflow promises to keep stable, how
deprecations happen, and what 1.0 requires. If a name is not listed here, it
is internal — it may change or disappear in any release without notice.

## Covered surface (semver applies)

Everything importable from the top-level `agentflow` package, i.e. the names
in `agentflow.__all__`:

| Area | Names |
|---|---|
| Core | `Agent`, `BaseAgent`, `LLM`, `Pipeline`, `SupervisorAgent` |
| Tools | `Tool`, `tool` |
| Cost | `estimate_cost`, `register_price` |
| Data models | `AgentResult`, `PipelineResult`, `LLMResponse`, `Event`, `EventEmitter` |
| Memory | `BaseMemory`, `InMemoryContext`, `RedisContext`, `VectorContext` |
| Ops | `RateLimiter`, `ResponseCache`, `InMemoryCache`, `RedisCache` |
| Observability | `Hooks`, `LoggingHooks`, `PipelineLogger`, `get_logger` |
| Exceptions | `AgentFlowError`, `AgentError`, `AgentTimeoutError`, `AgentOutputValidationError`, `BudgetExceededError`, `PipelineError`, `LLMError`, `ToolError` |
| HITL | `ApprovalPolicy`, `PauseExecution` |

Covered semantics (not just names):

- `Pipeline.add/run/resume/stream` signatures and event types.
- The context contract: each agent receives only its declared dependencies;
  values are `str`, or `dict` when the upstream agent declared an
  `output_schema` (its validated output).
- `LLMResponse` attribute names, and cost/token accounting fields on
  `AgentResult` / `PipelineResult`.

## Opt-in modules (best-effort, NOT covered by semver)

Importable by full path, excluded from the stability contract:

- `agentflow.sandbox` — code-execution sandboxes (this is a security
  surface; review it before use)
- `agentflow.triggers` — event-driven daemon triggers (MQTT)
- `agentflow.distillation` — background memory compression
- `agentflow.contrib.*` — third-party bridges (e.g. `contrib.otel.OTelHooks`)

## Current deprecations

| Deprecated | Use instead | Removal |
|---|---|---|
| Dict-style access on `LLMResponse` (`response["content"]`) | Attribute access (`response.content`) | 1.0 |
| `PipelineResult.total_duration` | `agent_seconds` (summed agent time) or `wall_time` (elapsed) | 1.0 |
| `set_session` / `set_approval_policy` on agents | `execute(..., session_id=, approval_policy=)` — mutating shared instances is unsafe under concurrency | 1.0 |
| Top-level imports of sandbox/trigger names | Full-path imports (`agentflow.sandbox`, `agentflow.triggers`) | done in 0.6 |

## Deprecation policy

- A covered name or behavior is deprecated in a minor release (documented
  here + docstring note) and kept working for at least one further minor
  release before removal in the next major.
- Pre-1.0 caveat: breaking changes may land in minor versions, but only
  with an entry in CHANGELOG.md and a migration note in this file.

## What 1.0 requires

1. The typed data plane finalized (`LLMResponse`, typed context contract).
2. Three consecutive minor releases with zero breaking changes to the
   covered surface.
3. Provider compatibility matrix (OpenAI, Groq, Ollama, OpenRouter, vLLM)
   exercised in CI with recorded responses.
4. All current deprecations removed.
5. Version/tag/changelog discipline: every release tagged, changelog entry,
   `agentflow.__version__` sourced from package metadata (done in 0.6).
