# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.7.0] — 2026-07-29

Correctness and honesty release. Three things this fixes were actively
misleading: cost tracking reported `$0.00` on the provider the quickstart
recommends, two shipped examples could not run at all, and the coverage gate
claimed 90% while the suite measured 78%.

**Upgrading.** Two behaviour changes are worth reading before you take this:

- A failing DAG level now cancels its remaining agents. If you relied on
  siblings completing after a failure, they no longer will — their results were
  discarded anyway.
- `pipe.add(agent, depends_on=["not_added_yet"])` no longer raises inside
  `add()`. Unknown dependencies raise when the graph is resolved
  (`run`/`stream`/`explain`). If you asserted on that early error, move the
  assertion.

`agentflow.distillation` is gone. It was never exported; if you imported it by
path, pin `0.6.1` or lift the module out of git history.

### Added
- **Cost visibility for unpriced models.** A model with no pricing entry still
  costs `$0.00`, but it is no longer silent: it is logged once per process, and
  `PipelineResult.unpriced_models` lists every model in a run whose cost is a
  placeholder. Priced the providers the docs actually recommend — Groq
  (llama/gemma), DeepSeek, Mistral, `claude-3-7-sonnet` — which previously all
  reported `total_cost: $0.000000`. See ADR 0003.
- **`BudgetExceededError.partial_result`** — the agents that completed before
  the ceiling tripped, instead of losing work already paid for.
- **Structured output that repairs itself.** `output_schema` now sends the JSON
  Schema to the model and, on a validation failure, shows it the errors and
  asks for a correction (`@Agent(output_retries=1)` by default, `0` disables).
  Repairs are billed to the agent and the budget. Markdown-fenced JSON is
  unwrapped locally rather than costing a round-trip. See ADR 0002.
- **`LLM.generate(**extra)`** — forwards any keyword to the provider (`seed`,
  `top_p`, `stop`, `response_format`, `extra_body`). Included in the cache key.
- **`Pipeline.explain()`** — renders the resolved DAG (levels, parallelism,
  dependencies, timeouts, conditional nodes) without running anything.
- **`Pipeline(max_concurrency=...)`** — caps agents in flight per run.
- **`scripts/check_sdist.py`** — fail-closed allowlist for source-distribution
  contents, the gate that was missing when 0.6.0 leaked a local directory.
- Tests for `contrib.otel` and the whole MQTT surface, which had none.

### Changed
- **A failing DAG level cancels its remaining agents** instead of waiting for
  them to finish producing output that is discarded anyway. A `PauseExecution`
  is control flow and still lets its siblings complete. See ADR 0001.
- **`depends_on` accepts forward references.** Unknown names now raise when the
  graph is resolved (`run`/`stream`/`explain`) rather than inside `add()`, so
  you are not forced to topologically sort your own source file. The message
  lists the agents that do exist.
- `ResponseCache.make_key()` takes an `extra` keyword so provider-specific
  parameters cannot collide in the cache.
- Trigger policies and `MQTTDaemon` moved from `agentflow.events` to
  `agentflow.triggers`, next to `BaseTrigger`/`MQTTTrigger`. `events` is now
  just the pipeline event emitter. These names were never exported.
- CI installs the `otel` and `mqtt` extras and builds the docs with
  `--strict`; `sandbox.py` is excluded from the coverage gate because its main
  paths need a live Docker daemon. Coverage is 93% against a 90% gate (it was
  78%, i.e. the gate would have failed the moment CI ran).

### Fixed
- **Two broken examples.** `robotics_mqtt_agent.py` and
  `drone_telemetry_agent.py` used `@Agent(timeout=)`, `LLM(provider=)` and
  `hooks=PipelineLogger(...)` — none of which exist — and imported a name
  dropped in 0.6. `tests/test_examples.py` now imports every example in CI.
- **`MQTTDaemon` busy-loop.** A cleanly-ended message stream caused an
  immediate reconnect with no await, spinning the loop and starving the event
  loop. A clean end now backs off like a dropped connection.
- **`PydanticTriggerPolicy` prompt crash.** The `KeyError` fallback in
  `build_task_prompt()` re-raised the same `KeyError`, and a payload model with
  a `data` field raised `TypeError`. Templates now render leniently: an unknown
  placeholder is left visible and logged instead of killing the daemon.
- Abandoning a `stream()` generator now cancels the agents still in flight.

### Removed
- **`agentflow.distillation`** and the `InMemoryContext.enable_distillation()`
  hook it served. 160 lines with no tests, no coverage, no export and no docs —
  unreachable without importing a private module. Recoverable from git history
  if it is ever wanted back, with tests. `swarm_routing.py` is also unexported
  but is 92% covered, so it stays and is now documented as an opt-in module.

---

## [0.6.1] — 2026-07-16

Packaging-hygiene release. **0.6.0 has been yanked from PyPI.** Its source
distribution (`.tar.gz`) accidentally bundled a local, untracked
`.bridgespace/` agent-workspace directory whose contents included a leaked
PyPI API token (since revoked by PyPI). The Python package code was
identical and unaffected; only the sdist contained the stray directory.

### Fixed
- **Security/packaging**: `.bridgespace/` (a local multi-agent workspace, never
  part of the library) was not excluded from the build and leaked into the
  0.6.0 sdist. It is now git-ignored and excluded from all distributions. The
  wheel was never affected. If you installed 0.6.0, upgrade to 0.6.1.

---

## [0.6.0] — 2026-07-16

First release shipping a pure-Python `py3-none-any` wheel. The only wheel
published for 0.5.0 was a Windows/CPython-3.11 binary left over from the
(since removed) C++ extension era — every other platform fell back to an
sdist build. If 0.4.0/0.5.0 failed to install for you, this release is the fix.

### Added
- **Typed data plane**: `LLM.generate()` returns a typed `LLMResponse` pydantic
  model. Dict-style access is kept as a deprecated shim (see Deprecated).
- **Validated output flows downstream**: when an agent declares
  `output_schema`, its validated output (`AgentResult.data`) is what downstream
  agents receive in `context` — not the raw JSON string.
- **Cost budgets**: `Pipeline(budget_usd=0.25)` enforces a hard USD ceiling per
  run, raising `BudgetExceededError` when exceeded (checked after each level).
- **OpenTelemetry adapter**: `agentflow.contrib.otel.OTelHooks` (install the
  `otel` extra) — one import to spans.
- **`AgentSpec`** is the public name of the object `@Agent` returns (previously
  the private `_DecoratorAgent`); `@Agent(system_prompt=...)` fully replaces
  the default role-based system prompt.
- **`wall_time`** (elapsed) and **`agent_seconds`** (summed agent time) on
  `PipelineResult`.
- **`PUBLIC_API.md`** stability contract and **`SECURITY.md`**.
- Showcase example `examples/earnings_triage.py`: diamond DAG with tools,
  parallel analysts, typed output, budget, and cache — zero-key via Ollama/Groq.
- CI test matrix extended to Python 3.13 and 3.14.

### Fixed
- `SupervisorAgent` crashed with `TypeError` when an upstream context value was
  a validated-output dict.
- `SupervisorAgent` billing race: concurrent runs sharing one instance
  corrupted each other's worker token/cost accounting (now a run-scoped ledger).
- A real agent failure alongside an HITL pause in the same DAG level was
  silently swallowed; errors now take precedence and raise (the discarded
  pause is logged).
- Resumed agents dropped their validated `output_schema` result;
  `AgentResult.data` is now populated on the resume path too.
- `Pipeline.serve()` silently discarded the trigger's `context_data`; it is now
  appended to the task prompt as a JSON block.
- `Pipeline.resume()` could raise a spurious `ValidationError` when the last
  saved context value was a dict.
- Dependency-cycle errors now name the agents involved.

### Changed
- `run()`, `resume()`, and `stream()` now share a single execution driver
  (previously three divergent copies of the same loop — the source of the
  pause/error bugs above). `resume()` and `stream()` fire the same
  `on_agent_start`/`on_agent_end`/`on_agent_error` hooks as `run()`.
- `stream()` persists HITL pause state under a real `run_id` distinct from the
  session id.
- `Event.type` and `EventEmitter.emit` are typed with the new `EventType`
  Literal (now includes `pipeline_paused`); `PipelineResult.status` is
  `Literal["completed", "paused"]`.
- `__version__` is single-sourced from package metadata (`importlib.metadata`).
- Sandbox and trigger names are no longer re-exported from the top-level
  package — import via `agentflow.sandbox` / `agentflow.triggers` (best-effort
  modules outside the semver contract).
- The C++ DAG engine and its build scaffolding were removed entirely: DAG
  resolution is microseconds against multi-second LLM calls.

### Deprecated (warn since 0.6, removal at 1.0)
- Dict-style access on `LLMResponse` (`response["content"]`, `.get(...)`) —
  use attribute access.
- `set_session()` / `set_approval_policy()` — pass `session_id=` /
  `approval_policy=` to `execute()`; mutating shared agent instances is unsafe
  under concurrency.
- `PipelineResult.total_duration` — use `agent_seconds` or `wall_time`.
- The `_DecoratorAgent` name — use `AgentSpec`.

---

## [0.5.0] — 2026-07-03 (retroactive entry)

Written after the fact on 2026-07-16: 0.4.0 and 0.5.0 shipped without
changelog entries, violating this file's own policy. Reconstructed from git
history. **Packaging note:** the only 0.5.0 wheel on PyPI is
`cp311-cp311-win_amd64`; every other platform builds from the sdist. Prefer 0.6.0.

### Added
- **Swarm routing**: `SupervisorAgent` delegates sub-tasks to worker agents via
  a generated `delegate_task` tool; worker tokens/cost bubble up to the
  supervisor's result. Unexported `swarm_routing.DynamicSupervisorAgent`
  prototype with depth-capped dynamic agent creation.
- **Background memory distillation** (`agentflow.distillation`): compresses
  long session memory with a version lock against concurrent writes.
- **Human-in-the-loop**: `ApprovalPolicy` + `PauseExecution` +
  `Pipeline.resume()` — pause on blocked tool calls, persist state to memory,
  resume with approve/reject.
- **Sandboxes** (`agentflow.sandbox`): Docker/subprocess code-execution tools.
- **MQTT triggers** (`agentflow.triggers`) and `Pipeline.serve()` daemon mode.

### Changed
- The C++ DAG engine introduced in 0.4.0 was made optional (skipped when no
  compiler is available) — and removed entirely in 0.6.0.

---

## [0.4.0] — 2026-07-02 (retroactive entry)

### Added
- C++ DAG engine via pybind11 (Kahn's algorithm) — later judged unnecessary
  and removed; see the 0.5.0/0.6.0 notes.
- Memory module (`BaseMemory`, `InMemoryContext`, `RedisContext`,
  `VectorContext`) with per-session context injection into agent prompts.
- ReAct loop hardening: duplicate-call detection, tool-output truncation,
  sliding message window, per-iteration LLM retry.

### Note
- 0.2.0 and 0.3.0 were developed and documented below but never published to
  PyPI, which jumped 0.1.0 → 0.4.0.

---

## [0.3.0]

### Added
- **Tool / function calling** (`tools.py`): the `@tool` decorator turns any sync
  or async Python function into an LLM-callable tool. Argument JSON schemas are
  generated automatically from type hints via Pydantic. Agents given `tools=[...]`
  run a bounded **ReAct loop** (call → execute tools → observe → repeat) up to
  `max_tool_iterations` (default 6). Tool errors are fed back to the model for
  recovery. New `ToolError` exception; tool-call traces recorded in
  `AgentResult.metadata["tool_calls"]`.
- **Cost tracking** (`pricing.py`): built-in USD price tables for common OpenAI
  and Anthropic models with longest-prefix matching. `LLM.generate()` returns a
  `cost` (and `prompt_tokens`/`completion_tokens`); `AgentResult.cost` and
  `PipelineResult.total_cost` aggregate spend. Cache hits bill `0.0`.
  `register_price()` / `estimate_cost()` are public.
- **Token streaming**: `LLM.astream()` yields content deltas token-by-token for
  interactive UIs (honours the rate limiter; no cache/retry mid-stream).
- **Observability hooks** (`observability.py`): `Hooks` base class + `LoggingHooks`
  wire the previously-unused `PipelineLogger` into `Pipeline.run()` (which was
  silent before). A raising hook is caught and warned, never crashing the run.
- **Production-grade retry**: unified exponential backoff with jitter and
  `Retry-After` header support. New `LLM(retry_base_delay=, retry_jitter=)` args.
- **Documentation site**: MkDocs Material + mkdocstrings under `docs/`, deployed
  via a new `docs.yml` workflow. New `docs` optional-dependency group.
- New examples: `tool_agent.py`, `streaming_and_cost.py`; and
  `benchmarks/parallel_speedup.py`.

### Fixed
- **`py.typed` marker** added — the package advertised `Typing :: Typed` but
  shipped no marker, so downstream type-checkers saw no types.
- **Red CI made green**: resolved 1 `ruff` error (B904) and 7 `mypy --strict`
  errors across `llm.py`, `cache.py`, `logging.py`, `events.py`, `pipeline.py`.

### Changed
- `__version__` bumped to `0.3.0`.
- Test coverage raised to ~91%; `fail_under` tightened from 80 → 90.
- `Pipeline.__init__` gains a `hooks` parameter; `run()` now emits lifecycle
  events and generates `run_id` up front.

---

## [0.2.0]

### Added
- **Parallel execution**: Agents at the same DAG level now run concurrently via
  `asyncio.gather()`. `_resolve_levels()` replaces `_resolve_order()` and uses
  Kahn's algorithm to group independent agents.
- **Per-agent timeout**: `Pipeline.add(timeout=N)` wraps each agent coroutine with
  `asyncio.wait_for`; raises `AgentTimeoutError` on expiry.
- **Conditional branching**: `Pipeline.add(condition=lambda ctx: ...)` allows
  dynamic skipping of agents based on upstream outputs. Skipped agents emit
  `agent_skipped` events in streaming mode.
- **Pipeline-level retry**: `Pipeline(retry_failed_agents=N)` retries failed
  agents up to N times with exponential backoff (1s, 2s, 4s). Timeouts are
  non-retriable.
- **LLM response caching** (`cache.py`): `ResponseCache` ABC with `InMemoryCache`
  (SHA-256 key, lazy TTL eviction, max-size LRU) and `RedisCache` (optional dep).
  Cache is wired into `LLM(cache=...)`.
- **Rate limiting** (`rate_limiter.py`): `RateLimiter(requests_per_minute, max_concurrent)`
  using `asyncio.Semaphore` + sliding-window counter. Async context manager interface.
  Wired into `LLM(rate_limiter=...)`.
- **Structured logging** (`logging.py`): `PipelineLogger` (LoggerAdapter with JSON
  formatter) carrying `run_id` and `pipeline` through all log records.
- **Agent output validation**: `@Agent(output_schema=MyPydanticModel)` validates
  LLM response JSON against a Pydantic v2 model; raises `AgentOutputValidationError`
  on failure.
- New exception classes: `AgentTimeoutError`, `AgentOutputValidationError`.
- `AgentResult` gains: `cached`, `level`, `timestamp` fields.
- `PipelineResult` gains: `run_id`, `levels_executed`, `agents_with_cache_hits` fields.
- `Event.type` now documents all valid values including `"agent_skipped"`.
- GitHub Actions CI workflow (lint, test matrix py3.10-3.12, codecov, build check).
- GitHub Actions publish workflow (OIDC trusted publishing on version tags).
- Issue templates for bug reports and feature requests.
- `CONTRIBUTING.md` with development setup and PR checklist.
- `pythonpath = ["src"]` in pytest config to fix editable install on non-ASCII paths.

### Changed
- `__version__` bumped to `0.2.0`.
- `pipeline.py`: `Pipeline.__init__` gains `retry_failed_agents` parameter.
- `pipeline.py`: `Pipeline.add` gains `timeout` and `condition` parameters.
- `llm.py`: `LLM.__init__` gains `cache` and `rate_limiter` parameters.
  Return dict from `generate()` now includes `"cached"` key.
- `pyproject.toml`: classifier updated to Beta; dev extras expanded;
  `ruff`, `mypy`, `coverage` tool config sections added.

### Performance
- Two independent agents that each take 0.1s now complete in ~0.1s (parallel),
  not ~0.2s (sequential).

---

## [0.1.0] — 2026-02-27

### Added
- Initial release.
- `@Agent` decorator and `BaseAgent` ABC for defining agents.
- `Pipeline` with topological sort (`_resolve_order`) for dependency resolution.
- `LLM` provider abstraction with OpenAI-compatible API, retry logic.
- `EventEmitter` + `pipeline.stream()` for async event streaming.
- Pydantic v2 data models: `AgentResult`, `PipelineResult`, `Event`.
- Custom exception hierarchy: `AgentFlowError`, `AgentError`, `LLMError`, `PipelineError`.
- Published to PyPI as `agentflowkit`.
