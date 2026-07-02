# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
