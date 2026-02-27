# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
