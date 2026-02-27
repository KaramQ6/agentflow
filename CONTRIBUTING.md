# Contributing to agentflowkit

Thank you for your interest in contributing! This guide explains how to work on the project.

## Development Setup

```bash
git clone https://github.com/KaramQ6/agentflow.git
cd agentflow
pip install -e ".[dev]"
```

Verify your setup:

```bash
pytest tests/ -v
```

All tests should pass before you start making changes.

## Project Structure

```
src/agentflow/
├── agent.py          # @Agent decorator + BaseAgent ABC
├── cache.py          # ResponseCache, InMemoryCache, RedisCache
├── events.py         # EventEmitter for streaming
├── exceptions.py     # Exception hierarchy
├── llm.py            # LLM provider abstraction
├── logging.py        # PipelineLogger (structured JSON)
├── pipeline.py       # Pipeline DAG orchestration (core)
├── rate_limiter.py   # RateLimiter (RPM + concurrency)
└── types.py          # Pydantic data models
```

## Code Style

We use **ruff** for linting + formatting and **mypy** for type checking.

```bash
ruff check src/ tests/          # Lint
ruff format src/ tests/         # Format
mypy src/agentflow/             # Type check
```

All code must:
- Have full type hints
- Pass `ruff check` with no errors
- Pass `mypy` in strict mode

## Adding a Feature

### New cache backend

1. Subclass `ResponseCache` in `cache.py`
2. Implement `async get(key)` and `async set(key, value, ttl)`
3. Add to `__all__` in `__init__.py`
4. Write tests in `tests/test_cache.py`

### New exception type

1. Add to `exceptions.py` (subclass the appropriate parent)
2. Add to `__all__` in `__init__.py`
3. Import and use in the relevant module

### New pipeline feature

1. Add to `pipeline.py` — keep changes localized to `_PipelineNode`, `Pipeline.add()`, or the execution loop
2. Emit appropriate events for streaming consumers
3. Write tests in `tests/test_parallel.py` or a new test file

## Pull Request Checklist

- [ ] Tests added for new functionality
- [ ] All existing tests still pass (`pytest tests/ -v`)
- [ ] Type hints on all new functions and classes
- [ ] Docstrings on public API additions
- [ ] Entry added to `[Unreleased]` section in `CHANGELOG.md`
- [ ] No new dependencies added without discussion in an issue first

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Redis cache backend
fix: handle asyncio.TimeoutError in _execute_node
docs: add caching section to README
test: add parallel execution timing test
refactor: extract _resolve_levels from _resolve_order
```

## Running the Full Suite

```bash
pytest tests/ -v --cov=agentflow --cov-report=term-missing
```

Coverage should remain ≥ 80%.
