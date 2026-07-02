# agentflow

**Lightweight multi-agent AI pipeline framework** — define agents with decorators,
wire them into a DAG, and run independent stages **in parallel** with tool calling,
caching, cost tracking, timeouts, and real-time streaming.

```python
from agentflow import Agent, Pipeline, LLM, tool

@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression."""
    return eval(expression, {"__builtins__": {}}, {})

@Agent(name="analyst", role="Analyst", tools=[calculator])
async def analyst(task: str, context: dict) -> str:
    return task

pipe = Pipeline(llm=LLM(model="gpt-4o-mini", api_key="..."))
pipe.add(analyst)
result = await pipe.run("What is 12.5% of 4,800?")
print(result.output, result.total_cost)
```

## Why agentflow?

- **Parallel DAG execution** — independent agents at the same level run concurrently via `asyncio.gather()`.
- **Real tools, not just chains** — the `@tool` decorator turns any Python function into an LLM tool; agents run a ReAct loop.
- **Cost tracking** — per-agent and per-pipeline USD cost from built-in pricing tables.
- **Token streaming** — `LLM.astream()` yields tokens for interactive UIs.
- **Production resilience** — exponential backoff with jitter, `Retry-After` support, per-agent timeouts, pipeline retries.
- **Observability** — lifecycle hooks for logging, metrics, or tracing; structured JSON logs.
- **Minimal deps** — only `openai` + `pydantic`. Fully typed, `py.typed` shipped.

## Install

```bash
pip install agentflowkit
pip install "agentflowkit[redis]"   # optional Redis cache backend
```

Continue to [Getting Started](getting-started.md).
