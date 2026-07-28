# Cost & Streaming

## Cost tracking

Every result carries an estimated USD cost, computed from built-in per-model
pricing and the actual prompt/completion token split.

```python
result = await pipe.run("Summarize the news")

agent = result.get("summarizer")
print(f"Agent cost:    ${agent.cost:.6f}")
print(f"Pipeline cost: ${result.total_cost:.6f}")
```

Cache hits bill nothing, so their `cost` is `0.0` while token counts remain for
reference.

Prices use longest-prefix matching, so `gpt-4o-2024-08-06` resolves to `gpt-4o`.

### Models with no price

A model that is not in the pricing table costs `0.0`. For a local Ollama or
vLLM model that is the truth; for a hosted model it is a placeholder, and
`total_cost` is an undercount. agentflow never lets that pass silently — the
model is logged once per process, and the run reports it:

```python
if result.unpriced_models:
    print(f"no price for: {result.unpriced_models}")
```

### Custom or self-hosted models

`register_price()` is the authoritative override. The bundled table is
indicative and drifts as providers change their prices, so anything you bill
against should be registered explicitly:

```python
from agentflow import register_price

register_price("my-finetuned-model", prompt_per_1m=0.50, completion_per_1m=1.50)
```

## Budgets

`budget_usd` is a hard ceiling on a single run, checked after each DAG level.
The error carries the work that already completed, so tripping the ceiling does
not throw away what you paid for:

```python
from agentflow import BudgetExceededError

pipe = Pipeline(llm=llm, budget_usd=0.25)
try:
    result = await pipe.run("Analyze the filing")
except BudgetExceededError as exc:
    print(f"stopped at ${exc.spent_usd:.4f} of ${exc.budget_usd:.2f}")
    result = exc.partial_result

for name, agent_result in result.results.items():
    print(name, agent_result.cost)
```

An in-flight LLM call cannot be aborted, so a run may overshoot the ceiling by
at most one level's spend.

## Token streaming

For interactive UIs, stream the model's output token-by-token with
`LLM.astream()`:

```python
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain async pipelines in one line."},
]
async for token in llm.astream(messages):
    print(token, end="", flush=True)
```

`astream()` honours the rate limiter but does not cache or retry (both are
ambiguous mid-stream); use `generate()` for those.

See the full API in the [reference](../reference.md).
