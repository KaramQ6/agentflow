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

### Custom or self-hosted models

Unknown models (e.g. local Ollama) cost `0.0`. Register a price to change that:

```python
from agentflow import register_price

register_price("my-finetuned-model", prompt_per_1m=0.50, completion_per_1m=1.50)
```

Prices use longest-prefix matching, so `gpt-4o-2024-08-06` resolves to `gpt-4o`.

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
