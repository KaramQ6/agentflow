# Observability

`Pipeline.run()` is silent by default. Pass a `Hooks` instance to observe the
full lifecycle and bridge it to logging, metrics, OpenTelemetry, or Langfuse.
All hook methods are no-ops by default, so override only what you need — and a
hook that raises is caught and warned rather than crashing the pipeline.

## Structured logging out of the box

`LoggingHooks` emits structured JSON logs via `PipelineLogger`:

```python
from agentflow import Pipeline, LoggingHooks

pipe = Pipeline(llm=llm, hooks=LoggingHooks("research-pipeline"))
result = await pipe.run("AI in Healthcare")
# {"timestamp": "...", "event": "agent_complete", "agent": "writer", "tokens": 812, ...}
```

## Custom hooks

Subclass `Hooks` to send spans to your own backend:

```python
from agentflow import Hooks

class OTelHooks(Hooks):
    def on_agent_start(self, agent: str, level: int) -> None:
        self._spans[agent] = tracer.start_span(agent)

    def on_agent_end(self, result) -> None:
        span = self._spans.pop(result.agent)
        span.set_attribute("tokens", result.tokens_used)
        span.set_attribute("cost_usd", result.cost)
        span.end()

pipe = Pipeline(llm=llm, hooks=OTelHooks())
```

## Streaming events

For live progress in the same process, use `pipeline.stream()` instead — it yields
`Event` objects (`agent_start`, `agent_complete`, `agent_skipped`,
`pipeline_complete`, …) as they happen:

```python
async for event in pipe.stream("AI in Healthcare"):
    if event.type == "agent_complete":
        print(event.agent, event.data["tokens"], event.data["cached"])
```

See the full API in the [reference](../reference.md).
