# Tools & Function Calling

Tools turn an agent from a one-shot prompt into a real **ReAct agent**: the model
decides which functions to call, agentflow executes them, feeds the results back,
and repeats until the model produces a final answer.

## Define a tool

Decorate any Python function with `@tool`. The JSON schema is generated
automatically from the type hints — you never hand-write it. Sync and async
functions both work (sync tools run in a thread so they never block the loop).

```python
from agentflow import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Look up the current weather for a city."""  # (1)!
    return f"18°{unit[0].upper()} and sunny in {city}"
```

1. The docstring becomes the tool description the model sees. `city` is required
   (no default); `unit` is optional.

## Attach tools to an agent

```python
from agentflow import Agent, Pipeline, LLM

@Agent(name="assistant", role="Helpful Assistant", tools=[get_weather])
async def assistant(task: str, context: dict) -> str:
    return task

pipe = Pipeline(llm=LLM(model="gpt-4o-mini", api_key="..."))
pipe.add(assistant)
result = await pipe.run("What's the weather in Amman in fahrenheit?")
```

The loop is bounded by `max_tool_iterations` (default 6) so a misbehaving model
can never spin forever.

## Inspect what happened

Every tool call is recorded in the agent result's metadata:

```python
for call in result.get("assistant").metadata["tool_calls"]:
    print(call["tool"], call["arguments"], "->", call["result"])
```

## Error handling

If a tool's arguments fail validation or the function raises, the error is fed
back to the model as the tool result (prefixed `Error:`) so it can recover,
rather than crashing the pipeline. A `ToolError` is raised only when you call a
tool directly via `Tool.acall`.

See the full API in the [reference](../reference.md).
