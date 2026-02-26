# agentflow

[![PyPI version](https://badge.fury.io/py/agentflow-py.svg)](https://pypi.org/project/agentflow-py/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Lightweight multi-agent AI pipeline framework. Define agents with decorators, wire them into pipelines, stream events in real-time.

- **Decorator-based** - Define agents as simple async functions
- **Async-first** - Built on asyncio, no sync bottlenecks
- **Event streaming** - Real-time pipeline monitoring via async generators
- **Provider agnostic** - Works with any OpenAI-compatible API (OpenAI, Groq, Together, Ollama, etc.)
- **Minimal deps** - Just `openai` + `pydantic`

## Install

```bash
pip install agentflow-py
```

## Quick Start

```python
import asyncio
from agentflow import Agent, Pipeline, LLM

# 1. Configure LLM (any OpenAI-compatible provider)
llm = LLM(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key="your-api-key",
)

# 2. Define agents with decorators
@Agent(name="researcher", role="Research Analyst")
async def researcher(task: str, context: dict) -> str:
    return f"Research this topic thoroughly: {task}"

@Agent(name="writer", role="Content Writer")
async def writer(task: str, context: dict) -> str:
    research = context["researcher"]
    return f"Write an article based on:\n{research}"

# 3. Build pipeline
pipe = Pipeline(llm=llm)
pipe.add(researcher)
pipe.add(writer, depends_on=["researcher"])

# 4. Run
async def main():
    result = await pipe.run("AI in Healthcare")
    print(result.output)
    print(f"Tokens: {result.total_tokens}")

asyncio.run(main())
```

## Event Streaming

Stream real-time events as agents execute:

```python
async for event in pipe.stream("AI in Healthcare"):
    if event.type == "agent_start":
        print(f"{event.agent} started...")
    elif event.type == "agent_complete":
        print(f"{event.agent} done ({event.data['tokens']} tokens)")
    elif event.type == "pipeline_complete":
        print(f"Total: {event.data['total_tokens']} tokens")
```

## Pipeline Results

Access individual agent results:

```python
result = await pipe.run("AI in Healthcare")

# Final output (last agent)
print(result.output)

# Individual agent results
research = result.get("researcher")
print(research.output)
print(research.tokens_used)
print(research.duration)

# Totals
print(result.total_tokens)
print(result.total_duration)
```

## Advanced: Class-Based Agents

For complex agents that need custom logic:

```python
from agentflow import BaseAgent, AgentResult

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="custom", role="Custom Processor")

    async def execute(self, task, context, llm):
        # Custom logic here
        response = await llm.generate([
            {"role": "system", "content": f"You are a {self.role}."},
            {"role": "user", "content": task},
        ])
        return AgentResult(
            agent=self.name,
            output=response["content"],
            tokens_used=response["tokens"],
            duration=response["duration"],
        )

pipe.add(CustomAgent())
```

## Supported Providers

Any OpenAI-compatible API works:

```python
# OpenAI
llm = LLM(model="gpt-4o-mini", api_key="sk-...")

# Groq (free tier)
llm = LLM(model="llama-3.3-70b-versatile",
           base_url="https://api.groq.com/openai/v1",
           api_key="gsk_...")

# Ollama (local)
llm = LLM(model="llama3", base_url="http://localhost:11434/v1")

# Together AI
llm = LLM(model="meta-llama/Llama-3-70b-chat-hf",
           base_url="https://api.together.xyz/v1",
           api_key="...")
```

## Examples

- [`examples/research_crew.py`](examples/research_crew.py) - Multi-agent research pipeline
- [`examples/code_reviewer.py`](examples/code_reviewer.py) - AI code review pipeline

## License

MIT
