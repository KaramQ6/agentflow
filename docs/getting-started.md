# Getting Started

## Install

```bash
pip install agentflowkit
```

## Configure a provider

`LLM` speaks any OpenAI-compatible API. Point `base_url` at your provider:

=== "OpenAI"

    ```python
    from agentflow import LLM
    llm = LLM(model="gpt-4o-mini", api_key="sk-...")
    ```

=== "Groq (free)"

    ```python
    llm = LLM(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_...",
    )
    ```

=== "Anthropic (via OpenRouter)"

    ```python
    llm = LLM(
        model="anthropic/claude-3.5-sonnet",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-...",
    )
    ```

=== "Ollama (local)"

    ```python
    llm = LLM(model="llama3.2", base_url="http://localhost:11434/v1", api_key="ollama")
    ```

## Your first pipeline

Independent agents at the same DAG level run **in parallel**. Dependent agents wait
for their prerequisites.

```python
import asyncio
from agentflow import Agent, Pipeline, LLM

llm = LLM(model="gpt-4o-mini", api_key="sk-...")

@Agent(name="researcher", role="Research Analyst")
async def researcher(task: str, context: dict) -> str:
    return f"Research this topic thoroughly: {task}"

@Agent(name="fact_checker", role="Fact Checker")
async def fact_checker(task: str, context: dict) -> str:
    return f"Find key facts and statistics about: {task}"

@Agent(name="writer", role="Content Writer")
async def writer(task: str, context: dict) -> str:
    return f"Write an article using:\n{context['researcher']}\n{context['fact_checker']}"

pipe = Pipeline(llm=llm)
pipe.add(researcher)                                       # Level 0
pipe.add(fact_checker)                                     # Level 0 — parallel
pipe.add(writer, depends_on=["researcher", "fact_checker"])  # Level 1

async def main():
    result = await pipe.run("AI in Healthcare")
    print(result.output)
    print(f"Tokens: {result.total_tokens} | Cost: ${result.total_cost:.6f}")

asyncio.run(main())
```

Next: give agents [tools](guides/tools.md), track [cost and streaming](guides/cost-streaming.md),
or add [observability](guides/observability.md).
