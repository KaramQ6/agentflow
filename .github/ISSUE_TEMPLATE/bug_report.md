---
name: Bug Report
about: Report a reproducible bug in agentflowkit
title: "[Bug] "
labels: bug
assignees: ""
---

## Description

A clear, concise description of the bug.

## Reproduction Steps

```python
# Minimal reproducible example
from agentflow import Agent, LLM, Pipeline

@Agent(name="example", role="Example")
async def example(task: str, context: dict) -> str:
    return task

# Code that triggers the bug
```

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened (include full traceback if applicable).

## Environment

- `agentflowkit` version: <!-- e.g. 0.2.0 -->
- Python version: <!-- e.g. 3.11.5 -->
- OS: <!-- e.g. Ubuntu 22.04 / macOS 14 / Windows 11 -->
- LLM provider: <!-- e.g. OpenAI, Groq, Ollama -->

## Additional Context

Any other context, screenshots, or logs that may help diagnose the issue.
