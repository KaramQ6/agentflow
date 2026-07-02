"""Token streaming + cost tracking example.

Shows two things:
  1. LLM.astream() — token-by-token output for interactive UIs.
  2. Per-agent and per-pipeline USD cost, computed from the model's pricing.

Run: python examples/streaming_and_cost.py
Requires OPENAI_API_KEY (cost tables cover OpenAI/Anthropic models).
"""

import asyncio
import os

from agentflow import LLM, Agent, Pipeline

llm = LLM(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY", ""))


async def stream_demo() -> None:
    print("Streaming tokens live:\n")
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Name three benefits of async pipelines in one line each."},
    ]
    async for token in llm.astream(messages):
        print(token, end="", flush=True)
    print("\n" + "=" * 60)


@Agent(name="summarizer", role="Summarizer")
async def summarizer(task: str, context: dict) -> str:
    return f"Summarize in 2 sentences: {task}"


async def cost_demo() -> None:
    pipe = Pipeline(llm=llm)
    pipe.add(summarizer)
    result = await pipe.run("Multi-agent systems coordinate specialized agents to solve tasks.")

    agent = result.get("summarizer")
    print(f"Output: {agent.output}\n")
    print(f"Agent tokens: {agent.tokens_used} | Agent cost: ${agent.cost:.6f}")
    print(f"Pipeline total cost: ${result.total_cost:.6f}")


async def main() -> None:
    await stream_demo()
    await cost_demo()


if __name__ == "__main__":
    asyncio.run(main())
