"""Tool-calling agent example — an agent that calls Python functions.

The agent is given two tools and runs a ReAct loop: the model decides which
tools to call, agentflow executes them, feeds the results back, and repeats
until the model produces a final answer.

Run: python examples/tool_agent.py
Requires GROQ_API_KEY (free at console.groq.com) or edit the api_key below.
"""

import asyncio
import os

from agentflow import LLM, Agent, Pipeline, tool

llm = LLM(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", ""),
)


# --- Tools: plain Python functions. Schemas are generated from the hints. --- #
@tool
def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression, e.g. '3 * (4 + 5)'."""
    # Restrict eval to arithmetic only — no names, no builtins.
    return eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - demo only


@tool
def get_stock_price(ticker: str) -> dict:
    """Look up the latest price for a stock ticker symbol."""
    # A real tool would hit an API; here we mock a few tickers.
    prices = {"AAPL": 229.87, "MSFT": 430.16, "NVDA": 138.07}
    return {"ticker": ticker.upper(), "price": prices.get(ticker.upper(), 0.0)}


@Agent(
    name="analyst",
    role="Financial Analyst who uses tools to compute exact answers",
    tools=[calculator, get_stock_price],
)
async def analyst(task: str, context: dict) -> str:
    return task


async def main() -> None:
    pipe = Pipeline(llm=llm)
    pipe.add(analyst)

    task = "What is the total cost of buying 10 shares of AAPL and 5 shares of NVDA?"
    print(f"Task: {task}\n{'=' * 60}")

    result = await pipe.run(task)

    print("\nTOOL CALLS MADE:")
    for call in result.get("analyst").metadata.get("tool_calls", []):
        print(f"  - {call['tool']}({call['arguments']}) -> {call['result']}")

    print(f"\nANSWER:\n{result.output}")
    print(f"\nTokens: {result.total_tokens} | Cost: ${result.total_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
