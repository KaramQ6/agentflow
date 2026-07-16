"""Tool-calling agent example — an agent that calls Python functions.

The agent is given two tools and runs a ReAct loop: the model decides which
tools to call, agentflow executes them, feeds the results back, and repeats
until the model produces a final answer.

Run: python examples/tool_agent.py
Requires GROQ_API_KEY (free at console.groq.com) or edit the api_key below.
"""

import ast
import asyncio
import operator
import os

from agentflow import LLM, Agent, Pipeline, tool

llm = LLM(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", ""),
)

# Arithmetic-only expression evaluator: parse to an AST and walk it, allowing
# nothing but numbers and these operators. Unlike eval(), there is no code path
# to names, attributes, or calls.
_OPERATORS: dict[type[ast.AST], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_arithmetic(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        op = _OPERATORS[type(node.op)]
        return op(_eval_arithmetic(node.left), _eval_arithmetic(node.right))  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        op = _OPERATORS[type(node.op)]
        return op(_eval_arithmetic(node.operand))  # type: ignore[operator]
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


# --- Tools: plain Python functions. Schemas are generated from the hints. --- #
@tool
def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression, e.g. '3 * (4 + 5)'."""
    return _eval_arithmetic(ast.parse(expression, mode="eval").body)


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
