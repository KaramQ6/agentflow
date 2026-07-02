"""Benchmark: parallel DAG execution vs. sequential.

Uses a fake LLM with a fixed per-call delay so the numbers reflect agentflow's
scheduling, not network variance. Three independent agents (Level 0) run
concurrently; a fourth (Level 1) waits for them.

Run: python benchmarks/parallel_speedup.py
"""

import asyncio
import time

from agentflow import AgentResult, Pipeline
from agentflow.agent import BaseAgent

CALL_LATENCY = 0.3  # simulated seconds per LLM call


class SlowLLM:
    """Fake LLM that sleeps to simulate network latency."""

    model = "bench-model"

    async def generate(self, messages, **kwargs):
        await asyncio.sleep(CALL_LATENCY)
        return {"content": "ok", "tokens": 100, "cost": 0.0, "duration": CALL_LATENCY, "model": self.model}


class Worker(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name=name, role="worker")

    async def execute(self, task, context, llm):
        r = await llm.generate([{"role": "user", "content": task}])
        return AgentResult(agent=self.name, output=r["content"], tokens_used=r["tokens"])


async def run_parallel() -> float:
    pipe = Pipeline(llm=SlowLLM())
    pipe.add(Worker("a"))
    pipe.add(Worker("b"))
    pipe.add(Worker("c"))
    pipe.add(Worker("d"), depends_on=["a", "b", "c"])  # Level 1
    start = time.perf_counter()
    await pipe.run("task")
    return time.perf_counter() - start


async def run_sequential() -> float:
    # Force a chain so every agent waits for the previous one.
    pipe = Pipeline(llm=SlowLLM())
    pipe.add(Worker("a"))
    pipe.add(Worker("b"), depends_on=["a"])
    pipe.add(Worker("c"), depends_on=["b"])
    pipe.add(Worker("d"), depends_on=["c"])
    start = time.perf_counter()
    await pipe.run("task")
    return time.perf_counter() - start


async def main() -> None:
    parallel = await run_parallel()
    sequential = await run_sequential()
    print(f"Per-call latency : {CALL_LATENCY:.2f}s")
    print(f"Parallel  (3 || + 1) : {parallel:.2f}s  (2 levels)")
    print(f"Sequential (chain of 4): {sequential:.2f}s  (4 levels)")
    print(f"Speedup: {sequential / parallel:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
