"""The budget is a hard ceiling: every LLM call reserves its worst case before it is sent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import agentflow.pipeline as pipeline_module
from agentflow import (
    LLM,
    Agent,
    BaseAgent,
    BudgetExceededError,
    Pipeline,
    PipelineError,
    register_price,
    tool,
)
from agentflow.types import AgentResult

MODEL = "budget-test-model"
# $0.001 per token each way. A reply bills 2 tokens = $0.002. A reservation is the
# prompt estimate plus max_tokens (100) worth of completion, so roughly $0.11 to $0.13.
register_price(MODEL, prompt_per_1m=1_000.0, completion_per_1m=1_000.0)


def _reply(model: str, tool_call: bool = False, completion_tokens: int = 1) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "" if tool_call else "ok"
    if tool_call:
        call = MagicMock()
        call.id = "c"
        call.type = "function"
        call.function.name = "read"
        call.function.arguments = '{"n": 1}'
        response.choices[0].message.tool_calls = [call]
    else:
        response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "tool_calls" if tool_call else "stop"
    response.usage.prompt_tokens = 1
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = 1 + completion_tokens
    response.model = model  # the billed model is the one the provider reports
    return response


def _llm(model: str = MODEL, *, delay: float = 0.0, tool_calls_forever: bool = False,
         completion_tokens: int = 1) -> LLM:
    llm = LLM(model=model, api_key="fake", max_retries=0, max_tokens=100)

    async def create(**_kwargs):
        if delay:
            await asyncio.sleep(delay)
        return _reply(model, tool_call=tool_calls_forever, completion_tokens=completion_tokens)

    llm._client = MagicMock()
    llm._client.chat.completions.create = AsyncMock(side_effect=create)
    return llm


class OneCall(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "one call")

    async def execute(self, task, context, llm):
        response = await llm.generate([{"role": "user", "content": task}])
        return AgentResult(agent=self.name, output=response.content, tokens_used=response.tokens,
                           cost=response.cost, metadata={"model": response.model})


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def instant(seconds):
        await real_sleep(0)

    monkeypatch.setattr(pipeline_module.asyncio, "sleep", instant)


@pytest.mark.asyncio
async def test_the_tool_loop_stops_before_the_budget_is_crossed():
    @tool
    async def read(n: int) -> str:
        return "data"

    @Agent(name="reader", role="r", tools=[read], max_tool_iterations=50)
    async def reader(task, context):
        return task

    llm = _llm(tool_calls_forever=True)
    pipe = Pipeline(llm=llm, budget_usd=0.5)
    pipe.add(reader)

    with pytest.raises(BudgetExceededError) as caught:
        await pipe.run("go")

    calls = llm._client.chat.completions.create.await_count
    assert caught.value.spent_usd <= 0.5, "a hard budget must never be crossed (COST-3)"
    assert calls < 50, f"the loop must stop on the budget, not on the iteration cap; made {calls} calls"
    assert caught.value.partial_result is not None, "the partial result must still be handed back"


@pytest.mark.asyncio
async def test_parallel_siblings_cannot_overshoot_the_budget_by_a_level():
    # Each reply uses its whole max_tokens, so each call really costs about $0.10.
    llm = _llm(delay=0.01, completion_tokens=100)
    pipe = Pipeline(llm=llm, budget_usd=0.3)
    for i in range(10):
        pipe.add(OneCall(f"a{i}"))

    with pytest.raises(BudgetExceededError) as caught:
        await pipe.run("go")

    calls = llm._client.chat.completions.create.await_count
    assert calls < 10, f"reservations must hold back siblings in the same level (COST-2); {calls} calls made"
    assert caught.value.spent_usd <= 0.3, "spend must stay under the ceiling"


@pytest.mark.asyncio
async def test_a_run_that_fits_its_budget_completes_unchanged():
    pipe = Pipeline(llm=_llm(), budget_usd=10.0)
    pipe.add(OneCall("a"))

    result = await pipe.run("go")

    assert result.status == "completed", "a budget with room must not change the outcome"
    assert result.total_cost == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_an_unpriced_model_with_a_budget_raises_before_any_call():
    llm = _llm(model="no-price-for-this-model")
    pipe = Pipeline(llm=llm, budget_usd=1.0)
    pipe.add(OneCall("a"))

    with pytest.raises(PipelineError, match="register_price") as caught:
        await pipe.run("go")

    assert "no-price-for-this-model" in str(caught.value), "the error must name the model"
    assert llm._client.chat.completions.create.await_count == 0, "nothing may be sent on an unmeasurable budget"


@pytest.mark.asyncio
async def test_an_unpriced_model_without_a_budget_still_runs_at_zero_cost():
    pipe = Pipeline(llm=_llm(model="no-price-either"))
    pipe.add(OneCall("a"))

    result = await pipe.run("go")

    assert result.status == "completed", "ADR 0003 still holds when no budget is set"
    assert result.unpriced_models == ["no-price-either"]


@pytest.mark.asyncio
async def test_a_model_registered_at_zero_price_runs_under_any_budget():
    register_price("local-free-model", 0, 0)
    pipe = Pipeline(llm=_llm(model="local-free-model"), budget_usd=0.000001)
    pipe.add(OneCall("a"))

    result = await pipe.run("go")

    assert result.status == "completed", "register_price(model, 0, 0) is the documented path for local models"


@pytest.mark.asyncio
async def test_a_budget_stop_is_not_retried():
    agent = OneCall("a")
    llm = _llm()
    pipe = Pipeline(llm=llm, budget_usd=0.05, retry_failed_agents=3)
    pipe.add(agent)

    with pytest.raises(BudgetExceededError):
        await pipe.run("go")

    assert llm._client.chat.completions.create.await_count == 0, (
        "the first reservation (about $0.11) does not fit $0.05, and retrying cannot change that"
    )
