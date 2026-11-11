"""Run-scoped accounting: every billed call of a run counts, including discarded attempts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import agentflow.pipeline as pipeline_module
from agentflow.llm import _estimate_prompt_tokens
from agentflow import LLM, AgentError, BaseAgent, InMemoryContext, Pipeline, register_price
from agentflow.types import AgentResult

PRICE = 1_000_000.0  # $1 per token each way, so every number below is exact.


def _billed_llm(model: str = "scope-test-model") -> LLM:
    """A real LLM whose provider call is faked: generate() runs unchanged and bills."""
    register_price(model, prompt_per_1m=PRICE, completion_per_1m=PRICE)
    llm = LLM(model=model, api_key="fake", max_retries=0)
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "ok"
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = 1
    response.usage.completion_tokens = 1
    response.usage.total_tokens = 2
    response.model = model
    llm._client = MagicMock()
    llm._client.chat.completions.create = AsyncMock(return_value=response)
    return llm


class FlakyAgent(BaseAgent):
    """Makes one billed call per attempt and fails the first ``failures`` attempts."""

    def __init__(self, name: str, failures: int):
        super().__init__(name, "flaky")
        self.failures = failures
        self.attempts = 0

    async def execute(self, task, context, llm):
        self.attempts += 1
        response = await llm.generate([{"role": "user", "content": task}])
        if self.attempts <= self.failures:
            raise AgentError(self.name, "transient failure after a billed call")
        return AgentResult(agent=self.name, output=response.content, tokens_used=response.tokens,
                           cost=response.cost, metadata={"model": response.model})


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    async def instant(_seconds):
        return None
    monkeypatch.setattr(pipeline_module.asyncio, "sleep", instant)


@pytest.mark.asyncio
async def test_a_discarded_retry_attempt_is_billed_to_the_run():
    pipe = Pipeline(llm=_billed_llm(), retry_failed_agents=1)
    pipe.add(FlakyAgent("a", failures=1))

    result = await pipe.run("go")

    assert result.total_cost == pytest.approx(4.0), (
        "two billed calls of $2 each were made; the failed attempt's call must not vanish (COST-1)"
    )
    assert result.total_tokens == 4, "tokens of the discarded attempt must be counted too"
    assert result.results["a"].cost == pytest.approx(2.0), "the agent's own result keeps only its own call"


@pytest.mark.asyncio
async def test_concurrent_runs_on_one_pipeline_keep_separate_ledgers():
    llm = _billed_llm()

    async def run_with(failures: int):
        # Agent state is per run here, so each run gets its own pipeline sharing one LLM.
        own = Pipeline(llm=llm, retry_failed_agents=2)
        own.add(FlakyAgent("a", failures=failures))
        return await own.run("go")

    clean, flaky_run = await asyncio.gather(run_with(0), run_with(2))

    assert clean.total_cost == pytest.approx(2.0), "a clean run must not absorb another run's failed spend"
    assert flaky_run.total_cost == pytest.approx(6.0), "three billed attempts of $2 each"


@pytest.mark.asyncio
async def test_the_budget_counts_the_spend_of_discarded_attempts():
    llm = _billed_llm()
    llm.max_tokens = 1
    reservation = PRICE / 1_000_000 * (_estimate_prompt_tokens([{"role": "user", "content": "go"}], None) + 1)
    # Two discarded attempts bill $2 each. The third reservation fits only if they are forgotten.
    pipe = Pipeline(llm=llm, retry_failed_agents=2, budget_usd=4.0 + reservation - 0.5)
    pipe.add(FlakyAgent("a", failures=2))

    with pytest.raises(pipeline_module.BudgetExceededError) as caught:
        await pipe.run("go")

    assert caught.value.spent_usd == pytest.approx(4.0), (
        "the budget must see the discarded attempts' spend, not only surviving results"
    )
    assert caught.value.partial_result.total_cost == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_run_ids_have_32_hex_characters():
    pipe = Pipeline(llm=_billed_llm())
    pipe.add(FlakyAgent("a", failures=0))

    result = await pipe.run("go")

    assert len(result.run_id) == 32, "8 hex chars collide at about 65k runs; a durable ledger keys on run_id"
    int(result.run_id, 16)


@pytest.mark.asyncio
async def test_streamed_totals_include_discarded_attempts():
    pipe = Pipeline(llm=_billed_llm(), retry_failed_agents=1)
    pipe.add(FlakyAgent("a", failures=1))

    events = [event async for event in pipe.stream("go")]

    complete = [e for e in events if e.type == "pipeline_complete"]
    assert complete, f"stream must finish, got {[e.type for e in events]}"
    assert complete[0].data["total_cost"] == pytest.approx(4.0), "stream and run must report the same total"


@pytest.mark.asyncio
async def test_failed_spend_before_a_pause_survives_resume():
    from agentflow import Agent, ApprovalPolicy, tool

    @tool
    async def actuate() -> str:
        return "done"

    @Agent(name="b", role="r", tools=[actuate])
    async def b(task, context):
        return task

    llm = _billed_llm()
    llm._client.chat.completions.create = AsyncMock(side_effect=_pause_then_finish(llm.model))
    pipe = Pipeline(llm=llm, retry_failed_agents=1, memory=InMemoryContext(),
                    approval_policy=ApprovalPolicy(blocked_tools=["actuate"]))
    pipe.add(FlakyAgent("a", failures=1))
    pipe.add(b, depends_on=["a"])

    paused = await pipe.run("go")
    assert paused.status == "paused", "the second agent must pause on the blocked tool"
    resumed = await pipe.resume(paused.pause_info["session_id"], "ok", approved=True)

    # a: 2 calls ($4, one discarded). b: 1 call before the pause + 1 after ($4).
    assert resumed.total_cost == pytest.approx(8.0), (
        "the discarded attempt before the pause must still be counted after resume"
    )


def _pause_then_finish(model: str):
    """Provider replies: a for both attempts, b's tool call, then b's final answer."""

    def reply(content: str, tool_calls=None):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        response.choices[0].message.tool_calls = tool_calls
        response.choices[0].finish_reason = "tool_calls" if tool_calls else "stop"
        response.usage.prompt_tokens = 1
        response.usage.completion_tokens = 1
        response.usage.total_tokens = 2
        response.model = model
        return response

    call = MagicMock()
    call.id = "c1"
    call.type = "function"
    call.function.name = "actuate"
    call.function.arguments = "{}"
    return [reply("a1"), reply("a2"), reply("", [call]), reply("final")]
