"""Structured output: instruct the model, then repair rather than die.

Before this, `output_schema` only validated. Users had to embed
`Model.model_json_schema()` in their own prompt (the README did exactly that),
and a single malformed reply killed the run.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentflow import Agent, Pipeline
from agentflow.agent import _strip_json_fence
from agentflow.exceptions import AgentOutputValidationError
from agentflow.types import LLMResponse


class Report(BaseModel):
    title: str
    confidence: float


class ScriptedLLM:
    """Returns queued contents in order and records what it was sent."""

    model = "gpt-4o-mini"

    def __init__(self, *contents: str):
        self._contents = list(contents)
        self.calls: list[list[dict]] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        content = self._contents.pop(0) if self._contents else "{}"
        return LLMResponse(
            content=content, tokens=10, cost=0.001, model=self.model, duration=0.0
        )


def _agent(**kwargs):
    @Agent(name="analyst", role="Analyst", output_schema=Report, **kwargs)
    async def analyst(task: str, context: dict) -> str:
        return task

    return analyst


# ── The model is told the shape ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_schema_is_injected_into_the_system_prompt():
    llm = ScriptedLLM('{"title": "T", "confidence": 0.9}')
    await _agent().execute("analyse", {}, llm)

    system_prompt = llm.calls[0][0]["content"]
    assert "JSON Schema" in system_prompt
    assert '"confidence"' in system_prompt
    assert '"title"' in system_prompt


@pytest.mark.asyncio
async def test_no_schema_block_when_no_output_schema():
    @Agent(name="plain", role="Writer")
    async def plain(task: str, context: dict) -> str:
        return task

    llm = ScriptedLLM("just prose")
    await plain.execute("write", {}, llm)

    assert "JSON Schema" not in llm.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_schema_block_survives_a_custom_system_prompt():
    """A custom system_prompt replaces the role line, not the format contract."""
    llm = ScriptedLLM('{"title": "T", "confidence": 0.9}')
    await _agent(system_prompt="You are terse.").execute("analyse", {}, llm)

    system_prompt = llm.calls[0][0]["content"]
    assert system_prompt.startswith("You are terse.")
    assert "JSON Schema" in system_prompt


# ── Repair ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_output_is_repaired_on_the_second_attempt():
    llm = ScriptedLLM(
        '{"title": "T"}',  # missing `confidence`
        '{"title": "T", "confidence": 0.75}',
    )

    result = await _agent().execute("analyse", {}, llm)

    assert result.data == {"title": "T", "confidence": 0.75}
    assert len(llm.calls) == 2, "the model was not asked to correct itself"


@pytest.mark.asyncio
async def test_the_repair_prompt_shows_the_model_its_errors():
    llm = ScriptedLLM('{"title": "T"}', '{"title": "T", "confidence": 0.5}')
    await _agent().execute("analyse", {}, llm)

    repair_messages = llm.calls[1]
    assert repair_messages[-2]["role"] == "assistant"
    assert repair_messages[-2]["content"] == '{"title": "T"}'
    assert "confidence" in repair_messages[-1]["content"]
    assert repair_messages[-1]["role"] == "user"


@pytest.mark.asyncio
async def test_repair_tokens_and_cost_are_billed_to_the_agent():
    llm = ScriptedLLM('{"title": "T"}', '{"title": "T", "confidence": 0.5}')

    result = await _agent().execute("analyse", {}, llm)

    assert result.tokens_used == 20, "the repair call must be counted"
    assert result.cost == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_repair_budget_is_bounded_and_then_it_raises():
    llm = ScriptedLLM('{"title": "T"}', '{"still": "wrong"}', '{"and": "again"}')

    with pytest.raises(AgentOutputValidationError):
        await _agent().execute("analyse", {}, llm)

    assert len(llm.calls) == 2, "output_retries=1 means one repair, not a loop"


@pytest.mark.asyncio
async def test_output_retries_zero_raises_on_the_first_failure():
    llm = ScriptedLLM('{"title": "T"}', '{"title": "T", "confidence": 0.5}')

    with pytest.raises(AgentOutputValidationError):
        await _agent(output_retries=0).execute("analyse", {}, llm)

    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_repair_does_not_mutate_the_original_conversation():
    llm = ScriptedLLM('{"title": "T"}', '{"title": "T", "confidence": 0.5}')
    await _agent().execute("analyse", {}, llm)

    first_call, second_call = llm.calls
    assert len(first_call) == 2, "the first request must not have grown a repair turn"
    assert len(second_call) == 4


@pytest.mark.asyncio
async def test_repaired_output_flows_downstream_as_the_validated_dict():
    llm = ScriptedLLM('{"title": "T"}', '{"title": "Final", "confidence": 0.9}')
    seen: dict = {}

    @Agent(name="consumer", role="Consumer")
    async def consumer(task: str, context: dict) -> str:
        seen.update(context)
        return task

    pipe = Pipeline(llm=llm)
    pipe.add(_agent())
    pipe.add(consumer, depends_on=["analyst"])
    await pipe.run("analyse")

    assert seen["analyst"] == {"title": "Final", "confidence": 0.9}


# ── Fenced JSON is fixed locally, not with a paid round-trip ──────────────────


@pytest.mark.asyncio
async def test_markdown_fenced_json_validates_without_a_repair_call():
    llm = ScriptedLLM('```json\n{"title": "T", "confidence": 0.4}\n```')

    result = await _agent().execute("analyse", {}, llm)

    assert result.data == {"title": "T", "confidence": 0.4}
    assert len(llm.calls) == 1, "a formatting artefact must not cost a round-trip"


@pytest.mark.asyncio
async def test_output_keeps_what_the_model_actually_said():
    raw = '```json\n{"title": "T", "confidence": 0.4}\n```'
    llm = ScriptedLLM(raw)

    result = await _agent().execute("analyse", {}, llm)

    assert result.output == raw, "output is the model's reply; data is the parse"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('```JSON\n{"a": 1}```', '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ("  not json at all  ", "  not json at all  "),
        ("```json\nunterminated", "unterminated"),
    ],
)
def test_strip_json_fence(raw, expected):
    assert _strip_json_fence(raw) == expected
