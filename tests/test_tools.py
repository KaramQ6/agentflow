"""Tests for tool/function calling."""

import json

import pytest
from agentflow import Agent, Tool, tool
from agentflow.exceptions import AgentError, ToolError


# --------------------------------------------------------------------------- #
# Schema generation
# --------------------------------------------------------------------------- #
def test_tool_decorator_bare():
    @tool
    async def search(query: str) -> str:
        """Search the web."""
        return query

    assert isinstance(search, Tool)
    assert search.name == "search"
    assert search.description == "Search the web."


def test_tool_decorator_with_overrides():
    @tool(name="lookup", description="custom desc")
    def raw(x: int) -> int:
        return x

    assert raw.name == "lookup"
    assert raw.description == "custom desc"


def test_tool_schema_types_and_required():
    @tool
    def weather(city: str, unit: str = "celsius") -> str:
        """Get weather."""
        return city

    schema = weather.openai_schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "weather"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"]) == {"city", "unit"}
    assert params["required"] == ["city"]  # unit has a default → not required
    assert "title" not in params  # pydantic titles stripped


# --------------------------------------------------------------------------- #
# Tool execution
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_tool_call():
    @tool
    async def add(a: int, b: int) -> int:
        return a + b

    assert await add.acall({"a": 2, "b": 3}) == "5"


@pytest.mark.asyncio
async def test_sync_tool_runs_in_thread():
    @tool
    def multiply(a: int, b: int) -> int:
        return a * b

    assert await multiply.acall({"a": 4, "b": 5}) == "20"


@pytest.mark.asyncio
async def test_tool_accepts_json_string_arguments():
    @tool
    async def echo(msg: str) -> str:
        return msg

    assert await echo.acall('{"msg": "hi"}') == "hi"


@pytest.mark.asyncio
async def test_tool_dict_return_is_json_encoded():
    @tool
    async def profile(name: str) -> dict:
        return {"name": name, "active": True}

    out = await echo_json(profile, {"name": "sam"})
    assert out == {"name": "sam", "active": True}


async def echo_json(t: Tool, args: dict) -> dict:
    return json.loads(await t.acall(args))


@pytest.mark.asyncio
async def test_tool_invalid_arguments_raise_toolerror():
    @tool
    async def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(ToolError):
        await add.acall({"a": "not-an-int", "b": 3})


@pytest.mark.asyncio
async def test_tool_bad_json_raises_toolerror():
    @tool
    async def echo(msg: str) -> str:
        return msg

    with pytest.raises(ToolError):
        await echo.acall("{not valid json")


@pytest.mark.asyncio
async def test_tool_function_exception_wrapped():
    @tool
    async def boom(x: int) -> int:
        raise ValueError("kaboom")

    with pytest.raises(ToolError) as exc:
        await boom.acall({"x": 1})
    assert "kaboom" in str(exc.value)


# --------------------------------------------------------------------------- #
# ReAct tool loop inside an agent
# --------------------------------------------------------------------------- #
def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _response(content: str, tool_calls=None) -> dict:
    return {
        "content": content,
        "tokens": 10,
        "prompt_tokens": 6,
        "completion_tokens": 4,
        "duration": 0.0,
        "model": "fake-model",
        "cached": False,
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else "stop",
    }


class ScriptedLLM:
    """LLM stub that returns queued responses and records requests."""

    model = "fake-model"

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.requests: list[dict] = []

    async def generate(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_agent_runs_tool_then_answers():
    @tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @Agent(name="calc", role="Calculator", tools=[add])
    async def calc(task: str, context: dict) -> str:
        return task

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "add", {"a": 2, "b": 3})]),
            _response("The sum is 5."),
        ]
    )

    result = await calc.execute("add 2 and 3", {}, llm)

    assert result.output == "The sum is 5."
    assert result.tokens_used == 20  # accumulated across both turns
    trace = result.metadata["tool_calls"]
    assert len(trace) == 1
    assert trace[0]["tool"] == "add"
    assert trace[0]["result"] == "5"
    # the second request must include the tool result message
    roles = [m["role"] for m in llm.requests[1]["messages"]]
    assert "tool" in roles


@pytest.mark.asyncio
async def test_agent_recovers_from_unknown_tool():
    @tool
    async def add(a: int, b: int) -> int:
        return a + b

    @Agent(name="calc", role="Calculator", tools=[add])
    async def calc(task: str, context: dict) -> str:
        return task

    llm = ScriptedLLM(
        [
            _response("", [_tool_call("c1", "nope", {"x": 1})]),
            _response("Recovered."),
        ]
    )

    result = await calc.execute("go", {}, llm)
    assert result.output == "Recovered."
    assert "unknown tool" in result.metadata["tool_calls"][0]["result"]


@pytest.mark.asyncio
async def test_agent_exceeds_max_tool_iterations():
    @tool
    async def loop(x: int) -> int:
        return x

    @Agent(name="looper", role="Looper", tools=[loop], max_tool_iterations=2)
    async def looper(task: str, context: dict) -> str:
        return task

    # Always returns a tool call → never terminates
    llm = ScriptedLLM([_response("", [_tool_call("c", "loop", {"x": 1})]) for _ in range(5)])

    with pytest.raises(AgentError) as exc:
        await looper.execute("go", {}, llm)
    assert "max_tool_iterations" in str(exc.value)
