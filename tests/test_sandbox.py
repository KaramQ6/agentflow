"""Tests for the sandbox module."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from unittest import mock

import pytest

from agentflow import Agent
from agentflow.exceptions import ToolError
from agentflow.sandbox import (
    _DOCKER_AVAILABLE,
    DockerSandbox,
    SandboxError,
    SandboxTimeoutError,
    SubprocessSandbox,
    create_sandbox,
    execute_code,
    sandboxed_tool,
)
from agentflow.tools import Tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpp_works() -> bool:
    """Check if g++ is installed AND can compile a trivial program."""
    import subprocess as _sp

    gpp = shutil.which("g++")
    if not gpp:
        return False
    d = tempfile.mkdtemp(prefix="agentflow_test_")
    try:
        src = os.path.join(d, "test.cpp")
        exe = os.path.join(d, "test.exe")
        with open(src, "w") as f:
            f.write("int main() { return 0; }")
        result = _sp.run(
            [gpp, "-o", exe, src],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


_GPP_WORKS = _gpp_works()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    model = "fake-model"

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.requests: list[dict] = []

    async def generate(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# SubprocessSandbox
# ---------------------------------------------------------------------------


class TestSubprocessSandbox:
    def test_initialization_warns(self):
        sb = SubprocessSandbox(default_timeout=30)
        assert sb._default_timeout == 30

    @pytest.mark.asyncio
    async def test_python_hello_world(self):
        sb = SubprocessSandbox()
        result = await sb.execute_code("python", "print('hello, sandbox')")
        assert result["exit_code"] == 0
        assert "hello, sandbox" in result["stdout"]
        assert result["stderr"] == ""

    @pytest.mark.asyncio
    async def test_python_error(self):
        sb = SubprocessSandbox()
        result = await sb.execute_code("python", "1 / 0")
        assert result["exit_code"] != 0
        assert "ZeroDivisionError" in result["stderr"]

    @pytest.mark.asyncio
    async def test_python_timeout(self):
        sb = SubprocessSandbox()
        code = "import time; time.sleep(30)"
        with pytest.raises(SandboxTimeoutError, match="timed out"):
            await sb.execute_code("python", code, timeout=2)

    @pytest.mark.asyncio
    async def test_cpp_hello_world(self):
        if not _GPP_WORKS:
            pytest.skip("working g++ not available")
        sb = SubprocessSandbox()
        code = '#include <iostream>\nint main() { std::cout << "hello cpp"; return 0; }'
        result = await sb.execute_code("cpp", code)
        assert result["exit_code"] == 0
        assert "hello cpp" in result["stdout"]

    @pytest.mark.asyncio
    async def test_cpp_compilation_error(self):
        if not _GPP_WORKS:
            pytest.skip("working g++ not available")
        sb = SubprocessSandbox()
        result = await sb.execute_code("cpp", "not valid c++ code")
        assert result["exit_code"] != 0

    @pytest.mark.asyncio
    async def test_cpp_timeout(self):
        if not _GPP_WORKS:
            pytest.skip("working g++ not available")
        sb = SubprocessSandbox()
        code = "#include <thread>\n#include <chrono>\nint main() { std::this_thread::sleep_for(std::chrono::seconds(30)); return 0; }"
        with pytest.raises(SandboxTimeoutError, match="timed out"):
            await sb.execute_code("cpp", code, timeout=2)

    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        sb = SubprocessSandbox()
        with pytest.raises(SandboxError, match="Unsupported language"):
            await sb.execute_code("ruby", "puts 'hi'")

    @pytest.mark.asyncio
    async def test_python_returns_stdout(self):
        sb = SubprocessSandbox()
        result = await sb.execute_code("python", "print(42)")
        assert result["stdout"].strip() == "42"

    @pytest.mark.asyncio
    async def test_python_multiline_code(self):
        sb = SubprocessSandbox()
        code = "x = 7\ny = 8\nprint(x + y)"
        result = await sb.execute_code("python", code)
        assert result["stdout"].strip() == "15"


# ---------------------------------------------------------------------------
# DockerSandbox (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker SDK not installed")
class TestDockerSandbox:
    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        sandbox = DockerSandbox(auto_pull=False)
        async with sandbox:
            with pytest.raises(SandboxError, match="Unsupported language"):
                await sandbox.execute_code("ruby", "puts 'hi'")

    @pytest.mark.asyncio
    async def test_execute_python_mocked(self):
        """Verify execute_code calls Docker SDK with correct parameters."""
        mock_client = mock.MagicMock()
        mock_container = mock.MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [
            b"output here\n",
            b"",
        ]
        mock_client.containers.run.return_value = mock_container

        sandbox = DockerSandbox(auto_pull=False)
        sandbox._client = mock_client

        result = await sandbox.execute_code("python", "print('hi')", timeout=5)

        assert result == {"stdout": "output here\n", "stderr": "", "exit_code": 0}
        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args.kwargs
        assert call_kwargs["mem_limit"] == "128m"
        assert call_kwargs["cpu_quota"] == 50000
        assert call_kwargs["network_mode"] == "none"
        assert call_kwargs["read_only"] is True
        assert call_kwargs["cap_drop"] == ["ALL"]

    @pytest.mark.asyncio
    async def test_execute_python_exit_code_nonzero(self):
        mock_client = mock.MagicMock()
        mock_container = mock.MagicMock()
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.side_effect = [b"", b"SyntaxError: bad input\n"]
        mock_client.containers.run.return_value = mock_container

        sandbox = DockerSandbox(auto_pull=False)
        sandbox._client = mock_client

        result = await sandbox.execute_code("python", "bad code")
        assert result["exit_code"] == 1
        assert "SyntaxError" in result["stderr"]

    @pytest.mark.asyncio
    async def test_timeout_kills_container(self):
        mock_client = mock.MagicMock()
        mock_container = mock.MagicMock()

        async def slow_wait():
            await asyncio.sleep(60)
            return {"StatusCode": 0}

        mock_container.wait.side_effect = slow_wait
        mock_client.containers.run.return_value = mock_container

        sandbox = DockerSandbox(auto_pull=False)
        sandbox._client = mock_client

        with pytest.raises(SandboxTimeoutError, match="timed out"):
            await sandbox.execute_code("python", "while True: pass", timeout=1)

        mock_container.kill.assert_called_once()
        mock_container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_cleanup_runs_even_on_error(self):
        mock_client = mock.MagicMock()
        mock_container = mock.MagicMock()
        mock_container.wait.side_effect = RuntimeError("container crash")
        mock_client.containers.run.return_value = mock_container

        sandbox = DockerSandbox(auto_pull=False)
        sandbox._client = mock_client

        with pytest.raises(RuntimeError, match="container crash"):
            await sandbox.execute_code("python", "print('hi')")

        mock_container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_cpp_execution_mocked(self):
        mock_client = mock.MagicMock()
        mock_container = mock.MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [b"hello cpp\n", b""]
        mock_client.containers.run.return_value = mock_container

        sandbox = DockerSandbox(auto_pull=False)
        sandbox._client = mock_client

        result = await sandbox.execute_code(
            "cpp",
            '#include <iostream>\nint main() { std::cout << "hello cpp"; return 0; }',
            timeout=5,
        )
        assert result["exit_code"] == 0
        assert "hello cpp" in result["stdout"]


# ---------------------------------------------------------------------------
# sandboxed_tool decorator
# ---------------------------------------------------------------------------


class TestSandboxedTool:
    def test_returns_tool_instance(self):
        @sandboxed_tool
        async def run_python(code: str) -> str:
            """Execute Python in a sandbox."""

        assert isinstance(run_python, Tool)
        assert run_python.name == "run_python"
        assert "Execute Python in a sandbox" in run_python.description

    def test_dual_mode_with_args(self):
        tool_instance = sandboxed_tool(
            sandbox=None,
            language="cpp",
            timeout=5,
            memory="256m",
            cpu_quota=10000,
        )
        assert callable(tool_instance)

        @tool_instance
        async def compile_cpp(code: str) -> str:
            """Compile C++ code."""

        assert isinstance(compile_cpp, Tool)
        assert compile_cpp.name == "compile_cpp"

    def test_openai_schema_includes_code_parameter(self):
        @sandboxed_tool
        async def run_code(code: str) -> str:
            """Run code."""

        schema = run_code.openai_schema
        params = schema["function"]["parameters"]
        assert "code" in params["properties"]
        assert params["required"] == ["code"]

    @pytest.mark.asyncio
    async def test_python_execution(self):
        @sandboxed_tool
        async def run_python(code: str) -> str:
            """Run Python."""

        result = await run_python.acall('{"code": "print(7 + 8)"}')
        assert result == "15"

    @pytest.mark.asyncio
    async def test_python_execution_error_propagates(self):
        @sandboxed_tool
        async def run_python(code: str) -> str:
            """Run Python."""

        with pytest.raises(ToolError, match="ZeroDivisionError"):
            await run_python.acall('{"code": "1 / 0"}')

    @pytest.mark.asyncio
    async def test_cpp_execution(self):
        if not _GPP_WORKS:
            pytest.skip("working g++ not available")

        @sandboxed_tool(language="cpp")
        async def run_cpp(code: str) -> str:
            """Run C++."""

        code = '#include <iostream>\nint main() { std::cout << "42"; return 0; }'
        result = await run_cpp.acall(json.dumps({"code": code}))
        assert "42" in result

    @pytest.mark.asyncio
    async def test_json_string_arguments(self):
        @sandboxed_tool
        async def run_python(code: str) -> str:
            """Run Python."""

        result = await run_python.acall('{"code": "print(1)"}')
        assert result == "1"

    @pytest.mark.asyncio
    async def test_code_param_extracted_by_position(self):
        @sandboxed_tool
        async def execute(source_code: str) -> str:
            """Execute source."""

        result = await execute.acall('{"source_code": "print(99)"}')
        assert result == "99"

    @pytest.mark.asyncio
    async def test_integration_with_agent(self):
        """Simulate an LLM calling the sandboxed tool through an Agent."""

        @sandboxed_tool
        async def run_python(code: str) -> str:
            """Execute Python code in a secure sandbox."""

        @Agent(name="coder", role="Engineer", tools=[run_python])
        async def coder(task: str, context: dict) -> str:
            return task

        llm = ScriptedLLM(
            [
                _response("", [_tool_call("c1", "run_python", {"code": "print(3 * 4)"})]),
                _response("Output was 12."),
            ]
        )

        result = await coder.execute("what is 3*4?", {}, llm)
        assert result.output == "Output was 12."
        trace = result.metadata["tool_calls"]
        assert trace[0]["tool"] == "run_python"
        assert trace[0]["result"] == "12"


# ---------------------------------------------------------------------------
# create_sandbox factory
# ---------------------------------------------------------------------------


class TestCreateSandbox:
    def test_returns_subprocess_when_docker_missing(self):
        sb = create_sandbox(prefer_docker=True, allow_insecure_fallback=True)
        # Docker may or may not be installed; with allow_insecure_fallback
        # the factory always returns something.
        assert isinstance(sb, DockerSandbox | SubprocessSandbox)

    def test_forwards_kwargs_to_subprocess(self):
        sb = create_sandbox(prefer_docker=False, default_timeout=42)
        assert isinstance(sb, SubprocessSandbox)
        assert sb._default_timeout == 42


# ---------------------------------------------------------------------------
# execute_code convenience function
# ---------------------------------------------------------------------------


class TestExecuteCode:
    @pytest.mark.asyncio
    async def test_simple_python(self):
        result = await execute_code("python", "print('hello')")
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_python_with_error(self):
        result = await execute_code("python", "raise ValueError('bad')")
        assert result["exit_code"] != 0

    @pytest.mark.asyncio
    async def test_timeout(self):
        code = "import time; time.sleep(30)"
        with pytest.raises(SandboxTimeoutError):
            await execute_code("python", code, timeout=1)

    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        with pytest.raises(SandboxError, match="Unsupported language"):
            await execute_code("javascript", "console.log(1)")

    @pytest.mark.asyncio
    async def test_cpp(self):
        if not _GPP_WORKS:
            pytest.skip("working g++ not available")
        code = '#include <iostream>\nint main() { std::cout << "ok"; return 0; }'
        result = await execute_code("cpp", code)
        assert result["exit_code"] == 0
        assert "ok" in result["stdout"]

    @pytest.mark.asyncio
    async def test_reuses_existing_sandbox(self):
        sb = SubprocessSandbox()
        result = await execute_code("python", "print(100)", sandbox=sb)
        assert result["exit_code"] == 0
        assert "100" in result["stdout"]
