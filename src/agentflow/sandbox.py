"""Secure code execution sandbox for agentflow agents.

Provides container-based and subprocess-based sandboxes for running
untrusted code safely. Includes a :func:`sandboxed_tool` decorator that
wraps tool functions so their code argument always executes inside an
isolated environment with resource limits.

Usage::

    from agentflow import sandboxed_tool, Agent

    @sandboxed_tool(language="python", timeout=5)
    async def run_python(code: str) -> str:
        \"\"\"Execute Python code in a secure sandbox.\"\"\"

    @sandboxed_tool(language="cpp", timeout=10)
    async def run_cpp(code: str) -> str:
        \"\"\"Compile and run C++ code in a secure sandbox.\"\"\"

    @Agent(name="coder", role="Software Engineer", tools=[run_python, run_cpp])
    async def coder(task: str, context: dict) -> str:
        return task
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from .exceptions import ToolError
from .tools import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Docker support
# ---------------------------------------------------------------------------
_DOCKER_AVAILABLE = False

try:
    import docker  # type: ignore[import-untyped,unused-ignore]

    _DOCKER_AVAILABLE = True
except ImportError:
    pass


class SandboxError(Exception):
    """Raised when sandboxed code execution fails."""


class SandboxTimeoutError(SandboxError):
    """Raised when sandboxed code exceeds its timeout."""


def _create_runner_script(language: str, code: str) -> list[str]:
    """Build the shell command for a given language and code payload."""
    if language == "python":
        return ["python", "-c", code]
    if language in ("cpp", "c++"):
        return [
            "sh",
            "-c",
            (
                "cat > /tmp/code.cpp << 'AGENTFLOW_EOF'\n"
                f"{code}\n"
                "AGENTFLOW_EOF\n"
                "g++ -std=c++17 -O2 -o /tmp/prog /tmp/code.cpp 2>&1 && /tmp/prog"
            ),
        ]
    raise SandboxError(f"Unsupported language: {language}")


# ===================================================================
# DockerSandbox
# ===================================================================


class DockerSandbox:
    """Isolated code execution via Docker containers.

    Spins up lightweight alpine containers with memory limits, CPU
    quotas, and strict timeout enforcement to safely run untrusted
    code.  Network access is disabled by default and all Linux
    capabilities are dropped.

    Typical usage (async context manager)::

        sandbox = DockerSandbox()
        async with sandbox:
            result = await sandbox.execute_code("python", "print(1 + 1)")
            print(result["stdout"])  # "2\\n"

    Parameters:
        python_image: Docker image tag for Python execution.
        cpp_image: Docker image tag for C++ compilation/execution.
        memory: Memory limit per container (Docker format, e.g. ``"128m"``).
        cpu_quota: CPU quota in microseconds per 100ms (50000 = 0.5 CPU).
        default_timeout: Default timeout in seconds for each execution.
        network_disabled: If True (default), containers have no network.
        auto_pull: If True, pre-pull images in ``__aenter__``.
    """

    DEFAULT_PYTHON_IMAGE = "python:3.12-alpine"
    DEFAULT_CPP_IMAGE = "gcc:14-alpine"
    DEFAULT_MEMORY = "128m"
    DEFAULT_CPU_QUOTA = 50000  # 0.5 CPU
    DEFAULT_TIMEOUT = 10

    def __init__(
        self,
        *,
        python_image: str | None = None,
        cpp_image: str | None = None,
        memory: str = DEFAULT_MEMORY,
        cpu_quota: int = DEFAULT_CPU_QUOTA,
        default_timeout: int = DEFAULT_TIMEOUT,
        network_disabled: bool = True,
        auto_pull: bool = True,
    ) -> None:
        if not _DOCKER_AVAILABLE:
            raise SandboxError(
                "Docker SDK is required for DockerSandbox. "
                "Install with: pip install agentflowkit[docker]"
            )
        self._python_image = python_image or self.DEFAULT_PYTHON_IMAGE
        self._cpp_image = cpp_image or self.DEFAULT_CPP_IMAGE
        self._memory = memory
        self._cpu_quota = cpu_quota
        self._default_timeout = default_timeout
        self._network_disabled = network_disabled
        self._auto_pull = auto_pull
        self._client: Any = None

    # -- context manager -------------------------------------------------

    async def __aenter__(self) -> DockerSandbox:
        self._client = await asyncio.to_thread(docker.from_env)
        if self._auto_pull:
            await self._pull_images()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None

    async def _pull_images(self) -> None:
        """Pull required images in background threads (best-effort)."""
        for image in {self._python_image, self._cpp_image}:
            try:
                await asyncio.to_thread(self._client.images.pull, image)
            except Exception:
                logger.debug("Pre-pull of %s skipped", image)

    # -- public API ------------------------------------------------------

    async def execute_code(
        self,
        language: str,
        code: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute *code* inside a sandboxed Docker container.

        Args:
            language: ``"python"`` or ``"cpp"`` (``"c++"`` is also accepted).
            code: Source code to execute.
            timeout: Max seconds before hard-kill (default:
                ``self._default_timeout``).

        Returns:
            ``{"stdout": str, "stderr": str, "exit_code": int}``
        """
        language = language.lower()
        if language not in ("python", "cpp", "c++"):
            raise SandboxError(f"Unsupported language: {language}")

        timeout = timeout or self._default_timeout
        command = _create_runner_script(language, code)
        image = self._python_image if language == "python" else self._cpp_image

        container = None
        try:
            container = await asyncio.to_thread(
                self._client.containers.run,
                image,
                command,
                detach=True,
                mem_limit=self._memory,
                cpu_quota=self._cpu_quota,
                network_mode="none" if self._network_disabled else None,
                read_only=True,
                tmpfs={"/tmp": ""},
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
            )
            return await self._wait_for_container(container, timeout)
        finally:
            if container is not None:
                await self._cleanup_container(container)

    # -- internals -------------------------------------------------------

    async def _wait_for_container(self, container: Any, timeout: int) -> dict[str, Any]:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(container.wait),
                timeout=timeout,
            )
            exit_code: int = result.get("StatusCode", -1) or 0
            stdout_bytes = await asyncio.to_thread(container.logs, stdout=True, stderr=False)
            stderr_bytes = await asyncio.to_thread(container.logs, stdout=False, stderr=True)
            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "exit_code": exit_code,
            }
        except asyncio.TimeoutError:
            await self._kill_container(container)
            raise SandboxTimeoutError(
                f"Code execution timed out after {timeout}s"
            ) from None

    async def _kill_container(self, container: Any) -> None:
        with suppress(Exception):
            await asyncio.to_thread(container.kill)

    async def _cleanup_container(self, container: Any) -> None:
        with suppress(Exception):
            await asyncio.to_thread(container.remove, force=True)


# ===================================================================
# SubprocessSandbox  (Docker-free fallback)
# ===================================================================


class SubprocessSandbox:
    """Minimal sandbox using subprocess with resource limits.

    Falls back to local ``subprocess`` when Docker is unavailable.
    **Significantly less secure** — only use with trusted code.
    Every invocation emits a prominent security warning via the
    configured logger.

    Parameters:
        default_timeout: Default timeout in seconds.
    """

    DEFAULT_TIMEOUT = 10

    def __init__(self, default_timeout: int = DEFAULT_TIMEOUT) -> None:
        self._default_timeout = default_timeout

    async def execute_code(
        self,
        language: str,
        code: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute *code* via a local subprocess with timeout.

        Args:
            language: ``"python"`` or ``"cpp"``.
            code: Source code to execute.
            timeout: Max seconds.

        Returns:
            ``{"stdout": str, "stderr": str, "exit_code": int}``
        """
        language = language.lower()
        if language not in ("python", "cpp", "c++"):
            raise SandboxError(f"Unsupported language: {language}")

        timeout = timeout or self._default_timeout
        logger.warning(
            "SECURITY: executing %s code without container isolation. "
            "Use DockerSandbox for untrusted code.",
            language,
        )

        if language == "python":
            return await self._execute_python(code, timeout)
        return await self._execute_cpp(code, timeout)

    # -- Python ----------------------------------------------------------

    async def _execute_python(self, code: str, timeout: int) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode or 0,
            }
        except asyncio.TimeoutError:
            with suppress(Exception):
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
            raise SandboxTimeoutError(
                f"Code execution timed out after {timeout}s"
            ) from None

    # -- C++ -------------------------------------------------------------

    async def _execute_cpp(self, code: str, timeout: int) -> dict[str, Any]:
        gpp = shutil.which("g++")
        if not gpp:
            raise SandboxError(
                "g++ compiler not found. Install build-essential (Linux) "
                "or Xcode command-line tools (macOS)."
            )

        tmpdir = await asyncio.to_thread(
            tempfile.mkdtemp, prefix="agentflow_sandbox_"
        )
        proc: asyncio.subprocess.Process | None = None
        try:
            src = os.path.join(tmpdir, "code.cpp")
            exe = os.path.join(tmpdir, "prog.exe" if sys.platform == "win32" else "prog")

            with open(src, "w", encoding="utf-8") as f:
                f.write(code)

            # compile
            compile_proc = await asyncio.create_subprocess_exec(
                gpp, "-std=c++17", "-O2", "-o", exe, src,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, compile_stderr = await asyncio.wait_for(
                compile_proc.communicate(), timeout=30
            )
            if compile_proc.returncode != 0:
                return {
                    "stdout": "",
                    "stderr": compile_stderr.decode("utf-8", errors="replace"),
                    "exit_code": compile_proc.returncode or 1,
                }

            # run
            proc = await asyncio.create_subprocess_exec(
                exe,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, run_stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": (
                    compile_stderr.decode("utf-8", errors="replace")
                    + run_stderr.decode("utf-8", errors="replace")
                ),
                "exit_code": proc.returncode or 0,
            }
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                with suppress(Exception):
                    proc.kill()
                    await proc.wait()
            raise SandboxTimeoutError(
                f"Code execution timed out after {timeout}s"
            ) from None
        finally:
            await asyncio.to_thread(shutil.rmtree, tmpdir, ignore_errors=True)


# ===================================================================
# Factory
# ===================================================================


def create_sandbox(
    *,
    prefer_docker: bool = True,
    allow_insecure_fallback: bool = False,
    **kwargs: Any,
) -> DockerSandbox | SubprocessSandbox:
    """Create the best-available sandbox for the current environment.

    Args:
        prefer_docker: Try Docker first; fall back to subprocess on failure.
        allow_insecure_fallback: If True, silently fall back to
            :class:`SubprocessSandbox` when Docker is unavailable.
            If False (default), raises :class:`RuntimeError` instead.
        **kwargs: Forwarded to the sandbox constructor. Keys not accepted
            by SubprocessSandbox (``memory``, ``cpu_quota``, etc.) are
            silently dropped when falling back.

    Returns:
        A :class:`DockerSandbox` or :class:`SubprocessSandbox` instance.

    Raises:
        RuntimeError: If Docker is unavailable and *allow_insecure_fallback*
            is ``False``.
    """
    if prefer_docker:
        if _DOCKER_AVAILABLE:
            try:
                return DockerSandbox(**kwargs)
            except Exception:
                if not allow_insecure_fallback:
                    raise RuntimeError(
                        "DockerSandbox is unavailable and insecure local "
                        "execution is disabled. Set "
                        "allow_insecure_fallback=True to fall back to "
                        "SubprocessSandbox, or install Docker."
                    ) from None
                logger.debug("DockerSandbox unavailable, falling back to subprocess")
        else:
            if not allow_insecure_fallback:
                raise RuntimeError(
                    "Docker is not available and insecure local execution "
                    "is disabled. Install Docker or set "
                    "allow_insecure_fallback=True to use SubprocessSandbox."
                )
    # Drop kwargs not accepted by SubprocessSandbox
    subprocess_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ("default_timeout",)
    }
    return SubprocessSandbox(**subprocess_kwargs)


# ===================================================================
# @sandboxed_tool  decorator
# ===================================================================


def sandboxed_tool(
    func: Callable[..., Any] | None = None,
    *,
    sandbox: DockerSandbox | SubprocessSandbox | None = None,
    language: str = "python",
    timeout: int = 10,
    memory: str = "128m",
    cpu_quota: int = 50000,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator that wraps a function as a sandboxed :class:`Tool`.

    The decorated function's **name** and **docstring** become the
    tool metadata exposed to the LLM.  When the tool is invoked the
    ``code`` argument (or equivalent — see below) is executed inside
    the sandbox instead of running the function body directly.

    Dual-mode (bare or with arguments)::

        @sandboxed_tool
        async def run_python(code: str) -> str: ...

        @sandboxed_tool(language="cpp", timeout=5)
        async def compile_and_run(code: str) -> str: ...

    Parameters:
        func: The decorated function (when used bare).
        sandbox: Existing sandbox instance to reuse; created fresh if ``None``.
        language: ``"python"`` or ``"cpp"``.
        timeout: Max seconds per execution.
        memory: Memory limit (Docker format, e.g. ``"128m"``).
        cpu_quota: CPU limit in microseconds per 100ms.
    """

    def decorator(fn: Callable[..., Any]) -> Tool:
        fn_name = fn.__name__
        fn_doc = inspect.getdoc(fn) or ""
        sig_params = list(inspect.signature(fn).parameters.keys())
        code_param = sig_params[0] if sig_params else "code"

        async def sandboxed_call(**kwargs: Any) -> str:
            code = kwargs.get(code_param, "")
            if not isinstance(code, str):
                code = str(code)

            sb = sandbox or create_sandbox(
                memory=memory,
                cpu_quota=cpu_quota,
                default_timeout=timeout,
            )

            if isinstance(sb, DockerSandbox):
                async with sb as box:
                    result = await box.execute_code(language, code, timeout)
            else:
                result = await sb.execute_code(language, code, timeout)

            if result["exit_code"] != 0:
                raise ToolError(
                    fn_name,
                    result["stderr"].strip() or f"Exit code {result['exit_code']}",
                )
            stdout: str = result["stdout"].strip()
            return stdout

        sandboxed_call.__name__ = fn_name
        sandboxed_call.__qualname__ = fn.__qualname__
        sandboxed_call.__doc__ = fn_doc
        sandboxed_call.__wrapped__ = fn  # type: ignore[attr-defined]

        return Tool(sandboxed_call, name=fn_name, description=fn_doc)

    if func is not None:
        return decorator(func)
    return decorator


# ===================================================================
# Standalone convenience
# ===================================================================


async def execute_code(
    language: str,
    code: str,
    timeout: int = 10,
    *,
    sandbox: DockerSandbox | SubprocessSandbox | None = None,
) -> dict[str, Any]:
    """One-shot code execution in the best available sandbox.

    Creates a temporary sandbox, executes *code*, and returns the
    result dict.  Pass an existing *sandbox* to avoid creating a
    fresh container each call.

    Args:
        language: ``"python"`` or ``"cpp"``.
        code: Source code to execute.
        timeout: Max seconds.
        sandbox: Reuse an existing sandbox instance.

    Returns:
        ``{"stdout": str, "stderr": str, "exit_code": int}``
    """
    if sandbox is not None:
        return await sandbox.execute_code(language, code, timeout)

    sb = create_sandbox(default_timeout=timeout)
    if isinstance(sb, DockerSandbox):
        async with sb as box:
            return await box.execute_code(language, code, timeout)
    return await sb.execute_code(language, code, timeout)
