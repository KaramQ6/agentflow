"""C++ Build Pipeline — multi-agent write/compile/test pipeline.

A two-agent pipeline:
  Agent A (cpp_writer) — writes C++ code from a natural-language prompt.
  Agent B (cpp_tester) — receives the code, compiles it, runs the binary,
                          feeds compile errors back to the LLM for fixes,
                          and reports results.

The pipeline uses two tools:
  `write_file` — saves source code to disk (used by Agent A).
  `compile_and_run` — compiles with g++/cl, runs the binary, returns
                      stdout/stderr and the exit code (used by Agent B).

Requirements:
  - g++ (MinGW / Linux / macOS) or cl.exe (MSVC) on PATH
  - GROQ_API_KEY or OPENAI_API_KEY

Run: python examples/cpp_build_pipeline.py "Write a C++ function that checks if a number is prime"
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from agentflow import LLM, Agent, Pipeline, tool


# ─── Tools ─────────────────────────────────────────────────────────────────────

# Shared state between agents — current working source file path
_SOURCE_PATH: Path | None = None


@tool
def write_file(filename: str, content: str) -> str:
    """Write C++ source code to a file. Use this to save generated code
    before compilation. Provide the full filename (e.g. 'main.cpp') and
    the complete source code as a string.

    Args:
        filename: The filename to write (e.g. 'solution.cpp').
        content: The complete C++ source code.
    """
    global _SOURCE_PATH
    tmpdir = Path(tempfile.gettempdir()) / "agentflow_cpp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    _SOURCE_PATH = tmpdir / filename
    _SOURCE_PATH.write_text(content, encoding="utf-8")
    return (
        f"Written {len(content)} bytes to {_SOURCE_PATH}\n"
        f"First 3 lines preview:\n"
        + "\n".join(f"  {i + 1}: {line}" for i, line in enumerate(content.splitlines()[:3]))
    )


@tool
def compile_and_run(compiler_args: str = "") -> str:
    """Compile the current C++ source file and run the resulting binary.
    Capture stdout, stderr, and exit code.

    Args:
        compiler_args: Extra flags for the compiler (e.g. '-O2 -Wall').
                       Leave empty for defaults.
    """
    global _SOURCE_PATH
    if _SOURCE_PATH is None or not _SOURCE_PATH.exists():
        return "Error: no source file written yet. Call write_file first."

    # Detect compiler
    compiler = _find_compiler()
    if compiler is None:
        return (
            "Error: no C++ compiler found on PATH. "
            "Install g++ (MinGW on Windows) or cl.exe (MSVC)."
        )

    exe_path = _SOURCE_PATH.with_suffix(".exe" if platform.system() == "Windows" else "")

    # Compile
    extra_args = shlex_split(compiler_args)
    compile_cmd = [compiler, "-std=c++17", "-o", str(exe_path), str(_SOURCE_PATH)] + extra_args
    compile_result = subprocess.run(
        compile_cmd, capture_output=True, text=True, timeout=30,
    )

    if compile_result.returncode != 0:
        return (
            f"COMPILATION FAILED (exit code {compile_result.returncode}):\n"
            f"{compile_result.stderr or compile_result.stdout}"
        )

    # Run
    try:
        run_result = subprocess.run(
            [str(exe_path)], capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutError:
        return "RUNTIME ERROR: program execution timed out after 10 seconds."
    finally:
        # Cleanup binary
        try:
            exe_path.unlink(missing_ok=True)
        except OSError:
            pass

    return (
        f"COMPILATION OK (command: {' '.join(compile_cmd)})\n"
        f"EXECUTION OK (exit code {run_result.returncode}):\n"
        f"STDOUT:\n{run_result.stdout or '(no output)'}\n"
        f"STDERR:\n{run_result.stderr or '(none)'}"
    )


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _find_compiler() -> str | None:
    """Locate a C++ compiler available on PATH."""
    candidates = ["g++", "clang++", "c++", "cl"]
    for name in candidates:
        if shutil_which(name):
            return name
    return None


def shutil_which(cmd: str) -> str | None:
    """Simple which() without importing shutil (to keep tool modules lean)."""
    path_ext = os.environ.get("PATHEXT", "").split(os.pathsep)
    for dirname in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(dirname, cmd)
        if os.path.isfile(candidate):
            return candidate
        for ext in path_ext:
            full = candidate + ext
            if os.path.isfile(full):
                return full
    return None


def shlex_split(s: str) -> list[str]:
    """Parse a shell-like string into arguments (no shlex dependency)."""
    return re.findall(r'(?:[^\s"']+|"[^"]*"|\'[^\']*\')', s)


# ─── Agent A: C++ Writer ───────────────────────────────────────────────────────

@Agent(
    name="cpp_writer",
    role="Expert C++17 developer who writes clean, modern, well-documented code "
          "with proper error handling and RAII patterns",
    tools=[write_file],
    max_tool_iterations=4,
)
async def cpp_writer(task: str, context: dict) -> str:
    return (
        f"Write a complete, compilable C++17 program based on this prompt:\n\n"
        f"  {task}\n\n"
        f"Requirements:\n"
        f"- Include all necessary headers.\n"
        f"- Use a `main()` function that demonstrates the functionality.\n"
        f"- Print results clearly to stdout.\n"
        f"- Handle edge cases (empty input, invalid values, etc.).\n"
        f"- Use modern C++17 features where appropriate (auto, structured bindings, "
        f"  std::optional, etc.).\n"
        f"- Add brief comments explaining non-obvious logic.\n\n"
        f"After writing the code, use the `write_file` tool to save it as "
        f"'solution.cpp' — the tester agent will compile and run it automatically."
    )


# ─── Agent B: C++ Tester ───────────────────────────────────────────────────────

@Agent(
    name="cpp_tester",
    role="Build engineer who compiles, tests, and iteratively fixes C++ code",
    tools=[compile_and_run, write_file],
    max_tool_iterations=6,
)
async def cpp_tester(task: str, context: dict) -> str:
    """Receives the previous agent's output (the C++ code) and attempts to
    compile and run it. If compilation fails, it can fix the code in a
    ReAct loop using write_file + compile_and_run until it succeeds."""
    code = context.get("cpp_writer", "")
    if not code.strip():
        return "Error: cpp_writer did not produce any code. Aborting."

    return (
        f"You are testing C++ code that was just generated. The code is "
        f"saved at the path written by the cpp_writer agent.\n\n"
        f"The original task was:\n  {task}\n\n"
        f"Steps:\n"
        f"1. Call `compile_and_run` to build and execute the program.\n"
        f"2. If compilation fails, read the error output, fix the source "
        f"   using `write_file`, and re-run `compile_and_run`.\n"
        f"3. Keep iterating until the program compiles and runs successfully, "
        f"   or you have exhausted your attempts.\n"
        f"4. Report the final status: success/failure, output, and any fixes applied."
    )


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(llm: LLM) -> Pipeline:
    pipe = Pipeline(llm=llm, retry_failed_agents=0)
    pipe.add(cpp_writer)
    pipe.add(cpp_tester, depends_on=["cpp_writer"])
    return pipe


# ─── Terminal Display ──────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def divider() -> None:
    print(f"{DIM}{'─' * 60}{RESET}")


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main(task: str) -> None:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    use_groq = bool(os.environ.get("GROQ_API_KEY"))
    llm = LLM(
        model="llama-3.3-70b-versatile" if use_groq else "gpt-4o-mini",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1" if use_groq else None,
        max_tokens=2048,
    )

    pipe = build_pipeline(llm)

    print(f"\n{BOLD}C++ Build Pipeline{RESET}")
    print(f"{DIM}Two-agent pipeline: Writer → Tester{RESET}")
    print(f"{DIM}Task: {task}{RESET}")
    divider()

    result = await pipe.run(task)

    # Writer output
    writer = result.get("cpp_writer")
    if writer:
        print(f"\n{CYAN}▶ C++ WRITER{RESET}  {DIM}{writer.tokens_used} tokens · "
              f"{writer.duration:.1f}s{RESET}")
        tool_trace = writer.metadata.get("tool_calls", [])
        for call in tool_trace:
            print(f"  {YELLOW}⬢{RESET} {call['tool']}(...) "
                  f"{DIM}→ {str(call['result'])[:100]}{RESET}")

    # Tester output
    tester = result.get("cpp_tester")
    if tester:
        print(f"\n{GREEN}▶ C++ TESTER{RESET}  {DIM}{tester.tokens_used} tokens · "
              f"{tester.duration:.1f}s{RESET}")
        tool_trace = tester.metadata.get("tool_calls", [])
        for call in tool_trace:
            result_preview = str(call["result"])[:120]
            print(f"  {YELLOW}⬢{RESET} {call['tool']}(...) "
                  f"{DIM}→ {result_preview.replace(chr(10), ' ')}{RESET}")

    divider()
    print(f"\n{BOLD}Final Tester Report:{RESET}")
    print(tester.output if tester else "(no output)")

    print(f"\n{DIM}Total tokens: {result.total_tokens} | "
          f"Cost: ${result.total_cost:.6f} | "
          f"Wall time: {result.total_duration:.1f}s{RESET}\n")


if __name__ == "__main__":
    task_str = sys.argv[1] if len(sys.argv) > 1 else None
    if not task_str:
        task_str = input("What C++ program should I write? ").strip()
    if not task_str:
        print("No task provided.")
        sys.exit(1)
    asyncio.run(main(task_str))
