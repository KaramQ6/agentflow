"""Every shipped example must stay importable against the current public API.

This is the gate that was missing when `examples/robotics_mqtt_agent.py` and
`examples/drone_telemetry_agent.py` drifted onto APIs that never existed
(`@Agent(timeout=)`, `LLM(provider=)`, `hooks=PipelineLogger(...)`): ruff does
not resolve imports or check call signatures, and nothing else in CI ever
executed these files. Importing each module runs its decorators, its top-level
`LLM(...)` construction, and its `Pipeline` wiring — which is where that class
of drift shows up.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.py"))


def test_examples_directory_is_not_empty() -> None:
    """Guard the guard: an empty glob would make every check below vacuous."""
    assert EXAMPLE_FILES, f"no example modules found under {EXAMPLES_DIR}"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_imports(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Examples build an LLM at module scope; give them a key so the import
    # exercises the real constructor instead of bailing out early.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-not-a-real-key")

    module_name = f"_agentflow_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
