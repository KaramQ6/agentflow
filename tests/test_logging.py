"""Tests for structured JSON logging and LoggingHooks."""

import json
import logging

from agentflow import AgentResult, LoggingHooks, PipelineLogger, PipelineResult


def _last_json_line(captured: str) -> dict:
    return json.loads(captured.strip().splitlines()[-1])


def test_pipeline_logger_emits_structured_json(capsys):
    # Unique name → a fresh StreamHandler binds to the captured stdout.
    log = PipelineLogger("test-pipe-start", run_id="abcd1234")
    log.log_pipeline_start("my task", agent_count=3, level_count=2)

    rec = _last_json_line(capsys.readouterr().out)
    assert rec["event"] == "pipeline_start"
    assert rec["run_id"] == "abcd1234"
    assert rec["pipeline"] == "test-pipe-start"
    assert rec["agent_count"] == 3


def test_pipeline_logger_agent_and_complete(capsys):
    log = PipelineLogger("test-pipe-agent", run_id="ffff0000")
    log.log_agent_start("writer", level=1)
    log.log_agent_complete("writer", tokens=812, duration=1.2, cached=False)
    log.log_agent_error("writer", "boom")
    log.log_pipeline_complete("ffff0000", total_tokens=900, total_duration=2.0)

    lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events == [
        "agent_start",
        "agent_complete",
        "agent_error",
        "pipeline_complete",
    ]


def test_json_formatter_includes_exception(capsys):
    log = PipelineLogger("test-pipe-exc")
    try:
        raise ValueError("kaboom")
    except ValueError:
        log.error("failed", exc_info=True)

    rec = _last_json_line(capsys.readouterr().out)
    assert "kaboom" in rec["exception"]


def test_logging_hooks_full_lifecycle(capsys):
    hooks = LoggingHooks("hooked-pipe", level=logging.INFO)
    hooks.on_pipeline_start("task", run_id="run12345", agent_count=1)
    hooks.on_agent_start("a", level=0)
    hooks.on_agent_end(AgentResult(agent="a", output="ok", tokens_used=5, cached=False))
    hooks.on_agent_error("a", RuntimeError("nope"))
    hooks.on_pipeline_end(PipelineResult(output="ok", run_id="run12345", total_tokens=5))

    events = [json.loads(line)["event"] for line in capsys.readouterr().out.strip().splitlines()]
    assert events[0] == "pipeline_start"
    assert "agent_complete" in events
    assert events[-1] == "pipeline_complete"


def test_logging_hooks_noop_before_start():
    # Calling agent hooks before on_pipeline_start must not raise (logger is None).
    hooks = LoggingHooks("unstarted")
    hooks.on_agent_start("a", level=0)
    hooks.on_agent_end(AgentResult(agent="a", output="x"))
    hooks.on_agent_error("a", RuntimeError("x"))
