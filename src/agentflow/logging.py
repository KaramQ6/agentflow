"""Structured logging for agentflow pipelines."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, MutableMapping


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields injected via LoggerAdapter
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message", "module",
                "msecs", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "taskName", "thread", "threadName",
            }:
                log[key] = value
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        return json.dumps(log, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a structured JSON logger.

    Args:
        name: Logger name (e.g. ``"agentflow.pipeline"``).
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class PipelineLogger(logging.LoggerAdapter):
    """Context-aware logger that attaches run_id and pipeline name to every record.

    Args:
        pipeline_name: Human-readable pipeline identifier.
        run_id: Short unique ID from PipelineResult.run_id.
        level: Logging level (default INFO).

    Usage:
        log = PipelineLogger("research-pipeline", run_id="a1b2c3d4")
        log.log_pipeline_start("AI in Healthcare", agent_count=3, level_count=2)
    """

    def __init__(self, pipeline_name: str, run_id: str = "", level: int = logging.INFO):
        logger = get_logger(f"agentflow.{pipeline_name}", level)
        super().__init__(logger, {"run_id": run_id, "pipeline": pipeline_name})

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs

    def log_pipeline_start(self, task: str, agent_count: int, level_count: int) -> None:
        self.info(
            "Pipeline started",
            extra={"event": "pipeline_start", "task_preview": task[:100],
                   "agent_count": agent_count, "level_count": level_count},
        )

    def log_agent_start(self, agent_name: str, level: int) -> None:
        self.info(
            f"Agent started: {agent_name}",
            extra={"event": "agent_start", "agent": agent_name, "level": level},
        )

    def log_agent_complete(
        self, agent_name: str, tokens: int, duration: float, cached: bool
    ) -> None:
        self.info(
            f"Agent complete: {agent_name}",
            extra={
                "event": "agent_complete",
                "agent": agent_name,
                "tokens": tokens,
                "duration_s": duration,
                "cached": cached,
            },
        )

    def log_agent_error(self, agent_name: str, error: str) -> None:
        self.error(
            f"Agent failed: {agent_name}",
            extra={"event": "agent_error", "agent": agent_name, "error": error},
        )

    def log_pipeline_complete(
        self, run_id: str, total_tokens: int, total_duration: float
    ) -> None:
        self.info(
            "Pipeline complete",
            extra={
                "event": "pipeline_complete",
                "run_id": run_id,
                "total_tokens": total_tokens,
                "total_duration_s": total_duration,
            },
        )
