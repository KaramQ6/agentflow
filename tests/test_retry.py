"""Tests for retry backoff and Retry-After handling."""

from types import SimpleNamespace

from agentflow import LLM
from agentflow.llm import _retry_after_seconds


def _error_with_retry_after(value):
    """A minimal duck-typed API error carrying a Retry-After header."""
    return SimpleNamespace(response=SimpleNamespace(headers={"retry-after": value}))


def test_retry_after_parsed_when_present():
    assert _retry_after_seconds(_error_with_retry_after("7")) == 7.0


def test_retry_after_absent_returns_none():
    assert _retry_after_seconds(ValueError("no response")) is None
    assert _retry_after_seconds(SimpleNamespace(response=None)) is None


def test_retry_after_non_numeric_returns_none():
    assert _retry_after_seconds(_error_with_retry_after("not-a-number")) is None


def test_backoff_is_exponential_without_jitter():
    llm = LLM(api_key="test", retry_base_delay=1.0, retry_jitter=False)
    plain = ValueError("boom")
    assert llm._backoff_delay(0, plain) == 1.0
    assert llm._backoff_delay(1, plain) == 2.0
    assert llm._backoff_delay(2, plain) == 4.0


def test_backoff_prefers_retry_after_header():
    llm = LLM(api_key="test", retry_base_delay=1.0, retry_jitter=False)
    assert llm._backoff_delay(3, _error_with_retry_after("2.5")) == 2.5


def test_backoff_jitter_within_bounds():
    llm = LLM(api_key="test", retry_base_delay=1.0, retry_jitter=True)
    plain = ValueError("boom")
    # attempt 1 → base*2 = 2.0, plus jitter in [0, 1.0)
    for _ in range(50):
        delay = llm._backoff_delay(1, plain)
        assert 2.0 <= delay < 3.0
