"""Tests for token-cost estimation."""

from agentflow import estimate_cost, register_price
from agentflow.pricing import get_price


def test_known_model_prefix_match():
    # gpt-4o-mini: (0.15, 0.60) per 1M tokens
    cost = estimate_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0)
    assert cost == 0.15


def test_versioned_model_resolves_to_prefix():
    # "gpt-4o-2024-08-06" should match "gpt-4o", not "gpt-4o-mini"
    assert get_price("gpt-4o-2024-08-06") == (2.50, 10.00)


def test_longest_prefix_wins():
    # "gpt-4o-mini-2024" must resolve to gpt-4o-mini, not gpt-4o
    assert get_price("gpt-4o-mini-2024") == (0.15, 0.60)


def test_prompt_and_completion_summed():
    cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
    # 1000/1e6*2.5 + 1000/1e6*10 = 0.0025 + 0.010 = 0.0125
    assert cost == 0.0125


def test_unknown_model_is_free():
    assert estimate_cost("some-local-llama", 1000, 1000) == 0.0
    assert get_price("some-local-llama") is None


def test_register_custom_price():
    register_price("my-model-v1", 1.0, 2.0)
    assert estimate_cost("my-model-v1", 1_000_000, 1_000_000) == 3.0
