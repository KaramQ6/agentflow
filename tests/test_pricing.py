"""Tests for token-cost estimation."""

import logging

import pytest

from agentflow import estimate_cost, register_price
from agentflow.pricing import _unpriced_seen, get_price, is_priced, warn_if_unpriced


@pytest.fixture(autouse=True)
def _reset_unpriced_warnings():
    """warn_if_unpriced dedupes per process; isolate that state per test."""
    _unpriced_seen.clear()
    yield
    _unpriced_seen.clear()


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


# ── Unpriced models must be loud, not silently $0 ─────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        # The README quickstart points at Groq; these used to bill $0.00, which
        # made the headline cost-tracking feature report nothing for the exact
        # provider the docs recommend.
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "deepseek-chat",
        "mistral-large-latest",
    ],
)
def test_recommended_provider_models_are_priced(model):
    assert is_priced(model), f"{model} is recommended in the docs but has no price"
    assert estimate_cost(model, 1_000_000, 1_000_000) > 0


def test_is_priced_false_for_unknown_model():
    assert not is_priced("totally-unknown-model-xyz")


def test_warn_if_unpriced_logs_once_per_model(caplog):
    with caplog.at_level(logging.WARNING, logger="agentflow.pricing"):
        warn_if_unpriced("mystery-model-a")
        warn_if_unpriced("mystery-model-a")
        warn_if_unpriced("mystery-model-b")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2, "each unknown model warns exactly once per process"
    assert "mystery-model-a" in warnings[0].getMessage()
    assert "register_price" in warnings[0].getMessage()


def test_warn_if_unpriced_is_silent_for_known_models(caplog):
    with caplog.at_level(logging.WARNING, logger="agentflow.pricing"):
        warn_if_unpriced("gpt-4o-mini")
        warn_if_unpriced("llama-3.3-70b-versatile")
    assert not caplog.records


def test_warn_if_unpriced_ignores_empty_model_name(caplog):
    with caplog.at_level(logging.WARNING, logger="agentflow.pricing"):
        warn_if_unpriced("")
    assert not caplog.records


def test_registering_a_price_stops_the_warning(caplog):
    warn_if_unpriced("late-priced-model")
    register_price("late-priced-model", 1.0, 2.0)

    caplog.clear()  # drop the warning from before the price was registered
    with caplog.at_level(logging.WARNING, logger="agentflow.pricing"):
        warn_if_unpriced("late-priced-model")
    assert not caplog.records


def test_model_family_variants_do_not_inherit_the_wrong_price():
    """Longest-prefix matching is only safe when variants are listed too."""
    assert get_price("llama-3.1-8b-instant") != get_price("llama-3.3-70b-versatile")
    assert get_price("deepseek-reasoner") != get_price("deepseek-chat")
    # A dated variant still resolves to its own family, not the shorter sibling.
    assert get_price("llama-3.1-8b-instant-0125") == get_price("llama-3.1-8b-instant")
