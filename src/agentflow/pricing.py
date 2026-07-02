"""Token-cost estimation for LLM calls.

Prices are USD per 1,000,000 tokens as ``(prompt, completion)`` and are
*indicative* — providers change them, and self-hosted models (Ollama, vLLM) are
free. Match is by longest name prefix so versioned ids like
``gpt-4o-2024-08-06`` resolve to ``gpt-4o``. Register or override any model with
:func:`register_price`.
"""

from __future__ import annotations

# model prefix -> (prompt_usd_per_1m, completion_usd_per_1m)
_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4-turbo": (10.00, 30.00),
    "o1-mini": (3.00, 12.00),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
}


def register_price(model: str, prompt_per_1m: float, completion_per_1m: float) -> None:
    """Add or override the price for a model (USD per 1M tokens)."""
    _PRICES[model] = (prompt_per_1m, completion_per_1m)


def get_price(model: str) -> tuple[float, float] | None:
    """Return ``(prompt, completion)`` per-1M price for ``model``, or None.

    Uses longest-prefix matching so versioned model ids resolve correctly.
    """
    match: tuple[float, float] | None = None
    match_len = -1
    for prefix, price in _PRICES.items():
        if model.startswith(prefix) and len(prefix) > match_len:
            match, match_len = price, len(prefix)
    return match


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of a call. Unknown models return ``0.0``."""
    price = get_price(model)
    if price is None:
        return 0.0
    prompt_price, completion_price = price
    cost = (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000
    return round(cost, 6)
