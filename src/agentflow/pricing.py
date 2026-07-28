"""Token-cost estimation for LLM calls.

Prices are USD per 1,000,000 tokens as ``(prompt, completion)``.

**These numbers drift.** Providers change them, self-hosted models (Ollama,
vLLM) are free, and a table baked into a release is stale the moment a provider
posts a price cut. Treat them as indicative and override anything that matters
to you with :func:`register_price` — that is the authoritative path, not this
table.

A model with no entry is billed ``0.0``. Silence there was the dangerous part:
a run on an unpriced model reported ``total_cost: $0.000000`` and looked free.
It is now surfaced twice — :func:`warn_if_unpriced` logs the model once per
process, and :attr:`agentflow.PipelineResult.unpriced_models` lists every model
in the run whose cost is a placeholder rather than an estimate.

Match is by longest name prefix, so versioned ids like ``gpt-4o-2024-08-06``
resolve to ``gpt-4o``. Prefix matching means a family and its variants must
both be listed or neither: with only ``gpt-4o`` present, ``gpt-4o-mini`` would
silently inherit the full-size price. Every family below lists its variants.
"""

from __future__ import annotations

import logging

_log = logging.getLogger("agentflow.pricing")

# model prefix -> (prompt_usd_per_1m, completion_usd_per_1m)
# Last reviewed: 2026-07. Verify against your provider's pricing page before
# relying on these for billing; use register_price() to correct any of them.
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
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
    # Groq — the provider the README quickstart points at, and the reason
    # unpriced models mattered: every run on Groq used to report $0.00.
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama3-70b-8192": (0.59, 0.79),
    "llama3-8b-8192": (0.05, 0.08),
    "gemma2-9b-it": (0.20, 0.20),
    # DeepSeek
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-chat": (0.27, 1.10),
    # Mistral
    "mistral-large": (2.00, 6.00),
    "mistral-small": (0.20, 0.60),
    "open-mixtral-8x7b": (0.70, 0.70),
}

_unpriced_seen: set[str] = set()


def register_price(model: str, prompt_per_1m: float, completion_per_1m: float) -> None:
    """Add or override the price for a model (USD per 1M tokens).

    This is the supported way to price self-hosted, fine-tuned, or
    newly-released models, and to correct a stale built-in entry.
    """
    _PRICES[model] = (prompt_per_1m, completion_per_1m)
    _unpriced_seen.discard(model)


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


def is_priced(model: str) -> bool:
    """Whether ``model`` has a price entry, i.e. whether its cost is real."""
    return get_price(model) is not None


def warn_if_unpriced(model: str) -> None:
    """Log once per process that ``model`` has no price and bills as ``0.0``.

    Kept separate from :func:`estimate_cost` so cost estimation stays a pure
    function; this is the effectful edge, called from the LLM client where a
    model name first arrives.
    """
    if not model or model in _unpriced_seen or is_priced(model):
        return
    _unpriced_seen.add(model)
    _log.warning(
        "No price registered for model %r — its cost is reported as $0.00, "
        "which is a placeholder and not an estimate. Register it with "
        "agentflow.register_price(%r, prompt_per_1m=..., completion_per_1m=...) "
        "to get real numbers.",
        model,
        model,
    )


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of a call. Unknown models return ``0.0``.

    A ``0.0`` from an unknown model is not a claim that the call was free — see
    :func:`is_priced` and :func:`warn_if_unpriced`.
    """
    price = get_price(model)
    if price is None:
        return 0.0
    prompt_price, completion_price = price
    cost = (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000
    return round(cost, 6)
