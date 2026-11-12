"""LLM provider abstraction for agentflow."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from ._scope import current_scope, record_spend
from .exceptions import LLMError, PipelineError
from .pricing import estimate_cost, is_priced, warn_if_unpriced
from .types import LLMResponse

if TYPE_CHECKING:
    from .cache import ResponseCache
    from .rate_limiter import RateLimiter


_log = logging.getLogger("agentflow.llm")


class LLM:
    """OpenAI-compatible LLM provider.

    Works with any provider that exposes an OpenAI-compatible API:
    OpenAI, Groq, Together, Ollama, vLLM, etc.

    Args:
        model: Model name (e.g. "gpt-4o", "llama-3.3-70b-versatile").
        api_key: API key for the provider.
        base_url: Base URL for the API (default: OpenAI).
        temperature: Sampling temperature (0.0-2.0). ``None`` leaves it out of
            the request, which reasoning models require.
        max_tokens: Maximum tokens in response. ``None`` leaves it out.
        max_retries: Retries on transient failures only (connection errors,
            timeouts, 429 and 5xx). This is the only retry layer: the SDK's own
            retries are off, and agents add none.
        timeout: Seconds to wait for one request (default 120).
        retry_base_delay: Base seconds for exponential backoff (default 1.0).
        retry_jitter: Add random jitter to backoff to avoid thundering herds.
        cache: Optional ResponseCache for deduplicating identical requests.
        rate_limiter: Optional RateLimiter for throttling API calls.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = 4096,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        retry_jitter: bool = True,
        cache: ResponseCache | None = None,
        rate_limiter: RateLimiter | None = None,
        context_window: int | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        # Tokens the model accepts per request. When set, agents trim whole turns to
        # fit and refuse a request that cannot fit, instead of letting a local server
        # truncate it silently.
        self.context_window = context_window
        if context_window is None and _is_loopback(base_url):
            _log.warning(
                "LLM at %s has no context_window. Local servers such as Ollama cut a request "
                "that exceeds their context without an error. Pass context_window= (e.g. the "
                "model's num_ctx) so agents trim to fit.",
                base_url,
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_jitter = retry_jitter
        self._cache = cache
        self._rate_limiter = rate_limiter

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        # One retry layer (LLM-1): the SDK's own retries would multiply ours.
        self._client = AsyncOpenAI(max_retries=0, timeout=timeout, **kwargs)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        **extra: Any,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            messages: OpenAI-format chat messages.
            model: Override the default model for this call.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.
            tools: Optional list of tool schemas (OpenAI function format). When
                provided the response may contain ``tool_calls`` and the cache is
                bypassed (tools may have side effects).
            tool_choice: Optional OpenAI ``tool_choice`` ("auto", "none", etc.).
            **extra: Passed straight through to the provider's chat-completions
                call. The escape hatch for anything agentflow does not model —
                ``seed``, ``top_p``, ``stop``, ``response_format``,
                ``extra_body``. agentflow never sets these itself: a
                provider-native ``response_format`` is not portable across every
                OpenAI-compatible endpoint, so opting in is your call. These
                values are part of the cache key.

        Returns:
            An :class:`~agentflow.types.LLMResponse`. Dict-style access
            (``response["content"]``) still works as a deprecated shim.
        """
        effective_model = model or self.model
        use_cache = self._cache is not None and not tools

        # Cache lookup (skipped when tools are in play)
        cache_key = ""
        if use_cache:
            assert self._cache is not None
            cache_key = self._cache.make_key(
                messages,
                effective_model,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                extra=extra or None,
            )
            cached = await self._cache.get(cache_key)
            if cached is not None:
                # A cache hit bills nothing — surface zero cost while keeping
                # the (informational) token counts from the original call.
                return LLMResponse(**{**cached, "cached": True, "cost": 0.0})

        params: dict[str, Any] = {}
        effective_temperature = temperature if temperature is not None else self.temperature
        if effective_temperature is not None:
            params["temperature"] = effective_temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if effective_max_tokens is not None:
            params["max_tokens"] = effective_max_tokens
        params.update(extra)
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice or "auto"

        scope = current_scope()
        budget = scope.budget if scope is not None else None
        if budget is None:
            return await self._request(effective_model, messages, params, use_cache, cache_key)
        if not is_priced(effective_model):
            raise budget.block(unpriced_budget_error(effective_model))
        # Without max_tokens the provider decides; reserve a generous completion.
        reservation = estimate_cost(
            effective_model,
            _estimate_prompt_tokens(messages, tools),
            effective_max_tokens if effective_max_tokens is not None else _UNBOUNDED_COMPLETION,
        )
        await budget.reserve(reservation)
        billed = 0.0
        try:
            result = await self._request(effective_model, messages, params, use_cache, cache_key)
            billed = result.cost
            return result
        finally:
            budget.settle(reservation, billed)

    async def _request(
        self,
        effective_model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        use_cache: bool,
        cache_key: str,
    ) -> LLMResponse:
        """Retries on transient errors, then pricing, accounting and the cache."""
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            # Rate limiting per attempt; acquire outside the try so a failed
            # acquire is never paired with a release in the finally below.
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()
            try:
                result = await self._complete(effective_model, messages, params)
                break
            except LLMError:
                raise
            except Exception as e:
                if not _transient(e) or attempt == self.max_retries:
                    raise LLMError(f"LLM call failed after {attempt + 1} attempt(s): {e}") from e
                await asyncio.sleep(self._backoff_delay(attempt, e))
            finally:
                if self._rate_limiter is not None:
                    self._rate_limiter.release()

        # The response's model id is the one that was actually billed —
        # providers routinely return a more specific id than requested.
        warn_if_unpriced(result.model)
        cost = result.cost or estimate_cost(result.model, result.prompt_tokens, result.completion_tokens)
        result = result.model_copy(update={"cost": cost, "duration": round(time.perf_counter() - start, 3)})
        record_spend(result.cost, result.tokens)

        # Write to cache (never cache tool-calling turns)
        if use_cache:
            assert self._cache is not None
            await self._cache.set(cache_key, result.model_dump(exclude={"cached"}))
        return result

    async def _complete(
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> LLMResponse:
        """One provider round trip, nothing else: no retry, no pricing, no cache.

        The replacement seam. Override it to swap the provider, or use
        :class:`agentflow.testing.ScriptedLLM`; ``generate()`` keeps caching,
        budget reservation and accounting around it.
        """
        response = await self._client.chat.completions.create(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            **params,
        )
        if not response.choices:
            raise LLMError(f"provider returned no choices for model {model!r}")
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            tokens=usage.total_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            cached=False,
            tool_calls=_serialize_tool_calls(choice.message.tool_calls),
            finish_reason=choice.finish_reason,
        )

    def _backoff_delay(self, attempt: int, error: Exception) -> float:
        """Seconds to wait before the next retry.

        Honours a server-sent ``Retry-After`` header when present, otherwise
        uses exponential backoff (``base * 2**attempt``) plus optional jitter.
        """
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            return retry_after
        delay: float = self._retry_base_delay * (2 ** attempt)
        if self._retry_jitter:
            delay += random.uniform(0, self._retry_base_delay)
        return delay

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion token-by-token, yielding content deltas.

        A low-level primitive for interactive UIs. Unlike :meth:`generate` it
        does not cache or retry (streaming makes both ambiguous), but it still
        honours the rate limiter.

        Yields:
            Content string fragments as the model produces them.
        """
        effective_model = model or self.model
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        try:
            sampling: dict[str, Any] = {}
            effective_temperature = temperature if temperature is not None else self.temperature
            if effective_temperature is not None:
                sampling["temperature"] = effective_temperature
            effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            if effective_max_tokens is not None:
                sampling["max_tokens"] = effective_max_tokens
            stream = await self._client.chat.completions.create(
                model=effective_model,
                messages=cast("list[ChatCompletionMessageParam]", messages),
                stream=True,
                **sampling,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except APIError as e:
            raise LLMError(f"LLM stream failed: {e}") from e
        finally:
            if self._rate_limiter is not None:
                self._rate_limiter.release()


def _retry_after_seconds(error: Exception) -> float | None:
    """Extract a ``Retry-After`` header (in seconds) from an API error, if any."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


_UNBOUNDED_COMPLETION = 4096  # tokens reserved when max_tokens is None


def _transient(error: Exception) -> bool:
    """Worth retrying: the same request may succeed. A 4xx other than 429 never will."""
    if isinstance(error, (APIConnectionError, RateLimitError)):  # includes timeouts
        return True
    return isinstance(error, APIStatusError) and error.status_code >= 500


def _is_loopback(base_url: str | None) -> bool:
    if not base_url:
        return False
    from urllib.parse import urlparse

    return (urlparse(base_url).hostname or "") in {"localhost", "127.0.0.1", "::1"}


def unpriced_budget_error(model: str) -> PipelineError:
    return PipelineError(
        f"budget_usd is set, but model {model!r} has no registered price, so the budget cannot be "
        f"measured. For a local or free model call agentflow.register_price({model!r}, "
        f"prompt_per_1m=0, completion_per_1m=0); otherwise register its real price."
    )


def _estimate_prompt_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    """Deliberately pessimistic: 3 characters per token, where real text averages about 4."""
    chars = len(json.dumps(messages, default=str)) + (len(json.dumps(tools)) if tools else 0)
    return chars // 3 + 1


def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool-call objects into plain, JSON-serializable dicts."""
    if not tool_calls:
        return None
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in tool_calls
    ]
