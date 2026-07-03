"""LLM provider abstraction for agentflow."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from openai import APIError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from .exceptions import LLMError
from .pricing import estimate_cost

if TYPE_CHECKING:
    from .cache import ResponseCache
    from .rate_limiter import RateLimiter


class LLM:
    """OpenAI-compatible LLM provider.

    Works with any provider that exposes an OpenAI-compatible API:
    OpenAI, Groq, Together, Ollama, vLLM, etc.

    Args:
        model: Model name (e.g. "gpt-4o", "llama-3.3-70b-versatile").
        api_key: API key for the provider.
        base_url: Base URL for the API (default: OpenAI).
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in response.
        max_retries: Number of retries on transient failures.
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
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        retry_jitter: bool = True,
        cache: ResponseCache | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.model = model
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

        self._client = AsyncOpenAI(**kwargs)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
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

        Returns:
            Dict with keys: content, tokens, prompt_tokens, completion_tokens,
            duration, model, cached, tool_calls, finish_reason.
        """
        effective_model = model or self.model
        use_cache = self._cache is not None and not tools

        # Cache lookup (skipped when tools are in play)
        cache_key = ""
        if use_cache:
            assert self._cache is not None
            cache_key = self._cache.make_key(messages, effective_model)
            cached = await self._cache.get(cache_key)
            if cached is not None:
                # A cache hit bills nothing — surface zero cost while keeping
                # the (informational) token counts from the original call.
                return {**cached, "cached": True, "cost": 0.0}

        start = time.perf_counter()
        last_error: Exception | None = None

        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
            extra["tool_choice"] = tool_choice or "auto"

        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting per attempt
                if self._rate_limiter is not None:
                    await self._rate_limiter.acquire()
                response = await self._client.chat.completions.create(
                    model=effective_model,
                    messages=cast("list[ChatCompletionMessageParam]", messages),
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    **extra,
                )
                duration = time.perf_counter() - start
                choice = response.choices[0]
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                result: dict[str, Any] = {
                    "content": choice.message.content or "",
                    "tokens": usage.total_tokens if usage else 0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": estimate_cost(response.model, prompt_tokens, completion_tokens),
                    "duration": round(duration, 3),
                    "model": response.model,
                    "cached": False,
                    "tool_calls": _serialize_tool_calls(choice.message.tool_calls),
                    "finish_reason": choice.finish_reason,
                }

                # Write to cache (never cache tool-calling turns)
                if use_cache:
                    assert self._cache is not None
                    await self._cache.set(cache_key, {k: v for k, v in result.items() if k != "cached"})

                return result

            except (RateLimitError, APIError) as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff_delay(attempt, e))
                    continue
            finally:
                if self._rate_limiter is not None:
                    self._rate_limiter.release()

        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_error}")

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
        try:
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()
            stream = await self._client.chat.completions.create(
                model=effective_model,
                messages=cast("list[ChatCompletionMessageParam]", messages),
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
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
