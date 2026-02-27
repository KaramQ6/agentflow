"""LLM provider abstraction for agentflow."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from openai import APIError, AsyncOpenAI, RateLimitError

from .exceptions import LLMError

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
        cache: ResponseCache | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
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
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate a completion from the LLM.

        Returns:
            Dict with keys: content, tokens, duration, model, cached.
        """
        effective_model = model or self.model

        # Cache lookup
        if self._cache is not None:
            cache_key = self._cache.make_key(messages, effective_model)
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True}

        start = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries + 1):
            # Rate limiting per attempt
            if self._rate_limiter is not None:
                await self._rate_limiter.acquire()

            try:
                response = await self._client.chat.completions.create(
                    model=effective_model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                duration = time.perf_counter() - start
                choice = response.choices[0]
                usage = response.usage

                result: dict[str, Any] = {
                    "content": choice.message.content or "",
                    "tokens": usage.total_tokens if usage else 0,
                    "duration": round(duration, 3),
                    "model": response.model,
                    "cached": False,
                }

                # Write to cache
                if self._cache is not None:
                    await self._cache.set(cache_key, {k: v for k, v in result.items() if k != "cached"})

                return result

            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except APIError as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                    continue
            finally:
                if self._rate_limiter is not None:
                    self._rate_limiter.release()

        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_error}")
