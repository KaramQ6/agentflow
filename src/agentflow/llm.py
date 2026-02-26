"""LLM provider abstraction for agentflow."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from openai import AsyncOpenAI, APIError, RateLimitError

from .exceptions import LLMError


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
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

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
            Dict with keys: content, tokens, duration, model.
        """
        start = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                duration = time.perf_counter() - start
                choice = response.choices[0]
                usage = response.usage

                return {
                    "content": choice.message.content or "",
                    "tokens": usage.total_tokens if usage else 0,
                    "duration": round(duration, 3),
                    "model": response.model,
                }

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

        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_error}")
