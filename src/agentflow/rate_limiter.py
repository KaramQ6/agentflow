"""Rate limiting for LLM API calls."""

from __future__ import annotations

import asyncio
import time
from collections import deque


class RateLimiter:
    """Async rate limiter enforcing RPM and concurrency limits.

    Uses a sliding-window counter for requests-per-minute and an
    asyncio.Semaphore for max-concurrent requests.

    Args:
        requests_per_minute: Maximum requests allowed per 60-second window.
        max_concurrent: Maximum simultaneous in-flight requests (default 10).

    Usage:
        limiter = RateLimiter(requests_per_minute=60, max_concurrent=5)
        llm = LLM(model="gpt-4o", rate_limiter=limiter)

        # Or use directly as an async context manager:
        async with limiter:
            result = await some_api_call()
    """

    def __init__(self, requests_per_minute: int, max_concurrent: int = 10):
        self._rpm = requests_per_minute
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # Track timestamps of recent requests within the 60s window
        self._request_times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available, then acquire it."""
        await self._semaphore.acquire()
        await self._wait_for_window()

    def release(self) -> None:
        """Release the concurrency slot."""
        self._semaphore.release()

    async def _wait_for_window(self) -> None:
        """Block until we are below the RPM limit."""
        async with self._lock:
            now = time.monotonic()
            window_start = now - 60.0

            # Drop requests older than the 60s window
            while self._request_times and self._request_times[0] < window_start:
                self._request_times.popleft()

            if len(self._request_times) >= self._rpm:
                # Need to wait until the oldest request in the window expires
                sleep_for = 60.0 - (now - self._request_times[0]) + 0.01
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                # Re-clean after sleeping
                now = time.monotonic()
                window_start = now - 60.0
                while self._request_times and self._request_times[0] < window_start:
                    self._request_times.popleft()

            self._request_times.append(time.monotonic())

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        self.release()
