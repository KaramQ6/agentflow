"""Tests for the async RateLimiter."""

import asyncio

import pytest

from agentflow import RateLimiter


@pytest.mark.asyncio
async def test_concurrency_limit_blocks_and_releases():
    limiter = RateLimiter(requests_per_minute=10_000, max_concurrent=2)
    await limiter.acquire()
    await limiter.acquire()

    # A third concurrent acquire must block while both slots are held.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(limiter.acquire(), timeout=0.1)

    limiter.release()  # free one slot
    await asyncio.wait_for(limiter.acquire(), timeout=0.5)  # now succeeds
    limiter.release()
    limiter.release()


@pytest.mark.asyncio
async def test_context_manager_acquires_and_releases():
    limiter = RateLimiter(requests_per_minute=10_000, max_concurrent=1)
    async with limiter:
        # slot held → a second concurrent acquire blocks
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire(), timeout=0.1)
    # After exit the slot is free again.
    async with limiter:
        pass


@pytest.mark.asyncio
async def test_rpm_window_forces_wait(monkeypatch):
    limiter = RateLimiter(requests_per_minute=2, max_concurrent=10)

    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        slept.append(seconds)
        await real_sleep(0)  # record but don't actually wait

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    # Fill the per-minute window (2 requests), then a third must wait.
    for _ in range(2):
        await limiter.acquire()
        limiter.release()
    await limiter.acquire()
    limiter.release()

    assert slept and slept[0] > 0
