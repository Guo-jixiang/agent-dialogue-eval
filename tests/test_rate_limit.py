"""Tests for TokenBucket rate limiter."""
from __future__ import annotations

import asyncio
import time

import pytest

from core.rate_limit import TokenBucket, get_rate_limiter


class TestTokenBucket:
    """Unit tests for the async token bucket."""

    @pytest.mark.asyncio
    async def test_acquire_immediate_when_tokens_available(self):
        """With burst capacity, first acquires should be instant."""
        bucket = TokenBucket(rate=100.0, burst=10)
        start = time.monotonic()
        for _ in range(10):
            await bucket.acquire()
        elapsed = time.monotonic() - start
        # 10 tokens from burst → all immediate
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_acquire_waits_when_exhausted(self):
        """Once burst is exhausted, further acquires must wait for refill."""
        bucket = TokenBucket(rate=10.0, burst=2)  # 2 burst, 10/sec refill
        # Exhaust burst
        await bucket.acquire()
        await bucket.acquire()
        # 3rd acquire should wait ~0.1s (1 token / 10 rps)
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # at least some wait
        assert elapsed < 0.5    # not absurdly long

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """Tokens refill while waiting."""
        bucket = TokenBucket(rate=20.0, burst=3)
        # Exhaust
        for _ in range(3):
            await bucket.acquire()
        # Wait for refill
        await asyncio.sleep(0.15)  # refill ~3 tokens at 20/sec
        start = time.monotonic()
        await bucket.acquire()
        elapsed = time.monotonic() - start
        # Should have at least 1 token after 0.15s refill
        assert elapsed < 0.05  # near-instant

    @pytest.mark.asyncio
    async def test_multiple_coroutines_throttled(self):
        """Concurrent coroutines share the bucket fairly."""
        bucket = TokenBucket(rate=10.0, burst=4)
        completed = []

        async def worker(i: int):
            await bucket.acquire()
            completed.append(i)

        # 10 concurrent workers with burst=4 → first 4 instant, last 6 wait
        start = time.monotonic()
        await asyncio.gather(*(worker(i) for i in range(10)))
        elapsed = time.monotonic() - start

        assert len(completed) == 10
        # 6 waiters at 10/sec → at least ~0.4s total wall time
        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_total_waited_tracks_cumulative_wait(self):
        """total_waited accumulates across acquires."""
        bucket = TokenBucket(rate=10.0, burst=1)  # burst=1: first instant, rest wait
        for _ in range(4):
            await bucket.acquire()
        # 3 waiters × ~0.1s each → total ~0.3s
        assert bucket.total_waited >= 0.15

    def test_rejects_invalid_params(self):
        """Rate <= 0 or burst < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            TokenBucket(rate=0.0)
        with pytest.raises(ValueError):
            TokenBucket(rate=-1.0)
        with pytest.raises(ValueError):
            TokenBucket(rate=5.0, burst=0)

    def test_get_rate_limiter_returns_none_when_disabled(self):
        """rate=0 disables and returns None."""
        assert get_rate_limiter(rate=0.0) is None

    def test_get_rate_limiter_returns_same_instance(self):
        """Same params return the same singleton."""
        b1 = get_rate_limiter(rate=5.0, burst=10)
        b2 = get_rate_limiter(rate=5.0, burst=10)
        assert b1 is b2
        assert b1 is not None
