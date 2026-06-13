"""Shared token-bucket rate limiter for outbound LLM API calls.

All :class:`LLMClient` instances share a single :class:`TokenBucket` so that
the three client roles (eval / sim / agent) don't exceed the provider's rate
limit in aggregate.
"""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Async token-bucket rate limiter.

    Tokens refill continuously at *rate* tokens/second up to *burst* capacity.
    Callers ``await acquire()`` before making an API call; if no token is
    available the coroutine sleeps until one becomes available.

    Thread-unsafe — designed for use within a single asyncio event loop.
    """

    def __init__(self, rate: float, burst: int = 10) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self._rate = float(rate)
        self._burst = float(burst)
        self._tokens = self._burst
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._total_waited = 0.0  # cumulative wait time for diagnostics

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def burst(self) -> int:
        return int(self._burst)

    @property
    def total_waited(self) -> float:
        return self._total_waited

    async def acquire(self) -> None:
        """Acquire one token, sleeping if necessary."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Not enough tokens — compute wait time
                deficit = 1.0 - self._tokens
                wait = deficit / self._rate
                self._tokens = 0.0

            # Sleep outside the lock so other coroutines can refill
            # After waking, loop back to re-check — another coroutine
            # may have taken the token while we slept
            self._total_waited += wait
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# Global singleton — created lazily from settings
# ---------------------------------------------------------------------------

_bucket: TokenBucket | None = None
_bucket_rate: float | None = None
_bucket_burst: int | None = None


def get_rate_limiter(rate: float = 0, burst: int = 10) -> TokenBucket | None:
    """Return the global :class:`TokenBucket`, creating it on first call.

    Pass ``rate=0`` to disable rate limiting (returns ``None``).
    """
    global _bucket, _bucket_rate, _bucket_burst
    if rate <= 0:
        return None
    if _bucket is None or rate != _bucket_rate or burst != _bucket_burst:
        _bucket = TokenBucket(rate=rate, burst=burst)
        _bucket_rate = rate
        _bucket_burst = burst
    return _bucket


# ---------------------------------------------------------------------------
# Global concurrency semaphore — caps in-flight LLM calls
# ---------------------------------------------------------------------------

_sem: asyncio.Semaphore | None = None
_sem_limit: int | None = None


def get_concurrency_sem(max_concurrency: int = 0) -> asyncio.Semaphore | None:
    """Return a global :class:`asyncio.Semaphore` capping concurrent LLM calls.

    Pass ``max_concurrency=0`` to disable (returns ``None``).
    """
    global _sem, _sem_limit
    if max_concurrency <= 0:
        return None
    if _sem is None or max_concurrency != _sem_limit:
        _sem = asyncio.Semaphore(max_concurrency)
        _sem_limit = max_concurrency
    return _sem
