"""Retry helpers for transient LLM API failures."""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Awaitable, Callable, Optional

from ..utils.logger import setup_logger


logger = setup_logger("llm_retry")

# ---------------------------------------------------------------------------
# Global LLM rate limiter (token bucket)
#
# Controls the flow rate of LLM calls rather than capping concurrency.
# Tokens refill continuously at LLM_CALLS_PER_SECOND per second.
# When the bucket is full, calls pass through immediately; when empty,
# callers wait only as long as needed to earn the next token.
#
# This smooths startup bursts (thundering herd) without hard-closing the
# pipeline: slow periods let tokens accumulate so brief bursts are absorbed,
# and sustained high load is paced to the configured rate.
#
# Set LLM_CALLS_PER_SECOND in .env (default: 20). Set to 0 to disable.
# ---------------------------------------------------------------------------
class _TokenBucket:
    """Async token bucket rate limiter."""

    def __init__(self, rate: float) -> None:
        self._rate = rate          # tokens replenished per second
        self._tokens = rate        # start full (absorb an initial burst)
        self._last_refill: float = 0.0  # set on first acquire
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = asyncio.get_event_loop().time()
                if self._last_refill == 0.0:
                    self._last_refill = now
                elapsed = now - self._last_refill
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # How long until the next token arrives
                wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)


_llm_rate_limiter: Optional[_TokenBucket] = None


def _get_rate_limiter() -> Optional[_TokenBucket]:
    """Return the module-level rate limiter, initialising it on first call."""
    global _llm_rate_limiter
    if _llm_rate_limiter is not None:
        return _llm_rate_limiter
    rate = float(os.getenv("LLM_CALLS_PER_SECOND", "20"))
    if rate <= 0:
        return None
    _llm_rate_limiter = _TokenBucket(rate)
    logger.info("LLM rate limiter initialised: %.1f calls/sec", rate)
    return _llm_rate_limiter


# ---------------------------------------------------------------------------
# Global CDP connection rate limiter (token bucket)
#
# Controls how fast new CDP connections (Playwright connect_over_cdp) are
# established.  Each connection spawns or attaches to a Chromium process and
# allocates memory; bursting them all simultaneously can cause OOM or
# Playwright connection timeouts under high concurrency.
#
# Tokens refill at CDP_CONNECTS_PER_SECOND per second.  Set to 0 to disable.
# Default: 5 connects/sec (conservative — each connect is heavy).
# ---------------------------------------------------------------------------
_cdp_rate_limiter: Optional[_TokenBucket] = None


def _get_cdp_rate_limiter() -> Optional[_TokenBucket]:
    """Return the module-level CDP rate limiter, initialising it on first call."""
    global _cdp_rate_limiter
    if _cdp_rate_limiter is not None:
        return _cdp_rate_limiter
    rate = float(os.getenv("CDP_CONNECTS_PER_SECOND", "5"))
    if rate <= 0:
        return None
    _cdp_rate_limiter = _TokenBucket(rate)
    logger.info("CDP rate limiter initialised: %.1f connects/sec", rate)
    return _cdp_rate_limiter


async def acquire_cdp_permit() -> None:
    """Acquire a CDP connection rate-limit token before calling start_service."""
    limiter = _get_cdp_rate_limiter()
    if limiter is not None:
        await limiter.acquire()

# Global LLM connection-error counter.  Every "connection error" / "connection reset"
# / "connection aborted" transient failure increments this.  The batch drain loop
# in open_core.py reads and resets it to decide when to drain.
_llm_connection_error_count: int = 0

# Markers that count as "connection errors" (a subset of the broader transient set).
_CONNECTION_ERROR_MARKERS = frozenset({
    "connection reset", "connection aborted", "connection error", "apiconnectionerror",
})


def _is_connection_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    error_type = error.__class__.__name__.lower()
    return any(m in error_msg or m in error_type for m in _CONNECTION_ERROR_MARKERS)


def get_llm_connection_error_count() -> int:
    """Return the global LLM connection-error count since last reset."""
    return _llm_connection_error_count


def reset_llm_connection_error_count() -> None:
    """Reset the global connection-error counter to zero."""
    global _llm_connection_error_count
    _llm_connection_error_count = 0


def is_transient_llm_error(error: Exception) -> bool:
    """Return True when an LLM exception appears transient and safe to retry."""
    error_msg = str(error).lower()
    error_type = error.__class__.__name__.lower()
    transient_markers = (
        "rate limit",
        "rate-limited",
        "ratelimit",
        "too many requests",
        "quota",
        "overloaded",
        "overload",
        "temporarily unavailable",
        "service unavailable",
        "server error",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection error",
        "apiconnectionerror",
        "apitimeouterror",
        "unavailable",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in error_msg or marker in error_type for marker in transient_markers)


async def invoke_with_llm_api_retry(
    operation: Callable[[], Awaitable[Any]],
    *,
    retries: int = 5,
    base_delay_seconds: float = 1.0,
    jitter_seconds: float = 0.35,
    operation_name: str = "LLM request",
    should_retry: Optional[Callable[[Exception], bool]] = None,
) -> Any:
    """Execute an async LLM operation with retry for transient provider failures."""
    retry_predicate = should_retry or is_transient_llm_error
    rate_limiter = _get_rate_limiter()
    _pre_jitter_ms = int(os.getenv("LLM_PRE_CALL_JITTER_MS", "100"))
    for retry_index in range(retries + 1):
        try:
            if rate_limiter is not None:
                await rate_limiter.acquire()
            if _pre_jitter_ms > 0:
                await asyncio.sleep(random.uniform(0, _pre_jitter_ms / 1000.0))
            return await operation()
        except Exception as error:
            if not retry_predicate(error):
                raise
            # Count every connection-type error globally (not just final failure).
            if _is_connection_error(error):
                global _llm_connection_error_count
                _llm_connection_error_count += 1
            if retry_index >= retries:
                logger.error(
                    "%s failed after %s retries [%s]: %s",
                    operation_name,
                    retries,
                    error.__class__.__name__,
                    error,
                )
                raise

            delay = max(0.0, base_delay_seconds) * (2 ** retry_index)
            if jitter_seconds > 0:
                delay += random.uniform(0.0, jitter_seconds)
            logger.warning(
                "%s transient failure; retrying in %.2fs (%s/%s): [%s] %s",
                operation_name,
                delay,
                retry_index + 1,
                retries,
                error.__class__.__name__,
                error,
            )
            import traceback
            logger.warning("Full traceback:\n%s", traceback.format_exc())
            await asyncio.sleep(delay)