import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from vlm_annotation.src.models.base import ModelMemoryError

logger = logging.getLogger("vlm_annotation.retry")

T = TypeVar("T")


class RateLimiter:
    """Token-bucket rate limiter with concurrency control."""

    def __init__(self, requests_per_minute: int = 30, max_concurrency: int = 5):
        self.rpm = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        await self.semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                await asyncio.sleep(sleep_time)
            self.last_request_time = time.monotonic()

    def release(self):
        self.semaphore.release()


async def execute_with_retry(
    func: Callable[..., Any],
    *args: Any,
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    max_backoff: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    model_instance: Optional[Any] = None,
    **kwargs: Any
) -> Any:
    """
    Executes function with rate limiter acquisition, exponential backoff, jitter,
    and increments rate_limit_hits counter on model_instance when HTTP 429 is encountered.
    """
    retries = 0
    backoff = initial_backoff

    while True:
        if rate_limiter:
            await rate_limiter.acquire()
        try:
            start_t = time.monotonic()
            result = await func(*args, **kwargs)
            return result
        except Exception as exc:
            err_msg = str(exc).lower()

            # Deterministic memory errors must fail fast, not retry.
            if isinstance(exc, ModelMemoryError) or any(
                kw in err_msg
                for kw in ["requires more system memory", "out of memory", "not enough memory", "insufficient memory", "cuda out of memory", "allocation failed"]
            ):
                if model_instance:
                    model_instance.failed_requests += 1
                logger.error(f"Non-retryable memory error: {exc}")
                raise exc

            is_rate_limit = "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg or "quota" in err_msg

            if is_rate_limit and model_instance:
                model_instance.rate_limit_hits += 1

            retries += 1
            if retries > max_retries:
                logger.error(f"Exhausted retries ({max_retries}) for request. Last error: {exc}")
                raise exc

            current_backoff = min(backoff, max_backoff)
            if jitter:
                current_backoff *= (0.8 + 0.4 * random.random())

            logger.warning(
                f"Attempt {retries}/{max_retries} failed ({exc}). Retrying in {current_backoff:.2f}s..."
            )
            await asyncio.sleep(current_backoff)
            backoff *= backoff_factor
        finally:
            if rate_limiter:
                rate_limiter.release()
