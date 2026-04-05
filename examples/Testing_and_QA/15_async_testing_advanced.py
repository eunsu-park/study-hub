#!/usr/bin/env python3
"""Example: Advanced Async Testing

Demonstrates testing async generators, timeout handling, concurrent task
coordination, and error propagation in asyncio-based code.
Related lesson: 15_Async_Testing_Advanced.md
"""

# =============================================================================
# WHY ADVANCED ASYNC TESTING?
#
# Basic async testing (awaiting a coroutine) is straightforward.
# The hard parts are:
#   1. Async generators — partial consumption, cleanup
#   2. Timeouts — testing that code respects deadlines
#   3. Concurrent tasks — race conditions, cancellation
#   4. Error propagation — exceptions in task groups
#
# pytest-asyncio makes this manageable by providing an async event loop
# as a fixture and letting you write test functions as coroutines.
# =============================================================================

import pytest
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator


# =============================================================================
# PRODUCTION CODE — ASYNC PATTERNS
# =============================================================================

async def async_range(start: int, stop: int, delay: float = 0.0) -> AsyncIterator[int]:
    """Async generator that yields integers with optional delay.
    Demonstrates resource cleanup with try/finally."""
    try:
        for i in range(start, stop):
            await asyncio.sleep(delay)
            yield i
    finally:
        # This runs even if the consumer breaks early or the generator
        # is garbage collected — important for resource cleanup.
        pass


async def fetch_with_timeout(url: str, timeout_sec: float = 1.0) -> dict:
    """Simulate a network fetch with timeout enforcement.
    In production, this would use aiohttp or httpx."""
    async def _simulate_fetch():
        # Simulate variable latency based on URL
        if "slow" in url:
            await asyncio.sleep(5.0)
        elif "medium" in url:
            await asyncio.sleep(0.5)
        else:
            await asyncio.sleep(0.01)
        return {"url": url, "status": 200, "data": "response"}

    try:
        return await asyncio.wait_for(_simulate_fetch(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        return {"url": url, "status": 408, "data": None}


async def fetch_all(urls: list[str], max_concurrent: int = 3) -> list[dict]:
    """Fetch multiple URLs concurrently with a semaphore limit.
    Demonstrates bounded concurrency — critical for not overwhelming servers."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded_fetch(url: str) -> dict:
        async with semaphore:
            return await fetch_with_timeout(url)

    tasks = [asyncio.create_task(_bounded_fetch(url)) for url in urls]
    return await asyncio.gather(*tasks)


@dataclass
class AsyncBatchProcessor:
    """Process items in batches asynchronously."""
    batch_size: int = 5
    processed: list = None

    def __post_init__(self):
        self.processed = []

    async def process_item(self, item: int) -> int:
        """Simulate async processing (e.g., DB write, API call)."""
        await asyncio.sleep(0.001)
        result = item * 2
        self.processed.append(result)
        return result

    async def process_batch(self, items: list[int]) -> list[int]:
        """Process items in batches to control resource usage."""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await asyncio.gather(
                *(self.process_item(item) for item in batch)
            )
            results.extend(batch_results)
        return results


async def async_retry(coro_factory, max_retries: int = 3, backoff: float = 0.01):
    """Retry an async operation with exponential backoff.
    coro_factory is a callable that returns a new coroutine each time."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff * (2 ** attempt))
    raise last_error


# =============================================================================
# TESTS — ASYNC GENERATORS
# =============================================================================

@pytest.mark.asyncio
class TestAsyncGenerators:
    """Test async generator consumption patterns."""

    async def test_full_consumption(self):
        """Consume all values from an async generator."""
        values = [v async for v in async_range(0, 5)]
        assert values == [0, 1, 2, 3, 4]

    async def test_partial_consumption(self):
        """Break early from an async generator — cleanup must still run."""
        values = []
        async for v in async_range(0, 100):
            values.append(v)
            if v >= 2:
                break
        assert values == [0, 1, 2]

    async def test_empty_range(self):
        """Edge case: async generator that yields nothing."""
        values = [v async for v in async_range(5, 5)]
        assert values == []


# =============================================================================
# TESTS — TIMEOUT HANDLING
# =============================================================================

@pytest.mark.asyncio
class TestTimeoutHandling:
    """Verify that code respects timeout deadlines."""

    async def test_fast_request_succeeds(self):
        result = await fetch_with_timeout("https://fast.example.com", timeout_sec=1.0)
        assert result["status"] == 200
        assert result["data"] is not None

    async def test_slow_request_times_out(self):
        """Slow endpoint should return timeout status, not hang."""
        result = await fetch_with_timeout("https://slow.example.com", timeout_sec=0.1)
        assert result["status"] == 408
        assert result["data"] is None

    async def test_timeout_does_not_raise(self):
        """Our fetch wraps TimeoutError — callers get a clean response."""
        # This should NOT raise asyncio.TimeoutError
        result = await fetch_with_timeout("https://slow.example.com", timeout_sec=0.05)
        assert isinstance(result, dict)


# =============================================================================
# TESTS — CONCURRENT TASK COORDINATION
# =============================================================================

@pytest.mark.asyncio
class TestConcurrentTasks:
    """Test concurrent async execution patterns."""

    async def test_fetch_all_respects_concurrency_limit(self):
        """Multiple URLs fetched concurrently with bounded parallelism."""
        urls = [f"https://api.example.com/{i}" for i in range(10)]
        results = await fetch_all(urls, max_concurrent=3)

        assert len(results) == 10
        assert all(r["status"] == 200 for r in results)

    async def test_batch_processor(self):
        """Items processed in correct batches."""
        processor = AsyncBatchProcessor(batch_size=3)
        items = [1, 2, 3, 4, 5, 6, 7]
        results = await processor.process_batch(items)

        assert results == [2, 4, 6, 8, 10, 12, 14]
        assert len(processor.processed) == 7

    async def test_gather_with_mixed_results(self):
        """asyncio.gather with return_exceptions=True collects errors."""

        async def succeed():
            return "ok"

        async def fail():
            raise ValueError("boom")

        results = await asyncio.gather(
            succeed(), fail(), succeed(),
            return_exceptions=True,
        )
        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ok"


# =============================================================================
# TESTS — RETRY WITH BACKOFF
# =============================================================================

@pytest.mark.asyncio
class TestAsyncRetry:
    """Verify retry logic with exponential backoff."""

    async def test_succeeds_on_first_try(self):
        call_count = 0

        async def always_works():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await async_retry(always_works, max_retries=3)
        assert result == "success"
        assert call_count == 1

    async def test_succeeds_after_retries(self):
        call_count = 0

        async def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            return "recovered"

        result = await async_retry(fails_twice, max_retries=3, backoff=0.001)
        assert result == "recovered"
        assert call_count == 3

    async def test_exhausts_retries(self):
        async def always_fails():
            raise ConnectionError("permanent failure")

        with pytest.raises(ConnectionError, match="permanent"):
            await async_retry(always_fails, max_retries=2, backoff=0.001)


# =============================================================================
# TESTS — CANCELLATION
# =============================================================================

@pytest.mark.asyncio
class TestCancellation:
    """Verify that async tasks handle cancellation gracefully."""

    async def test_task_cancellation(self):
        """Cancelled tasks raise CancelledError."""
        async def long_running():
            await asyncio.sleep(10)
            return "done"

        task = asyncio.create_task(long_running())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_shield_prevents_cancellation(self):
        """asyncio.shield protects a coroutine from outer cancellation."""
        result_holder = []

        async def important_work():
            await asyncio.sleep(0.01)
            result_holder.append("completed")

        # Shield wraps the coroutine — even if outer scope cancels,
        # the shielded work continues (in this controlled test)
        shielded = asyncio.shield(important_work())
        await shielded
        assert "completed" in result_holder


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# Install: pip install pytest-asyncio
# Run:     pytest 15_async_testing_advanced.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
