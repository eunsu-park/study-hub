"""
Exercises for Lesson 08: Async Programming
Topic: Python

Solutions to practice problems from the lesson.
"""

import asyncio
import time
from pathlib import Path
import tempfile
import os


# === Exercise 1: Web Crawler ===
# Problem: Write an async crawler that fetches multiple URLs concurrently.

async def fetch_url(url: str, delay: float = 0.0) -> dict:
    """Simulate fetching a URL with an async delay.

    In production, you would use aiohttp.ClientSession.get(url).
    We simulate network latency with asyncio.sleep to demonstrate
    concurrent execution without requiring real HTTP requests.
    """
    await asyncio.sleep(delay)  # Simulate network latency
    return {
        "url": url,
        "status": 200,
        "content_length": len(url) * 100,  # Fake content
    }


async def crawl(urls: list[str], max_concurrent: int = 3) -> list[dict]:
    """Fetch multiple URLs concurrently with a concurrency limit.

    Uses a semaphore to cap the number of simultaneous requests.
    Without this, launching thousands of requests at once would
    overwhelm the server and exhaust local resources.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def fetch_with_limit(url: str) -> dict:
        async with semaphore:
            # Simulate varying response times
            delay = len(url) % 3 * 0.1
            return await fetch_url(url, delay)

    # Launch all requests concurrently (semaphore limits actual concurrency)
    tasks = [fetch_with_limit(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return list(results)


def exercise_1():
    """Demonstrate async web crawler."""
    urls = [
        "https://example.com",
        "https://example.com/about",
        "https://example.com/products",
        "https://example.com/contact",
        "https://example.com/blog",
    ]

    start = time.time()
    results = asyncio.run(crawl(urls, max_concurrent=3))
    elapsed = time.time() - start

    for r in results:
        print(f"  {r['url']} -> status={r['status']}, size={r['content_length']}")
    print(f"  Total time: {elapsed:.2f}s (concurrent, not sequential)")


# === Exercise 2: Concurrent File Processing ===
# Problem: Write a function that reads and processes multiple files concurrently.

async def read_file_async(filepath: str) -> str:
    """Read a file asynchronously using asyncio.to_thread.

    File I/O is blocking, so we offload it to a thread pool via
    asyncio.to_thread. This lets other coroutines run while the
    OS performs the actual disk read.
    """
    def _read(path):
        with open(path, "r") as f:
            return f.read()
    return await asyncio.to_thread(_read, filepath)


async def process_files(filepaths: list[str]) -> dict[str, dict]:
    """Read and process multiple files concurrently.

    Returns a dict mapping each filepath to its stats
    (line count, word count, char count).
    """
    async def process_one(filepath: str) -> tuple[str, dict]:
        content = await read_file_async(filepath)
        lines = content.splitlines()
        words = content.split()
        return filepath, {
            "lines": len(lines),
            "words": len(words),
            "chars": len(content),
        }

    tasks = [process_one(fp) for fp in filepaths]
    results = await asyncio.gather(*tasks)
    return dict(results)


def exercise_2():
    """Demonstrate concurrent file processing."""
    # Create temporary test files
    tmpdir = tempfile.mkdtemp()
    filepaths = []
    for i in range(5):
        path = os.path.join(tmpdir, f"file_{i}.txt")
        with open(path, "w") as f:
            f.write(f"This is file number {i}.\n" * (i + 1))
            f.write(f"It contains {(i + 1) * 5} words approximately.\n")
        filepaths.append(path)

    results = asyncio.run(process_files(filepaths))

    for filepath, stats in results.items():
        name = os.path.basename(filepath)
        print(f"  {name}: {stats['lines']} lines, {stats['words']} words, {stats['chars']} chars")

    # Clean up
    for fp in filepaths:
        os.remove(fp)
    os.rmdir(tmpdir)


# === Exercise 3: Rate Limiter ===
# Problem: Write an async function that limits requests per second.

class AsyncRateLimiter:
    """Async rate limiter using a token bucket algorithm.

    Tokens are added at a fixed rate (tokens_per_second). Each request
    consumes one token. If no tokens are available, the coroutine
    awaits until a token is replenished. This provides smooth
    rate limiting without bursty behavior.
    """

    def __init__(self, max_requests: int, per_seconds: float = 1.0):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.tokens = max_requests
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a token is available, then consume it."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            # Replenish tokens based on elapsed time
            self.tokens = min(
                self.max_requests,
                self.tokens + elapsed * (self.max_requests / self.per_seconds)
            )
            self.last_refill = now

            if self.tokens < 1:
                # Calculate wait time for next token
                wait = (1 - self.tokens) / (self.max_requests / self.per_seconds)
                await asyncio.sleep(wait)
                self.tokens = 0
            else:
                self.tokens -= 1


async def rate_limited_requests(urls: list[str], rate: int = 3):
    """Fetch URLs with a rate limit of `rate` requests per second."""
    limiter = AsyncRateLimiter(max_requests=rate, per_seconds=1.0)
    results = []

    async def fetch(url: str) -> dict:
        await limiter.acquire()
        timestamp = time.time()
        response = await fetch_url(url, delay=0.05)
        response["fetched_at"] = f"{timestamp:.3f}"
        return response

    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results


def exercise_3():
    """Demonstrate async rate limiter."""
    urls = [f"https://api.example.com/data/{i}" for i in range(6)]

    start = time.time()
    results = asyncio.run(rate_limited_requests(urls, rate=3))
    elapsed = time.time() - start

    for r in results:
        print(f"  {r['url']} fetched at t={r['fetched_at']}")
    print(f"  Total time: {elapsed:.2f}s (rate-limited to 3/s for 6 requests)")


if __name__ == "__main__":
    print("=== Exercise 1: Web Crawler ===")
    exercise_1()

    print("\n=== Exercise 2: Concurrent File Processing ===")
    exercise_2()

    print("\n=== Exercise 3: Rate Limiter ===")
    exercise_3()

    print("\nAll exercises completed!")
