#!/bin/bash
# Exercises for Lesson 09: Rate Limiting and Throttling
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: Token Bucket Implementation ===
# Problem: Implement a token bucket rate limiter from scratch and demonstrate
# how it handles burst traffic.
exercise_1() {
    echo "=== Exercise 1: Token Bucket Implementation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import time


class TokenBucket:
    """Token bucket rate limiter.

    The bucket starts full (capacity tokens). Each request consumes 1 token.
    Tokens are refilled at a constant rate. If the bucket is empty, reject.

    This allows bursts up to `capacity`, then throttles to `refill_rate`.
    """

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def _refill(self):
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def allow(self) -> bool:
        """Check if a request is allowed."""
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    @property
    def remaining(self) -> int:
        self._refill()
        return int(self.tokens)


# Demo: burst of 12 requests with capacity=10, refill=2/sec
bucket = TokenBucket(capacity=10, refill_rate=2.0)

print("Sending 12 requests in a burst:")
for i in range(12):
    allowed = bucket.allow()
    print(f"  Request {i+1}: {'ALLOWED' if allowed else 'REJECTED'} "
          f"(remaining: {bucket.remaining})")

print(f"\nWaiting 3 seconds for refill...")
time.sleep(3)  # Refills 6 tokens (3s * 2/sec)

print(f"After 3s: {bucket.remaining} tokens available")
for i in range(8):
    allowed = bucket.allow()
    print(f"  Request {i+1}: {'ALLOWED' if allowed else 'REJECTED'}")

# Output:
# Request 1-10: ALLOWED
# Request 11-12: REJECTED
# After 3s: 6 tokens available
# Request 1-6: ALLOWED
# Request 7-8: REJECTED
SOLUTION
}

# === Exercise 2: Sliding Window Rate Limiter ===
# Problem: Implement a sliding window counter that prevents the boundary
# burst problem of fixed windows.
exercise_2() {
    echo "=== Exercise 2: Sliding Window Rate Limiter ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import time
from collections import defaultdict


class SlidingWindowCounter:
    """Sliding window rate limiter using weighted counting.

    Instead of storing every request timestamp (memory-heavy), this uses
    two fixed windows and weights the previous window's count by how much
    of it overlaps with the current sliding window.

    Memory: O(1) per client (just two counters + timestamps).
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: dict[str, dict] = {}

    def _get_client(self, client_id: str) -> dict:
        if client_id not in self.clients:
            self.clients[client_id] = {
                "current_window_start": 0,
                "current_count": 0,
                "previous_count": 0,
            }
        return self.clients[client_id]

    def allow(self, client_id: str) -> tuple[bool, dict]:
        """Check if a request is allowed within the sliding window."""
        now = time.time()
        state = self._get_client(client_id)
        window_start = now - (now % self.window_seconds)

        # Rotate windows if we moved to a new fixed window
        if window_start != state["current_window_start"]:
            state["previous_count"] = state["current_count"]
            state["current_count"] = 0
            state["current_window_start"] = window_start

        # Weight of the previous window that overlaps with sliding window
        elapsed_in_current = now - window_start
        weight = 1.0 - (elapsed_in_current / self.window_seconds)

        # Estimated count in the sliding window
        estimated = state["previous_count"] * weight + state["current_count"]

        info = {
            "limit": self.max_requests,
            "remaining": max(0, int(self.max_requests - estimated - 1)),
            "reset": int(window_start + self.window_seconds),
        }

        if estimated < self.max_requests:
            state["current_count"] += 1
            return True, info
        else:
            info["retry_after"] = int(self.window_seconds - elapsed_in_current) + 1
            return False, info


# Demo
limiter = SlidingWindowCounter(max_requests=5, window_seconds=10)

for i in range(7):
    allowed, info = limiter.allow("user_1")
    status = "ALLOWED" if allowed else "REJECTED"
    print(f"Request {i+1}: {status} | remaining={info['remaining']}")
SOLUTION
}

# === Exercise 3: Rate Limit Response Headers ===
# Problem: Implement FastAPI middleware that adds standard rate limit headers
# to every response.
exercise_3() {
    echo "=== Exercise 3: Rate Limit Response Headers ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import time
from collections import defaultdict
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()

# Simple in-memory rate limiter
requests_per_client: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 100         # requests
RATE_WINDOW = 3600       # per hour


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next) -> Response:
    """Add rate limit headers to every response.

    Standard headers (IETF draft-ietf-httpapi-ratelimit-headers):
    - RateLimit-Limit:     Maximum requests per window
    - RateLimit-Remaining: Remaining requests in current window
    - RateLimit-Reset:     Seconds until the window resets
    - Retry-After:         Seconds to wait (only on 429 responses)
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_WINDOW

    # Clean old entries
    timestamps = requests_per_client[client_ip]
    requests_per_client[client_ip] = [t for t in timestamps if t > window_start]
    timestamps = requests_per_client[client_ip]

    remaining = max(0, RATE_LIMIT - len(timestamps))
    reset_seconds = int(RATE_WINDOW - (now - window_start))

    # Check rate limit
    if len(timestamps) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={
                "type": "https://api.example.com/problems/rate-limited",
                "title": "Rate Limit Exceeded",
                "status": 429,
                "detail": f"Limit of {RATE_LIMIT} requests per hour exceeded.",
            },
            headers={
                "RateLimit-Limit": str(RATE_LIMIT),
                "RateLimit-Remaining": "0",
                "RateLimit-Reset": str(reset_seconds),
                "Retry-After": str(reset_seconds),
            },
        )

    # Record request
    timestamps.append(now)

    # Process request and add headers to response
    response = await call_next(request)
    response.headers["RateLimit-Limit"] = str(RATE_LIMIT)
    response.headers["RateLimit-Remaining"] = str(remaining - 1)
    response.headers["RateLimit-Reset"] = str(reset_seconds)
    return response


@app.get("/api/v1/data")
def get_data():
    return {"data": "Check the response headers for rate limit info"}
SOLUTION
}

# === Exercise 4: Client Retry with Exponential Backoff ===
# Problem: Implement a robust HTTP client that respects rate limits and
# retries with exponential backoff and jitter.
exercise_4() {
    echo "=== Exercise 4: Client Retry with Exponential Backoff ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import time
import random
import httpx


class RateLimitAwareClient:
    """HTTP client that handles 429 responses with exponential backoff.

    Features:
    - Respects Retry-After header
    - Exponential backoff with jitter (prevents thundering herd)
    - Configurable max retries
    - Pre-emptive throttling based on RateLimit-Remaining header
    """

    def __init__(self, base_url: str, max_retries: int = 5):
        self.base_url = base_url
        self.max_retries = max_retries
        self.client = httpx.Client(base_url=base_url, timeout=30.0)

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make an HTTP request with automatic retry on rate limit."""
        for attempt in range(self.max_retries + 1):
            response = self.client.request(method, path, **kwargs)

            # Success — return immediately
            if response.status_code != 429:
                self._log_rate_limit(response)
                return response

            # Rate limited — calculate wait time
            if attempt == self.max_retries:
                return response  # Give up after max retries

            wait = self._calculate_wait(response, attempt)
            print(f"  Rate limited (attempt {attempt + 1}). "
                  f"Waiting {wait:.1f}s before retry...")
            time.sleep(wait)

        return response

    def _calculate_wait(self, response: httpx.Response, attempt: int) -> float:
        """Calculate wait time with exponential backoff + jitter.

        Priority:
        1. Use Retry-After header if present (server knows best)
        2. Fall back to exponential backoff: 2^attempt seconds
        3. Add random jitter to prevent thundering herd
        """
        # Check Retry-After header
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            base_wait = float(retry_after)
        else:
            # Exponential backoff: 1, 2, 4, 8, 16 seconds
            base_wait = min(2 ** attempt, 60)

        # Add jitter: random 0-25% of base wait
        jitter = random.uniform(0, base_wait * 0.25)
        return base_wait + jitter

    def _log_rate_limit(self, response: httpx.Response):
        """Monitor rate limit headers for preemptive throttling."""
        remaining = response.headers.get("RateLimit-Remaining")
        limit = response.headers.get("RateLimit-Limit")
        if remaining and limit:
            pct = int(remaining) / int(limit) * 100
            if pct < 10:
                print(f"  WARNING: Only {remaining}/{limit} requests remaining ({pct:.0f}%)")

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)


# Usage:
# client = RateLimitAwareClient("https://api.example.com")
# response = client.get("/api/v1/data")  # Automatically retries on 429
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 09: Rate Limiting and Throttling"
echo "================================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
