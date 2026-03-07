#!/usr/bin/env python3
"""Example: Rate Limiting

Demonstrates two rate limiting algorithms as FastAPI middleware:
1. Token Bucket — smooth traffic, allows bursts
2. Sliding Window — precise per-window counting

Also shows: rate limit response headers, per-client tracking, and retry logic.

Related lesson: 09_Rate_Limiting_and_Throttling.md

Run:
    pip install "fastapi[standard]"
    uvicorn 05_rate_limiting:app --reload --port 8000

Test:
    # Rapid requests to trigger rate limiting
    for i in $(seq 1 15); do http GET :8000/api/v1/bucket/data; done
    for i in $(seq 1 15); do http GET :8000/api/v1/window/data; done
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

# =============================================================================
# 1. TOKEN BUCKET ALGORITHM
# =============================================================================
# How it works:
# - Each client has a "bucket" that holds up to `capacity` tokens.
# - Tokens are consumed (1 per request). If the bucket is empty, reject.
# - Tokens are refilled at a constant `rate` (tokens per second).
# - Allows bursts up to `capacity`, then throttles to `rate`.
#
# Analogy: A bucket with a hole — water (tokens) drains out at a fixed rate,
# and you can pour in at most `capacity` at once.
#
# Used by: AWS API Gateway, Stripe, many CDNs.


class TokenBucket:
    """Per-client token bucket rate limiter."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum tokens (burst size).
            refill_rate: Tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: dict[str, dict] = {}

    def _get_bucket(self, client_id: str) -> dict:
        """Get or initialize a bucket for a client."""
        now = time.monotonic()
        if client_id not in self.buckets:
            self.buckets[client_id] = {
                "tokens": self.capacity,
                "last_refill": now,
            }
        return self.buckets[client_id]

    def consume(self, client_id: str) -> tuple[bool, dict]:
        """Attempt to consume a token.

        Returns:
            (allowed, info) where info contains remaining tokens and reset time.
        """
        bucket = self._get_bucket(client_id)
        now = time.monotonic()

        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_refill"]
        refill = elapsed * self.refill_rate
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + refill)
        bucket["last_refill"] = now

        info = {
            "limit": self.capacity,
            "remaining": max(0, int(bucket["tokens"]) - 1),
            "reset": int(time.time()) + int((self.capacity - bucket["tokens"]) / self.refill_rate),
        }

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            info["remaining"] = int(bucket["tokens"])
            return True, info
        else:
            info["remaining"] = 0
            retry_after = (1 - bucket["tokens"]) / self.refill_rate
            info["retry_after"] = max(1, int(retry_after))
            return False, info


# =============================================================================
# 2. SLIDING WINDOW ALGORITHM
# =============================================================================
# How it works:
# - Track timestamps of all requests within the window.
# - Count requests in the last `window_seconds`.
# - If count >= limit, reject.
#
# Pros: More precise than fixed windows (no boundary burst problem).
# Cons: Higher memory usage (stores all timestamps).
#
# Used by: Redis-based rate limiters, Cloudflare.


class SlidingWindow:
    """Per-client sliding window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Window duration in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def consume(self, client_id: str) -> tuple[bool, dict]:
        """Check if a request is allowed within the sliding window."""
        now = time.monotonic()
        window_start = now - self.window_seconds

        # Remove expired timestamps
        timestamps = self.requests[client_id]
        self.requests[client_id] = [t for t in timestamps if t > window_start]
        timestamps = self.requests[client_id]

        count = len(timestamps)
        info = {
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - count - 1),
            "reset": int(time.time()) + self.window_seconds,
            "window": self.window_seconds,
        }

        if count < self.max_requests:
            timestamps.append(now)
            info["remaining"] = self.max_requests - len(timestamps)
            return True, info
        else:
            # Calculate when the oldest request in the window expires
            oldest = timestamps[0]
            retry_after = int(oldest - window_start) + 1
            info["retry_after"] = max(1, retry_after)
            info["remaining"] = 0
            return False, info


# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(title="Rate Limiting API", version="1.0.0")

# Create rate limiters
# Token bucket: 10 burst capacity, refills at 2 tokens/second
token_bucket = TokenBucket(capacity=10, refill_rate=2.0)

# Sliding window: 10 requests per 60-second window
sliding_window = SlidingWindow(max_requests=10, window_seconds=60)


# =============================================================================
# HELPER — Extract client identifier
# =============================================================================

def get_client_id(request: Request) -> str:
    """Identify the client for rate limiting.

    In production, use a combination of:
    - API key (best: identifies the actual consumer)
    - User ID from JWT (good: per-user limits)
    - IP address (fallback: can be shared by NAT users)
    """
    # Check for API key first, fall back to IP
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return f"ip:{request.client.host}" if request.client else "ip:unknown"


# =============================================================================
# RATE LIMIT HEADERS
# =============================================================================
# Standard headers inform clients of their rate limit status.
# These are described in IETF draft-ietf-httpapi-ratelimit-headers.

def add_rate_limit_headers(response: Response, info: dict) -> None:
    """Add standard rate limit headers to the response.

    Headers:
    - X-RateLimit-Limit: Maximum requests allowed
    - X-RateLimit-Remaining: Requests left in current window
    - X-RateLimit-Reset: Unix timestamp when the limit resets
    - Retry-After: Seconds to wait before retrying (only on 429)
    """
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(info["reset"])


def rate_limit_response(info: dict) -> JSONResponse:
    """Build a 429 Too Many Requests response with proper headers."""
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "type": "https://api.example.com/problems/rate-limited",
            "title": "Rate Limit Exceeded",
            "status": 429,
            "detail": f"Rate limit exceeded. Retry after {info.get('retry_after', 1)} seconds.",
        },
        media_type="application/problem+json",
    )
    add_rate_limit_headers(response, info)
    response.headers["Retry-After"] = str(info.get("retry_after", 1))
    return response


# =============================================================================
# MIDDLEWARE — Apply rate limiting to all requests
# =============================================================================

def create_rate_limit_middleware(
    limiter: TokenBucket | SlidingWindow,
    path_prefix: str,
) -> Callable:
    """Factory for rate-limiting middleware targeting specific path prefixes."""

    async def middleware(request: Request, call_next: Callable) -> Response:
        # Only rate-limit matching paths
        if not request.url.path.startswith(path_prefix):
            return await call_next(request)

        client_id = get_client_id(request)
        allowed, info = limiter.consume(client_id)

        if not allowed:
            return rate_limit_response(info)

        response = await call_next(request)
        add_rate_limit_headers(response, info)
        return response

    return middleware


# Register middleware for each rate limiter (different path prefixes)
app.middleware("http")(create_rate_limit_middleware(token_bucket, "/api/v1/bucket"))
app.middleware("http")(create_rate_limit_middleware(sliding_window, "/api/v1/window"))


# =============================================================================
# ROUTES — Token Bucket endpoints
# =============================================================================

@app.get("/api/v1/bucket/data", tags=["Token Bucket"])
def bucket_data():
    """Endpoint protected by token bucket rate limiter.

    Allows bursts of up to 10 requests, then refills at 2 req/sec.
    Check X-RateLimit-Remaining header to see your remaining quota.
    """
    return {"message": "Data from token bucket endpoint", "timestamp": time.time()}


@app.get("/api/v1/bucket/info", tags=["Token Bucket"])
def bucket_info():
    """Get current token bucket configuration."""
    return {
        "algorithm": "token_bucket",
        "capacity": token_bucket.capacity,
        "refill_rate": f"{token_bucket.refill_rate} tokens/sec",
        "description": "Allows bursts up to capacity, then throttles to refill rate",
    }


# =============================================================================
# ROUTES — Sliding Window endpoints
# =============================================================================

@app.get("/api/v1/window/data", tags=["Sliding Window"])
def window_data():
    """Endpoint protected by sliding window rate limiter.

    Allows 10 requests per 60-second window. Unlike fixed windows, the
    sliding window prevents boundary bursts (e.g., 10 at :59, 10 at :01).
    """
    return {"message": "Data from sliding window endpoint", "timestamp": time.time()}


@app.get("/api/v1/window/info", tags=["Sliding Window"])
def window_info():
    """Get current sliding window configuration."""
    return {
        "algorithm": "sliding_window",
        "max_requests": sliding_window.max_requests,
        "window_seconds": sliding_window.window_seconds,
        "description": "Precise per-window counting with no boundary burst problem",
    }


# =============================================================================
# CLIENT-SIDE RETRY LOGIC EXAMPLE
# =============================================================================

@app.get("/api/v1/retry-example", tags=["Info"])
def retry_example():
    """Shows the recommended client-side retry logic as pseudocode."""
    return {
        "description": "Client-side retry with exponential backoff",
        "pseudocode": [
            "max_retries = 3",
            "for attempt in range(max_retries):",
            "    response = requests.get(url, headers=headers)",
            "    if response.status_code != 429:",
            "        break",
            "    retry_after = int(response.headers.get('Retry-After', 1))",
            "    backoff = min(retry_after * (2 ** attempt), 60)",
            "    time.sleep(backoff + random.uniform(0, 1))  # jitter",
        ],
        "key_points": [
            "Always respect the Retry-After header",
            "Use exponential backoff: 1s, 2s, 4s, 8s, ...",
            "Add random jitter to prevent thundering herd",
            "Set a maximum retry limit to avoid infinite loops",
        ],
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("05_rate_limiting:app", host="127.0.0.1", port=8000, reload=True)
