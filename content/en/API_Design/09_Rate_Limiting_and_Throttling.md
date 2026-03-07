# 09. Rate Limiting and Throttling

**Previous**: [Lesson 8](./08_Caching_Strategies.md) | **Next**: [API Documentation](./10_API_Documentation.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Compare token bucket, sliding window, and fixed window rate limiting algorithms and select the appropriate one for a given use case
- Implement per-user and per-IP rate limiting using Redis as a backing store
- Design proper 429 Too Many Requests responses with standard rate limit headers (X-RateLimit-*)
- Configure tiered rate limits for different API consumer classes (free, pro, enterprise)
- Integrate rate limiting into API gateways and middleware layers
- Apply throttling strategies that degrade gracefully under load

---

## Table of Contents

1. [Why Rate Limiting Matters](#1-why-rate-limiting-matters)
2. [Rate Limiting Algorithms](#2-rate-limiting-algorithms)
3. [Rate Limit Headers](#3-rate-limit-headers)
4. [Per-User and Per-IP Limits](#4-per-user-and-per-ip-limits)
5. [Redis-Backed Rate Limiter](#5-redis-backed-rate-limiter)
6. [Tiered Rate Limiting](#6-tiered-rate-limiting)
7. [Rate Limiting in API Gateways](#7-rate-limiting-in-api-gateways)
8. [Throttling and Graceful Degradation](#8-throttling-and-graceful-degradation)
9. [Exercises](#9-exercises)
10. [References](#10-references)

---

## 1. Why Rate Limiting Matters

Rate limiting controls the number of requests a client can make to an API within a given time window. Without it, a single client can overwhelm the server, degrade performance for everyone, and increase infrastructure costs.

### Goals of Rate Limiting

- **Availability**: Prevent a single client from monopolizing server resources
- **Fairness**: Ensure equitable access across all API consumers
- **Cost control**: Limit expensive operations (database queries, external API calls)
- **Security**: Mitigate brute-force attacks, credential stuffing, and DDoS attempts
- **Compliance**: Enforce usage limits tied to pricing tiers or SLAs

### Rate Limiting vs. Throttling

| Concept | Behavior | Response |
|---------|----------|----------|
| Rate limiting | Rejects requests that exceed the limit | 429 Too Many Requests |
| Throttling | Slows down or queues excess requests | Delayed response (200 OK, but slower) |

In practice, the terms are often used interchangeably, but the distinction matters when designing user-facing behavior.

---

## 2. Rate Limiting Algorithms

### Fixed Window

The simplest algorithm. Divide time into fixed intervals (e.g., 1-minute windows). Count requests per window and reject when the count exceeds the limit.

```
Window: [12:00:00 - 12:01:00]  limit=100
Requests: ████████████ (87)    → allowed
Window: [12:01:00 - 12:02:00]  limit=100
Requests: ████████████████████████████ (112) → last 12 rejected
```

```python
import time
import redis

class FixedWindowLimiter:
    """Fixed window rate limiter using Redis.

    Divides time into fixed intervals and counts requests per window.
    Simple but susceptible to burst traffic at window boundaries.
    """

    def __init__(self, redis_client: redis.Redis, limit: int, window_seconds: int):
        self.redis = redis_client
        self.limit = limit
        self.window = window_seconds

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        # Current window identifier: floor(timestamp / window_size)
        current_window = int(time.time() // self.window)
        redis_key = f"ratelimit:{key}:{current_window}"

        # Increment counter atomically
        current_count = self.redis.incr(redis_key)

        # Set expiry on first request in this window
        if current_count == 1:
            self.redis.expire(redis_key, self.window)

        remaining = max(0, self.limit - current_count)
        reset_at = (current_window + 1) * self.window

        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_at),
        }

        return current_count <= self.limit, headers
```

**Pros**: Simple to implement, low memory usage.
**Cons**: Burst problem at window boundaries --- a client can send `2 * limit` requests across two adjacent windows (e.g., 100 requests at 12:00:59 and 100 at 12:01:01).

### Sliding Window Log

Track the timestamp of every request. To check if a new request is allowed, count how many timestamps fall within the past N seconds.

```python
class SlidingWindowLogLimiter:
    """Sliding window log rate limiter using Redis sorted sets.

    Stores each request timestamp and counts requests within the
    sliding window. Precise but uses more memory per client.
    """

    def __init__(self, redis_client: redis.Redis, limit: int, window_seconds: int):
        self.redis = redis_client
        self.limit = limit
        self.window = window_seconds

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        now = time.time()
        window_start = now - self.window
        redis_key = f"ratelimit:swl:{key}"

        pipe = self.redis.pipeline()
        # Remove timestamps outside the current window
        pipe.zremrangebyscore(redis_key, 0, window_start)
        # Add current request timestamp
        pipe.zadd(redis_key, {str(now): now})
        # Count requests in window
        pipe.zcard(redis_key)
        # Set expiry to auto-cleanup
        pipe.expire(redis_key, self.window)
        results = pipe.execute()

        request_count = results[2]
        remaining = max(0, self.limit - request_count)

        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(now + self.window)),
        }

        return request_count <= self.limit, headers
```

**Pros**: No boundary burst problem, precise counting.
**Cons**: Higher memory usage (stores every timestamp), more expensive operations.

### Sliding Window Counter

A hybrid that combines fixed window efficiency with sliding window accuracy. It uses two adjacent fixed windows and weights the counts based on the elapsed fraction.

```python
class SlidingWindowCounterLimiter:
    """Sliding window counter: a memory-efficient approximation.

    Uses the current and previous fixed windows with weighted counting
    to approximate a true sliding window. Good balance of accuracy
    and performance.
    """

    def __init__(self, redis_client: redis.Redis, limit: int, window_seconds: int):
        self.redis = redis_client
        self.limit = limit
        self.window = window_seconds

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        now = time.time()
        current_window = int(now // self.window)
        previous_window = current_window - 1

        # How far into the current window we are (0.0 to 1.0)
        elapsed_fraction = (now % self.window) / self.window

        current_key = f"ratelimit:swc:{key}:{current_window}"
        previous_key = f"ratelimit:swc:{key}:{previous_window}"

        pipe = self.redis.pipeline()
        pipe.get(current_key)
        pipe.get(previous_key)
        results = pipe.execute()

        current_count = int(results[0] or 0)
        previous_count = int(results[1] or 0)

        # Weighted sum: full current window + remaining portion of previous
        estimated_count = current_count + previous_count * (1 - elapsed_fraction)

        if estimated_count < self.limit:
            self.redis.incr(current_key)
            self.redis.expire(current_key, self.window * 2)

        remaining = max(0, int(self.limit - estimated_count))
        reset_at = (current_window + 1) * self.window

        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(reset_at)),
        }

        return estimated_count < self.limit, headers
```

### Token Bucket

Tokens are added to a bucket at a fixed rate. Each request consumes one token. If the bucket is empty, the request is rejected. The bucket has a maximum capacity, allowing short bursts up to that capacity.

```
Bucket capacity: 10 tokens
Refill rate: 1 token/second

Time 0s:  [##########] 10 tokens → request → [#########] 9 tokens
Time 0s:  [#########]  9 tokens → request → [########]  8 tokens
Time 0s:  [########]   8 tokens → 5 burst requests → [###] 3 tokens
Time 3s:  [######]     6 tokens (refilled 3 tokens)
```

```python
class TokenBucketLimiter:
    """Token bucket rate limiter.

    Allows short bursts (up to bucket capacity) while enforcing
    a sustained average rate. Widely used in production systems
    (e.g., AWS API Gateway, Stripe).
    """

    def __init__(
        self, redis_client: redis.Redis, capacity: int, refill_rate: float
    ):
        self.redis = redis_client
        self.capacity = capacity        # max tokens (burst size)
        self.refill_rate = refill_rate  # tokens per second

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        redis_key = f"ratelimit:tb:{key}"
        now = time.time()

        # Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now

        -- Calculate tokens to add since last refill
        local elapsed = now - last_refill
        tokens = math.min(capacity, tokens + elapsed * refill_rate)

        local allowed = 0
        if tokens >= 1 then
            tokens = tokens - 1
            allowed = 1
        end

        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) * 2)

        return {allowed, math.floor(tokens)}
        """

        result = self.redis.eval(lua_script, 1, redis_key,
                                  self.capacity, self.refill_rate, now)
        allowed = bool(result[0])
        remaining = int(result[1])

        headers = {
            "X-RateLimit-Limit": str(self.capacity),
            "X-RateLimit-Remaining": str(remaining),
        }

        return allowed, headers
```

**Pros**: Allows controlled bursts, smooth rate enforcement, widely understood.
**Cons**: Slightly more complex to implement, requires atomic operations.

### Algorithm Comparison

| Algorithm | Burst Handling | Memory | Accuracy | Complexity |
|-----------|---------------|--------|----------|------------|
| Fixed window | Allows 2x burst at boundaries | Low | Low | Simple |
| Sliding window log | No bursts | High | Exact | Moderate |
| Sliding window counter | Approximated | Low | Good | Moderate |
| Token bucket | Controlled bursts | Low | Good | Moderate |

---

## 3. Rate Limit Headers

Standard headers communicate rate limit status to API consumers. While not yet an official RFC, the `X-RateLimit-*` headers are a widely adopted convention. The IETF draft `RateLimit` header fields (draft-ietf-httpapi-ratelimit-headers) standardize these.

### Standard Response Headers

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100          # Maximum requests allowed per window
X-RateLimit-Remaining: 57       # Requests remaining in current window
X-RateLimit-Reset: 1709827200   # Unix timestamp when the window resets
```

### 429 Too Many Requests Response

When a client exceeds the limit, respond with a `429` status code and include a `Retry-After` header:

```
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1709827200
Retry-After: 30

{
    "type": "https://api.example.com/errors/rate-limit-exceeded",
    "title": "Rate Limit Exceeded",
    "status": 429,
    "detail": "You have exceeded the rate limit of 100 requests per minute. Please retry after 30 seconds.",
    "retry_after": 30
}
```

### FastAPI Middleware for Rate Limit Headers

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting and inject standard headers into every response."""
    client_key = get_client_identifier(request)
    limiter = get_limiter_for_route(request.url.path)

    allowed, headers = limiter.is_allowed(client_key)

    if not allowed:
        reset_time = int(headers.get("X-RateLimit-Reset", 0))
        retry_after = max(1, reset_time - int(time.time()))
        headers["Retry-After"] = str(retry_after)

        return JSONResponse(
            status_code=429,
            content={
                "type": "https://api.example.com/errors/rate-limit-exceeded",
                "title": "Rate Limit Exceeded",
                "status": 429,
                "detail": f"Rate limit exceeded. Retry after {retry_after} seconds.",
                "retry_after": retry_after,
            },
            headers=headers,
        )

    response = await call_next(request)

    # Attach rate limit headers to all successful responses
    for header_name, header_value in headers.items():
        response.headers[header_name] = header_value

    return response
```

---

## 4. Per-User and Per-IP Limits

Different identification strategies serve different purposes.

### Identification Strategies

| Strategy | Key Source | Use Case | Limitations |
|----------|-----------|----------|-------------|
| IP address | `X-Forwarded-For` or client IP | Unauthenticated endpoints | Shared IPs (NAT, proxies) |
| API key | `X-API-Key` header | Machine-to-machine APIs | Key sharing |
| User ID | JWT `sub` claim | Authenticated endpoints | Requires auth |
| Composite | IP + User-Agent + route | Fine-grained control | Complex key management |

### Extracting the Client Identifier

```python
from fastapi import Request


def get_client_ip(request: Request) -> str:
    """Extract the real client IP, accounting for reverse proxies.

    X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2.
    The leftmost IP is the original client (if proxies are trusted).
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(",")[0].strip()

    x_real_ip = request.headers.get("X-Real-IP")
    if x_real_ip:
        return x_real_ip

    return request.client.host if request.client else "unknown"


def get_client_identifier(request: Request) -> str:
    """Build a rate limit key based on authentication status.

    Authenticated users are identified by user ID (more accurate).
    Anonymous users fall back to IP address.
    """
    # Check for authenticated user (set by auth middleware)
    user = getattr(request.state, "user", None)
    if user:
        return f"user:{user.id}"

    # Check for API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key[:16]}"  # Use prefix for key privacy

    # Fall back to IP
    return f"ip:{get_client_ip(request)}"
```

### Per-Route Rate Limits

Different endpoints have different cost profiles. A search endpoint that queries a full-text index is more expensive than a simple GET by ID.

```python
from functools import wraps
from fastapi import Request, HTTPException


# Route-specific rate limits
ROUTE_LIMITS = {
    "/api/search": {"limit": 20, "window": 60},       # 20/min (expensive)
    "/api/users": {"limit": 100, "window": 60},        # 100/min (standard)
    "/api/health": {"limit": 1000, "window": 60},      # 1000/min (cheap)
    "default": {"limit": 60, "window": 60},             # 60/min fallback
}


def rate_limit(limit: int, window: int = 60):
    """Decorator for per-route rate limiting."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_key = get_client_identifier(request)
            redis_key = f"rl:{request.url.path}:{client_key}"

            current = redis_client.incr(redis_key)
            if current == 1:
                redis_client.expire(redis_key, window)

            if current > limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {limit} requests per {window}s",
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "Retry-After": str(window),
                    },
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


@app.get("/api/search")
@rate_limit(limit=20, window=60)
async def search(request: Request, q: str):
    return await perform_search(q)
```

---

## 5. Redis-Backed Rate Limiter

Redis is the standard backing store for production rate limiters due to its atomic operations, sub-millisecond latency, and built-in key expiration.

### Production-Ready Implementation

```python
import time
import redis.asyncio as redis
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    limit: int
    remaining: int
    reset_at: int
    retry_after: int | None = None

    @property
    def headers(self) -> dict[str, str]:
        h = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_at),
        }
        if self.retry_after is not None:
            h["Retry-After"] = str(self.retry_after)
        return h


class AsyncSlidingWindowLimiter:
    """Async sliding window counter rate limiter with Redis.

    Uses the sliding window counter algorithm for a good balance
    between accuracy and memory usage. All operations are atomic
    via Redis Lua scripting.
    """

    # Lua script ensures atomicity — no race conditions between
    # reading the count and incrementing it
    LUA_SCRIPT = """
    local current_key = KEYS[1]
    local previous_key = KEYS[2]
    local limit = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    local current_window = math.floor(now / window)
    local elapsed_fraction = (now % window) / window

    local current_count = tonumber(redis.call('GET', current_key) or '0')
    local previous_count = tonumber(redis.call('GET', previous_key) or '0')

    local estimated = current_count + previous_count * (1 - elapsed_fraction)

    if estimated < limit then
        redis.call('INCR', current_key)
        redis.call('EXPIRE', current_key, window * 2)
        current_count = current_count + 1
        estimated = current_count + previous_count * (1 - elapsed_fraction)
        return {1, math.floor(limit - estimated), (current_window + 1) * window}
    end

    return {0, 0, (current_window + 1) * window}
    """

    def __init__(self, redis_client: redis.Redis, limit: int, window: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window = window
        self._script_sha = None

    async def _ensure_script(self):
        if self._script_sha is None:
            self._script_sha = await self.redis.script_load(self.LUA_SCRIPT)

    async def check(self, key: str) -> RateLimitResult:
        await self._ensure_script()

        now = time.time()
        current_window = int(now // self.window)

        current_key = f"rl:{key}:{current_window}"
        previous_key = f"rl:{key}:{current_window - 1}"

        result = await self.redis.evalsha(
            self._script_sha, 2,
            current_key, previous_key,
            self.limit, self.window, now
        )

        allowed = bool(result[0])
        remaining = max(0, int(result[1]))
        reset_at = int(result[2])

        retry_after = None
        if not allowed:
            retry_after = max(1, reset_at - int(now))

        return RateLimitResult(
            allowed=allowed,
            limit=self.limit,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )
```

### Using the Limiter as FastAPI Middleware

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import redis.asyncio as redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage Redis connection lifecycle."""
    app.state.redis = redis.Redis(host="localhost", port=6379, db=0)
    app.state.limiter = AsyncSlidingWindowLimiter(
        app.state.redis, limit=100, window=60
    )
    yield
    await app.state.redis.close()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health checks
    if request.url.path == "/health":
        return await call_next(request)

    client_key = get_client_identifier(request)
    result = await request.app.state.limiter.check(client_key)

    if not result.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "type": "https://api.example.com/errors/rate-limit",
                "title": "Rate Limit Exceeded",
                "status": 429,
                "detail": f"Retry after {result.retry_after} seconds.",
            },
            headers=result.headers,
        )

    response = await call_next(request)
    for name, value in result.headers.items():
        response.headers[name] = value
    return response
```

---

## 6. Tiered Rate Limiting

Different API consumers need different limits. Free tier users get basic access; paying customers get higher throughput.

```python
from enum import Enum
from dataclasses import dataclass


class Tier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierConfig:
    requests_per_minute: int
    requests_per_day: int
    burst_size: int


TIER_LIMITS: dict[Tier, TierConfig] = {
    Tier.FREE: TierConfig(
        requests_per_minute=60,
        requests_per_day=1_000,
        burst_size=10,
    ),
    Tier.PRO: TierConfig(
        requests_per_minute=600,
        requests_per_day=50_000,
        burst_size=100,
    ),
    Tier.ENTERPRISE: TierConfig(
        requests_per_minute=6_000,
        requests_per_day=1_000_000,
        burst_size=1_000,
    ),
}


class TieredRateLimiter:
    """Rate limiter that applies different limits based on user tier.

    Checks both per-minute and per-day limits. Returns the most
    restrictive result.
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    async def check(self, user_id: str, tier: Tier) -> RateLimitResult:
        config = TIER_LIMITS[tier]

        # Check per-minute limit
        minute_limiter = AsyncSlidingWindowLimiter(
            self.redis, limit=config.requests_per_minute, window=60
        )
        minute_result = await minute_limiter.check(f"{user_id}:min")

        # Check per-day limit
        day_limiter = AsyncSlidingWindowLimiter(
            self.redis, limit=config.requests_per_day, window=86400
        )
        day_result = await day_limiter.check(f"{user_id}:day")

        # Return the most restrictive result
        if not minute_result.allowed:
            return minute_result
        if not day_result.allowed:
            return day_result

        # Both allowed — return minute-level headers (more relevant)
        return minute_result


# Usage in a dependency
async def check_rate_limit(request: Request):
    user = request.state.user
    tier = Tier(user.subscription_tier)
    limiter = TieredRateLimiter(request.app.state.redis)
    result = await limiter.check(str(user.id), tier)

    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers=result.headers,
        )
    return result
```

---

## 7. Rate Limiting in API Gateways

In a microservices architecture, rate limiting is best applied at the API gateway level rather than in each individual service. This centralizes policy enforcement and avoids duplicating logic.

### Gateway-Level vs. Service-Level

```
                          Gateway-Level Rate Limiting
Client → [API Gateway (rate limit)] → Service A
                                     → Service B
                                     → Service C

                         Service-Level Rate Limiting
Client → [API Gateway] → Service A (rate limit)
                        → Service B (rate limit)
                        → Service C (rate limit)
```

| Approach | Pros | Cons |
|----------|------|------|
| Gateway-level | Centralized policy, single enforcement point | Cannot differentiate per-service costs |
| Service-level | Fine-grained control per service | Duplicated logic, harder to manage globally |
| Hybrid | Best of both: global limits at gateway, specific limits in services | More complex to configure |

### Kong Rate Limiting Plugin

```yaml
# kong.yml — declarative configuration
plugins:
  - name: rate-limiting
    config:
      minute: 100
      hour: 5000
      policy: redis
      redis_host: redis
      redis_port: 6379

# Per-consumer overrides
consumers:
  - username: premium-client
    plugins:
      - name: rate-limiting
        config:
          minute: 1000
          hour: 50000
```

### AWS API Gateway Throttling

```python
import boto3

client = boto3.client("apigateway")

# Set account-level throttle
client.update_account(
    patchOperations=[
        {"op": "replace", "path": "/throttle/rateLimit", "value": "1000"},
        {"op": "replace", "path": "/throttle/burstLimit", "value": "2000"},
    ]
)

# Set per-method throttle via usage plan
client.create_usage_plan(
    name="ProPlan",
    throttle={"rateLimit": 500.0, "burstLimit": 1000},
    quota={"limit": 100000, "period": "MONTH"},
    apiStages=[{"apiId": "abc123", "stage": "prod"}],
)
```

---

## 8. Throttling and Graceful Degradation

Instead of hard-rejecting requests, throttling slows responses to manage load. This provides a better user experience under pressure.

### Backpressure Pattern

```python
import asyncio


class BackpressureThrottler:
    """Applies progressive delay instead of hard rejection.

    As the client approaches their rate limit, responses are
    progressively delayed. This signals to well-behaved clients
    that they should slow down, while still serving their requests.
    """

    def __init__(self, redis_client, limit: int, window: int = 60):
        self.redis = redis_client
        self.limit = limit
        self.window = window

    async def apply(self, key: str) -> float:
        """Returns the delay in seconds to apply. 0.0 means no delay."""
        now = time.time()
        current_window = int(now // self.window)
        redis_key = f"throttle:{key}:{current_window}"

        count = await self.redis.incr(redis_key)
        if count == 1:
            await self.redis.expire(redis_key, self.window)

        usage_ratio = count / self.limit

        if usage_ratio < 0.5:
            return 0.0          # No delay under 50% usage
        elif usage_ratio < 0.8:
            return 0.1          # 100ms delay at 50-80%
        elif usage_ratio < 1.0:
            return 0.5          # 500ms delay at 80-100%
        else:
            return -1.0         # Signal to reject (over 100%)


@app.middleware("http")
async def throttle_middleware(request: Request, call_next):
    client_key = get_client_identifier(request)
    throttler = BackpressureThrottler(request.app.state.redis, limit=100)

    delay = await throttler.apply(client_key)

    if delay < 0:
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})

    if delay > 0:
        await asyncio.sleep(delay)

    return await call_next(request)
```

### Load Shedding

Under extreme load, it is better to reject some requests quickly than to attempt serving all of them slowly (which can lead to cascading failures).

```python
import random


class LoadShedder:
    """Probabilistic load shedding based on current server load.

    When the server is overloaded, randomly reject a percentage of
    requests with 503 Service Unavailable. This prevents cascading
    failures by keeping the server responsive for the requests it
    does accept.
    """

    def __init__(self, max_concurrent: int = 1000):
        self.max_concurrent = max_concurrent
        self.current_requests = 0

    def should_shed(self) -> bool:
        load_factor = self.current_requests / self.max_concurrent

        if load_factor < 0.7:
            return False    # Under 70%: accept all
        elif load_factor < 0.9:
            # 70-90%: reject proportionally
            reject_probability = (load_factor - 0.7) / 0.2
            return random.random() < reject_probability
        else:
            return True     # Over 90%: reject all new requests
```

---

## 9. Exercises

### Exercise 1: Implement a Leaky Bucket

Implement a leaky bucket rate limiter. Unlike the token bucket, the leaky bucket processes requests at a constant rate. If the bucket is full, new requests are rejected. Use Redis for state management. Include proper rate limit headers in responses and write tests that verify:

- Requests within the limit are allowed
- Burst requests fill the bucket
- The bucket drains at a constant rate
- Requests are rejected when the bucket is full

### Exercise 2: Multi-Key Rate Limiting

Design a rate limiting system that enforces limits across multiple dimensions simultaneously:

- Per-IP: 60 requests/minute (unauthenticated)
- Per-user: 200 requests/minute
- Per-endpoint: varies by route (e.g., `/search` = 20/min, `/users` = 100/min)
- Global: 10,000 requests/minute across all clients

All four limits must be checked atomically. If any limit is exceeded, return 429 with headers indicating which limit was hit.

### Exercise 3: Rate Limit Dashboard

Build a FastAPI endpoint `GET /admin/rate-limits` that returns the current rate limit status for all active clients. For each client, show:

- Client identifier (IP or user ID)
- Current request count
- Limit and remaining quota
- Time until window reset

Use Redis `SCAN` to iterate over rate limit keys efficiently without blocking the server.

### Exercise 4: Client-Side Rate Limiting

Write a Python API client class that respects the server's rate limit headers. The client should:

- Parse `X-RateLimit-Remaining` and `X-RateLimit-Reset` from responses
- Automatically slow down when remaining quota is low
- Sleep and retry when receiving a `429` response (respecting `Retry-After`)
- Implement exponential backoff for consecutive 429 responses

### Exercise 5: Distributed Rate Limiting

Design a rate limiting solution for a multi-region API deployment where requests from the same user may hit different data centers. Consider:

- Eventual consistency between regions
- Local vs. global rate limit enforcement
- What happens when Redis is temporarily unavailable (fail-open vs. fail-closed)
- How to handle clock skew between regions

Write pseudocode or a design document for your approach.

---

## 10. References

- [RFC 6585: Additional HTTP Status Codes (429)](https://tools.ietf.org/html/rfc6585)
- [IETF Draft: RateLimit Header Fields](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/)
- [Stripe Rate Limiting](https://stripe.com/docs/rate-limits)
- [Cloudflare Rate Limiting](https://developers.cloudflare.com/waf/rate-limiting-rules/)
- [Redis Rate Limiting Patterns](https://redis.io/glossary/rate-limiting/)
- [Kong Rate Limiting Plugin](https://docs.konghq.com/hub/kong-inc/rate-limiting/)
- [slowapi - Rate Limiting for FastAPI](https://github.com/laurentS/slowapi)

---

**Previous**: [Lesson 8](./08_Caching_Strategies.md) | [Overview](./00_Overview.md) | **Next**: [API Documentation](./10_API_Documentation.md)

**License**: CC BY-NC 4.0
