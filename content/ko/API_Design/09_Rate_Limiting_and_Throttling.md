# 09. Rate Limiting과 Throttling

**이전**: [Lesson 8](./08_Caching_Strategies.md) | **다음**: [API 문서화](./10_API_Documentation.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- Token bucket, sliding window, fixed window rate limiting 알고리즘을 비교하고 주어진 사용 사례에 적합한 알고리즘 선택하기
- Redis를 백킹 스토어로 사용하여 사용자별 및 IP별 rate limiting 구현하기
- 표준 rate limit 헤더(X-RateLimit-*)를 포함한 적절한 429 Too Many Requests 응답 설계하기
- 다양한 API 소비자 등급(free, pro, enterprise)에 대한 계층형 rate limit 설정하기
- API 게이트웨이 및 미들웨어 계층에 rate limiting 통합하기
- 부하 상황에서 점진적으로 성능을 낮추는 throttling 전략 적용하기

---

## 목차

1. [Rate Limiting이 중요한 이유](#1-rate-limiting이-중요한-이유)
2. [Rate Limiting 알고리즘](#2-rate-limiting-알고리즘)
3. [Rate Limit 헤더](#3-rate-limit-헤더)
4. [사용자별 및 IP별 제한](#4-사용자별-및-ip별-제한)
5. [Redis 기반 Rate Limiter](#5-redis-기반-rate-limiter)
6. [계층형 Rate Limiting](#6-계층형-rate-limiting)
7. [API 게이트웨이에서의 Rate Limiting](#7-api-게이트웨이에서의-rate-limiting)
8. [Throttling과 점진적 성능 저하](#8-throttling과-점진적-성능-저하)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. Rate Limiting이 중요한 이유

Rate limiting은 클라이언트가 주어진 시간 윈도우 내에 API에 보낼 수 있는 요청 수를 제어합니다. 이것이 없으면 단일 클라이언트가 서버를 압도하고, 모든 사용자의 성능을 저하시키며, 인프라 비용을 증가시킬 수 있습니다.

### Rate Limiting의 목적

- **가용성**: 단일 클라이언트가 서버 리소스를 독점하는 것을 방지
- **공정성**: 모든 API 소비자에게 공평한 접근 보장
- **비용 제어**: 비용이 많이 드는 작업(데이터베이스 쿼리, 외부 API 호출) 제한
- **보안**: 브루트포스 공격, 크리덴셜 스터핑, DDoS 시도 완화
- **규정 준수**: 가격 등급 또는 SLA에 연결된 사용 제한 시행

### Rate Limiting vs Throttling

| 개념 | 동작 | 응답 |
|---------|----------|----------|
| Rate limiting | 제한을 초과하는 요청을 거부 | 429 Too Many Requests |
| Throttling | 초과 요청을 느리게 하거나 큐에 넣음 | 지연된 응답 (200 OK, 단 더 느림) |

실무에서는 두 용어가 혼용되는 경우가 많지만, 사용자 대면 동작을 설계할 때 이 구분은 중요합니다.

---

## 2. Rate Limiting 알고리즘

### Fixed Window

가장 단순한 알고리즘입니다. 시간을 고정 간격(예: 1분 윈도우)으로 나눕니다. 윈도우당 요청 수를 세고 제한을 초과하면 거부합니다.

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

**장점**: 구현이 간단하고, 메모리 사용량이 적습니다.
**단점**: 윈도우 경계에서 버스트 문제 발생 --- 클라이언트가 인접한 두 윈도우에 걸쳐 `2 * limit` 요청을 보낼 수 있습니다(예: 12:00:59에 100개, 12:01:01에 100개).

### Sliding Window Log

모든 요청의 타임스탬프를 추적합니다. 새 요청이 허용되는지 확인하려면 과거 N초 이내에 해당하는 타임스탬프가 몇 개인지 셉니다.

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

**장점**: 경계 버스트 문제가 없고, 정확한 카운팅이 가능합니다.
**단점**: 메모리 사용량이 높고(모든 타임스탬프 저장), 연산 비용이 더 큽니다.

### Sliding Window Counter

Fixed window의 효율성과 sliding window의 정확성을 결합한 하이브리드 방식입니다. 인접한 두 개의 fixed window를 사용하고 경과 비율에 따라 카운트에 가중치를 부여합니다.

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

토큰이 고정 속도로 버킷에 추가됩니다. 각 요청은 하나의 토큰을 소비합니다. 버킷이 비면 요청이 거부됩니다. 버킷은 최대 용량을 가지며, 해당 용량까지의 짧은 버스트를 허용합니다.

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

**장점**: 제어된 버스트를 허용하고, 부드러운 속도 제한을 시행하며, 널리 알려져 있습니다.
**단점**: 구현이 약간 더 복잡하고, 원자적 연산이 필요합니다.

### 알고리즘 비교

| 알고리즘 | 버스트 처리 | 메모리 | 정확도 | 복잡도 |
|-----------|---------------|--------|----------|------------|
| Fixed window | 경계에서 2배 버스트 허용 | 낮음 | 낮음 | 단순 |
| Sliding window log | 버스트 없음 | 높음 | 정확 | 보통 |
| Sliding window counter | 근사값 | 낮음 | 양호 | 보통 |
| Token bucket | 제어된 버스트 | 낮음 | 양호 | 보통 |

---

## 3. Rate Limit 헤더

표준 헤더는 API 소비자에게 rate limit 상태를 전달합니다. 아직 공식 RFC는 아니지만, `X-RateLimit-*` 헤더는 널리 채택된 관례입니다. IETF 초안인 `RateLimit` 헤더 필드(draft-ietf-httpapi-ratelimit-headers)가 이를 표준화하고 있습니다.

### 표준 응답 헤더

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100          # Maximum requests allowed per window
X-RateLimit-Remaining: 57       # Requests remaining in current window
X-RateLimit-Reset: 1709827200   # Unix timestamp when the window resets
```

### 429 Too Many Requests 응답

클라이언트가 제한을 초과하면, `429` 상태 코드와 함께 `Retry-After` 헤더를 포함하여 응답합니다:

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

### Rate Limit 헤더를 위한 FastAPI 미들웨어

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

## 4. 사용자별 및 IP별 제한

다양한 식별 전략은 각각 다른 목적에 사용됩니다.

### 식별 전략

| 전략 | Key 소스 | 사용 사례 | 제한 사항 |
|----------|-----------|----------|-------------|
| IP 주소 | `X-Forwarded-For` 또는 클라이언트 IP | 비인증 엔드포인트 | 공유 IP (NAT, 프록시) |
| API key | `X-API-Key` 헤더 | 기계 간 API | Key 공유 |
| 사용자 ID | JWT `sub` claim | 인증된 엔드포인트 | 인증 필요 |
| 복합 | IP + User-Agent + route | 세밀한 제어 | 복잡한 key 관리 |

### 클라이언트 식별자 추출

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

### 라우트별 Rate Limit

엔드포인트마다 비용 프로필이 다릅니다. 전문 검색 인덱스를 쿼리하는 검색 엔드포인트는 단순한 ID 조회보다 비용이 더 높습니다.

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

## 5. Redis 기반 Rate Limiter

Redis는 원자적 연산, 서브밀리초 레이턴시, 내장 키 만료 기능 덕분에 프로덕션 rate limiter의 표준 백킹 스토어입니다.

### 프로덕션 레디 구현

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

### FastAPI 미들웨어로 Limiter 사용

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

## 6. 계층형 Rate Limiting

API 소비자마다 다른 제한이 필요합니다. 무료 등급 사용자는 기본 접근을 제공받고, 유료 고객은 더 높은 처리량을 얻습니다.

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

## 7. API 게이트웨이에서의 Rate Limiting

마이크로서비스 아키텍처에서 rate limiting은 개별 서비스가 아닌 API 게이트웨이 수준에서 적용하는 것이 가장 좋습니다. 이렇게 하면 정책 시행이 중앙화되고 로직 중복을 방지합니다.

### 게이트웨이 수준 vs 서비스 수준

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

| 접근 방식 | 장점 | 단점 |
|----------|------|------|
| 게이트웨이 수준 | 중앙화된 정책, 단일 시행 지점 | 서비스별 비용 차별화 불가 |
| 서비스 수준 | 서비스별 세밀한 제어 | 중복된 로직, 전역 관리 어려움 |
| 하이브리드 | 양쪽의 장점: 게이트웨이의 전역 제한, 서비스의 특화 제한 | 설정이 더 복잡 |

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

## 8. Throttling과 점진적 성능 저하

요청을 엄격하게 거부하는 대신, throttling은 부하를 관리하기 위해 응답을 느리게 합니다. 이는 부하 상황에서 더 나은 사용자 경험을 제공합니다.

### Backpressure 패턴

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

극심한 부하 상황에서는 모든 요청을 느리게 처리하려 시도하는 것(연쇄 장애로 이어질 수 있음)보다 일부 요청을 빠르게 거부하는 것이 더 나은 선택입니다.

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

## 9. 연습 문제

### 연습 문제 1: Leaky Bucket 구현

Leaky bucket rate limiter를 구현하십시오. Token bucket과 달리 leaky bucket은 일정한 속도로 요청을 처리합니다. 버킷이 가득 차면 새 요청이 거부됩니다. 상태 관리를 위해 Redis를 사용하십시오. 응답에 적절한 rate limit 헤더를 포함하고, 다음을 검증하는 테스트를 작성하십시오:

- 제한 내의 요청이 허용됨
- 버스트 요청이 버킷을 채움
- 버킷이 일정한 속도로 비워짐
- 버킷이 가득 차면 요청이 거부됨

### 연습 문제 2: 다중 키 Rate Limiting

여러 차원에서 동시에 제한을 시행하는 rate limiting 시스템을 설계하십시오:

- IP별: 60 요청/분 (비인증)
- 사용자별: 200 요청/분
- 엔드포인트별: 라우트에 따라 다름 (예: `/search` = 20/분, `/users` = 100/분)
- 전역: 모든 클라이언트에 걸쳐 10,000 요청/분

네 가지 제한이 모두 원자적으로 확인되어야 합니다. 어떤 제한이 초과되면 어떤 제한에 걸렸는지를 나타내는 헤더와 함께 429를 반환하십시오.

### 연습 문제 3: Rate Limit 대시보드

모든 활성 클라이언트의 현재 rate limit 상태를 반환하는 FastAPI 엔드포인트 `GET /admin/rate-limits`를 구축하십시오. 각 클라이언트에 대해 다음을 표시하십시오:

- 클라이언트 식별자 (IP 또는 사용자 ID)
- 현재 요청 수
- 제한 및 남은 할당량
- 윈도우 리셋까지 남은 시간

서버를 차단하지 않고 효율적으로 rate limit 키를 반복하기 위해 Redis `SCAN`을 사용하십시오.

### 연습 문제 4: 클라이언트 측 Rate Limiting

서버의 rate limit 헤더를 준수하는 Python API 클라이언트 클래스를 작성하십시오. 클라이언트는 다음을 수행해야 합니다:

- 응답에서 `X-RateLimit-Remaining`과 `X-RateLimit-Reset`을 파싱
- 남은 할당량이 적을 때 자동으로 속도를 낮춤
- `429` 응답을 받으면 대기 후 재시도 (`Retry-After` 준수)
- 연속적인 429 응답에 대해 지수 백오프 구현

### 연습 문제 5: 분산 Rate Limiting

동일 사용자의 요청이 서로 다른 데이터센터에 도달할 수 있는 다중 리전 API 배포를 위한 rate limiting 솔루션을 설계하십시오. 다음을 고려하십시오:

- 리전 간 최종 일관성(eventual consistency)
- 로컬 vs 전역 rate limit 시행
- Redis가 일시적으로 사용 불가할 때의 동작 (fail-open vs fail-closed)
- 리전 간 클록 스큐(clock skew) 처리 방법

접근 방식에 대한 의사 코드 또는 설계 문서를 작성하십시오.

---

## 10. 참고 자료

- [RFC 6585: Additional HTTP Status Codes (429)](https://tools.ietf.org/html/rfc6585)
- [IETF Draft: RateLimit Header Fields](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/)
- [Stripe Rate Limiting](https://stripe.com/docs/rate-limits)
- [Cloudflare Rate Limiting](https://developers.cloudflare.com/waf/rate-limiting-rules/)
- [Redis Rate Limiting Patterns](https://redis.io/glossary/rate-limiting/)
- [Kong Rate Limiting Plugin](https://docs.konghq.com/hub/kong-inc/rate-limiting/)
- [slowapi - Rate Limiting for FastAPI](https://github.com/laurentS/slowapi)

---

**이전**: [Lesson 8](./08_Caching_Strategies.md) | [개요](./00_Overview.md) | **다음**: [API 문서화](./10_API_Documentation.md)

**License**: CC BY-NC 4.0
