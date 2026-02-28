"""
Exercises for Lesson 05: Reverse Proxy and API Gateway
Topic: System_Design

Solutions to practice problems and hands-on exercises.
Covers rate limiting algorithms, API gateway routing, and caching proxy.
"""

import time
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


# === Exercise 1: Rate Limiting Selection ===
# Problem: Choose appropriate Rate Limiting algorithm for scenarios.

def exercise_1():
    """Rate limiting algorithm selection."""
    scenarios = [
        {
            "scenario": "Public API (allow bursts, average limiting)",
            "choice": "Token Bucket",
            "reason": "Allows burst traffic for better user experience. "
                      "Average rate limiting prevents sustained abuse.",
        },
        {
            "scenario": "Real-time streaming service",
            "choice": "Leaky Bucket",
            "reason": "Constant output rate maintains stream quality. "
                      "No bursts ensures stable transmission.",
        },
        {
            "scenario": "Login endpoint (prevent brute force)",
            "choice": "Fixed/Sliding Window",
            "reason": "Strict limiting (e.g., 5 attempts/min). "
                      "Simplicity is fine since boundary issues are minor for login.",
        },
        {
            "scenario": "Simple requirements, quick implementation",
            "choice": "Fixed Window",
            "reason": "Simplest implementation. Only needs a counter + timestamp. "
                      "Good enough for many use cases.",
        },
    ]

    print("Rate Limiting Algorithm Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {s['scenario']}")
        print(f"   Choice: {s['choice']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 2: Token Bucket Rate Limiter ===
# Problem: Implement Token Bucket algorithm.

class TokenBucket:
    """Token Bucket rate limiter.

    Tokens accumulate at a constant rate up to a maximum capacity.
    Each request consumes one token. If no tokens are available,
    the request is rejected.
    """

    def __init__(self, capacity=100, refill_rate=10.0):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def allow_request(self):
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class SlidingWindowRateLimiter:
    """Sliding window rate limiter.

    Tracks request timestamps within a sliding time window.
    More accurate than fixed window, no boundary issues.
    """

    def __init__(self, max_requests=10, window_seconds=1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []

    def allow_request(self):
        now = time.time()
        # Remove requests outside the window
        cutoff = now - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False


class PerClientRateLimiter:
    """Per-client rate limiter that tracks separate limits per client IP."""

    def __init__(self, capacity=10, refill_rate=2.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, TokenBucket] = {}

    def allow_request(self, client_ip):
        if client_ip not in self.buckets:
            self.buckets[client_ip] = TokenBucket(self.capacity, self.refill_rate)
        return self.buckets[client_ip].allow_request()


def exercise_2():
    """Token Bucket rate limiter implementation and comparison."""
    print("Token Bucket Implementation:")
    print("=" * 60)

    bucket = TokenBucket(capacity=100, refill_rate=10)

    # Burst test: send 120 requests instantly
    allowed = sum(1 for _ in range(120) if bucket.allow_request())
    print(f"Burst test: {allowed}/120 requests allowed (capacity=100)")

    # Wait for refill and try again
    time.sleep(0.5)  # Should refill ~5 tokens
    allowed_after_wait = sum(1 for _ in range(10) if bucket.allow_request())
    print(f"After 0.5s wait: {allowed_after_wait}/10 requests allowed")

    # Comparison: Token Bucket vs Sliding Window
    print("\n--- Token Bucket vs Sliding Window ---")
    print("Bursty pattern: 20 requests every 100ms for 1 second")

    tb = TokenBucket(capacity=10, refill_rate=10)
    sw = SlidingWindowRateLimiter(max_requests=10, window_seconds=1.0)

    tb_results = []
    sw_results = []

    for burst in range(10):
        for _ in range(2):  # 2 requests per burst
            tb_results.append(tb.allow_request())
            sw_results.append(sw.allow_request())
        time.sleep(0.1)

    print(f"Token Bucket: {sum(tb_results)}/20 allowed")
    print(f"Sliding Window: {sum(sw_results)}/20 allowed")

    # Per-client rate limiter
    print("\n--- Per-Client Rate Limiter ---")
    per_client = PerClientRateLimiter(capacity=5, refill_rate=1.0)

    clients = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]
    for client in clients:
        allowed = sum(1 for _ in range(10) if per_client.allow_request(client))
        print(f"  {client}: {allowed}/10 allowed (capacity=5 per client)")


# === Exercise 3: API Gateway Router ===
# Problem: Build a simple API gateway with routing, middleware, and path parameters.

class Request:
    """Simulated HTTP request."""
    def __init__(self, method, path, headers=None, client_ip="127.0.0.1"):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self.client_ip = client_ip
        self.params = {}


class Response:
    """Simulated HTTP response."""
    def __init__(self, status=200, body="", headers=None):
        self.status = status
        self.body = body
        self.headers = headers or {}


class Gateway:
    """Simple API Gateway with routing, middleware, and path parameters."""

    def __init__(self):
        self.routes = []
        self.middleware_chain = []
        self.rate_limiters = {}

    def route(self, pattern, service_handler):
        """Register a route pattern -> service handler."""
        # Convert /api/users/:id to regex
        regex_pattern = re.sub(r':(\w+)', r'(?P<\1>[^/]+)', pattern)
        regex_pattern = f"^{regex_pattern}$"
        self.routes.append((re.compile(regex_pattern), pattern, service_handler))

    def use(self, middleware_fn):
        """Add middleware to the chain."""
        self.middleware_chain.append(middleware_fn)

    def handle(self, request):
        """Process a request through middleware chain and routing."""
        # Run middleware chain
        for middleware in self.middleware_chain:
            result = middleware(request)
            if isinstance(result, Response):
                return result  # Middleware rejected the request

        # Find matching route
        for regex, pattern, handler in self.routes:
            match = regex.match(request.path)
            if match:
                request.params = match.groupdict()
                return handler(request)

        return Response(404, "Not Found")


def auth_middleware(request):
    """JWT authentication middleware (simulated)."""
    # Skip auth for /api/auth/* paths
    if request.path.startswith("/api/auth"):
        return None  # Continue

    token = request.headers.get("Authorization", "")
    if not token.startswith("Bearer "):
        return Response(401, "Unauthorized: Missing JWT token")

    # Simulate token validation
    if token == "Bearer valid_token":
        return None  # Continue
    return Response(403, "Forbidden: Invalid token")


def logging_middleware(request):
    """Request logging middleware."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [LOG] {timestamp} {request.method} {request.path} "
          f"from {request.client_ip}")
    return None


def cors_middleware(request):
    """CORS headers middleware - adds headers to response (simplified)."""
    # In real implementation, would modify response headers
    return None


# Service handlers
def user_service(request):
    user_id = request.params.get("id", "list")
    return Response(200, f"User Service: user_id={user_id}")


def order_service(request):
    order_id = request.params.get("id", "list")
    return Response(200, f"Order Service: order_id={order_id}")


def auth_service(request):
    return Response(200, "Auth Service: token generated")


def exercise_3():
    """API Gateway with routing and middleware."""
    print("API Gateway Router:")
    print("=" * 60)

    gateway = Gateway()

    # Register middleware
    gateway.use(logging_middleware)
    gateway.use(auth_middleware)

    # Register routes
    gateway.route("/api/users", user_service)
    gateway.route("/api/users/:id", user_service)
    gateway.route("/api/orders", order_service)
    gateway.route("/api/orders/:id", order_service)
    gateway.route("/api/auth/login", auth_service)

    # Test requests
    test_cases = [
        Request("POST", "/api/auth/login", client_ip="10.0.0.1"),
        Request("GET", "/api/users/123",
                headers={"Authorization": "Bearer valid_token"},
                client_ip="10.0.0.2"),
        Request("GET", "/api/orders/456",
                headers={"Authorization": "Bearer valid_token"},
                client_ip="10.0.0.3"),
        Request("GET", "/api/users/789", client_ip="10.0.0.4"),  # No token
        Request("GET", "/api/unknown", headers={"Authorization": "Bearer valid_token"},
                client_ip="10.0.0.5"),
    ]

    print("\nProcessing requests:")
    for req in test_cases:
        resp = gateway.handle(req)
        auth = req.headers.get("Authorization", "none")[:20]
        print(f"  -> {req.method} {req.path} (auth={auth}...)")
        print(f"     Response: {resp.status} - {resp.body}")
        print()


# === Exercise 4: Caching Reverse Proxy ===
# Problem: Implement a caching reverse proxy with TTL and Cache-Control.

class CachingProxy:
    """Caching reverse proxy that caches GET responses with TTL."""

    def __init__(self, backend_handler, default_ttl=60):
        self.backend = backend_handler
        self.default_ttl = default_ttl
        self.cache = {}  # path -> (response, expiry_time)
        self.stats = {"hits": 0, "misses": 0, "invalidations": 0}

    def _parse_cache_control(self, headers):
        """Parse Cache-Control header."""
        cc = headers.get("Cache-Control", "")
        directives = {}
        for part in cc.split(","):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                directives[key.strip()] = val.strip()
            else:
                directives[part] = True
        return directives

    def handle(self, request):
        """Handle request with caching logic."""
        # Only cache GET requests
        if request.method != "GET":
            # Invalidate cache for this path on writes
            if request.path in self.cache:
                del self.cache[request.path]
                self.stats["invalidations"] += 1
            return self.backend(request)

        # Check cache
        if request.path in self.cache:
            resp, expiry = self.cache[request.path]
            if time.time() < expiry:
                self.stats["hits"] += 1
                return Response(resp.status, resp.body,
                                {**resp.headers, "X-Cache": "HIT"})
            else:
                del self.cache[request.path]

        # Cache miss - fetch from backend
        self.stats["misses"] += 1
        resp = self.backend(request)

        # Parse Cache-Control
        cc = self._parse_cache_control(resp.headers)

        if "no-store" in cc:
            return resp

        ttl = int(cc.get("max-age", self.default_ttl))
        if "no-cache" not in cc:
            self.cache[request.path] = (resp, time.time() + ttl)

        resp.headers["X-Cache"] = "MISS"
        return resp

    def get_stats(self):
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {**self.stats, "total": total, "hit_rate": f"{hit_rate:.1%}"}


def exercise_4():
    """Caching reverse proxy implementation."""
    print("Caching Reverse Proxy:")
    print("=" * 60)

    call_count = {"value": 0}

    def backend(request):
        call_count["value"] += 1
        if "products" in request.path:
            return Response(200, f"Product data #{call_count['value']}",
                            {"Cache-Control": "max-age=300"})
        elif "login" in request.path:
            return Response(200, f"Login status",
                            {"Cache-Control": "no-store"})
        return Response(200, f"Data #{call_count['value']}",
                        {"Cache-Control": "max-age=60"})

    proxy = CachingProxy(backend, default_ttl=60)

    # Read-heavy workload (90% GET)
    print("\n--- Read-heavy workload (90% GET, 10% POST) ---")
    random.seed(42)
    paths = ["/api/products/1", "/api/products/2", "/api/products/3",
             "/api/users/1", "/api/login"]

    for _ in range(100):
        path = random.choice(paths)
        if random.random() < 0.9:
            req = Request("GET", path)
        else:
            req = Request("POST", path)
        resp = proxy.handle(req)

    stats = proxy.get_stats()
    print(f"  Requests: {stats['total']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Cache invalidations: {stats['invalidations']}")
    print(f"  Hit rate: {stats['hit_rate']}")
    print(f"  Backend calls: {call_count['value']}")

    # Write-heavy workload
    print("\n--- Write-heavy workload (50% GET, 50% POST) ---")
    proxy2 = CachingProxy(backend, default_ttl=60)
    call_count["value"] = 0

    for _ in range(100):
        path = random.choice(paths[:3])  # Only product paths
        if random.random() < 0.5:
            req = Request("GET", path)
        else:
            req = Request("POST", path)
        resp = proxy2.handle(req)

    stats2 = proxy2.get_stats()
    print(f"  Requests: {stats2['total']}")
    print(f"  Cache hits: {stats2['hits']}")
    print(f"  Cache misses: {stats2['misses']}")
    print(f"  Cache invalidations: {stats2['invalidations']}")
    print(f"  Hit rate: {stats2['hit_rate']}")
    print(f"  Backend calls: {call_count['value']}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Rate Limiting Selection ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Token Bucket Rate Limiter ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: API Gateway Router ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Caching Reverse Proxy ===")
    print("=" * 60)
    exercise_4()

    print("\nAll exercises completed!")
