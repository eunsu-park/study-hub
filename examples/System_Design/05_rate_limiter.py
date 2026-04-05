"""
Rate Limiter Implementations

Demonstrates:
- Token bucket algorithm
- Sliding window counter
- Fixed window counter
- Leaky bucket

Theory:
- Rate limiters control the rate of requests to protect services.
- Token bucket: tokens added at fixed rate, each request consumes one.
  Allows bursts up to bucket capacity.
- Sliding window: counts requests in a sliding time window.
  More accurate than fixed window but more memory.
- Leaky bucket: requests queue and drain at constant rate.
  Smooths traffic, no bursts.

Adapted from System Design Lesson 05.
"""

import time
from collections import deque
from threading import Lock


# Why: Token bucket is the most common rate limiter in production (used by AWS,
# Stripe). It naturally allows bursts up to the bucket capacity while enforcing
# a long-term average rate — a good fit for APIs with bursty but bounded traffic.
class TokenBucket:
    """Token bucket rate limiter.

    - Tokens refill at `rate` tokens per second
    - Maximum `capacity` tokens
    - Each request consumes 1 token
    """

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        # Why: Start with a full bucket so the service can handle an initial
        # burst immediately at startup without waiting for tokens to accumulate.
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def allow(self) -> bool:
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self.tokens


# Why: Sliding window avoids the "boundary burst" problem of fixed windows,
# where a user could send 2x the limit by timing requests at a window boundary.
# The trade-off is higher memory (storing every timestamp) vs fixed window's O(1).
class SlidingWindowCounter:
    """Sliding window counter rate limiter.

    Counts requests in a sliding time window.
    """

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        # Why: deque enables O(1) removal of expired timestamps from the front,
        # since timestamps are naturally ordered by insertion time.
        self.requests: deque[float] = deque()
        self.lock = Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            # Remove expired entries
            while self.requests and self.requests[0] <= now - self.window:
                self.requests.popleft()

            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False

    @property
    def current_count(self) -> int:
        now = time.monotonic()
        while self.requests and self.requests[0] <= now - self.window:
            self.requests.popleft()
        return len(self.requests)


# Why: Fixed window is the cheapest to implement (just a counter + timestamp),
# making it suitable for distributed systems where per-request state is costly.
# The known weakness: a client can send 2x the limit across a window boundary.
class FixedWindowCounter:
    """Fixed window counter rate limiter.

    Counts requests in fixed time windows.
    Simpler but has boundary burst issue.
    """

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self.count = 0
        self.window_start = time.monotonic()
        self.lock = Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            if now - self.window_start >= self.window:
                self.window_start = now
                self.count = 0

            if self.count < self.limit:
                self.count += 1
                return True
            return False


# Why: Leaky bucket enforces a strictly uniform output rate, which is ideal
# when the downstream service cannot handle any bursts (e.g., hardware with
# fixed processing capacity). Unlike token bucket, it smooths all traffic.
class LeakyBucket:
    """Leaky bucket rate limiter.

    Requests queue and drain at a constant rate.
    """

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # requests per second
        self.capacity = capacity
        self.queue: deque[float] = deque()
        self.last_drain = time.monotonic()
        self.lock = Lock()

    def _drain(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_drain
        to_drain = int(elapsed * self.rate)
        for _ in range(min(to_drain, len(self.queue))):
            self.queue.popleft()
        if to_drain > 0:
            self.last_drain = now

    def allow(self) -> bool:
        with self.lock:
            self._drain()
            if len(self.queue) < self.capacity:
                self.queue.append(time.monotonic())
                return True
            return False


# ── Simulation (non-real-time) ──────────────────────────────────────────

# Why: Real-time rate limiters use wall-clock time, making them hard to test
# deterministically. These simulation variants accept explicit timestamps,
# enabling reproducible demos and unit tests without sleep() calls.
class SimTokenBucket:
    """Non-real-time token bucket for simulation."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_time = 0.0

    def allow(self, timestamp: float) -> bool:
        elapsed = timestamp - self.last_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_time = timestamp
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class SimSlidingWindow:
    """Non-real-time sliding window for simulation."""

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self.requests: deque[float] = deque()

    def allow(self, timestamp: float) -> bool:
        while self.requests and self.requests[0] <= timestamp - self.window:
            self.requests.popleft()
        if len(self.requests) < self.limit:
            self.requests.append(timestamp)
            return True
        return False


# ── Demos ───────────────────────────────────────────────────────────────

def demo_token_bucket():
    print("=" * 60)
    print("TOKEN BUCKET RATE LIMITER")
    print("=" * 60)

    tb = SimTokenBucket(rate=5, capacity=10)

    print(f"\n  Config: rate=5/sec, capacity=10 tokens")

    # Burst of 15 requests at t=0
    print(f"\n  Burst of 15 requests at t=0:")
    allowed = sum(1 for i in range(15) if tb.allow(0.0))
    print(f"    Allowed: {allowed}/15 (bucket capacity = 10)")

    # After 1 second, 5 more tokens
    print(f"\n  After 1 second, 8 more requests:")
    allowed = sum(1 for i in range(8) if tb.allow(1.0 + i * 0.001))
    print(f"    Allowed: {allowed}/8 (refilled 5 tokens)")

    # Steady stream
    print(f"\n  Steady stream (1 request every 0.2s for 2s):")
    tb2 = SimTokenBucket(rate=5, capacity=10)
    results = []
    for i in range(10):
        t = i * 0.2
        ok = tb2.allow(t)
        results.append(("✓" if ok else "✗", f"{t:.1f}s"))
    for ok, t in results:
        print(f"    {t}: {ok}")


def demo_sliding_window():
    print("\n" + "=" * 60)
    print("SLIDING WINDOW RATE LIMITER")
    print("=" * 60)

    sw = SimSlidingWindow(limit=5, window=1.0)

    print(f"\n  Config: limit=5 requests per 1s window")

    # Simulate requests over 3 seconds
    request_times = [
        0.0, 0.1, 0.2, 0.3, 0.4,   # 5 in first window
        0.5, 0.6,                     # these should be rejected
        1.1, 1.2, 1.3, 1.4, 1.5,    # old ones expired
        2.0,
    ]

    print(f"\n  {'Time':>6}  {'Result':>8}  {'Window Count':>13}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*13}")
    for t in request_times:
        ok = sw.allow(t)
        count = len(sw.requests)
        print(f"  {t:>5.1f}s  {'ALLOW' if ok else 'DENY':>8}  {count:>13}")


def demo_comparison():
    """Compare rate limiters under burst traffic."""
    print("\n" + "=" * 60)
    print("RATE LIMITER COMPARISON")
    print("=" * 60)

    # Generate bursty traffic: 20 requests in 0.5s, then 2s pause, repeat
    timestamps = []
    t = 0.0
    for _ in range(3):  # 3 bursts
        for i in range(20):
            timestamps.append(t + i * 0.025)  # 20 requests in 0.5s
        t += 2.5  # 2.5s between burst starts

    tb = SimTokenBucket(rate=10, capacity=15)
    sw = SimSlidingWindow(limit=10, window=1.0)

    tb_allowed = sum(1 for t in timestamps if tb.allow(t))
    sw_allowed = sum(1 for t in timestamps if sw.allow(t))

    total = len(timestamps)
    print(f"\n  Traffic: 3 bursts of 20 requests (0.5s each), 2s gaps")
    print(f"  Total requests: {total}")
    print(f"\n  {'Algorithm':<25} {'Allowed':>8} {'Denied':>8} {'Rate':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'Token Bucket (10/s, 15)':<25} {tb_allowed:>8} {total-tb_allowed:>8} "
          f"{tb_allowed/total*100:>5.0f}%")
    print(f"  {'Sliding Window (10/1s)':<25} {sw_allowed:>8} {total-sw_allowed:>8} "
          f"{sw_allowed/total*100:>5.0f}%")

    print(f"\n  Token bucket allows initial burst (up to capacity).")
    print(f"  Sliding window strictly enforces per-window limit.")


if __name__ == "__main__":
    demo_token_bucket()
    demo_sliding_window()
    demo_comparison()
