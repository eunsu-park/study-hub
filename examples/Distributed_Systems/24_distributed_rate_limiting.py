"""
Distributed Rate Limiting

Implements token bucket and sliding window rate limiters, then extends
them to a distributed setting with coordinated counters across multiple
nodes. Demonstrates local vs global rate limits and the tradeoffs of
synchronisation frequency.

Key concepts:
- Token bucket: fixed-rate token replenishment, burst capacity
- Sliding window: count requests in a rolling time window
- Distributed counters: synchronise rate state across nodes
- Local vs global rate limits: accuracy vs latency tradeoff
- Redis-based rate limiting pattern

Usage:
    python 24_distributed_rate_limiting.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Token Bucket
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Token bucket rate limiter.
    Tokens are added at a fixed rate; requests consume tokens.
    Burst is allowed up to the bucket capacity.
    """

    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: Tokens added per second.
            capacity: Maximum tokens in the bucket.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self._last_refill: float = 0.0

    def _refill(self, now: float) -> None:
        elapsed = now - self._last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self._last_refill = now

    def allow(self, now: float, cost: float = 1.0) -> bool:
        """Check if a request is allowed. Consumes tokens if yes."""
        self._refill(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def tokens_available(self, now: float) -> float:
        self._refill(now)
        return self.tokens


# ---------------------------------------------------------------------------
# Sliding Window
# ---------------------------------------------------------------------------

class SlidingWindowCounter:
    """
    Sliding window rate limiter using fixed sub-windows.
    Approximates the sliding window with weighted counts from current
    and previous windows.
    """

    def __init__(self, window_s: float, max_requests: int):
        self.window_s = window_s
        self.max_requests = max_requests
        self._prev_count = 0
        self._curr_count = 0
        self._curr_window_start = 0.0

    def _advance_window(self, now: float) -> None:
        window_start = (now // self.window_s) * self.window_s
        if window_start > self._curr_window_start:
            if window_start - self._curr_window_start >= 2 * self.window_s:
                self._prev_count = 0
                self._curr_count = 0
            else:
                self._prev_count = self._curr_count
                self._curr_count = 0
            self._curr_window_start = window_start

    def allow(self, now: float) -> bool:
        """Check if a request is allowed."""
        self._advance_window(now)

        # Weight previous window by remaining portion
        elapsed_in_window = now - self._curr_window_start
        prev_weight = 1.0 - (elapsed_in_window / self.window_s)
        estimated = self._prev_count * prev_weight + self._curr_count

        if estimated < self.max_requests:
            self._curr_count += 1
            return True
        return False


# ---------------------------------------------------------------------------
# Distributed Rate Limiter
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    """Rate limiter state on one node."""
    node_id: int
    local_count: int = 0
    global_count: int = 0
    local_limit: int = 0


class DistributedRateLimiter:
    """
    Distributed rate limiter with periodic synchronisation.
    Each node maintains a local counter and periodically syncs
    with a central coordinator.
    """

    def __init__(self, n_nodes: int, global_limit: int,
                 sync_interval: float = 1.0):
        self.n_nodes = n_nodes
        self.global_limit = global_limit
        self.sync_interval = sync_interval
        self.nodes = {i: NodeState(i) for i in range(n_nodes)}
        self._central_count = 0
        self._last_sync = 0.0
        self._redistribute_limits()
        self.log: list[str] = []

    def _redistribute_limits(self) -> None:
        """Distribute the global limit evenly across nodes."""
        remaining = self.global_limit - self._central_count
        per_node = max(0, remaining // self.n_nodes)
        leftover = max(0, remaining % self.n_nodes)

        for i, node in self.nodes.items():
            node.local_limit = per_node + (1 if i < leftover else 0)
            node.local_count = 0

    def allow(self, node_id: int, now: float) -> bool:
        """Check rate limit at a specific node."""
        if now - self._last_sync >= self.sync_interval:
            self._sync(now)

        node = self.nodes[node_id]
        if node.local_count < node.local_limit:
            node.local_count += 1
            return True
        return False

    def _sync(self, now: float) -> None:
        """Synchronise all nodes with central coordinator."""
        total_used = sum(n.local_count for n in self.nodes.values())
        self._central_count += total_used
        self._last_sync = now

        self.log.append(
            f"SYNC at t={now:.1f}: total_used={total_used}, "
            f"global={self._central_count}/{self.global_limit}")

        self._redistribute_limits()

    @property
    def total_allowed(self) -> int:
        return self._central_count + sum(n.local_count for n in self.nodes.values())


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_token_bucket() -> None:
    """Demonstrate token bucket rate limiter."""
    print("=" * 70)
    print("Token Bucket Rate Limiter")
    print("=" * 70)

    bucket = TokenBucket(rate=10.0, capacity=20.0)  # 10 req/s, burst 20

    print(f"\n  Rate: 10 tokens/s, Capacity: 20 tokens (burst)\n")

    # Burst of 25 requests at t=0
    allowed = 0
    denied = 0
    for i in range(25):
        if bucket.allow(0.0):
            allowed += 1
        else:
            denied += 1
    print(f"  Burst of 25 at t=0: {allowed} allowed, {denied} denied "
          f"(bucket had 20 tokens)")

    # After waiting 1 second (10 new tokens)
    allowed = 0
    denied = 0
    for i in range(15):
        if bucket.allow(1.0):
            allowed += 1
        else:
            denied += 1
    print(f"  15 requests at t=1: {allowed} allowed, {denied} denied "
          f"(10 tokens refilled)")

    # Steady state: 1 request per 100ms
    print(f"\n  Steady state (1 req per 100ms):")
    for t_ms in range(2000, 2500, 100):
        t = t_ms / 1000.0
        ok = bucket.allow(t)
        tokens = bucket.tokens_available(t)
        print(f"    t={t:.1f}s: {'ALLOW' if ok else 'DENY'} "
              f"(tokens={tokens:.1f})")


def demo_sliding_window() -> None:
    """Demonstrate sliding window rate limiter."""
    print("\n" + "=" * 70)
    print("Sliding Window Counter Rate Limiter")
    print("=" * 70)

    limiter = SlidingWindowCounter(window_s=10.0, max_requests=5)

    print(f"\n  Window: 10s, Max: 5 requests per window\n")

    # Submit requests over time
    request_times = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16]

    for t in request_times:
        ok = limiter.allow(float(t))
        print(f"    t={t:>3}s: {'ALLOW' if ok else 'DENY'}")


def demo_distributed_limiter() -> None:
    """Demonstrate distributed rate limiting."""
    print("\n" + "=" * 70)
    print("Distributed Rate Limiter (4 Nodes)")
    print("=" * 70)

    limiter = DistributedRateLimiter(
        n_nodes=4, global_limit=100, sync_interval=2.0)

    print(f"\n  4 nodes, global limit: 100 requests, sync every 2s")
    print(f"  Per-node limit: ~{100 // 4} requests between syncs\n")

    # Simulate requests across nodes
    node_counts: dict[int, dict[str, int]] = {
        i: {"allowed": 0, "denied": 0} for i in range(4)
    }

    for t_100ms in range(100):
        t = t_100ms / 10.0
        node_id = t_100ms % 4

        if limiter.allow(node_id, t):
            node_counts[node_id]["allowed"] += 1
        else:
            node_counts[node_id]["denied"] += 1

    print(f"  Results after 10 seconds:")
    total_allowed = 0
    for nid, counts in node_counts.items():
        total_allowed += counts["allowed"]
        print(f"    Node {nid}: {counts['allowed']} allowed, "
              f"{counts['denied']} denied")

    print(f"\n  Total allowed: {total_allowed} (limit: 100)")

    print(f"\n  Sync events:")
    for line in limiter.log:
        print(f"    {line}")


def demo_accuracy_tradeoff() -> None:
    """Show accuracy vs sync frequency tradeoff."""
    print("\n" + "=" * 70)
    print("Accuracy vs Sync Frequency Tradeoff")
    print("=" * 70)

    global_limit = 100
    intervals = [0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"\n  Global limit: {global_limit}, 4 nodes, 200 requests over 10s")
    print(f"\n  {'Sync Interval':>14}  {'Total Allowed':>14}  {'Overrun':>8}  "
          f"{'Syncs':>6}")
    print("  " + "-" * 50)

    for interval in intervals:
        limiter = DistributedRateLimiter(
            n_nodes=4, global_limit=global_limit, sync_interval=interval)

        allowed = 0
        for t_50ms in range(200):
            t = t_50ms / 20.0
            node_id = t_50ms % 4
            if limiter.allow(node_id, t):
                allowed += 1

        overrun = max(0, allowed - global_limit)
        syncs = len(limiter.log)
        print(f"  {interval:>13.1f}s  {allowed:>14}  {overrun:>8}  {syncs:>6}")

    print("""
  Tradeoff:
  - Frequent syncs: more accurate, but higher coordination overhead
  - Infrequent syncs: less accurate (may exceed limit), lower overhead
  - Redis-based: INCR + EXPIRE per key (very frequent, single source)
""")


def demo_redis_pattern() -> None:
    """Show the Redis-based rate limiting pattern."""
    print("=" * 70)
    print("Redis-Based Rate Limiting Pattern (Pseudocode)")
    print("=" * 70)

    print("""
  Fixed Window (simple):
    key = f"ratelimit:{client_ip}:{window}"
    count = INCR key
    if count == 1:
        EXPIRE key {window_size}
    if count > limit:
        return 429 Too Many Requests

  Sliding Window Log (precise):
    key = f"ratelimit:{client_ip}"
    now = current_timestamp_ms
    ZREMRANGEBYSCORE key 0 (now - window_ms)  # Remove old entries
    count = ZCARD key
    if count < limit:
        ZADD key now now                       # Add current request
        return 200 OK
    else:
        return 429 Too Many Requests

  Token Bucket (with Lua script for atomicity):
    -- Lua script executed atomically
    local tokens = tonumber(redis.call('get', KEYS[1])) or capacity
    local last = tonumber(redis.call('get', KEYS[2])) or now
    local elapsed = now - last
    tokens = math.min(capacity, tokens + elapsed * rate)
    if tokens >= 1 then
        redis.call('set', KEYS[1], tokens - 1)
        redis.call('set', KEYS[2], now)
        return 1  -- allowed
    end
    return 0      -- denied
""")


if __name__ == "__main__":
    demo_token_bucket()
    demo_sliding_window()
    demo_distributed_limiter()
    demo_accuracy_tradeoff()
    demo_redis_pattern()
    print("Done.")
