# Lesson 23: Distributed Rate Limiting

[Overview](./00_Overview.md) | [Previous: Service Discovery](./22_Service_Discovery.md) | [Next: Event Sourcing and CQRS](./24_Event_Sourcing_CQRS.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement token bucket and sliding window rate limiters for single-node and distributed settings
2. Build distributed counters using Redis-based atomic operations
3. Design rate limiting with configurable policies (per-user, per-API, global)
4. Handle clock skew and network partition edge cases in distributed rate limiting
5. Analyze the trade-offs between accuracy, latency, and consistency in rate limiting strategies

---

## Table of Contents

1. [Rate Limiting Fundamentals](#1-rate-limiting-fundamentals)
2. [Token Bucket Algorithm](#2-token-bucket-algorithm)
3. [Sliding Window Algorithms](#3-sliding-window-algorithms)
4. [Distributed Rate Limiting Challenges](#4-distributed-rate-limiting-challenges)
5. [Redis-Based Implementation](#5-redis-based-implementation)
6. [Distributed Counters](#6-distributed-counters)
7. [Policy Configuration](#7-policy-configuration)
8. [Edge Cases and Failure Modes](#8-edge-cases-and-failure-modes)
9. [Production Patterns](#9-production-patterns)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Rate Limiting Fundamentals

### 1.1 Why Rate Limit?

Rate limiting protects services from overload, prevents abuse, and enforces fair usage policies. In a distributed system, rate limiting is particularly challenging because requests arrive at different nodes and there is no shared memory for counters.

```
Without rate limiting:                With rate limiting:
  Client → [1000 req/s] → Server     Client → [1000 req/s] → Rate Limiter → [100 req/s] → Server
  Server overloaded, crashes          Server handles within capacity
```

### 1.2 Algorithm Overview

| Algorithm | Accuracy | Memory | Burst Handling | Complexity |
|-----------|----------|--------|----------------|------------|
| Token Bucket | High | O(1) | Allows controlled bursts | Low |
| Leaky Bucket | High | O(1) | Smooths to fixed rate | Low |
| Fixed Window | Low (boundary burst) | O(1) | 2x burst at boundary | Lowest |
| Sliding Window Log | Exact | O(N) | No burst | High |
| Sliding Window Counter | Approximate | O(1) | Minimal burst | Low |

---

## 2. Token Bucket Algorithm

### 2.1 Implementation

```python
import time
import random
import threading
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter.

    Tokens are added at a fixed rate. Each request consumes one token.
    If no tokens are available, the request is rejected.

    The bucket has a maximum capacity, allowing controlled bursts
    up to the capacity size.
    """
    rate: float           # Tokens added per second
    capacity: float       # Maximum tokens (burst size)
    tokens: float = 0.0   # Current tokens
    last_refill: float = field(default_factory=time.time)
    total_allowed: int = 0
    total_rejected: int = 0

    def __post_init__(self):
        self.tokens = self.capacity  # Start full

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def allow(self, tokens: float = 1.0) -> bool:
        """
        Check if a request is allowed.

        Returns True and consumes tokens if allowed.
        Returns False if not enough tokens.
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            self.total_allowed += 1
            return True
        else:
            self.total_rejected += 1
            return False

    def wait_time(self, tokens: float = 1.0) -> float:
        """Calculate how long to wait for enough tokens."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        deficit = tokens - self.tokens
        return deficit / self.rate

    def stats(self) -> dict:
        self._refill()
        return {
            "tokens": round(self.tokens, 2),
            "capacity": self.capacity,
            "rate": self.rate,
            "allowed": self.total_allowed,
            "rejected": self.total_rejected,
            "utilization": round(
                self.total_allowed / max(1, self.total_allowed + self.total_rejected) * 100, 1
            ),
        }


def demonstrate_token_bucket():
    """Demonstrate the token bucket algorithm."""
    print("=== Token Bucket Rate Limiter ===\n")

    bucket = TokenBucket(rate=10.0, capacity=20.0)  # 10 req/s, burst of 20

    # Phase 1: Burst of 25 requests
    print("Phase 1: Burst of 25 requests")
    allowed = sum(1 for _ in range(25) if bucket.allow())
    print(f"  Allowed: {allowed}/25 (bucket capacity = 20)")

    # Phase 2: Wait and try again
    time.sleep(0.5)  # 5 tokens refill in 0.5s
    print(f"\nPhase 2: After 500ms wait")
    allowed = sum(1 for _ in range(10) if bucket.allow())
    print(f"  Allowed: {allowed}/10 (5 tokens refilled)")

    # Phase 3: Steady state
    print(f"\nPhase 3: Steady state (10 req/s for 2 seconds)")
    for second in range(2):
        time.sleep(0.1)
        allowed = sum(1 for _ in range(15) if bucket.allow())
        print(f"  Second {second + 1}: Allowed {allowed}/15 requests")

    print(f"\nStats: {bucket.stats()}")


demonstrate_token_bucket()
```

---

## 3. Sliding Window Algorithms

### 3.1 Fixed Window Counter

```python
class FixedWindowCounter:
    """
    Fixed window rate limiter.

    Divides time into fixed windows (e.g., 1-second intervals).
    Counts requests per window and rejects when limit is reached.

    Problem: A burst of requests at the boundary of two windows
    can allow up to 2x the rate limit.
    """

    def __init__(self, limit: int, window_size: float = 1.0):
        self.limit = limit
        self.window_size = window_size
        self.current_window: int = 0
        self.count: int = 0

    def _current_window_id(self) -> int:
        return int(time.time() / self.window_size)

    def allow(self) -> bool:
        window = self._current_window_id()
        if window != self.current_window:
            self.current_window = window
            self.count = 0

        if self.count < self.limit:
            self.count += 1
            return True
        return False


class SlidingWindowLog:
    """
    Sliding window log rate limiter.

    Maintains a log of all request timestamps within the window.
    Provides exact rate limiting but uses O(N) memory per client.
    """

    def __init__(self, limit: int, window_size: float = 1.0):
        self.limit = limit
        self.window_size = window_size
        self.timestamps: list[float] = []

    def allow(self) -> bool:
        now = time.time()
        cutoff = now - self.window_size

        # Remove expired entries
        self.timestamps = [t for t in self.timestamps if t > cutoff]

        if len(self.timestamps) < self.limit:
            self.timestamps.append(now)
            return True
        return False


class SlidingWindowCounter:
    """
    Sliding window counter rate limiter.

    Approximation that combines two adjacent fixed windows.
    Uses weighted counting based on how far into the current window we are.

    Memory: O(1) — only stores two counters.
    Accuracy: Approximate, but much better than fixed window at boundaries.
    """

    def __init__(self, limit: int, window_size: float = 1.0):
        self.limit = limit
        self.window_size = window_size
        self.prev_count: int = 0
        self.curr_count: int = 0
        self.prev_window: int = 0
        self.curr_window: int = 0

    def _current_window_id(self) -> int:
        return int(time.time() / self.window_size)

    def _window_progress(self) -> float:
        """How far into the current window (0.0 to 1.0)."""
        return (time.time() % self.window_size) / self.window_size

    def allow(self) -> bool:
        window = self._current_window_id()

        if window != self.curr_window:
            if window == self.curr_window + 1:
                self.prev_count = self.curr_count
                self.prev_window = self.curr_window
            else:
                self.prev_count = 0
            self.curr_count = 0
            self.curr_window = window

        # Weighted estimate: prev_count * (1 - progress) + curr_count
        progress = self._window_progress()
        estimated = self.prev_count * (1.0 - progress) + self.curr_count

        if estimated < self.limit:
            self.curr_count += 1
            return True
        return False


def compare_window_algorithms():
    """Compare fixed window, sliding log, and sliding counter."""
    print("=== Window Algorithm Comparison ===\n")

    limit = 10  # 10 requests per second

    for name, limiter in [
        ("Fixed Window", FixedWindowCounter(limit)),
        ("Sliding Log", SlidingWindowLog(limit)),
        ("Sliding Counter", SlidingWindowCounter(limit)),
    ]:
        allowed = 0
        rejected = 0

        # Send 5 requests at end of one window, 5 at start of next
        for _ in range(15):
            if limiter.allow():
                allowed += 1
            else:
                rejected += 1

        print(f"  {name:20s}: allowed={allowed}, rejected={rejected}")


compare_window_algorithms()
```

---

## 4. Distributed Rate Limiting Challenges

### 4.1 The Multi-Node Problem

```python
def illustrate_distributed_challenge():
    """Illustrate why distributed rate limiting is hard."""
    print("=== Distributed Rate Limiting Challenges ===\n")

    print("Scenario: 100 req/s limit, 5 API servers\n")

    approaches = {
        "Local only (no coordination)": {
            "strategy": "Each server limits to 100/5 = 20 req/s",
            "problem": "Uneven traffic → some servers waste capacity",
            "effective_limit": "20-100 req/s depending on distribution",
        },
        "Central counter (Redis)": {
            "strategy": "All servers check/increment a shared counter",
            "problem": "Redis latency added to every request",
            "effective_limit": "~100 req/s (accurate)",
        },
        "Leaky bucket with sync": {
            "strategy": "Local buckets with periodic sync",
            "problem": "Brief over-limit between syncs",
            "effective_limit": "100-120 req/s (slightly over)",
        },
        "Token bucket with prefetch": {
            "strategy": "Prefetch tokens from central store",
            "problem": "Wasted tokens if traffic shifts",
            "effective_limit": "90-110 req/s (close)",
        },
    }

    for name, info in approaches.items():
        print(f"  {name}:")
        print(f"    Strategy: {info['strategy']}")
        print(f"    Problem:  {info['problem']}")
        print(f"    Effective limit: {info['effective_limit']}")
        print()


illustrate_distributed_challenge()
```

### 4.2 Distributed Token Bucket

```python
class DistributedTokenBucket:
    """
    Distributed token bucket with central coordination.

    Each node maintains a local bucket and periodically syncs
    with a central coordinator (e.g., Redis) to refill tokens.

    This avoids per-request network calls while maintaining
    approximate global rate limiting.
    """

    def __init__(self, node_id: str, global_rate: float, global_capacity: float,
                 num_nodes: int, sync_interval: float = 0.5):
        self.node_id = node_id
        self.global_rate = global_rate
        self.global_capacity = global_capacity
        self.num_nodes = num_nodes
        self.sync_interval = sync_interval

        # Local bucket: proportional share
        self.local_rate = global_rate / num_nodes
        self.local_capacity = global_capacity / num_nodes
        self.local_bucket = TokenBucket(
            rate=self.local_rate,
            capacity=self.local_capacity,
        )

        # Sync state
        self.last_sync = time.time()
        self.tokens_borrowed: float = 0
        self.tokens_returned: float = 0

    def allow(self) -> bool:
        """Check if request is allowed using local bucket."""
        return self.local_bucket.allow()

    def sync_with_coordinator(self, coordinator: 'RateLimitCoordinator'):
        """
        Sync local bucket with central coordinator.

        - Report unused tokens (return to pool)
        - Request additional tokens if needed
        """
        now = time.time()
        if now - self.last_sync < self.sync_interval:
            return

        self.last_sync = now

        # Calculate unused capacity
        unused = self.local_bucket.tokens
        used_pct = 1.0 - (unused / max(0.01, self.local_capacity))

        # If underutilized, return tokens
        if used_pct < 0.5:
            return_amount = unused * 0.3
            self.local_bucket.tokens -= return_amount
            coordinator.return_tokens(self.node_id, return_amount)
            self.tokens_returned += return_amount

        # If overloaded, request more
        elif used_pct > 0.9:
            request_amount = self.local_capacity * 0.5
            granted = coordinator.request_tokens(self.node_id, request_amount)
            self.local_bucket.tokens += granted
            self.tokens_borrowed += granted

    def stats(self) -> dict:
        return {
            "node": self.node_id,
            "local_tokens": round(self.local_bucket.tokens, 2),
            "local_capacity": self.local_capacity,
            "borrowed": round(self.tokens_borrowed, 2),
            "returned": round(self.tokens_returned, 2),
            **self.local_bucket.stats(),
        }


class RateLimitCoordinator:
    """
    Central rate limit coordinator (backed by Redis in production).

    Manages a global token pool that nodes can borrow from and return to.
    """

    def __init__(self, global_rate: float, global_capacity: float):
        self.global_rate = global_rate
        self.global_capacity = global_capacity
        self.pool: float = global_capacity * 0.2  # Reserve 20% in pool
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        # Add a fraction of the global rate to the pool
        self.pool = min(
            self.global_capacity * 0.5,  # Max 50% in reserve
            self.pool + elapsed * self.global_rate * 0.2,
        )
        self.last_refill = now

    def request_tokens(self, node_id: str, amount: float) -> float:
        """Grant tokens from the central pool."""
        with self.lock:
            self._refill()
            granted = min(amount, self.pool)
            self.pool -= granted
            return granted

    def return_tokens(self, node_id: str, amount: float):
        """Accept returned tokens into the pool."""
        with self.lock:
            self.pool = min(self.global_capacity * 0.5, self.pool + amount)


def demonstrate_distributed_rate_limiting():
    """Demonstrate distributed rate limiting with coordination."""
    print("=== Distributed Rate Limiting ===\n")

    global_rate = 100.0  # 100 req/s global
    global_capacity = 200.0
    num_nodes = 5

    coordinator = RateLimitCoordinator(global_rate, global_capacity)
    nodes = [
        DistributedTokenBucket(f"node-{i}", global_rate, global_capacity, num_nodes)
        for i in range(num_nodes)
    ]

    # Simulate traffic (uneven distribution)
    traffic_weights = [0.4, 0.25, 0.15, 0.1, 0.1]  # Node 0 gets 40% of traffic
    total_allowed = 0
    total_rejected = 0

    for _ in range(10):  # 10 rounds of 0.1s each
        for i, node in enumerate(nodes):
            # Sync with coordinator
            node.sync_with_coordinator(coordinator)

            # Process requests proportional to traffic weight
            num_requests = int(15 * traffic_weights[i])  # ~15 req per 0.1s at 100/s
            for _ in range(num_requests):
                if node.allow():
                    total_allowed += 1
                else:
                    total_rejected += 1

        time.sleep(0.05)

    print(f"Global limit: {global_rate} req/s")
    print(f"Total allowed: {total_allowed}")
    print(f"Total rejected: {total_rejected}")
    print(f"\nPer-node stats:")
    for node in nodes:
        s = node.stats()
        print(f"  {s['node']}: allowed={s['allowed']}, rejected={s['rejected']}, "
              f"tokens={s['local_tokens']}")


demonstrate_distributed_rate_limiting()
```

---

## 5. Redis-Based Implementation

### 5.1 Atomic Rate Limiting with Lua Scripts

```python
class RedisRateLimiter:
    """
    Simulated Redis-based distributed rate limiter.

    In production, the core logic runs as a Redis Lua script
    for atomicity. This simulation demonstrates the algorithm.
    """

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def _execute_lua(self, script_name: str, keys: list, args: list) -> Any:
        """Simulate atomic Redis Lua script execution."""
        with self.lock:
            if script_name == "token_bucket":
                return self._lua_token_bucket(keys, args)
            elif script_name == "sliding_window":
                return self._lua_sliding_window(keys, args)

    def _lua_token_bucket(self, keys: list, args: list) -> Tuple[bool, float]:
        """
        Token bucket implemented as a Redis Lua script.

        KEYS[1] = rate limit key
        ARGV[1] = rate (tokens/sec)
        ARGV[2] = capacity
        ARGV[3] = now (timestamp)
        ARGV[4] = requested tokens

        This runs atomically in Redis, preventing race conditions
        between multiple API servers.
        """
        key = keys[0]
        rate = float(args[0])
        capacity = float(args[1])
        now = float(args[2])
        requested = float(args[3])

        # Get current state
        state = self.store.get(key, {"tokens": capacity, "last_refill": now})

        # Refill
        elapsed = now - state["last_refill"]
        tokens = min(capacity, state["tokens"] + elapsed * rate)

        # Check
        allowed = tokens >= requested
        if allowed:
            tokens -= requested

        # Save
        self.store[key] = {"tokens": tokens, "last_refill": now}
        return allowed, tokens

    def _lua_sliding_window(self, keys: list, args: list) -> Tuple[bool, int]:
        """
        Sliding window counter as a Redis Lua script.

        Uses two sorted sets (current and previous window)
        with weighted counting.
        """
        key = keys[0]
        limit = int(args[0])
        window_size = float(args[1])
        now = float(args[2])

        window_id = int(now / window_size)
        progress = (now % window_size) / window_size

        curr_key = f"{key}:{window_id}"
        prev_key = f"{key}:{window_id - 1}"

        curr_count = self.store.get(curr_key, 0)
        prev_count = self.store.get(prev_key, 0)

        estimated = prev_count * (1.0 - progress) + curr_count

        if estimated < limit:
            self.store[curr_key] = curr_count + 1
            return True, int(estimated + 1)
        return False, int(estimated)

    def check_rate_limit(self, client_id: str, rate: float = 10.0,
                         capacity: float = 20.0) -> dict:
        """Check rate limit for a client using token bucket."""
        now = time.time()
        key = f"ratelimit:token:{client_id}"
        allowed, tokens = self._execute_lua(
            "token_bucket", [key], [rate, capacity, now, 1.0]
        )
        return {
            "allowed": allowed,
            "remaining_tokens": round(tokens, 2),
            "limit": rate,
        }

    def check_sliding_window(self, client_id: str, limit: int = 100,
                              window: float = 60.0) -> dict:
        """Check rate limit using sliding window counter."""
        now = time.time()
        key = f"ratelimit:window:{client_id}"
        allowed, count = self._execute_lua(
            "sliding_window", [key], [limit, window, now]
        )
        return {
            "allowed": allowed,
            "current_count": count,
            "limit": limit,
            "window_seconds": window,
        }


def demonstrate_redis_rate_limiter():
    """Demonstrate Redis-based rate limiting."""
    print("=== Redis-Based Rate Limiting ===\n")

    limiter = RedisRateLimiter()

    # Token bucket: 5 req/s, burst of 10
    print("Token Bucket (5 req/s, burst=10):")
    for i in range(15):
        result = limiter.check_rate_limit("user-123", rate=5.0, capacity=10.0)
        status = "ALLOW" if result["allowed"] else "DENY "
        print(f"  Request {i+1:2d}: {status} (remaining={result['remaining_tokens']})")

    # Sliding window: 10 requests per 1 second
    print(f"\nSliding Window (10 req/1s):")
    for i in range(15):
        result = limiter.check_sliding_window("user-456", limit=10, window=1.0)
        status = "ALLOW" if result["allowed"] else "DENY "
        print(f"  Request {i+1:2d}: {status} (count={result['current_count']}/{result['limit']})")


demonstrate_redis_rate_limiter()
```

---

## 6. Distributed Counters

### 6.1 CRDT-Based Counter for Approximate Limiting

```python
class CRDTCounter:
    """
    CRDT-based distributed counter for approximate rate limiting.

    Each node maintains a local counter. Counters are merged
    using max() (PN-Counter approach). This provides eventual
    consistency without coordination.
    """

    def __init__(self, node_id: str, num_nodes: int):
        self.node_id = node_id
        self.num_nodes = num_nodes
        # Positive counter per node
        self.increments: Dict[str, int] = defaultdict(int)
        self.increments[node_id] = 0

    def increment(self):
        """Increment the local counter."""
        self.increments[self.node_id] += 1

    def value(self) -> int:
        """Get the current counter value."""
        return sum(self.increments.values())

    def merge(self, other: 'CRDTCounter'):
        """Merge with another counter (take max of each node's count)."""
        for node_id, count in other.increments.items():
            self.increments[node_id] = max(self.increments[node_id], count)


def demonstrate_crdt_counter():
    """Demonstrate CRDT-based distributed counter for rate limiting."""
    print("=== CRDT Counter for Rate Limiting ===\n")

    # 3 nodes, each counting independently
    counters = {
        f"node-{i}": CRDTCounter(f"node-{i}", 3)
        for i in range(3)
    }

    # Each node gets some requests
    for _ in range(10):
        counters["node-0"].increment()
    for _ in range(7):
        counters["node-1"].increment()
    for _ in range(3):
        counters["node-2"].increment()

    print("Before merge:")
    for nid, c in counters.items():
        print(f"  {nid}: local={c.increments[nid]}, total={c.value()}")

    # Merge (gossip round)
    for nid1 in counters:
        for nid2 in counters:
            if nid1 != nid2:
                counters[nid1].merge(counters[nid2])

    print("\nAfter merge:")
    for nid, c in counters.items():
        print(f"  {nid}: total={c.value()}")

    limit = 25
    print(f"\nGlobal limit: {limit}")
    print(f"Global count: {counters['node-0'].value()}")
    print(f"Over limit: {counters['node-0'].value() > limit}")


demonstrate_crdt_counter()
```

---

## 7. Policy Configuration

### 7.1 Multi-Tier Rate Limiting

```python
@dataclass
class RateLimitPolicy:
    """Configuration for a rate limit policy."""
    name: str
    limit: int
    window_seconds: float
    scope: str  # "global", "per_user", "per_ip", "per_api_key"
    algorithm: str  # "token_bucket", "sliding_window", "fixed_window"
    burst_multiplier: float = 1.5  # Allow burst up to limit * multiplier
    retry_after_seconds: float = 1.0


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter with configurable policies.

    Applies multiple rate limits simultaneously (e.g., per-second
    and per-minute), and the most restrictive one wins.
    """

    def __init__(self):
        self.policies: Dict[str, list[RateLimitPolicy]] = defaultdict(list)
        self.buckets: Dict[str, TokenBucket] = {}

    def add_policy(self, scope_value: str, policy: RateLimitPolicy):
        """Add a rate limit policy for a scope value."""
        self.policies[scope_value].append(policy)
        key = f"{scope_value}:{policy.name}"
        self.buckets[key] = TokenBucket(
            rate=policy.limit / policy.window_seconds,
            capacity=policy.limit * policy.burst_multiplier,
        )

    def check(self, scope_value: str) -> dict:
        """
        Check all applicable rate limits.

        Returns the most restrictive result.
        """
        policies = self.policies.get(scope_value, [])
        if not policies:
            return {"allowed": True, "policy": None}

        for policy in policies:
            key = f"{scope_value}:{policy.name}"
            bucket = self.buckets.get(key)
            if bucket and not bucket.allow():
                return {
                    "allowed": False,
                    "policy": policy.name,
                    "retry_after": policy.retry_after_seconds,
                    "limit": policy.limit,
                    "window": policy.window_seconds,
                }

        return {"allowed": True, "policy": None}


def demonstrate_multi_tier():
    """Demonstrate multi-tier rate limiting."""
    print("=== Multi-Tier Rate Limiting ===\n")

    limiter = MultiTierRateLimiter()

    # User "alice" gets three tiers
    limiter.add_policy("user:alice", RateLimitPolicy(
        name="per_second", limit=10, window_seconds=1.0, scope="per_user",
        algorithm="token_bucket",
    ))
    limiter.add_policy("user:alice", RateLimitPolicy(
        name="per_minute", limit=100, window_seconds=60.0, scope="per_user",
        algorithm="token_bucket",
    ))
    limiter.add_policy("user:alice", RateLimitPolicy(
        name="per_hour", limit=1000, window_seconds=3600.0, scope="per_user",
        algorithm="token_bucket",
    ))

    print("Policies for user:alice:")
    for p in limiter.policies["user:alice"]:
        print(f"  {p.name}: {p.limit} per {p.window_seconds}s")

    # Burst of 20 requests
    allowed = 0
    first_reject_policy = None
    for i in range(20):
        result = limiter.check("user:alice")
        if result["allowed"]:
            allowed += 1
        elif first_reject_policy is None:
            first_reject_policy = result["policy"]

    print(f"\nBurst of 20 requests: {allowed} allowed")
    print(f"First rejection by: {first_reject_policy}")


demonstrate_multi_tier()
```

---

## 8. Edge Cases and Failure Modes

### 8.1 Clock Skew

```python
def analyze_clock_skew_impact():
    """Analyze the impact of clock skew on distributed rate limiting."""
    print("=== Clock Skew Impact ===\n")

    # Scenario: 3 nodes with clock skew
    node_offsets = {
        "node-0": 0.0,       # Reference clock
        "node-1": 0.5,       # 500ms ahead
        "node-2": -0.3,      # 300ms behind
    }

    window_size = 1.0
    limit = 10

    print("Node clock offsets:")
    for node, offset in node_offsets.items():
        print(f"  {node}: {offset:+.1f}s")

    # At the same real moment, each node thinks it's in a different window
    real_time = 100.5
    print(f"\nReal time: {real_time}")
    for node, offset in node_offsets.items():
        perceived = real_time + offset
        window_id = int(perceived / window_size)
        print(f"  {node}: perceived={perceived}, window={window_id}")

    print(f"\nImpact:")
    print(f"  node-1 and node-2 disagree on window by {0.5 + 0.3:.1f}s")
    print(f"  Requests near window boundaries may be counted in wrong window")
    print(f"  Max effective rate: {limit * 2} (2x burst at boundary with skew)")
    print(f"\nMitigation:")
    print(f"  1. Use NTP with tight synchronization (<10ms)")
    print(f"  2. Use sliding window (reduces boundary effect)")
    print(f"  3. Build clock skew tolerance into limits (set to 0.9 * desired)")


analyze_clock_skew_impact()
```

### 8.2 Partition Tolerance

```python
def analyze_partition_impact():
    """Analyze rate limiting during network partitions."""
    print("=== Network Partition Impact ===\n")

    scenarios = [
        {
            "name": "Redis unreachable",
            "strategy": "Local fallback with conservative limit",
            "risk": "Over-limiting (lost capacity) or under-limiting (no coordination)",
            "recommendation": "Use local bucket at 1/N rate as fallback",
        },
        {
            "name": "Partial partition (some nodes can reach Redis)",
            "strategy": "Nodes that can reach Redis rate-limit normally",
            "risk": "Unfair: reachable nodes are rate-limited, others aren't",
            "recommendation": "Track last successful sync; degrade gracefully",
        },
        {
            "name": "Full network partition",
            "strategy": "Each partition operates independently",
            "risk": "Each partition allows full rate → 2x total during partition",
            "recommendation": "Accept over-limit; set alerts for anomalies",
        },
    ]

    for s in scenarios:
        print(f"  {s['name']}:")
        print(f"    Strategy: {s['strategy']}")
        print(f"    Risk: {s['risk']}")
        print(f"    Recommendation: {s['recommendation']}")
        print()


analyze_partition_impact()
```

---

## 9. Production Patterns

### 9.1 Rate Limiting Architecture

```python
def production_patterns():
    """Describe production rate limiting patterns."""
    print("=== Production Rate Limiting Patterns ===\n")

    patterns = [
        {
            "name": "API Gateway Rate Limiting",
            "where": "Edge (API gateway / load balancer)",
            "why": "Protect backend from external abuse",
            "how": "Redis + sliding window per API key",
            "examples": "Kong, AWS API Gateway, Nginx",
        },
        {
            "name": "Application-Level Rate Limiting",
            "where": "Within the application code",
            "why": "Business-logic-aware limits",
            "how": "Token bucket per user/tenant",
            "examples": "Stripe API, GitHub API",
        },
        {
            "name": "Service Mesh Rate Limiting",
            "where": "Sidecar proxy (Envoy)",
            "why": "Protect internal services from each other",
            "how": "Local token bucket + global rate limit service",
            "examples": "Istio, Lyft Ratelimit",
        },
        {
            "name": "Database Rate Limiting",
            "where": "Database proxy / connection pool",
            "why": "Prevent query overload",
            "how": "Admission control + queue",
            "examples": "PgBouncer, ProxySQL",
        },
    ]

    for p in patterns:
        print(f"  {p['name']}:")
        print(f"    Where: {p['where']}")
        print(f"    Why: {p['why']}")
        print(f"    How: {p['how']}")
        print(f"    Examples: {p['examples']}")
        print()


production_patterns()
```

---

## 10. Summary and Key Takeaways

### Rate Limiting Algorithm Selection

> **RATE LIMITING DECISION TREE**
>
> Need exact limits?
>   Yes → Sliding Window Log (O(N) memory)
>   No → Need burst tolerance?
>     Yes → Token Bucket (configurable burst)
>     No → Sliding Window Counter (approximate, O(1) memory)
>
> Distributed?
>   Single node → Local algorithm
>   Multi-node → Redis Lua script or distributed token bucket with sync

### Key Principles

1. **Token bucket is the default choice**: Simple, supports bursts, O(1) memory.
2. **Redis Lua scripts provide atomicity**: Critical for correctness across multiple API servers.
3. **Accept approximate limits in distributed settings**: Exact limits require synchronization on every request.
4. **Multi-tier limits catch different abuse patterns**: Per-second catches bursts, per-hour catches sustained abuse.
5. **Always have a fallback**: If Redis is down, degrade to local rate limiting rather than dropping all requests.

---

## 11. Practice Problems

### Problem 1: Algorithm Comparison

Implement all five rate limiting algorithms and benchmark them under a workload of 10,000 requests/second. Compare memory usage, accuracy (% deviation from target), and CPU overhead.

### Problem 2: Distributed Challenge

Design a rate limiter for a 10-node cluster with 1000 req/s global limit. The limiter must handle Redis failures by falling back to local limiting. Calculate the maximum over-limit during a 30-second Redis outage.

### Problem 3: Fairness

A multi-tenant API serves 100 tenants with different rate limits (10-10000 req/s). Design a fair queuing system that prevents a high-rate tenant from starving low-rate tenants.

### Problem 4: Implementation Challenge

Build a complete distributed rate limiter with:
- Token bucket algorithm with Redis backend
- Sliding window for per-minute limits
- Automatic fallback to local limiting on Redis failure
- HTTP headers: X-RateLimit-Limit, X-RateLimit-Remaining, Retry-After

### Problem 5: Cost Analysis

A cloud rate limiter uses Redis for centralized counting. Each rate limit check costs 1 Redis call (0.5ms latency). With 100,000 req/s, calculate: total Redis operations, Redis cluster size needed, and added p50/p99 latency. Propose an optimization to reduce Redis calls by 80%.

---

## 12. References

1. Stripe Engineering (2017). "Rate Limiters and Load Shedders." Stripe Blog.
2. Cloudflare (2017). "How We Built Rate Limiting Capable of Scaling to Millions of Domains."
3. Redis documentation: "Rate Limiting with Redis."
4. Lyft Engineering (2017). "Ratelimit: A Generic Rate Limit Service."
5. Veeraraghavan, K. et al. (2016). "Maelstrom: Mitigating Datacenter-level Disasters." *OSDI*.
6. Google (2022). "Rate Limiting Strategies and Techniques." Cloud Architecture Center.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 4. O'Reilly Media.

---

[Next: Lesson 24 — Event Sourcing and CQRS](./24_Event_Sourcing_CQRS.md)
