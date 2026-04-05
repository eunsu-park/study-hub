"""
Exercises for Lesson 23: Distributed Rate Limiting
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# === Exercise 1: Algorithm Benchmark ===
def exercise_1():
    """Benchmark all five rate limiting algorithms."""
    print("=== Exercise 1: Algorithm Benchmark ===\n")

    class TokenBucket:
        def __init__(self, rate, capacity):
            self.rate, self.capacity = rate, capacity
            self.tokens, self.last = capacity, time.time()
        def allow(self):
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    class FixedWindow:
        def __init__(self, limit):
            self.limit, self.count, self.window = limit, 0, 0
        def allow(self):
            w = int(time.time())
            if w != self.window:
                self.window, self.count = w, 0
            if self.count < self.limit:
                self.count += 1
                return True
            return False

    class SlidingLog:
        def __init__(self, limit):
            self.limit, self.timestamps = limit, []
        def allow(self):
            now = time.time()
            self.timestamps = [t for t in self.timestamps if t > now - 1]
            if len(self.timestamps) < self.limit:
                self.timestamps.append(now)
                return True
            return False

    class SlidingCounter:
        def __init__(self, limit):
            self.limit = limit
            self.prev_count, self.curr_count = 0, 0
            self.curr_window = 0
        def allow(self):
            w = int(time.time())
            if w != self.curr_window:
                self.prev_count = self.curr_count
                self.curr_count = 0
                self.curr_window = w
            progress = time.time() % 1.0
            est = self.prev_count * (1 - progress) + self.curr_count
            if est < self.limit:
                self.curr_count += 1
                return True
            return False

    class LeakyBucket:
        def __init__(self, rate):
            self.rate, self.last = rate, time.time()
            self.water = 0.0
        def allow(self):
            now = time.time()
            self.water = max(0, self.water - (now - self.last) * self.rate)
            self.last = now
            if self.water < 1.0:
                self.water += 1.0
                return True
            return False

    limit = 100
    algorithms = {
        "Token Bucket": TokenBucket(limit, limit * 2),
        "Fixed Window": FixedWindow(limit),
        "Sliding Log": SlidingLog(limit),
        "Sliding Counter": SlidingCounter(limit),
        "Leaky Bucket": LeakyBucket(limit),
    }

    for name, algo in algorithms.items():
        allowed = sum(1 for _ in range(200) if algo.allow())
        print(f"  {name:20s}: {allowed}/200 allowed")


exercise_1()


# === Exercise 2: Distributed with Redis Failure ===
def exercise_2():
    """Calculate over-limit during Redis outage."""
    print("\n=== Exercise 2: Redis Outage Analysis ===\n")

    num_nodes = 10
    global_limit = 1000  # req/s
    outage_duration = 30  # seconds

    # During outage: each node falls back to local limit
    local_limit = global_limit / num_nodes  # 100 req/s per node
    # Worst case: all traffic hits one node
    max_rate_during_outage = local_limit  # Only 100/s if traffic is balanced
    # If unbalanced: one node at 100/s, others idle
    # Total effective limit: still 1000/s (sum of local limits)
    # But if traffic is concentrated: only 100/s per hot node

    # With conservative fallback at 1/N rate:
    conservative_total = num_nodes * (global_limit / num_nodes)  # = 1000
    print(f"  Global limit: {global_limit} req/s across {num_nodes} nodes")
    print(f"  Local fallback: {local_limit} req/s per node")
    print(f"  Outage duration: {outage_duration}s")
    print(f"  Max over-limit: 0 (if traffic is evenly distributed)")
    print(f"  Under-limit risk: concentrated traffic → {local_limit} req/s (10x under)")
    print(f"  Total excess requests during outage: 0 (conservative approach)")
    print(f"  Total missed capacity: up to {(global_limit - local_limit) * outage_duration} requests")


exercise_2()


# === Exercise 3: Multi-Tenant Fairness ===
def exercise_3():
    """Design fair queuing for multi-tenant rate limiting."""
    print("\n=== Exercise 3: Multi-Tenant Fair Queuing ===\n")

    class FairRateLimiter:
        """Weighted fair queuing across tenants."""
        def __init__(self, total_capacity):
            self.total_capacity = total_capacity
            self.tenant_limits = {}
            self.tenant_usage = defaultdict(int)

        def set_limit(self, tenant, limit):
            self.tenant_limits[tenant] = limit

        def allow(self, tenant):
            limit = self.tenant_limits.get(tenant, 0)
            if self.tenant_usage[tenant] < limit:
                self.tenant_usage[tenant] += 1
                return True
            return False

        def reset(self):
            self.tenant_usage.clear()

    limiter = FairRateLimiter(total_capacity=10000)
    tenants = {f"tenant-{i}": random.choice([10, 100, 1000, 5000, 10000])
               for i in range(10)}

    for tenant, limit in tenants.items():
        limiter.set_limit(tenant, limit)

    # Simulate: high-rate tenant sends many requests
    for tenant, limit in sorted(tenants.items(), key=lambda x: x[1]):
        requests = limit * 2  # Each tenant tries 2x their limit
        allowed = sum(1 for _ in range(requests) if limiter.allow(tenant))
        print(f"  {tenant} (limit={limit:>5}): sent={requests:>5}, allowed={allowed:>5}")


exercise_3()


# === Exercise 4: HTTP Rate Limiter ===
def exercise_4():
    """Rate limiter with HTTP headers."""
    print("\n=== Exercise 4: HTTP Rate Limit Headers ===\n")

    class HTTPRateLimiter:
        def __init__(self, limit, window):
            self.limit, self.window = limit, window
            self.counts = defaultdict(int)
            self.windows = {}

        def check(self, client_id):
            now = int(time.time() / self.window)
            if self.windows.get(client_id) != now:
                self.windows[client_id] = now
                self.counts[client_id] = 0

            remaining = self.limit - self.counts[client_id]
            if remaining > 0:
                self.counts[client_id] += 1
                return {
                    "allowed": True,
                    "headers": {
                        "X-RateLimit-Limit": str(self.limit),
                        "X-RateLimit-Remaining": str(remaining - 1),
                        "X-RateLimit-Reset": str(int((now + 1) * self.window)),
                    }
                }
            return {
                "allowed": False,
                "status": 429,
                "headers": {
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(int(self.window)),
                }
            }

    rl = HTTPRateLimiter(limit=5, window=1)
    for i in range(7):
        result = rl.check("user-1")
        status = "200" if result["allowed"] else "429"
        print(f"  Request {i+1}: {status} | {result['headers']}")


exercise_4()


# === Exercise 5: Redis Cost Analysis ===
def exercise_5():
    """Analyze Redis cost for centralized rate limiting."""
    print("\n=== Exercise 5: Redis Cost Analysis ===\n")

    req_per_sec = 100000
    redis_latency_ms = 0.5
    redis_ops_per_check = 1

    total_ops = req_per_sec * redis_ops_per_check
    redis_throughput = 1000 / redis_latency_ms  # ops per second per connection
    connections_needed = total_ops / redis_throughput

    print(f"  Request rate: {req_per_sec:,} req/s")
    print(f"  Redis ops: {total_ops:,} ops/s")
    print(f"  Added p50 latency: {redis_latency_ms}ms")
    print(f"  Added p99 latency: ~{redis_latency_ms * 3}ms")
    print(f"  Redis connections needed: ~{connections_needed:.0f}")
    print(f"\n  Optimization (80% reduction):")
    print(f"    Use local token bucket with periodic Redis sync")
    print(f"    Sync every 100ms → {req_per_sec / 10:,.0f} Redis ops/s (10x reduction)")
    print(f"    Or use batch token prefetch → {req_per_sec / 100:,.0f} ops/s (100x reduction)")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
