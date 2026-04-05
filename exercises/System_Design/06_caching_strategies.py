"""
Exercises for Lesson 06: Caching Strategies
Topic: System_Design

Solutions to practice problems and hands-on exercises.
Covers cache patterns, TTL configuration, cache avalanche prevention,
cache warming, and stampede prevention.
"""

import time
import random
import math
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Any
import threading


# === Exercise 1: Caching Strategy Selection ===
# Problem: Choose an appropriate caching pattern for scenarios.

def exercise_1():
    """Caching strategy selection for different scenarios."""
    scenarios = [
        {
            "scenario": "Read-heavy service (read:write = 100:1)",
            "choice": "Cache-Aside (Lazy Loading)",
            "reason": "Check cache first on read, query DB on cache miss, "
                      "then populate cache. Optimized for read-heavy workloads. "
                      "Only caches data that's actually requested.",
        },
        {
            "scenario": "Write-heavy log system",
            "choice": "Write-Behind (Write-Back)",
            "reason": "Write to cache quickly and respond immediately. "
                      "Batch save to DB asynchronously. "
                      "Maximizes write performance at cost of potential data loss.",
        },
        {
            "scenario": "Inventory management requiring strong consistency",
            "choice": "Write-Through",
            "reason": "Update cache and DB simultaneously on every write. "
                      "Always consistent inventory count. Prevents overselling. "
                      "Higher write latency is acceptable for correctness.",
        },
        {
            "scenario": "Simple web application",
            "choice": "Cache-Aside",
            "reason": "Simplest implementation. Works even if cache fails "
                      "(degrades to DB-only). Most versatile pattern.",
        },
    ]

    print("Caching Strategy Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {s['scenario']}")
        print(f"   Choice: {s['choice']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 2: TTL Configuration ===
# Problem: Set appropriate TTL for different data types.

def exercise_2():
    """TTL configuration for different data types."""
    data_types = [
        ("User profile (changed several times a day)", "1-4 hours",
         "Low change frequency. Use with explicit invalidation on updates."),
        ("Product price (real-time fluctuation)", "1-5 minutes or no cache",
         "Price accuracy important. Short TTL or event-based invalidation."),
        ("Country list (rarely changes)", "24 hours to 7 days",
         "Rarely changes. Long TTL maximizes cache efficiency."),
        ("Real-time stock price", "Do NOT cache",
         "Real-time accuracy critical. Stale data is financially risky."),
    ]

    print("TTL Configuration Guide:")
    print("=" * 60)
    for i, (data, ttl, reason) in enumerate(data_types, 1):
        label = chr(96 + i)
        print(f"\n{label}) {data}")
        print(f"   TTL: {ttl}")
        print(f"   Reason: {reason}")


# === Exercise 3: Cache Avalanche Prevention ===
# Problem: All cache expires at midnight, server becomes slow.
# Diagnosis: Cache Avalanche

class SimulatedDB:
    """Simulated database with configurable latency."""
    def __init__(self, latency_ms=50):
        self.latency_ms = latency_ms
        self.query_count = 0

    def get(self, key):
        self.query_count += 1
        # Simulate DB latency
        return f"db_value_{key}"


class CacheWithJitter:
    """Cache implementation with TTL jitter to prevent avalanche."""

    def __init__(self, db, base_ttl=86400, jitter_range=3600):
        self.db = db
        self.base_ttl = base_ttl
        self.jitter_range = jitter_range
        self.store = {}  # key -> (value, expiry_time)

    def get(self, key, current_time=None):
        if current_time is None:
            current_time = time.time()

        if key in self.store:
            value, expiry = self.store[key]
            if current_time < expiry:
                return value, "HIT"

        # Cache miss
        value = self.db.get(key)
        jitter = random.randint(-self.jitter_range, self.jitter_range)
        ttl = self.base_ttl + jitter
        self.store[key] = (value, current_time + ttl)
        return value, "MISS"


def exercise_3():
    """Cache avalanche prevention with TTL jitter."""
    print("Cache Avalanche Prevention:")
    print("=" * 60)

    print("\nProblem: Every day at midnight, all cache expires -> server overload")
    print("Diagnosis: Cache Avalanche")
    print("\nSolution: TTL Jitter")

    # Simulate without jitter
    print("\n--- Without Jitter ---")
    db1 = SimulatedDB()
    cache_no_jitter = {}
    base_ttl = 100  # All expire at t=100

    # Populate 1000 keys at t=0
    for i in range(1000):
        cache_no_jitter[f"key_{i}"] = (f"val_{i}", base_ttl)

    # At t=100, all expire simultaneously
    db1.query_count = 0
    for i in range(1000):
        key = f"key_{i}"
        _, expiry = cache_no_jitter[key]
        if 100 >= expiry:  # Expired
            db1.get(key)  # Must hit DB

    print(f"  At t=100: All {db1.query_count} keys expired -> {db1.query_count} DB queries!")

    # Simulate with jitter
    print("\n--- With Jitter (±10) ---")
    random.seed(42)
    db2 = SimulatedDB()
    cache_with_jitter = {}

    for i in range(1000):
        jitter = random.randint(-10, 10)
        ttl = base_ttl + jitter
        cache_with_jitter[f"key_{i}"] = (f"val_{i}", ttl)

    # Check expirations per time unit around t=100
    print("  Expirations per time unit:")
    for t in range(90, 111):
        expired = sum(1 for _, (_, exp) in cache_with_jitter.items() if exp == t)
        bar = "#" * (expired // 2)
        if expired > 0:
            print(f"  t={t:3d}: {expired:3d} expirations {bar}")

    print("\n  Result: Expirations spread over 20-second window instead of one spike!")


# === Exercise 4: Cache Strategy Comparison ===
# Problem: Compare cache-aside, write-through, write-back, write-around strategies.

class CacheAside:
    """Cache-Aside (Lazy Loading) pattern."""
    def __init__(self, db):
        self.db = db
        self.cache = {}
        self.stats = {"db_reads": 0, "db_writes": 0, "cache_hits": 0}

    def read(self, key):
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        value = self.db.get(key)
        self.stats["db_reads"] += 1
        self.cache[key] = value
        return value

    def write(self, key, value):
        self.db.query_count += 1
        self.stats["db_writes"] += 1
        # Invalidate cache
        self.cache.pop(key, None)


class WriteThrough:
    """Write-Through pattern."""
    def __init__(self, db):
        self.db = db
        self.cache = {}
        self.stats = {"db_reads": 0, "db_writes": 0, "cache_hits": 0}

    def read(self, key):
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        value = self.db.get(key)
        self.stats["db_reads"] += 1
        self.cache[key] = value
        return value

    def write(self, key, value):
        self.cache[key] = value
        self.db.query_count += 1
        self.stats["db_writes"] += 1


class WriteBack:
    """Write-Back (Write-Behind) pattern."""
    def __init__(self, db, flush_interval=10):
        self.db = db
        self.cache = {}
        self.dirty = set()
        self.flush_interval = flush_interval
        self.write_count = 0
        self.stats = {"db_reads": 0, "db_writes": 0, "cache_hits": 0}

    def read(self, key):
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        value = self.db.get(key)
        self.stats["db_reads"] += 1
        self.cache[key] = value
        return value

    def write(self, key, value):
        self.cache[key] = value
        self.dirty.add(key)
        self.write_count += 1
        # Flush periodically
        if self.write_count % self.flush_interval == 0:
            self._flush()

    def _flush(self):
        for key in self.dirty:
            self.db.query_count += 1
            self.stats["db_writes"] += 1
        self.dirty.clear()


class WriteAround:
    """Write-Around pattern: writes go directly to DB, cache only on read."""
    def __init__(self, db):
        self.db = db
        self.cache = {}
        self.stats = {"db_reads": 0, "db_writes": 0, "cache_hits": 0}

    def read(self, key):
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        value = self.db.get(key)
        self.stats["db_reads"] += 1
        self.cache[key] = value
        return value

    def write(self, key, value):
        self.db.query_count += 1
        self.stats["db_writes"] += 1
        # Do NOT update cache - stale until next read miss


def exercise_4():
    """Cache strategy comparison with mixed workload."""
    print("Cache Strategy Comparison:")
    print("=" * 60)
    print("Workload: 70% reads, 30% writes, 1000 operations")

    random.seed(42)
    keys = [f"key_{i}" for i in range(20)]
    operations = []
    for _ in range(1000):
        key = random.choice(keys)
        if random.random() < 0.7:
            operations.append(("read", key))
        else:
            operations.append(("write", key))

    strategies = {
        "Cache-Aside": CacheAside(SimulatedDB()),
        "Write-Through": WriteThrough(SimulatedDB()),
        "Write-Back": WriteBack(SimulatedDB(), flush_interval=10),
        "Write-Around": WriteAround(SimulatedDB()),
    }

    print(f"\n{'Strategy':<16} | {'DB Reads':>10} | {'DB Writes':>10} | "
          f"{'Cache Hits':>11} | {'Hit Rate':>9}")
    print("-" * 72)

    for name, strategy in strategies.items():
        for op, key in operations:
            if op == "read":
                strategy.read(key)
            else:
                strategy.write(key, f"new_{key}")

        # Flush remaining dirty entries for write-back
        if hasattr(strategy, '_flush'):
            strategy._flush()

        total_reads = strategy.stats["cache_hits"] + strategy.stats["db_reads"]
        hit_rate = strategy.stats["cache_hits"] / total_reads if total_reads > 0 else 0
        print(f"{name:<16} | {strategy.stats['db_reads']:>10} | "
              f"{strategy.stats['db_writes']:>10} | "
              f"{strategy.stats['cache_hits']:>11} | {hit_rate:>8.1%}")


# === Exercise 5: Cache Stampede Prevention ===
# Problem: Prevent thundering herd when cache key expires.

class CacheWithStampedePrevention:
    """Cache with stampede prevention using lock-based and probabilistic methods."""

    def __init__(self, db, ttl=60):
        self.db = db
        self.ttl = ttl
        self.store = {}  # key -> (value, expiry)
        self.locks = {}  # key -> lock
        self.stats = {"db_reads": 0, "concurrent_waits": 0}

    def get_no_protection(self, key, current_time):
        """No stampede protection - all miss simultaneously."""
        if key in self.store:
            value, expiry = self.store[key]
            if current_time < expiry:
                return value, False
        # Miss -> DB read
        value = self.db.get(key)
        self.stats["db_reads"] += 1
        self.store[key] = (value, current_time + self.ttl)
        return value, True

    def get_with_lock(self, key, current_time):
        """Lock-based: only one reader fetches, others wait."""
        if key in self.store:
            value, expiry = self.store[key]
            if current_time < expiry:
                return value, False

        # Try to acquire lock
        if key not in self.locks:
            self.locks[key] = True  # "Acquired"
            value = self.db.get(key)
            self.stats["db_reads"] += 1
            self.store[key] = (value, current_time + self.ttl)
            del self.locks[key]
            return value, True
        else:
            # Another reader is fetching, wait and use cached value
            self.stats["concurrent_waits"] += 1
            if key in self.store:
                return self.store[key][0], False
            return None, False

    def get_with_early_expiry(self, key, current_time, beta=1.0):
        """Probabilistic early expiration: XFetch algorithm."""
        if key in self.store:
            value, expiry = self.store[key]
            # Probabilistically refresh before actual expiry
            delta = expiry - current_time
            if delta > 0:
                # Probability of refresh increases as expiry approaches
                if delta > self.ttl * 0.2:  # Not close to expiry
                    return value, False
                # Close to expiry: random chance of refresh
                if random.random() > beta * math.log(random.random() + 0.001):
                    return value, False

        # Refresh
        value = self.db.get(key)
        self.stats["db_reads"] += 1
        self.store[key] = (value, current_time + self.ttl)
        return value, True


def exercise_5():
    """Cache stampede prevention comparison."""
    print("Cache Stampede Prevention:")
    print("=" * 60)

    # Simulate 100 concurrent readers hitting the same expired key
    print("\nScenario: 100 concurrent reads on expired key 'hot_key'")

    for method_name, method_fn_name in [
        ("No Protection", "get_no_protection"),
        ("Lock-Based", "get_with_lock"),
        ("Early Expiration", "get_with_early_expiry"),
    ]:
        db = SimulatedDB()
        cache = CacheWithStampedePrevention(db, ttl=60)
        # Pre-populate the cache, expired at t=60
        cache.store["hot_key"] = ("old_value", 60)
        cache.stats = {"db_reads": 0, "concurrent_waits": 0}

        method = getattr(cache, method_fn_name)
        current_time = 61  # Just past expiry

        for _ in range(100):
            if method_fn_name == "get_with_early_expiry":
                method("hot_key", current_time, beta=0.5)
            else:
                method("hot_key", current_time)

        print(f"\n  {method_name}:")
        print(f"    DB reads: {cache.stats['db_reads']}")
        if cache.stats.get("concurrent_waits"):
            print(f"    Concurrent waits: {cache.stats['concurrent_waits']}")


# === Exercise 6: Cache-Control Headers ===
# Problem: Write Cache-Control headers for requirements.

def exercise_6():
    """Cache-Control header configuration."""
    print("Cache-Control Headers:")
    print("=" * 60)

    headers = [
        {
            "scenario": "Static image (versioned URL, 1 year cache)",
            "header": "Cache-Control: public, max-age=31536000, immutable",
            "reason": "Versioned URLs mean the content never changes at this URL. "
                      "Immutable tells browsers not to revalidate.",
        },
        {
            "scenario": "API response (CDN 10 min, browser 1 min)",
            "header": "Cache-Control: public, max-age=60, s-maxage=600",
            "reason": "max-age=60 for browser cache, s-maxage=600 for CDN/proxy cache. "
                      "CDN caches longer to reduce origin load.",
        },
        {
            "scenario": "Login status check API (no caching)",
            "header": "Cache-Control: no-store",
            "reason": "Sensitive authentication data must never be cached. "
                      "no-store prevents both browser and proxy caching.",
        },
    ]

    for i, h in enumerate(headers, 1):
        label = chr(96 + i)
        print(f"\n{label}) {h['scenario']}")
        print(f"   Header: {h['header']}")
        print(f"   Reason: {h['reason']}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Caching Strategy Selection ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: TTL Configuration ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Cache Avalanche Prevention ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Cache Strategy Comparison ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Cache Stampede Prevention ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Cache-Control Headers ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
