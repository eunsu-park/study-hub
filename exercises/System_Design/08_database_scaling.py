"""
Exercises for Lesson 08: Database Scaling
Topic: System_Design

Solutions to practice problems and hands-on exercises.
Covers sharding strategies, shard key selection, hotspot detection,
and rebalancing.
"""

import hashlib
import random
import time
from collections import defaultdict
from bisect import bisect_right
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


# === Exercise 1: Choosing a Sharding Strategy ===
# Problem: Choose appropriate sharding strategy for services.

def exercise_1():
    """Sharding strategy selection."""
    scenarios = [
        {
            "service": "Chat app (messages per conversation room)",
            "strategy": "Hash(conversation_id)",
            "reason": "Messages in the same room go to the same shard. "
                      "Queries are per conversation, so co-location is efficient.",
        },
        {
            "service": "Log analysis system",
            "strategy": "Range(timestamp)",
            "reason": "Time range queries are the primary access pattern. "
                      "Easy to archive/delete old log shards. "
                      "Hot shard issue mitigated by rolling shards.",
        },
        {
            "service": "Global user service",
            "strategy": "Hash(user_id)",
            "reason": "Even distribution across shards. "
                      "Per-user queries are efficient. No geographic hotspots.",
        },
        {
            "service": "SaaS multi-tenant",
            "strategy": "tenant_id (directory-based)",
            "reason": "Complete isolation per tenant. No cross-tenant queries needed. "
                      "Large tenants can have dedicated shards.",
        },
    ]

    print("Sharding Strategy Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {s['service']}")
        print(f"   Strategy: {s['strategy']}")
        print(f"   Reason:   {s['reason']}")


# === Exercise 2: Shard Key Selection ===
# Problem: Choose shard key for e-commerce tables.

def exercise_2():
    """Shard key selection for e-commerce tables."""
    print("Shard Key Selection for E-Commerce:")
    print("=" * 60)

    tables = [
        {
            "table": "users (id, email, name)",
            "shard_key": "Hash(user_id)",
            "reason": "Primary key with even distribution.",
        },
        {
            "table": "orders (id, user_id, status, created_at)",
            "shard_key": "Hash(user_id)",
            "reason": "user_id is the main query condition. "
                      "Efficient per-user order lookups.",
        },
        {
            "table": "order_items (id, order_id, product_id, quantity)",
            "shard_key": "Hash(user_id)",
            "reason": "Co-locate with orders on the same shard. "
                      "Using order_id might place items on different shard than orders.",
        },
        {
            "table": "products (id, merchant_id, name, price)",
            "shard_key": "Hash(merchant_id)",
            "reason": "Query products by seller is common. "
                      "Seller's products stay on same shard.",
        },
    ]

    for t in tables:
        print(f"\n  {t['table']}")
        print(f"  Shard key: {t['shard_key']}")
        print(f"  Reason: {t['reason']}")

    print("\n  Cross-shard considerations:")
    print("  - Order -> Product info: Use cache layer")
    print("  - Seller order queries: Separate index/materialized view")


# === Exercise 3: Hotspot Resolution ===
# Problem: Comments concentrating on popular posts, overloading a shard.

def exercise_3():
    """Hotspot resolution strategies."""
    print("Hotspot Resolution: Popular Post Comments:")
    print("=" * 60)

    print("\nProblem: All comments for a viral post land on one shard")
    print()
    solutions = [
        ("Hot post caching",
         "Cache comments in Redis. Batch save to DB periodically. "
         "Absorbs most read traffic."),
        ("Salting",
         "post_123_0, post_123_1, ... Add random salt on write. "
         "Query all salts and merge on read."),
        ("Separate counters",
         "Manage comment counts in Redis only. "
         "Async DB updates for persistence."),
        ("Composite shard key",
         "Use (post_id, comment_id) composite. "
         "Same post's comments are distributed across shards."),
        ("Dedicated handling",
         "Detect popular posts (e.g., >1000 comments/hour). "
         "Route to dedicated cache layer or separate shard."),
    ]

    for i, (name, desc) in enumerate(solutions, 1):
        print(f"  {i}. {name}")
        print(f"     {desc}")
        print()

    # Demonstrate salting approach
    print("\n--- Salting Demo ---")
    num_shards = 4
    num_salt_buckets = 8
    comments_per_post = {"post_viral": 10000, "post_normal": 100}

    print("Without salting:")
    for post, count in comments_per_post.items():
        shard = hash(post) % num_shards
        print(f"  {post}: all {count} comments -> shard {shard}")

    print("\nWith salting (8 buckets):")
    for post, count in comments_per_post.items():
        distribution = defaultdict(int)
        for _ in range(count):
            salt = random.randint(0, num_salt_buckets - 1)
            salted_key = f"{post}_{salt}"
            shard = hash(salted_key) % num_shards
            distribution[shard] += 1
        for shard in sorted(distribution):
            print(f"  {post} -> shard {shard}: {distribution[shard]} comments")


# === Exercise 4: Rebalancing Plan ===
# Problem: Expand from 3 shards to 5 shards with zero downtime.

class ShardRouter:
    """Simple hash-based shard router with rebalancing support."""

    def __init__(self, num_shards):
        self.num_shards = num_shards
        self.shards = {i: {} for i in range(num_shards)}

    def get_shard(self, key):
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return h % self.num_shards

    def write(self, key, value):
        shard_id = self.get_shard(key)
        self.shards[shard_id][key] = value

    def read(self, key):
        shard_id = self.get_shard(key)
        return self.shards[shard_id].get(key)


def exercise_4():
    """Shard rebalancing from 3 to 5 shards."""
    print("Shard Rebalancing Plan (3 -> 5 shards):")
    print("=" * 60)

    # Phase 1: Initial state with 3 shards
    old_router = ShardRouter(3)
    num_keys = 10000

    for i in range(num_keys):
        old_router.write(f"key_{i}", f"value_{i}")

    print("\nPhase 1: Initial state (3 shards)")
    for shard_id, data in old_router.shards.items():
        print(f"  Shard {shard_id}: {len(data)} keys")

    # Phase 2: Calculate new mapping
    new_router = ShardRouter(5)
    keys_to_move = defaultdict(list)  # (old_shard, new_shard) -> keys

    for i in range(num_keys):
        key = f"key_{i}"
        old_shard = old_router.get_shard(key)
        new_shard = new_router.get_shard(key)
        if old_shard != new_shard:
            keys_to_move[(old_shard, new_shard)].append(key)

    total_moved = sum(len(keys) for keys in keys_to_move.values())
    print(f"\nPhase 2: Migration plan")
    print(f"  Total keys to move: {total_moved}/{num_keys} "
          f"({total_moved/num_keys:.1%})")

    move_summary = defaultdict(int)
    for (old, new), keys in keys_to_move.items():
        move_summary[old] += len(keys)

    for shard_id in sorted(move_summary):
        print(f"  Shard {shard_id} losing: {move_summary[shard_id]} keys")

    # Phase 3: Simulate migration
    print(f"\nPhase 3-7: Migration steps")
    steps = [
        "Phase 3: Prepare new shards (Shard 3, 4) - create schema",
        "Phase 4: Start dual write (new writes go to both old and new routing)",
        "Phase 5: Background migrate existing data (~{pct}% of keys move)",
        "Phase 6: Verify data counts and sample checksums",
        "Phase 7: Switch reads to new routing (gradual: 10% -> 50% -> 100%)",
        "Phase 8: Stop dual write, remove old routing",
        "Phase 9: Cleanup migrated data from old shards",
    ]
    for step in steps:
        if "{pct}" in step:
            step = step.format(pct=f"{total_moved/num_keys*100:.0f}")
        print(f"  {step}")

    # Verify new distribution
    print(f"\nFinal state (5 shards):")
    for i in range(num_keys):
        key = f"key_{i}"
        value = f"value_{i}"
        new_router.write(key, value)

    for shard_id, data in new_router.shards.items():
        print(f"  Shard {shard_id}: {len(data)} keys")


# === Exercise 5: Hot Shard Detection ===
# Problem: Build monitoring that detects hot shards.

class ShardLoadTracker:
    """Monitors request rates per shard and detects hotspots."""

    def __init__(self, num_shards, window_seconds=10):
        self.num_shards = num_shards
        self.window = window_seconds
        self.requests = defaultdict(list)  # shard_id -> [timestamps]
        self.alert_threshold = 2.0  # Alert if > 2x average

    def record_request(self, shard_id, timestamp):
        self.requests[shard_id].append(timestamp)

    def get_load(self, timestamp):
        """Get current load per shard within time window."""
        cutoff = timestamp - self.window
        loads = {}
        for shard_id in range(self.num_shards):
            recent = [t for t in self.requests[shard_id] if t > cutoff]
            loads[shard_id] = len(recent)
        return loads

    def detect_hotspot(self, timestamp):
        """Detect if any shard exceeds 2x average load."""
        loads = self.get_load(timestamp)
        if not loads:
            return None

        avg_load = sum(loads.values()) / len(loads)
        if avg_load == 0:
            return None

        hot_shards = {
            sid: load for sid, load in loads.items()
            if load > avg_load * self.alert_threshold
        }
        return hot_shards if hot_shards else None


def exercise_5():
    """Hot shard detection and mitigation."""
    print("Hot Shard Detection and Mitigation:")
    print("=" * 60)

    tracker = ShardLoadTracker(num_shards=4, window_seconds=10)
    random.seed(42)

    # Normal traffic for t=0-50
    for t in range(50):
        for _ in range(10):  # 10 requests per tick
            shard = random.randint(0, 3)
            tracker.record_request(shard, t)

    # Viral event: shard 1 gets 5x traffic for t=50-70
    for t in range(50, 70):
        for _ in range(10):  # Normal
            shard = random.randint(0, 3)
            tracker.record_request(shard, t)
        for _ in range(40):  # Extra load on shard 1
            tracker.record_request(1, t)

    # Check at various times
    print("\nLoad monitoring:")
    for t in [40, 55, 60, 65, 75]:
        loads = tracker.get_load(t)
        hot = tracker.detect_hotspot(t)
        avg = sum(loads.values()) / len(loads)
        print(f"\n  t={t:2d}: Loads={dict(loads)}, avg={avg:.0f}")
        if hot:
            for sid, load in hot.items():
                print(f"    ALERT: Shard {sid} is HOT! "
                      f"({load} requests, {load/avg:.1f}x average)")

    # Mitigation: split hot shard
    print("\n--- Mitigation: Split shard 1 into sub-shards ---")
    print("  Shard 1 split into: shard_1a and shard_1b")
    print("  Keys re-hashed with secondary hash within shard 1")
    print("  Result: Load redistributed evenly between sub-shards")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Sharding Strategy Selection ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Shard Key Selection ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Hotspot Resolution ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Shard Rebalancing ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Hot Shard Detection ===")
    print("=" * 60)
    exercise_5()

    print("\nAll exercises completed!")
