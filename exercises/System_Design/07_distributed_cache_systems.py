"""
Exercises for Lesson 07: Distributed Cache Systems
Topic: System_Design

Solutions to practice problems and hands-on exercises.
Covers Redis data structures, Sentinel vs Cluster, consistent hashing,
cache eviction policies, and replicated cache clusters.
"""

import hashlib
import random
import math
from collections import defaultdict, OrderedDict
from bisect import bisect_right
from typing import Dict, List, Optional, Any


# === Exercise 1: Redis Data Structure Selection ===
# Problem: Choose appropriate Redis data structure for requirements.

def exercise_1():
    """Redis data structure selection."""
    scenarios = [
        {
            "requirement": "Store user session (ID, name, permissions, etc.)",
            "structure": "Hash",
            "command": 'HSET session:abc123 user_id 123 name "John" role "admin"',
            "reason": "Hash stores field-value pairs. Perfect for structured session data "
                      "with O(1) field access.",
        },
        {
            "requirement": "Real-time game leaderboard",
            "structure": "Sorted Set (ZSET)",
            "command": 'ZADD leaderboard 1000 "user:123" 950 "user:456"',
            "reason": "Sorted Set maintains elements sorted by score. "
                      "ZRANGE/ZREVRANGE gives top-N in O(log N + M).",
        },
        {
            "requirement": "Store 10 recently viewed products",
            "structure": "List (with LTRIM)",
            "command": 'LPUSH user:123:recent "product:789"\nLTRIM user:123:recent 0 9',
            "reason": "List with LPUSH + LTRIM maintains a fixed-size recent items list.",
        },
        {
            "requirement": "List of posts user liked",
            "structure": "Set",
            "command": 'SADD user:123:likes "post:456" "post:789"',
            "reason": "Set provides O(1) membership check (SISMEMBER) "
                      "and prevents duplicates.",
        },
        {
            "requirement": "Chat room message queue",
            "structure": "List (as queue)",
            "command": 'RPUSH chat:room1 "{message...}"\nBLPOP chat:room1 0',
            "reason": "RPUSH/BLPOP provides FIFO queue behavior. "
                      "BLPOP blocks until message available (consumer pattern).",
        },
    ]

    print("Redis Data Structure Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {s['requirement']}")
        print(f"   Structure: {s['structure']}")
        print(f"   Command:   {s['command']}")
        print(f"   Reason:    {s['reason']}")


# === Exercise 2: Sentinel vs Cluster Selection ===
# Problem: Choose between Sentinel and Cluster.

def exercise_2():
    """Sentinel vs Cluster selection."""
    scenarios = [
        ("10GB data, high availability needed", "Sentinel",
         "Data size fits single server. Sentinel provides HA with simple config."),
        ("1TB data, horizontal scaling needed", "Cluster",
         "Large data needs distributed storage across multiple nodes."),
        ("Service with many complex transactions", "Sentinel",
         "Cluster has multi-key transaction limitations. Single master is easier."),
        ("Simple cache, 1M requests/second", "Cluster (or Memcached)",
         "Cluster distributes load across shards for high throughput."),
    ]

    print("Sentinel vs Cluster Selection:")
    print("=" * 60)
    for i, (scenario, choice, reason) in enumerate(scenarios, 1):
        label = chr(96 + i)
        print(f"\n{label}) {scenario}")
        print(f"   Choice: {choice}")
        print(f"   Reason: {reason}")


# === Exercise 3: Consistent Hashing ===
# Problem: Compare key redistribution with traditional vs consistent hashing.

def exercise_3():
    """Consistent hashing vs traditional hashing redistribution analysis."""
    print("Consistent Hashing Key Redistribution:")
    print("=" * 60)

    num_keys = 10000

    # Traditional hashing: key % N
    print("\n--- Traditional Hashing ---")
    old_servers = 4
    new_servers = 5
    moved = 0
    for key in range(num_keys):
        old_assignment = key % old_servers
        new_assignment = key % new_servers
        if old_assignment != new_assignment:
            moved += 1
    theoretical = old_servers / new_servers  # N/(N+1) of keys move
    print(f"  4 servers -> 5 servers")
    print(f"  Keys moved: {moved}/{num_keys} ({moved/num_keys:.1%})")
    print(f"  Theoretical: {theoretical:.1%}")

    # Consistent hashing
    print("\n--- Consistent Hashing ---")
    ring = ConsistentHashRing(virtual_nodes=150)
    for i in range(4):
        ring.add_node(f"server_{i}")

    # Map keys to servers
    old_mapping = {}
    for key in range(num_keys):
        old_mapping[key] = ring.get_node(str(key))

    # Add a 5th server
    ring.add_node("server_4")

    moved = 0
    for key in range(num_keys):
        new_server = ring.get_node(str(key))
        if new_server != old_mapping[key]:
            moved += 1
    theoretical_consistent = 1 / new_servers  # ~K/N keys move
    print(f"  4 servers -> 5 servers")
    print(f"  Keys moved: {moved}/{num_keys} ({moved/num_keys:.1%})")
    print(f"  Theoretical: ~{theoretical_consistent:.1%}")
    print(f"\n  Consistent hashing moves {(old_servers/new_servers - moved/num_keys)*100:.0f}% fewer keys!")


class ConsistentHashRing:
    """Consistent hash ring with virtual nodes."""

    def __init__(self, virtual_nodes=150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}          # hash -> node_name
        self.sorted_hashes = [] # sorted list of hashes
        self.nodes = set()

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def add_node(self, node_name):
        self.nodes.add(node_name)
        for i in range(self.virtual_nodes):
            vnode_key = f"{node_name}:vn{i}"
            h = self._hash(vnode_key)
            self.ring[h] = node_name
        self.sorted_hashes = sorted(self.ring.keys())

    def remove_node(self, node_name):
        self.nodes.discard(node_name)
        self.ring = {h: n for h, n in self.ring.items() if n != node_name}
        self.sorted_hashes = sorted(self.ring.keys())

    def get_node(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect_right(self.sorted_hashes, h) % len(self.sorted_hashes)
        return self.ring[self.sorted_hashes[idx]]


# === Exercise 4: Weighted Consistent Hashing ===
# Problem: Give "large" server 2x virtual nodes, verify ~2x keys.

def exercise_4():
    """Weighted consistent hashing."""
    print("Weighted Consistent Hashing:")
    print("=" * 60)

    # Standard: all servers get 150 virtual nodes
    ring = ConsistentHashRing(virtual_nodes=150)
    for i in range(4):
        ring.add_node(f"small_{i}")

    # Large server gets 2x virtual nodes
    for i in range(300):  # 2x virtual nodes
        vnode_key = f"large_0:vn{i}"
        h = ring._hash(vnode_key)
        ring.ring[h] = "large_0"
    ring.nodes.add("large_0")
    ring.sorted_hashes = sorted(ring.ring.keys())

    # Map 10000 keys
    distribution = defaultdict(int)
    for key in range(10000):
        node = ring.get_node(str(key))
        distribution[node] += 1

    total_weight = 4 * 150 + 300  # 4 small + 1 large
    print(f"\nNode distribution (10,000 keys):")
    for node in sorted(distribution):
        count = distribution[node]
        weight = 300 if "large" in node else 150
        expected_pct = weight / total_weight * 100
        actual_pct = count / 10000 * 100
        bar = "#" * int(actual_pct)
        print(f"  {node:>10} (w={weight:3d}): {count:>5} keys "
              f"({actual_pct:5.1f}%, expected ~{expected_pct:.1f}%)  [{bar}]")

    large_keys = distribution.get("large_0", 0)
    avg_small = sum(v for k, v in distribution.items() if "small" in k) / 4
    print(f"\n  Large server: {large_keys} keys")
    print(f"  Avg small server: {avg_small:.0f} keys")
    print(f"  Ratio: {large_keys / avg_small:.2f}x (expected ~2.0x)")


# === Exercise 5: Cache Eviction Policy Comparison ===
# Problem: Compare LRU, LFU, and Random eviction policies.

class LRUCache:
    """Least Recently Used cache."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value


class LFUCache:
    """Least Frequently Used cache (simplified)."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.freq = defaultdict(int)
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.freq[key] += 1
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.freq[key] += 1
        else:
            if len(self.cache) >= self.capacity:
                # Evict least frequently used
                min_freq_key = min(self.cache, key=lambda k: self.freq[k])
                del self.cache[min_freq_key]
                del self.freq[min_freq_key]
            self.freq[key] = 1
        self.cache[key] = value


class RandomCache:
    """Random eviction cache."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key not in self.cache and len(self.cache) >= self.capacity:
            evict_key = random.choice(list(self.cache.keys()))
            del self.cache[evict_key]
        self.cache[key] = value


def zipf_distribution(n_keys, n_requests, s=1.0):
    """Generate Zipf-distributed access pattern."""
    # Zipf: probability of key k ~ 1/k^s
    weights = [1.0 / (k ** s) for k in range(1, n_keys + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]

    keys = list(range(n_keys))
    return random.choices(keys, weights=probs, k=n_requests)


def exercise_5():
    """Cache eviction policy comparison."""
    print("Cache Eviction Policy Comparison:")
    print("=" * 60)

    capacity = 100
    n_keys = 500
    n_requests = 10000

    # Zipf distribution (80/20 rule)
    print("\n--- Zipf Distribution (20% keys get 80% access) ---")
    random.seed(42)
    zipf_accesses = zipf_distribution(n_keys, n_requests, s=1.0)

    for CacheClass, name in [(LRUCache, "LRU"), (LFUCache, "LFU"),
                              (RandomCache, "Random")]:
        cache = CacheClass(capacity)
        for key in zipf_accesses:
            if cache.get(key) is None:
                cache.put(key, f"value_{key}")

        total = cache.hits + cache.misses
        hit_rate = cache.hits / total
        print(f"  {name:>8}: Hit rate = {hit_rate:.1%} "
              f"(hits={cache.hits}, misses={cache.misses})")

    # Uniform distribution
    print("\n--- Uniform Distribution ---")
    random.seed(42)
    uniform_accesses = [random.randint(0, n_keys - 1) for _ in range(n_requests)]

    for CacheClass, name in [(LRUCache, "LRU"), (LFUCache, "LFU"),
                              (RandomCache, "Random")]:
        cache = CacheClass(capacity)
        for key in uniform_accesses:
            if cache.get(key) is None:
                cache.put(key, f"value_{key}")

        total = cache.hits + cache.misses
        hit_rate = cache.hits / total
        print(f"  {name:>8}: Hit rate = {hit_rate:.1%} "
              f"(hits={cache.hits}, misses={cache.misses})")

    print("\nAnalysis:")
    print("  - LFU benefits most from skewed access patterns (Zipf)")
    print("  - LRU is a good general-purpose choice")
    print("  - Random performs worst but has minimal overhead")
    print("  - Under uniform access, all policies converge (capacity/n_keys)")


# === Exercise 6: Social Media Follow Feature (Redis Design) ===
# Problem: Design follow feature using Redis Sets.

class RedisSimulator:
    """Simplified Redis simulator for Set operations."""
    def __init__(self):
        self.store = defaultdict(set)

    def sadd(self, key, *members):
        self.store[key].update(members)

    def srem(self, key, *members):
        self.store[key] -= set(members)

    def smembers(self, key):
        return self.store[key]

    def scard(self, key):
        return len(self.store[key])

    def sismember(self, key, member):
        return member in self.store[key]

    def sinter(self, *keys):
        if not keys:
            return set()
        result = self.store[keys[0]].copy()
        for key in keys[1:]:
            result &= self.store[key]
        return result


def exercise_6():
    """Social media follow feature using Redis."""
    print("Social Media Follow Feature (Redis Design):")
    print("=" * 60)

    redis = RedisSimulator()

    # A follows B
    def follow(follower, followee):
        redis.sadd(f"user:{follower}:following", followee)
        redis.sadd(f"user:{followee}:followers", follower)
        print(f"  {follower} followed {followee}")

    def unfollow(follower, followee):
        redis.srem(f"user:{follower}:following", followee)
        redis.srem(f"user:{followee}:followers", follower)
        print(f"  {follower} unfollowed {followee}")

    # Build relationships
    follow("Alice", "Bob")
    follow("Alice", "Charlie")
    follow("Alice", "Dave")
    follow("Bob", "Charlie")
    follow("Bob", "Eve")
    follow("Charlie", "Dave")
    follow("Dave", "Alice")

    print(f"\n  Alice's following: {redis.smembers('user:Alice:following')}")
    print(f"  Bob's following: {redis.smembers('user:Bob:following')}")
    print(f"  Charlie's followers: {redis.smembers('user:Charlie:followers')}")
    print(f"  Charlie's follower count: {redis.scard('user:Charlie:followers')}")

    # Common following
    common = redis.sinter("user:Alice:following", "user:Bob:following")
    print(f"\n  Common following (Alice & Bob): {common}")

    # Is Alice following Bob?
    print(f"  Alice follows Bob? {redis.sismember('user:Alice:following', 'Bob')}")
    print(f"  Alice follows Eve? {redis.sismember('user:Alice:following', 'Eve')}")

    # Unfollow
    unfollow("Alice", "Dave")
    print(f"  Alice's following after unfollow: {redis.smembers('user:Alice:following')}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Redis Data Structure Selection ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Sentinel vs Cluster ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Consistent Hashing Redistribution ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Weighted Consistent Hashing ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Cache Eviction Policy Comparison ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Social Media Follow Feature ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
