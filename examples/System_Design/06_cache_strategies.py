"""
Cache Strategies

Demonstrates:
- Cache-aside (lazy loading)
- Read-through
- Write-through
- Write-back (write-behind)
- Cache eviction policies (LRU, LFU, TTL)

Theory:
- Cache-aside: Application checks cache first; on miss, reads from DB
  and populates cache. Simple but risks stale data.
- Read-through: Cache itself loads from DB on miss. Transparent to app.
- Write-through: Writes go to both cache and DB synchronously.
  Consistent but higher write latency.
- Write-back: Writes go to cache only; async flush to DB.
  Fast writes but risk data loss on crash.

Adapted from System Design Lesson 06.
"""

import time
from collections import OrderedDict, defaultdict
from typing import Any


# ── Simulated Database ─────────────────────────────────────────────────

class SimDB:
    """Simulated database with latency."""

    def __init__(self, latency_ms: float = 50.0):
        self.store: dict[str, Any] = {}
        self.latency_ms = latency_ms
        self.reads = 0
        self.writes = 0

    def get(self, key: str) -> Any | None:
        self.reads += 1
        return self.store.get(key)

    def put(self, key: str, value: Any) -> None:
        self.writes += 1
        self.store[key] = value

    def stats(self) -> str:
        return f"DB reads={self.reads}, writes={self.writes}"


# ── LRU Cache ──────────────────────────────────────────────────────────

# Why: OrderedDict gives us O(1) move-to-end on access and O(1) popitem from
# the front — exactly the two operations LRU needs. This avoids implementing
# a custom doubly-linked list + hash map combination from scratch.
class LRUCache:
    """LRU eviction cache."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[bool, Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True, self.cache[key]
        self.misses += 1
        return False, None

    def put(self, key: str, value: Any) -> str | None:
        evicted = None
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                evicted_key, _ = self.cache.popitem(last=False)
                evicted = evicted_key
        self.cache[key] = value
        return evicted

    def remove(self, key: str) -> None:
        self.cache.pop(key, None)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> str:
        return (f"Cache hits={self.hits}, misses={self.misses}, "
                f"hit_rate={self.hit_rate():.1%}")


# ── Cache-Aside ────────────────────────────────────────────────────────

# Why: Cache-aside gives the application full control over what gets cached.
# This is the most common caching pattern (used by Memcached, Redis clients)
# because it works with any database and requires no special DB integration.
class CacheAside:
    """Cache-aside (lazy loading) pattern.

    App checks cache → miss → read from DB → populate cache.
    """

    def __init__(self, cache: LRUCache, db: SimDB):
        self.cache = cache
        self.db = db

    def read(self, key: str) -> Any | None:
        hit, value = self.cache.get(key)
        if hit:
            return value
        # Cache miss: read from DB
        value = self.db.get(key)
        if value is not None:
            self.cache.put(key, value)
        return value

    def write(self, key: str, value: Any) -> None:
        # Why: Invalidate (delete) rather than update the cache entry on write.
        # This avoids a race condition where a concurrent read could cache a
        # stale value between the DB write and cache update.
        self.db.put(key, value)
        self.cache.remove(key)


# ── Read-Through ───────────────────────────────────────────────────────

class ReadThrough:
    """Read-through cache.

    Cache automatically loads from DB on miss.
    """

    def __init__(self, cache: LRUCache, db: SimDB):
        self.cache = cache
        self.db = db

    def read(self, key: str) -> Any | None:
        hit, value = self.cache.get(key)
        if hit:
            return value
        # Cache loads from DB transparently
        value = self.db.get(key)
        if value is not None:
            self.cache.put(key, value)
        return value


# ── Write-Through ──────────────────────────────────────────────────────

# Why: Write-through guarantees cache and DB are always in sync, eliminating
# stale reads. The cost is higher write latency (two synchronous writes).
# Best when reads vastly outnumber writes and consistency is critical.
class WriteThrough:
    """Write-through cache.

    Writes go to both cache and DB synchronously.
    """

    def __init__(self, cache: LRUCache, db: SimDB):
        self.cache = cache
        self.db = db

    def read(self, key: str) -> Any | None:
        hit, value = self.cache.get(key)
        if hit:
            return value
        value = self.db.get(key)
        if value is not None:
            self.cache.put(key, value)
        return value

    def write(self, key: str, value: Any) -> None:
        # Write to both
        self.db.put(key, value)
        self.cache.put(key, value)


# ── Write-Back ─────────────────────────────────────────────────────────

# Why: Write-back absorbs write bursts entirely in cache, deferring DB writes.
# This dramatically reduces DB load for write-heavy workloads (e.g., counters,
# session updates). The risk: data loss if the cache crashes before flushing.
class WriteBack:
    """Write-back (write-behind) cache.

    Writes go to cache only; dirty entries flushed to DB later.
    """

    def __init__(self, cache: LRUCache, db: SimDB):
        self.cache = cache
        self.db = db
        # Why: The dirty set tracks which cache entries have unflushed writes.
        # On eviction, only dirty entries need DB writes — clean entries can be
        # silently discarded since the DB already has their current value.
        self.dirty: set[str] = set()
        self.flush_count = 0

    def read(self, key: str) -> Any | None:
        hit, value = self.cache.get(key)
        if hit:
            return value
        value = self.db.get(key)
        if value is not None:
            self.cache.put(key, value)
        return value

    def write(self, key: str, value: Any) -> None:
        # Write to cache only, mark dirty
        evicted = self.cache.put(key, value)
        self.dirty.add(key)
        # If eviction happened and it was dirty, flush it
        if evicted and evicted in self.dirty:
            self._flush_key(evicted)

    def _flush_key(self, key: str) -> None:
        if key in self.dirty:
            hit, value = self.cache.get(key)
            if hit:
                self.db.put(key, value)
            self.dirty.discard(key)
            self.flush_count += 1

    def flush_all(self) -> int:
        """Flush all dirty entries to DB."""
        flushed = 0
        for key in list(self.dirty):
            hit, value = self.cache.get(key)
            if hit:
                self.db.put(key, value)
                flushed += 1
            self.dirty.discard(key)
        self.flush_count += flushed
        return flushed


# ── TTL Cache ──────────────────────────────────────────────────────────

# Why: TTL-based expiration bounds the staleness of cached data — a pragmatic
# middle ground when you cannot afford cache invalidation complexity but need
# reasonably fresh data (e.g., DNS caching, API response caching).
class TTLCache:
    """Cache with Time-To-Live expiration."""

    def __init__(self, capacity: int, ttl_seconds: float):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache: dict[str, tuple[Any, float]] = {}
        self.hits = 0
        self.misses = 0
        self.expirations = 0

    def get(self, key: str, current_time: float) -> tuple[bool, Any]:
        if key in self.cache:
            value, expiry = self.cache[key]
            if current_time < expiry:
                self.hits += 1
                return True, value
            # Expired
            del self.cache[key]
            self.expirations += 1
        self.misses += 1
        return False, None

    def put(self, key: str, value: Any, current_time: float) -> None:
        # Evict expired first
        expired = [k for k, (_, exp) in self.cache.items()
                   if current_time >= exp]
        for k in expired:
            del self.cache[k]
            self.expirations += 1

        if len(self.cache) >= self.capacity and key not in self.cache:
            # Evict oldest
            oldest = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest]

        self.cache[key] = (value, current_time + self.ttl)


# ── Demos ──────────────────────────────────────────────────────────────

def demo_cache_aside():
    print("=" * 60)
    print("CACHE-ASIDE (LAZY LOADING)")
    print("=" * 60)

    db = SimDB()
    for i in range(20):
        db.put(f"key-{i}", f"value-{i}")
    db.reads = db.writes = 0  # reset stats

    cache = LRUCache(capacity=5)
    ca = CacheAside(cache, db)

    print(f"\n  DB has 20 entries, cache capacity = 5")

    # First reads: all misses
    print(f"\n  First pass (5 unique keys):")
    for i in range(5):
        val = ca.read(f"key-{i}")
        print(f"    read key-{i} → {val}")
    print(f"    {cache.stats()}")
    print(f"    {db.stats()}")

    # Second reads: all hits
    print(f"\n  Second pass (same 5 keys):")
    db.reads = 0
    for i in range(5):
        ca.read(f"key-{i}")
    print(f"    {cache.stats()}")
    print(f"    {db.stats()} (no new DB reads!)")

    # Write invalidates cache
    print(f"\n  Write key-0 = 'updated':")
    ca.write("key-0", "updated")
    hit, _ = cache.get("key-0")
    print(f"    key-0 in cache? {hit} (invalidated)")
    val = ca.read("key-0")
    print(f"    read key-0 → {val} (re-fetched from DB)")


def demo_write_strategies():
    print("\n" + "=" * 60)
    print("WRITE-THROUGH vs WRITE-BACK")
    print("=" * 60)

    # Write-through
    db_wt = SimDB()
    cache_wt = LRUCache(capacity=10)
    wt = WriteThrough(cache_wt, db_wt)

    # Write-back
    db_wb = SimDB()
    cache_wb = LRUCache(capacity=10)
    wb = WriteBack(cache_wb, db_wb)

    # Perform 20 writes
    print(f"\n  20 writes to 10 keys (each key written twice):")
    for round_num in range(2):
        for i in range(10):
            wt.write(f"key-{i}", f"v{round_num}-{i}")
            wb.write(f"key-{i}", f"v{round_num}-{i}")

    print(f"\n  {'Strategy':<20} {'DB Writes':>10} {'Dirty':>6}")
    print(f"  {'-'*20} {'-'*10} {'-'*6}")
    print(f"  {'Write-Through':<20} {db_wt.writes:>10} {'0':>6}")
    print(f"  {'Write-Back':<20} {db_wb.writes:>10} {len(wb.dirty):>6}")

    # Flush write-back
    flushed = wb.flush_all()
    print(f"\n  After flush: {flushed} entries written to DB")
    print(f"  Write-Back total DB writes: {db_wb.writes}")

    print(f"\n  Write-through: consistent but {db_wt.writes} DB writes")
    print(f"  Write-back: only {db_wb.writes} DB writes (batched)")

    # Both have same data
    print(f"\n  Data consistency check:")
    for i in range(3):
        wt_val = wt.read(f"key-{i}")
        wb_val = wb.read(f"key-{i}")
        print(f"    key-{i}: WT={wt_val}, WB={wb_val}, match={wt_val == wb_val}")


def demo_ttl():
    print("\n" + "=" * 60)
    print("TTL (TIME-TO-LIVE) CACHE")
    print("=" * 60)

    cache = TTLCache(capacity=5, ttl_seconds=2.0)
    print(f"\n  Capacity=5, TTL=2 seconds")

    # Populate at t=0
    for i in range(5):
        cache.put(f"key-{i}", f"val-{i}", current_time=0.0)

    # Read at t=1 (all valid)
    print(f"\n  At t=1.0s (within TTL):")
    for i in range(5):
        hit, val = cache.get(f"key-{i}", current_time=1.0)
        status = "HIT" if hit else "MISS"
        print(f"    key-{i}: {status} → {val}")
    print(f"    Hits={cache.hits}, Misses={cache.misses}")

    # Read at t=2.5 (all expired)
    print(f"\n  At t=2.5s (past TTL):")
    for i in range(5):
        hit, val = cache.get(f"key-{i}", current_time=2.5)
        status = "HIT" if hit else "EXPIRED"
        print(f"    key-{i}: {status}")
    print(f"    Expirations={cache.expirations}")


def demo_eviction_comparison():
    print("\n" + "=" * 60)
    print("CACHE EVICTION: LRU BEHAVIOR")
    print("=" * 60)

    cache = LRUCache(capacity=4)
    print(f"\n  Capacity=4, access pattern: A B C D A B E A B C D E")

    access_pattern = list("ABCDABEABCDE")
    print(f"\n  {'Access':>8}  {'Hit/Miss':>9}  Cache State")
    print(f"  {'-'*8}  {'-'*9}  {'-'*25}")

    for key in access_pattern:
        hit, _ = cache.get(key)
        if not hit:
            cache.put(key, key)
        state = list(cache.cache.keys())
        status = "HIT" if hit else "MISS"
        print(f"  {key:>8}  {status:>9}  {state}")

    print(f"\n  {cache.stats()}")


if __name__ == "__main__":
    demo_cache_aside()
    demo_write_strategies()
    demo_ttl()
    demo_eviction_comparison()
