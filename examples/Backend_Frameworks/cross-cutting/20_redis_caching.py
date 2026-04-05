"""
Redis Caching Patterns — Strategies and Cache Invalidation
Demonstrates: cache-aside, write-through, TTL management, cache stampede
              prevention, and tag-based invalidation.

Run: pip install redis fakeredis
     python 20_redis_caching.py
Note: Uses fakeredis for self-contained demo (no Redis server needed).
"""

import json
import time
import hashlib
import threading
from typing import Optional, Callable
from dataclasses import dataclass

# Use fakeredis for demo; swap with redis.Redis() in production
try:
    import fakeredis
    r = fakeredis.FakeRedis(decode_responses=True)
except ImportError:
    print("Install fakeredis for demo: pip install fakeredis")
    raise


# --- 1. Cache-Aside Pattern ---

def cache_aside(key: str, fetch_fn: Callable, ttl: int = 300) -> dict:
    """Look in cache first; on miss, fetch from source and populate cache."""
    cached = r.get(key)
    if cached:
        print(f"  [HIT] {key}")
        return json.loads(cached)
    print(f"  [MISS] {key}")
    data = fetch_fn()
    r.setex(key, ttl, json.dumps(data))
    return data


# Simulated DB query
def fetch_user(user_id: int) -> dict:
    time.sleep(0.01)  # simulate latency
    return {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}


# --- 2. Write-Through Pattern ---

class WriteThroughCache:
    """Writes go to both cache and DB atomically."""

    def __init__(self, prefix: str, ttl: int = 600):
        self.prefix = prefix
        self.ttl = ttl
        self._db: dict[str, dict] = {}  # simulated DB

    def _key(self, id_: str) -> str:
        return f"{self.prefix}:{id_}"

    def get(self, id_: str) -> Optional[dict]:
        cached = r.get(self._key(id_))
        if cached:
            return json.loads(cached)
        record = self._db.get(id_)
        if record:
            r.setex(self._key(id_), self.ttl, json.dumps(record))
        return record

    def put(self, id_: str, data: dict):
        """Write to DB and cache simultaneously."""
        self._db[id_] = data
        r.setex(self._key(id_), self.ttl, json.dumps(data))

    def delete(self, id_: str):
        self._db.pop(id_, None)
        r.delete(self._key(id_))


# --- 3. Cache Stampede Prevention (Lock-based) ---

def cache_with_lock(key: str, fetch_fn: Callable, ttl: int = 300, lock_timeout: int = 5) -> dict:
    """Prevent thundering herd with a distributed lock on cache miss."""
    cached = r.get(key)
    if cached:
        return json.loads(cached)

    lock_key = f"lock:{key}"
    acquired = r.set(lock_key, "1", nx=True, ex=lock_timeout)
    if acquired:
        try:
            data = fetch_fn()
            r.setex(key, ttl, json.dumps(data))
            return data
        finally:
            r.delete(lock_key)
    else:
        # Another process is populating; wait and retry
        for _ in range(lock_timeout * 10):
            time.sleep(0.1)
            cached = r.get(key)
            if cached:
                return json.loads(cached)
        # Fallback: fetch directly
        return fetch_fn()


# --- 4. Tag-Based Invalidation ---

def cache_with_tags(key: str, data: dict, tags: list[str], ttl: int = 300):
    """Store data with tag associations for group invalidation."""
    r.setex(key, ttl, json.dumps(data))
    for tag in tags:
        r.sadd(f"tag:{tag}", key)
        r.expire(f"tag:{tag}", ttl + 60)


def invalidate_tag(tag: str):
    """Delete all cache entries associated with a tag."""
    tag_key = f"tag:{tag}"
    keys = r.smembers(tag_key)
    if keys:
        r.delete(*keys)
        print(f"  [INVALIDATE] tag={tag}, keys={list(keys)}")
    r.delete(tag_key)


# --- 5. Cache Key Builder ---

def cache_key(*parts, **params) -> str:
    """Build deterministic cache key from parts and sorted params."""
    base = ":".join(str(p) for p in parts)
    if params:
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{base}:{hashlib.md5(param_str.encode()).hexdigest()[:8]}"
    return base


# ========== Demo ==========

if __name__ == "__main__":
    print("=== Cache-Aside Pattern ===")
    print(cache_aside("user:1", lambda: fetch_user(1)))  # MISS
    print(cache_aside("user:1", lambda: fetch_user(1)))  # HIT

    print("\n=== Write-Through Pattern ===")
    products = WriteThroughCache("product")
    products.put("p1", {"name": "Widget", "price": 9.99})
    print(products.get("p1"))  # from cache

    print("\n=== Cache Stampede Prevention ===")
    r.delete("expensive:query")
    result = cache_with_lock("expensive:query", lambda: {"rows": 1000})
    print(result)

    print("\n=== Tag-Based Invalidation ===")
    cache_with_tags("product:1", {"name": "A"}, tags=["catalog", "featured"])
    cache_with_tags("product:2", {"name": "B"}, tags=["catalog"])
    print(f"  Before: product:1={r.get('product:1')}")
    invalidate_tag("catalog")
    print(f"  After:  product:1={r.get('product:1')}")  # None

    print("\n=== Cache Key Builder ===")
    print(cache_key("api", "books", page=2, sort="title"))
