# Exercise: Redis Caching Patterns
# Practice with cache-aside, write-through, TTL management, and invalidation strategies.

import json
import time
import hashlib
from typing import Optional, Callable
from collections import defaultdict


# Exercise 1: In-Memory Cache with TTL
class TTLCache:
    """Simple cache with per-key time-to-live expiration."""

    def __init__(self):
        self._store = {}  # key -> {"value": ..., "expires_at": float}

    def set(self, key: str, value, ttl: int = 300):
        """Store a value with TTL in seconds."""
        # TODO: Implement
        pass

    def get(self, key: str):
        """Get value if key exists and not expired. Return None otherwise.
        Clean up expired key on access.
        """
        # TODO: Implement
        pass

    def delete(self, key: str) -> bool:
        """Delete a key. Return True if it existed."""
        # TODO: Implement
        pass

    def clear(self):
        """Remove all entries."""
        # TODO: Implement
        pass

    def cleanup(self) -> int:
        """Remove all expired entries. Return count of removed keys."""
        # TODO: Implement
        pass


# Test
# cache = TTLCache()
# cache.set("a", "hello", ttl=1)
# assert cache.get("a") == "hello"
# time.sleep(1.1)
# assert cache.get("a") is None
# cache.set("b", 42, ttl=60)
# assert cache.delete("b") is True
# assert cache.delete("b") is False


# Exercise 2: Cache-Aside Decorator
def cache_aside(cache: TTLCache, ttl: int = 300):
    """Decorator that implements cache-aside pattern.

    - Cache key is derived from function name and arguments
    - On cache hit, return cached value
    - On cache miss, call function, store result, return it
    """
    def decorator(func):
        # TODO: Implement
        # Key format: "{func_name}:{arg1}:{arg2}:kwarg1=val1"
        pass
    return decorator


# Test
# cache = TTLCache()
# call_count = {"n": 0}
# @cache_aside(cache, ttl=60)
# def fetch_user(user_id: int) -> dict:
#     call_count["n"] += 1
#     return {"id": user_id, "name": f"User {user_id}"}
# result1 = fetch_user(1)
# result2 = fetch_user(1)  # should be cached
# assert call_count["n"] == 1
# assert result1 == result2


# Exercise 3: Write-Through Cache
class WriteThroughCache:
    """Cache that writes to both cache and a backing store simultaneously."""

    def __init__(self, cache: TTLCache, ttl: int = 600):
        self.cache = cache
        self.ttl = ttl
        self._db = {}  # simulated database

    def get(self, key: str):
        """Try cache first, then DB. Populate cache on DB hit."""
        # TODO: Implement
        pass

    def put(self, key: str, value):
        """Write to both DB and cache atomically."""
        # TODO: Implement
        pass

    def delete(self, key: str):
        """Delete from both DB and cache."""
        # TODO: Implement
        pass

    def db_size(self) -> int:
        return len(self._db)


# Test
# cache = TTLCache()
# wt = WriteThroughCache(cache, ttl=60)
# wt.put("user:1", {"name": "Alice"})
# assert wt.get("user:1") == {"name": "Alice"}
# cache.delete("user:1")  # evict from cache
# assert wt.get("user:1") == {"name": "Alice"}  # should fetch from DB and re-cache
# assert cache.get("user:1") is not None  # should be re-cached


# Exercise 4: Tag-Based Invalidation
class TaggedCache:
    """Cache with tag-based group invalidation."""

    def __init__(self):
        self._cache = TTLCache()
        self._tags = defaultdict(set)  # tag -> set of keys

    def set(self, key: str, value, ttl: int = 300, tags: list[str] | None = None):
        """Store value with optional tags for group invalidation."""
        # TODO: Implement
        pass

    def get(self, key: str):
        """Get cached value."""
        # TODO: Implement
        pass

    def invalidate_tag(self, tag: str) -> int:
        """Delete all cache entries with this tag. Return count deleted."""
        # TODO: Implement
        pass


# Test
# tc = TaggedCache()
# tc.set("product:1", {"name": "A"}, tags=["catalog", "featured"])
# tc.set("product:2", {"name": "B"}, tags=["catalog"])
# tc.set("user:1", {"name": "Alice"}, tags=["users"])
# assert tc.get("product:1") is not None
# count = tc.invalidate_tag("catalog")
# assert count == 2
# assert tc.get("product:1") is None
# assert tc.get("product:2") is None
# assert tc.get("user:1") is not None  # unaffected


# Exercise 5: Cache Key Builder
def build_cache_key(prefix: str, *args, **kwargs) -> str:
    """Build a deterministic cache key.

    Format: "{prefix}:{arg1}:{arg2}:{hash}" where hash is MD5 of sorted kwargs.
    If no kwargs, omit the hash part.
    """
    # TODO: Implement
    pass


# Test
# k1 = build_cache_key("api", "books", page=2, sort="title")
# k2 = build_cache_key("api", "books", sort="title", page=2)  # same kwargs, different order
# assert k1 == k2  # deterministic
# k3 = build_cache_key("api", "books")
# assert k3 == "api:books"


if __name__ == "__main__":
    print("Redis Caching Patterns Exercise")
    print("Implement each class/function and verify with the test cases.")
