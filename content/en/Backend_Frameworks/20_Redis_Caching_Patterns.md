# 20. Redis Caching Patterns

**Previous**: [Go Web Basics](./19_Go_Web_Basics.md) | **Next**: [Job Queues](./21_Job_Queues.md)

**Difficulty**: ⭐⭐⭐

## Learning Objectives

- Understand Redis data types and their use cases in backend systems
- Implement the cache-aside (lazy loading) pattern for read-heavy workloads
- Apply write-through and write-behind patterns for write consistency
- Design effective cache invalidation strategies using TTL and event-driven approaches
- Use Redis as a session store for stateless application servers
- Build rate limiters with Redis atomic operations
- Implement Pub/Sub messaging for real-time communication
- Use Redis Streams for event sourcing and message processing
- Integrate Redis with FastAPI, Express, and Django applications

## Table of Contents

1. [Redis Fundamentals](#1-redis-fundamentals)
2. [Cache-Aside Pattern](#2-cache-aside-pattern)
3. [Write-Through and Write-Behind](#3-write-through-and-write-behind)
4. [Cache Invalidation Strategies](#4-cache-invalidation-strategies)
5. [Redis as Session Store](#5-redis-as-session-store)
6. [Rate Limiting with Redis](#6-rate-limiting-with-redis)
7. [Pub/Sub Messaging](#7-pubsub-messaging)
8. [Redis Streams](#8-redis-streams)
9. [Framework Integration](#9-framework-integration)
10. [Practice Exercises](#10-practice-exercises)

---

## 1. Redis Fundamentals

Redis (Remote Dictionary Server) is an in-memory data structure store used as a database, cache, message broker, and streaming engine. It supports sub-millisecond response times and can handle millions of requests per second.

### Core Data Types

| Type | Description | Common Use Cases |
|---|---|---|
| String | Binary-safe string (max 512 MB) | Cache values, counters, session tokens |
| Hash | Field-value pairs | User profiles, object storage |
| List | Ordered collection (linked list) | Message queues, activity feeds |
| Set | Unordered unique elements | Tags, unique visitors, set operations |
| Sorted Set | Scored unique elements | Leaderboards, rate limiting, time series |
| Stream | Append-only log | Event sourcing, message queues |

### Data Type Quick Reference

| Data Type | Use Case | Example Command | Time Complexity |
|-----------|----------|-----------------|----------------|
| String | Cache, counters | GET/SET/INCR | O(1) |
| Hash | Object storage | HGET/HSET/HGETALL | O(1) per field |
| List | Queues, feeds | LPUSH/RPOP/LRANGE | O(1) push/pop |
| Set | Tags, unique items | SADD/SMEMBERS/SINTER | O(1) add |
| Sorted Set | Leaderboards, scheduling | ZADD/ZRANGE/ZRANK | O(log N) |
| Stream | Event log, messaging | XADD/XREAD/XGROUP | O(1) add |

### Essential Commands

```bash
# Strings
SET user:1:name "Alice"              # Set a key
GET user:1:name                       # Get a key
SETEX session:abc 3600 "user_data"   # Set with TTL (seconds)
INCR page:views                       # Atomic increment
MSET k1 "v1" k2 "v2"                # Set multiple keys
MGET k1 k2                           # Get multiple keys

# Hashes
HSET user:1 name "Alice" email "alice@example.com" age "30"
HGET user:1 name
HGETALL user:1
HINCRBY user:1 age 1

# Lists
LPUSH queue:tasks "task1" "task2"     # Push to head
RPOP queue:tasks                      # Pop from tail (FIFO queue)
LRANGE queue:tasks 0 -1              # Get all elements
LLEN queue:tasks                      # Length

# Sets
SADD tags:post:1 "python" "redis" "backend"
SMEMBERS tags:post:1
SISMEMBER tags:post:1 "python"       # Check membership
SINTER tags:post:1 tags:post:2       # Intersection

# Sorted Sets
ZADD leaderboard 100 "alice" 85 "bob" 92 "charlie"
ZREVRANGE leaderboard 0 9 WITHSCORES  # Top 10
ZRANK leaderboard "alice"             # Rank (0-indexed)
ZINCRBY leaderboard 5 "bob"           # Increment score

# Key management
DEL key1 key2                         # Delete keys
EXISTS key1                           # Check existence
EXPIRE key1 300                       # Set TTL
TTL key1                              # Check remaining TTL
KEYS "user:*"                         # Find keys (avoid in production)
SCAN 0 MATCH "user:*" COUNT 100      # Iterate keys safely
```

### Python Redis Client Setup

```python
import redis
import json
from typing import Optional, Any

# Connection
r = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,       # Return strings instead of bytes
    socket_connect_timeout=5,
    retry_on_timeout=True,
)

# Connection pool (recommended for production)
pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    max_connections=20,
    decode_responses=True,
)
r = redis.Redis(connection_pool=pool)

# Health check
r.ping()  # Returns True
```

### Node.js Redis Client Setup

```javascript
import { createClient } from 'redis';

const client = createClient({
    url: 'redis://localhost:6379',
    socket: {
        connectTimeout: 5000,
        reconnectStrategy: (retries) => Math.min(retries * 100, 5000),
    },
});

client.on('error', (err) => console.error('Redis error:', err));
client.on('connect', () => console.log('Redis connected'));

await client.connect();

// Basic operations
await client.set('key', 'value');
const value = await client.get('key');
await client.setEx('session:abc', 3600, JSON.stringify({ userId: 1 }));
```

---

## 2. Cache-Aside Pattern

Cache-aside (also called lazy loading) is the most common caching pattern. The application manages the cache explicitly: it checks the cache first, and on a miss, fetches from the database and populates the cache.

### Theory: Cache Stampede / Dogpile

The cache holds key X with TTL 300s. At second 300, X expires. At second 301, 1000 concurrent requests all miss. All 1000 query the database simultaneously to recompute X. The database melts.

This is **cache stampede** (or "dogpile"). It is the single most common production-cache failure mode. Three defenses, ranging from simple to sophisticated.

#### C.1 Locking (single-flight)

When the first request misses, it takes a Redis lock keyed `lock:X`. Other requests that miss block on that lock. The first request computes X, populates the cache, releases the lock. Other requests retry, find X in the cache, return.

```python
def get(key):
    val = cache.get(key)
    if val is not None: return val
    with redis.lock(f"lock:{key}", timeout=5):
        # double-check after acquiring lock
        val = cache.get(key)
        if val is not None: return val
        val = db.get(key)
        cache.set(key, val, ttl=300)
        return val
```

The lock turns N concurrent computations into 1. Risk: lock contention if X is *very* hot — many requests serialize on the lock. Use a lock timeout to avoid deadlock if the holder dies.

#### C.2 Probabilistic early refresh (XFetch)

Instead of waiting for TTL to fire, refresh the cache *probabilistically* slightly before expiry. The closer to expiry, the higher the probability:

```python
def get(key):
    val, ttl_remaining = cache.get_with_ttl(key)
    # randomly recompute when close to expiry
    if val is None or random() < beta * ttl_remaining_factor:
        val = db.get(key)
        cache.set(key, val, ttl=300)
    return val
```

The XFetch algorithm formalizes this. The benefit: the cache rarely fully expires, so a stampede never starts. No locking needed.

#### C.3 Request coalescing

The cache library detects pending lookups for the same key and joins them onto one in-flight backend call. The first miss starts the database query; concurrent misses for the same key wait for that result instead of issuing their own queries.

This is what Go's `golang.org/x/sync/singleflight` does. Caffeine has a similar `LoadingCache`. The discipline is in the *cache library*, not the application — once you have it, every cache lookup is stampede-safe.

#### C.4 Comparing the defenses

| Defense | Complexity | Effectiveness | When to use |
|---------|------------|---------------|-------------|
| Locking | Low | High | Hot keys with expensive recompute |
| Probabilistic early refresh | Medium | Very high | Cache backed by a synchronous cache library |
| Request coalescing | Library-level | High | Built into modern cache libraries |

Production caches usually combine: probabilistic refresh keeps things populated; coalescing handles the residual concurrent misses.

### Flow

```
1. Application receives request
2. Check Redis cache for the key
3. Cache HIT  → return cached data
4. Cache MISS → query database → store result in Redis → return data
```

### Python Implementation

```python
import redis
import json
import hashlib
from typing import Optional
from datetime import timedelta

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

class CacheAside:
    def __init__(self, redis_client: redis.Redis, default_ttl: int = 300):
        self.redis = redis_client
        self.default_ttl = default_ttl

    def _make_key(self, prefix: str, identifier: str) -> str:
        return f"{prefix}:{identifier}"

    def get_user(self, user_id: int) -> Optional[dict]:
        cache_key = self._make_key("user", str(user_id))

        # Step 1: Check cache
        cached = self.redis.get(cache_key)
        if cached:
            print(f"Cache HIT: {cache_key}")
            return json.loads(cached)

        # Step 2: Cache miss — fetch from database
        print(f"Cache MISS: {cache_key}")
        user = self._fetch_user_from_db(user_id)
        if user is None:
            return None

        # Step 3: Populate cache
        self.redis.setex(cache_key, self.default_ttl, json.dumps(user))
        return user

    def invalidate_user(self, user_id: int) -> None:
        cache_key = self._make_key("user", str(user_id))
        self.redis.delete(cache_key)

    def _fetch_user_from_db(self, user_id: int) -> Optional[dict]:
        # Simulated database query
        return {"id": user_id, "name": "Alice", "email": "alice@example.com"}

# Usage
cache = CacheAside(r, default_ttl=600)
user = cache.get_user(42)       # Cache MISS → fetches from DB
user = cache.get_user(42)       # Cache HIT → returns from Redis
cache.invalidate_user(42)
user = cache.get_user(42)       # Cache MISS → fetches again
```

### Batch Cache-Aside

Fetching multiple items with a single round trip:

```python
def get_users_batch(self, user_ids: list[int]) -> list[dict]:
    cache_keys = [self._make_key("user", str(uid)) for uid in user_ids]

    # Step 1: Multi-get from cache
    cached_values = self.redis.mget(cache_keys)

    results = {}
    missing_ids = []

    for uid, cached in zip(user_ids, cached_values):
        if cached:
            results[uid] = json.loads(cached)
        else:
            missing_ids.append(uid)

    # Step 2: Fetch missing from database
    if missing_ids:
        db_users = self._fetch_users_batch_from_db(missing_ids)
        pipe = self.redis.pipeline()
        for user in db_users:
            results[user["id"]] = user
            key = self._make_key("user", str(user["id"]))
            pipe.setex(key, self.default_ttl, json.dumps(user))
        pipe.execute()

    return [results[uid] for uid in user_ids if uid in results]
```

### Cache Stampede Prevention

When a popular cache key expires, many concurrent requests may hit the database simultaneously. Use a lock to prevent this:

```python
import time

def get_user_with_lock(self, user_id: int) -> Optional[dict]:
    cache_key = self._make_key("user", str(user_id))
    lock_key = f"lock:{cache_key}"

    cached = self.redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # Try to acquire lock (NX = set if not exists, EX = expiry)
    acquired = self.redis.set(lock_key, "1", nx=True, ex=10)

    if acquired:
        try:
            user = self._fetch_user_from_db(user_id)
            if user:
                self.redis.setex(cache_key, self.default_ttl, json.dumps(user))
            return user
        finally:
            self.redis.delete(lock_key)
    else:
        # Another process is fetching; wait and retry
        for _ in range(50):  # 50 * 0.1s = 5 seconds max
            time.sleep(0.1)
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        # Fallback to database
        return self._fetch_user_from_db(user_id)
```

---

## 3. Write-Through and Write-Behind

### Theory: The Three Caching Strategies

Cache strategy is fundamentally about *who writes to the cache and when*. Three patterns dominate.

#### A.1 Cache-aside (lazy loading)

The application reads from the cache; on miss, it reads from the database and populates the cache.

```
Read:
  if cache.hit(key):  return cache.get(key)
  data = db.get(key)
  cache.set(key, data, ttl=300)
  return data

Write:
  db.update(key, value)
  cache.delete(key)   # or set to new value
```

Properties:

- The cache is a *side cache* — the application is the one in charge.
- The first read after a write or eviction is slow (cache miss).
- Strong consistency requires invalidating the cache on every write — easy in one process, hard across services.

This is the default pattern; reach for the others only when cache-aside doesn't fit.

#### A.2 Write-through

Writes go through the cache to the database. Reads always come from the cache.

```
Write:
  cache.set(key, value)
  db.update(key, value)  # synchronous

Read:
  return cache.get(key)  # always populated
```

Properties:

- Cache is always consistent with the database (assuming both writes succeed).
- Write latency is `cache_write + db_write` (slower than write-behind).
- Cache size must hold all reads — expensive for large datasets.
- If the cache is cold (after restart), every read misses until populated. Often paired with cache-warming strategies.

Use when reads must be fast and consistency matters more than write latency.

#### A.3 Write-behind (write-back)

Writes go to the cache; the database is updated asynchronously, often in batches.

```
Write:
  cache.set(key, value)
  queue.push(("write", key, value))  # async worker reads this

Worker (separate):
  for batch of writes from queue:
    db.bulk_update(batch)
```

Properties:

- Lowest write latency (only cache write blocks the request).
- Highest throughput (writes batched).
- Risk of data loss: if the cache crashes before the queue drains, queued writes are gone.
- Eventual consistency: a read of the database during the lag window sees stale data.

Use for high-throughput write workloads where the data is replaceable (analytics counters, telemetry) or where the cache is durable (Redis with AOF persistence).

#### A.4 Picking a strategy

| Strategy | Consistency | Read latency | Write latency | Complexity |
|----------|-------------|--------------|---------------|------------|
| Cache-aside | Eventual | DB on miss | Fast | Low |
| Write-through | Strong | Cache only | Slow | Medium |
| Write-behind | Eventual | Cache only | Fastest | High |

Most apps use cache-aside; specialized workloads use write-through (real-time dashboards) or write-behind (counters, analytics).

### Write-Through Pattern

Every write goes to both the cache and the database synchronously. This ensures cache consistency but increases write latency.

```python
class WriteThrough:
    def __init__(self, redis_client, db_session, ttl=600):
        self.redis = redis_client
        self.db = db_session
        self.ttl = ttl

    def update_user(self, user_id: int, data: dict) -> dict:
        # Step 1: Write to database
        user = self._update_db(user_id, data)

        # Step 2: Write to cache (same transaction context)
        cache_key = f"user:{user_id}"
        self.redis.setex(cache_key, self.ttl, json.dumps(user))

        return user

    def create_user(self, data: dict) -> dict:
        # Step 1: Insert into database
        user = self._insert_db(data)

        # Step 2: Populate cache
        cache_key = f"user:{user['id']}"
        self.redis.setex(cache_key, self.ttl, json.dumps(user))

        return user

    def _update_db(self, user_id, data):
        # Database UPDATE query
        return {"id": user_id, **data}

    def _insert_db(self, data):
        # Database INSERT query
        return {"id": 1, **data}
```

### Write-Behind (Write-Back) Pattern

Writes go to the cache immediately, and a background process asynchronously flushes changes to the database. This reduces write latency but risks data loss.

```python
import threading
import queue

class WriteBehind:
    def __init__(self, redis_client, db_session, flush_interval=5):
        self.redis = redis_client
        self.db = db_session
        self.write_queue = queue.Queue()
        self.flush_interval = flush_interval
        self._start_flusher()

    def update_user(self, user_id: int, data: dict) -> dict:
        user = {"id": user_id, **data}
        cache_key = f"user:{user_id}"

        # Step 1: Write to cache immediately
        self.redis.setex(cache_key, 3600, json.dumps(user))

        # Step 2: Queue the write for async database flush
        self.write_queue.put(("update", "user", user_id, data))

        return user

    def _start_flusher(self):
        def flush_worker():
            while True:
                batch = []
                try:
                    while len(batch) < 100:
                        item = self.write_queue.get(timeout=self.flush_interval)
                        batch.append(item)
                except queue.Empty:
                    pass

                if batch:
                    self._flush_to_db(batch)

        thread = threading.Thread(target=flush_worker, daemon=True)
        thread.start()

    def _flush_to_db(self, batch):
        for operation, table, entity_id, data in batch:
            try:
                if operation == "update":
                    # Execute UPDATE query
                    print(f"Flushed {operation} {table}:{entity_id}")
            except Exception as e:
                print(f"Flush error: {e}")
                # Re-queue or write to dead-letter queue
```

### Comparison

| Aspect | Cache-Aside | Write-Through | Write-Behind |
|---|---|---|---|
| Read latency | Miss penalty | Always fast | Always fast |
| Write latency | N/A (bypass cache) | Higher (dual write) | Lowest (cache only) |
| Consistency | Eventual | Strong | Eventual |
| Complexity | Low | Medium | High |
| Data loss risk | None | None | Possible |

---

## 4. Cache Invalidation Strategies

Cache invalidation is one of the hardest problems in computer science. Here are practical strategies.

### Theory: Eviction Policies

A cache holds finite RAM. When you set a new key and the cache is full, something must be evicted. The eviction policy decides what.

#### B.1 The classics

- **LRU (Least Recently Used).** Evict the key that has not been accessed in the longest time. Implemented as a doubly-linked list + hashmap; access moves the node to the front. Simple, low overhead, good for typical access patterns where recently-used items are likely to be used again.
- **LFU (Least Frequently Used).** Evict the key with the lowest access frequency. Better than LRU for workloads with a long-tail of rarely-accessed keys, but needs a counter per key. Vulnerable to "scan pollution" — a one-time scan of cold data inflates frequencies and evicts hot data.
- **FIFO (First In, First Out).** Evict the oldest key by insertion time. Simplest; rarely the best.

#### B.2 Modern hybrids

- **ARC (Adaptive Replacement Cache).** Maintains two LRU lists (recently used once, frequently used) and dynamically balances between them. Resists scan pollution. Patented for some uses; not in mainline Redis.
- **TinyLFU / W-TinyLFU.** Frequency sketch (Count-Min) admits a new key only if its predicted frequency exceeds the LFU candidate. Used in Caffeine (Java cache library), state-of-the-art for general caches.

#### B.3 Redis's eviction policies

Redis offers a menu via `maxmemory-policy`:

```
allkeys-lru           # LRU across all keys
allkeys-lfu           # LFU across all keys (Redis 4+)
volatile-lru          # LRU only on keys with a TTL set
volatile-lfu          # LFU only on keys with a TTL set
allkeys-random        # random eviction (rarely useful)
volatile-ttl          # evict the key closest to its TTL
noeviction            # refuse writes when full (return error)
```

The choice depends on your workload:

- **Pure cache** (everything is best-effort): `allkeys-lru` or `allkeys-lfu`.
- **Mixed cache + persistent data** in the same Redis: `volatile-lru` (only evict TTL-bearing keys).
- **No eviction acceptable** (e.g., session store): `noeviction` and over-provision RAM.

### TTL-Based Expiration

The simplest approach: set a time-to-live on every cached entry.

```python
# Short TTL for frequently changing data
r.setex("stock:AAPL:price", 30, "150.25")       # 30 seconds

# Medium TTL for user profiles
r.setex("user:42:profile", 600, json.dumps(profile))  # 10 minutes

# Long TTL for rarely changing data
r.setex("config:feature_flags", 3600, json.dumps(flags))  # 1 hour

# Adaptive TTL based on access patterns
def get_adaptive_ttl(key: str, base_ttl: int = 300) -> int:
    access_count = r.incr(f"access_count:{key}")
    r.expire(f"access_count:{key}", 3600)

    if access_count > 100:
        return base_ttl * 3   # Hot key: extend TTL
    elif access_count > 10:
        return base_ttl       # Normal
    else:
        return base_ttl // 2  # Cold key: shorter TTL
```

### Event-Driven Invalidation

Invalidate cache entries when the underlying data changes:

```python
class EventDrivenCache:
    def __init__(self, redis_client):
        self.redis = redis_client

    def on_user_updated(self, user_id: int):
        """Called after a database UPDATE on the user table."""
        # Delete the user cache
        self.redis.delete(f"user:{user_id}")

        # Delete related caches
        self.redis.delete(f"user:{user_id}:posts")
        self.redis.delete(f"user:{user_id}:stats")

        # Invalidate list caches that may contain this user
        self._invalidate_pattern(f"users:list:*")

    def on_post_created(self, post: dict):
        """Called after a new post is created."""
        author_id = post["author_id"]

        # Invalidate author's post list
        self.redis.delete(f"user:{author_id}:posts")

        # Invalidate paginated post lists
        self._invalidate_pattern("posts:page:*")

        # Increment version counter instead of deleting
        self.redis.incr("posts:version")

    def _invalidate_pattern(self, pattern: str):
        """Delete all keys matching a pattern using SCAN."""
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
```

### Version-Based Invalidation

Instead of deleting cache entries, use a version number in the cache key:

```python
def get_posts_versioned(self, page: int) -> list:
    version = self.redis.get("posts:version") or "0"
    cache_key = f"posts:v{version}:page:{page}"

    cached = self.redis.get(cache_key)
    if cached:
        return json.loads(cached)

    posts = self._fetch_posts_from_db(page)
    self.redis.setex(cache_key, 300, json.dumps(posts))
    return posts

def invalidate_posts(self):
    # Simply increment the version; old keys expire via TTL
    self.redis.incr("posts:version")
```

---

## 5. Redis as Session Store

Using Redis for session storage enables stateless application servers, making horizontal scaling straightforward.

### Express.js Session Store

```javascript
import express from 'express';
import session from 'express-session';
import RedisStore from 'connect-redis';
import { createClient } from 'redis';

const redisClient = createClient({ url: 'redis://localhost:6379' });
await redisClient.connect();

const app = express();

app.use(session({
    store: new RedisStore({ client: redisClient }),
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: process.env.NODE_ENV === 'production',
        httpOnly: true,
        maxAge: 24 * 60 * 60 * 1000,  // 24 hours
        sameSite: 'strict',
    },
}));

app.post('/login', (req, res) => {
    // After authentication
    req.session.userId = user.id;
    req.session.role = user.role;
    res.json({ message: 'Logged in' });
});

app.get('/profile', (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({ error: 'Not authenticated' });
    }
    res.json({ userId: req.session.userId });
});

app.post('/logout', (req, res) => {
    req.session.destroy((err) => {
        res.json({ message: 'Logged out' });
    });
});
```

### Python Flask Session Store

```python
from flask import Flask, session
from flask_session import Session
import redis

app = Flask(__name__)
app.config.update(
    SESSION_TYPE="redis",
    SESSION_REDIS=redis.Redis(host="localhost", port=6379, db=1),
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=86400,  # 24 hours
    SESSION_KEY_PREFIX="session:",
    SESSION_USE_SIGNER=True,
    SECRET_KEY="your-secret-key",
)
Session(app)

@app.post("/login")
def login():
    # After authentication
    session["user_id"] = user.id
    session["role"] = user.role
    return {"message": "Logged in"}

@app.get("/profile")
def profile():
    user_id = session.get("user_id")
    if not user_id:
        return {"error": "Not authenticated"}, 401
    return {"user_id": user_id}
```

---

## 6. Rate Limiting with Redis

Redis's atomic operations make it ideal for implementing distributed rate limiters.

### Fixed Window Rate Limiter

```python
def fixed_window_rate_limit(
    redis_client: redis.Redis,
    key: str,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int]:
    """
    Returns (allowed: bool, remaining: int).
    """
    window_key = f"ratelimit:{key}:{int(time.time()) // window_seconds}"

    pipe = redis_client.pipeline()
    pipe.incr(window_key)
    pipe.expire(window_key, window_seconds)
    count, _ = pipe.execute()

    allowed = count <= limit
    remaining = max(0, limit - count)
    return allowed, remaining

# Usage
allowed, remaining = fixed_window_rate_limit(r, "api:user:42", limit=100, window_seconds=60)
if not allowed:
    print("Rate limit exceeded")
```

### Sliding Window Rate Limiter

More accurate than fixed window, prevents burst at window boundaries:

```python
def sliding_window_rate_limit(
    redis_client: redis.Redis,
    key: str,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int]:
    now = time.time()
    window_start = now - window_seconds
    member = f"{now}"

    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
    pipe.zadd(key, {member: now})                 # Add current request
    pipe.zcard(key)                               # Count entries in window
    pipe.expire(key, window_seconds)              # Set TTL for cleanup
    _, _, count, _ = pipe.execute()

    allowed = count <= limit
    remaining = max(0, limit - count)
    return allowed, remaining
```

### Token Bucket Rate Limiter (Lua Script)

For highest performance, use a Lua script for atomic token bucket logic:

```python
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])  -- tokens per second
local now = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1]) or max_tokens
local last_refill = tonumber(data[2]) or now

-- Refill tokens
local elapsed = now - last_refill
local new_tokens = math.min(max_tokens, tokens + elapsed * refill_rate)

if new_tokens >= 1 then
    new_tokens = new_tokens - 1
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    redis.call('EXPIRE', key, math.ceil(max_tokens / refill_rate) * 2)
    return {1, math.floor(new_tokens)}  -- allowed, remaining
else
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    return {0, 0}  -- denied, remaining
end
"""

class TokenBucketLimiter:
    def __init__(self, redis_client, max_tokens=10, refill_rate=1.0):
        self.redis = redis_client
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.script = self.redis.register_script(TOKEN_BUCKET_SCRIPT)

    def allow(self, key: str) -> tuple[bool, int]:
        result = self.script(
            keys=[f"bucket:{key}"],
            args=[self.max_tokens, self.refill_rate, time.time()],
        )
        return bool(result[0]), int(result[1])
```

---

## 7. Pub/Sub Messaging

Redis Pub/Sub provides fire-and-forget messaging for real-time features.

### Publisher

```python
import redis
import json
import time

publisher = redis.Redis(host="localhost", port=6379, decode_responses=True)

def publish_event(channel: str, event_type: str, data: dict):
    message = json.dumps({
        "type": event_type,
        "data": data,
        "timestamp": time.time(),
    })
    subscriber_count = publisher.publish(channel, message)
    print(f"Published to {channel}, {subscriber_count} subscribers received")

# Usage
publish_event("notifications", "new_order", {"order_id": 123, "total": 59.99})
publish_event("chat:room:42", "message", {"user": "Alice", "text": "Hello!"})
```

### Subscriber

```python
import redis
import json

subscriber = redis.Redis(host="localhost", port=6379, decode_responses=True)
pubsub = subscriber.pubsub()

def handle_notification(message):
    if message["type"] == "message":
        data = json.loads(message["data"])
        print(f"Received: {data['type']} -> {data['data']}")

pubsub.subscribe(**{"notifications": handle_notification})

# Blocking listener
thread = pubsub.run_in_thread(sleep_time=0.01)

# To stop:
# thread.stop()
# pubsub.unsubscribe()
```

### Node.js Pub/Sub

```javascript
import { createClient } from 'redis';

// Publisher
const publisher = createClient({ url: 'redis://localhost:6379' });
await publisher.connect();

await publisher.publish('events', JSON.stringify({
    type: 'user_signup',
    data: { userId: 1, email: 'alice@example.com' },
}));

// Subscriber (must use a separate connection)
const subscriber = createClient({ url: 'redis://localhost:6379' });
await subscriber.connect();

await subscriber.subscribe('events', (message) => {
    const event = JSON.parse(message);
    console.log(`Event: ${event.type}`, event.data);
});
```

### Limitations of Pub/Sub

- **No persistence**: Messages are lost if no subscriber is listening
- **No acknowledgment**: No guarantee a subscriber processed the message
- **No replay**: Cannot re-read past messages
- For reliable messaging, use **Redis Streams** (next section) or a dedicated message broker

---

## 8. Redis Streams

Redis Streams provide a persistent, append-only log with consumer groups, acknowledgment, and message replay. They are a reliable alternative to Pub/Sub.

### Producing Messages

```python
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Add entries to a stream
message_id = r.xadd("orders", {
    "order_id": "1001",
    "customer_id": "42",
    "total": "59.99",
    "status": "pending",
})
print(f"Added message: {message_id}")  # e.g., "1678901234567-0"

# Add with max length (capped stream)
r.xadd("logs", {"level": "info", "msg": "Server started"}, maxlen=10000)
```

### Consumer Groups

Consumer groups enable multiple consumers to divide work from a single stream:

```python
# Create a consumer group (starting from the beginning)
try:
    r.xgroup_create("orders", "order_processors", id="0", mkstream=True)
except redis.exceptions.ResponseError:
    pass  # Group already exists

# Consumer 1: Read and process messages
def process_orders(consumer_name: str):
    while True:
        # Read new messages for this consumer
        messages = r.xreadgroup(
            groupname="order_processors",
            consumername=consumer_name,
            streams={"orders": ">"},   # ">" means undelivered messages only
            count=10,
            block=5000,               # Block for 5 seconds if no messages
        )

        if not messages:
            continue

        for stream, entries in messages:
            for message_id, data in entries:
                try:
                    print(f"[{consumer_name}] Processing order {data['order_id']}")
                    # ... process the order ...

                    # Acknowledge successful processing
                    r.xack("orders", "order_processors", message_id)
                except Exception as e:
                    print(f"Error processing {message_id}: {e}")
                    # Message will be re-delivered to another consumer

# Run consumers in separate threads/processes
import threading
for i in range(3):
    t = threading.Thread(target=process_orders, args=(f"worker-{i}",))
    t.daemon = True
    t.start()
```

### Claiming Pending Messages

Handle messages that a crashed consumer left unacknowledged:

```python
def claim_stale_messages(group: str, consumer: str, min_idle_ms: int = 60000):
    """Claim messages idle for more than min_idle_ms."""
    # Get pending messages
    pending = r.xpending_range("orders", group, "-", "+", count=100)

    stale_ids = [
        entry["message_id"]
        for entry in pending
        if entry["time_since_delivered"] > min_idle_ms
    ]

    if stale_ids:
        claimed = r.xclaim(
            "orders", group, consumer,
            min_idle_time=min_idle_ms,
            message_ids=stale_ids,
        )
        print(f"Claimed {len(claimed)} stale messages")
        return claimed
    return []
```

### Stream Information

```python
# Stream metadata
info = r.xinfo_stream("orders")
print(f"Length: {info['length']}")
print(f"First entry: {info['first-entry']}")
print(f"Last entry: {info['last-entry']}")

# Consumer group info
groups = r.xinfo_groups("orders")
for g in groups:
    print(f"Group: {g['name']}, Pending: {g['pending']}, Consumers: {g['consumers']}")
```

---

## 9. Framework Integration

### FastAPI + Redis

```python
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
import json

redis_client: aioredis.Redis = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
    yield
    await redis_client.close()

app = FastAPI(lifespan=lifespan)

async def get_redis() -> aioredis.Redis:
    return redis_client

@app.get("/products/{product_id}")
async def get_product(product_id: int, cache: aioredis.Redis = Depends(get_redis)):
    # Check cache
    cached = await cache.get(f"product:{product_id}")
    if cached:
        return json.loads(cached)

    # Fetch from database
    product = await fetch_product_from_db(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Cache for 5 minutes
    await cache.setex(f"product:{product_id}", 300, json.dumps(product))
    return product

@app.put("/products/{product_id}")
async def update_product(
    product_id: int,
    data: dict,
    cache: aioredis.Redis = Depends(get_redis),
):
    product = await update_product_in_db(product_id, data)

    # Invalidate cache
    await cache.delete(f"product:{product_id}")

    return product
```

### Express.js + Redis Middleware

```javascript
import express from 'express';
import { createClient } from 'redis';

const app = express();
const redis = createClient({ url: 'redis://localhost:6379' });
await redis.connect();

// Cache middleware factory
function cacheMiddleware(ttl = 300) {
    return async (req, res, next) => {
        if (req.method !== 'GET') return next();

        const key = `cache:${req.originalUrl}`;
        const cached = await redis.get(key);

        if (cached) {
            return res.json(JSON.parse(cached));
        }

        // Override res.json to cache the response
        const originalJson = res.json.bind(res);
        res.json = async (data) => {
            await redis.setEx(key, ttl, JSON.stringify(data));
            return originalJson(data);
        };

        next();
    };
}

// Usage
app.get('/api/products', cacheMiddleware(60), async (req, res) => {
    const products = await db.query('SELECT * FROM products');
    res.json(products);
});

app.put('/api/products/:id', async (req, res) => {
    const product = await db.updateProduct(req.params.id, req.body);

    // Invalidate related caches
    await redis.del(`cache:/api/products`);
    await redis.del(`cache:/api/products/${req.params.id}`);

    res.json(product);
});
```

### Django + Redis Cache Backend

```python
# settings.py
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": "redis://localhost:6379/0",
        "OPTIONS": {
            "db": 0,
            "parser_class": "redis.connection.DefaultParser",
            "pool_class": "redis.BlockingConnectionPool",
        },
    }
}

# views.py
from django.core.cache import cache
from django.views.decorators.cache import cache_page

# Low-level cache API
def get_product(request, product_id):
    cache_key = f"product:{product_id}"
    product = cache.get(cache_key)

    if product is None:
        product = Product.objects.get(id=product_id)
        cache.set(cache_key, product, timeout=300)

    return JsonResponse(product.to_dict())

# View-level caching decorator
@cache_page(60 * 5)  # Cache entire view for 5 minutes
def product_list(request):
    products = Product.objects.all()
    return JsonResponse([p.to_dict() for p in products], safe=False)
```

---

## 10. Practice Exercises

### Exercise 1: Build a Cache-Aside Layer

Create a Python class `ArticleCache` that:
- Implements cache-aside for articles (id, title, content, author, published_at)
- Supports batch fetching with `MGET`
- Includes cache stampede prevention with a distributed lock
- Tracks hit/miss statistics in Redis using `HINCRBY`

```python
# Starter code
class ArticleCache:
    def __init__(self, redis_client, db, ttl=300):
        self.redis = redis_client
        self.db = db
        self.ttl = ttl

    def get(self, article_id: int) -> dict:
        # TODO: Implement cache-aside with lock
        pass

    def get_batch(self, article_ids: list[int]) -> list[dict]:
        # TODO: Implement batch cache-aside with MGET
        pass

    def invalidate(self, article_id: int):
        # TODO: Invalidate article and related caches
        pass

    def stats(self) -> dict:
        # TODO: Return hit/miss counts from Redis
        pass
```

### Exercise 2: Sliding Window Rate Limiter

Build a rate limiting middleware for FastAPI using Redis sorted sets:
- 100 requests per minute per API key
- Return `429 Too Many Requests` with `Retry-After` header
- Include rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### Exercise 3: Real-Time Notifications with Pub/Sub

Build a notification system with two components:
- **Publisher**: A FastAPI endpoint `POST /notify` that publishes messages to a channel
- **Subscriber**: A background process that listens and logs notifications
- Support channel patterns (e.g., `user:*:notifications`)
- Add a message history feature using Redis Lists (last 50 messages)

### Exercise 4: Order Processing Pipeline with Streams

Implement an order processing system using Redis Streams:
- Producer: Adds orders to the `orders` stream
- Consumer group with 3 workers: Each worker processes orders and acknowledges them
- Dead-letter handling: Claim messages idle for more than 60 seconds
- Dashboard: An endpoint that returns stream length, pending counts, and consumer lag

```python
# Starter code
class OrderPipeline:
    STREAM = "orders"
    GROUP = "order_processors"

    def __init__(self, redis_client):
        self.redis = redis_client
        self._ensure_group()

    def submit_order(self, order: dict) -> str:
        # TODO: Add order to stream
        pass

    def process(self, consumer_name: str):
        # TODO: Read, process, and acknowledge orders
        pass

    def claim_stale(self, consumer_name: str, min_idle_ms=60000):
        # TODO: Claim unacknowledged messages
        pass

    def dashboard(self) -> dict:
        # TODO: Return stream and consumer group stats
        pass
```

---

## Further Reading

- [Redis Documentation](https://redis.io/docs/)
- [Redis University](https://university.redis.io/)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)
- [Caching Strategies and How to Choose the Right One](https://codeahoy.com/2017/08/11/caching-strategies-and-how-to-choose-the-right-one/)
- [redis-py Documentation](https://redis-py.readthedocs.io/)
- [ioredis (Node.js)](https://github.com/redis/ioredis)

---

**Previous**: [Go Web Basics](./19_Go_Web_Basics.md) | **Next**: [Job Queues](./21_Job_Queues.md)
