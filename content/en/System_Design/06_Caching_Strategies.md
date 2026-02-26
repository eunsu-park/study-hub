# Caching Strategies

**Previous**: [Reverse Proxy & API Gateway](./05_Reverse_Proxy_API_Gateway.md) | **Next**: [Distributed Cache Systems](./07_Distributed_Cache_Systems.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why caching is one of the most effective techniques for improving system performance and reducing load
2. Compare Cache-Aside, Read-Through, Write-Through, and Write-Behind patterns, identifying the consistency and performance trade-offs of each
3. Describe cache invalidation strategies (TTL-based, event-driven, version-based) and explain why invalidation is considered the hardest problem in caching
4. Diagnose and solve common cache failure modes: cache penetration, cache avalanche, and hot key problems
5. Design a multi-layer caching architecture that includes browser, CDN, application, and database caches
6. Select the appropriate caching pattern for a given read/write workload

**Difficulty**: ⭐⭐⭐
**Estimated Learning Time**: 2-3 hours
**Prerequisites**: [05_Reverse_Proxy_API_Gateway.md](./05_Reverse_Proxy_API_Gateway.md)

---

The fastest database query is the one you never have to make. Caching stores frequently accessed data closer to where it is needed, turning expensive millisecond-level database lookups into microsecond in-memory reads. Done well, caching can reduce backend load by orders of magnitude; done poorly, it introduces stale data, thundering herds, and mysterious inconsistencies. This lesson equips you with the patterns and pitfalls you need to cache effectively.

## Table of Contents

1. [What is Caching?](#1-what-is-caching)
2. [Caching Strategy Patterns](#2-caching-strategy-patterns)
3. [Cache Invalidation](#3-cache-invalidation)
4. [Cache Problems and Solutions](#4-cache-problems-and-solutions)
5. [CDN Caching](#5-cdn-caching)
6. [Practice Problems](#6-practice-problems)
7. [References](#7-references)

---

## 1. What is Caching?

### 1.1 Definition of Caching

```
┌─────────────────────────────────────────────────────────────────┐
│                      What is Caching?                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Temporarily store frequently used data in fast storage"       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Without cache:                                            │ │
│  │                                                            │ │
│  │  Client ──▶ Server ──▶ Database                            │ │
│  │         ◀──────────────    (Query DB every time)           │ │
│  │         ~100ms                                             │ │
│  │                                                            │ │
│  │  With cache:                                               │ │
│  │                                                            │ │
│  │  Client ──▶ Server ──▶ Cache (Hit!) ──▶ Response          │ │
│  │         ◀────────────    ~1ms                              │ │
│  │                                                            │ │
│  │  On Cache Miss:                                            │ │
│  │  Server ──▶ Database ──▶ Store in cache ──▶ Response      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  100x faster response possible!                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Cache Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cache Layers                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Client                                                         │
│    │                                                            │
│    ├── Browser cache (fastest)                                 │
│    │     L1: Memory cache                                       │
│    │     L2: Disk cache                                         │
│    │                                                            │
│    ├── CDN cache (Edge)                                         │
│    │     Geographically close servers                           │
│    │                                                            │
│    ├── Reverse proxy cache                                      │
│    │     Nginx, Varnish                                         │
│    │                                                            │
│    ├── Application cache                                        │
│    │     Local cache (in-memory)                                │
│    │     Distributed cache (Redis, Memcached)                   │
│    │                                                            │
│    └── Database cache                                           │
│          Query Cache, Buffer Pool                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │       Fast                              Slow                │ │
│  │  ◀───────────────────────────────────────────────▶         │ │
│  │  Browser → CDN → Proxy → App Cache → DB                    │ │
│  │                                                            │ │
│  │       Small                             Large              │ │
│  │  ◀───────────────────────────────────────────────▶         │ │
│  │        (Storage capacity)                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Cache Hit Rate

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cache Hit Rate                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Cache Hit Rate = (Cache Hits) / (Total Requests) × 100%        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Example: 1000 requests, 950 Hit, 50 Miss                 │ │
│  │  Hit Rate = 950 / 1000 = 95%                               │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Effect by hit rate:                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Hit Rate   │ DB Load Reduction  │ Avg Response Time      │ │
│  │  ───────────┼───────────────────┼────────────────        │ │
│  │  50%        │ 50%               │ 50ms (example)          │ │
│  │  90%        │ 90%               │ 10ms                    │ │
│  │  95%        │ 95%               │ 5ms                     │ │
│  │  99%        │ 99%               │ 2ms                     │ │
│  │                                                            │ │
│  │  High hit rate = Faster response + DB protection          │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Hit rate improvement strategies:                                │
│  • Appropriate TTL configuration                                 │
│  • Cache size optimization                                       │
│  • Hot data preloading                                           │
│  • Cache key design optimization                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Caching Strategy Patterns

> **Analogy -- The Convenience Store**
>
> Think of caching like the difference between a convenience store and a warehouse. The **warehouse** (database) has everything, but it is far away and takes time to retrieve items. The **convenience store** (cache) is right around the corner with a small selection of the most popular items -- you get what you need in seconds.
>
> **Cache-Aside** is like you checking the convenience store first; if they do not have it, you drive to the warehouse and bring a copy back to the store for next time. **Write-Through** is like the store automatically restocking from the warehouse every time a new shipment arrives. **Write-Behind** is like the store accepting new products immediately and sending the paperwork to the warehouse later in batch.
>
> The tricky part? Knowing when the convenience store's stock is outdated -- that is the cache invalidation problem, and it is why Phil Karlton famously said there are only two hard things in computer science: cache invalidation and naming things.

### 2.1 Cache-Aside (Lazy Loading)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cache-Aside Pattern                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Application directly manages cache"                           │
│                                                                  │
│  Read:                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Application                                               │ │
│  │      │                                                     │ │
│  │      │ 1. Check cache                                      │ │
│  │      ▼                                                     │ │
│  │   [Cache] ──Hit?──▶ Yes ──▶ Return data                    │ │
│  │      │                                                     │ │
│  │     No (Miss)                                              │ │
│  │      │                                                     │ │
│  │      │ 2. Query DB                                         │ │
│  │      ▼                                                     │ │
│  │  [Database] ──▶ Return data                                │ │
│  │      │                                                     │ │
│  │      │ 3. Store in cache                                   │ │
│  │      ▼                                                     │ │
│  │   [Cache] (update)                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Write:                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Application                                               │ │
│  │      │                                                     │ │
│  │      │ 1. Write to DB                                      │ │
│  │      ▼                                                     │ │
│  │  [Database] (update)                                       │ │
│  │      │                                                     │ │
│  │      │ 2. Invalidate cache (delete)                        │ │
│  │      ▼                                                     │ │
│  │   [Cache] (delete)                                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Code example:                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  def get_user(user_id):                                    │ │
│  │      # 1. Check cache                                      │ │
│  │      user = cache.get(f"user:{user_id}")                   │ │
│  │      if user:                                              │ │
│  │          return user  # Cache Hit                          │ │
│  │                                                            │ │
│  │      # 2. Query DB (Cache Miss)                            │ │
│  │      user = db.query(f"SELECT * FROM users WHERE id={id}") │ │
│  │                                                            │ │
│  │      # 3. Store in cache                                   │ │
│  │      cache.set(f"user:{user_id}", user, ttl=3600)          │ │
│  │                                                            │ │
│  │      return user                                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                           │
│  • Only cache needed data (memory efficient)                    │
│  • Works with DB even if cache fails                            │
│                                                                  │
│  Cons:                                                           │
│  • First request always slow (Cache Miss)                       │
│  • Possible cache-DB inconsistency                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Annotated implementation:**

```python
def get_user(user_id):
    cache_key = f"user:{user_id}"

    # Step 1: Check cache first — avoids a DB round-trip on the
    # majority of reads (assuming a warm cache with reasonable hit rate)
    user = cache.get(cache_key)
    if user is not None:           # Explicit None check: cached empty results are valid
        return user

    # Step 2: Cache miss — query DB *before* writing to cache,
    # so we cache the actual current value, not a stale or speculative one
    user = db.query("SELECT * FROM users WHERE id = %s", (user_id,))

    # Step 3: Populate cache with TTL.
    # TTL = 3600s (1 hour): balances freshness vs DB load.
    # Without TTL, stale data could persist indefinitely after a write
    # that bypasses cache invalidation (e.g., direct DB migration).
    if user is not None:
        cache.set(cache_key, user, ttl=3600)

    return user


def update_user(user_id, data):
    # Write to DB first — DB is the source of truth.
    # If we invalidated cache first and the DB write failed,
    # the next read would re-cache the old value (acceptable),
    # but if we updated cache first and DB failed, we'd serve wrong data.
    db.execute("UPDATE users SET ... WHERE id = %s", (user_id,))

    # Invalidate (delete) rather than update the cache entry.
    # Delete is safer: avoids race conditions where a concurrent read
    # re-caches stale data between our DB write and cache update.
    cache.delete(f"user:{user_id}")
```

### 2.2 Read-Through

```
┌─────────────────────────────────────────────────────────────────┐
│                   Read-Through Pattern                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Cache handles DB queries"                                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Application                                               │ │
│  │      │                                                     │ │
│  │      │ 1. Request to cache (always)                        │ │
│  │      ▼                                                     │ │
│  │   [Cache] ──Hit?──▶ Yes ──▶ Return data                    │ │
│  │      │                                                     │ │
│  │     No (Miss)                                              │ │
│  │      │                                                     │ │
│  │      │ 2. Cache queries DB (automatic)                     │ │
│  │      ▼                                                     │ │
│  │  [Database]                                                │ │
│  │      │                                                     │ │
│  │      │ 3. Store in cache + return                          │ │
│  │      ▼                                                     │ │
│  │  Application                                               │ │
│  │                                                            │ │
│  │  Application only sees cache!                              │ │
│  │  DB query logic built into cache                           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Cache-Aside vs Read-Through:                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Cache-Aside: App manages both cache and DB                │ │
│  │  Read-Through: App only uses cache, cache manages DB       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                           │
│  • Simplified application code                                  │
│  • Centralized cache logic                                      │
│                                                                  │
│  Cons:                                                           │
│  • Need to implement loader in cache system                     │
│  • Reduced flexibility                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Write-Through

```
┌─────────────────────────────────────────────────────────────────┐
│                  Write-Through Pattern                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Update cache and DB simultaneously on write (synchronous)"    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Application                                               │ │
│  │      │                                                     │ │
│  │      │ 1. Write to cache                                   │ │
│  │      ▼                                                     │ │
│  │   [Cache] ─────────────────────────────────────┐           │ │
│  │      │                                         │           │ │
│  │      │ 2. Cache writes to DB (sync)            │           │ │
│  │      ▼                                         │ Both      │ │
│  │  [Database]                                    │ complete  │ │
│  │      │                                         │           │ │
│  │      │ 3. Respond after completion             │           │ │
│  │      ▼                                         │           │ │
│  │  Application ◀─────────────────────────────────┘           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Characteristics:                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • Cache and DB always synchronized (strong consistency)   │ │
│  │  • Effective when used with Read-Through                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                           │
│  • Data consistency guaranteed                                  │
│  • No data loss                                                  │
│                                                                  │
│  Cons:                                                           │
│  • Write latency (write to two places)                          │
│  • May cache unnecessary data                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Write-Behind (Write-Back)

```
┌─────────────────────────────────────────────────────────────────┐
│                 Write-Behind (Write-Back) Pattern                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Write to cache first, DB later (asynchronous)"                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Application                                               │ │
│  │      │                                                     │ │
│  │      │ 1. Write to cache                                   │ │
│  │      ▼                                                     │ │
│  │   [Cache] ──────────────▶ Immediate response               │ │
│  │      │                                                     │ │
│  │      │ 2. Write to DB later (async)                        │ │
│  │      │    ┌─────────────────────────────┐                  │ │
│  │      │    │ Write Queue                 │                  │ │
│  │      │    │ [data1, data2, data3, ...]  │                  │ │
│  │      │    └─────────────────────────────┘                  │ │
│  │      │              │                                      │ │
│  │      │              │ Batch processing                     │ │
│  │      │              ▼                                      │ │
│  │      └─────────▶ [Database]                                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Timeline:                                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  T=0: Write 1 → Cache (immediate)                          │ │
│  │  T=1: Write 2 → Cache (immediate)                          │ │
│  │  T=2: Write 3 → Cache (immediate)                          │ │
│  │  ...                                                       │ │
│  │  T=10: Batch write 1,2,3 to DB at once                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Pros:                                                           │
│  • Excellent write performance (write to cache and respond)     │
│  • Reduced DB load (batch processing)                           │
│  • Can merge writes (multiple writes to same key → save last)   │
│                                                                  │
│  Cons:                                                           │
│  • Risk of data loss (if cache fails)                           │
│  • Complex implementation                                        │
│  • Consistency issues (data not yet in DB)                      │
│                                                                  │
│  Use cases:                                                      │
│  • Log collection                                                │
│  • View/like counters                                            │
│  • When performance more important than real-time consistency   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Pattern Comparison Summary

| Pattern | Read | Write | Consistency | Performance | Complexity |
|---------|------|-------|------------|----------|-----------|
| Cache-Aside | App manages | Direct to DB, invalidate cache | Weak | Good | Low |
| Read-Through | Cache manages | - | - | Good | Medium |
| Write-Through | - | Cache→DB sync | Strong | Fair | Medium |
| Write-Behind | - | Cache first, DB later | Weak | Excellent | High |

---

## 3. Cache Invalidation

### 3.1 Invalidation Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cache Invalidation Strategies                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "When and how to delete/update cached data?"                   │
│                                                                  │
│  1. TTL (Time To Live) based                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  cache.set("user:123", data, ttl=3600)  # Expires in 1h   │ │
│  │                                                            │ │
│  │  T=0:     [user:123] stored                                │ │
│  │  T=3600:  [user:123] automatically deleted                 │ │
│  │                                                            │ │
│  │  Pros: Simple, automatic                                   │ │
│  │  Cons: May serve stale data during TTL                     │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Explicit Invalidation                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  # On data modification                                    │ │
│  │  db.update_user(user_id, new_data)                         │ │
│  │  cache.delete(f"user:{user_id}")  # Delete cache          │ │
│  │                                                            │ │
│  │  Pros: Immediate consistency recovery                      │ │
│  │  Cons: Need deletion logic, error-prone                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Event-based Invalidation                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  DB Change ──▶ Publish event ──▶ Cache invalidation        │ │
│  │                                                            │ │
│  │  [DB] ──CDC──▶ [Kafka] ──▶ [Cache Invalidator]             │ │
│  │                                                            │ │
│  │  Pros: Loose coupling, scalability                         │ │
│  │  Cons: Increased complexity, possible latency              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 TTL Configuration Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                     TTL Configuration Guide                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Recommended TTL by data characteristics:                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Rarely changing data:                                     │ │
│  │  • Configuration values, metadata: 24 hours ~ 7 days       │ │
│  │  • Country/region info: 7+ days                            │ │
│  │                                                            │ │
│  │  Infrequently changing data:                               │ │
│  │  • User profiles: 1 hour ~ 24 hours                        │ │
│  │  • Product info: 1 hour ~ 6 hours                          │ │
│  │                                                            │ │
│  │  Frequently changing data:                                 │ │
│  │  • Inventory quantity: 1 min ~ 5 min                       │ │
│  │  • Real-time stats: 30 sec ~ 1 min                         │ │
│  │                                                            │ │
│  │  Real-time data:                                           │ │
│  │  • Don't cache or very short TTL (seconds)                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  TTL decision formula:                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  TTL = f(change frequency, consistency requirements,       │ │
│  │           traffic pattern)                                 │ │
│  │                                                            │ │
│  │  • Low change frequency → Longer TTL                       │ │
│  │  • Important consistency → Short TTL or explicit invalidation│ │
│  │  • High traffic → Longer TTL (hit rate important)          │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  TTL Jitter:                                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Problem: Same TTL → Simultaneous expiry → Cache stampede │ │
│  │                                                            │ │
│  │  Solution: Add random value to TTL                         │ │
│  │                                                            │ │
│  │  base_ttl = 3600  # 1 hour                                 │ │
│  │  jitter = random(-300, 300)  # ±5 minutes                  │ │
│  │  ttl = base_ttl + jitter  # 55 min ~ 65 min               │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Cache Problems and Solutions

### 4.1 Cache Penetration

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cache Penetration                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Continuous requests for non-existent data → DB overload"      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Attacker: GET /user?id=-1 (non-existent ID)               │ │
│  │                                                            │ │
│  │  Every time:                                               │ │
│  │  1. Check cache → Miss                                     │ │
│  │  2. Query DB → No result                                   │ │
│  │  3. Don't cache (because empty)                            │ │
│  │  4. Next request repeats same process!                     │ │
│  │                                                            │ │
│  │  Request ──▶ [Cache] ──Miss──▶ [DB] ──▶ Empty              │ │
│  │  Request ──▶ [Cache] ──Miss──▶ [DB] ──▶ Empty (repeat)     │ │
│  │  Request ──▶ [Cache] ──Miss──▶ [DB] ──▶ Empty              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 1: Cache null values                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  data = cache.get(key)                                     │ │
│  │  if data is None:                                          │ │
│  │      data = db.query(key)                                  │ │
│  │      if data is None:                                      │ │
│  │          cache.set(key, "NULL", ttl=60)  # Short TTL       │ │
│  │      else:                                                 │ │
│  │          cache.set(key, data, ttl=3600)                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 2: Bloom Filter                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  [Bloom Filter] ──"Does key possibly exist?"               │ │
│  │       │                                                    │ │
│  │       ├── Definitely doesn't exist → Return (no DB query)  │ │
│  │       │                                                    │ │
│  │       └── Might exist → Query cache/DB                     │ │
│  │                                                            │ │
│  │  Bloom Filter: Memory-efficient existence check            │ │
│  │  False Positive possible, False Negative impossible        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Cache Avalanche

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cache Avalanche                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Mass cache expiry at once → DB surge"                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  T=0: Cache warming (1000 items, TTL=3600)                 │ │
│  │                                                            │ │
│  │  T=3600: 1000 items expire simultaneously!                │ │
│  │          ──▶ 1000 simultaneous DB queries                  │ │
│  │          ──▶ DB overload / failure                         │ │
│  │                                                            │ │
│  │  [Cache] ═════════════════════════════════════             │ │
│  │          │  All Expired!  │                                │ │
│  │          └────────────────┘                                │ │
│  │                   │                                        │ │
│  │          ┌────────┴────────┐                               │ │
│  │          │ Simultaneous    │                               │ │
│  │          │ DB requests     │                               │ │
│  │          │ 1000!          │                                │ │
│  │          └────────────────┘                                │ │
│  │                   │                                        │ │
│  │                   ▼                                        │ │
│  │              [DB] 💥 Overload                              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 1: TTL distribution (Jitter)                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ttl = base_ttl + random(0, 600)  # Add 0~10 min random   │ │
│  │                                                            │ │
│  │  → Expiry times distributed                                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 2: Cache warming (Pre-loading)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • Preload hot data on server start                        │ │
│  │  • Background refresh before expiry                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 3: Multi-layer cache                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  [L1 Cache] ──Miss──▶ [L2 Cache] ──Miss──▶ [DB]            │ │
│  │   (local)             (distributed)                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 4: Mutex/Lock                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  On cache miss, only one request queries DB, others wait  │ │
│  │                                                            │ │
│  │  def get_with_lock(key):                                   │ │
│  │      data = cache.get(key)                                 │ │
│  │      if data:                                              │ │
│  │          return data                                       │ │
│  │                                                            │ │
│  │      if cache.setnx(f"lock:{key}", 1, ttl=10):             │ │
│  │          # Lock acquired → Query DB                        │ │
│  │          data = db.query(key)                              │ │
│  │          cache.set(key, data)                              │ │
│  │          cache.delete(f"lock:{key}")                       │ │
│  │      else:                                                 │ │
│  │          # Lock failed → Wait and retry                    │ │
│  │          sleep(0.1)                                        │ │
│  │          return cache.get(key)                             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Hot Key Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                      Hot Key Problem                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Traffic concentration on specific key → Node overload"        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Example: Celebrity post, flash sale product               │ │
│  │                                                            │ │
│  │       product:12345 (hot product)                          │ │
│  │             │                                              │ │
│  │    ┌────────┼────────┐                                     │ │
│  │    │        │        │                                     │ │
│  │    ▼        ▼        ▼                                     │ │
│  │  [Request] [Request] [Request]  ×10000                     │ │
│  │             │                                              │ │
│  │             ▼                                              │ │
│  │    [Redis Node 1] 💥 Overload                              │ │
│  │                                                            │ │
│  │  Other nodes idle but one node overloaded!                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 1: Add local cache                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  [App Server 1] ──[Local Cache]                            │ │
│  │  [App Server 2] ──[Local Cache]                            │ │
│  │  [App Server 3] ──[Local Cache]                            │ │
│  │          │               │                                 │ │
│  │          └───────────────┘                                 │ │
│  │                  │                                         │ │
│  │                  ▼                                         │ │
│  │            [Redis Cluster]                                 │ │
│  │                                                            │ │
│  │  Distribute hot key with local cache!                      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 2: Key replication                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Original: product:12345                                   │ │
│  │                                                            │ │
│  │  Replicate: product:12345:0                                │ │
│  │             product:12345:1                                │ │
│  │             product:12345:2                                │ │
│  │             ...                                            │ │
│  │             product:12345:N                                │ │
│  │                                                            │ │
│  │  On query: product:12345:{random(0, N)}                    │ │
│  │  → Distributed across multiple nodes!                      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution 3: Read replicas                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  [Master] ──replicate──▶ [Replica 1]                       │ │
│  │               ──▶ [Replica 2]                              │ │
│  │               ──▶ [Replica 3]                              │ │
│  │                                                            │ │
│  │  Distribute read requests across replicas                  │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Problem Summary

| Problem | Cause | Impact | Solution |
|---------|-------|--------|----------|
| Cache Penetration | Query non-existent data | DB overload | Null caching, Bloom Filter |
| Cache Avalanche | Simultaneous expiry | DB surge | TTL Jitter, lock, warming |
| Hot Key | Traffic on specific key | Node overload | Local cache, key replication |

---

## 5. CDN Caching

### 5.1 CDN Caching Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CDN Caching                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Cache static content on edge servers"                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │              ┌─────────────┐                               │ │
│  │              │   Origin    │                               │ │
│  │              │   Server    │                               │ │
│  │              └──────┬──────┘                               │ │
│  │                     │                                      │ │
│  │        ┌────────────┼────────────┐                         │ │
│  │        │            │            │                         │ │
│  │        ▼            ▼            ▼                         │ │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐                   │ │
│  │   │ Edge    │  │ Edge    │  │ Edge    │                   │ │
│  │   │ Seoul   │  │ Tokyo   │  │ NYC     │                   │ │
│  │   └────┬────┘  └────┬────┘  └────┬────┘                   │ │
│  │        │            │            │                         │ │
│  │        ▼            ▼            ▼                         │ │
│  │   Korea users   Japan users   US users                     │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  CDN caching targets:                                            │
│  • Images, CSS, JavaScript                                       │
│  • Fonts, videos                                                 │
│  • API responses (GET, cacheable)                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Cache-Control Headers

```
┌─────────────────────────────────────────────────────────────────┐
│                  Cache-Control Headers                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Main directives:                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  max-age=3600                                              │ │
│  │    Browser/CDN cache time (seconds)                        │ │
│  │                                                            │ │
│  │  s-maxage=86400                                            │ │
│  │    CDN/shared cache specific time (overrides max-age)      │ │
│  │                                                            │ │
│  │  public                                                    │ │
│  │    Can be cached by any cache                              │ │
│  │                                                            │ │
│  │  private                                                   │ │
│  │    Can only be cached by browser (not CDN)                 │ │
│  │                                                            │ │
│  │  no-cache                                                  │ │
│  │    Must validate with origin before using cache            │ │
│  │                                                            │ │
│  │  no-store                                                  │ │
│  │    Do not cache                                            │ │
│  │                                                            │ │
│  │  stale-while-revalidate=60                                 │ │
│  │    Serve stale cache while revalidating in background      │ │
│  │                                                            │ │
│  │  immutable                                                 │ │
│  │    Content never changes (version in URL)                  │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Examples:                                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  # Static assets (1 year)                                  │ │
│  │  Cache-Control: public, max-age=31536000, immutable        │ │
│  │                                                            │ │
│  │  # API response (CDN 1 hour, browser 5 min)                │ │
│  │  Cache-Control: public, max-age=300, s-maxage=3600         │ │
│  │                                                            │ │
│  │  # Personal data (browser only)                            │ │
│  │  Cache-Control: private, max-age=600                       │ │
│  │                                                            │ │
│  │  # Sensitive data (no caching)                             │ │
│  │  Cache-Control: no-store                                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Practice Problems

### Problem 1: Caching Strategy Selection

Choose an appropriate caching pattern for the following scenarios:

a) Read-heavy service (read:write = 100:1)
b) Write-heavy log system
c) Inventory management requiring strong consistency
d) Simple web application

### Problem 2: TTL Configuration

Set appropriate TTL for the following data:

a) User profile (changed several times a day)
b) Product price (real-time fluctuation)
c) Country list (rarely changes)
d) Real-time stock price

### Problem 3: Cache Problem Resolution

Identify the problem and propose a solution for the following situation:

Situation: "Every day at midnight, all cache expires and server becomes slow."

### Problem 4: Write Cache-Control

Write Cache-Control headers for the following requirements:

a) Static image (versioned URL, 1 year cache)
b) API response (CDN 10 min, browser 1 min)
c) Login status check API (no caching)

---

## Answers

### Problem 1 Answer

```
a) Read-heavy service: Cache-Aside
   - Check cache first on read
   - Query DB on cache miss then cache
   - Optimized for reads

b) Write-heavy log system: Write-Behind
   - Write to cache quickly and respond
   - Batch save to DB
   - Maximizes write performance

c) Strong consistency inventory: Write-Through
   - Update cache and DB simultaneously
   - Always consistent inventory count
   - Prevents overselling

d) Simple web application: Cache-Aside
   - Simple implementation
   - Works with DB even if cache fails
   - Most versatile
```

### Problem 2 Answer

```
a) User profile: 1-4 hours
   - Low change frequency
   - Use with explicit invalidation

b) Product price: 1-5 minutes or don't cache
   - Price accuracy important
   - Short TTL or event-based invalidation

c) Country list: 24 hours ~ 7 days
   - Rarely changes
   - Long TTL maximizes cache efficiency

d) Real-time stock price: Don't cache
   - Real-time critical
   - Stale data is risky
```

### Problem 3 Answer

```
Problem: Cache Avalanche
         All cache expires at midnight simultaneously

Solutions:
1. TTL Jitter
   base_ttl = 86400  # 24 hours
   jitter = random(-3600, 3600)  # ±1 hour
   ttl = base_ttl + jitter

2. Cache warming
   - Background refresh before midnight
   - Preload DB query 1 hour before expiry

3. Stale-While-Revalidate
   - Serve stale cache while async refresh

4. Distribute cache refresh
   - Cache items at different times
```

### Problem 4 Answer

```
a) Static image (versioned)
   Cache-Control: public, max-age=31536000, immutable

b) API response
   Cache-Control: public, max-age=60, s-maxage=600

c) Login status API
   Cache-Control: no-store
   (or: Cache-Control: private, no-cache)
```

---

## Hands-On Exercises

### Exercise 1: Cache Strategy Comparison

Use `examples/System_Design/06_cache_strategies.py` to explore cache behavior.

**Tasks:**
1. Run all demos and observe the difference between cache-aside, write-through, and write-back
2. Add a **write-around** strategy: writes go directly to DB, cache is only populated on read miss
3. Compare all four strategies (cache-aside, write-through, write-back, write-around) for a mixed workload: 70% reads, 30% writes on 1000 operations
4. Track metrics: DB reads, DB writes, cache hits, and total "simulated latency" for each strategy

### Exercise 2: Cache Warming

Implement a cache warming strategy for cold-start scenarios.

**Tasks:**
1. Extend the `CacheAside` class with a `warm(keys)` method that pre-loads frequently accessed keys
2. Simulate a cold start: compare performance (hit rate over first 100 requests) with and without warming
3. Implement a "top-K" warming strategy: analyze access logs to find the K most frequently accessed keys, then warm only those
4. What's the optimal K for a cache of size 100 with Zipf-distributed access patterns?

### Exercise 3: Cache Stampede Prevention

Implement techniques to prevent cache stampede (thundering herd).

**Tasks:**
1. Simulate a stampede: 100 concurrent readers all miss the same key simultaneously, causing 100 DB reads
2. Implement **lock-based prevention**: only one reader fetches from DB while others wait
3. Implement **probabilistic early expiration**: refresh cache entries before they actually expire (add jitter to TTL)
4. Compare the three approaches (no protection, locking, early expiration) for DB load and average latency

---

## 7. References

### Documentation
- [Redis Caching](https://redis.io/docs/manual/client-side-caching/)
- [HTTP Caching - MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching)
- [Cloudflare Cache](https://developers.cloudflare.com/cache/)

### Tools
- [Redis](https://redis.io/)
- [Memcached](https://memcached.org/)
- [Varnish Cache](https://varnish-cache.org/)

### References
- [Caching Strategies](https://aws.amazon.com/caching/best-practices/)

---

**Previous**: [Reverse Proxy & API Gateway](./05_Reverse_Proxy_API_Gateway.md) | **Next**: [Distributed Cache Systems](./07_Distributed_Cache_Systems.md)

---

**Document Information**
- Last Modified: 2024
- Difficulty: ⭐⭐⭐
- Estimated Learning Time: 2-3 hours
