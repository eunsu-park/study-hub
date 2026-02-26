# Distributed Cache Systems

**Previous**: [Caching Strategies](./06_Caching_Strategies.md) | **Next**: [Database Scaling](./08_Database_Scaling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why a single-node cache is insufficient for large-scale systems and how distributed caches solve this problem
2. Describe key Redis data structures (strings, hashes, lists, sets, sorted sets) and identify appropriate use cases for each
3. Compare Redis Cluster and Redis Sentinel architectures for high availability and horizontal scaling
4. Evaluate the trade-offs between Redis and Memcached and select the right tool for a given workload
5. Implement consistent hashing with virtual nodes to distribute keys evenly across cache nodes
6. Design a distributed caching tier that handles node failures and rebalancing with minimal disruption

**Difficulty**: ⭐⭐⭐
**Estimated Learning Time**: 2-3 hours
**Prerequisites**: [06_Caching_Strategies.md](./06_Caching_Strategies.md)

---

In the previous lesson you learned caching patterns; now the question becomes: where does that cache actually live when your system spans dozens or hundreds of servers? A single Redis instance can handle impressive throughput, but it is still a single point of failure with limited memory. Distributed cache systems like Redis Cluster and Memcached pools spread data across multiple nodes, giving you both the capacity and resilience that production systems demand.

## Table of Contents

1. [What is Distributed Cache?](#1-what-is-distributed-cache)
2. [Redis Data Structures](#2-redis-data-structures)
3. [Redis Cluster and Sentinel](#3-redis-cluster-and-sentinel)
4. [Memcached Comparison](#4-memcached-comparison)
5. [Consistent Hashing](#5-consistent-hashing)
6. [Practice Problems](#6-practice-problems)
7. [References](#7-references)

---

## 1. What is Distributed Cache?

### 1.1 Need for Distributed Cache

```
┌─────────────────────────────────────────────────────────────────┐
│                   Need for Distributed Cache                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Local cache problems:                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  [Server 1]     [Server 2]     [Server 3]                  │ │
│  │  Local Cache    Local Cache    Local Cache                 │ │
│  │  user:123 ✓     user:123 ✗     user:123 ✗                  │ │
│  │                                                            │ │
│  │  Problems:                                                 │ │
│  │  1. Different cache on each server (inconsistency)         │ │
│  │  2. Memory waste (duplicate data)                          │ │
│  │  3. Difficult cache invalidation                           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  With distributed cache:                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  [Server 1]     [Server 2]     [Server 3]                  │ │
│  │       │              │              │                      │ │
│  │       └──────────────┼──────────────┘                      │ │
│  │                      │                                     │ │
│  │                      ▼                                     │ │
│  │            ┌────────────────────┐                          │ │
│  │            │   Distributed      │                          │ │
│  │            │   Cache (Redis)    │                          │ │
│  │            │   user:123 ✓       │                          │ │
│  │            └────────────────────┘                          │ │
│  │                                                            │ │
│  │  Benefits:                                                 │ │
│  │  1. All servers share same cache                           │ │
│  │  2. Cache consistency maintained                           │ │
│  │  3. Centralized management                                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Local Cache vs Distributed Cache

| Item | Local Cache | Distributed Cache |
|------|-------------|------------------|
| Storage location | Application memory | Separate server (Redis etc.) |
| Speed | Very fast (ns~us) | Fast (ms) |
| Capacity | Server memory limit | Scalable |
| Consistency | Different per server | Shared/consistent |
| Failure | Lost on server restart | Can be managed independently |
| Use case | Single server, read-only | Multiple servers, sessions etc. |

---

## 2. Redis Data Structures

### 2.1 String

```
┌─────────────────────────────────────────────────────────────────┐
│                       String                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Most basic key-value storage                                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  SET user:123:name "John"                                  │ │
│  │  GET user:123:name  → "John"                               │ │
│  │                                                            │ │
│  │  # Set expiry time (seconds)                               │ │
│  │  SET session:abc "data" EX 3600                            │ │
│  │  SETEX session:abc 3600 "data"  # Same                     │ │
│  │                                                            │ │
│  │  # Atomic increment/decrement                              │ │
│  │  SET counter 0                                             │ │
│  │  INCR counter  → 1                                         │ │
│  │  INCRBY counter 10  → 11                                   │ │
│  │  DECR counter  → 10                                        │ │
│  │                                                            │ │
│  │  # NX (Not eXists): Set only if key doesn't exist          │ │
│  │  SET lock:resource "owner" NX EX 30  # Distributed lock    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Use cases:                                                      │
│  • Session storage                                               │
│  • Cache (JSON serialization)                                    │
│  • Counters (views, likes)                                       │
│  • Distributed locks                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Hash

```
┌─────────────────────────────────────────────────────────────────┐
│                        Hash                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Collection of field-value pairs (good for storing objects)     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  # Store user object                                       │ │
│  │  HSET user:123 name "John" age 30 email "john@example.com" │ │
│  │                                                            │ │
│  │  # Get single field                                        │ │
│  │  HGET user:123 name  → "John"                              │ │
│  │                                                            │ │
│  │  # Get all fields                                          │ │
│  │  HGETALL user:123                                          │ │
│  │  → {"name": "John", "age": "30", "email": "..."}           │ │
│  │                                                            │ │
│  │  # Increment field                                         │ │
│  │  HINCRBY user:123 age 1  → 31                              │ │
│  │                                                            │ │
│  │  # Check field existence                                   │ │
│  │  HEXISTS user:123 name  → 1                                │ │
│  │                                                            │ │
│  │  Structure:                                                │ │
│  │  ┌─────────────────────────────────────────────────┐       │ │
│  │  │ user:123                                        │       │ │
│  │  │   ├── name: "John"                              │       │ │
│  │  │   ├── age: 30                                   │       │ │
│  │  │   └── email: "john@example.com"                 │       │ │
│  │  └─────────────────────────────────────────────────┘       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Advantages (vs String + JSON):                                 │
│  • Access/modify individual fields                              │
│  • Efficient partial updates                                    │
│  • Memory efficient (for small hashes)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 List

```
┌─────────────────────────────────────────────────────────────────┐
│                        List                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Ordered collection of strings (Linked List)                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  # Push to right (Queue use)                               │ │
│  │  RPUSH queue:jobs "job1" "job2" "job3"                     │ │
│  │                                                            │ │
│  │  # Pop from left                                           │ │
│  │  LPOP queue:jobs  → "job1"                                 │ │
│  │                                                            │ │
│  │  # Push to left (Stack use)                                │ │
│  │  LPUSH stack:items "item1"                                 │ │
│  │  LPOP stack:items  → "item1"                               │ │
│  │                                                            │ │
│  │  # Range query                                             │ │
│  │  LRANGE queue:jobs 0 -1  → ["job2", "job3"]                │ │
│  │                                                            │ │
│  │  # Blocking Pop (for job queue)                            │ │
│  │  BLPOP queue:jobs 30  # Wait 30 seconds                    │ │
│  │                                                            │ │
│  │  Structure:                                                │ │
│  │  ┌─────────────────────────────────────────────────┐       │ │
│  │  │ queue:jobs                                      │       │ │
│  │  │   [job2] ←→ [job3]                              │       │ │
│  │  │    Head        Tail                             │       │ │
│  │  └─────────────────────────────────────────────────┘       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Use cases:                                                      │
│  • Job Queue                                                     │
│  • Recent items list (recently viewed products)                 │
│  • Timeline                                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Set

```
┌─────────────────────────────────────────────────────────────────┐
│                         Set                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Unordered collection of unique strings                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  # Add members                                             │ │
│  │  SADD user:123:followers "user:456" "user:789"             │ │
│  │                                                            │ │
│  │  # Check membership                                        │ │
│  │  SISMEMBER user:123:followers "user:456"  → 1              │ │
│  │                                                            │ │
│  │  # Get all members                                         │ │
│  │  SMEMBERS user:123:followers  → {"user:456", "user:789"}   │ │
│  │                                                            │ │
│  │  # Set operations                                          │ │
│  │  SINTER user:123:followers user:456:followers  # Intersection│ │
│  │  SUNION user:123:followers user:456:followers  # Union     │ │
│  │  SDIFF user:123:followers user:456:followers   # Difference│ │
│  │                                                            │ │
│  │  # Count                                                   │ │
│  │  SCARD user:123:followers  → 2                             │ │
│  │                                                            │ │
│  │  Structure:                                                │ │
│  │  ┌─────────────────────────────────────────────────┐       │ │
│  │  │ user:123:followers                              │       │ │
│  │  │   { "user:456", "user:789" }                    │       │ │
│  │  └─────────────────────────────────────────────────┘       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Use cases:                                                      │
│  • Tags                                                          │
│  • Followers/Following                                           │
│  • Likes user list                                               │
│  • Unique visitors                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Sorted Set (ZSet)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sorted Set                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Set sorted by score                                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  # Add with scores                                         │ │
│  │  ZADD leaderboard 100 "user:123" 85 "user:456" 92 "user:789"│ │
│  │                                                            │ │
│  │  # Range query (ascending)                                 │ │
│  │  ZRANGE leaderboard 0 2 WITHSCORES                         │ │
│  │  → [("user:456", 85), ("user:789", 92), ("user:123", 100)] │ │
│  │                                                            │ │
│  │  # Range query (descending) - Top 3                        │ │
│  │  ZREVRANGE leaderboard 0 2 WITHSCORES                      │ │
│  │  → [("user:123", 100), ("user:789", 92), ("user:456", 85)] │ │
│  │                                                            │ │
│  │  # Score range query                                       │ │
│  │  ZRANGEBYSCORE leaderboard 80 95 WITHSCORES                │ │
│  │                                                            │ │
│  │  # Rank query                                              │ │
│  │  ZRANK leaderboard "user:123"  → 2 (0-based)               │ │
│  │  ZREVRANK leaderboard "user:123"  → 0 (1st place)          │ │
│  │                                                            │ │
│  │  # Increment score                                         │ │
│  │  ZINCRBY leaderboard 10 "user:456"  → 95                   │ │
│  │                                                            │ │
│  │  Structure:                                                │ │
│  │  ┌─────────────────────────────────────────────────┐       │ │
│  │  │ leaderboard                                     │       │ │
│  │  │   score: 85  → "user:456"                       │       │ │
│  │  │   score: 92  → "user:789"                       │       │ │
│  │  │   score: 100 → "user:123"                       │       │ │
│  │  └─────────────────────────────────────────────────┘       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Use cases:                                                      │
│  • Leaderboard                                                   │
│  • Priority queue                                                │
│  • Timeline sorted by time (score = timestamp)                  │
│  • Rate Limiting (sliding window)                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.6 Data Structure Summary

| Type | Structure | Time Complexity | Use Case |
|------|-----------|----------------|---------|
| String | Key-Value | O(1) | Cache, session, counter |
| Hash | Field-Value Map | O(1) | Objects, user profiles |
| List | Linked List | O(1) Push/Pop | Queue, recent items |
| Set | HashSet | O(1) | Tags, unique visitors |
| Sorted Set | Skip List | O(log N) | Leaderboard, ranking |

---

## 3. Redis Cluster and Sentinel

### 3.1 Redis Sentinel

```
┌─────────────────────────────────────────────────────────────────┐
│                    Redis Sentinel                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Monitoring and automatic failover for Redis high availability"│
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │        ┌─────────────────────────────────────┐             │ │
│  │        │         Sentinels                   │             │ │
│  │        │  ┌───────┐ ┌───────┐ ┌───────┐     │             │ │
│  │        │  │Sent 1 │ │Sent 2 │ │Sent 3 │     │             │ │
│  │        │  └───┬───┘ └───┬───┘ └───┬───┘     │             │ │
│  │        │      │         │         │         │             │ │
│  │        └──────┼─────────┼─────────┼─────────┘             │ │
│  │               │         │         │                        │ │
│  │        ┌──────┴─────────┴─────────┴──────┐                 │ │
│  │        │                                 │                 │ │
│  │        ▼                                 ▼                 │ │
│  │   ┌─────────┐                      ┌─────────┐             │ │
│  │   │ Master  │ ════replication══▶   │ Replica │             │ │
│  │   │  (R/W)  │                      │  (R/O)  │             │ │
│  │   └─────────┘                      └─────────┘             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Roles:                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. Monitoring: Watch Master/Replica status                │ │
│  │  2. Notification: Alert admins on failure                  │ │
│  │  3. Automatic failover: Promote Replica to Master on failure│ │
│  │  4. Configuration provider: Provide current Master address │ │
│  │     to clients                                             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Failover process:                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. Detect Master failure (majority Sentinel agreement)    │ │
│  │  2. Elect one Sentinel as leader                           │ │
│  │  3. Leader promotes one Replica to Master                  │ │
│  │  4. Other Replicas connect to new Master                   │ │
│  │  5. Notify clients of new Master address                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Limitations:                                                    │
│  • No horizontal scaling (single Master)                        │
│  • Writes only to Master                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Redis Cluster

```
┌─────────────────────────────────────────────────────────────────┐
│                    Redis Cluster                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Redis horizontal scaling and automatic sharding"              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  16384 Hash Slots distribution:                            │ │
│  │                                                            │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │ │
│  │  │   Node A    │ │   Node B    │ │   Node C    │          │ │
│  │  │ Master      │ │ Master      │ │ Master      │          │ │
│  │  │ Slots:      │ │ Slots:      │ │ Slots:      │          │ │
│  │  │ 0-5460      │ │ 5461-10922  │ │ 10923-16383 │          │ │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘          │ │
│  │         │               │               │                  │ │
│  │         ▼               ▼               ▼                  │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │ │
│  │  │  Replica A  │ │  Replica B  │ │  Replica C  │          │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘          │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Hash Slot calculation:                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  slot = CRC16(key) % 16384                                 │ │
│  │                                                            │ │
│  │  Example: key = "user:123"                                 │ │
│  │      CRC16("user:123") = 12345                             │ │
│  │      slot = 12345 % 16384 = 12345 → Node C                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Hash Tag (force same slot):                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  # {user:123} is hash target                               │ │
│  │  user:{user:123}:profile                                   │ │
│  │  user:{user:123}:settings                                  │ │
│  │  user:{user:123}:orders                                    │ │
│  │                                                            │ │
│  │  → All stored in same slot                                 │ │
│  │  → Multi-key operations possible                           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Features:                                                       │
│  • Horizontal scaling (add nodes)                               │
│  • Automatic sharding                                            │
│  • High availability (auto Replica failover)                    │
│  • Some limitations: Multi-key ops only within same slot        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Sentinel vs Cluster

| Item | Sentinel | Cluster |
|------|----------|---------|
| Purpose | High availability | HA + Horizontal scaling |
| Sharding | None (single Master) | Auto sharding |
| Capacity | Single server memory | Distributed storage |
| Complexity | Low | High |
| Multi-key ops | Free | Hash Tag needed |
| Suitable for | Small-medium scale | Large scale |

---

## 4. Memcached Comparison

### 4.1 Redis vs Memcached

```
┌─────────────────────────────────────────────────────────────────┐
│                  Redis vs Memcached                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Redis:                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • Various data structures (String, Hash, List, Set, ZSet) │ │
│  │  • Persistence options (RDB, AOF)                          │ │
│  │  • Replication and cluster support                         │ │
│  │  • Pub/Sub, Lua scripts                                    │ │
│  │  • Transactions (MULTI/EXEC)                               │ │
│  │  • Single-threaded (event loop)                            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Memcached:                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • Simple key-value storage (String only)                  │ │
│  │  • No persistence (pure cache)                             │ │
│  │  • Multi-threaded (utilizes multi-core)                    │ │
│  │  • Simple and lightweight                                  │ │
│  │  • LRU cache policy                                        │ │
│  │  • Slab allocator (memory efficient)                       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Comparison Table

| Item | Redis | Memcached |
|------|-------|-----------|
| Data structures | Various | String only |
| Persistence | RDB, AOF | None |
| Replication | Supported | None |
| Cluster | Supported | Client sharding |
| Threading | Single (6.0+ multi) | Multi |
| Memory efficiency | Good | Very good |
| Max value size | 512MB | 1MB |
| Use cases | General purpose, session, queue | Simple cache |

### 4.3 Selection Criteria

```
┌─────────────────────────────────────────────────────────────────┐
│                     Selection Guide                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Choose Redis when:                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Need complex data structures (List, Set, Sorted Set)     │ │
│  │ • Need data persistence                                    │ │
│  │ • Need Pub/Sub, message queue features                     │ │
│  │ • Need replication and high availability                   │ │
│  │ • Session storage                                          │ │
│  │ • Leaderboard, Rate Limiting                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Choose Memcached when:                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Only need simple key-value cache                         │ │
│  │ • Multi-core utilization important                         │ │
│  │ • Memory efficiency critical                               │ │
│  │ • Persistence not needed                                   │ │
│  │ • Very high throughput needed                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Most cases recommend Redis (feature diversity, ecosystem)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Consistent Hashing

### 5.1 Traditional Hashing Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                   Traditional Hashing Problem                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Modular hashing: hash(key) % N                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  3 servers:                                                │ │
│  │  hash("user:1") % 3 = 1 → Server 1                         │ │
│  │  hash("user:2") % 3 = 2 → Server 2                         │ │
│  │  hash("user:3") % 3 = 0 → Server 0                         │ │
│  │                                                            │ │
│  │  Add 1 server (4 total):                                   │ │
│  │  hash("user:1") % 4 = 0 → Server 0  (changed!)             │ │
│  │  hash("user:2") % 4 = 2 → Server 2                         │ │
│  │  hash("user:3") % 4 = 3 → Server 3  (changed!)             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Problem: Most keys redistributed on server add/remove!         │
│        → Cache miss surge → Database load                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  N servers → N+1 servers: N/(N+1) keys redistributed       │ │
│  │  Example: 100 → 101 servers: ~99% keys redistributed!      │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Consistent Hashing Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consistent Hashing                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Minimize key redistribution on server add/remove"             │
│                                                                  │
│  Hash Ring:                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │                         0                                  │ │
│  │                         │                                  │ │
│  │                    ┌────┴────┐                             │ │
│  │                    │         │                             │ │
│  │           Node A ──●         ●── key1                      │ │
│  │                   /           \                            │ │
│  │                  /             \                           │ │
│  │       key2 ●───●                 ●─── Node B               │ │
│  │              Node C               \                        │ │
│  │                  \                 \                       │ │
│  │                   \                 ●── key3               │ │
│  │                    \               /                       │ │
│  │                     ●─────────────●                        │ │
│  │                                                            │ │
│  │  Key → Assigned to first node clockwise                   │ │
│  │                                                            │ │
│  │  key1 → Node B                                             │ │
│  │  key2 → Node C                                             │ │
│  │  key3 → Node A                                             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  When adding node:                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │           Node A ──●                                       │ │
│  │                   /                                        │ │
│  │       key2 ●───●                    ●── Node D (newly added)│ │
│  │              Node C                 │                      │ │
│  │                                     ●── key1 (moved to D)  │ │
│  │                                     │                      │ │
│  │                    ●─────────────●──┘                      │ │
│  │                         Node B                             │ │
│  │                                                            │ │
│  │  Only key1 moved: Node B → Node D                          │ │
│  │  Others stay!                                              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Theory: Only K/N keys redistributed (K=total keys, N=nodes)    │
│  Example: 100 nodes → 101: Only ~1% redistributed!              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Virtual Nodes

```
┌─────────────────────────────────────────────────────────────────┐
│                    Virtual Nodes                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Problem: Uneven distribution with few nodes                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  With only 2 nodes:                                        │ │
│  │                                                            │ │
│  │           Node A ──●                                       │ │
│  │                   /   \                                    │ │
│  │                  /     \                                   │ │
│  │                 /       \                                  │ │
│  │                /    Many keys                              │ │
│  │               /     go to                                  │ │
│  │              ●───── Node B                                 │ │
│  │                                                            │ │
│  │  → Uneven!                                                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solution: Use virtual nodes                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Represent each physical node as multiple virtual nodes   │ │
│  │                                                            │ │
│  │  Node A → A-1, A-2, A-3, A-4, ...                          │ │
│  │  Node B → B-1, B-2, B-3, B-4, ...                          │ │
│  │                                                            │ │
│  │            A-1 ●                                           │ │
│  │               /  B-2 ●                                     │ │
│  │              /       / A-3 ●                               │ │
│  │         B-1 ●       /     /                                │ │
│  │                    /     /                                 │ │
│  │              A-2 ●─────●── B-3                             │ │
│  │                                                            │ │
│  │  → More even distribution!                                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Virtual node count:                                             │
│  • More nodes = more even, but higher memory usage              │
│  • Typically 100-200 virtual nodes per physical node            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Practice Problems

### Problem 1: Redis Data Structure Selection

Choose the appropriate Redis data structure for the following requirements:

a) Store user session (ID, name, permissions, etc.)
b) Real-time game leaderboard
c) Store 10 recently viewed products
d) List of posts user liked
e) Chat room message queue

### Problem 2: Sentinel vs Cluster

Choose between Sentinel and Cluster for the following scenarios:

a) 10GB data, high availability needed
b) 1TB data, horizontal scaling needed
c) Service with many complex transactions
d) Simple cache, 1 million requests per second

### Problem 3: Consistent Hashing

With 4 cache servers, when adding 1 server:
a) What % of keys redistributed with traditional hashing?
b) What % of keys redistributed with consistent hashing?

### Problem 4: Redis Design

Design a social media follow feature using Redis.

Requirements:
- User A follows B
- View A's following list
- View B's follower list
- Find common following between A and B

---

## Answers

### Problem 1 Answer

```
a) User session: Hash
   HSET session:abc123 user_id 123 name "John" role "admin"

b) Leaderboard: Sorted Set
   ZADD leaderboard 1000 "user:123" 950 "user:456"

c) Recently viewed: List (with size limit)
   LPUSH user:123:recent "product:789"
   LTRIM user:123:recent 0 9  # Keep only 10

d) Liked posts: Set
   SADD user:123:likes "post:456" "post:789"

e) Chat message queue: List
   RPUSH chat:room1 "{message...}"
   BLPOP chat:room1 0  # Consumer waits
```

### Problem 2 Answer

```
a) 10GB, HA: Sentinel
   - Data size fits single server
   - Sentinel provides HA
   - Simple configuration

b) 1TB, horizontal scaling: Cluster
   - Large data needs distributed storage
   - Horizontal scaling for throughput

c) Complex transactions: Sentinel
   - Cluster has multi-key transaction limitations
   - Single Master easier for transactions

d) Simple cache, high throughput: Cluster or Memcached
   - Cluster for load distribution
   - Memcached also viable for simple cache
```

### Problem 3 Answer

```
a) Traditional hashing: ~80%
   N/(N+1) = 4/5 = 80% keys redistributed

b) Consistent hashing: ~20%
   K/N = 1/5 = 20% keys only
   (keys in new node's range)
```

### Problem 4 Answer

```redis
# Follow relationship
# A follows B
SADD user:A:following "B"
SADD user:B:followers "A"

# A's following list
SMEMBERS user:A:following

# B's follower list
SMEMBERS user:B:followers

# Common following between A and B
SINTER user:A:following user:B:following

# Follower count
SCARD user:B:followers

# Check if following
SISMEMBER user:A:following "B"

# Unfollow
SREM user:A:following "B"
SREM user:B:followers "A"
```

---

## Hands-On Exercises

### Exercise 1: Consistent Hashing Deep Dive

Use `examples/System_Design/07_consistent_hashing.py` to explore consistent hashing.

**Tasks:**
1. Run all demos and observe the impact of virtual nodes on distribution balance
2. Experiment: what's the minimum number of virtual nodes needed to keep all servers within ±5% of ideal distribution for 10,000 keys?
3. Implement **weighted consistent hashing**: give a "large" server 2× the virtual nodes of a "small" server. Verify it gets ~2× the keys
4. Simulate a rolling deployment: add 1 node, verify redistribution, add another. Track cumulative key movement

### Exercise 2: Cache Cluster Replication

Build a multi-node cache cluster with replication.

**Tasks:**
1. Create 3 cache nodes using consistent hashing for primary assignment
2. For each key, replicate to N-1 clockwise neighbors on the ring (replication factor = 2)
3. Implement read: try primary, fall back to replica on failure
4. Simulate node failure: kill one node, verify all keys are still readable from replicas
5. Measure read latency (1 hop vs. 2 hops) and write amplification (writes × replication factor)

### Exercise 3: Cache Eviction Policy Comparison

Compare eviction policies for a distributed cache workload.

**Tasks:**
1. Implement LRU, LFU, and Random eviction in a cache with capacity 100
2. Generate Zipf-distributed access patterns (80/20 rule): 20% of keys get 80% of accesses
3. Run 10,000 requests and compare hit rates for each policy
4. Repeat with a uniform access pattern. Which policy benefits most from skewed access?

---

## 7. References

### Official Documentation
- [Redis Documentation](https://redis.io/documentation)
- [Memcached Wiki](https://github.com/memcached/memcached/wiki)

### Tools
- [Redis Commander](https://github.com/joeferner/redis-commander) - GUI
- [RedisInsight](https://redis.com/redis-enterprise/redis-insight/) - Official GUI

### Papers
- "Consistent Hashing and Random Trees" - Karger et al.

---

**Previous**: [Caching Strategies](./06_Caching_Strategies.md) | **Next**: [Database Scaling](./08_Database_Scaling.md)

---

**Document Information**
- Last Modified: 2024
- Difficulty: ⭐⭐⭐
- Estimated Learning Time: 2-3 hours
