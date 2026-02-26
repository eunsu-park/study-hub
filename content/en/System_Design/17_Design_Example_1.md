# Practical Design Examples 1

**Previous**: [Consensus Algorithms](./16_Consensus_Algorithms.md) | **Next**: [Practical Design Examples 2](./18_Design_Example_2.md)

Difficulty: ⭐⭐⭐⭐

## Learning Objectives

1. Apply a structured design methodology (requirements, capacity estimation, high-level design, detailed design) to real-world system problems
2. Design a URL shortener service addressing key generation, collision avoidance, redirection performance, and analytics
3. Design a Pastebin-like content storage service handling variable-size payloads, expiration policies, and access control
4. Design a rate limiter using token bucket, sliding window, and fixed window algorithms, evaluating each algorithm's trade-offs
5. Perform back-of-the-envelope capacity estimation for storage, bandwidth, and QPS requirements
6. Identify and resolve bottlenecks in each design through caching, database partitioning, and replication strategies

---

System design interviews test whether you can synthesize everything you know about distributed systems into a coherent, working architecture under time pressure. The three systems in this lesson -- URL shortener, Pastebin, and rate limiter -- are among the most commonly asked because they are deceptively simple on the surface yet touch on hashing, storage trade-offs, caching, and distributed coordination at scale. Working through these examples builds the structured thinking you will apply to any design problem.

## Table of Contents

1. [URL Shortener](#1-url-shortener)
2. [Pastebin](#2-pastebin)
3. [Rate Limiter](#3-rate-limiter)
4. [Practice Problems](#4-practice-problems)

---

## 1. URL Shortener

### 1.1 Requirements Definition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Functional Requirements                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Core Features:                                                        │
│  1. URL Shortening: Long URL → Short URL generation                    │
│  2. URL Redirection: Short URL → Redirect to original URL              │
│                                                                         │
│  Additional Features:                                                  │
│  3. Custom short URLs (optional)                                       │
│  4. URL expiration time setting                                        │
│  5. Click analytics/statistics                                         │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                     Non-Functional Requirements                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  - High availability: 99.9% uptime                                     │
│  - Low latency: Redirection < 100ms                                    │
│  - Scalability: Store hundreds of millions of URLs                     │
│  - Security: Unpredictable short URLs                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Capacity Estimation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Capacity Estimation (Back-of-envelope)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Assumptions:                                                          │
│  - Monthly new URLs: 100M (100 million)                                │
│  - Read/Write ratio: 100:1                                             │
│  - URL retention period: 5 years                                       │
│  - Average URL length: 100 bytes                                       │
│                                                                         │
│  Calculations:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Write QPS:                                                      │   │
│  │  100M / 30 days / 24 hours / 3600 seconds ≈ 40 writes/sec       │   │
│  │                                                                  │   │
│  │  Read QPS:                                                       │   │
│  │  40 * 100 = 4,000 reads/sec                                     │   │
│  │                                                                  │   │
│  │  Total URLs over 5 years:                                        │   │
│  │  100M * 12 months * 5 years = 6B (6 billion)                    │   │
│  │                                                                  │   │
│  │  Storage capacity:                                               │   │
│  │  6B * (7 bytes short + 100 bytes long) ≈ 640GB                  │   │
│  │                                                                  │   │
│  │  Bandwidth:                                                      │   │
│  │  4,000 reads/sec * 500 bytes = 2 MB/sec                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Determining Short URL Length:                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Base62: [a-zA-Z0-9] = 62 characters                            │   │
│  │                                                                  │   │
│  │  6 characters: 62^6 = 56.8 billion (sufficient!)                │   │
│  │  7 characters: 62^7 = 3.5 trillion                              │   │
│  │                                                                  │   │
│  │  → Use 7 characters (with buffer)                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     System Architecture                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │   Client                                                          │ │
│  │     │                                                             │ │
│  │     ▼                                                             │ │
│  │  ┌─────────────────┐                                              │ │
│  │  │  Load Balancer  │                                              │ │
│  │  └────────┬────────┘                                              │ │
│  │           │                                                       │ │
│  │     ┌─────┴─────┐                                                 │ │
│  │     ▼           ▼                                                 │ │
│  │  ┌─────────┐ ┌─────────┐                                          │ │
│  │  │API Srv 1│ │API Srv 2│ ...                                      │ │
│  │  └────┬────┘ └────┬────┘                                          │ │
│  │       │           │                                               │ │
│  │       └─────┬─────┘                                               │ │
│  │             │                                                     │ │
│  │       ┌─────┴─────┐                                               │ │
│  │       ▼           ▼                                               │ │
│  │  ┌─────────┐ ┌─────────────┐                                      │ │
│  │  │  Cache  │ │   Database  │                                      │ │
│  │  │ (Redis) │ │ (MySQL/Mongo)│                                     │ │
│  │  └─────────┘ └─────────────┘                                      │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  API Design:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  POST /api/shorten                                               │   │
│  │  Body: { "long_url": "https://...", "expiry": "2024-12-31" }    │   │
│  │  Response: { "short_url": "https://tinyurl.com/abc123" }        │   │
│  │                                                                  │   │
│  │  GET /{short_code}                                               │   │
│  │  Response: 301 Redirect to original URL                         │   │
│  │                                                                  │   │
│  │  GET /api/stats/{short_code}                                    │   │
│  │  Response: { "clicks": 1234, "created_at": "..." }              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Detailed Design: Short URL Generation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Method 1: Hash + Collision Handling                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  long_url = "https://example.com/very/long/path"                │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  MD5(long_url) = "e4d909c290d0fb1ca068ffaddf22cbd0"             │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  Base62(first 43 bits) = "abc123d"                              │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  Check collision                                                 │   │
│  │       │                                                          │   │
│  │  ┌────┴────┐                                                     │   │
│  │  │         │                                                     │   │
│  │  ▼         ▼                                                     │   │
│  │ None    Exists                                                   │   │
│  │  │         │                                                     │   │
│  │  │    Add salt to long_url                                      │   │
│  │  │    Re-hash                                                    │   │
│  │  │         │                                                     │   │
│  │  └────┬────┘                                                     │   │
│  │       ▼                                                          │   │
│  │     Store                                                        │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Pros: Same URL → Same short URL (cache efficiency)                   │
│  Cons: Collision handling logic required                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     Method 2: ID Generator (Recommended)               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │                 ID Generator                               │ │   │
│  │  │                                                            │ │   │
│  │  │  Method A: Auto-increment (single DB)                     │ │   │
│  │  │  - Simple but SPOF                                        │ │   │
│  │  │                                                            │ │   │
│  │  │  Method B: Range-based                                    │ │   │
│  │  │  ┌────────────────────────────────────────────────────┐   │ │   │
│  │  │  │  ZooKeeper/etcd                                    │   │ │   │
│  │  │  │  ┌──────────────────────────────────────────────┐  │   │ │   │
│  │  │  │  │ Server 1: 1-1,000,000                        │  │   │ │   │
│  │  │  │  │ Server 2: 1,000,001-2,000,000               │  │   │ │   │
│  │  │  │  │ Server 3: 2,000,001-3,000,000               │  │   │ │   │
│  │  │  │  └──────────────────────────────────────────────┘  │   │ │   │
│  │  │  └────────────────────────────────────────────────────┘   │ │   │
│  │  │                                                            │ │   │
│  │  │  Method C: Snowflake ID                                   │ │   │
│  │  │  [timestamp: 41bits][machine: 10bits][sequence: 12bits]   │ │   │
│  │  │                                                            │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │                           │                                     │   │
│  │                           ▼                                     │   │
│  │              ID = 123456789                                     │   │
│  │                           │                                     │   │
│  │                           ▼                                     │   │
│  │              Base62(123456789) = "8M0kX"                       │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Pros: No collisions, easy to scale                                   │
│  Cons: Same URL gets different short URLs                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.5 Detailed Design: Redirection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Redirection Flow                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GET /abc123                                                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────┐                                                       │
│  │ Load Balancer│                                                       │
│  └──────┬──────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐     ┌─────────────┐                                  │
│  │  API Server │────►│ Redis Cache │                                  │
│  └──────┬──────┘     └──────┬──────┘                                  │
│         │                   │                                          │
│         │              Cache hit?                                      │
│         │                   │                                          │
│         │            ┌──────┴──────┐                                   │
│         │            │             │                                   │
│         │           Yes           No                                   │
│         │            │             │                                   │
│         │            │      ┌──────▼──────┐                           │
│         │            │      │   Database  │                           │
│         │            │      └──────┬──────┘                           │
│         │            │             │                                   │
│         │            │       Store in cache                           │
│         │            │             │                                   │
│         │            └──────┬──────┘                                   │
│         │                   │                                          │
│         ▼                   ▼                                          │
│  ┌─────────────────────────────────────┐                              │
│  │  HTTP 301 (permanent) or 302 (temp) │                              │
│  │  Location: https://original-url.com │                              │
│  └─────────────────────────────────────┘                              │
│                                                                         │
│  301 vs 302:                                                           │
│  - 301: Browser caching → Reduced server load, inaccurate stats       │
│  - 302: Every request hits server → Accurate stats, higher load       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.6 Database Schema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Database Design                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  urls table:                                                           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Column          Type           Description                    │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │  id              BIGINT         PK, auto-increment            │    │
│  │  short_code      VARCHAR(7)     UK, indexed                   │    │
│  │  long_url        VARCHAR(2048)  Original URL                  │    │
│  │  user_id         BIGINT         FK, nullable                  │    │
│  │  created_at      DATETIME       Creation time                 │    │
│  │  expires_at      DATETIME       Expiration time, nullable     │    │
│  │  click_count     BIGINT         Click count                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Indexes:                                                              │
│  - PRIMARY KEY (id)                                                    │
│  - UNIQUE INDEX idx_short_code (short_code)                           │
│  - INDEX idx_user_id (user_id)                                        │
│  - INDEX idx_expires_at (expires_at)                                  │
│                                                                         │
│  click_analytics table (optional):                                     │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  url_id          BIGINT         FK                            │    │
│  │  clicked_at      DATETIME       Click time                    │    │
│  │  ip_address      VARCHAR(45)    IPv6 support                  │    │
│  │  user_agent      VARCHAR(255)   Browser info                  │    │
│  │  referrer        VARCHAR(2048)  Traffic source                │    │
│  │  country         VARCHAR(2)     Country code                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pastebin

### 2.1 Requirements Definition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Functional Requirements                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Core Features:                                                        │
│  1. Paste text and generate URL                                        │
│  2. Retrieve text via URL                                              │
│  3. Set expiration time                                                │
│                                                                         │
│  Additional Features:                                                  │
│  4. Syntax highlighting                                                │
│  5. Password protection                                                │
│  6. One-time view (delete after reading)                               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                     Constraints                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  - Maximum text size: 10MB                                             │
│  - Default expiration: 30 days                                         │
│  - Anonymous users allowed                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Capacity Estimation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Capacity Estimation                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Assumptions:                                                          │
│  - Daily new pastes: 1M                                                │
│  - Read/Write ratio: 5:1                                               │
│  - Average text size: 10KB                                             │
│  - Retention period: 1 year                                            │
│                                                                         │
│  Calculations:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Write QPS: 1M / 86400 ≈ 12 writes/sec                          │   │
│  │  Read QPS: 12 * 5 = 60 reads/sec                                │   │
│  │                                                                  │   │
│  │  Daily storage: 1M * 10KB = 10GB                                │   │
│  │  Annual storage: 10GB * 365 = 3.65TB                            │   │
│  │                                                                  │   │
│  │  Bandwidth:                                                      │   │
│  │  - Write: 12 * 10KB = 120KB/sec                                 │   │
│  │  - Read: 60 * 10KB = 600KB/sec                                  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     System Architecture                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │   Client                                                          │ │
│  │     │                                                             │ │
│  │     ▼                                                             │ │
│  │  ┌─────────────────┐                                              │ │
│  │  │  Load Balancer  │                                              │ │
│  │  └────────┬────────┘                                              │ │
│  │           │                                                       │ │
│  │     ┌─────┴─────┐                                                 │ │
│  │     ▼           ▼                                                 │ │
│  │  ┌─────────┐ ┌─────────┐                                          │ │
│  │  │API Srv 1│ │API Srv 2│                                          │ │
│  │  └────┬────┘ └────┬────┘                                          │ │
│  │       │           │                                               │ │
│  │       └─────┬─────┘                                               │ │
│  │             │                                                     │ │
│  │    ┌────────┼────────┐                                            │ │
│  │    ▼        ▼        ▼                                            │ │
│  │  ┌────┐ ┌───────┐ ┌──────────────┐                               │ │
│  │  │Cache│ │MetaDB │ │Object Storage│                               │ │
│  │  │Redis│ │MySQL  │ │S3/MinIO      │                               │ │
│  │  └────┘ └───────┘ └──────────────┘                               │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Storage Strategy:                                                     │
│  - Metadata (ID, created_at, expires_at): MySQL                       │
│  - Actual text content: Object Storage (S3)                           │
│  - Frequently accessed text: Redis cache                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Detailed Design: Storage Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Data Storage Flow                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Text Storage:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  POST /api/paste                                                 │   │
│  │  { "content": "...", "expires_in": "7d" }                       │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  ┌────────────────────┐                                         │   │
│  │  │ 1. Generate ID     │  (same approach as URL shortener)       │   │
│  │  │    → paste_abc123  │                                         │   │
│  │  └─────────┬──────────┘                                         │   │
│  │            │                                                     │   │
│  │            ▼                                                     │   │
│  │  ┌────────────────────┐     ┌─────────────────────┐             │   │
│  │  │ 2. Store content   │────►│  Object Storage     │             │   │
│  │  │    Apply compress  │     │  Key: paste_abc123  │             │   │
│  │  │    (gzip)          │     │  Value: [gzipped]   │             │   │
│  │  └─────────┬──────────┘     └─────────────────────┘             │   │
│  │            │                                                     │   │
│  │            ▼                                                     │   │
│  │  ┌────────────────────┐     ┌─────────────────────┐             │   │
│  │  │ 3. Store metadata  │────►│  MySQL              │             │   │
│  │  │    (atomic)        │     │  id, created_at,    │             │   │
│  │  │                    │     │  expires_at, size   │             │   │
│  │  └─────────┬──────────┘     └─────────────────────┘             │   │
│  │            │                                                     │   │
│  │            ▼                                                     │   │
│  │  Response: { "url": "https://paste.io/abc123" }                 │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Caching Strategy:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Cache only popular pastes (based on view count)              │   │
│  │  - LRU policy                                                   │   │
│  │  - Cache size: 20% of storage (approximately 700GB)             │   │
│  │  - TTL: 1 hour (frequent refresh)                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Expiration Policy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Expired Data Cleanup                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Method 1: Lazy Deletion                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  GET /abc123                                                     │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  expires_at < now()?                                             │   │
│  │       │                                                          │   │
│  │  ┌────┴────┐                                                     │   │
│  │  │         │                                                     │   │
│  │ Yes       No                                                     │   │
│  │  │         │                                                     │   │
│  │ 404       Return                                                 │   │
│  │ + Add to deletion queue                                         │   │
│  │                                                                  │   │
│  │  Pros: Simple implementation, immediate                         │   │
│  │  Cons: Unaccessed data remains                                  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Method 2: Background Cleanup                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Cron Job (hourly):                                             │   │
│  │                                                                  │   │
│  │  SELECT id, storage_key                                         │   │
│  │  FROM pastes                                                    │   │
│  │  WHERE expires_at < NOW()                                       │   │
│  │  LIMIT 1000;                                                    │   │
│  │                                                                  │   │
│  │  for each expired:                                              │   │
│  │      1. Delete from Object Storage                              │   │
│  │      2. Delete from MySQL                                       │   │
│  │      3. Invalidate Cache                                        │   │
│  │                                                                  │   │
│  │  Pros: Reclaims storage space                                   │   │
│  │  Cons: Avoid peak hours                                         │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Recommended: Combine both methods                                     │
│  - Lazy: Immediate expiration handling                                 │
│  - Background: Regular cleanup to reclaim storage                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Rate Limiter

### 3.1 Requirements Definition

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Functional Requirements                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Core Features:                                                        │
│  1. Request limiting: By IP, user, API key                             │
│  2. Various time windows: Per second, minute, hour                     │
│  3. Return 429 when exceeded                                           │
│                                                                         │
│  Non-Functional Requirements:                                          │
│  - Distributed environment support                                     │
│  - Low latency (checked on every API call)                            │
│  - High availability                                                   │
│  - Accuracy (prevent race conditions)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Algorithm Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Rate Limiting Algorithms                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Token Bucket                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │     ┌───────────────────────────────────────┐                   │   │
│  │     │              Bucket                    │                   │   │
│  │     │   Capacity: 10 tokens                  │                   │   │
│  │     │   ┌───┬───┬───┬───┬───┬───┬───┐       │ ← Token added     │   │
│  │     │   │ ● │ ● │ ● │ ● │ ● │   │   │       │   (1/sec)        │   │
│  │     │   └───┴───┴───┴───┴───┴───┴───┘       │                   │   │
│  │     │        │                               │                   │   │
│  │     └────────┼───────────────────────────────┘                   │   │
│  │              │                                                   │   │
│  │              ▼ Token consumed on request                        │   │
│  │         ┌─────────┐                                              │   │
│  │         │ Request │                                              │   │
│  │         └─────────┘                                              │   │
│  │                                                                  │   │
│  │  Characteristics:                                                │   │
│  │  - Allows bursts (if bucket has tokens)                         │   │
│  │  - Tokens refilled at constant rate                             │   │
│  │  - Memory efficient                                             │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. Leaky Bucket                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │          │ Request inflow                                       │   │
│  │          ▼                                                       │   │
│  │     ┌─────────┐                                                  │   │
│  │     │  Queue  │  ← Reject if queue full                         │   │
│  │     │ (FIFO)  │                                                  │   │
│  │     └────┬────┘                                                  │   │
│  │          │                                                       │   │
│  │          │ Leak (process) at constant rate                      │   │
│  │          ▼                                                       │   │
│  │     ┌─────────┐                                                  │   │
│  │     │ Process │                                                  │   │
│  │     └─────────┘                                                  │   │
│  │                                                                  │   │
│  │  Characteristics:                                                │   │
│  │  - Guarantees constant processing rate                          │   │
│  │  - Smooths out bursts                                           │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  3. Fixed Window                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Time ──────────────────────────────────────────────────────►   │   │
│  │                                                                  │   │
│  │  │◄── Window 1 ──►│◄── Window 2 ──►│◄── Window 3 ──►│          │   │
│  │  │    (limit: 5)  │    (limit: 5)  │    (limit: 5)  │          │   │
│  │  │  ●●●●●         │  ●●            │  ●●●●          │          │   │
│  │  │  count: 5      │  count: 2      │  count: 4      │          │   │
│  │                                                                  │   │
│  │  Problem: Bursts at window boundaries                           │   │
│  │       5 at end of Window 1 + 5 at start of Window 2 = 10!      │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  4. Sliding Window Log                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Store timestamp of each request:                               │   │
│  │  [12:00:01, 12:00:15, 12:00:32, 12:00:45, 12:01:02, ...]        │   │
│  │                                                                  │   │
│  │  Count requests within current time - 1 minute                  │   │
│  │                                                                  │   │
│  │  Pros: Accurate                                                 │   │
│  │  Cons: High memory usage                                        │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  5. Sliding Window Counter (Recommended)                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Weighted average of current + previous window                  │   │
│  │                                                                  │   │
│  │  Previous window: 3 requests                                    │   │
│  │  Current window: 5 requests                                     │   │
│  │  Current position: 70% into window                              │   │
│  │                                                                  │   │
│  │  Estimated count: 3 * 0.3 + 5 = 5.9                            │   │
│  │                                                                  │   │
│  │  Pros: Memory efficient + accurate                              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Rate Limiter Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Deployed as middleware (API Gateway or service level)                 │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │   Client                                                          │ │
│  │     │                                                             │ │
│  │     ▼                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                    API Gateway                               │ │ │
│  │  │  ┌───────────────────────────────────────────────────────┐  │ │ │
│  │  │  │               Rate Limiter Middleware                  │  │ │ │
│  │  │  │                                                        │  │ │ │
│  │  │  │   ┌────────────┐      ┌────────────────────────────┐  │  │ │ │
│  │  │  │   │ Rate Rules │      │     Redis Cluster          │  │  │ │ │
│  │  │  │   │            │─────►│  ┌─────┐ ┌─────┐ ┌─────┐  │  │  │ │ │
│  │  │  │   │ - 100/min  │      │  │Node1│ │Node2│ │Node3│  │  │  │ │ │
│  │  │  │   │ - 1000/hr  │      │  └─────┘ └─────┘ └─────┘  │  │  │ │ │
│  │  │  │   └────────────┘      └────────────────────────────┘  │  │ │ │
│  │  │  │                                                        │  │ │ │
│  │  │  └───────────────────────────────────────────────────────┘  │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │           │                                                       │ │
│  │           │ Allow                  │ Deny                         │ │
│  │           ▼                       ▼                               │ │
│  │  ┌─────────────────┐     ┌─────────────────────┐                 │ │
│  │  │  Backend API    │     │  429 Too Many       │                 │ │
│  │  │  Servers        │     │  Requests           │                 │ │
│  │  └─────────────────┘     │  Retry-After: 60    │                 │ │
│  │                          └─────────────────────┘                 │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Redis Implementation: Token Bucket

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Redis Token Bucket Implementation                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Data Structure:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Key: rate_limit:{user_id}                                      │   │
│  │  Value: HASH                                                    │   │
│  │    - tokens: current token count                                │   │
│  │    - last_refill: last refill time                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Lua Script (atomic execution):                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  local key = KEYS[1]                                             │   │
│  │  local capacity = tonumber(ARGV[1])     -- bucket capacity      │   │
│  │  local refill_rate = tonumber(ARGV[2])  -- tokens per second    │   │
│  │  local now = tonumber(ARGV[3])          -- current time (ms)    │   │
│  │  local requested = tonumber(ARGV[4])    -- requested tokens     │   │
│  │                                                                  │   │
│  │  local bucket = redis.call('HGETALL', key)                      │   │
│  │  local tokens = capacity                                        │   │
│  │  local last_refill = now                                        │   │
│  │                                                                  │   │
│  │  if #bucket > 0 then                                            │   │
│  │      tokens = tonumber(bucket[2])                               │   │
│  │      last_refill = tonumber(bucket[4])                          │   │
│  │  end                                                             │   │
│  │                                                                  │   │
│  │  -- Refill tokens                                                │   │
│  │  local elapsed = (now - last_refill) / 1000                     │   │
│  │  local refill = elapsed * refill_rate                           │   │
│  │  tokens = math.min(capacity, tokens + refill)                   │   │
│  │                                                                  │   │
│  │  -- Process request                                              │   │
│  │  local allowed = 0                                              │   │
│  │  if tokens >= requested then                                    │   │
│  │      tokens = tokens - requested                                │   │
│  │      allowed = 1                                                │   │
│  │  end                                                             │   │
│  │                                                                  │   │
│  │  redis.call('HSET', key, 'tokens', tokens, 'last_refill', now) │   │
│  │  redis.call('EXPIRE', key, capacity / refill_rate * 2)         │   │
│  │                                                                  │   │
│  │  return {allowed, tokens}                                       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Distributed Environment Considerations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Distributed Rate Limiter Issues                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem 1: Race Condition                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Server 1: GET counter → 99                                     │   │
│  │  Server 2: GET counter → 99                                     │   │
│  │  Server 1: SET counter → 100 (allowed)                          │   │
│  │  Server 2: SET counter → 100 (allowed!) ← Limit exceeded!       │   │
│  │                                                                  │   │
│  │  Solution: Atomic execution with Lua Script                     │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Problem 2: Redis Cluster Sync Delay                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Solutions:                                                     │   │
│  │  - Same user goes to same Redis node (Consistent Hashing)      │   │
│  │  - Or tolerate some margin of error                             │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Problem 3: Redis Failure                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Fallback strategies:                                           │   │
│  │  1. Allow all requests (availability priority)                  │   │
│  │  2. Use local cache (reduced accuracy)                          │   │
│  │  3. Deny all requests (security priority)                       │   │
│  │                                                                  │   │
│  │  Recommended: Choose based on service characteristics           │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Rate Limit Rules Configuration:                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  {                                                               │   │
│  │    "rules": [                                                   │   │
│  │      {                                                          │   │
│  │        "key": "user:{user_id}",                                │   │
│  │        "limit": 100,                                           │   │
│  │        "window": "60s"                                         │   │
│  │      },                                                         │   │
│  │      {                                                          │   │
│  │        "key": "ip:{client_ip}",                                │   │
│  │        "limit": 1000,                                          │   │
│  │        "window": "1h"                                          │   │
│  │      },                                                         │   │
│  │      {                                                          │   │
│  │        "key": "api:{api_key}:/expensive-endpoint",             │   │
│  │        "limit": 10,                                            │   │
│  │        "window": "1m"                                          │   │
│  │      }                                                          │   │
│  │    ]                                                            │   │
│  │  }                                                               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Practice Problems

### Exercise 1: URL Shortener Extension

Design extensions to the URL shortener with the following features:
- Redirect to different URLs based on country
- A/B testing support (50% to URL-A, 50% to URL-B)
- Dashboard for analyzing 1 million clicks per day

### Exercise 2: Pastebin Security

Design to meet the following security requirements:
- Password-protected pastes
- Burn after read (auto-delete after viewing)
- Client-side encryption option

### Exercise 3: Dynamic Rate Limiting

Design a Rate Limiter with the following requirements:
- Different limits per user tier (free/paid/enterprise)
- Automatic adjustment during peak hours
- Granular limits per API endpoint

---

**Previous**: [Consensus Algorithms](./16_Consensus_Algorithms.md) | **Next**: [Practical Design Examples 2](./18_Design_Example_2.md)

---

## References

- "System Design Interview" - Alex Xu
- "Designing Data-Intensive Applications" - Martin Kleppmann
- bit.ly, TinyURL Architecture Analysis
- Stripe Rate Limiting Best Practices
- GitHub API Rate Limiting
