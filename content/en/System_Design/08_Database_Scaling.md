# Database Scaling

**Previous**: [Distributed Cache Systems](./07_Distributed_Cache_Systems.md) | **Next**: [Database Replication](./09_Database_Replication.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why database scaling becomes necessary as data volume and query throughput grow beyond single-node capacity
2. Distinguish between partitioning (single database) and sharding (multiple databases) and describe when each is appropriate
3. Compare sharding strategies -- range-based, hash-based, and directory-based -- and evaluate their trade-offs for different access patterns
4. Select an effective shard key by analyzing query patterns, data distribution, and growth projections
5. Identify and prevent hotspot problems caused by uneven data distribution or skewed access patterns
6. Design a rebalancing strategy that redistributes data across shards with minimal downtime

**Difficulty**: ⭐⭐⭐
**Estimated Learning Time**: 2-3 hours
**Prerequisites**: [07_Distributed_Cache_Systems.md](./07_Distributed_Cache_Systems.md), [PostgreSQL Folder](../PostgreSQL/00_Overview.md)

---

Databases are often the hardest component to scale because, unlike stateless application servers, they hold state that must remain consistent and durable. When a single database can no longer keep up -- whether limited by storage capacity, write throughput, or query latency -- you need to split the data across multiple machines. Sharding is powerful but introduces complexity in routing, joins, and rebalancing that every senior engineer must understand.

## Table of Contents

1. [The Need for Database Scaling](#1-the-need-for-database-scaling)
2. [Partitioning vs Sharding](#2-partitioning-vs-sharding)
3. [Sharding Strategies](#3-sharding-strategies)
4. [Shard Key Selection](#4-shard-key-selection)
5. [Hotspot Prevention](#5-hotspot-prevention)
6. [Rebalancing](#6-rebalancing)
7. [Practice Problems](#7-practice-problems)
8. [References](#8-references)

---

## 1. The Need for Database Scaling

### 1.1 Limitations of a Single Database

```
┌─────────────────────────────────────────────────────────────────┐
│              Single Database Limitations                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Issues with Traffic/Data Growth:                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. Storage Capacity Limits                                │ │
│  │     • Physical limits of single disk/server                │ │
│  │     • Difficult to manage data exceeding tens of TB        │ │
│  │                                                            │ │
│  │  2. Throughput Limits                                      │ │
│  │     • Single server CPU/memory limitations                 │ │
│  │     • Limited concurrent connections                       │ │
│  │                                                            │ │
│  │  3. Increased Response Time                                │ │
│  │     • Large tables → Slow queries                          │ │
│  │     • Growing index sizes                                  │ │
│  │                                                            │ │
│  │  4. Single Point of Failure (SPOF)                         │ │
│  │     • Server failure = Entire service failure              │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Solutions:                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Vertical Scaling (Scale Up)                               │ │
│  │    → More powerful hardware (has limits)                   │ │
│  │                                                            │ │
│  │  Horizontal Scaling (Scale Out)                            │ │
│  │    → Distribute across multiple servers (partitioning/     │ │
│  │      sharding)                                             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Scaling Strategy Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 Database Scaling Strategies                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. Read Replica                                           │ │
│  │     • Distribute read load                                 │ │
│  │     • Covered in detail in next lesson                     │ │
│  │                                                            │ │
│  │  2. Partitioning                                           │ │
│  │     • Split tables within a single DB                      │ │
│  │     • PostgreSQL native partitioning                       │ │
│  │                                                            │ │
│  │  3. Sharding                                               │ │
│  │     • Distribute data across multiple DB servers           │ │
│  │     • Core of horizontal scaling                           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│                    [Single DB]                                   │
│                        │                                        │
│         ┌──────────────┼──────────────┐                         │
│         ▼              ▼              ▼                         │
│    [Replication]  [Partitioning]   [Sharding]                   │
│    Read scaling   Single DB split  Multi-DB distributed         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Partitioning vs Sharding

### 2.1 Partitioning

```
┌─────────────────────────────────────────────────────────────────┐
│                        Partitioning                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Splitting a table into multiple partitions within a single    │
│   DB server"                                                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  ┌───────────────────────────────────────────────────────┐ │ │
│  │  │              Database Server                          │ │ │
│  │  │                                                       │ │ │
│  │  │  ┌─────────────────────────────────────────────────┐  │ │ │
│  │  │  │              orders (logical table)             │  │ │ │
│  │  │  └─────────────────────────────────────────────────┘  │ │ │
│  │  │                        │                              │ │ │
│  │  │         ┌──────────────┼──────────────┐               │ │ │
│  │  │         ▼              ▼              ▼               │ │ │
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐         │ │ │
│  │  │  │orders_2022│  │orders_2023│  │orders_2024│         │ │ │
│  │  │  │(partition1)│  │(partition2)│  │(partition3)│         │ │ │
│  │  │  └───────────┘  └───────────┘  └───────────┘         │ │ │
│  │  │                                                       │ │ │
│  │  └───────────────────────────────────────────────────────┘ │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PostgreSQL Example:                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  -- Create partitioned table                               │ │
│  │  CREATE TABLE orders (                                     │ │
│  │      id SERIAL,                                            │ │
│  │      order_date DATE,                                      │ │
│  │      amount DECIMAL                                        │ │
│  │  ) PARTITION BY RANGE (order_date);                        │ │
│  │                                                            │ │
│  │  -- Create partition                                       │ │
│  │  CREATE TABLE orders_2024 PARTITION OF orders              │ │
│  │      FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Advantages:                                                     │
│  • Improved query performance (partition pruning)               │
│  • Easier data management (delete/backup per partition)         │
│  • Reduced index size                                           │
│                                                                  │
│  Limitations:                                                    │
│  • Still a single server (capacity, throughput limits)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Sharding

```
┌─────────────────────────────────────────────────────────────────┐
│                          Sharding                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Distributing data across multiple DB servers"                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │         ┌─────────────────────────────────────────┐        │ │
│  │         │           Application                   │        │ │
│  │         └──────────────────┬──────────────────────┘        │ │
│  │                            │                               │ │
│  │                   ┌────────┴────────┐                      │ │
│  │                   │  Shard Router   │                      │ │
│  │                   │ (Shard Selector)│                      │ │
│  │                   └────────┬────────┘                      │ │
│  │                            │                               │ │
│  │         ┌──────────────────┼──────────────────┐            │ │
│  │         │                  │                  │            │ │
│  │         ▼                  ▼                  ▼            │ │
│  │  ┌────────────┐     ┌────────────┐     ┌────────────┐     │ │
│  │  │  Shard 1   │     │  Shard 2   │     │  Shard 3   │     │ │
│  │  │ (Server 1) │     │ (Server 2) │     │ (Server 3) │     │ │
│  │  │            │     │            │     │            │     │ │
│  │  │ user_id    │     │ user_id    │     │ user_id    │     │ │
│  │  │ 1-1000000  │     │ 1000001-   │     │ 2000001-   │     │ │
│  │  │            │     │ 2000000    │     │ 3000000    │     │ │
│  │  └────────────┘     └────────────┘     └────────────┘     │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Advantages:                                                     │
│  • Theoretically unlimited scaling                              │
│  • Reduced load on each shard                                   │
│  • Fault isolation                                              │
│                                                                  │
│  Disadvantages:                                                  │
│  • Increased complexity                                         │
│  • Difficult cross-shard queries                                │
│  • Transaction constraints                                      │
│  • Difficult rebalancing                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Comparison

| Aspect | Partitioning | Sharding |
|--------|--------------|----------|
| Location | Within single server | Across multiple servers |
| Scalability | Limited | High |
| Complexity | Low | High |
| Transactions | Full support | Within shard only |
| JOIN | Possible | Cross-shard difficult |
| Management | DB handles automatically | Application manages |

---

## 3. Sharding Strategies

### 3.1 Range-Based Sharding

```
┌─────────────────────────────────────────────────────────────────┐
│                    Range-Based Sharding                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Determine shard based on value ranges"                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  user_id based Range sharding:                             │ │
│  │                                                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │  Shard 1    │  │  Shard 2    │  │  Shard 3    │        │ │
│  │  │             │  │             │  │             │        │ │
│  │  │ user_id     │  │ user_id     │  │ user_id     │        │ │
│  │  │ 1 ~ 1M      │  │ 1M+1 ~ 2M   │  │ 2M+1 ~ 3M   │        │ │
│  │  │             │  │             │  │             │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  │                                                            │ │
│  │  Date based Range sharding:                                │ │
│  │                                                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │  Shard 1    │  │  Shard 2    │  │  Shard 3    │        │ │
│  │  │             │  │             │  │             │        │ │
│  │  │ 2022        │  │ 2023        │  │ 2024        │        │ │
│  │  │ data        │  │ data        │  │ data        │        │ │
│  │  │             │  │             │  │             │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Routing Logic:                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  def get_shard(user_id):                                   │ │
│  │      if user_id <= 1_000_000:                              │ │
│  │          return "shard_1"                                  │ │
│  │      elif user_id <= 2_000_000:                            │ │
│  │          return "shard_2"                                  │ │
│  │      else:                                                 │ │
│  │          return "shard_3"                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Advantages:                                                     │
│  • Efficient range queries                                      │
│  • Simple implementation                                        │
│                                                                  │
│  Disadvantages:                                                  │
│  • Possible hotspots (new users concentrated in last shard)     │
│  • Uneven distribution possible                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Hash-Based Sharding

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hash-Based Sharding                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Determine shard by hash value of the key"                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  shard = hash(user_id) % N                                 │ │
│  │                                                            │ │
│  │  user_id: 12345                                            │ │
│  │  hash(12345) = 67890                                       │ │
│  │  67890 % 3 = 0 → Shard 0                                   │ │
│  │                                                            │ │
│  │  user_id: 12346                                            │ │
│  │  hash(12346) = 12347                                       │ │
│  │  12347 % 3 = 1 → Shard 1                                   │ │
│  │                                                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │  Shard 0    │  │  Shard 1    │  │  Shard 2    │        │ │
│  │  │             │  │             │  │             │        │ │
│  │  │ user: 12345 │  │ user: 12346 │  │ user: 12347 │        │ │
│  │  │ user: 12348 │  │ user: 12349 │  │ user: 12350 │        │ │
│  │  │     ...     │  │     ...     │  │     ...     │        │ │
│  │  │             │  │             │  │             │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Routing Logic:                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  def get_shard(user_id, num_shards=3):                     │ │
│  │      hash_value = hash(str(user_id))                       │ │
│  │      shard_id = hash_value % num_shards                    │ │
│  │      return f"shard_{shard_id}"                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Advantages:                                                     │
│  • Even distribution                                            │
│  • Hotspot prevention                                           │
│                                                                  │
│  Disadvantages:                                                  │
│  • Difficult range queries (requires querying all shards)       │
│  • Large-scale redistribution when adding/removing shards       │
│  • → Solved with consistent hashing                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Directory-Based Sharding

```
┌─────────────────────────────────────────────────────────────────┐
│                  Directory-Based Sharding                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Determine shard using a lookup table (directory)"             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │                   Application                              │ │
│  │                        │                                   │ │
│  │                        │ user_id: 12345                    │ │
│  │                        ▼                                   │ │
│  │              ┌──────────────────┐                          │ │
│  │              │ Directory Service│                          │ │
│  │              │ (Lookup Table)   │                          │ │
│  │              │                  │                          │ │
│  │              │ user_id │ shard  │                          │ │
│  │              │ ────────┼─────── │                          │ │
│  │              │ 12345   │ shard_2│                          │ │
│  │              │ 12346   │ shard_1│                          │ │
│  │              │ 12347   │ shard_3│                          │ │
│  │              │   ...   │  ...   │                          │ │
│  │              └─────────┬────────┘                          │ │
│  │                        │                                   │ │
│  │                        │ shard_2                           │ │
│  │                        ▼                                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │  Shard 1    │  │  Shard 2    │  │  Shard 3    │        │ │
│  │  └─────────────┘  └──────┬──────┘  └─────────────┘        │ │
│  │                          │                                 │ │
│  │                          ▼                                 │ │
│  │                    user: 12345                             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Advantages:                                                     │
│  • Flexible data movement                                       │
│  • Easy to resolve imbalances                                   │
│  • Can move specific users to specific shards                   │
│                                                                  │
│  Disadvantages:                                                  │
│  • Directory service is SPOF                                    │
│  • Additional lookup overhead                                   │
│  • Directory size growth                                        │
│                                                                  │
│  Solutions:                                                      │
│  • Cache the directory (Redis)                                  │
│  • Replicate the directory                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Strategy Comparison

| Strategy | Even Distribution | Range Queries | Rebalancing | Complexity |
|----------|------------------|---------------|-------------|------------|
| Range | Low | Good | Difficult | Low |
| Hash | Good | Difficult | Difficult | Low |
| Directory | Good | Possible | Easy | High |
| Consistent Hash | Good | Difficult | Easy | Medium |

---

## 4. Shard Key Selection

### 4.1 Characteristics of a Good Shard Key

```
┌─────────────────────────────────────────────────────────────────┐
│                Good Shard Key Characteristics                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. High Cardinality                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Good: user_id (millions of unique values)                 │ │
│  │  Bad:  country (dozens)                                    │ │
│  │  Bad:  status (few)                                        │ │
│  │                                                            │ │
│  │  → Many unique values enable even distribution             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Even Distribution                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Good: hashed user_id                                      │ │
│  │  Bad:  created_date (new data concentrated on one shard)   │ │
│  │  Bad:  popular product_id (queries concentrated on certain │ │
│  │        products)                                           │ │
│  │                                                            │ │
│  │  → Data and queries should be evenly distributed           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Query Pattern Alignment                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Most queries: "WHERE user_id = ?"                         │ │
│  │  → Use user_id as shard key!                               │ │
│  │                                                            │ │
│  │  Most queries: "WHERE order_date BETWEEN ... AND ..."      │ │
│  │  → Use order_date as shard key!                            │ │
│  │                                                            │ │
│  │  → Frequent queries should be processed within a single    │ │
│  │    shard                                                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. Immutability                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Good: user_id (doesn't change)                            │ │
│  │  Bad:  email (user can change it)                          │ │
│  │  Bad:  status (changes)                                    │ │
│  │                                                            │ │
│  │  → Changing shard key = Data migration (expensive)         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Shard Key Examples by Domain

```
┌─────────────────────────────────────────────────────────────────┐
│                   Shard Key Examples by Domain                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Social Media (User-centric):                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • users: user_id                                          │ │
│  │  • posts: user_id (author)                                 │ │
│  │  • followers: user_id (the one being followed)             │ │
│  │  • messages: conversation_id                               │ │
│  │                                                            │ │
│  │  → User's data stays in the same shard!                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  E-commerce (Tenant-centric):                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • products: merchant_id                                   │ │
│  │  • orders: user_id or order_id                             │ │
│  │  • reviews: product_id                                     │ │
│  │                                                            │ │
│  │  Note: If orders use user_id,                              │ │
│  │        seller order queries require all shards             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  SaaS (Tenant-centric):                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • All tables: tenant_id                                   │ │
│  │                                                            │ │
│  │  → Complete isolation per tenant                           │ │
│  │  → No cross-tenant queries needed                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Logs/Analytics (Time-series):                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  • logs: timestamp (Range sharding)                        │ │
│  │  • metrics: (device_id, timestamp)                         │ │
│  │                                                            │ │
│  │  → Efficient for range queries                             │ │
│  │  → Easy to delete old shards                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Hotspot Prevention

### 5.1 The Hotspot Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                     The Hotspot Problem                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "A phenomenon where traffic/data concentrates on a specific    │
│   shard"                                                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Normal:                                                   │ │
│  │  ┌───────┐ ┌───────┐ ┌───────┐                            │ │
│  │  │Shard 1│ │Shard 2│ │Shard 3│                            │ │
│  │  │  33%  │ │  33%  │ │  33%  │                            │ │
│  │  └───────┘ └───────┘ └───────┘                            │ │
│  │                                                            │ │
│  │  Hotspot:                                                  │ │
│  │  ┌───────┐ ┌───────┐ ┌───────┐                            │ │
│  │  │Shard 1│ │Shard 2│ │Shard 3│                            │ │
│  │  │  80%  │ │  10%  │ │  10%  │ Overloaded!                │ │
│  │  └───────┘ └───────┘ └───────┘                            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Causes:                                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. Range sharding + Sequential keys                       │ │
│  │     → New data concentrated in last shard                  │ │
│  │                                                            │ │
│  │  2. Popular entities                                       │ │
│  │     → Celebrity accounts, popular products                 │ │
│  │                                                            │ │
│  │  3. Time-based patterns                                    │ │
│  │     → Specific shard concentration at certain times        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Hotspot Prevention Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                Hotspot Prevention Strategies                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Use Hash-Based Sharding                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Use Hash instead of Range for even distribution           │ │
│  │  user_id: 1, 2, 3 → distributed, not in same shard         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Composite Shard Key                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Single key: user_id                                       │ │
│  │  → Popular users become hotspots                           │ │
│  │                                                            │ │
│  │  Composite key: (user_id, post_id)                         │ │
│  │  → Same user's posts are also distributed                  │ │
│  │                                                            │ │
│  │  shard = hash(user_id + "_" + post_id) % N                 │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Salting                                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Original key: celebrity_123                               │ │
│  │  Salting:     celebrity_123_0                              │ │
│  │               celebrity_123_1                              │ │
│  │               celebrity_123_2                              │ │
│  │                                                            │ │
│  │  # Write: Add random salt                                  │ │
│  │  salt = random(0, 10)                                      │ │
│  │  key = f"{user_id}_{salt}"                                 │ │
│  │                                                            │ │
│  │  # Read: Query all salts and combine                       │ │
│  │  for salt in range(10):                                    │ │
│  │      results += query(f"{user_id}_{salt}")                 │ │
│  │                                                            │ │
│  │  → Write distribution, increased read complexity           │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. Hot Entity Special Handling                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  • Cache popular entities (Redis)                          │ │
│  │  • Dedicated shard/server                                  │ │
│  │  • Async processing (counters, etc.)                       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Rebalancing

### 6.1 When Rebalancing is Needed

```
┌─────────────────────────────────────────────────────────────────┐
│                  When Rebalancing is Needed                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Adding Shards (Expansion)                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Before: [Shard 1] [Shard 2] [Shard 3]                     │ │
│  │  After:  [Shard 1] [Shard 2] [Shard 3] [Shard 4]           │ │
│  │                                                            │ │
│  │  Move some data to new shard                               │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Removing Shards (Contraction)                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Before: [Shard 1] [Shard 2] [Shard 3]                     │ │
│  │  After:  [Shard 1] [Shard 2]                               │ │
│  │                                                            │ │
│  │  Move Shard 3 data to other shards                         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Resolving Imbalance                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Before: [Shard 1: 80%] [Shard 2: 10%] [Shard 3: 10%]      │ │
│  │  After:  [Shard 1: 33%] [Shard 2: 33%] [Shard 3: 33%]      │ │
│  │                                                            │ │
│  │  Move data from Shard 1 to other shards                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Rebalancing Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rebalancing Strategies                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Consistent Hashing                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Minimal data movement when adding/removing shards         │ │
│  │  Learned in previous lesson                                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. Dual Write                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  Phase 1: Add new shard, write to both                     │ │
│  │  ┌──────┐     ┌──────┐     ┌──────┐                        │ │
│  │  │Old S1│     │Old S2│     │New S3│                        │ │
│  │  │ R/W  │     │ R/W  │     │  W   │                        │ │
│  │  └──────┘     └──────┘     └──────┘                        │ │
│  │                                                            │ │
│  │  Phase 2: Migrate existing data (background)               │ │
│  │                                                            │ │
│  │  Phase 3: Read from new shard as well                      │ │
│  │  ┌──────┐     ┌──────┐     ┌──────┐                        │ │
│  │  │Shard1│     │Shard2│     │Shard3│                        │ │
│  │  │ R/W  │     │ R/W  │     │ R/W  │                        │ │
│  │  └──────┘     └──────┘     └──────┘                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. Version-Based Routing                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  shard_config_v1:                                          │ │
│  │    range 1-1M → shard_1                                    │ │
│  │    range 1M-2M → shard_2                                   │ │
│  │                                                            │ │
│  │  shard_config_v2: (after migration)                        │ │
│  │    range 1-700K → shard_1                                  │ │
│  │    range 700K-1.4M → shard_2                               │ │
│  │    range 1.4M-2M → shard_3                                 │ │
│  │                                                            │ │
│  │  # New writes use v2, existing reads use v1                │ │
│  │  # Switch to v2 after migration complete                   │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Problem 1: Choosing a Sharding Strategy

Choose the appropriate sharding strategy for the following services.

a) Chat app (messages per conversation room)
b) Log analysis system
c) Global user service
d) SaaS multi-tenant

### Problem 2: Shard Key Selection

Choose the shard key for the following tables in an e-commerce service.

- users (id, email, name)
- orders (id, user_id, status, created_at)
- order_items (id, order_id, product_id, quantity)
- products (id, merchant_id, name, price)

Requirements:
- Frequent queries for user orders
- Need to query products by seller
- Orders and order items queried together

### Problem 3: Hotspot Resolution

Comments are concentrating on popular posts, overloading that shard. Propose solutions.

### Problem 4: Rebalancing Plan

You want to expand from 3 shards to 5 shards. Plan a zero-downtime migration.

---

## Answers

### Problem 1 Answer

```
a) Chat app: Hash(conversation_id)
   - Messages in the same conversation room go to the same shard
   - Efficient as queries are per conversation room

b) Log analysis: Range(timestamp)
   - Time range queries are efficient
   - Easy to delete old log shards

c) Global users: Hash(user_id)
   - Even distribution
   - Efficient per-user data queries

d) SaaS multi-tenant: tenant_id
   - Complete isolation per tenant
   - No cross-tenant queries needed
```

### Problem 2 Answer

```
users: Hash(user_id)
  - Use as primary key
  - Even distribution

orders: Hash(user_id)
  - user_id is the main query condition
  - Efficient user order queries

order_items: Hash(order_id)
  - Or Hash(user_id) (same shard as orders)
  - Using order_id may put it on different shard than orders
  - → Recommend user_id (same shard as orders)

products: Hash(merchant_id)
  - Query products by seller
  - Seller unit isolation

Cross-shard considerations:
- Order product info → Use cache
- Seller order queries → Separate index table
```

### Problem 3 Answer

```
Solutions:

1. Hot post caching
   - Cache comments in Redis
   - Batch save to DB periodically

2. Apply salting
   post_123_0, post_123_1, ...
   - Random salt on write
   - Query all salts and combine on read

3. Separate comment counters
   - Manage counters in Redis only
   - Async DB updates

4. Change shard key
   - Use (post_id, comment_id) composite key
   - Same post's comments are also distributed

5. Dedicated handling
   - Detect popular posts
   - Add dedicated cache layer
```

### Problem 4 Answer

```
Zero-downtime migration plan:

Phase 1: Prepare new shards
   - Prepare Shard 4, 5 servers
   - Create schema

Phase 2: Start dual write
   - Update router
   - New writes → Old shards + New shards (based on consistent hashing)

Phase 3: Migrate existing data
   - Background copy
   - Determine targets by consistent hashing (about 40% moves)

Phase 4: Verification
   - Compare data counts
   - Verify sample data

Phase 5: Switch reads
   - Activate new routing rules
   - Gradual switch (10% → 50% → 100%)

Phase 6: Stop dual write
   - Remove old routing

Phase 7: Cleanup
   - Delete migrated data from old shards
```

---

## Hands-On Exercises

### Exercise 1: Sharding Strategy Comparison

Use `examples/System_Design/08_sharding_sim.py` to explore sharding behavior.

**Tasks:**
1. Run all demos and compare hash-based, range-based, and directory-based sharding
2. Create a realistic dataset: 10,000 user records with IDs and creation timestamps
3. Shard by user_id (hash) and by creation_date (range). Which gives better balance?
4. Implement a cross-shard query: "find all users created in January" — compare query complexity for hash vs. range sharding

### Exercise 2: Shard Rebalancing

Implement a shard rebalancing algorithm for adding a new node.

**Tasks:**
1. Start with 4 hash-based shards and 10,000 keys
2. Add a 5th shard. Implement a migration plan that moves the minimum number of keys
3. Use consistent hashing (from `07_consistent_hashing.py`) to determine the new key-to-shard mapping
4. Simulate the migration with a "live" system: reads continue working during migration, writes go to both old and new shard until migration completes
5. Report: keys moved, migration duration (simulated), and data inconsistency window

### Exercise 3: Hot Shard Detection and Mitigation

Build a monitoring system that detects and mitigates hot shards.

**Tasks:**
1. Create a load tracker that counts requests per shard per time window
2. Simulate a hot-spot: one shard receives 5× the traffic of others (e.g., a viral user's data)
3. Implement detection: alert when any shard exceeds 2× the average load
4. Implement mitigation: split the hot shard into 2 sub-shards using a secondary hash
5. Verify that the load rebalances after splitting

---

## 8. References

### Books
- Designing Data-Intensive Applications - Ch. 6

### Databases
- [Vitess](https://vitess.io/) - MySQL Sharding
- [Citus](https://www.citusdata.com/) - PostgreSQL Sharding
- [MongoDB Sharding](https://docs.mongodb.com/manual/sharding/)

### Case Studies
- [Instagram Sharding](https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c)
- [Pinterest Sharding](https://medium.com/pinterest-engineering/sharding-pinterest-how-we-scaled-our-mysql-fleet-3f341e96ca6f)

---

**Previous**: [Distributed Cache Systems](./07_Distributed_Cache_Systems.md) | **Next**: [Database Replication](./09_Database_Replication.md)

---

**Document Information**
- Last Updated: 2024
- Difficulty: ⭐⭐⭐
- Estimated Learning Time: 2-3 hours
