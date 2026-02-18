# NewSQL and Modern Trends

**Previous**: [14. Distributed Databases](./14_Distributed_Databases.md) | **Next**: [16. Database Design Case Study](./16_Database_Design_Case_Study.md)

---

The database landscape has evolved dramatically since the NoSQL revolution. A new generation of systems aims to combine the best of both worlds: the horizontal scalability and availability of NoSQL with the ACID guarantees and SQL interface of relational databases. Beyond NewSQL, entirely new categories of databases have emerged to serve specialized workloads -- vector similarity search for AI, time-series storage for IoT, and graph analytics at scale. This lesson surveys the cutting edge of database technology and the architectural innovations that make these systems possible.

**Difficulty**: ⭐⭐⭐

**Learning Objectives**:
- Explain the NewSQL design philosophy and how it reconciles ACID with horizontal scalability
- Describe Google Spanner's TrueTime mechanism and external consistency
- Compare CockroachDB, TiDB, and Spanner architectures
- Understand vector databases and similarity search algorithms (HNSW, IVF)
- Explain time-series database optimizations
- Evaluate graph analytics platforms for large-scale analysis
- Assess Database-as-a-Service offerings
- Understand the data lakehouse paradigm and its relationship to databases

---

## Table of Contents

1. [NewSQL: Motivation and Design Philosophy](#1-newsql-motivation-and-design-philosophy)
2. [Google Spanner](#2-google-spanner)
3. [CockroachDB](#3-cockroachdb)
4. [TiDB](#4-tidb)
5. [Vector Databases](#5-vector-databases)
6. [Time-Series Databases](#6-time-series-databases)
7. [Graph Analytics at Scale](#7-graph-analytics-at-scale)
8. [Database-as-a-Service](#8-database-as-a-service)
9. [Data Lakehouse](#9-data-lakehouse)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. NewSQL: Motivation and Design Philosophy

### 1.1 The Gap Between SQL and NoSQL

By the early 2010s, the database world had split into two camps:

```
Traditional SQL                              NoSQL
┌──────────────────────────┐    ┌──────────────────────────┐
│ ✓ ACID transactions      │    │ ✓ Horizontal scalability │
│ ✓ SQL interface          │    │ ✓ High availability      │
│ ✓ Rich query capability  │    │ ✓ Flexible schemas       │
│ ✓ Strong consistency     │    │ ✓ Low latency            │
│ ✗ Horizontal scaling     │    │ ✗ No ACID (usually)      │
│ ✗ Geographic distribution│    │ ✗ No SQL (or limited)    │
│ ✗ Auto-failover          │    │ ✗ Weak consistency       │
└──────────────────────────┘    └──────────────────────────┘

                    GAP: Can we have both?

NewSQL
┌──────────────────────────┐
│ ✓ ACID transactions      │
│ ✓ SQL interface          │
│ ✓ Horizontal scalability │
│ ✓ High availability      │
│ ✓ Strong consistency     │
│ ✓ Geographic distribution│
│ ✓ Auto-failover          │
└──────────────────────────┘
```

### 1.2 NewSQL Definition

The term "NewSQL" was coined by Matthew Aslett of the 451 Group in 2011 to describe a new class of relational database management systems that:

1. **Provide ACID guarantees** for read-write transactions
2. **Use SQL** as the primary interface
3. **Scale horizontally** across commodity hardware using shared-nothing architecture
4. **Achieve throughput** comparable to or exceeding NoSQL systems for OLTP workloads

### 1.3 Architectural Innovations

NewSQL systems typically employ three key innovations:

**1. Sharding with distributed transactions**: Data is automatically partitioned (sharded) across nodes, but the system provides transparent distributed transactions via consensus protocols (Paxos/Raft), making the sharding invisible to the application.

**2. Multi-version concurrency control (MVCC)**: Instead of locking, NewSQL systems use MVCC with globally ordered timestamps, allowing reads to proceed without blocking writes.

**3. Consensus-based replication**: Each partition (shard) is replicated across multiple nodes using Raft or Paxos, providing both high availability and strong consistency.

```
NewSQL Architecture (Generic):

Client ──▶ SQL Layer (Parse, Plan, Optimize)
                │
                ▼
           Transaction Layer (Distributed MVCC, 2PC/Raft)
                │
                ▼
           Storage Layer (Sharded, Replicated)
           ┌──────┐  ┌──────┐  ┌──────┐
           │Shard 1│  │Shard 2│  │Shard 3│
           │R1,R2,R3│  │R1,R2,R3│  │R1,R2,R3│  ← Each shard has
           └──────┘  └──────┘  └──────┘     3 replicas (Raft group)
```

### 1.4 NewSQL vs Sharded PostgreSQL

A common question is: "Why not just shard PostgreSQL?" The answer reveals what makes NewSQL special:

| Feature | Sharded PostgreSQL (Citus) | NewSQL (CockroachDB) |
|---------|---------------------------|----------------------|
| **Cross-shard transactions** | Limited (2PC across shards) | Full ACID (Raft + MVCC) |
| **Automatic resharding** | Manual or semi-automatic | Automatic range splitting |
| **Schema changes** | Rolling DDL required | Online schema changes |
| **Replication** | Async/sync (per shard) | Raft (per range) |
| **Failover** | Manual or scripted | Automatic (Raft leader election) |
| **Global distribution** | Possible but complex | Built-in (geo-partitioning) |
| **Compatibility** | Full PostgreSQL | PostgreSQL wire protocol (subset) |

---

## 2. Google Spanner

### 2.1 Overview

Google Spanner, introduced in a 2012 paper, is the first system to distribute data at global scale while providing **externally consistent** (linearizable) transactions. It is the original NewSQL system that inspired all others.

```
Google Spanner Architecture:

┌─────────────────────────────────────────────────────────────┐
│                        Universe                              │
│                                                             │
│  Zone (US-East)        Zone (EU-West)        Zone (Asia)   │
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────┐ │
│  │ Spanserver 1  │    │ Spanserver 4  │    │ Spanserver 7│ │
│  │ Spanserver 2  │    │ Spanserver 5  │    │ Spanserver 8│ │
│  │ Spanserver 3  │    │ Spanserver 6  │    │ Spanserver 9│ │
│  │               │    │               │    │             │ │
│  │ ┌─TrueTime─┐  │    │ ┌─TrueTime─┐  │    │ ┌─TrueTime┐│ │
│  │ │GPS+Atomic│  │    │ │GPS+Atomic│  │    │ │GPS+Atom.││ │
│  │ │  Clocks  │  │    │ │  Clocks  │  │    │ │ Clocks  ││ │
│  │ └──────────┘  │    │ └──────────┘  │    │ └─────────┘│ │
│  └───────────────┘    └───────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 TrueTime

The key innovation in Spanner is **TrueTime**, a globally synchronized clock API that returns a time interval rather than a single timestamp.

```
Traditional clock:    now() → T           (single point, unknown error)
TrueTime:            TT.now() → [T-ε, T+ε]  (interval with bounded error)

Where ε is the clock uncertainty, typically 1-7 milliseconds.

TrueTime guarantees:
  For any invocation of TT.now() returning [earliest, latest]:
    earliest ≤ actual_time ≤ latest

TrueTime Implementation:
  - GPS receivers: provide time accurate to nanoseconds
  - Atomic clocks: drift slowly, used when GPS signal is lost
  - Multiple time sources per datacenter for redundancy
  - Marzullo's algorithm to compute tight bounds
```

### 2.3 External Consistency via Commit Wait

Spanner uses TrueTime to achieve **external consistency** (also called strict serializability or linearizability): if transaction T1 commits before transaction T2 starts (in real time), then T1's commit timestamp is less than T2's commit timestamp.

```
Commit Wait Protocol:

1. Transaction T acquires all locks
2. T picks commit timestamp s = TT.now().latest
3. COMMIT WAIT: T waits until TT.after(s) is true
   i.e., waits until the clock uncertainty has passed
4. T commits with timestamp s and releases locks

Why this works:
  - T picked s = TT.now().latest at some real time t_pick
  - After commit wait, real time t_commit satisfies t_commit > s
  - Any future transaction T' starting after t_commit will get
    s' = TT.now().latest > t_commit > s
  - Therefore s < s' → T serializes before T' (external consistency)

Timeline:
  ├──────┤ε├──────┤
  |      |s|      |
  |      |        |
  |   TT.now()    |
  |   [T-ε, T+ε]  |
  |               |
  |    Commit     |
  |    Wait       |
  |               |
  |               ├── TT.after(s) = true → COMMIT
```

**Cost of commit wait**: Average ~7ms latency added to every write transaction (the average clock uncertainty).

### 2.4 Spanner SQL and Features

Spanner supports a dialect of SQL with extensions for distributed semantics:

```sql
-- Create a table with interleaving (co-locate child rows with parent)
CREATE TABLE Customers (
  CustomerId INT64 NOT NULL,
  Name STRING(100),
  Email STRING(256)
) PRIMARY KEY (CustomerId);

CREATE TABLE Orders (
  CustomerId INT64 NOT NULL,
  OrderId INT64 NOT NULL,
  Total NUMERIC,
  CreatedAt TIMESTAMP
) PRIMARY KEY (CustomerId, OrderId),
  INTERLEAVE IN PARENT Customers ON DELETE CASCADE;

-- Interleaving physically co-locates orders with their customer
-- This eliminates network round-trips for customer-order joins!

-- Read-write transaction (serializable)
BEGIN TRANSACTION;
  UPDATE Accounts SET Balance = Balance - 100 WHERE AccountId = 1;
  UPDATE Accounts SET Balance = Balance + 100 WHERE AccountId = 2;
COMMIT;

-- Read-only transaction (no locks, no commit wait)
-- Uses a snapshot timestamp for consistent reads
SET TRANSACTION READ ONLY;
SELECT * FROM Orders WHERE CustomerId = 42;

-- Stale reads (bounded staleness, for lower latency)
SELECT * FROM Orders
  WHERE CustomerId = 42
  WITH (MAX_STALENESS = 10s);
```

### 2.5 Spanner Summary

| Feature | Detail |
|---------|--------|
| **Consistency** | External consistency (linearizability) |
| **Clock** | TrueTime (GPS + atomic clocks) |
| **Replication** | Multi-Paxos per split |
| **Partitioning** | Range-based, automatic split/merge |
| **Transactions** | 2PC across Paxos groups |
| **SQL** | ANSI SQL with extensions |
| **Availability** | 99.999% SLA (5 nines) |
| **Latency** | ~7ms write (commit wait), low ms reads |

---

## 3. CockroachDB

### 3.1 Overview

CockroachDB (CRDB) is an open-source NewSQL database inspired by Spanner but designed to run without specialized hardware (no GPS/atomic clocks). It is named after the cockroach for its resilience -- it is designed to survive failures at every level.

### 3.2 Architecture

```
CockroachDB Architecture:

┌──────────────────────────────────────────────────────────┐
│                     SQL Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Gateway  │  │ Gateway  │  │ Gateway  │  (any node)   │
│  │ Node     │  │ Node     │  │ Node     │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │              │              │                    │
│       └──────────────┼──────────────┘                    │
│                      │                                   │
│  ┌───────────────────┴────────────────────────────────┐  │
│  │              Distribution Layer                     │  │
│  │  Key space divided into Ranges (~512MB each)       │  │
│  │  Each Range is a Raft group (3+ replicas)          │  │
│  │  Range 1: [a, f)   Range 2: [f, m)   Range 3: ... │  │
│  └────────────────────────────────────────────────────┘  │
│                      │                                   │
│  ┌───────────────────┴────────────────────────────────┐  │
│  │              Storage Layer (Pebble)                 │  │
│  │  LSM-tree based key-value store                    │  │
│  │  MVCC timestamps on every key                      │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Multi-Active Availability

Unlike traditional primary-standby setups, CockroachDB has no single leader for the entire database. Each Range has its own Raft leader, and leaders are distributed across all nodes:

```
Node 1                  Node 2                  Node 3
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│ Range A: LEADER│     │ Range A: Follower│     │ Range A: Follower│
│ Range B: Follower│     │ Range B: LEADER│     │ Range B: Follower│
│ Range C: Follower│     │ Range C: Follower│     │ Range C: LEADER│
└────────────────┘     └────────────────┘     └────────────────┘

Every node serves both reads and writes (for ranges it leads).
No single point of failure!
```

### 3.4 Serializable Isolation Without TrueTime

Since CRDB runs on commodity hardware without TrueTime, it uses a different mechanism for transaction ordering:

**Hybrid Logical Clocks (HLC)**: Combine a physical clock component with a logical counter.

```
HLC = (physical_time, logical_counter, node_id)

Rules:
1. Local event: HLC.physical = max(HLC.physical, wall_clock)
                HLC.logical += 1
2. Send message: attach current HLC
3. Receive message with HLC_msg:
   HLC.physical = max(HLC.physical, HLC_msg.physical, wall_clock)
   if HLC.physical == HLC_msg.physical:
     HLC.logical = max(HLC.logical, HLC_msg.logical) + 1
   else:
     HLC.logical = 0
```

**Clock skew handling**: CRDB enforces a maximum clock offset (default 500ms). If clocks drift beyond this, nodes are taken out of the cluster. For transactions that might be affected by clock skew, CRDB uses **uncertainty intervals** and **read refreshes**:

```
Transaction T1 reads key K at timestamp t:
  - T1 observes value V written at timestamp t_w
  - If t_w is within T1's uncertainty interval [t, t + max_offset]:
    T1 cannot determine if V was written before or after T1 started
    → T1 pushes its timestamp forward past t_w and retries the read

This is called a "read restart" and ensures serializability.
```

### 3.5 Key Features

```sql
-- Geo-partitioning: pin data to specific regions
ALTER TABLE users PARTITION BY LIST (country) (
  PARTITION us VALUES IN ('US'),
  PARTITION eu VALUES IN ('DE', 'FR', 'UK'),
  PARTITION asia VALUES IN ('JP', 'KR', 'SG')
);

ALTER PARTITION us OF TABLE users
  CONFIGURE ZONE USING constraints='[+region=us-east1]';
ALTER PARTITION eu OF TABLE users
  CONFIGURE ZONE USING constraints='[+region=europe-west1]';

-- Follows leader (low-latency reads from nearest replica)
ALTER TABLE users CONFIGURE ZONE USING
  lease_preferences='[[+region=us-east1],[+region=europe-west1]]';

-- Online schema changes (no downtime)
ALTER TABLE orders ADD COLUMN discount DECIMAL DEFAULT 0;
-- Runs as a background job, no table locks!

-- Change data capture (CDC)
CREATE CHANGEFEED FOR TABLE orders INTO 'kafka://broker:9092';
```

### 3.6 CockroachDB Summary

| Feature | Detail |
|---------|--------|
| **Consistency** | Serializable isolation |
| **Clock** | HLC (Hybrid Logical Clock) |
| **Replication** | Raft per Range |
| **Partitioning** | Automatic range splitting (~512MB) |
| **SQL** | PostgreSQL wire protocol compatible |
| **Geo-distribution** | Geo-partitioning, follower reads |
| **Schema changes** | Online, non-blocking |
| **License** | BSL (free for most uses) |

---

## 4. TiDB

### 4.1 Overview

TiDB (Ti stands for Titanium) is an open-source NewSQL database created by PingCAP. Its distinguishing feature is **HTAP (Hybrid Transactional/Analytical Processing)**: it can handle both OLTP and OLAP workloads in a single system.

### 4.2 Architecture

```
TiDB Architecture:

                    ┌─────────┐
                    │  Client │
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │ TiDB   │ │ TiDB   │ │ TiDB   │   SQL Layer
         │ Server │ │ Server │ │ Server │   (stateless)
         └────┬───┘ └────┬───┘ └────┬───┘
              │          │          │
              └──────────┼──────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │ TiKV   │ │ TiKV   │ │ TiKV   │   Row Store (OLTP)
         │(Raft)  │ │(Raft)  │ │(Raft)  │   (transactional)
         └────────┘ └────────┘ └────────┘
              │          │          │
              └──────────┼──────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │TiFlash │ │TiFlash │ │TiFlash │   Column Store (OLAP)
         │(Raft   │ │(Raft   │ │(Raft   │   (analytical)
         │Learner)│ │Learner)│ │Learner)│
         └────────┘ └────────┘ └────────┘

Placement Driver (PD): Cluster metadata, timestamp oracle, scheduling
```

### 4.3 HTAP: Bridging OLTP and OLAP

Traditional architectures separate OLTP (write-heavy, row-oriented) and OLAP (read-heavy, column-oriented) into different systems, with ETL pipelines between them. TiDB unifies both:

```
Traditional Architecture:
  OLTP (MySQL) ──ETL──▶ OLAP (Data Warehouse)
                        (hours of delay)

TiDB HTAP Architecture:
  ┌──────────────────────────────────────────┐
  │              TiDB                         │
  │                                           │
  │  OLTP query ──▶ TiKV (row store)         │
  │  OLAP query ──▶ TiFlash (column store)   │
  │                                           │
  │  Raft replication: TiKV ──▶ TiFlash      │
  │  (real-time, asynchronous via Raft Learner)│
  └──────────────────────────────────────────┘
```

**TiKV** (row store): Stores data in key-value pairs ordered by primary key. Optimized for point reads and writes (OLTP). Uses Raft for replication and RocksDB for storage.

**TiFlash** (column store): Stores the same data in columnar format. Optimized for scans, aggregations, and analytical queries (OLAP). Receives data via Raft Learner protocol (asynchronous, does not affect TiKV write latency).

**Query routing**: The TiDB SQL optimizer automatically decides whether to use TiKV or TiFlash for each part of a query:

```sql
-- OLTP query → routed to TiKV
SELECT * FROM orders WHERE order_id = 12345;

-- OLAP query → routed to TiFlash
SELECT product_id, SUM(quantity), AVG(price)
FROM order_items
GROUP BY product_id
ORDER BY SUM(quantity) DESC
LIMIT 100;

-- Hybrid query → TiKV for point lookup, TiFlash for aggregation
SELECT c.name, stats.total_spent
FROM customers c
JOIN (
  SELECT customer_id, SUM(total) as total_spent
  FROM orders
  GROUP BY customer_id
  HAVING SUM(total) > 10000
) stats ON c.id = stats.customer_id
WHERE c.id = 42;
```

### 4.4 TiDB Summary

| Feature | Detail |
|---------|--------|
| **Consistency** | Snapshot isolation (default), serializable (optional) |
| **Clock** | Timestamp Oracle (centralized, via PD) |
| **Replication** | Raft per Region |
| **HTAP** | TiKV (row) + TiFlash (column) |
| **SQL** | MySQL wire protocol compatible |
| **Storage engine** | RocksDB (TiKV), ClickHouse-derived (TiFlash) |
| **License** | Apache 2.0 (open source) |

---

## 5. Vector Databases

### 5.1 Motivation

The rise of deep learning has created a new data type: **embeddings** -- high-dimensional vectors that represent the semantic meaning of text, images, audio, or any object.

```
Traditional DB query:          Vector DB query:
"Find user with id = 42"      "Find items similar to this image"
Exact match on a key           Nearest neighbors in vector space

                    ┌─ Vector Space ──────────────────┐
                    │                                   │
                    │    ●(cat image)                   │
                    │      ● (kitten image)             │
                    │        ● (query: cat photo)       │
                    │                                   │
                    │                     ●(car image)  │
                    │                   ● (truck image) │
                    │                                   │
                    │ ●(sunset photo)                   │
                    │   ● (beach photo)                 │
                    │                                   │
                    └───────────────────────────────────┘
```

### 5.2 Embeddings

An embedding is a learned representation that maps an object to a fixed-dimensional vector such that semantically similar objects have nearby vectors.

```
Text embeddings (e.g., OpenAI text-embedding-3-small, 1536 dimensions):
  "The cat sat on the mat"   → [0.023, -0.041, 0.108, ..., 0.055]  (1536 dims)
  "A kitten rested on a rug" → [0.021, -0.039, 0.112, ..., 0.051]  (similar!)
  "Stock prices fell today"  → [-0.087, 0.032, -0.005, ..., 0.019] (different)

Image embeddings (e.g., CLIP, 512 dimensions):
  photo_of_cat.jpg → [0.15, -0.23, ..., 0.08]   (512 dims)
  photo_of_dog.jpg → [0.12, -0.19, ..., 0.11]   (nearby in vector space)
```

### 5.3 Similarity Metrics

| Metric | Formula | Range | Use Case |
|--------|---------|-------|----------|
| **Cosine similarity** | `cos(A,B) = (A . B) / (|A| * |B|)` | [-1, 1] | Text similarity (direction matters) |
| **Euclidean distance (L2)** | `d = sqrt(sum((a_i - b_i)^2))` | [0, inf) | Image similarity, spatial data |
| **Dot product** | `A . B = sum(a_i * b_i)` | (-inf, inf) | Recommendation (magnitude matters) |
| **Manhattan distance (L1)** | `d = sum(|a_i - b_i|)` | [0, inf) | Sparse vectors |

### 5.4 Approximate Nearest Neighbor (ANN) Search

Exact nearest neighbor search in high dimensions is computationally prohibitive (curse of dimensionality). Vector databases use **approximate** algorithms that trade a small accuracy loss for massive speed gains.

**Inverted File Index (IVF)**:

```
IVF Algorithm:

1. TRAINING: Cluster vectors into K centroids (K-means)
   ┌────────────────────────────────────┐
   │  ●  ● C1        ●  ●  ●           │
   │    ●       ●        C2  ●         │
   │                                    │
   │       ● C3                         │
   │     ●    ●                         │
   │    ●                  ● ●          │
   │                      ●  C4 ●      │
   └────────────────────────────────────┘
   C1, C2, C3, C4 are cluster centroids

2. INDEXING: Assign each vector to its nearest centroid
   Inverted lists:
   C1 → [v1, v5, v12, v33, ...]
   C2 → [v2, v7, v15, v41, ...]
   C3 → [v3, v9, v22, v37, ...]
   C4 → [v4, v11, v19, v45, ...]

3. SEARCH: For query q, find nprobe nearest centroids,
   then search only those clusters' vectors
   nprobe=1: search 1 cluster (fast, less accurate)
   nprobe=K: search all clusters (slow, exact)
```

**HNSW (Hierarchical Navigable Small World)**:

```
HNSW Algorithm:

Build a multi-layer graph where:
- Layer 0 (bottom): contains ALL vectors, densely connected
- Layer 1: contains a SUBSET of vectors, sparser connections
- Layer 2: even fewer vectors, even sparser
- ...
- Top layer: very few vectors, long-range connections

Search traversal:

Layer 3:  A ─────────────────── B          (few nodes, long jumps)
          │                     │
Layer 2:  A ──── C ──── D ──── B          (more nodes)
          │      │      │      │
Layer 1:  A ─ E ─ C ─ F ─ D ─ G ─ B      (even more nodes)
          │   │   │   │   │   │   │
Layer 0:  A E H C I F J D K G L B M      (all nodes, short connections)
                              ↑
                          query point

Search: Start at top layer, greedily move to nearest neighbor,
        descend to next layer, repeat.
        At Layer 0, explore local neighborhood for final result.

Time complexity: O(log N) per query (vs O(N) for brute force)
```

### 5.5 Vector Database Systems

**Pinecone** (fully managed):

```python
import pinecone

# Initialize
pinecone.init(api_key="...", environment="us-east-1-aws")
index = pinecone.Index("product-search")

# Upsert vectors
index.upsert(vectors=[
    ("prod_1", [0.1, 0.2, ..., 0.5], {"category": "electronics", "price": 99.99}),
    ("prod_2", [0.3, 0.1, ..., 0.8], {"category": "clothing", "price": 49.99}),
])

# Query: find 5 most similar products
results = index.query(
    vector=[0.15, 0.22, ..., 0.48],  # query embedding
    top_k=5,
    filter={"category": {"$eq": "electronics"}},  # metadata filter
    include_metadata=True
)
```

**Milvus** (open source):

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
]
schema = CollectionSchema(fields)
collection = Collection("documents", schema)

# Create HNSW index
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index("embedding", index_params)

# Search
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
    output_fields=["text"]
)
```

**pgvector** (PostgreSQL extension):

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(1536)  -- 1536-dimensional vector
);

-- Create HNSW index
CREATE INDEX ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

-- Insert
INSERT INTO documents (content, embedding)
VALUES ('The cat sat on the mat', '[0.023, -0.041, ...]');

-- Similarity search (cosine distance)
SELECT content, 1 - (embedding <=> '[0.025, -0.038, ...]') AS similarity
FROM documents
ORDER BY embedding <=> '[0.025, -0.038, ...]'
LIMIT 5;

-- Hybrid search: vector similarity + metadata filter
SELECT content, 1 - (embedding <=> $1) AS similarity
FROM documents
WHERE category = 'science'
ORDER BY embedding <=> $1
LIMIT 10;
```

### 5.6 Use Cases for Vector Databases

| Use Case | Description |
|----------|-------------|
| **Semantic search** | Find documents by meaning, not just keywords |
| **RAG (Retrieval-Augmented Generation)** | Store knowledge base embeddings; retrieve context for LLM prompts |
| **Recommendation systems** | Find items similar to user preferences (collaborative filtering via embeddings) |
| **Image search** | Find visually similar images |
| **Anomaly detection** | Identify vectors far from normal clusters |
| **Deduplication** | Find near-duplicate documents or images |
| **Multimodal search** | Search across text, images, and audio using shared embedding spaces (CLIP) |

### 5.7 Choosing a Vector Database

| System | Type | Hosting | Strengths | Limitations |
|--------|------|---------|-----------|-------------|
| **Pinecone** | Managed | Cloud only | Simple API, auto-scaling | Vendor lock-in, cost |
| **Milvus** | Open source | Self-hosted / Zilliz Cloud | Feature-rich, GPU support | Operational complexity |
| **Weaviate** | Open source | Self-hosted / Cloud | GraphQL API, modules | Newer, smaller ecosystem |
| **Qdrant** | Open source | Self-hosted / Cloud | Rust-based (fast), filtering | Smaller community |
| **pgvector** | Extension | Any PostgreSQL | No new infra, full SQL | Limited scale, no GPU |
| **Chroma** | Open source | Embedded / Server | Simple, Python-native | Limited production features |

---

## 6. Time-Series Databases

### 6.1 What is Time-Series Data?

Time-series data consists of measurements or events indexed by time. It has unique characteristics that general-purpose databases handle poorly.

```
Time-series data examples:

IoT sensor readings:
  timestamp            | device_id | temperature | humidity
  2024-11-15T10:00:00  | sensor_42 | 23.5        | 65.2
  2024-11-15T10:00:01  | sensor_42 | 23.6        | 65.1
  2024-11-15T10:00:02  | sensor_42 | 23.5        | 65.3
  ...

Application metrics:
  timestamp            | service   | metric     | value
  2024-11-15T10:00:00  | api-gw    | latency_ms | 42
  2024-11-15T10:00:00  | api-gw    | req_count  | 1547
  2024-11-15T10:00:00  | api-gw    | error_rate | 0.02
  ...
```

### 6.2 Characteristics of Time-Series Data

| Characteristic | Implication |
|---------------|-------------|
| **Write-heavy** | Continuous ingestion, rarely updated after write |
| **Time-ordered** | Data naturally ordered by timestamp; most queries involve time ranges |
| **Append-only** | New data is always newer; old data is rarely modified |
| **High cardinality** | Millions of unique series (device_id x metric combinations) |
| **Downsampling** | Old data can be aggregated (1-second → 1-minute → 1-hour) |
| **TTL (expiry)** | Old data has diminishing value; auto-delete after retention period |
| **Compression** | Temporal locality enables high compression ratios (10-20x) |

### 6.3 TSDB Optimizations

**Time-based partitioning**: Data is automatically partitioned by time, enabling efficient time-range queries and data lifecycle management.

```
┌─────────────────────────────────────────────────────┐
│  Time-based partitioning                             │
│                                                     │
│  Chunk 1        Chunk 2        Chunk 3        ...   │
│  [Nov 1-7]     [Nov 8-14]    [Nov 15-21]          │
│  ┌──────┐      ┌──────┐      ┌──────┐              │
│  │ Data │      │ Data │      │ Data │              │
│  │ Index│      │ Index│      │ Index│              │
│  └──────┘      └──────┘      └──────┘              │
│                                                     │
│  Query: WHERE time > 'Nov 10' AND time < 'Nov 20'  │
│  → Only scans Chunk 2 and Chunk 3!                  │
│                                                     │
│  Retention: DROP Chunk 1 (instant, no row-by-row)   │
└─────────────────────────────────────────────────────┘
```

**Delta-of-delta compression**: Consecutive timestamps are similar (e.g., every 10 seconds). Store the delta between deltas.

```
Raw timestamps:     1700000000, 1700000010, 1700000020, 1700000030
Deltas:                        10,          10,          10
Delta-of-deltas:                            0,           0

Compression: store base + first delta + delta-of-deltas
  1700000000, 10, 0, 0  (4 values instead of 4 full timestamps)

For values: Gorilla compression (XOR-based, from Facebook's Gorilla paper)
  If consecutive values are similar, XOR is mostly zeros → compress well
```

**Downsampling**: Automatically aggregate old data to reduce storage.

```
Raw data (1-second resolution):     → 86,400 points/day/series
After 7 days: downsample to 1-min  → 1,440 points/day/series
After 30 days: downsample to 1-hr  → 24 points/day/series
After 1 year: downsample to 1-day  → 1 point/day/series

Storage reduction: 86,400x over 1 year!
```

### 6.4 TimescaleDB

TimescaleDB is a time-series extension for PostgreSQL, preserving full SQL capability.

```sql
-- Create a hypertable (auto-partitioned by time)
CREATE TABLE metrics (
  time TIMESTAMPTZ NOT NULL,
  device_id TEXT NOT NULL,
  temperature DOUBLE PRECISION,
  humidity DOUBLE PRECISION
);

SELECT create_hypertable('metrics', 'time');

-- Automatic partitioning by time (chunks)
-- Each chunk is a separate PostgreSQL table

-- Continuous aggregate (materialized, auto-refreshing)
CREATE MATERIALIZED VIEW hourly_stats
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', time) AS bucket,
  device_id,
  AVG(temperature) AS avg_temp,
  MAX(temperature) AS max_temp,
  MIN(temperature) AS min_temp
FROM metrics
GROUP BY bucket, device_id;

-- Retention policy (auto-delete old data)
SELECT add_retention_policy('metrics', INTERVAL '90 days');

-- Compression policy
ALTER TABLE metrics SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'device_id',
  timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('metrics', INTERVAL '7 days');

-- Query: last 24 hours, 5-minute averages
SELECT
  time_bucket('5 minutes', time) AS bucket,
  device_id,
  AVG(temperature) AS avg_temp
FROM metrics
WHERE time > NOW() - INTERVAL '24 hours'
  AND device_id = 'sensor_42'
GROUP BY bucket, device_id
ORDER BY bucket;
```

### 6.5 InfluxDB

InfluxDB is a purpose-built time-series database with its own query language (Flux) and line protocol.

```
# InfluxDB Line Protocol (write)
cpu,host=server01,region=us-east usage=0.72,system=0.15 1700000000000000000
cpu,host=server02,region=eu-west usage=0.45,system=0.08 1700000000000000000

# Structure: measurement,tag_key=tag_value field_key=field_value timestamp

# Flux query: average CPU usage by host, last hour, 5-minute windows
from(bucket: "monitoring")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu" and r._field == "usage")
  |> aggregateWindow(every: 5m, fn: mean)
  |> group(columns: ["host"])
```

### 6.6 TSDB Comparison

| Feature | TimescaleDB | InfluxDB | Prometheus |
|---------|-------------|----------|------------|
| **Base** | PostgreSQL extension | Purpose-built | Purpose-built |
| **Query language** | Full SQL | Flux / InfluxQL | PromQL |
| **Compression** | Columnar compression | TSM engine | Gorilla + delta |
| **Clustering** | Self-managed or cloud | Enterprise only | Thanos/Cortex |
| **Cardinality** | Good (PostgreSQL indexes) | Limited (high cardinality = slow) | Limited |
| **Joins** | Full SQL joins | Limited | No |
| **Best for** | General time-series + SQL | Metrics, IoT | Infrastructure monitoring |

---

## 7. Graph Analytics at Scale

### 7.1 Beyond Graph Databases

While graph databases like Neo4j excel at transactional graph queries (OLTP), graph analytics requires processing the entire graph (OLAP): PageRank, community detection, shortest paths across billions of edges.

### 7.2 Apache Spark GraphX

GraphX is Spark's API for graph-parallel computation. It extends the RDD (Resilient Distributed Dataset) abstraction with a Resilient Distributed Property Graph.

```python
from pyspark.sql import SparkSession
from graphframes import GraphFrame

spark = SparkSession.builder.appName("GraphAnalytics").getOrCreate()

# Vertices DataFrame
vertices = spark.createDataFrame([
    ("1", "Alice", 30),
    ("2", "Bob", 25),
    ("3", "Charlie", 35),
    ("4", "Diana", 28),
], ["id", "name", "age"])

# Edges DataFrame
edges = spark.createDataFrame([
    ("1", "2", "follows"),
    ("2", "3", "follows"),
    ("3", "1", "follows"),
    ("1", "4", "follows"),
    ("4", "3", "follows"),
], ["src", "dst", "relationship"])

# Create graph
g = GraphFrame(vertices, edges)

# PageRank
pagerank = g.pageRank(resetProbability=0.15, maxIter=10)
pagerank.vertices.select("id", "name", "pagerank").show()

# Connected components
cc = g.connectedComponents()
cc.show()

# Shortest paths
sp = g.shortestPaths(landmarks=["3"])
sp.show()

# Triangle count
tc = g.triangleCount()
tc.show()
```

### 7.3 TigerGraph

TigerGraph is a distributed graph analytics platform designed for deep link analytics (multi-hop queries at scale).

```gsql
// GSQL: TigerGraph's query language

// Define schema
CREATE VERTEX Person (PRIMARY_ID id STRING, name STRING, age INT)
CREATE VERTEX Company (PRIMARY_ID id STRING, name STRING)
CREATE DIRECTED EDGE works_at (FROM Person, TO Company, since DATETIME)
CREATE UNDIRECTED EDGE knows (FROM Person, TO Person, strength FLOAT)

// Install a query: 2-hop fraud detection
CREATE QUERY fraud_ring_detection(VERTEX<Person> seed) FOR GRAPH social {
  // Find all people within 2 hops who share financial connections
  Start = {seed};

  // 1-hop: direct connections
  hop1 = SELECT t FROM Start:s -(knows:e)- Person:t
         WHERE e.strength > 0.8;

  // 2-hop: connections of connections
  hop2 = SELECT t FROM hop1:s -(knows:e)- Person:t
         WHERE e.strength > 0.8 AND t != seed;

  // Find shared financial patterns
  suspicious = SELECT t FROM hop2:t -(works_at)- Company:c
               WHERE c.name == "shell_company_pattern";

  PRINT suspicious;
}
```

### 7.4 Graph Analytics Comparison

| Feature | Neo4j | Spark GraphX | TigerGraph |
|---------|-------|-------------|------------|
| **Type** | Graph DB + Analytics | Distributed compute | Distributed graph DB |
| **Scale** | Billions of nodes | Billions of edges | Trillions of edges |
| **Query** | Cypher | Scala/Python API | GSQL |
| **Real-time** | Yes (OLTP + OLAP) | No (batch) | Yes |
| **Best for** | Mixed OLTP/OLAP graph | Batch analytics on Spark | Deep link analytics |

---

## 8. Database-as-a-Service

### 8.1 The Serverless Database Trend

Database-as-a-Service (DBaaS) abstracts away server provisioning, scaling, patching, and backup management. The latest wave of DBaaS offerings goes further with **serverless** architectures that scale to zero when not in use.

### 8.2 Neon

Neon is a serverless PostgreSQL with a separation of storage and compute, enabling instant branching and scale-to-zero.

```
Neon Architecture:

┌────────────────────────────────────────────────────┐
│  Compute Layer (stateless, scale-to-zero)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Postgres │  │ Postgres │  │ Postgres │  (auto) │
│  │ Instance │  │ Instance │  │  (idle)  │         │
│  └────┬─────┘  └────┬─────┘  └──────────┘         │
│       │              │                              │
│       └──────────────┘                              │
│                │                                    │
│  ┌─────────────┴──────────────────────────────────┐ │
│  │  Pageserver (page cache + WAL processing)       │ │
│  └─────────────┬──────────────────────────────────┘ │
│                │                                    │
│  ┌─────────────┴──────────────────────────────────┐ │
│  │  SafeKeeper (WAL durability, consensus)         │ │
│  └─────────────┬──────────────────────────────────┘ │
│                │                                    │
│  ┌─────────────┴──────────────────────────────────┐ │
│  │  Object Storage (S3) - bottomless storage       │ │
│  └────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

**Key features**:
- **Scale to zero**: When no queries are running, compute scales down to zero (no cost)
- **Instant branching**: Create a copy-on-write branch of the entire database (like git branch for data)
- **Bottomless storage**: Data stored in S3, no upper limit

```bash
# Create a branch (instant, copy-on-write)
neonctl branches create --name staging --parent main

# Each branch gets its own compute endpoint
# Useful for: development, testing, previews
```

### 8.3 PlanetScale

PlanetScale is a serverless MySQL platform built on Vitess (YouTube's MySQL sharding middleware).

**Key features**:
- **Non-blocking schema changes**: Schema changes run as background operations, no table locks
- **Database branching**: Create branches for schema changes, merge via deploy requests
- **Sharding**: Built on Vitess, automatic horizontal sharding

```bash
# Git-like workflow for schema changes
pscale branch create main add-column
pscale shell main add-column

# On the branch:
mysql> ALTER TABLE users ADD COLUMN avatar_url VARCHAR(500);

# Create a deploy request (like a pull request)
pscale deploy-request create main add-column

# Review and merge
pscale deploy-request deploy main 1
# Schema change applied to production with zero downtime!
```

### 8.4 Supabase

Supabase is an open-source Firebase alternative built on PostgreSQL.

```
Supabase Stack:
┌──────────────────────────────────────────────┐
│  Dashboard / Studio (Admin UI)                │
│  ┌─────────────────────────────────────────┐  │
│  │  Auth (GoTrue)                          │  │
│  │  Realtime (WebSocket subscriptions)     │  │
│  │  Storage (S3-compatible)                │  │
│  │  Edge Functions (Deno runtime)          │  │
│  │  PostgREST (auto-generated REST API)    │  │
│  │  pgvector (vector search)               │  │
│  └─────────────────────────────────────────┘  │
│                    │                          │
│           ┌────────┴────────┐                 │
│           │   PostgreSQL    │                 │
│           │  (the source    │                 │
│           │   of truth)     │                 │
│           └─────────────────┘                 │
└──────────────────────────────────────────────┘
```

```javascript
// Supabase client (auto-generated API from PostgreSQL schema)
import { createClient } from '@supabase/supabase-js'

const supabase = createClient('https://xxx.supabase.co', 'anon-key')

// Query (translates to SQL automatically)
const { data, error } = await supabase
  .from('products')
  .select('name, price, categories(name)')
  .gte('price', 10)
  .order('price', { ascending: true })
  .limit(20)

// Real-time subscription
const subscription = supabase
  .channel('orders')
  .on('postgres_changes',
    { event: 'INSERT', schema: 'public', table: 'orders' },
    (payload) => console.log('New order:', payload.new)
  )
  .subscribe()

// Row Level Security (RLS) - security at the database level
// Users can only see their own orders
// CREATE POLICY "Users see own orders" ON orders
//   FOR SELECT USING (auth.uid() = user_id);
```

### 8.5 DBaaS Comparison

| Feature | Neon | PlanetScale | Supabase |
|---------|------|-------------|----------|
| **Engine** | PostgreSQL | MySQL (Vitess) | PostgreSQL |
| **Serverless** | Yes (scale to zero) | Yes | Partial |
| **Branching** | Database branching | Schema branching | No |
| **Sharding** | No (single node) | Yes (Vitess) | No |
| **Auto-generated API** | No | No | Yes (PostgREST) |
| **Realtime** | No | No | Yes (WebSocket) |
| **Vector search** | pgvector | No | pgvector |
| **Open source** | Yes (storage) | No (Vitess is OSS) | Yes |
| **Best for** | Dev/test, branching | High-scale MySQL | Full-stack apps |

---

## 9. Data Lakehouse

### 9.1 The Evolution of Data Architecture

```
Era 1: Data Warehouse (1990s-2010s)
  Structured data → ETL → Data Warehouse (Redshift, Snowflake)
  + Strong schema, ACID, SQL
  - Expensive, limited to structured data

Era 2: Data Lake (2010s)
  All data → Dump into HDFS/S3 → Process with Spark
  + Cheap storage, all data types
  - No ACID, no schema enforcement, "data swamp"

Era 3: Data Lakehouse (2020s)
  All data → Lake storage (S3) + Table format (Delta/Iceberg)
  + Cheap storage + ACID + schema + SQL
  Best of both worlds!
```

### 9.2 What is a Lakehouse?

A data lakehouse combines the low-cost storage of a data lake (object storage like S3) with the data management features of a data warehouse (ACID transactions, schema enforcement, indexing).

```
Data Lakehouse Architecture:

┌──────────────────────────────────────────────────────┐
│  Query Engines                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │  Spark  │  │  Trino  │  │ DuckDB  │  ...         │
│  └────┬────┘  └────┬────┘  └────┬────┘              │
│       │            │            │                    │
│       └────────────┼────────────┘                    │
│                    │                                  │
│  ┌─────────────────┴────────────────────────────────┐ │
│  │  Table Format (metadata layer)                    │ │
│  │  ┌────────────┐  ┌────────────┐                   │ │
│  │  │ Delta Lake │  │  Apache    │                   │ │
│  │  │            │  │  Iceberg   │                   │ │
│  │  └────────────┘  └────────────┘                   │ │
│  │  Provides: ACID, schema evolution, time travel,   │ │
│  │  partition pruning, file-level statistics          │ │
│  └──────────────────────────────────────────────────┘ │
│                    │                                  │
│  ┌─────────────────┴────────────────────────────────┐ │
│  │  Object Storage                                   │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐                    │ │
│  │  │  S3  │  │ GCS  │  │ ADLS │                    │ │
│  │  └──────┘  └──────┘  └──────┘                    │ │
│  │  Data stored as Parquet/ORC files                 │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 9.3 Delta Lake

Delta Lake, created by Databricks, adds ACID transactions to Apache Spark on data lakes.

**Transaction log**: Delta Lake maintains a transaction log (`_delta_log/`) in JSON that records every change to the table. This provides:

```
my_table/
├── _delta_log/
│   ├── 00000000000000000000.json   ← initial table creation
│   ├── 00000000000000000001.json   ← INSERT 1000 rows
│   ├── 00000000000000000002.json   ← UPDATE some rows
│   └── 00000000000000000003.json   ← DELETE old rows
├── part-00000-xxxx.parquet
├── part-00001-xxxx.parquet
└── part-00002-xxxx.parquet
```

```python
from delta import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .getOrCreate()

# Write data (ACID)
df.write.format("delta").mode("overwrite").save("/data/events")

# Time travel: read data as of a specific version
df_v1 = spark.read.format("delta") \
    .option("versionAsOf", 1) \
    .load("/data/events")

# Time travel: read data as of a specific timestamp
df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-11-14") \
    .load("/data/events")

# MERGE (upsert): update existing rows, insert new ones
deltaTable = DeltaTable.forPath(spark, "/data/events")
deltaTable.alias("target").merge(
    new_data.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# Schema evolution
df_new_columns.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("/data/events")
```

### 9.4 Apache Iceberg

Apache Iceberg is an open table format for large analytic tables, designed to solve the limitations of Hive tables.

**Key advantages over Delta Lake**:

| Feature | Delta Lake | Apache Iceberg |
|---------|------------|----------------|
| **Engine lock-in** | Optimized for Spark | Engine-agnostic (Spark, Trino, Flink, Dremio) |
| **Hidden partitioning** | Explicit partition columns | Automatic (partition evolution without rewriting) |
| **Metadata** | JSON log + checkpoints | Manifest files + snapshot metadata |
| **Schema evolution** | Add/rename/drop columns | Full evolution including nested types |
| **Governance** | Databricks-led | Apache Foundation (vendor-neutral) |

```sql
-- Iceberg in Spark SQL
CREATE TABLE catalog.db.events (
  event_id BIGINT,
  user_id BIGINT,
  event_type STRING,
  event_time TIMESTAMP,
  properties MAP<STRING, STRING>
) USING iceberg
PARTITIONED BY (days(event_time));  -- hidden partitioning by day

-- Time travel
SELECT * FROM catalog.db.events VERSION AS OF 12345;
SELECT * FROM catalog.db.events TIMESTAMP AS OF '2024-11-14 00:00:00';

-- Snapshot management
SELECT * FROM catalog.db.events.snapshots;

-- Expire old snapshots (reclaim storage)
CALL catalog.system.expire_snapshots('db.events', TIMESTAMP '2024-10-01');
```

### 9.5 Lakehouse vs Traditional Architectures

| Feature | Data Warehouse | Data Lake | Data Lakehouse |
|---------|---------------|-----------|----------------|
| **Storage cost** | High | Low | Low |
| **Data types** | Structured only | All types | All types |
| **ACID** | Yes | No | Yes |
| **Schema** | Schema-on-write | Schema-on-read | Schema-on-write (flexible) |
| **Query performance** | Excellent | Variable | Good (with indexing) |
| **Time travel** | Limited | No | Yes |
| **Governance** | Strong | Weak | Strong |
| **ML support** | Limited | Good (raw data) | Good (raw + structured) |

---

## 10. Exercises

### Exercise 1: NewSQL Comparison

Compare Google Spanner, CockroachDB, and TiDB across the following dimensions. Fill in the table:

| Dimension | Spanner | CockroachDB | TiDB |
|-----------|---------|-------------|------|
| Clock mechanism | | | |
| Default isolation level | | | |
| Replication protocol | | | |
| SQL compatibility | | | |
| HTAP support | | | |
| Open source? | | | |
| Typical deployment | | | |

### Exercise 2: TrueTime and Commit Wait

Google Spanner's TrueTime reports a clock uncertainty of `[T-ε, T+ε]` where ε is typically 1-7ms.

1. Explain why Spanner must "wait out" the uncertainty before committing a transaction.
2. If ε = 5ms, what is the minimum commit latency for a write transaction?
3. If ε were 0 (perfect clock), how would this simplify the protocol?
4. Why can't CockroachDB use the same approach? What does it do instead?
5. Explain a scenario where CockroachDB's HLC approach might result in a "read restart." What is the user-visible impact?

### Exercise 3: Vector Database Design

You are building a RAG (Retrieval-Augmented Generation) system for a customer support chatbot. The knowledge base contains:
- 100,000 support articles (average 500 words each)
- Articles are updated weekly
- Expected query load: 1,000 queries per minute
- Latency requirement: < 100ms per query

Design the vector database component:
1. Choose a vector database system and justify your choice.
2. Choose an embedding model and dimension size.
3. Describe your indexing strategy (IVF, HNSW, or hybrid).
4. How would you handle article updates? (re-embed entire article? chunk-level updates?)
5. How would you implement hybrid search (vector similarity + keyword matching)?

### Exercise 4: Time-Series Schema Design

Design a TimescaleDB schema for a smart building monitoring system:
- 500 sensors across 50 floors
- Each sensor reports temperature, humidity, CO2, and occupancy every 10 seconds
- Common queries: last 24 hours for a specific floor, average per floor per hour, anomaly detection
- Data retention: raw data for 90 days, hourly aggregates for 2 years

Write the SQL for:
1. Table creation with hypertable
2. Continuous aggregate for hourly averages
3. Retention policy
4. Compression policy
5. A query for "floors where average temperature exceeded 28C in the last 6 hours"

### Exercise 5: HTAP Scenario Analysis

A financial trading company currently runs:
- An OLTP database (PostgreSQL) for trade execution
- An OLAP warehouse (Snowflake) with 2-hour ETL lag for risk analysis

They are considering migrating to TiDB for HTAP.

1. What are the benefits of eliminating the ETL pipeline?
2. What risks does running analytics on the same system as transactions introduce?
3. How does TiDB's architecture (TiKV + TiFlash) mitigate the resource contention risk?
4. In what scenarios would you still recommend keeping separate OLTP and OLAP systems?

### Exercise 6: Database Selection

For each of the following applications, select the most appropriate database technology from the options covered in this lesson. Justify your choice.

1. A genomics research platform storing DNA sequence embeddings for similarity search across 10 billion sequences.
2. A factory monitoring system with 10,000 machines, each reporting 50 metrics every second.
3. A global banking application requiring ACID transactions with operations in 30 countries.
4. A startup building a collaborative document editor (like Google Docs) with 10,000 users.
5. A data analytics platform that needs to run SQL queries over 100 TB of event data stored in S3.

### Exercise 7: Lakehouse Design

You are designing a data platform for a ride-sharing company. Data sources include:
- Ride events (10 million rides/day)
- GPS traces (100 points per ride)
- Payment transactions
- Driver/rider profiles
- Surge pricing calculations

Design a lakehouse architecture:
1. Choose between Delta Lake and Apache Iceberg. Justify.
2. Define the table schemas for the three most important tables.
3. Describe the partitioning strategy for each table.
4. How would you implement time travel for auditing payment transactions?
5. What query engines would you use for (a) real-time dashboards, (b) monthly reports, (c) ML feature engineering?

### Exercise 8: Serverless Database Evaluation

Your team is building a SaaS application with these characteristics:
- Multi-tenant (each tenant has isolated data)
- Variable load: peaks at 10x the average
- Development team needs frequent schema changes
- Budget is limited (startup)

Evaluate Neon, PlanetScale, and Supabase for this use case:
1. How does each handle multi-tenancy?
2. How does each handle variable load?
3. How does each handle schema changes in production?
4. What would be your recommendation and why?

### Exercise 9: Consistent Hashing in CockroachDB

CockroachDB divides the key space into Ranges (approximately 512 MB each). Each Range is a Raft group.

1. When a Range grows beyond 512 MB, how does CockroachDB split it? What happens to the Raft group?
2. When a node joins the cluster, how are Ranges rebalanced?
3. What is the difference between a Range split and a Range rebalance?
4. How does CockroachDB decide which node should be the Raft leader for a Range? How does this relate to geo-partitioning?

### Exercise 10: Essay Question

Write a 600-word essay on the following topic:

"The relational database is dead" has been proclaimed many times, yet PostgreSQL usage continues to grow year over year. Based on the material in this lesson (NewSQL, vector databases, time-series databases, lakehouse), argue whether the relational model is becoming more or less relevant. Support your argument with specific examples of how relational databases are adapting (e.g., pgvector, TimescaleDB as extensions) versus being replaced by specialized systems.

---

## 11. References

1. Corbett, J. et al. (2013). "Spanner: Google's Globally-Distributed Database." ACM TODS.
2. Taft, R. et al. (2020). "CockroachDB: The Resilient Geo-Distributed SQL Database." ACM SIGMOD.
3. Huang, D. et al. (2020). "TiDB: A Raft-based HTAP Database." VLDB.
4. Aslett, M. (2011). "How Will the Database Incumbents Respond to NoSQL and NewSQL?" The 451 Group.
5. Johnson, J., Douze, M., Jegou, H. (2019). "Billion-Scale Similarity Search with GPUs." IEEE Trans. on Big Data.
6. Malkov, Y. & Yashunin, D. (2018). "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs." IEEE TPAMI.
7. Armbrust, M. et al. (2021). "Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics." CIDR.
8. Armbrust, M. et al. (2020). "Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores." VLDB.
9. Apache Iceberg Documentation. https://iceberg.apache.org/
10. Pelkonen, T. et al. (2015). "Gorilla: A Fast, Scalable, In-Memory Time Series Database." VLDB.
11. Neon Documentation. https://neon.tech/docs
12. CockroachDB Architecture Documentation. https://www.cockroachlabs.com/docs/stable/architecture/overview.html

---

**Previous**: [14. Distributed Databases](./14_Distributed_Databases.md) | **Next**: [16. Database Design Case Study](./16_Database_Design_Case_Study.md)
