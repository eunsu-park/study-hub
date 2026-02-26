# NoSQL Data Models

**Previous**: [12. Concurrency Control](./12_Concurrency_Control.md) | **Next**: [14. Distributed Databases](./14_Distributed_Databases.md)

---

The relational model has served as the dominant paradigm for data management since Codd's seminal 1970 paper. Yet as the internet evolved from thousands of users to billions, and as data grew from megabytes to petabytes, practitioners discovered scenarios where the relational model's strict guarantees became liabilities rather than assets. This lesson explores the NoSQL revolution: why it happened, what models emerged, and how to choose the right data model for a given problem.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why the relational model struggles at web scale
2. State and interpret the CAP theorem with its formal proof sketch
3. Contrast BASE and ACID consistency models
4. Design data models using key-value, document, wide-column, and graph paradigms
5. Write basic queries in each NoSQL paradigm
6. Apply a decision framework to select the appropriate data model
7. Understand polyglot persistence and its architectural implications

---

## Table of Contents

1. [Motivation: Limits of the Relational Model](#1-motivation-limits-of-the-relational-model)
2. [The CAP Theorem](#2-the-cap-theorem)
3. [BASE vs ACID](#3-base-vs-acid)
4. [Key-Value Stores](#4-key-value-stores)
5. [Document Stores](#5-document-stores)
6. [Wide-Column Stores](#6-wide-column-stores)
7. [Graph Databases](#7-graph-databases)
8. [Comparison Matrix: When to Use Which Model](#8-comparison-matrix-when-to-use-which-model)
9. [Polyglot Persistence](#9-polyglot-persistence)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Motivation: Limits of the Relational Model

### 1.1 The Impedance Mismatch

Relational databases store data in flat, two-dimensional tables. Modern applications, however, work with rich, nested objects: a single "order" in an e-commerce application includes line items, shipping addresses, payment details, and promotional codes. Mapping these hierarchical objects to normalized tables requires complex JOIN operations at read time and multi-table INSERT operations at write time.

```
Application Object              Relational Tables
┌──────────────────┐            ┌──────────┐   ┌────────────┐
│ Order             │            │ orders   │   │ line_items │
│  ├─ customer      │    ──▶    ├──────────┤   ├────────────┤
│  ├─ items[]       │            │ order_id │──▶│ order_id   │
│  ├─ shipping_addr │            │ cust_id  │   │ product_id │
│  └─ payment       │            │ total    │   │ quantity   │
└──────────────────┘            └──────────┘   └────────────┘
                                      │
                                      ▼
                                ┌──────────────┐   ┌──────────┐
                                │ addresses    │   │ payments │
                                ├──────────────┤   ├──────────┤
                                │ order_id     │   │ order_id │
                                │ street       │   │ method   │
                                │ city         │   │ amount   │
                                └──────────────┘   └──────────┘
```

This "impedance mismatch" between the application's object model and the database's relational model creates overhead in code complexity, development time, and runtime performance.

### 1.2 Scalability Challenges

Relational databases were designed for vertical scaling (bigger machines). When you need more capacity, you buy a faster CPU, more RAM, or faster disks. This approach has hard limits:

| Scaling Dimension | Vertical (Scale Up) | Horizontal (Scale Out) |
|-------------------|---------------------|------------------------|
| **Approach** | Bigger machine | More machines |
| **Cost curve** | Exponential | Linear |
| **Theoretical limit** | Hardware ceiling | Virtually unlimited |
| **Downtime for upgrade** | Usually required | Rolling upgrades |
| **Relational DB support** | Natural fit | Extremely difficult |
| **NoSQL support** | Possible | Natural fit |

Horizontal scaling requires distributing data across multiple nodes, which fundamentally conflicts with several relational guarantees:

- **JOINs across partitions** become network operations with unpredictable latency.
- **Distributed transactions** require complex coordination protocols (2PC) that reduce throughput.
- **Schema changes** (ALTER TABLE) on billions of rows across hundreds of nodes are operationally dangerous.

### 1.3 Schema Rigidity

Relational databases enforce a rigid schema: every row in a table has the same columns. In agile development, requirements change frequently. Each schema change requires:

1. Writing a migration script
2. Testing the migration against production-sized data
3. Coordinating the deployment with application code changes
4. Potentially locking the table during ALTER TABLE operations

For applications with rapidly evolving data models (social media feeds, IoT sensor data, content management), this rigidity imposes significant operational overhead.

### 1.4 The NoSQL Movement

The term "NoSQL" emerged around 2009, initially standing for "Not Only SQL" to emphasize that these systems complement rather than replace relational databases. The key motivations were:

- **Horizontal scalability**: distribute data across commodity hardware
- **Flexible schemas**: adapt to changing data structures without migrations
- **High availability**: remain operational even when nodes fail
- **Performance**: optimize for specific access patterns rather than general-purpose queries
- **Developer productivity**: store data in formats that match application objects

> **Historical Note**: Google's Bigtable paper (2006) and Amazon's Dynamo paper (2007) are considered the foundational papers of the NoSQL movement. They demonstrated that relaxing relational guarantees could achieve performance and scalability that was previously impossible.

---

## 2. The CAP Theorem

### 2.1 Statement

The CAP theorem, formulated by Eric Brewer in 2000 and formally proved by Gilbert and Lynch in 2002, states:

> **CAP Theorem**: In a distributed data store, it is impossible to simultaneously guarantee all three of the following properties:
> - **Consistency (C)**: Every read receives the most recent write or an error.
> - **Availability (A)**: Every request receives a (non-error) response, without the guarantee that it contains the most recent write.
> - **Partition Tolerance (P)**: The system continues to operate despite an arbitrary number of messages being dropped (or delayed) by the network between nodes.

```
                    ┌─────────────────┐
                    │   Consistency   │
                    │       (C)       │
                    └────────┬────────┘
                             │
                   CA        │        CP
               (impossible   │    (sacrifice
               in distributed│    availability)
               systems)      │
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────┴─────┐                            ┌─────┴─────┐
   │Availability│                           │ Partition │
   │    (A)    │                            │ Tolerance │
   │           │                            │    (P)    │
   └────┬──────┘                            └─────┬─────┘
        │                                         │
        └──────────────── AP ─────────────────────┘
                    (sacrifice
                     consistency)
```

### 2.2 Understanding Each Property

**Consistency** (Linearizability): If client A writes value `v` to key `k`, then any subsequent read of `k` by any client must return `v` (or a more recent value). This is linearizable consistency -- the strongest form. It means the system behaves as if there is a single copy of the data.

**Availability**: Every request received by a non-failing node must result in a response. The system cannot simply ignore requests or return errors when it is uncertain about the state.

**Partition Tolerance**: A network partition occurs when messages between nodes are lost or arbitrarily delayed. In any real distributed system, network partitions are inevitable (cables get cut, switches fail, datacenters lose connectivity). Therefore, P is not optional in practice.

### 2.3 Proof Sketch

The proof by Gilbert and Lynch (2002) proceeds by contradiction:

**Setup**: Consider the simplest possible distributed system with two nodes, N1 and N2, connected by a network. Both nodes store a copy of a variable `v`, initially set to `v0`.

**Assume** the system guarantees all three properties: C, A, and P.

**Step 1**: A network partition occurs between N1 and N2. Messages sent from N1 to N2 (and vice versa) are lost. By P, the system must continue to operate.

**Step 2**: A client sends a write request to N1, setting `v = v1`. By A, N1 must accept this write and respond to the client. However, because of the partition, N1 cannot propagate this update to N2.

**Step 3**: Another client sends a read request to N2 for the value of `v`. By A, N2 must respond. The only value N2 has is `v0` (the old value), because the update from N1 was lost in the partition.

**Step 4**: N2 returns `v0`, but the most recent write was `v1`. This violates C (linearizability).

**Contradiction**: We assumed C, A, and P, but derived a violation of C. Therefore, it is impossible to guarantee all three simultaneously.

```
    N1                          N2
    ┌──┐                       ┌──┐
    │v0│                       │v0│    Initial state
    └──┘                       └──┘
     │                          │
     │    ╳╳╳ PARTITION ╳╳╳    │     Network partition
     │                          │
   write(v1)                    │
     │                          │
    ┌──┐                       ┌──┐
    │v1│    cannot replicate   │v0│    N1 updated, N2 stale
    └──┘    ──────╳──────▶     └──┘
                                │
                             read(v) → returns v0  ← VIOLATES C
```

### 2.4 CAP in Practice

Since network partitions are unavoidable in distributed systems, the real choice is between:

- **CP systems**: Sacrifice availability during partitions. When a partition occurs, the system may refuse to answer some requests to maintain consistency. Examples: HBase, MongoDB (with majority read concern), etcd, ZooKeeper.

- **AP systems**: Sacrifice consistency during partitions. When a partition occurs, the system continues to serve requests but may return stale data. Examples: Cassandra, DynamoDB, CouchDB, Riak.

**Important nuances**:

1. **CAP is about behavior during partitions only.** When there is no partition, a system can provide both C and A.

2. **Consistency is a spectrum.** Between strict linearizability and eventual consistency, there are many intermediate levels (causal consistency, read-your-writes, monotonic reads).

3. **The choice is per-operation, not per-system.** A single database can offer different consistency guarantees for different operations. For example, MongoDB allows you to choose `readConcern: "majority"` (CP) or `readConcern: "local"` (AP) on a per-query basis.

### 2.5 PACELC Extension

Daniel Abadi proposed the PACELC theorem as a refinement:

> **PACELC**: If there is a **P**artition, choose between **A**vailability and **C**onsistency; **E**lse (when the system is running normally), choose between **L**atency and **C**onsistency.

| System | P → A or C? | E → L or C? | Classification |
|--------|-------------|-------------|----------------|
| DynamoDB | A | L | PA/EL |
| Cassandra | A | L | PA/EL |
| MongoDB | C | C | PC/EC |
| HBase | C | C | PC/EC |
| PNUTS (Yahoo) | A | C | PA/EC |
| Spanner | C | C | PC/EC |

This captures the observation that even when no partition exists, systems make latency-consistency tradeoffs.

---

## 3. BASE vs ACID

We covered ACID in detail in [Lesson 11](./11_Transaction_Theory.md). Let us now contrast it with the BASE model that most NoSQL systems adopt.

### 3.1 ACID Recap

| Property | Meaning |
|----------|---------|
| **Atomicity** | All operations in a transaction succeed or all fail |
| **Consistency** | A transaction brings the database from one valid state to another |
| **Isolation** | Concurrent transactions do not interfere with each other |
| **Durability** | Once committed, data survives system failures |

### 3.2 BASE Properties

BASE is a backronym proposed by Eric Brewer as the counterpoint to ACID:

| Property | Meaning |
|----------|---------|
| **Basically Available** | The system guarantees availability (in the CAP sense) |
| **Soft state** | The state of the system may change over time, even without input, due to eventual consistency propagation |
| **Eventually consistent** | Given enough time without new updates, all replicas will converge to the same state |

> **Analogy -- News Spreading Through a Town**:
> Imagine a small town without internet or TV, where news spreads only by word of mouth. When something happens (a new store opens), the people nearby hear about it first. Over the next few hours, they tell their neighbors, who tell their neighbors, and so on. At any given moment, some residents know the news while others do not -- the town is in a "soft state." But given enough time with no new events, *everyone* eventually hears the same story. This is eventual consistency: there is no single broadcast that updates everyone simultaneously (unlike ACID's "all-or-nothing" commit), but the system converges to a consistent state over time. The trade-off is that if you ask two residents at the same moment, you might get different answers -- which is acceptable for a social media "likes" counter but unacceptable for a bank balance.

### 3.3 Detailed Comparison

```
          ACID                                    BASE
┌──────────────────────┐            ┌──────────────────────┐
│ Strong consistency    │            │ Eventual consistency  │
│ Pessimistic locking   │            │ Optimistic replication│
│ Centralized           │            │ Distributed           │
│ Schema-first          │            │ Schema-flexible       │
│ Scale up              │            │ Scale out             │
│ Complex query support │            │ Simple query patterns │
│ Lower availability    │            │ Higher availability   │
│ Higher latency (2PC)  │            │ Lower latency         │
└──────────────────────┘            └──────────────────────┘
```

| Dimension | ACID | BASE |
|-----------|------|------|
| **Target** | Correctness above all | Availability above all |
| **Reads** | Always see latest write | May see stale data |
| **Conflict resolution** | Prevent conflicts (locks) | Detect and resolve (CRDTs, LWW) |
| **Application complexity** | Simple (DB handles consistency) | Complex (app handles conflicts) |
| **Use cases** | Banking, inventory, booking | Social feeds, analytics, caching |

### 3.4 Eventual Consistency Models

"Eventually consistent" is not a single model but a family of consistency guarantees:

| Model | Guarantee |
|-------|-----------|
| **Strong eventual consistency** | Replicas that have received the same set of updates are in the same state (no conflicts) |
| **Causal consistency** | If operation A causally precedes B, all nodes see A before B |
| **Read-your-writes** | A process always sees its own writes |
| **Monotonic reads** | If a process reads value v, subsequent reads will never return a value older than v |
| **Monotonic writes** | Writes from a process are applied in order at all replicas |
| **Session consistency** | Within a session, read-your-writes + monotonic reads |

---

## 4. Key-Value Stores

### 4.1 Data Model

The key-value store is the simplest NoSQL data model. It is essentially a distributed hash table:

```
┌─────────────────────────────────────────────────┐
│                  Key-Value Store                 │
│                                                 │
│   Key (String)          Value (Opaque Blob)     │
│   ─────────────         ───────────────────     │
│   "user:1001"     →     {"name": "Alice", ...}  │
│   "session:abc"   →     {token_data...}         │
│   "cache:page:42" →     "<html>...</html>"      │
│   "counter:views" →     "1547382"               │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Characteristics**:
- **Keys** are unique strings (or byte arrays)
- **Values** are opaque blobs -- the store does not interpret their contents
- **Operations** are simple: GET, PUT, DELETE (and sometimes atomic increment, TTL)
- **No secondary indexes** (unless added as a separate feature)
- **No JOINs, no aggregations, no complex queries**

### 4.2 Operations

The API is minimal by design:

```
# Fundamental operations
PUT(key, value)         # Store a key-value pair
GET(key) → value        # Retrieve value by key
DELETE(key)             # Remove a key-value pair

# Extended operations (vendor-specific)
EXISTS(key) → bool      # Check if key exists
EXPIRE(key, ttl)        # Set time-to-live
INCR(key)               # Atomic increment
MGET(key1, key2, ...)   # Batch retrieval
```

### 4.3 Redis

Redis (Remote Dictionary Server) is an in-memory key-value store that supports rich data structures.

**Data structures in Redis**:

```
┌───────────────────────────────────────────────────────┐
│  Redis Data Types                                      │
│                                                       │
│  STRING:  "hello"                                     │
│  LIST:    [a, b, c, d]                                │
│  SET:     {a, b, c}                                   │
│  SORTED SET: {(a,1.0), (b,2.5), (c,3.7)}            │
│  HASH:    {field1: val1, field2: val2}                │
│  BITMAP:  01101001...                                 │
│  STREAM:  [(id1, {k:v}), (id2, {k:v}), ...]         │
│  GEOSPATIAL: {(member, lat, lon), ...}                │
└───────────────────────────────────────────────────────┘
```

**Example Redis session**:

```redis
# Strings
SET user:1001:name "Alice"
GET user:1001:name                    # → "Alice"

# Hash (like a mini-document)
HSET user:1001 name "Alice" email "alice@example.com" age "30"
HGET user:1001 name                   # → "Alice"
HGETALL user:1001                     # → name, Alice, email, alice@example.com, age, 30

# List (message queue pattern)
LPUSH notifications:1001 "New order #42"
LPUSH notifications:1001 "Payment received"
LRANGE notifications:1001 0 -1       # → ["Payment received", "New order #42"]

# Sorted Set (leaderboard)
ZADD leaderboard 1500 "player:alice"
ZADD leaderboard 2300 "player:bob"
ZADD leaderboard 1800 "player:charlie"
ZREVRANGE leaderboard 0 2 WITHSCORES # → bob:2300, charlie:1800, alice:1500

# Atomic increment (page view counter)
INCR page:views:homepage              # → 1
INCR page:views:homepage              # → 2

# TTL (session management)
SET session:abc123 "{user_id: 1001}" EX 3600   # Expires in 1 hour
TTL session:abc123                              # → 3598 (seconds remaining)
```

**Use cases for Redis**:
- **Caching**: Store frequently accessed data with TTL
- **Session storage**: User sessions with automatic expiration
- **Rate limiting**: Use INCR with EXPIRE for sliding windows
- **Real-time leaderboards**: Sorted sets for ranking
- **Pub/Sub messaging**: Lightweight message broker
- **Distributed locks**: SETNX (SET if Not eXists) for mutual exclusion

### 4.4 Amazon DynamoDB

DynamoDB is a fully managed, serverless key-value and document database by AWS.

**Key concepts**:

```
┌─────────────────────────────────────────────────────────┐
│  DynamoDB Table                                          │
│                                                         │
│  Partition Key (PK)  │  Sort Key (SK)  │  Attributes    │
│  ────────────────────┼─────────────────┼────────────     │
│  "USER#1001"         │  "PROFILE"      │  {name, email} │
│  "USER#1001"         │  "ORDER#001"    │  {total, date} │
│  "USER#1001"         │  "ORDER#002"    │  {total, date} │
│  "USER#1002"         │  "PROFILE"      │  {name, email} │
│                                                         │
│  PK → determines partition (physical location)          │
│  PK + SK → uniquely identifies an item                  │
└─────────────────────────────────────────────────────────┘
```

**Single-table design**: Unlike relational databases where you normalize data into many tables, DynamoDB best practices recommend putting all entities into a single table and using composite keys to model relationships:

```
PK              SK                  Attributes
───────────     ───────────────     ─────────────────────
USER#1001       PROFILE             name=Alice, email=...
USER#1001       ORDER#2024-001      total=99.99, status=shipped
USER#1001       ORDER#2024-002      total=45.50, status=pending
PRODUCT#ABC     METADATA            name=Widget, price=9.99
PRODUCT#ABC     REVIEW#1001         rating=5, text="Great!"
ORDER#2024-001  ITEM#ABC            qty=2, price=9.99
```

**Access patterns** drive the design:
- Get user profile: `PK = "USER#1001", SK = "PROFILE"`
- List user's orders: `PK = "USER#1001", SK begins_with "ORDER#"`
- Get order items: `PK = "ORDER#2024-001", SK begins_with "ITEM#"`

### 4.5 Limitations of Key-Value Stores

- **No ad-hoc queries**: You can only look up by key; no way to say "find all users in city X"
- **No relationships**: No JOINs, no foreign keys
- **Application-level consistency**: The application must manage data consistency across keys
- **Data modeling complexity**: Denormalization and single-table design require careful upfront planning

---

## 5. Document Stores

### 5.1 Data Model

Document stores extend the key-value model by making the value a structured, queryable document (typically JSON or BSON):

```json
{
  "_id": "order_1001",
  "customer": {
    "name": "Alice Chen",
    "email": "alice@example.com",
    "address": {
      "street": "123 Main St",
      "city": "San Francisco",
      "state": "CA",
      "zip": "94105"
    }
  },
  "items": [
    {"product_id": "P001", "name": "Widget", "qty": 2, "price": 9.99},
    {"product_id": "P002", "name": "Gadget", "qty": 1, "price": 24.99}
  ],
  "total": 44.97,
  "status": "shipped",
  "created_at": "2024-11-15T10:30:00Z",
  "tags": ["electronics", "priority"]
}
```

**Key characteristics**:
- **Self-describing**: Each document contains its structure
- **Nested structures**: Objects within objects, arrays of objects
- **Flexible schema**: Different documents in the same collection can have different fields
- **Rich querying**: Unlike key-value stores, you can query by any field, including nested fields
- **Indexes**: Secondary indexes on any field, including fields within nested objects

### 5.2 JSON vs BSON

| Feature | JSON | BSON |
|---------|------|------|
| **Format** | Text-based | Binary |
| **Readability** | Human-readable | Machine-optimized |
| **Data types** | String, Number, Boolean, null, Object, Array | 20+ types including Date, ObjectId, Decimal128, Binary |
| **Size** | Compact for text | Slightly larger (type prefixes) |
| **Parsing** | Slower (text parsing) | Faster (direct memory mapping) |
| **Used by** | CouchDB, Elasticsearch | MongoDB |

### 5.3 MongoDB

MongoDB is the most popular document database, using BSON as its storage format.

**Core concepts**:

```
Relational           MongoDB
─────────            ─────────
Database      →      Database
Table         →      Collection
Row           →      Document
Column        →      Field
JOIN          →      Embedding / $lookup
Primary Key   →      _id field
Index         →      Index
```

**CRUD Operations**:

```javascript
// INSERT — Customer data is embedded directly inside the order document.
// This is intentional denormalization: we duplicate the customer's name and email
// so that displaying an order never requires a second query (JOIN equivalent).
// Trade-off: if Alice changes her email, we must update every order document.
db.orders.insertOne({
  customer: { name: "Alice", email: "alice@example.com" },
  items: [
    { product: "Widget", qty: 2, price: 9.99 },
    { product: "Gadget", qty: 1, price: 24.99 }
  ],
  total: 44.97,
  status: "pending",
  created_at: new Date()
});

// FIND (Query)
// Dot notation ("customer.name") reaches into the embedded document —
// this works because MongoDB indexes nested fields natively.
// In a relational DB, this would require a JOIN between orders and customers tables.
db.orders.find({
  "customer.name": "Alice",
  "status": "pending"
});

// Find orders with total > 100, sorted by date.
// The sort + limit pattern is efficient when a compound index on
// {total: 1, created_at: -1} exists — MongoDB uses the index to avoid in-memory sorting.
db.orders.find({ total: { $gt: 100 } })
         .sort({ created_at: -1 })
         .limit(10);

// Querying inside an array: "items.product" matches any element in the items array.
// MongoDB automatically creates a "multikey index" for array fields,
// so this query can still use an index despite items being an array.
db.orders.find({ "items.product": "Widget" });

// UPDATE
// $set modifies only the specified field; other fields remain untouched.
// $currentDate automatically sets a timestamp — useful for audit trails.
db.orders.updateOne(
  { _id: ObjectId("...") },
  {
    $set: { status: "shipped" },
    $currentDate: { updated_at: true }
  }
);

// $push appends to an array in place — no need to read-modify-write the whole document.
// This is atomic at the document level (MongoDB guarantees single-document atomicity).
db.orders.updateOne(
  { _id: ObjectId("...") },
  { $push: { items: { product: "Doohickey", qty: 3, price: 5.99 } } }
);

// DELETE — deleteMany removes all matching documents in one operation.
// No cascading deletes like relational FKs; the application must handle related cleanup.
db.orders.deleteMany({ status: "cancelled" });
```

**Aggregation Pipeline**: MongoDB's framework for complex data processing:

```javascript
// Revenue by product category in the last 30 days.
// The aggregation pipeline processes documents through sequential stages,
// similar to UNIX pipes: each stage transforms and passes data to the next.
db.orders.aggregate([
  // Stage 1: Filter early — $match at the top of the pipeline uses indexes
  // and reduces the data volume for all subsequent stages (like WHERE in SQL).
  { $match: {
    created_at: { $gte: new Date(Date.now() - 30*24*60*60*1000) },
    status: { $ne: "cancelled" }
  }},
  // Stage 2: $unwind "flattens" the items array — each array element becomes
  // its own document. This is necessary because we want to group by individual
  // products, but items are embedded inside order documents (denormalized).
  { $unwind: "$items" },
  // Stage 3: Group by product and compute aggregates.
  // Because items were embedded (not in a separate collection), we can compute
  // this without any JOIN — the trade-off of denormalization pays off here.
  { $group: {
    _id: "$items.product",
    total_revenue: { $sum: { $multiply: ["$items.qty", "$items.price"] } },
    total_units: { $sum: "$items.qty" },
    order_count: { $sum: 1 }
  }},
  // Stage 4: Sort by revenue descending
  { $sort: { total_revenue: -1 } },
  // Stage 5: Limit to top 10
  { $limit: 10 }
]);
```

### 5.4 Schema Design Patterns

In document databases, the key design decision is **embedding vs referencing**:

**Embedding** (denormalization):

```json
// Embedding orders inside the user document — a single read fetches the user
// AND all their orders. This avoids the equivalent of a JOIN.
// Trade-off: the document grows with each order, and MongoDB has a 16 MB document
// size limit. Suitable when the number of embedded items is bounded (one-to-few).
{
  "_id": "user_1001",
  "name": "Alice",
  "orders": [
    { "order_id": "O001", "total": 44.97, "items": ["..."] },
    { "order_id": "O002", "total": 89.50, "items": ["..."] }
  ]
}
```

**Referencing** (normalization):

```json
// Users collection — the user document stays small and stable.
{ "_id": "user_1001", "name": "Alice" }

// Orders collection — orders reference the user by ID, like a foreign key.
// A second query (or $lookup) is needed to fetch the user's orders.
// This is better when the number of orders per user is unbounded.
{ "_id": "O001", "user_id": "user_1001", "total": 44.97, "items": ["..."] }
{ "_id": "O002", "user_id": "user_1001", "total": 89.50, "items": ["..."] }
```

**Decision criteria**:

| Factor | Embed | Reference |
|--------|-------|-----------|
| Data is read together? | Yes → Embed | No → Reference |
| Data size unbounded? | No → Embed | Yes → Reference (16MB doc limit) |
| Data updated independently? | No → Embed | Yes → Reference |
| Cardinality | One-to-few | One-to-many / Many-to-many |

**Common patterns**:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Subset** | Embed most-used fields, reference the rest | Product summary embedded, full specs referenced |
| **Extended Reference** | Store a copy of frequently accessed fields from a referenced document | Order stores customer name + email (not full profile) |
| **Computed** | Pre-compute and store derived values | Store order total rather than computing from items every time |
| **Bucket** | Group time-series data into buckets | One document per hour of sensor readings |
| **Outlier** | Handle documents that exceed embedding limits differently | Flag "has_overflow" and store excess in overflow collection |

### 5.5 Indexing in MongoDB

```javascript
// Single-field index
db.orders.createIndex({ status: 1 });

// Compound index
db.orders.createIndex({ "customer.email": 1, created_at: -1 });

// Multikey index (on array fields)
db.orders.createIndex({ "items.product": 1 });

// Text index (full-text search)
db.orders.createIndex({ "items.product": "text", notes: "text" });

// TTL index (auto-delete after expiration)
db.sessions.createIndex({ created_at: 1 }, { expireAfterSeconds: 3600 });

// Unique index
db.users.createIndex({ email: 1 }, { unique: true });
```

### 5.6 Limitations of Document Stores

- **No multi-document ACID transactions** (historically; MongoDB added them in v4.0, but with performance overhead)
- **Data duplication**: Denormalization means updates may need to touch many documents
- **Unbounded document growth**: Documents that grow without limit degrade performance
- **No built-in referential integrity**: The application must enforce foreign key-like constraints

---

## 6. Wide-Column Stores

### 6.1 Data Model

Wide-column stores (also called column-family stores) organize data by columns rather than rows. They should not be confused with columnar analytical databases (like Parquet or ClickHouse); wide-column stores are designed for operational workloads.

```
┌────────────────────────────────────────────────────────────────┐
│  Wide-Column Store                                             │
│                                                                │
│  Row Key      │ Column Family: "profile"  │ CF: "activity"     │
│  ─────────────┼───────────────────────────┼────────────────     │
│  user:1001    │ name: Alice               │ last_login: ...     │
│               │ email: alice@ex.com       │ page_views: 1547    │
│               │ city: San Francisco       │                     │
│  ─────────────┼───────────────────────────┼────────────────     │
│  user:1002    │ name: Bob                 │ last_login: ...     │
│               │ email: bob@ex.com         │ page_views: 892     │
│               │ phone: +1-555-0123        │ last_purchase: ...  │
│               │                           │                     │
│  Note: user:1001 has no "phone" column,                        │
│        user:1002 has an extra "last_purchase" column.           │
│        Each row can have different columns!                     │
└────────────────────────────────────────────────────────────────┘
```

**Key concepts**:
- **Row Key**: The primary identifier, determines data distribution across nodes
- **Column Family**: A group of related columns, defined at table creation time
- **Column**: A name-value pair within a column family
- **Sparse storage**: Rows do not need to have the same columns; empty columns consume no storage
- **Timestamps**: Each cell (row key + column) can store multiple timestamped versions

### 6.2 Cassandra

Apache Cassandra is a distributed wide-column store designed for high availability with no single point of failure.

**Architecture**:

```
                  ┌─────────────────────────────┐
                  │      Cassandra Ring          │
                  │                              │
                  │     N1 ─── N2                │
                  │    /         \               │
                  │  N6           N3             │
                  │    \         /               │
                  │     N5 ─── N4                │
                  │                              │
                  │  Every node is equal          │
                  │  (no master/slave)           │
                  │  Data distributed by         │
                  │  consistent hashing          │
                  └─────────────────────────────┘
```

**CQL (Cassandra Query Language)**:

```sql
-- Create keyspace (like a database).
-- NetworkTopologyStrategy with 3 replicas per datacenter ensures that
-- even if 2 nodes fail in a DC, one copy of every partition survives.
CREATE KEYSPACE ecommerce
WITH replication = {
  'class': 'NetworkTopologyStrategy',
  'dc1': 3, 'dc2': 3
};

-- Create table with compound primary key.
-- The partition key choice is the single most important design decision in Cassandra:
-- it determines data distribution AND query efficiency.
CREATE TABLE ecommerce.orders (
  customer_id UUID,
  order_date TIMESTAMP,
  order_id UUID,
  total DECIMAL,
  status TEXT,
  items LIST<FROZEN<item_type>>,
  PRIMARY KEY ((customer_id), order_date, order_id)
) WITH CLUSTERING ORDER BY (order_date DESC);

-- PRIMARY KEY anatomy:
--   Partition key: (customer_id) — Cassandra hashes this value to decide which
--     node stores the data. All orders for one customer live on the SAME node,
--     making "get all orders for customer X" a single-node read (fast).
--   Clustering key: order_date, order_id — within a partition, rows are stored
--     sorted by these columns on disk. DESC order means the most recent orders
--     are physically first, making "latest N orders" a sequential disk read.

-- Insert data
INSERT INTO ecommerce.orders (customer_id, order_date, order_id, total, status)
VALUES (uuid(), '2024-11-15', uuid(), 44.97, 'shipped');

-- Query by partition key (FAST — the coordinator hashes customer_id to locate
-- the exact node; no other nodes are contacted). O(1) node lookup.
SELECT * FROM ecommerce.orders
WHERE customer_id = 550e8400-e29b-41d4-a716-446655440000;

-- Query with clustering key range (FAST — because order_date is the clustering
-- key, rows within this partition are sorted by date on disk. Cassandra reads
-- a contiguous byte range — essentially a sequential scan, not a random seek).
SELECT * FROM ecommerce.orders
WHERE customer_id = 550e8400-e29b-41d4-a716-446655440000
  AND order_date >= '2024-01-01'
  AND order_date < '2025-01-01';

-- Query without partition key (SLOW — Cassandra has no idea which node holds
-- 'pending' orders, so it must broadcast to ALL nodes and scan every partition.
-- ALLOW FILTERING is required to acknowledge this full-cluster scan).
-- ANTI-PATTERN: avoid this in production! Create a separate table with
-- status as the partition key if you need this query.
SELECT * FROM ecommerce.orders WHERE status = 'pending' ALLOW FILTERING;
```

**Data modeling principle**: In Cassandra, you model tables around queries, not around entities. If you have N different queries, you may need N different tables (each optimized for one query).

### 6.3 HBase

Apache HBase is a wide-column store built on top of HDFS (Hadoop Distributed File System), inspired by Google's Bigtable.

**Key differences from Cassandra**:

| Feature | Cassandra | HBase |
|---------|-----------|-------|
| **Architecture** | Peer-to-peer (no master) | Master-RegionServer |
| **Consistency** | Tunable (AP by default) | Strong (CP) |
| **Write path** | Log-structured merge tree | Log-structured merge tree |
| **Storage** | Own storage engine | HDFS |
| **Query language** | CQL (SQL-like) | Java API / HBase Shell |
| **Best for** | Write-heavy, multi-datacenter | Random read/write on HDFS data |

### 6.4 Use Cases for Wide-Column Stores

- **Time-series data**: IoT sensor readings, metrics, logs (row key = device_id, clustering key = timestamp)
- **Event logging**: Store application events partitioned by user or service
- **Content management**: Store articles, comments, and metadata
- **Recommendation engines**: Store user-item interaction matrices
- **Messaging systems**: Store messages partitioned by conversation

### 6.5 Limitations

- **No JOINs**: All data for a query must be in one table (denormalized)
- **Limited secondary indexes**: Querying by non-key columns is expensive
- **Complex data modeling**: Designing effective partition keys requires deep understanding of access patterns
- **No multi-partition transactions** (Cassandra added lightweight transactions with Paxos, but they are slow)

---

## 7. Graph Databases

### 7.1 The Property Graph Model

Graph databases store data as a network of nodes and edges, making them ideal for data with complex, interconnected relationships.

```
┌──────────────────────────────────────────────────────────────┐
│  Property Graph Model                                        │
│                                                              │
│  NODE (Vertex)                    EDGE (Relationship)        │
│  ┌─────────────────┐             ────────────────────        │
│  │ Label: Person   │             Type: FOLLOWS               │
│  │ Properties:     │             Properties:                 │
│  │   name: "Alice" │──FOLLOWS──▶   since: "2023-01"          │
│  │   age: 30       │             Direction: Alice → Bob      │
│  └─────────────────┘                                         │
│                                                              │
│  Nodes have:                     Edges have:                 │
│  - One or more labels            - Exactly one type          │
│  - Zero or more properties       - A direction               │
│  - Unique ID                     - Zero or more properties   │
│                                  - Start node and end node   │
└──────────────────────────────────────────────────────────────┘
```

**Example graph**:

```
(Alice:Person)──[:FOLLOWS]──▶(Bob:Person)
     │                            │
     │                            │
  [:LIKES]                    [:WROTE]
     │                            │
     ▼                            ▼
(Neo4j:Product)              (Review:Review)
  {name: "Neo4j"}             {rating: 5,
                               text: "Great DB!"}
     ▲
     │
  [:MADE_BY]
     │
(Neo4j Inc:Company)
  {founded: 2007}
```

### 7.2 Why Graphs?

Consider a social network query: "Find friends of friends who also like the same products as me."

**In SQL** (relational):

```sql
-- Friends of friends who like the same products
SELECT DISTINCT fof.name
FROM users me
JOIN friendships f1 ON me.id = f1.user_id
JOIN friendships f2 ON f1.friend_id = f2.user_id
JOIN user_likes ul1 ON me.id = ul1.user_id
JOIN user_likes ul2 ON f2.friend_id = ul2.user_id
WHERE me.name = 'Alice'
  AND ul1.product_id = ul2.product_id
  AND f2.friend_id != me.id;

-- This requires multiple JOINs, and performance degrades
-- exponentially with the depth of traversal.
```

**In Cypher** (Neo4j):

```cypher
// Same query, expressed naturally as a graph traversal
MATCH (me:Person {name: "Alice"})-[:FRIEND]->()-[:FRIEND]->(fof:Person),
      (me)-[:LIKES]->(product:Product)<-[:LIKES]-(fof)
WHERE fof <> me
RETURN DISTINCT fof.name;
```

**Performance comparison**: For relationship-heavy queries, graph databases maintain constant performance regardless of dataset size (because traversal is local), while relational databases degrade as tables grow (because JOINs scan larger index structures).

```
Query Time
    │
    │  Relational (JOIN-based)
    │  ╱
    │ ╱
    │╱                    Graph DB (traversal-based)
    │─────────────────────────────────────────
    │
    └──────────────────────────────────────── Data Size
```

### 7.3 Cypher Query Language

Cypher is Neo4j's declarative graph query language. Its syntax uses ASCII art to represent graph patterns.

**Pattern syntax**:
```
(node)              -- a node
(n:Label)           -- a node with a label
(n:Label {prop: v}) -- a node with a property filter
-[r:TYPE]->         -- a directed relationship
-[r:TYPE*1..3]->    -- a variable-length path (1 to 3 hops)
```

**CREATE operations**:

```cypher
// Create nodes
CREATE (alice:Person {name: "Alice", age: 30})
CREATE (bob:Person {name: "Bob", age: 28})
CREATE (widget:Product {name: "Widget", price: 9.99})

// Create relationships
MATCH (alice:Person {name: "Alice"}), (bob:Person {name: "Bob"})
CREATE (alice)-[:FOLLOWS {since: date("2023-01-15")}]->(bob);

MATCH (alice:Person {name: "Alice"}), (w:Product {name: "Widget"})
CREATE (alice)-[:PURCHASED {date: date("2024-03-20"), qty: 2}]->(w);
```

**READ operations**:

```cypher
// Find all people Alice follows
MATCH (alice:Person {name: "Alice"})-[:FOLLOWS]->(friend)
RETURN friend.name;

// Find shortest path between two people
MATCH path = shortestPath(
  (alice:Person {name: "Alice"})-[:FOLLOWS*]-(bob:Person {name: "Bob"})
)
RETURN path;

// Recommendation: products bought by people who bought the same products as Alice
MATCH (alice:Person {name: "Alice"})-[:PURCHASED]->(p:Product)<-[:PURCHASED]-(other),
      (other)-[:PURCHASED]->(rec:Product)
WHERE NOT (alice)-[:PURCHASED]->(rec)
RETURN rec.name, COUNT(*) AS score
ORDER BY score DESC
LIMIT 5;

// PageRank-style influence: who has the most followers?
MATCH (p:Person)<-[:FOLLOWS]-(follower)
RETURN p.name, COUNT(follower) AS followers
ORDER BY followers DESC
LIMIT 10;

// Community detection: find clusters of mutual follows
MATCH (a:Person)-[:FOLLOWS]->(b:Person)-[:FOLLOWS]->(a)
RETURN a.name, b.name;
```

**UPDATE and DELETE**:

```cypher
// Update a property
MATCH (alice:Person {name: "Alice"})
SET alice.age = 31;

// Add a label
MATCH (alice:Person {name: "Alice"})
SET alice:PremiumUser;

// Delete a relationship
MATCH (alice:Person {name: "Alice"})-[r:FOLLOWS]->(bob:Person {name: "Bob"})
DELETE r;

// Delete a node and all its relationships
MATCH (n:Person {name: "Eve"})
DETACH DELETE n;
```

### 7.4 Graph Use Cases

| Use Case | Why Graphs Excel |
|----------|------------------|
| **Social networks** | Friend-of-friend queries, influence, community detection |
| **Recommendation engines** | Collaborative filtering via graph traversal |
| **Fraud detection** | Identify suspicious patterns in transaction networks |
| **Knowledge graphs** | Represent entities and their relationships (Google Knowledge Graph, Wikidata) |
| **Network/IT operations** | Model network topology, trace dependencies, root cause analysis |
| **Supply chain** | Track products through manufacturing, shipping, distribution |
| **Access control** | Model permission hierarchies and inheritance |
| **Biology** | Protein interaction networks, gene regulatory networks |

### 7.5 Limitations of Graph Databases

- **Not designed for aggregations**: SUM, AVG, GROUP BY are not where graphs shine
- **Storage overhead**: Storing relationship pointers for every edge uses more space than relational foreign keys
- **Limited horizontal scaling**: Partitioning a highly connected graph across nodes is an NP-hard problem (graph partitioning)
- **Smaller ecosystem**: Fewer tools, fewer developers, less community support compared to relational or document databases
- **Write throughput**: Generally lower than key-value or wide-column stores

---

## 8. Comparison Matrix: When to Use Which Model

### 8.1 Decision Matrix

| Criterion | Key-Value | Document | Wide-Column | Graph | Relational |
|-----------|-----------|----------|-------------|-------|------------|
| **Schema flexibility** | High (opaque values) | High (JSON) | Medium (column families fixed) | High (property graph) | Low (fixed schema) |
| **Query complexity** | Key lookup only | Rich queries on fields | Partition-key based | Graph traversal | Full SQL |
| **Relationships** | None | Embedded/referenced | Denormalized | First-class | JOINs |
| **Write throughput** | Very high | High | Very high | Medium | Medium |
| **Read latency** | Sub-millisecond | Low | Low (partition key) | Variable (depth) | Variable |
| **Horizontal scaling** | Excellent | Good | Excellent | Limited | Poor |
| **ACID transactions** | Limited | Per-document (multi-doc in some) | Limited | Per-node (some support multi) | Full |
| **Aggregations** | None | Aggregation pipeline | Limited | Limited | Full SQL |

### 8.2 Decision Flowchart

```
                         What is your primary need?
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
          High-speed         Rich queries       Complex
          simple lookups     on structured      relationships
                │            data                    │
                │                  │                  │
           Key-Value          Document           Graph DB
           (Redis,            (MongoDB,          (Neo4j,
            DynamoDB)          CouchDB)           Neptune)
                                   │
                              Need massive
                              write scale?
                              ┌────┴────┐
                              │         │
                             Yes       No
                              │         │
                         Wide-Column  Document
                         (Cassandra)  (MongoDB)
```

### 8.3 Concrete Recommendations

| Scenario | Recommended Model | Reasoning |
|----------|-------------------|-----------|
| User sessions | Key-Value (Redis) | Simple GET/SET with TTL |
| Product catalog | Document (MongoDB) | Flexible attributes per product category |
| IoT time-series | Wide-Column (Cassandra) | Massive write throughput, time-ordered partitions |
| Social graph | Graph (Neo4j) | Relationship traversal is the core operation |
| Financial transactions | Relational (PostgreSQL) | ACID compliance mandatory |
| Content management | Document (MongoDB) | Nested, variable structures |
| Real-time analytics | Wide-Column + Key-Value | Cassandra for storage, Redis for caching |
| Fraud detection | Graph (Neo4j) | Pattern matching across transaction networks |
| Shopping cart | Key-Value (Redis) or Document | Fast read/write, flexible structure |
| Recommendation engine | Graph + Document | Graph for relationships, Document for item metadata |

---

## 9. Polyglot Persistence

### 9.1 Definition

Polyglot persistence is the practice of using different data storage technologies for different parts of an application, based on the specific access patterns and requirements of each component.

The term was coined by Martin Fowler and Pramod Sadalage in 2011.

### 9.2 Architecture Example

Consider an e-commerce platform:

```
┌─────────────────────────────────────────────────────────────────┐
│                    E-Commerce Platform                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │  Product      │  │  Order       │  │  User          │        │
│  │  Service      │  │  Service     │  │  Service       │        │
│  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘        │
│         │                 │                   │                  │
│         ▼                 ▼                   ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │  MongoDB     │  │  PostgreSQL  │  │  PostgreSQL    │        │
│  │  (Catalog)   │  │  (Orders)    │  │  (Users)       │        │
│  │              │  │  ACID txns   │  │  Auth + RBAC   │        │
│  └──────────────┘  └──────────────┘  └────────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │  Redis       │  │  Neo4j       │  │  Elasticsearch │        │
│  │  (Cache +    │  │  (Recom-     │  │  (Search)      │        │
│  │   Sessions)  │  │   mendations)│  │                │        │
│  └──────────────┘  └──────────────┘  └────────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  Cassandra   │  │  S3 / Blob   │                             │
│  │  (Analytics  │  │  (Images,    │                             │
│  │   Events)    │  │   Files)     │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 Benefits

- **Optimized performance**: Each service uses the database best suited to its access patterns
- **Independent scaling**: Scale each database independently based on its workload
- **Technology flexibility**: Teams can adopt the best tool for their specific problem

### 9.4 Challenges

- **Operational complexity**: Multiple database technologies require diverse expertise
- **Data consistency**: Keeping data synchronized across databases is difficult
- **Monitoring and debugging**: Different monitoring tools, different log formats
- **Cross-store queries**: Joining data from MongoDB and PostgreSQL requires application-level logic
- **Transaction boundaries**: A single business operation may span multiple databases (e.g., "place order" writes to PostgreSQL for the order and Redis for the cache)

### 9.5 Mitigating Challenges

| Challenge | Mitigation Strategy |
|-----------|---------------------|
| Data consistency | Event-driven architecture (Kafka/RabbitMQ) for eventual consistency |
| Cross-store queries | API composition at the application/gateway layer |
| Operational complexity | Platform team with shared infrastructure (Kubernetes, Terraform) |
| Transaction boundaries | Saga pattern or outbox pattern for distributed transactions |
| Monitoring | Unified observability (OpenTelemetry, Grafana) |

### 9.6 When NOT to Use Polyglot Persistence

- **Small teams**: The operational overhead is not justified
- **Simple applications**: A single PostgreSQL instance handles most workloads well
- **Strict consistency requirements**: Multiple databases make global consistency nearly impossible
- **Early-stage startups**: Premature optimization; start with one database and split later

> **Rule of Thumb**: Start with PostgreSQL for everything. When you hit specific, measurable limitations (performance, scale, data model mismatch), then consider introducing a specialized database for that specific use case.

---

## 10. Exercises

### Exercise 1: CAP Theorem Analysis

For each of the following systems, classify them as CP or AP. Justify your answer.

1. A banking system that processes wire transfers between accounts.
2. A social media "likes" counter that shows approximate counts.
3. A DNS system.
4. A distributed configuration store (like etcd or ZooKeeper).
5. A shopping cart service.

### Exercise 2: Data Model Selection

For each of the following scenarios, choose the most appropriate NoSQL data model (key-value, document, wide-column, or graph) and design a basic schema.

1. **Real-time multiplayer game**: Store player positions, scores, and game state for 100,000 concurrent players. Updates happen 60 times per second per player.

2. **Recipe website**: Store recipes with variable ingredients, steps, nutritional information, user ratings, and comments. Users should be able to search by ingredient.

3. **Genealogy application**: Store family trees with parent-child relationships, marriages, and historical records spanning centuries.

4. **IoT fleet management**: Store GPS coordinates, speed, and fuel level from 50,000 vehicles, sampled every 5 seconds.

### Exercise 3: Redis Design

Design a Redis data model for a rate limiter that:
- Allows 100 API requests per user per minute
- Returns the remaining request count with each response
- Resets the counter at the start of each minute window

Write the pseudocode for the `check_rate_limit(user_id)` function using Redis commands.

### Exercise 4: MongoDB Schema Design

You are building a blog platform. Design a MongoDB schema considering these access patterns:
1. Display a blog post with all its comments
2. Display the 10 most recent posts by a specific author
3. Display the 10 most recent posts across all authors
4. Count the total number of comments per post

Consider:
- A post can have 0 to 10,000 comments
- Comments can have replies (up to 3 levels deep)
- Authors have profiles with name, bio, and avatar

Provide your schema as JSON document examples and explain your embedding vs referencing decisions.

### Exercise 5: Cassandra Data Modeling

Design a Cassandra schema for a messaging application with these queries:
1. Get all messages in a conversation, sorted by timestamp (most recent first)
2. Get all conversations for a user, sorted by last activity
3. Get unread message count per conversation for a user

Write the CQL CREATE TABLE statements and explain your partition key and clustering key choices.

### Exercise 6: Cypher Queries

Given the following graph:

```
(:Person {name: "Alice"})-[:WORKS_AT]->(:Company {name: "TechCorp"})
(:Person {name: "Bob"})-[:WORKS_AT]->(:Company {name: "TechCorp"})
(:Person {name: "Charlie"})-[:WORKS_AT]->(:Company {name: "DataInc"})
(:Person {name: "Alice"})-[:KNOWS]-(:Person {name: "Bob"})
(:Person {name: "Bob"})-[:KNOWS]-(:Person {name: "Charlie"})
(:Person {name: "Alice"})-[:KNOWS]-(:Person {name: "Diana"})
(:Person {name: "Diana"})-[:WORKS_AT]->(:Company {name: "DataInc"})
(:Person {name: "Alice"})-[:SKILL]->(:Technology {name: "Python"})
(:Person {name: "Bob"})-[:SKILL]->(:Technology {name: "Python"})
(:Person {name: "Bob"})-[:SKILL]->(:Technology {name: "Java"})
(:Person {name: "Charlie"})-[:SKILL]->(:Technology {name: "Python"})
(:Person {name: "Diana"})-[:SKILL]->(:Technology {name: "Java"})
```

Write Cypher queries for:
1. Find all people who know someone at "DataInc" (but do not themselves work at DataInc).
2. Find all people who share at least 2 skills with Bob.
3. Find the shortest path between Alice and Charlie.
4. Recommend companies to Alice based on where her connections work (exclude her own company).

### Exercise 7: Polyglot Persistence Design

You are architecting a ride-sharing platform (similar to Uber/Lyft). The application requires:
- Real-time driver location tracking (updated every 2 seconds)
- Ride booking and payment processing
- Driver and rider profiles with ratings
- Route calculation and ETA estimation
- Ride history and analytics
- Fraud detection (identifying suspicious patterns in ride requests)

Design a polyglot persistence architecture:
1. Identify each data domain and its access patterns.
2. Choose a database technology for each domain. Justify your choice.
3. Draw an architecture diagram showing how data flows between services.
4. Identify the main data consistency challenges and propose solutions.

### Exercise 8: CAP Theorem Proof Extension

Extend the Gilbert-Lynch proof sketch from Section 2.3 to show that a **three-node** system (N1, N2, N3) also cannot satisfy C, A, and P simultaneously when the network partitions into two groups: {N1} and {N2, N3}.

Specifically:
1. Define the initial state.
2. Describe the partition.
3. Show the write to the minority partition (N1).
4. Show the read from the majority partition (N2 or N3).
5. Identify the contradiction.

### Exercise 9: Consistency Model Classification

For each scenario below, determine the minimum consistency model required (from strongest to weakest: linearizability, sequential consistency, causal consistency, read-your-writes, eventual consistency):

1. A user updates their profile picture and immediately views their profile page.
2. In a group chat, if Alice replies to Bob's message, everyone should see Bob's message before Alice's reply.
3. An inventory system must never oversell a product.
4. A "like" counter that can be slightly inaccurate but must eventually reflect all likes.
5. An email system where a user always sees their sent messages in the Sent folder.

### Exercise 10: Comparative Analysis

Write a 500-word essay comparing how you would model a university enrollment system (students, courses, enrollments, professors, departments) using:
1. A relational database (PostgreSQL)
2. A document database (MongoDB)
3. A graph database (Neo4j)

For each model:
- Draw/describe the schema
- Write two representative queries
- List one advantage and one disadvantage of the model for this use case

---

## 11. References

1. Brewer, E. (2000). "Towards Robust Distributed Systems" (CAP Theorem keynote). ACM PODC.
2. Gilbert, S. & Lynch, N. (2002). "Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services." ACM SIGACT News.
3. DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-Value Store." SOSP.
4. Chang, F. et al. (2006). "Bigtable: A Distributed Storage System for Structured Data." OSDI.
5. Cattell, R. (2011). "Scalable SQL and NoSQL Data Stores." ACM SIGMOD Record.
6. Sadalage, P. & Fowler, M. (2012). *NoSQL Distilled*. Addison-Wesley.
7. Robinson, I., Webber, J., & Eifrem, E. (2015). *Graph Databases*. O'Reilly Media.
8. Abadi, D. (2012). "Consistency Tradeoffs in Modern Distributed Database System Design." IEEE Computer.
9. MongoDB Manual. "Data Modeling Introduction." https://www.mongodb.com/docs/manual/core/data-modeling-introduction/
10. Apache Cassandra Documentation. https://cassandra.apache.org/doc/latest/

---

**Previous**: [12. Concurrency Control](./12_Concurrency_Control.md) | **Next**: [14. Distributed Databases](./14_Distributed_Databases.md)
