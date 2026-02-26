# NoSQL Databases

**Previous**: [Managed Relational Databases](./11_Managed_Relational_DB.md) | **Next**: [Identity and Access Management](./13_Identity_Access_Management.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the CAP theorem and how NoSQL databases make trade-offs between consistency and availability
2. Distinguish between key-value, document, wide-column, and in-memory NoSQL database types
3. Compare AWS DynamoDB and GCP Firestore/Bigtable in features and pricing models
4. Design a DynamoDB table with appropriate partition keys and sort keys
5. Configure read/write capacity modes (provisioned vs. on-demand) for cost optimization
6. Implement secondary indexes for flexible query patterns
7. Identify workloads where NoSQL is a better fit than relational databases

---

Not every workload fits the relational model. Applications with massive scale, flexible schemas, or single-digit-millisecond latency requirements often benefit from NoSQL databases. Cloud-managed NoSQL services provide automatic scaling, built-in replication, and zero server management, making them a natural fit for modern distributed applications.

## 1. NoSQL Overview

### 1.1 NoSQL vs RDBMS

| Item | RDBMS | NoSQL |
|------|-------|-------|
| Schema | Strict schema | Flexible schema |
| Scalability | Vertical scaling | Horizontal scaling |
| Transaction | ACID | BASE (some ACID) |
| Query | SQL | Various APIs |
| Use Cases | Transactions, complex relationships | Large volume, flexible data |

### 1.2 Service Comparison

| Type | AWS | GCP |
|------|-----|-----|
| Key-Value / Document | DynamoDB | Firestore |
| Wide Column | - | Bigtable |
| In-Memory Cache | ElastiCache | Memorystore |
| Document (MongoDB) | DocumentDB | MongoDB Atlas (Marketplace) |

---

## 2. AWS DynamoDB

### 2.1 DynamoDB Overview

**Features:**
- Fully managed Key-Value / Document DB
- Millisecond latency
- Unlimited scaling
- Serverless (on-demand capacity)

**Core Concepts:**
- **Table**: Data container
- **Item**: Record
- **Attribute**: Field
- **Primary Key**: Partition key + (optional) sort key

### 2.2 Table Creation

```bash
# Create table (partition key only)
aws dynamodb create-table \
    --table-name Users \
    --attribute-definitions \
        AttributeName=userId,AttributeType=S \
    --key-schema \
        AttributeName=userId,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

# Create table (partition key + sort key)
aws dynamodb create-table \
    --table-name Orders \
    --attribute-definitions \
        AttributeName=customerId,AttributeType=S \
        AttributeName=orderId,AttributeType=S \
    --key-schema \
        AttributeName=customerId,KeyType=HASH \
        AttributeName=orderId,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST

# List tables
aws dynamodb list-tables

# Table information
aws dynamodb describe-table --table-name Users
```

### 2.3 CRUD Operations

```bash
# Add item (PutItem)
aws dynamodb put-item \
    --table-name Users \
    --item '{
        "userId": {"S": "user-001"},
        "name": {"S": "John Doe"},
        "email": {"S": "john@example.com"},
        "age": {"N": "30"}
    }'

# Get item (GetItem)
aws dynamodb get-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}'

# Update item (UpdateItem)
aws dynamodb update-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}' \
    --update-expression "SET age = :newAge" \
    --expression-attribute-values '{":newAge": {"N": "31"}}'

# Delete item (DeleteItem)
aws dynamodb delete-item \
    --table-name Users \
    --key '{"userId": {"S": "user-001"}}'

# Scan (entire table)
aws dynamodb scan --table-name Users

# Query (partition key based)
aws dynamodb query \
    --table-name Orders \
    --key-condition-expression "customerId = :cid" \
    --expression-attribute-values '{":cid": {"S": "customer-001"}}'
```

### 2.4 Python SDK (boto3)

```python
import boto3
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')

# Add item
table.put_item(Item={
    'userId': 'user-002',
    'name': 'Jane Doe',
    'email': 'jane@example.com',
    'age': 25
})

# Get item
response = table.get_item(Key={'userId': 'user-002'})
item = response.get('Item')

# Query (with GSI)
response = table.query(
    IndexName='email-index',
    KeyConditionExpression='email = :email',
    ExpressionAttributeValues={':email': 'jane@example.com'}
)

# Batch write
with table.batch_writer() as batch:
    for i in range(100):
        batch.put_item(Item={'userId': f'user-{i}', 'name': f'User {i}'})
```

### 2.5 Global Secondary Index (GSI)

```bash
# Add GSI
aws dynamodb update-table \
    --table-name Users \
    --attribute-definitions \
        AttributeName=email,AttributeType=S \
    --global-secondary-index-updates '[
        {
            "Create": {
                "IndexName": "email-index",
                "KeySchema": [{"AttributeName": "email", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"}
            }
        }
    ]'
```

### 2.6 DynamoDB Streams

Stream for Change Data Capture (CDC).

```bash
# Enable stream
aws dynamodb update-table \
    --table-name Users \
    --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES

# Connect Lambda trigger
aws lambda create-event-source-mapping \
    --function-name process-dynamodb \
    --event-source-arn arn:aws:dynamodb:...:table/Users/stream/xxx \
    --starting-position LATEST
```

---

## 3. GCP Firestore

### 3.1 Firestore Overview

**Features:**
- Document-based NoSQL DB
- Real-time synchronization
- Offline support
- Automatic scaling

**Core Concepts:**
- **Collection**: Group of documents
- **Document**: JSON-like data
- **Subcollection**: Hierarchical structure

### 3.2 Firestore Setup

```bash
# Enable Firestore API
gcloud services enable firestore.googleapis.com

# Create database (Native mode)
gcloud firestore databases create \
    --location=asia-northeast3 \
    --type=firestore-native
```

### 3.3 Python SDK

```python
from google.cloud import firestore

db = firestore.Client()

# Add document (auto ID)
doc_ref = db.collection('users').add({
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
})

# Add/update document (specified ID)
db.collection('users').document('user-001').set({
    'name': 'Jane Doe',
    'email': 'jane@example.com',
    'age': 25
})

# Get document
doc = db.collection('users').document('user-001').get()
if doc.exists:
    print(doc.to_dict())

# Partial update
db.collection('users').document('user-001').update({
    'age': 26
})

# Delete document
db.collection('users').document('user-001').delete()

# Query
users = db.collection('users').where('age', '>=', 25).stream()
for user in users:
    print(f'{user.id} => {user.to_dict()}')

# Complex query (index required)
users = db.collection('users') \
    .where('age', '>=', 25) \
    .order_by('age') \
    .limit(10) \
    .stream()
```

### 3.4 Real-time Listeners

```python
# Document change detection
def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        print(f'Received document snapshot: {doc.id}')

doc_ref = db.collection('users').document('user-001')
doc_watch = doc_ref.on_snapshot(on_snapshot)

# Collection change detection
col_ref = db.collection('users')
col_watch = col_ref.on_snapshot(on_snapshot)
```

### 3.5 Security Rules

```javascript
// firestore.rules
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Authenticated users can access only their own documents
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }

    // Public read
    match /public/{document=**} {
      allow read: if true;
      allow write: if request.auth != null;
    }
  }
}
```

```bash
# Deploy security rules
firebase deploy --only firestore:rules
```

---

## 4. In-Memory Cache

### 4.1 AWS ElastiCache

**Supported Engines:**
- Redis
- Memcached

```bash
# Create Redis cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id my-redis \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-cache-nodes 1 \
    --cache-subnet-group-name my-subnet-group \
    --security-group-ids sg-12345678

# Create replication group (high availability)
aws elasticache create-replication-group \
    --replication-group-id my-redis-cluster \
    --replication-group-description "Redis cluster" \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-node-groups 1 \
    --replicas-per-node-group 1 \
    --automatic-failover-enabled \
    --cache-subnet-group-name my-subnet-group

# Check endpoint
aws elasticache describe-cache-clusters \
    --cache-cluster-id my-redis \
    --show-cache-node-info
```

**Python Connection:**
```python
import redis

# Single node
r = redis.Redis(
    host='my-redis.xxx.cache.amazonaws.com',
    port=6379,
    decode_responses=True
)

# SET/GET
r.set('key', 'value')
value = r.get('key')

# Hash
r.hset('user:1000', mapping={'name': 'John', 'email': 'john@example.com'})
user = r.hgetall('user:1000')

# TTL
r.setex('session:abc', 3600, 'user_data')
```

### 4.2 GCP Memorystore

**Supported Engines:**
- Redis
- Memcached

```bash
# Create Redis instance
gcloud redis instances create my-redis \
    --region=asia-northeast3 \
    --tier=BASIC \
    --size=1 \
    --redis-version=redis_6_x

# Instance information
gcloud redis instances describe my-redis \
    --region=asia-northeast3

# Connection information (host/port)
gcloud redis instances describe my-redis \
    --region=asia-northeast3 \
    --format='value(host,port)'
```

**Connection:**
```python
import redis

# Memorystore Redis (Private IP)
r = redis.Redis(
    host='10.0.0.3',  # Private IP
    port=6379,
    decode_responses=True
)

r.set('hello', 'world')
print(r.get('hello'))
```

---

## 5. Capacity Modes

### 5.1 DynamoDB Capacity Modes

| Mode | Features | Best For |
|------|------|-----------|
| **On-Demand** | Auto scaling, pay per request | Unpredictable traffic |
| **Provisioned** | Pre-specified capacity | Stable traffic |

```bash
# On-demand mode
aws dynamodb update-table \
    --table-name Users \
    --billing-mode PAY_PER_REQUEST

# Provisioned mode
aws dynamodb update-table \
    --table-name Users \
    --billing-mode PROVISIONED \
    --provisioned-throughput ReadCapacityUnits=100,WriteCapacityUnits=100

# Auto Scaling setup
aws application-autoscaling register-scalable-target \
    --service-namespace dynamodb \
    --resource-id "table/Users" \
    --scalable-dimension "dynamodb:table:ReadCapacityUnits" \
    --min-capacity 5 \
    --max-capacity 1000
```

### 5.2 Firestore Capacity

Firestore is fully serverless with automatic scaling.

**Pricing:**
- Document reads: $0.06 / 100,000
- Document writes: $0.18 / 100,000
- Document deletes: $0.02 / 100,000
- Storage: $0.18 / GB / month

---

## 6. Cost Comparison

### 6.1 DynamoDB

| Item | On-Demand | Provisioned |
|------|---------|-----------|
| Reads | $0.25 / 1M RRU | $0.00013 / RCU / hour |
| Writes | $1.25 / 1M WRU | $0.00065 / WCU / hour |
| Storage | $0.25 / GB / month | $0.25 / GB / month |

### 6.2 Firestore

| Item | Cost |
|------|------|
| Document reads | $0.06 / 100,000 |
| Document writes | $0.18 / 100,000 |
| Storage | $0.18 / GB / month |

### 6.3 ElastiCache / Memorystore

| Service | Node Type | Hourly Cost |
|--------|----------|------------|
| ElastiCache | cache.t3.micro | ~$0.02 |
| ElastiCache | cache.r5.large | ~$0.20 |
| Memorystore | 1GB Basic | ~$0.05 |
| Memorystore | 1GB Standard (HA) | ~$0.10 |

---

## 7. Use Case Selection

| Use Case | Recommended Service |
|----------|-----------|
| Session management | ElastiCache / Memorystore |
| User profiles | DynamoDB / Firestore |
| Real-time chat | Firestore (real-time sync) |
| Game leaderboard | ElastiCache Redis |
| IoT data | DynamoDB / Bigtable |
| Shopping cart | DynamoDB / Firestore |
| Caching | ElastiCache / Memorystore |

---

## 8. Design Patterns

### 8.1 DynamoDB Single Table Design

```
PK              | SK              | Attributes
----------------|-----------------|------------------
USER#123        | USER#123        | name, email
USER#123        | ORDER#001       | product, quantity
USER#123        | ORDER#002       | product, quantity
PRODUCT#A       | PRODUCT#A       | name, price
PRODUCT#A       | REVIEW#001      | rating, comment
```

### 8.2 Cache Patterns

**Cache-Aside (Lazy Loading):**
```python
def get_user(user_id):
    # Check cache
    cached = cache.get(f'user:{user_id}')
    if cached:
        return cached

    # Query from DB
    user = db.get_user(user_id)

    # Save to cache
    cache.setex(f'user:{user_id}', 3600, user)
    return user
```

**Write-Through:**
```python
def update_user(user_id, data):
    # Update DB
    db.update_user(user_id, data)

    # Update cache
    cache.set(f'user:{user_id}', data)
```

---

## 9. Next Steps

- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - IAM
- [11_Managed_Relational_DB.md](./11_Managed_Relational_DB.md) - RDB

---

## Exercises

### Exercise 1: NoSQL vs RDBMS Selection

For each scenario, determine whether a relational database (RDS/Cloud SQL) or a NoSQL database (DynamoDB/Firestore) is the better fit. Justify your answer.

1. An e-commerce platform needs to store orders, customers, and products with strong referential integrity and complex joins for reporting.
2. A real-time gaming leaderboard that must handle 100,000 score updates per second with consistent single-digit millisecond reads.
3. A social media application stores user profiles where each user can have different optional fields (some have bios, others have company info, etc.).
4. A banking application that processes fund transfers and requires atomicity across debit and credit operations.

<details>
<summary>Show Answer</summary>

1. **Relational (RDBMS)** — Complex joins across orders/customers/products and strong referential integrity (foreign keys) are RDBMS strengths. NoSQL databases do not support multi-table joins natively; you'd have to denormalize everything or perform multiple round trips.

2. **NoSQL (DynamoDB)** — Leaderboards require extreme write throughput and low-latency reads. DynamoDB's on-demand mode scales to handle 100,000 WPS automatically. A DynamoDB table with `gameId` as partition key and `score` as sort key, plus a Global Secondary Index, enables sorted leaderboard queries in milliseconds.

3. **NoSQL (DynamoDB or Firestore)** — Flexible, schemaless document storage is a NoSQL strength. DynamoDB attributes are optional — you only store what exists for each user. Adding a new field type (e.g., "company info") requires no schema migration.

4. **Relational (RDBMS)** — ACID transactions are critical for financial operations. A fund transfer must atomically debit one account and credit another. RDS supports multi-row, multi-table transactions. DynamoDB supports transactions but only within a single table or across a limited set of items.

</details>

### Exercise 2: DynamoDB Key Design

You are designing a DynamoDB table for an order management system. Orders have the following query patterns:
- Retrieve a specific order by order ID.
- List all orders for a specific customer, sorted by order date.
- Get all orders placed in the last 7 days (across all customers).

Design the primary key (partition key and sort key) and any Global Secondary Index (GSI) needed to support these query patterns.

<details>
<summary>Show Answer</summary>

**Primary key design**:

| Key | Attribute | Reason |
|-----|-----------|--------|
| Partition Key (HASH) | `customerId` | Groups all orders for a customer on the same partition for efficient retrieval |
| Sort Key (RANGE) | `orderDate#orderId` | Enables range queries by date. Prefix `orderDate` (ISO format: `2026-02-24`) ensures lexicographic sort by date; append `orderId` to guarantee uniqueness |

This design supports:
- **Query 2** (all orders for a customer, sorted by date): `Query(customerId="C-001")` with optional date range filter on the sort key.
- **Query 1** (specific order): Provide both `customerId` and the full sort key `orderDate#orderId`.

**GSI for cross-customer date queries (Query 3)**:

| | Attribute |
|--|-----------|
| GSI Partition Key | `orderDate` (date only, e.g., `2026-02-24`) |
| GSI Sort Key | `orderId` |

To get all orders in the last 7 days: Query the GSI with `orderDate` for each of the 7 date values. This is a scatter-gather pattern.

```bash
aws dynamodb create-table \
    --table-name Orders \
    --attribute-definitions \
        AttributeName=customerId,AttributeType=S \
        AttributeName=orderDateOrderId,AttributeType=S \
        AttributeName=orderDate,AttributeType=S \
        AttributeName=orderId,AttributeType=S \
    --key-schema \
        AttributeName=customerId,KeyType=HASH \
        AttributeName=orderDateOrderId,KeyType=RANGE \
    --global-secondary-indexes '[{
        "IndexName": "OrdersByDate",
        "KeySchema": [
            {"AttributeName": "orderDate", "KeyType": "HASH"},
            {"AttributeName": "orderId", "KeyType": "RANGE"}
        ],
        "Projection": {"ProjectionType": "ALL"}
    }]' \
    --billing-mode PAY_PER_REQUEST
```

</details>

### Exercise 3: Capacity Mode Selection

A DynamoDB table serves a product catalog. Usage patterns are:
- Weekdays: ~50 reads/second, ~5 writes/second consistently.
- Black Friday: peaks at 2,000 reads/second and 500 writes/second for ~6 hours.
- Weekends: minimal traffic (~10 reads/second).

Should you use Provisioned or On-Demand capacity? Justify your answer and estimate the cost difference.

<details>
<summary>Show Answer</summary>

**Recommendation: On-Demand capacity**

**Reasoning**:
- The traffic pattern is highly variable with a 40x peak-to-average ratio during Black Friday.
- On-Demand scales instantly to any level without pre-provisioning, handling the 2,000 RPS peak without throttling.
- With Provisioned capacity set for Black Friday peaks, you'd pay for 2,000 RCU and 500 WCU all year round — even during off-peak weekends when only 10 reads/second are needed.

**Cost comparison** (approximate, based on `ap-northeast-2` pricing):

**On-Demand**:
- Normal weekday: 50 reads × 3600 × 16 hours × $0.000000125/RCU = ~$0.36/day
- Black Friday (6 hours peak): 2000 × 6 × 3600 × $0.000000125 = ~$5.40 + writes
- Total for the year: roughly **$200–300/year**

**Provisioned at Black Friday peak (2,000 RCU, 500 WCU)**:
- 2,000 RCU × $0.00013/hour × 8,760 hours = **~$2,277/year** (just for reads)
- Plus write capacity

**Savings with On-Demand**: Several thousand dollars per year.

**When Provisioned would be better**: If the load was consistently 50 reads/second all year with no spikes, Provisioned at 50 RCU would cost: 50 × $0.00013 × 8,760 = ~$57/year — much cheaper than On-Demand for predictable workloads. Use Provisioned for steady, predictable traffic; On-Demand for variable or unpredictable traffic.

</details>

### Exercise 4: ElastiCache Use Case

An e-commerce application currently queries RDS MySQL for product details on every page load. The product catalog has 50,000 products that change at most once per day. Response time for product pages is 800ms, mostly due to database queries.

Describe how you would use ElastiCache (Redis) to improve response time. Include: the caching pattern, cache key design, and TTL strategy.

<details>
<summary>Show Answer</summary>

**Caching pattern: Cache-Aside (Lazy Loading)**

```
Application request for product detail:
1. Check cache: GET product:{product_id}
2. If CACHE HIT → return cached data (1-5ms response time)
3. If CACHE MISS → query RDS, store in cache, return data
```

**Cache key design**:
- Individual product: `product:{product_id}` (e.g., `product:12345`)
- Product list by category: `products:category:{category_id}:page:{page}` (e.g., `products:category:electronics:page:1`)

**TTL strategy**:
- Products are updated at most once per day → set TTL to **3600 seconds (1 hour)** or **86400 seconds (24 hours)**.
- A 1-hour TTL means: after a product update, the stale cached value expires within 1 hour. For most product details (price, description), this is acceptable.
- For time-sensitive data (inventory count, flash sale prices): use a shorter TTL (60 seconds) or actively invalidate the cache key when data changes.

**Cache invalidation on update**:
```python
# When product is updated
def update_product(product_id, data):
    db.update(f"UPDATE products SET ... WHERE id={product_id}")
    redis.delete(f"product:{product_id}")  # Invalidate immediately
```

**Expected improvement**: Database queries for cached products drop by ~95% (only cache misses hit the database). Response time for product pages drops from 800ms to ~50ms (Redis in-memory lookup + minor processing).

</details>

### Exercise 5: DynamoDB Global Tables

A gaming company has players in South Korea (Asia-Pacific) and the United States. They want their DynamoDB player profile table to be available with low latency in both regions and survive a complete regional outage.

1. What DynamoDB feature enables this?
2. What are the consistency implications to understand before using this feature?
3. Write the CLI commands to add a replica region.

<details>
<summary>Show Answer</summary>

1. **DynamoDB Global Tables** — A fully managed, multi-region, multi-active replication feature. Writes to the table in any region are automatically replicated to all other regions, typically within 1 second. Each region has a fully writable local copy.

2. **Consistency implications**:
   - **Eventual consistency between regions**: A write in Seoul may take up to ~1 second to appear in Virginia. If a Korean player updates their profile, a US player (or a US-region read) may briefly see the old value.
   - **Conflict resolution**: If two regions simultaneously update the same item (e.g., player buys something in both regions due to a network partition), DynamoDB uses **"last writer wins"** (based on timestamp). The losing write is silently discarded.
   - **Within-region reads**: Strongly consistent reads are available within a single region; cross-region reads are always eventually consistent.
   - **Design recommendation**: Design the application so that writes for a given user are always routed to their home region to minimize cross-region conflicts.

3. **CLI to add a replica region**:
```bash
# First, the table must be in on-demand or provisioned mode
# Add us-east-1 as a replica region
aws dynamodb update-table \
    --table-name PlayerProfiles \
    --replica-updates '[{
        "Create": {
            "RegionName": "us-east-1"
        }
    }]' \
    --region ap-northeast-2

# Verify replica status
aws dynamodb describe-table \
    --table-name PlayerProfiles \
    --region ap-northeast-2 \
    --query 'Table.Replicas'
```

</details>

---

## References

- [AWS DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
- [AWS ElastiCache Documentation](https://docs.aws.amazon.com/elasticache/)
- [GCP Firestore Documentation](https://cloud.google.com/firestore/docs)
- [GCP Memorystore Documentation](https://cloud.google.com/memorystore/docs)
