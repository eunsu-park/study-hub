"""
Exercises for Lesson 12: NoSQL Databases
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates NoSQL vs RDBMS decisions, key design, capacity planning, and caching strategies.
"""


# === Exercise 1: NoSQL vs RDBMS Selection ===
def exercise_1():
    """Determine whether RDBMS or NoSQL is the better fit for each scenario."""

    scenarios = [
        {
            "description": (
                "E-commerce platform: orders, customers, products with strong "
                "referential integrity and complex joins for reporting"
            ),
            "choice": "Relational (RDBMS)",
            "reason": (
                "Complex joins across orders/customers/products and strong "
                "referential integrity (foreign keys) are RDBMS strengths. "
                "NoSQL does not support multi-table joins natively."
            ),
        },
        {
            "description": (
                "Real-time gaming leaderboard: 100,000 score updates/sec "
                "with consistent single-digit millisecond reads"
            ),
            "choice": "NoSQL (DynamoDB)",
            "reason": (
                "Extreme write throughput + low-latency reads. DynamoDB on-demand "
                "scales to 100k WPS automatically. Partition key=gameId, sort "
                "key=score enables sorted leaderboard queries in milliseconds."
            ),
        },
        {
            "description": (
                "Social media profiles: each user has different optional fields "
                "(bios, company info, etc.)"
            ),
            "choice": "NoSQL (DynamoDB or Firestore)",
            "reason": (
                "Flexible, schemaless document storage is a NoSQL strength. "
                "Attributes are optional -- no schema migration needed when "
                "adding new field types."
            ),
        },
        {
            "description": (
                "Banking: fund transfers requiring atomicity across "
                "debit and credit operations"
            ),
            "choice": "Relational (RDBMS)",
            "reason": (
                "ACID transactions are critical for financial operations. "
                "A fund transfer must atomically debit one account and credit "
                "another. RDS supports multi-row, multi-table transactions."
            ),
        },
    ]

    print("NoSQL vs RDBMS Selection:")
    print("=" * 70)
    for i, s in enumerate(scenarios, 1):
        print(f"\n  Scenario {i}: {s['description']}")
        print(f"    Choice: {s['choice']}")
        print(f"    Reason: {s['reason']}")

    # Quick decision guide
    print("\n  Decision Guide:")
    print("    Use RDBMS when: complex joins, referential integrity, ACID transactions")
    print("    Use NoSQL when: flexible schema, extreme throughput, simple access patterns")


# === Exercise 2: DynamoDB Key Design ===
def exercise_2():
    """Design DynamoDB primary key and GSI for an order management system."""

    query_patterns = [
        "Retrieve a specific order by order ID",
        "List all orders for a specific customer, sorted by order date",
        "Get all orders placed in the last 7 days (across all customers)",
    ]

    print("DynamoDB Key Design for Order Management:")
    print("=" * 70)
    print("\n  Query Patterns:")
    for i, q in enumerate(query_patterns, 1):
        print(f"    {i}. {q}")
    print()

    # Primary key design
    print("  Primary Key Design:")
    print(f"    {'Key':<20} {'Attribute':<25} {'Reason'}")
    print("    " + "-" * 70)
    print(f"    {'Partition (HASH)':<20} {'customerId':<25} "
          "Groups all orders for a customer on same partition")
    print(f"    {'Sort (RANGE)':<20} {'orderDate#orderId':<25} "
          "Enables date-range queries; orderId ensures uniqueness")
    print()

    # How primary key supports queries
    print("  How this key supports query patterns:")
    print("    Pattern 1 (specific order): Provide both customerId + orderDate#orderId")
    print("    Pattern 2 (customer orders by date): Query(customerId='C-001') with date range")
    print()

    # GSI for cross-customer date queries
    # Why GSI: The primary key partitions by customerId, so querying across ALL
    # customers by date requires a separate index with date as the partition key.
    print("  GSI for cross-customer date queries (Pattern 3):")
    print(f"    {'GSI Key':<20} {'Attribute':<25} {'Reason'}")
    print("    " + "-" * 70)
    print(f"    {'GSI Partition':<20} {'orderDate':<25} "
          "Groups orders by date (e.g., '2026-02-24')")
    print(f"    {'GSI Sort':<20} {'orderId':<25} "
          "Uniquely identifies each order within a date")
    print()

    print("  Query for last 7 days: Query GSI for each of 7 date values (scatter-gather).")
    print()

    # CLI command
    print("  CLI Command:")
    print("    aws dynamodb create-table \\")
    print("        --table-name Orders \\")
    print("        --attribute-definitions \\")
    print("            AttributeName=customerId,AttributeType=S \\")
    print("            AttributeName=orderDateOrderId,AttributeType=S \\")
    print("            AttributeName=orderDate,AttributeType=S \\")
    print("            AttributeName=orderId,AttributeType=S \\")
    print("        --key-schema \\")
    print("            AttributeName=customerId,KeyType=HASH \\")
    print("            AttributeName=orderDateOrderId,KeyType=RANGE \\")
    print("        --global-secondary-indexes '[{")
    print('            "IndexName": "OrdersByDate",')
    print('            "KeySchema": [')
    print('                {"AttributeName": "orderDate", "KeyType": "HASH"},')
    print('                {"AttributeName": "orderId", "KeyType": "RANGE"}')
    print("            ],")
    print('            "Projection": {"ProjectionType": "ALL"}')
    print("        }]' \\")
    print("        --billing-mode PAY_PER_REQUEST")


# === Exercise 3: Capacity Mode Selection ===
def exercise_3():
    """Evaluate Provisioned vs On-Demand capacity for a product catalog."""

    # Traffic patterns
    patterns = {
        "weekday": {"reads_per_sec": 50, "writes_per_sec": 5, "hours": 16},
        "black_friday": {"reads_per_sec": 2000, "writes_per_sec": 500, "hours": 6},
        "weekend": {"reads_per_sec": 10, "writes_per_sec": 1, "hours": 48},
    }

    print("DynamoDB Capacity Mode Selection:")
    print("=" * 70)
    print("\n  Traffic Patterns:")
    print(f"    {'Period':<18} {'Reads/sec':<15} {'Writes/sec':<15} {'Duration'}")
    print("    " + "-" * 60)
    for period, p in patterns.items():
        print(f"    {period:<18} {p['reads_per_sec']:<15} "
              f"{p['writes_per_sec']:<15} {p['hours']}h")
    print()

    peak_to_avg_ratio = patterns["black_friday"]["reads_per_sec"] / patterns["weekday"]["reads_per_sec"]
    print(f"  Peak-to-average ratio: {peak_to_avg_ratio:.0f}x")
    print()

    # Cost comparison
    # On-Demand pricing (ap-northeast-2)
    rru_price = 0.000000125  # per RRU
    wru_price = 0.000000625  # per WRU

    # Provisioned pricing
    rcu_hourly = 0.00013
    wcu_hourly = 0.00065

    # On-Demand estimate: sum of reads across all periods
    weekday_reads_year = patterns["weekday"]["reads_per_sec"] * 3600 * patterns["weekday"]["hours"] * 260
    bf_reads = patterns["black_friday"]["reads_per_sec"] * 3600 * patterns["black_friday"]["hours"]
    weekend_reads_year = patterns["weekend"]["reads_per_sec"] * 3600 * patterns["weekend"]["hours"] * 52
    total_reads = weekday_reads_year + bf_reads + weekend_reads_year
    on_demand_read_cost = total_reads * rru_price

    # Provisioned at peak (2000 RCU all year)
    provisioned_read_cost = 2000 * rcu_hourly * 8760

    print("  Cost Comparison (approximate, reads only):")
    print(f"    On-Demand: ~${on_demand_read_cost:,.0f}/year")
    print(f"    Provisioned at peak (2000 RCU): ~${provisioned_read_cost:,.0f}/year")
    print()

    print("  Recommendation: On-Demand capacity")
    print("    - 40x peak-to-average ratio makes provisioned wasteful.")
    print("    - On-Demand scales instantly for Black Friday without pre-provisioning.")
    print("    - No risk of throttling during traffic spikes.")
    print()
    print("  When Provisioned is better:")
    print("    - Consistent 50 reads/sec all year -> 50 RCU provisioned is much cheaper.")
    print("    - Use Provisioned for steady, predictable traffic.")


# === Exercise 4: ElastiCache Use Case ===
def exercise_4():
    """Design a caching strategy with ElastiCache Redis for product pages."""

    print("ElastiCache Redis Caching Strategy:")
    print("=" * 70)
    print()
    print("  Context: 50,000 products, updated at most once/day.")
    print("  Current response time: 800ms (mostly DB queries).")
    print()

    # Caching pattern
    print("  Caching Pattern: Cache-Aside (Lazy Loading)")
    print("  " + "-" * 50)
    print("    1. Check cache: GET product:{product_id}")
    print("    2. If CACHE HIT -> return cached data (1-5ms)")
    print("    3. If CACHE MISS -> query RDS, store in cache, return data")
    print()

    # Cache key design
    print("  Cache Key Design:")
    keys = [
        ("Individual product", "product:{product_id}", "product:12345"),
        ("Category page", "products:category:{id}:page:{n}", "products:category:electronics:page:1"),
    ]
    for name, pattern, example in keys:
        print(f"    {name}:")
        print(f"      Pattern: {pattern}")
        print(f"      Example: {example}")
    print()

    # TTL strategy
    # Why 1 hour: Products update at most once/day, so a 1-hour TTL means
    # users see stale data for at most 1 hour after an update -- acceptable
    # for product descriptions. For price-sensitive data, use shorter TTLs.
    print("  TTL Strategy:")
    ttl_configs = [
        ("Product details (description, specs)", "3600s (1 hour)",
         "Updated at most once/day -> 1h stale window acceptable."),
        ("Time-sensitive data (inventory, flash sale)", "60s",
         "Reduces stale data risk for rapidly changing fields."),
    ]
    for data_type, ttl, reason in ttl_configs:
        print(f"    {data_type}: TTL = {ttl}")
        print(f"      Why: {reason}")
    print()

    # Cache invalidation
    print("  Cache Invalidation on Update:")
    print("    def update_product(product_id, data):")
    print("        db.update(f'UPDATE products SET ... WHERE id={product_id}')")
    print("        redis.delete(f'product:{product_id}')  # Invalidate immediately")
    print()

    # Expected improvement
    cache_hit_rate = 0.95
    db_latency_ms = 800
    cache_latency_ms = 5
    avg_latency = cache_hit_rate * cache_latency_ms + (1 - cache_hit_rate) * db_latency_ms
    print(f"  Expected Improvement:")
    print(f"    Cache hit rate (estimated): {cache_hit_rate * 100:.0f}%")
    print(f"    Before: {db_latency_ms}ms (every request hits DB)")
    print(f"    After:  ~{avg_latency:.0f}ms average "
          f"({cache_hit_rate * 100:.0f}% x {cache_latency_ms}ms + "
          f"{(1 - cache_hit_rate) * 100:.0f}% x {db_latency_ms}ms)")
    print(f"    DB load reduction: ~{cache_hit_rate * 100:.0f}% fewer queries")


# === Exercise 5: DynamoDB Global Tables ===
def exercise_5():
    """Configure DynamoDB Global Tables for multi-region gaming."""

    print("DynamoDB Global Tables for Multi-Region Gaming:")
    print("=" * 70)
    print()
    print("  Scenario: Gaming company with players in South Korea and US.")
    print("  Goal: Low-latency reads in both regions + survive regional outage.")
    print()

    # Feature identification
    print("  1. Feature: DynamoDB Global Tables")
    print("     Multi-region, multi-active replication. Writes in any region")
    print("     are automatically replicated to all other regions (~1 sec).")
    print()

    # Consistency implications
    # Why this matters: A Korean player updates their profile. A US player
    # queries that profile 500ms later. They may see the old value because
    # cross-region replication hasn't completed yet.
    print("  2. Consistency Implications:")
    implications = [
        ("Eventual consistency between regions",
         "A write in Seoul may take ~1s to appear in Virginia."),
        ("Conflict resolution: last writer wins",
         "Simultaneous writes to the same item in different regions -> "
         "timestamp-based resolution. Losing write is silently discarded."),
        ("Within-region reads",
         "Strongly consistent reads available within a single region only."),
        ("Design recommendation",
         "Route writes for a given user to their home region to "
         "minimize cross-region conflicts."),
    ]
    for name, detail in implications:
        print(f"    - {name}:")
        print(f"      {detail}")
    print()

    # CLI commands
    print("  3. CLI Commands to Add Replica Region:")
    print()
    print("    # Add us-east-1 as a replica region")
    print("    aws dynamodb update-table \\")
    print("        --table-name PlayerProfiles \\")
    print("        --replica-updates '[{")
    print('            "Create": {')
    print('                "RegionName": "us-east-1"')
    print("            }")
    print("        }]' \\")
    print("        --region ap-northeast-2")
    print()
    print("    # Verify replica status")
    print("    aws dynamodb describe-table \\")
    print("        --table-name PlayerProfiles \\")
    print("        --region ap-northeast-2 \\")
    print("        --query 'Table.Replicas'")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: NoSQL vs RDBMS Selection ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: DynamoDB Key Design ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Capacity Mode Selection ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: ElastiCache Use Case ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: DynamoDB Global Tables ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
