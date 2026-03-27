"""
Exercise Solutions: Lesson 09 - Spark SQL Optimization

Covers:
  - Problem 1: Execution Plan Analysis
  - Problem 2: Join Optimization (100M + 1M records)
  - Problem 3: Skew Handling (concentrated categories)

Note: Pure Python simulation of Spark optimization concepts.
"""

import random
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# Problem 1: Execution Plan Analysis
# Analyze the execution plan of a given query and find optimization points.
# ---------------------------------------------------------------------------

def problem1_execution_plan_analysis():
    """Simulate analyzing a Spark SQL execution plan.

    PySpark equivalent:

        df = spark.sql('''
            SELECT category, COUNT(*) as cnt, SUM(amount) as total
            FROM orders
            WHERE status = 'completed' AND order_date >= '2024-01-01'
            GROUP BY category
            ORDER BY total DESC
        ''')
        df.explain(True)  # Shows all plan stages

    The explain() output has 4 levels:
    - Parsed Logical Plan:  raw SQL -> abstract syntax tree
    - Analyzed Logical Plan: resolve table/column names
    - Optimized Logical Plan: Catalyst optimizations applied
    - Physical Plan:         actual execution strategy (scans, shuffles, sorts)
    """

    plan = {
        "query": """
            SELECT category, COUNT(*) as cnt, SUM(amount) as total
            FROM orders
            WHERE status = 'completed' AND order_date >= '2024-01-01'
            GROUP BY category
            ORDER BY total DESC
        """,
        "parsed_logical_plan": [
            "Sort [total DESC]",
            "  Aggregate [category], [category, count(1) as cnt, sum(amount) as total]",
            "    Filter [status = 'completed' AND order_date >= '2024-01-01']",
            "      Relation [orders]",
        ],
        "optimized_logical_plan": [
            "Sort [total DESC]",
            "  Aggregate [category], [category, count(1) as cnt, sum(amount) as total]",
            "    Filter [status = 'completed' AND order_date >= '2024-01-01']",
            "      FileScan parquet [category, amount, status, order_date]",
            "          PushedFilters: [IsNotNull(status), EqualTo(status, completed),",
            "                          GreaterThanOrEqual(order_date, 2024-01-01)]",
            "          ReadSchema: struct<category:string, amount:double>",
        ],
        "catalyst_optimizations": [
            "1. Predicate Pushdown: WHERE clause pushed to file scan level",
            "   -> Parquet row groups with status != 'completed' are skipped entirely",
            "   -> Date filter prunes partitions if table is partitioned by order_date",
            "",
            "2. Column Pruning: Only 4 columns read (category, amount, status, order_date)",
            "   -> Remaining columns (order_id, customer_id, etc.) are never loaded",
            "   -> Reduces I/O by ~60% for a wide table",
            "",
            "3. Constant Folding: '2024-01-01' string converted to date once at plan time",
        ],
        "optimization_recommendations": [
            "1. PARTITION BY order_date: enables partition pruning for the date filter",
            "   -> Spark skips entire directories for dates before 2024-01-01",
            "",
            "2. Bucketing BY category: if category has low cardinality, bucketing avoids",
            "   the shuffle in the GROUP BY. Pre-sorting within buckets speeds up the final sort.",
            "",
            "3. Cache if reused: If this query feeds multiple downstream queries,",
            "   df.cache() avoids re-scanning the Parquet files.",
            "",
            "4. Consider AQE: spark.sql.adaptive.enabled=true (default in Spark 3.x)",
            "   -> Automatically coalesces small partitions after the shuffle",
            "   -> Handles skew in the GROUP BY stage",
        ],
    }

    print("\n  Query:")
    print(f"  {plan['query'].strip()}")

    print("\n  Parsed Logical Plan:")
    for line in plan["parsed_logical_plan"]:
        print(f"    {line}")

    print("\n  Optimized Logical Plan (after Catalyst):")
    for line in plan["optimized_logical_plan"]:
        print(f"    {line}")

    print("\n  Catalyst Optimizations Applied:")
    for line in plan["catalyst_optimizations"]:
        print(f"    {line}")

    print("\n  Optimization Recommendations:")
    for line in plan["optimization_recommendations"]:
        print(f"    {line}")

    return plan


# ---------------------------------------------------------------------------
# Problem 2: Join Optimization
# Design the optimal method to join a transaction table (100M records)
# with a customer table (1M records).
# ---------------------------------------------------------------------------

def problem2_join_optimization():
    """Demonstrate join optimization strategies for large/small table join.

    PySpark equivalent (broadcast join):

        from pyspark.sql import functions as F

        # Customer table is small (1M rows, ~200MB) -> broadcast it
        result = (
            transactions_df
            .join(
                F.broadcast(customers_df),  # Force broadcast
                on="customer_id",
                how="inner"
            )
        )

    Why broadcast join?
    - Sort-Merge Join (default for large-large): requires shuffling BOTH
      tables by customer_id. Shuffling 100M rows is very expensive (network I/O).
    - Broadcast Join: sends the small table (1M rows, ~200MB) to every executor.
      No shuffle of the large table at all. Each partition performs a local
      hash-map lookup. Orders of magnitude faster.

    Rule of thumb: broadcast when small table < 10MB (default) or up to 1-2GB
    with spark.sql.autoBroadcastJoinThreshold tuning.
    """

    # Simulate the data sizes
    large_table_rows = 100_000_000
    small_table_rows = 1_000_000
    small_table_size_mb = 200  # ~200 bytes per row

    strategies = [
        {
            "name": "Sort-Merge Join (Default)",
            "description": "Both tables shuffled by customer_id, then merge-joined",
            "shuffle_data": f"{large_table_rows // 1_000_000}M + {small_table_rows // 1_000_000}M rows",
            "pros": "Works for any table size",
            "cons": "Full shuffle of 100M rows is very expensive (network + disk I/O)",
            "recommended": False,
        },
        {
            "name": "Broadcast Hash Join",
            "description": "Small table broadcast to all executors, local hash-map lookup",
            "shuffle_data": f"0 rows (only {small_table_size_mb}MB broadcast)",
            "pros": "No shuffle of the large table; O(1) lookups per row",
            "cons": "Small table must fit in executor memory",
            "recommended": True,
        },
        {
            "name": "Bucketed Join",
            "description": "Both tables pre-bucketed by customer_id (write-time optimization)",
            "shuffle_data": "0 rows (pre-organized on disk)",
            "pros": "Zero shuffle even for large-large joins; great if join is frequent",
            "cons": "Requires pre-bucketing at write time; both tables must have same bucket count",
            "recommended": False,  # Overkill for 1M-row table
        },
    ]

    print(f"\n  Scenario: Join {large_table_rows:,} transactions with {small_table_rows:,} customers")
    print(f"  Customer table size: ~{small_table_size_mb}MB\n")

    for s in strategies:
        marker = " ** RECOMMENDED **" if s["recommended"] else ""
        print(f"  Strategy: {s['name']}{marker}")
        print(f"    Mechanism : {s['description']}")
        print(f"    Shuffle   : {s['shuffle_data']}")
        print(f"    Pros      : {s['pros']}")
        print(f"    Cons      : {s['cons']}")
        print()

    print("  Configuration for Broadcast Join:")
    print("    spark.sql.autoBroadcastJoinThreshold = 209715200  # 200MB")
    print("    # or use F.broadcast(customers_df) to force it explicitly")

    # Simulate timing comparison
    print("\n  Simulated Timing Comparison (relative):")
    print(f"    Sort-Merge Join:    ~300 seconds (shuffle-heavy)")
    print(f"    Broadcast Hash Join: ~30 seconds (no shuffle)")
    print(f"    Speedup:            ~10x")

    return strategies


# ---------------------------------------------------------------------------
# Problem 3: Skew Handling
# Improve aggregation performance when data is concentrated in specific
# categories.
# ---------------------------------------------------------------------------

def generate_skewed_data(n: int = 100_000) -> list[dict]:
    """Generate data with skewed category distribution.

    'Electronics' gets ~70% of rows, creating a hot partition in GROUP BY.
    """
    categories_weighted = (
        ["Electronics"] * 70 +
        ["Clothing"] * 15 +
        ["Books"] * 10 +
        ["Food"] * 5
    )
    data = []
    for i in range(n):
        data.append({
            "order_id": i + 1,
            "category": random.choice(categories_weighted),
            "amount": round(random.uniform(10, 500), 2),
        })
    return data


def problem3_skew_handling():
    """Demonstrate skew handling techniques for aggregation.

    The problem:
    - 'Electronics' has 70% of rows. After shuffle (GROUP BY category),
      one partition handles 70K rows while others handle only 5-15K.
    - That one partition becomes a bottleneck (all other tasks finish
      fast and wait).

    Solutions:

    1. Adaptive Query Execution (AQE) - Automatic (Spark 3.x default)
       spark.sql.adaptive.enabled = true
       spark.sql.adaptive.skewJoin.enabled = true
       Spark detects skewed partitions at runtime and splits them.

    2. Salting - Manual technique for severe skew
       Add a random salt column, aggregate with salt, then aggregate again
       to remove the salt. Distributes the hot key across multiple partitions.

    3. Two-phase aggregation - Conceptually similar to salting
       First partial aggregation per partition (local combine), then
       final aggregation. This is what Spark already does (partial_sum),
       but explicit salting helps when partitions themselves are skewed.
    """
    data = generate_skewed_data(100_000)

    # Show distribution
    dist: dict[str, int] = defaultdict(int)
    for d in data:
        dist[d["category"]] += 1

    print("\n  Data Distribution (skewed):")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        bar = "#" * int(pct / 2)
        print(f"    {cat:<15} {count:>7,} ({pct:.1f}%) {bar}")

    # --- Approach 1: Naive GROUP BY (simulated) ---
    print("\n  Approach 1: Naive GROUP BY")
    print("  Problem: 'Electronics' partition processes ~70K rows while")
    print("  others process only 5-15K. Stragglers waste cluster time.")

    start = time.time()
    naive_agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total": 0.0})
    for d in data:
        naive_agg[d["category"]]["count"] += 1
        naive_agg[d["category"]]["total"] += d["amount"]
    naive_time = time.time() - start

    # --- Approach 2: Salted Aggregation ---
    print("\n  Approach 2: Salted Aggregation")
    print("  Step 1: Add salt (random 0-9) to category key")
    print("  Step 2: GROUP BY (category, salt) -> partial aggregates")
    print("  Step 3: GROUP BY category -> final aggregates (removes salt)")

    SALT_BUCKETS = 10

    # PySpark equivalent:
    # salted = df.withColumn("salt", (F.rand() * SALT_BUCKETS).cast("int"))
    # partial = salted.groupBy("category", "salt").agg(
    #     F.count("*").alias("cnt"), F.sum("amount").alias("total"))
    # final = partial.groupBy("category").agg(
    #     F.sum("cnt").alias("count"), F.sum("total").alias("total"))

    start = time.time()
    # Phase 1: Partial aggregation with salt
    partial_agg: dict[tuple[str, int], dict] = defaultdict(lambda: {"count": 0, "total": 0.0})
    for d in data:
        salt = random.randint(0, SALT_BUCKETS - 1)
        key = (d["category"], salt)
        partial_agg[key]["count"] += 1
        partial_agg[key]["total"] += d["amount"]

    # Phase 2: Final aggregation (remove salt)
    salted_agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total": 0.0})
    for (cat, _salt), stats in partial_agg.items():
        salted_agg[cat]["count"] += stats["count"]
        salted_agg[cat]["total"] += stats["total"]
    salted_time = time.time() - start

    # Display results
    print(f"\n  Results:")
    print(f"  {'Category':<15} {'Count':>8} {'Total Revenue':>15} {'Avg':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*15} {'-'*10}")
    for cat, stats in sorted(salted_agg.items(), key=lambda x: -x[1]["total"]):
        avg = stats["total"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {cat:<15} {stats['count']:>8,} ${stats['total']:>13,.2f} ${avg:>8,.2f}")

    # Show why salting helps
    print(f"\n  Why Salting Helps:")
    print(f"    Without salt: 'Electronics' = 1 partition with ~70K rows")
    print(f"    With salt={SALT_BUCKETS}: 'Electronics' split across {SALT_BUCKETS} partitions")
    print(f"    Each 'Electronics' partition handles ~{70_000 // SALT_BUCKETS:,} rows")
    print(f"    -> Parallel processing time reduced by ~{SALT_BUCKETS}x for the hot key")

    # AQE recommendation
    print(f"\n  Recommended: Enable AQE (automatic in Spark 3.x):")
    print(f"    spark.sql.adaptive.enabled = true")
    print(f"    spark.sql.adaptive.skewJoin.enabled = true")
    print(f"    spark.sql.adaptive.skewJoin.skewedPartitionFactor = 5")
    print(f"    spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes = 256MB")

    return dict(salted_agg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Execution Plan Analysis")
    print("=" * 70)
    problem1_execution_plan_analysis()

    print()
    print("=" * 70)
    print("Problem 2: Join Optimization (100M + 1M Records)")
    print("=" * 70)
    problem2_join_optimization()

    print()
    print("=" * 70)
    print("Problem 3: Skew Handling")
    print("=" * 70)
    problem3_skew_handling()
