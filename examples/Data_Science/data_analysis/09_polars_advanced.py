"""
Advanced Polars Operations
==========================
Demonstrates:
- Lazy evaluation and query optimization
- Expression API (select, filter, group_by, with_columns)
- Window functions in Polars
- Joining strategies
- String and datetime operations
- Performance comparison with Pandas patterns

Requirements:
    pip install polars

Theory:
- Polars uses Apache Arrow columnar format in memory.
- Lazy evaluation builds a query plan that Polars optimizes before execution:
  predicate pushdown, projection pushdown, slice pushdown, common
  subexpression elimination, and parallel execution.
- Expressions are the building blocks: col(), lit(), when/then/otherwise.

Adapted from Data_Science Lesson 29.
"""

try:
    import polars as pl
except ImportError:
    print("Polars not installed. Run: pip install polars")
    print("Showing conceptual examples with print statements instead.\n")
    pl = None

import time
import numpy as np


def create_sample_data() -> "pl.DataFrame":
    """Create a sample sales dataset."""
    np.random.seed(42)
    n = 100_000

    data = {
        "date": pl.date_range(
            pl.date(2023, 1, 1),
            pl.date(2025, 12, 31),
            eager=True,
        ).sample(n, with_replacement=True, seed=42),
        "product": np.random.choice(
            ["Laptop", "Mouse", "Keyboard", "Monitor", "Headset"],
            n,
        ).tolist(),
        "region": np.random.choice(
            ["North", "South", "East", "West"],
            n,
        ).tolist(),
        "quantity": np.random.randint(1, 20, n).tolist(),
        "unit_price": np.round(
            np.random.uniform(10, 1500, n), 2
        ).tolist(),
        "customer_id": np.random.randint(1, 5000, n).tolist(),
    }
    return pl.DataFrame(data)


# ── Demos ─────────────────────────────────────────────────────────

def demo_lazy_evaluation():
    print("=" * 60)
    print("LAZY EVALUATION & QUERY OPTIMIZATION")
    print("=" * 60)

    if pl is None:
        print("  (Polars not available — showing concepts only)")
        return

    df = create_sample_data()

    # Lazy query — nothing executes until .collect()
    query = (
        df.lazy()
        .filter(pl.col("unit_price") > 100)
        .group_by("product", "region")
        .agg([
            pl.col("quantity").sum().alias("total_qty"),
            (pl.col("quantity") * pl.col("unit_price")).sum().alias("revenue"),
            pl.col("customer_id").n_unique().alias("unique_customers"),
        ])
        .sort("revenue", descending=True)
    )

    # Show the optimized plan
    print("\n  Query plan:")
    print(f"  {query.explain()[:200]}...")

    # Execute
    result = query.collect()
    print(f"\n  Result shape: {result.shape}")
    print(f"\n  Top 5 by revenue:")
    print(result.head(5))


def demo_expressions():
    print("\n" + "=" * 60)
    print("EXPRESSION API")
    print("=" * 60)

    if pl is None:
        print("  (Polars not available)")
        return

    df = create_sample_data()

    # with_columns — add/transform columns
    enriched = df.with_columns([
        (pl.col("quantity") * pl.col("unit_price")).alias("line_total"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.year().alias("year"),
        pl.when(pl.col("unit_price") > 500)
        .then(pl.lit("Premium"))
        .when(pl.col("unit_price") > 100)
        .then(pl.lit("Standard"))
        .otherwise(pl.lit("Budget"))
        .alias("tier"),
    ])

    print("\n  Schema after enrichment:")
    for name, dtype in enriched.schema.items():
        print(f"    {name}: {dtype}")

    # Aggregation with multiple expressions
    summary = enriched.group_by("tier").agg([
        pl.count().alias("order_count"),
        pl.col("line_total").mean().alias("avg_order_value"),
        pl.col("line_total").quantile(0.95).alias("p95_order_value"),
        pl.col("customer_id").n_unique().alias("unique_customers"),
    ]).sort("avg_order_value", descending=True)

    print(f"\n  Tier summary:")
    print(summary)


def demo_window_functions():
    print("\n" + "=" * 60)
    print("WINDOW FUNCTIONS")
    print("=" * 60)

    if pl is None:
        print("  (Polars not available)")
        return

    df = create_sample_data()
    df = df.with_columns(
        (pl.col("quantity") * pl.col("unit_price")).alias("revenue")
    )

    # Window functions with over()
    result = (
        df.lazy()
        .with_columns([
            # Rank within each region by revenue
            pl.col("revenue")
            .rank(descending=True)
            .over("region")
            .alias("rank_in_region"),

            # Running total per product
            pl.col("revenue")
            .cum_sum()
            .over("product")
            .alias("cumulative_revenue"),

            # Percentage of region total
            (pl.col("revenue") / pl.col("revenue").sum().over("region") * 100)
            .alias("pct_of_region"),
        ])
        .filter(pl.col("rank_in_region") <= 3)
        .sort("region", "rank_in_region")
        .select(["region", "product", "revenue", "rank_in_region", "pct_of_region"])
        .collect()
    )

    print(f"\n  Top 3 sales per region:")
    print(result.head(12))


def demo_joins():
    print("\n" + "=" * 60)
    print("JOIN OPERATIONS")
    print("=" * 60)

    if pl is None:
        print("  (Polars not available)")
        return

    # Products dimension table
    products = pl.DataFrame({
        "product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Headset"],
        "category": ["Computing", "Peripherals", "Peripherals", "Computing", "Audio"],
        "weight_kg": [2.0, 0.1, 0.5, 5.0, 0.3],
    })

    # Regions dimension table
    regions = pl.DataFrame({
        "region": ["North", "South", "East", "West"],
        "timezone": ["EST", "CST", "EST", "PST"],
        "manager": ["Alice", "Bob", "Charlie", "Diana"],
    })

    df = create_sample_data().head(20)

    # Join with both dimension tables
    enriched = (
        df.join(products, on="product", how="left")
          .join(regions, on="region", how="left")
    )

    print(f"\n  Enriched schema: {list(enriched.columns)}")
    print(f"\n  Sample (5 rows):")
    print(enriched.select([
        "date", "product", "category", "region", "manager", "quantity"
    ]).head(5))


def demo_string_datetime():
    print("\n" + "=" * 60)
    print("STRING & DATETIME OPERATIONS")
    print("=" * 60)

    if pl is None:
        print("  (Polars not available)")
        return

    df = create_sample_data()

    # Datetime operations
    time_analysis = (
        df.lazy()
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.strftime("%B").alias("month_name"),
        ])
        .group_by("year", "quarter")
        .agg([
            pl.count().alias("orders"),
            (pl.col("quantity") * pl.col("unit_price")).sum().alias("revenue"),
        ])
        .sort("year", "quarter")
        .collect()
    )

    print(f"\n  Quarterly revenue:")
    print(time_analysis)

    # String operations
    string_demo = df.select([
        pl.col("product"),
        pl.col("product").str.to_uppercase().alias("upper"),
        pl.col("product").str.len_chars().alias("name_length"),
        pl.col("product").str.contains("o|e").alias("has_vowel_oe"),
        pl.col("region").str.slice(0, 1).alias("region_code"),
    ]).unique().sort("product")

    print(f"\n  String operations:")
    print(string_demo)


def demo_performance_comparison():
    print("\n" + "=" * 60)
    print("PERFORMANCE: LAZY vs EAGER")
    print("=" * 60)

    if pl is None:
        print("  (Polars not available)")
        return

    df = create_sample_data()

    # Eager execution
    start = time.perf_counter()
    for _ in range(10):
        _ = (
            df.filter(pl.col("unit_price") > 100)
            .group_by("product")
            .agg(pl.col("quantity").sum())
            .sort("quantity", descending=True)
        )
    eager_time = (time.perf_counter() - start) / 10

    # Lazy execution
    start = time.perf_counter()
    for _ in range(10):
        _ = (
            df.lazy()
            .filter(pl.col("unit_price") > 100)
            .group_by("product")
            .agg(pl.col("quantity").sum())
            .sort("quantity", descending=True)
            .collect()
        )
    lazy_time = (time.perf_counter() - start) / 10

    print(f"\n  Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Eager: {eager_time*1000:.2f} ms (avg of 10 runs)")
    print(f"  Lazy:  {lazy_time*1000:.2f} ms (avg of 10 runs)")
    print(f"  Ratio: {eager_time/lazy_time:.2f}x")

    print(f"""
  Polars vs Pandas — Key Differences:
    {'Feature':<25} {'Polars':<20} {'Pandas':<20}
    {'-'*25} {'-'*20} {'-'*20}
    {'Backend':<25} {'Apache Arrow':<20} {'NumPy':<20}
    {'Index':<25} {'No index':<20} {'Row index':<20}
    {'Lazy evaluation':<25} {'Yes':<20} {'No':<20}
    {'Multithreading':<25} {'Built-in':<20} {'Limited':<20}
    {'String handling':<25} {'Arrow strings':<20} {'Python objects':<20}
    {'Null handling':<25} {'Native':<20} {'NaN/None mix':<20}
    {'Memory efficiency':<25} {'Better':<20} {'Good':<20}""")


if __name__ == "__main__":
    demo_lazy_evaluation()
    demo_expressions()
    demo_window_functions()
    demo_joins()
    demo_string_datetime()
    demo_performance_comparison()
