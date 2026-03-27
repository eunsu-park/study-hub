"""
Modern Data Tools: Polars and DuckDB
=====================================
Demonstrates:
- Polars basics and expression API
- Polars lazy evaluation
- DuckDB SQL analytics
- Arrow interoperability
- Pandas vs Polars vs DuckDB performance

Requirements:
    pip install polars duckdb pyarrow pandas numpy
"""

import numpy as np
import time


# ── 1. Polars Basics ──────────────────────────────────────────────

def demo_polars_basics():
    """Basic Polars operations."""
    import polars as pl

    # Create DataFrame
    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age": [30, 25, 35, 28, 32],
        "city": ["NYC", "LA", "NYC", "LA", "NYC"],
        "salary": [75000, 65000, 90000, 70000, 85000],
    })
    print("DataFrame:")
    print(df)

    # Filter
    print("\nAge > 28:")
    print(df.filter(pl.col("age") > 28))

    # Select + transform
    print("\nWith bonus:")
    print(df.select([
        "name",
        "salary",
        (pl.col("salary") * 0.1).alias("bonus"),
        (pl.col("salary") * 1.1).alias("total_comp"),
    ]))

    # Group by
    print("\nBy city:")
    print(df.group_by("city").agg(
        pl.col("salary").mean().alias("avg_salary"),
        pl.col("age").mean().alias("avg_age"),
        pl.len().alias("count"),
    ))

    # Sort
    print("\nSorted by salary (desc):")
    print(df.sort("salary", descending=True))


# ── 2. Polars Expression API ──────────────────────────────────────

def demo_expressions():
    """Polars expression API with larger dataset."""
    import polars as pl

    np.random.seed(42)
    N = 100_000

    df = pl.DataFrame({
        "user_id": np.random.randint(1, 1001, N),
        "product": np.random.choice(["A", "B", "C", "D", "E"], N),
        "amount": np.random.exponential(50, N).round(2),
        "date": pl.Series(np.random.choice(
            pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 12, 31), eager=True), N
        )),
    })

    # Complex expression chain
    result = (
        df
        .filter(pl.col("amount") > 10)
        .with_columns(
            pl.col("date").dt.month().alias("month"),
            pl.col("amount").log().alias("log_amount"),
            (pl.col("amount") > pl.col("amount").mean()).alias("above_avg"),
        )
        .group_by(["product", "month"])
        .agg(
            pl.col("amount").sum().alias("total"),
            pl.col("amount").mean().alias("avg"),
            pl.col("user_id").n_unique().alias("users"),
            pl.col("amount").quantile(0.95).alias("p95"),
        )
        .sort(["product", "month"])
    )

    print("Monthly product summary (first 15 rows):")
    print(result.head(15))

    # Window functions
    window_result = df.head(1000).with_columns(
        pl.col("amount").rank().over("product").alias("rank_in_product"),
        pl.col("amount").cum_sum().over("user_id").alias("cum_spend"),
        (pl.col("amount") - pl.col("amount").mean().over("product")).alias("diff_from_avg"),
    )
    print("\nWindow functions (first 10 rows):")
    print(window_result.select(["user_id", "product", "amount",
                                 "rank_in_product", "cum_spend", "diff_from_avg"]).head(10))

    return df


# ── 3. Polars Lazy Evaluation ─────────────────────────────────────

def demo_lazy(df):
    """Lazy evaluation and query optimization."""
    import polars as pl

    # Build lazy query
    lazy_q = (
        df.lazy()
        .filter(pl.col("amount") > 20)
        .group_by("product")
        .agg(
            pl.col("amount").sum().alias("total"),
            pl.col("amount").mean().alias("avg"),
            pl.len().alias("count"),
        )
        .sort("total", descending=True)
    )

    # Show optimized plan
    print("Optimized Query Plan:")
    print(lazy_q.explain())

    # Execute
    result = lazy_q.collect()
    print("\nResult:")
    print(result)


# ── 4. DuckDB SQL ─────────────────────────────────────────────────

def demo_duckdb(pl_df):
    """DuckDB SQL analytics on a Polars DataFrame."""
    import duckdb

    # Query Polars DataFrame directly
    print("Basic aggregation:")
    result = duckdb.sql("""
        SELECT
            product,
            COUNT(*) as orders,
            ROUND(SUM(amount), 2) as revenue,
            ROUND(AVG(amount), 2) as avg_order,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount), 2) as p95
        FROM pl_df
        WHERE amount > 10
        GROUP BY product
        ORDER BY revenue DESC
    """)
    print(result)

    # Window functions
    print("\nMonthly growth:")
    result = duckdb.sql("""
        WITH monthly AS (
            SELECT
                product,
                EXTRACT(MONTH FROM date) as month,
                SUM(amount) as revenue
            FROM pl_df
            GROUP BY product, month
        )
        SELECT
            product, month, ROUND(revenue, 0) as revenue,
            ROUND(LAG(revenue) OVER (PARTITION BY product ORDER BY month), 0) as prev,
            ROUND(100.0 * (revenue - LAG(revenue) OVER (PARTITION BY product ORDER BY month))
                  / NULLIF(LAG(revenue) OVER (PARTITION BY product ORDER BY month), 0), 1) as growth_pct
        FROM monthly
        WHERE product = 'A'
        ORDER BY month
    """)
    print(result)


# ── 5. Performance Comparison ─────────────────────────────────────

def demo_benchmark():
    """Benchmark Pandas vs Polars vs DuckDB."""
    import polars as pl
    import pandas as pd
    import duckdb

    np.random.seed(42)
    N = 5_000_000
    print(f"Benchmarking on {N:,} rows...\n")

    data = {
        "category": np.random.choice(["A", "B", "C", "D", "E"], N),
        "value": np.random.randn(N),
        "amount": np.random.exponential(100, N),
    }

    # Pandas
    pd_df = pd.DataFrame(data)
    t0 = time.time()
    pd_result = pd_df.groupby("category").agg(
        total=("amount", "sum"),
        avg_val=("value", "mean"),
        count=("value", "count"),
    )
    pd_time = time.time() - t0

    # Polars (eager)
    pl_df = pl.DataFrame(data)
    t0 = time.time()
    pl_result = pl_df.group_by("category").agg(
        pl.col("amount").sum().alias("total"),
        pl.col("value").mean().alias("avg_val"),
        pl.len().alias("count"),
    )
    pl_eager_time = time.time() - t0

    # Polars (lazy)
    t0 = time.time()
    pl_result_lazy = (
        pl_df.lazy()
        .group_by("category")
        .agg(
            pl.col("amount").sum().alias("total"),
            pl.col("value").mean().alias("avg_val"),
            pl.len().alias("count"),
        )
        .collect()
    )
    pl_lazy_time = time.time() - t0

    # DuckDB
    t0 = time.time()
    db_result = duckdb.sql("""
        SELECT category, SUM(amount) as total,
               AVG(value) as avg_val, COUNT(*) as count
        FROM pl_df GROUP BY category
    """)
    db_time = time.time() - t0

    print(f"{'Tool':<18} {'Time':>8} {'Speedup':>10}")
    print("-" * 38)
    print(f"{'Pandas':<18} {pd_time:>7.3f}s {'1.0x':>10}")
    print(f"{'Polars (eager)':<18} {pl_eager_time:>7.3f}s {pd_time/pl_eager_time:>9.1f}x")
    print(f"{'Polars (lazy)':<18} {pl_lazy_time:>7.3f}s {pd_time/pl_lazy_time:>9.1f}x")
    print(f"{'DuckDB':<18} {db_time:>7.3f}s {pd_time/db_time:>9.1f}x")


# ── 6. Arrow Interoperability ─────────────────────────────────────

def demo_arrow():
    """Arrow-based data exchange between tools."""
    import polars as pl
    import pandas as pd
    import pyarrow as pa
    import duckdb

    # Create in Polars
    pl_df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"]})
    print("Polars DataFrame:")
    print(pl_df)

    # Polars → Arrow
    arrow = pl_df.to_arrow()
    print(f"\nArrow schema: {arrow.schema}")

    # Arrow → Pandas
    pd_df = arrow.to_pandas()
    print(f"\nPandas dtypes:\n{pd_df.dtypes}")

    # Pandas → Polars (round trip)
    pl_back = pl.from_pandas(pd_df)
    print(f"\nRound trip OK: {pl_df.frame_equal(pl_back)}")

    # DuckDB queries all
    print("\nDuckDB on Polars:")
    print(duckdb.sql("SELECT b, SUM(a) as total FROM pl_df GROUP BY b"))


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    demos = {
        "basics": demo_polars_basics,
        "expressions": lambda: demo_lazy(demo_expressions()),
        "duckdb": lambda: demo_duckdb(demo_expressions()),
        "benchmark": demo_benchmark,
        "arrow": demo_arrow,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in demos:
        print("Usage: python 08_polars_duckdb.py <demo>")
        print(f"Available: {', '.join(demos.keys())}")
        print("\nRunning benchmark...")
        demo_benchmark()
    else:
        demos[sys.argv[1]]()
