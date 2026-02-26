"""
DuckDB Analytics
================
Demonstrates:
- In-process OLAP database for analytics
- SQL on DataFrames and CSV/Parquet files
- Window functions and CTEs
- Integration with Pandas and Polars
- Performance on analytical queries

Requirements:
    pip install duckdb pandas

Theory:
- DuckDB is an in-process columnar database optimized for OLAP workloads.
- It runs inside the application process (like SQLite for OLTP).
- Vectorized execution engine processes data in batches for cache efficiency.
- Can query Pandas DataFrames, Parquet files, and CSV files directly
  without loading them into the database first.

Adapted from Data_Science Lesson 29.
"""

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    print("Showing conceptual examples instead.\n")
    duckdb = None

import numpy as np
import pandas as pd
import time


def create_sample_data() -> pd.DataFrame:
    """Create sample sales data as Pandas DataFrame."""
    np.random.seed(42)
    n = 100_000

    dates = pd.date_range("2023-01-01", "2025-12-31", periods=n)
    products = np.random.choice(
        ["Laptop", "Mouse", "Keyboard", "Monitor", "Headset"], n
    )
    regions = np.random.choice(["North", "South", "East", "West"], n)
    quantities = np.random.randint(1, 20, n)
    prices = np.round(np.random.uniform(10, 1500, n), 2)
    customer_ids = np.random.randint(1, 5000, n)

    return pd.DataFrame({
        "date": dates,
        "product": products,
        "region": regions,
        "quantity": quantities,
        "unit_price": prices,
        "customer_id": customer_ids,
    })


# ── Demos ─────────────────────────────────────────────────────────

def demo_basic_queries():
    print("=" * 60)
    print("DUCKDB: SQL ON DATAFRAMES")
    print("=" * 60)

    if duckdb is None:
        print("  (DuckDB not available)")
        return

    df = create_sample_data()

    # Query Pandas DataFrame directly with SQL
    result = duckdb.sql("""
        SELECT
            product,
            region,
            SUM(quantity * unit_price) AS revenue,
            COUNT(*) AS order_count,
            COUNT(DISTINCT customer_id) AS unique_customers
        FROM df
        GROUP BY product, region
        ORDER BY revenue DESC
        LIMIT 10
    """).fetchdf()

    print(f"\n  Top 10 product-region combinations by revenue:")
    print(result.to_string(index=False))


def demo_window_functions():
    print("\n" + "=" * 60)
    print("DUCKDB: WINDOW FUNCTIONS")
    print("=" * 60)

    if duckdb is None:
        print("  (DuckDB not available)")
        return

    df = create_sample_data()

    result = duckdb.sql("""
        WITH monthly_revenue AS (
            SELECT
                DATE_TRUNC('month', date) AS month,
                product,
                SUM(quantity * unit_price) AS revenue
            FROM df
            GROUP BY 1, 2
        )
        SELECT
            month,
            product,
            revenue,
            LAG(revenue) OVER (
                PARTITION BY product ORDER BY month
            ) AS prev_month,
            ROUND(
                (revenue - LAG(revenue) OVER (
                    PARTITION BY product ORDER BY month
                )) / NULLIF(LAG(revenue) OVER (
                    PARTITION BY product ORDER BY month
                ), 0) * 100, 1
            ) AS growth_pct,
            SUM(revenue) OVER (
                PARTITION BY product
                ORDER BY month
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_revenue
        FROM monthly_revenue
        WHERE month >= '2025-01-01' AND product = 'Laptop'
        ORDER BY month
        LIMIT 12
    """).fetchdf()

    print(f"\n  Laptop monthly revenue with window functions:")
    print(result.to_string(index=False))


def demo_cte_analytics():
    print("\n" + "=" * 60)
    print("DUCKDB: CTE-BASED ANALYTICS")
    print("=" * 60)

    if duckdb is None:
        print("  (DuckDB not available)")
        return

    df = create_sample_data()

    # Customer segmentation using CTEs
    result = duckdb.sql("""
        WITH customer_metrics AS (
            SELECT
                customer_id,
                COUNT(*) AS order_count,
                SUM(quantity * unit_price) AS total_spent,
                MIN(date) AS first_order,
                MAX(date) AS last_order,
                COUNT(DISTINCT product) AS products_bought
            FROM df
            GROUP BY customer_id
        ),
        customer_segments AS (
            SELECT *,
                CASE
                    WHEN total_spent > 50000 THEN 'Platinum'
                    WHEN total_spent > 20000 THEN 'Gold'
                    WHEN total_spent > 5000 THEN 'Silver'
                    ELSE 'Bronze'
                END AS segment,
                NTILE(10) OVER (ORDER BY total_spent DESC) AS decile
            FROM customer_metrics
        )
        SELECT
            segment,
            COUNT(*) AS customers,
            ROUND(AVG(total_spent), 2) AS avg_spent,
            ROUND(AVG(order_count), 1) AS avg_orders,
            ROUND(AVG(products_bought), 1) AS avg_products,
            MIN(decile) AS top_decile,
            MAX(decile) AS bottom_decile
        FROM customer_segments
        GROUP BY segment
        ORDER BY avg_spent DESC
    """).fetchdf()

    print(f"\n  Customer segmentation:")
    print(result.to_string(index=False))


def demo_in_memory_tables():
    print("\n" + "=" * 60)
    print("DUCKDB: IN-MEMORY TABLES")
    print("=" * 60)

    if duckdb is None:
        print("  (DuckDB not available)")
        return

    con = duckdb.connect(":memory:")

    # Create tables
    con.execute("""
        CREATE TABLE products AS
        SELECT * FROM (VALUES
            ('Laptop', 'Computing', 1299.99),
            ('Mouse', 'Peripherals', 29.99),
            ('Keyboard', 'Peripherals', 89.99),
            ('Monitor', 'Computing', 499.99),
            ('Headset', 'Audio', 149.99)
        ) AS t(name, category, base_price)
    """)

    # Use persistent table with DataFrame queries
    df = create_sample_data()
    con.register("sales", df)

    result = con.execute("""
        SELECT
            p.category,
            COUNT(*) AS orders,
            ROUND(SUM(s.quantity * s.unit_price), 2) AS revenue,
            ROUND(AVG(s.unit_price), 2) AS avg_price
        FROM sales s
        JOIN products p ON s.product = p.name
        GROUP BY p.category
        ORDER BY revenue DESC
    """).fetchdf()

    print(f"\n  Revenue by product category:")
    print(result.to_string(index=False))

    con.close()


def demo_performance():
    print("\n" + "=" * 60)
    print("DUCKDB vs PANDAS: PERFORMANCE")
    print("=" * 60)

    if duckdb is None:
        print("  (DuckDB not available)")
        return

    df = create_sample_data()

    # Pandas aggregation
    start = time.perf_counter()
    for _ in range(10):
        _ = (
            df.assign(revenue=df["quantity"] * df["unit_price"])
            .groupby(["product", "region"])
            .agg(
                total_revenue=("revenue", "sum"),
                avg_price=("unit_price", "mean"),
                order_count=("quantity", "count"),
            )
            .sort_values("total_revenue", ascending=False)
        )
    pandas_time = (time.perf_counter() - start) / 10

    # DuckDB SQL
    start = time.perf_counter()
    for _ in range(10):
        _ = duckdb.sql("""
            SELECT
                product, region,
                SUM(quantity * unit_price) AS total_revenue,
                AVG(unit_price) AS avg_price,
                COUNT(*) AS order_count
            FROM df
            GROUP BY product, region
            ORDER BY total_revenue DESC
        """).fetchdf()
    duckdb_time = (time.perf_counter() - start) / 10

    print(f"\n  Dataset: {len(df):,} rows")
    print(f"  Pandas: {pandas_time*1000:.2f} ms")
    print(f"  DuckDB: {duckdb_time*1000:.2f} ms")
    print(f"  Speedup: {pandas_time/duckdb_time:.1f}x")

    print(f"""
  When to use DuckDB vs Pandas:
    {'Use Case':<35} {'Best Tool':<15}
    {'-'*35} {'-'*15}
    {'Aggregation on large data':<35} {'DuckDB':<15}
    {'Complex SQL (CTEs, window funcs)':<35} {'DuckDB':<15}
    {'Row-by-row transformations':<35} {'Pandas':<15}
    {'ML pipeline preprocessing':<35} {'Pandas':<15}
    {'Ad-hoc data exploration':<35} {'Either':<15}
    {'Query Parquet/CSV without loading':<35} {'DuckDB':<15}""")


if __name__ == "__main__":
    demo_basic_queries()
    demo_window_functions()
    demo_cte_analytics()
    demo_in_memory_tables()
    demo_performance()
