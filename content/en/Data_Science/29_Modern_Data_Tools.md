# Modern Data Tools: Polars and DuckDB

[Previous: Survival Analysis](./28_Survival_Analysis.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify the performance limitations of Pandas and explain when modern alternatives are warranted
2. Apply Polars' expression API for filtering, aggregation, and column transformations on large datasets
3. Explain how Polars' lazy evaluation and query optimization (predicate pushdown, projection pushdown) improve performance
4. Use DuckDB to run SQL queries directly against DataFrames, Parquet files, and CSV files without a server
5. Describe the role of Apache Arrow as a zero-copy interchange format between Polars, Pandas, and DuckDB
6. Translate common Pandas operations into their Polars equivalents
7. Evaluate when to use Pandas, Polars, or DuckDB based on dataset size, workflow requirements, and team familiarity

---

Pandas is the workhorse of data science, but it was designed in an era of single-core processors and datasets that fit comfortably in memory. As data volumes grow, its single-threaded, eager-evaluation model becomes a bottleneck. Polars and DuckDB represent the next generation of data tools -- leveraging multi-core parallelism, lazy evaluation, and columnar storage to deliver order-of-magnitude speedups without requiring a distributed computing cluster.

---

## 1. Why Modern Tools?

### 1.1 Limitations of Pandas

```python
"""
Pandas Pain Points (for medium-large data):

1. Single-threaded: Can't leverage multi-core CPUs
2. Memory hungry: Copies data frequently, GIL overhead
3. Eager evaluation: Every operation executes immediately
4. Inconsistent API: Multiple ways to do the same thing
5. Slow string operations: Not optimized for string columns

When Pandas is still fine:
  - Small datasets (< 1M rows)
  - Quick exploration / prototyping
  - Deep ecosystem (sklearn, statsmodels, etc.)

When to consider Polars/DuckDB:
  - Dataset > 1M rows
  - Need faster iteration on transforms
  - Doing aggregations on 10M+ row datasets
  - Want SQL-based analysis without a server

Performance comparison (10M rows, group-by aggregation):
  Pandas:  ~5 seconds
  Polars:  ~0.3 seconds (16x faster)
  DuckDB:  ~0.2 seconds (25x faster)
"""
```

---

## 2. Polars

### 2.1 Basics

```python
import polars as pl
import numpy as np

# Create DataFrame
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age": [30, 25, 35, 28, 32],
    "city": ["NYC", "LA", "NYC", "LA", "NYC"],
    "salary": [75000, 65000, 90000, 70000, 85000],
})
print(df)

# Read from file (much faster than Pandas for large files)
# df = pl.read_csv("data.csv")
# df = pl.read_parquet("data.parquet")

# Basic operations
print(df.filter(pl.col("age") > 28))           # Filter
print(df.select(["name", "salary"]))            # Select columns
print(df.sort("salary", descending=True))       # Sort
print(df.with_columns(                          # Add column
    (pl.col("salary") * 1.1).alias("new_salary")
))
```

### 2.2 Expression API

```python
"""
Polars Expression API:
  - pl.col("name")           → reference a column
  - pl.lit(value)            → literal value
  - Chainable transformations → .filter().group_by().agg()
  - No index (unlike Pandas) → cleaner semantics
"""

import polars as pl
import numpy as np

# Generate larger dataset
np.random.seed(42)
N = 1_000_000
df = pl.DataFrame({
    "user_id": np.random.randint(1, 10001, N),
    "product": np.random.choice(["A", "B", "C", "D", "E"], N),
    "amount": np.random.exponential(50, N).round(2),
    "timestamp": pl.Series(
        np.random.choice(
            pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 12, 31), eager=True),
            N,
        )
    ),
})

# Complex expressions
result = (
    df
    .filter(pl.col("amount") > 10)
    .with_columns(
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("amount").log().alias("log_amount"),
        (pl.col("amount") > pl.col("amount").mean()).alias("above_avg"),
    )
    .group_by(["product", "month"])
    .agg(
        pl.col("amount").sum().alias("total_revenue"),
        pl.col("amount").mean().alias("avg_order"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("amount").quantile(0.95).alias("p95_amount"),
    )
    .sort(["product", "month"])
)
print(result.head(10))
```

### 2.3 Lazy Evaluation

```python
"""
Lazy Evaluation: Build a query plan, optimize, then execute.

Eager (Pandas):
  df = read_csv("big.csv")          # Read ALL data into memory
  df = df[df.amount > 100]          # Filter (processes all rows)
  df = df.groupby("product").sum()  # Aggregate

Lazy (Polars):
  q = scan_csv("big.csv")           # Build query plan (no I/O yet)
  q = q.filter(col("amount") > 100) # Add filter to plan
  q = q.group_by("product").agg(sum("amount"))  # Add aggregation
  result = q.collect()               # Execute optimized plan

Optimizations applied automatically:
  - Predicate pushdown: filter before read
  - Projection pushdown: only read needed columns
  - Common subexpression elimination
  - Join optimization
  - Parallel execution
"""

# Lazy query (in production, use scan_csv or scan_parquet)
lazy_df = df.lazy()

query = (
    lazy_df
    .filter(pl.col("amount") > 20)
    .group_by("product")
    .agg(
        pl.col("amount").sum().alias("total"),
        pl.col("amount").mean().alias("avg"),
        pl.len().alias("count"),
    )
    .sort("total", descending=True)
)

# Inspect the optimized query plan
print("Query Plan:")
print(query.explain())

# Execute
result = query.collect()
print("\nResult:")
print(result)

# Streaming mode for datasets larger than RAM
# result = query.collect(streaming=True)
```

### 2.4 Window Functions

```python
# Window functions in Polars
result = df.with_columns(
    # Rank within each product
    pl.col("amount").rank().over("product").alias("rank_in_product"),

    # Running total per user
    pl.col("amount").cum_sum().over("user_id").alias("cumulative_spend"),

    # Difference from product mean
    (pl.col("amount") - pl.col("amount").mean().over("product")).alias("diff_from_avg"),

    # Lead/lag
    pl.col("amount").shift(1).over("user_id").alias("prev_amount"),
)
print(result.head(10))
```

---

## 3. DuckDB

### 3.1 In-Process SQL Engine

```python
"""
DuckDB: In-process OLAP database.
  - No server needed (like SQLite, but for analytics)
  - Columnar storage → fast aggregations
  - Can query Pandas, Polars, Parquet, CSV directly
  - Full SQL support (window functions, CTEs, etc.)
"""

import duckdb

# Query Polars DataFrame directly with SQL!
result = duckdb.sql("""
    SELECT
        product,
        COUNT(*) as order_count,
        ROUND(SUM(amount), 2) as total_revenue,
        ROUND(AVG(amount), 2) as avg_order,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount), 2) as p95
    FROM df
    WHERE amount > 10
    GROUP BY product
    ORDER BY total_revenue DESC
""")
print(result)

# Query Parquet files directly (no loading into memory)
# duckdb.sql("SELECT * FROM 'data/*.parquet' WHERE year = 2024")

# Query CSV files with auto-detection
# duckdb.sql("SELECT * FROM read_csv_auto('data.csv')")
```

### 3.2 Advanced SQL Features

```python
import duckdb

# Window functions
result = duckdb.sql("""
    WITH monthly AS (
        SELECT
            product,
            EXTRACT(MONTH FROM timestamp) as month,
            SUM(amount) as revenue
        FROM df
        GROUP BY product, month
    )
    SELECT
        product,
        month,
        revenue,
        LAG(revenue) OVER (PARTITION BY product ORDER BY month) as prev_month,
        ROUND(100.0 * (revenue - LAG(revenue) OVER (PARTITION BY product ORDER BY month))
              / NULLIF(LAG(revenue) OVER (PARTITION BY product ORDER BY month), 0), 1)
              as growth_pct,
        SUM(revenue) OVER (PARTITION BY product ORDER BY month) as ytd_revenue
    FROM monthly
    ORDER BY product, month
""")
print(result)
```

### 3.3 DuckDB with Persistent Storage

```python
import duckdb

# Create a persistent database
con = duckdb.connect("analytics.duckdb")

# Create table and insert data
con.execute("""
    CREATE TABLE IF NOT EXISTS orders AS
    SELECT * FROM df
""")

# Create indexes for faster queries
con.execute("CREATE INDEX IF NOT EXISTS idx_product ON orders(product)")

# Analytical queries
result = con.execute("""
    SELECT
        product,
        DATE_TRUNC('month', timestamp) as month,
        COUNT(DISTINCT user_id) as unique_users,
        COUNT(*) as orders,
        ROUND(SUM(amount), 2) as revenue
    FROM orders
    GROUP BY product, month
    ORDER BY month, product
""").fetchdf()  # Returns Pandas DataFrame
print(result.head(10))

# Export results
con.execute("COPY (SELECT * FROM orders WHERE amount > 100) TO 'high_value.parquet' (FORMAT PARQUET)")

con.close()

# Clean up
import os
os.remove("analytics.duckdb")
```

---

## 4. Arrow Interoperability

### 4.1 Zero-Copy Data Exchange

```python
"""
Apache Arrow: Common in-memory format for columnar data.

  Polars DataFrame ←→ Arrow Table ←→ DuckDB
       ↕                  ↕
  Pandas DataFrame   Parquet File

Zero-copy: No data duplication when converting between formats.
"""

import polars as pl
import pandas as pd
import pyarrow as pa
import duckdb

# Polars → Arrow → Pandas (zero-copy where possible)
polars_df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

# Polars → Arrow
arrow_table = polars_df.to_arrow()
print(f"Arrow: {arrow_table.schema}")

# Arrow → Pandas
pandas_df = arrow_table.to_pandas()
print(f"Pandas: {pandas_df.dtypes.to_dict()}")

# Pandas → Polars
polars_from_pandas = pl.from_pandas(pandas_df)
print(f"Polars: {polars_from_pandas.dtypes}")

# DuckDB can query all three directly
result = duckdb.sql("SELECT * FROM polars_df WHERE a > 1")
print(result)
```

---

## 5. Migration from Pandas

### 5.1 Common Operations Comparison

```python
"""
Pandas → Polars Translation Guide:

| Operation       | Pandas                         | Polars                               |
|----------------|--------------------------------|--------------------------------------|
| Read CSV       | pd.read_csv("f.csv")           | pl.read_csv("f.csv")                |
| Read Parquet   | pd.read_parquet("f.parquet")   | pl.read_parquet("f.parquet")         |
| Filter         | df[df.age > 30]                | df.filter(pl.col("age") > 30)       |
| Select         | df[["a", "b"]]                 | df.select(["a", "b"])                |
| Add column     | df["new"] = df.a + 1           | df.with_columns((pl.col("a")+1).alias("new")) |
| Group + agg    | df.groupby("g").agg({"a":"sum"})| df.group_by("g").agg(pl.col("a").sum()) |
| Sort           | df.sort_values("a")            | df.sort("a")                         |
| Rename         | df.rename(columns={"a":"b"})   | df.rename({"a": "b"})                |
| Drop nulls     | df.dropna()                    | df.drop_nulls()                      |
| Fill nulls     | df.fillna(0)                   | df.fill_null(0)                      |
| Value counts   | df.a.value_counts()            | df["a"].value_counts()               |
| Join           | pd.merge(df1, df2, on="key")   | df1.join(df2, on="key")              |
| Apply          | df.a.apply(func)               | df.with_columns(pl.col("a").map_elements(func)) |
"""

import pandas as pd
import polars as pl
import numpy as np

# Performance comparison
np.random.seed(42)
N = 5_000_000

# Create same data in both
data = {
    "category": np.random.choice(["A", "B", "C", "D", "E"], N),
    "value": np.random.randn(N),
    "amount": np.random.exponential(100, N),
}

import time

# Pandas
pd_df = pd.DataFrame(data)
t0 = time.time()
pd_result = pd_df.groupby("category").agg(
    total=("amount", "sum"),
    avg_val=("value", "mean"),
    count=("value", "count"),
)
pd_time = time.time() - t0

# Polars
pl_df = pl.DataFrame(data)
t0 = time.time()
pl_result = pl_df.group_by("category").agg(
    pl.col("amount").sum().alias("total"),
    pl.col("value").mean().alias("avg_val"),
    pl.len().alias("count"),
)
pl_time = time.time() - t0

# DuckDB
import duckdb
t0 = time.time()
db_result = duckdb.sql("""
    SELECT category,
           SUM(amount) as total,
           AVG(value) as avg_val,
           COUNT(*) as count
    FROM pl_df
    GROUP BY category
""")
db_time = time.time() - t0

print(f"Group-by aggregation on {N:,} rows:")
print(f"  Pandas:  {pd_time:.3f}s")
print(f"  Polars:  {pl_time:.3f}s  ({pd_time/pl_time:.1f}x faster)")
print(f"  DuckDB:  {db_time:.3f}s  ({pd_time/db_time:.1f}x faster)")
```

---

## 6. Practical Patterns

### 6.1 Large File Processing

```python
"""
Processing files larger than RAM:

Polars Streaming:
  pl.scan_parquet("big_file.parquet")
    .filter(...)
    .group_by(...)
    .agg(...)
    .collect(streaming=True)

DuckDB:
  duckdb.sql("SELECT ... FROM 'big_file.parquet' WHERE ...")
  → Processes in chunks, never loads full file
"""

# Polars: Lazy scan of multiple files
# query = (
#     pl.scan_parquet("data/year=*/*.parquet")
#     .filter(pl.col("amount") > 100)
#     .group_by("product")
#     .agg(pl.col("amount").sum())
#     .collect(streaming=True)
# )

# DuckDB: Query partitioned Parquet directly
# duckdb.sql("""
#     SELECT product, SUM(amount)
#     FROM 'data/year=*/month=*/*.parquet'
#     WHERE year = 2024 AND amount > 100
#     GROUP BY product
# """)
```

### 6.2 When to Use What

```python
"""
Decision Guide:

┌─────────────────────────────────────────────────┐
│                Dataset Size                     │
├──────────┬──────────────┬───────────────────────┤
│ < 1M rows│ 1M - 100M    │ > 100M rows           │
│          │              │                       │
│ Pandas   │ Polars or    │ Polars (streaming)    │
│ (fine)   │ DuckDB       │ DuckDB                │
│          │              │ Spark (distributed)   │
└──────────┴──────────────┴───────────────────────┘

Use Pandas when:
  - Small data + rapid prototyping
  - Need sklearn/statsmodels integration
  - Team is familiar with Pandas

Use Polars when:
  - Need speed for transforms/aggregations
  - Building data pipelines (lazy evaluation)
  - Prefer method-chaining API

Use DuckDB when:
  - Prefer SQL over Python API
  - Need to query files directly (Parquet/CSV)
  - Want persistent analytical database
  - Mixed SQL + Python workflow
"""
```

---

## 7. Practice Problems

### Exercise 1: Polars Pipeline

```python
"""
Build a data pipeline with Polars:
1. Generate 10M row sales dataset (product, region, date, amount, quantity)
2. Using lazy evaluation, compute:
   a. Monthly revenue by product and region
   b. Month-over-month growth rate per product
   c. Running 3-month average per region
3. Compare execution time: eager vs lazy
4. Export results to Parquet with partitioning
"""
```

### Exercise 2: DuckDB Analytics

```python
"""
Perform analytics using DuckDB SQL:
1. Create a DuckDB database with 3 tables (orders, customers, products)
2. Write queries for:
   a. Top 10 customers by lifetime value
   b. Monthly cohort retention analysis
   c. Product affinity (frequently bought together)
3. Compare DuckDB query time vs equivalent Pandas operations
4. Export results to Parquet and query across files
"""
```

---

## 8. Summary

### Key Takeaways

| Tool | Type | Best For |
|------|------|----------|
| **Polars** | DataFrame library | Fast transforms, lazy evaluation, pipelines |
| **DuckDB** | In-process SQL | SQL analytics, querying files directly |
| **Arrow** | Memory format | Zero-copy interop between tools |
| **Pandas** | DataFrame library | Prototyping, small data, ecosystem |

### Best Practices

1. **Use lazy evaluation** in Polars — let the optimizer plan your query
2. **Query files directly** with DuckDB — no need to load into memory
3. **Arrow for interchange** — convert between Polars/Pandas/DuckDB without copies
4. **Profile first** — don't switch from Pandas unless you have a performance problem
5. **Use Parquet** — columnar format benefits all three tools

### Navigation

- **Previous**: L28 — Survival Analysis
- Return to **L01** (NumPy Fundamentals) for foundational data tools
