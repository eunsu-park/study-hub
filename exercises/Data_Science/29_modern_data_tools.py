"""
Exercises for Lesson 29: Modern Data Tools
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Columnar vs Row-Based Storage ===
# Problem: Demonstrate why columnar storage is faster for analytical
#          queries by comparing column-oriented vs row-oriented access.
def exercise_1():
    """Solution for understanding columnar vs row-based data layout.

    Row-based storage (Pandas/CSV):
    [row0_col0, row0_col1, ...], [row1_col0, row1_col1, ...], ...
    Good for: fetching entire rows, transactional workloads.

    Columnar storage (Parquet/Arrow/Polars):
    [col0_row0, col0_row1, ...], [col1_row0, col1_row1, ...], ...
    Good for: aggregations on specific columns, analytical queries.

    Columnar is faster for analytics because:
    1. Only reads needed columns (projection pushdown)
    2. Better compression (similar values together)
    3. CPU cache-friendly for column scans
    """
    import time

    np.random.seed(42)

    n_rows = 2_000_000
    n_cols = 10

    # Row-oriented storage (list of tuples)
    print("Columnar vs Row-Based Access Patterns:")
    print(f"  Dataset: {n_rows:,} rows x {n_cols} columns")

    # Create data as columns (columnar layout)
    columns = {
        f"col_{i}": np.random.randn(n_rows) for i in range(n_cols)
    }

    # Simulate row-oriented access: sum one column
    # In row-oriented, accessing one column touches all rows of all columns
    col_data = columns["col_0"]

    t0 = time.time()
    # Column-oriented aggregation: contiguous memory access
    col_sum = np.sum(col_data)
    col_time = time.time() - t0

    # Row-oriented simulation: interleaved memory access
    # Store data interleaved (row-major)
    interleaved = np.zeros((n_rows, n_cols))
    for i, key in enumerate(columns):
        interleaved[:, i] = columns[key]

    t0 = time.time()
    row_sum = np.sum(interleaved[:, 0])
    row_time = time.time() - t0

    print(f"\n  Single column aggregation (sum):")
    print(f"    Columnar access: {col_time*1000:.2f} ms")
    print(f"    Row-based access: {row_time*1000:.2f} ms")
    if col_time > 0:
        print(f"    Speedup: {row_time/col_time:.1f}x")
    print(f"    Both produce same result: {np.isclose(col_sum, row_sum)}")

    # Multi-column aggregation
    t0 = time.time()
    group_col = (columns["col_0"] * 10).astype(int)  # fake grouping
    target_col = columns["col_1"]
    # Group-by sum using numpy
    unique_groups = np.unique(group_col)
    group_sums = np.array([np.sum(target_col[group_col == g])
                           for g in unique_groups[:20]])
    col_group_time = time.time() - t0

    t0 = time.time()
    group_col_r = (interleaved[:, 0] * 10).astype(int)
    target_col_r = interleaved[:, 1]
    unique_groups_r = np.unique(group_col_r)
    group_sums_r = np.array([np.sum(target_col_r[group_col_r == g])
                             for g in unique_groups_r[:20]])
    row_group_time = time.time() - t0

    print(f"\n  Group-by aggregation (20 groups):")
    print(f"    Columnar: {col_group_time*1000:.2f} ms")
    print(f"    Row-based: {row_group_time*1000:.2f} ms")

    print(f"\n  Key takeaway: Columnar layout is optimal for analytical")
    print(f"  queries that touch few columns but many rows.")
    print(f"  This is why Parquet, Arrow, Polars, and DuckDB are fast.")


# === Exercise 2: Lazy vs Eager Evaluation ===
# Problem: Demonstrate the performance difference between eager
#          and lazy evaluation for a multi-step data pipeline.
def exercise_2():
    """Solution for comparing eager vs lazy evaluation.

    Eager evaluation (Pandas-style):
    - Each operation executes immediately and materializes results
    - Intermediate results consume memory
    - No cross-operation optimization

    Lazy evaluation (Polars/Spark-style):
    - Operations build a query plan without executing
    - Optimizer can fuse, reorder, and prune operations
    - Execution happens only when results are requested (.collect())

    Key optimizations in lazy evaluation:
    1. Predicate pushdown: filter before reading/processing
    2. Projection pushdown: only process needed columns
    3. Operation fusion: combine multiple operations into one pass
    """
    import time

    np.random.seed(42)

    n = 3_000_000

    # Simulated dataset
    categories = np.random.choice(['A', 'B', 'C', 'D', 'E'], n)
    amounts = np.random.exponential(50, n)
    quantities = np.random.randint(1, 20, n)
    flags = np.random.binomial(1, 0.3, n)

    print("Lazy vs Eager Evaluation:")
    print(f"  Dataset: {n:,} rows")

    # Eager approach: execute each step immediately
    t0 = time.time()

    # Step 1: Filter
    mask = amounts > 10
    e_amounts = amounts[mask]
    e_categories = categories[mask]
    e_quantities = quantities[mask]
    e_flags = flags[mask]

    # Step 2: Add derived column
    e_revenue = e_amounts * e_quantities

    # Step 3: Filter again
    mask2 = e_flags == 1
    e_amounts2 = e_amounts[mask2]
    e_categories2 = e_categories[mask2]
    e_revenue2 = e_revenue[mask2]

    # Step 4: Group-by aggregation
    unique_cats = np.unique(e_categories2)
    eager_result = {}
    for cat in unique_cats:
        cat_mask = e_categories2 == cat
        eager_result[cat] = {
            'total_revenue': np.sum(e_revenue2[cat_mask]),
            'avg_amount': np.mean(e_amounts2[cat_mask]),
            'count': np.sum(cat_mask),
        }

    eager_time = time.time() - t0

    # Lazy approach: fused operations (simulate optimizer)
    t0 = time.time()

    # Optimizer combines: filter1 AND filter2 into single pass,
    # only computes needed columns, and fuses with aggregation
    combined_mask = (amounts > 10) & (flags == 1)
    l_amounts = amounts[combined_mask]
    l_categories = categories[combined_mask]
    l_quantities = quantities[combined_mask]
    l_revenue = l_amounts * l_quantities

    unique_cats_l = np.unique(l_categories)
    lazy_result = {}
    for cat in unique_cats_l:
        cat_mask = l_categories == cat
        lazy_result[cat] = {
            'total_revenue': np.sum(l_revenue[cat_mask]),
            'avg_amount': np.mean(l_amounts[cat_mask]),
            'count': np.sum(cat_mask),
        }

    lazy_time = time.time() - t0

    print(f"\n  Eager (step-by-step): {eager_time*1000:.2f} ms")
    print(f"  Lazy (optimized):     {lazy_time*1000:.2f} ms")
    if lazy_time > 0:
        print(f"  Speedup: {eager_time/lazy_time:.2f}x")

    # Verify results match
    results_match = all(
        np.isclose(eager_result[c]['total_revenue'],
                   lazy_result[c]['total_revenue'], rtol=1e-10)
        for c in unique_cats
    )
    print(f"  Results match: {results_match}")

    print(f"\n  Optimizations: predicate pushdown, projection pushdown, op fusion")

    # Show intermediate materialization cost
    print(f"\n  Memory impact:")
    n_after_filter1 = np.sum(amounts > 10)
    n_after_filter2 = np.sum(combined_mask)
    print(f"    After filter 1: {n_after_filter1:,} rows (intermediate)")
    print(f"    After filter 2: {n_after_filter2:,} rows (final)")
    print(f"    Eager materializes {n_after_filter1:,} rows unnecessarily")


# === Exercise 3: SQL-Style Analytics with NumPy ===
# Problem: Implement common SQL analytical patterns (window functions,
#          CTEs, self-joins) using pure numpy, demonstrating the
#          concepts behind DuckDB's query processing.
def exercise_3():
    """Solution for SQL-style analytics using numpy.

    This exercise implements core SQL analytics concepts:
    1. GROUP BY with multiple aggregations
    2. Window functions (ROW_NUMBER, RANK, cumulative sum)
    3. HAVING clause (filter after aggregation)

    These are the patterns that DuckDB optimizes heavily using
    vectorized columnar execution.
    """
    np.random.seed(42)

    n = 10000
    user_ids = np.random.randint(1, 201, n)
    products = np.random.choice(['Widget', 'Gadget', 'Doohickey',
                                  'Thingamajig', 'Whatchamacallit'], n)
    amounts = np.random.exponential(75, n).round(2)
    months = np.random.randint(1, 13, n)

    print("SQL-Style Analytics with NumPy:")
    print(f"  Dataset: {n:,} orders, {len(np.unique(user_ids))} users, "
          f"{len(np.unique(products))} products")

    # Query 1: GROUP BY with aggregations
    # SQL: SELECT product, COUNT(*), SUM(amount), AVG(amount)
    #      FROM orders GROUP BY product ORDER BY SUM(amount) DESC
    print(f"\n  Query 1: Revenue by Product")
    print(f"    {'Product':<20} {'Count':>8} {'Total':>12} {'Average':>10}")
    print(f"    {'-' * 54}")

    unique_products = np.unique(products)
    product_stats = []
    for prod in unique_products:
        mask = products == prod
        count = np.sum(mask)
        total = np.sum(amounts[mask])
        avg = np.mean(amounts[mask])
        product_stats.append((prod, count, total, avg))

    # Sort by total descending
    product_stats.sort(key=lambda x: -x[2])
    for prod, count, total, avg in product_stats:
        print(f"    {prod:<20} {count:>8,} {total:>12,.2f} {avg:>10.2f}")

    # Query 2: Monthly trend with cumulative sum
    # SQL: SELECT month, SUM(amount), SUM(SUM(amount)) OVER (ORDER BY month)
    print(f"\n  Query 2: Monthly Revenue with YTD Cumulative")
    print(f"    {'Month':>6} {'Monthly':>12} {'YTD':>14} {'MoM Growth':>12}")
    print(f"    {'-' * 48}")

    monthly_rev = np.zeros(12)
    for m in range(1, 13):
        monthly_rev[m - 1] = np.sum(amounts[months == m])

    ytd = np.cumsum(monthly_rev)
    for m in range(12):
        if m == 0:
            growth = "N/A"
        else:
            pct = (monthly_rev[m] - monthly_rev[m - 1]) / monthly_rev[m - 1] * 100
            growth = f"{pct:>+.1f}%"
        print(f"    {m+1:>6} {monthly_rev[m]:>12,.2f} {ytd[m]:>14,.2f} "
              f"{growth:>12}")

    # Query 3: HAVING clause
    print(f"\n  Query 3: Products with Above-Average Order Count")
    counts = np.array([np.sum(products == p) for p in unique_products])
    avg_count = np.mean(counts)
    print(f"    Average: {avg_count:.0f}")
    for prod, count in zip(unique_products, counts):
        if count > avg_count:
            print(f"    {prod}: {count:,} orders")


# === Exercise 4: Data Pipeline Design Patterns ===
# Problem: Implement a multi-stage data pipeline with validation,
#          transformation, and quality checks at each stage.
def exercise_4():
    """Solution for data pipeline with validation and quality checks.

    A production data pipeline should include:
    1. Input validation (schema, types, ranges)
    2. Deduplication
    3. Null handling
    4. Transformation
    5. Output validation (row counts, aggregate checks)

    This mirrors what tools like Polars, DuckDB, and dbt provide
    for data quality assurance.
    """
    np.random.seed(42)

    n_raw = 5000
    raw_ids = np.random.randint(1000, 9999, n_raw)
    raw_amounts = np.random.exponential(100, n_raw)
    valid_cats = ['A', 'B', 'C', 'D']
    raw_categories = np.random.choice(valid_cats + ['ERROR', ''], n_raw,
                                       p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05])

    # Inject nulls (represented as -1)
    null_idx = np.random.choice(n_raw, 150, replace=False)
    raw_amounts[null_idx] = -1

    # Inject duplicates
    dup_idx = np.random.choice(n_raw, 200, replace=False)
    raw_ids = np.concatenate([raw_ids, raw_ids[dup_idx]])
    raw_amounts = np.concatenate([raw_amounts, raw_amounts[dup_idx]])
    raw_categories = np.concatenate([raw_categories, raw_categories[dup_idx]])
    n_total = len(raw_ids)

    print("Data Pipeline with Quality Checks:")
    print(f"  Raw records: {n_total:,}")

    # Stage 1: Validation
    cat_ok = np.array([c in valid_cats for c in raw_categories])
    amt_ok = raw_amounts > 0
    valid = cat_ok & amt_ok
    print(f"\n  Stage 1 - Validation:")
    print(f"    Invalid categories: {np.sum(~cat_ok)}")
    print(f"    Invalid amounts: {np.sum(~amt_ok)}")
    print(f"    Passing: {np.sum(valid):,}/{n_total:,}")

    s1_ids = raw_ids[valid]
    s1_amounts = raw_amounts[valid]
    s1_categories = raw_categories[valid]

    # Stage 2: Deduplication
    seen = set()
    keep = np.ones(len(s1_ids), dtype=bool)
    for i in range(len(s1_ids)):
        if int(s1_ids[i]) in seen:
            keep[i] = False
        else:
            seen.add(int(s1_ids[i]))

    n_dupes = np.sum(~keep)
    s2_amounts = s1_amounts[keep]
    s2_categories = s1_categories[keep]
    print(f"\n  Stage 2 - Deduplication: removed {n_dupes}, "
          f"remaining {len(s2_amounts):,}")

    # Stage 3: Outlier capping
    q1, q3 = np.percentile(s2_amounts, [25, 75])
    cap = q3 + 3 * (q3 - q1)
    n_outliers = np.sum(s2_amounts > cap)
    s3_amounts = s2_amounts.clip(max=cap)
    print(f"\n  Stage 3 - Outlier capping: {n_outliers} capped at {cap:.2f}")

    # Stage 4: Aggregation
    print(f"\n  Stage 4 - Aggregation:")
    print(f"    {'Cat':>6} {'Count':>8} {'Total':>12} {'Mean':>10}")
    print(f"    {'-' * 40}")
    for cat in sorted(valid_cats):
        m = s2_categories == cat
        print(f"    {cat:>6} {np.sum(m):>8,} {np.sum(s3_amounts[m]):>12,.2f} "
              f"{np.mean(s3_amounts[m]):>10.2f}")

    # Stage 5: Validation checks
    print(f"\n  Stage 5 - Output Validation:")
    checks = [
        ("Records > 0", len(s2_amounts) > 0),
        ("No negatives", np.all(s3_amounts >= 0)),
        ("Valid categories", all(c in valid_cats for c in s2_categories)),
    ]
    for name, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    print(f"\n  Summary: {n_total:,} -> {len(s2_amounts):,} records "
          f"({(n_total - len(s2_amounts))/n_total*100:.1f}% dropped)")


if __name__ == "__main__":
    print("=== Exercise 1: Columnar vs Row-Based Storage ===")
    exercise_1()
    print("\n=== Exercise 2: Lazy vs Eager Evaluation ===")
    exercise_2()
    print("\n=== Exercise 3: SQL-Style Analytics with NumPy ===")
    exercise_3()
    print("\n=== Exercise 4: Data Pipeline Design Patterns ===")
    exercise_4()
    print("\nAll exercises completed!")
