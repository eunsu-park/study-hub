# 17. Window Functions & Analytics Queries

**Previous**: [Replication and High Availability](./16_Replication_HA.md) | **Next**: [Table Partitioning](./18_Table_Partitioning.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how window functions differ from regular aggregate functions and why they preserve individual rows
2. Construct OVER clauses with PARTITION BY, ORDER BY, and frame specifications
3. Apply ranking functions (ROW_NUMBER, RANK, DENSE_RANK, NTILE) to produce ordered rankings within partitions
4. Use analytical functions (LAG, LEAD, FIRST_VALUE, LAST_VALUE) to compare rows across a result set
5. Calculate running totals, moving averages, and cumulative statistics using aggregate window functions
6. Distinguish between ROWS, RANGE, and GROUPS frame types and select the correct one for a given analysis
7. Solve practical analytical problems including growth rates, outlier detection, sessionization, and gap filling

---

Business intelligence and data analysis often require calculations that look across multiple rows -- rankings, running totals, period-over-period comparisons, and moving averages. Traditional GROUP BY collapses rows into summaries, losing the individual detail you need. Window functions solve this elegantly: they perform calculations across a set of related rows while keeping every row in the result. Once you master window functions, complex analytical queries that would otherwise require multiple subqueries or procedural code become concise single-pass SQL statements.

## Table of Contents
1. [Window Function Basics](#1-window-function-basics)
2. [Ranking Functions](#2-ranking-functions)
3. [Analytical Functions](#3-analytical-functions)
4. [Aggregate Window Functions](#4-aggregate-window-functions)
5. [Frame Details](#5-frame-details)
6. [Practical Use Patterns](#6-practical-use-patterns)
7. [Practice Problems](#7-practice-problems)

---

## 1. Window Function Basics

### 1.1 What are Window Functions?

```
┌─────────────────────────────────────────────────────────────────┐
│                 Regular Aggregate vs Window Functions            │
│                                                                 │
│   Regular Aggregate (GROUP BY)     Window Function (OVER)      │
│   ┌───────────────┐            ┌───────────────┐               │
│   │ A | B | SUM   │            │ A | B | val | SUM             │
│   ├───────────────┤            ├───────────────────┤           │
│   │ X |   | 150   │            │ X | 1 | 50  | 150 │           │
│   │ Y |   | 120   │            │ X | 2 | 100 | 150 │           │
│   └───────────────┘            │ Y | 3 | 70  | 120 │           │
│   (rows collapsed to groups)   │ Y | 4 | 50  | 120 │           │
│                                └───────────────────┘           │
│                                (all rows kept + aggregate)      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Basic Syntax

```sql
-- Window function basic structure
function_name() OVER (
    [PARTITION BY column]    -- Group division (optional)
    [ORDER BY column]        -- Ordering (optional)
    [frame_clause]           -- Range specification (optional)
)

-- Example
SELECT
    department,
    employee_name,
    salary,
    SUM(salary) OVER (PARTITION BY department) AS dept_total,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg
FROM employees;
```

### 1.3 Sample Data

```sql
-- Create test table
CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    salesperson VARCHAR(50),
    region VARCHAR(20),
    sale_date DATE,
    amount NUMERIC(10,2)
);

INSERT INTO sales (salesperson, region, sale_date, amount) VALUES
    ('Alice', 'East', '2024-01-15', 1000),
    ('Alice', 'East', '2024-01-20', 1500),
    ('Alice', 'East', '2024-02-10', 2000),
    ('Bob', 'East', '2024-01-18', 800),
    ('Bob', 'East', '2024-02-15', 1200),
    ('Charlie', 'West', '2024-01-10', 900),
    ('Charlie', 'West', '2024-01-25', 1100),
    ('Charlie', 'West', '2024-02-20', 1300),
    ('Diana', 'West', '2024-01-30', 700),
    ('Diana', 'West', '2024-02-05', 1600);
```

---

## 2. Ranking Functions

### 2.1 ROW_NUMBER, RANK, DENSE_RANK Comparison

```sql
-- Compare ranking functions
SELECT
    salesperson,
    amount,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS row_num,
    RANK() OVER (ORDER BY amount DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY amount DESC) AS dense_rank
FROM sales;
```

```
Example result (tie handling difference):
┌────────────┬────────┬─────────┬──────┬────────────┐
│ salesperson│ amount │ row_num │ rank │ dense_rank │
├────────────┼────────┼─────────┼──────┼────────────┤
│ Alice      │ 2000   │ 1       │ 1    │ 1          │
│ Diana      │ 1600   │ 2       │ 2    │ 2          │
│ Alice      │ 1500   │ 3       │ 3    │ 3          │
│ Charlie    │ 1300   │ 4       │ 4    │ 4          │
│ Bob        │ 1200   │ 5       │ 5    │ 5          │
│ Charlie    │ 1100   │ 6       │ 6    │ 6          │
│ Alice      │ 1000   │ 7       │ 7    │ 7          │  -- no ties
│ Charlie    │  900   │ 8       │ 8    │ 8          │
│ Bob        │  800   │ 9       │ 9    │ 9          │
│ Diana      │  700   │ 10      │ 10   │ 10         │
└────────────┴────────┴─────────┴──────┴────────────┘

When there are ties:
│ A          │ 1000   │ 1       │ 1    │ 1          │
│ B          │ 1000   │ 2       │ 1    │ 1          │  -- tie!
│ C          │  900   │ 3       │ 3    │ 2          │
                      (sequential) (skip) (sequential)
```

### 2.2 Ranking with PARTITION BY

```sql
-- Regional sales ranking
SELECT
    region,
    salesperson,
    amount,
    RANK() OVER (
        PARTITION BY region
        ORDER BY amount DESC
    ) AS region_rank
FROM sales;
```

```
Result:
┌────────┬────────────┬────────┬─────────────┐
│ region │ salesperson│ amount │ region_rank │
├────────┼────────────┼────────┼─────────────┤
│ East   │ Alice      │ 2000   │ 1           │
│ East   │ Alice      │ 1500   │ 2           │
│ East   │ Bob        │ 1200   │ 3           │
│ East   │ Alice      │ 1000   │ 4           │
│ East   │ Bob        │  800   │ 5           │
│ West   │ Diana      │ 1600   │ 1           │  ← partition reset
│ West   │ Charlie    │ 1300   │ 2           │
│ West   │ Charlie    │ 1100   │ 3           │
│ West   │ Charlie    │  900   │ 4           │
│ West   │ Diana      │  700   │ 5           │
└────────┴────────────┴────────┴─────────────┘
```

### 2.3 NTILE - Assign Quantiles

```sql
-- Divide into 4 quartiles
SELECT
    salesperson,
    amount,
    NTILE(4) OVER (ORDER BY amount DESC) AS quartile
FROM sales;

-- Use case: identify top 25% customers
SELECT *
FROM (
    SELECT
        customer_id,
        total_purchase,
        NTILE(4) OVER (ORDER BY total_purchase DESC) AS quartile
    FROM customer_summary
) sub
WHERE quartile = 1;  -- top 25%
```

### 2.4 Top-N Queries

```sql
-- Extract top 2 sales per region
SELECT *
FROM (
    SELECT
        region,
        salesperson,
        sale_date,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY region
            ORDER BY amount DESC
        ) AS rn
    FROM sales
) ranked
WHERE rn <= 2;
```

---

## 3. Analytical Functions

### 3.1 LAG and LEAD

```sql
-- LAG: reference previous row value
-- LEAD: reference next row value
SELECT
    salesperson,
    sale_date,
    amount,
    LAG(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
    ) AS prev_amount,
    LEAD(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
    ) AS next_amount
FROM sales
ORDER BY salesperson, sale_date;
```

```
Result:
┌────────────┬────────────┬────────┬─────────────┬─────────────┐
│ salesperson│ sale_date  │ amount │ prev_amount │ next_amount │
├────────────┼────────────┼────────┼─────────────┼─────────────┤
│ Alice      │ 2024-01-15 │ 1000   │ NULL        │ 1500        │
│ Alice      │ 2024-01-20 │ 1500   │ 1000        │ 2000        │
│ Alice      │ 2024-02-10 │ 2000   │ 1500        │ NULL        │
│ Bob        │ 2024-01-18 │  800   │ NULL        │ 1200        │
│ Bob        │ 2024-02-15 │ 1200   │  800        │ NULL        │
└────────────┴────────────┴────────┴─────────────┴─────────────┘
```

### 3.2 Calculate Growth Rate

```sql
-- Month-over-month growth rate
SELECT
    salesperson,
    sale_date,
    amount,
    LAG(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
    ) AS prev_amount,
    ROUND(
        (amount - LAG(amount) OVER (
            PARTITION BY salesperson ORDER BY sale_date
        )) * 100.0 /
        NULLIF(LAG(amount) OVER (
            PARTITION BY salesperson ORDER BY sale_date
        ), 0),
        2
    ) AS growth_pct
FROM sales
ORDER BY salesperson, sale_date;
```

### 3.3 FIRST_VALUE, LAST_VALUE, NTH_VALUE

```sql
-- First/last value within partition
SELECT
    salesperson,
    sale_date,
    amount,
    FIRST_VALUE(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_sale,
    LAST_VALUE(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_sale,
    NTH_VALUE(amount, 2) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS second_sale
FROM sales;
```

---

## 4. Aggregate Window Functions

### 4.1 SUM, AVG, COUNT

```sql
-- Window aggregate functions
SELECT
    salesperson,
    sale_date,
    amount,
    -- Total by salesperson
    SUM(amount) OVER (PARTITION BY salesperson) AS person_total,
    -- Average by salesperson
    AVG(amount) OVER (PARTITION BY salesperson) AS person_avg,
    -- Percentage of total
    ROUND(amount * 100.0 / SUM(amount) OVER (), 2) AS pct_of_total
FROM sales
ORDER BY salesperson, sale_date;
```

### 4.2 Running Total

```sql
-- Cumulative sum
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM sales
ORDER BY sale_date;

-- Cumulative sum by salesperson
SELECT
    salesperson,
    sale_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
    ) AS cumulative_sales
FROM sales
ORDER BY salesperson, sale_date;
```

### 4.3 Moving Average

```sql
-- Moving average of last 3 records
SELECT
    sale_date,
    amount,
    AVG(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3
FROM sales
ORDER BY sale_date;

-- Centered moving average (including 1 before and after)
SELECT
    sale_date,
    amount,
    AVG(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS centered_avg
FROM sales
ORDER BY sale_date;
```

---

## 5. Frame Details

### 5.1 Frame Syntax

```
ROWS | RANGE | GROUPS BETWEEN start_point AND end_point

start_point/end_point:
- UNBOUNDED PRECEDING  -- partition start
- n PRECEDING          -- n rows before
- CURRENT ROW          -- current row
- n FOLLOWING          -- n rows after
- UNBOUNDED FOLLOWING  -- partition end
```

### 5.2 ROWS vs RANGE

```sql
-- Test with tie data to see difference
CREATE TABLE test_frame (
    id INT,
    val INT
);
INSERT INTO test_frame VALUES (1, 100), (2, 100), (3, 200), (4, 200), (5, 300);

-- ROWS: physical row-based
SELECT
    id, val,
    SUM(val) OVER (ORDER BY val ROWS UNBOUNDED PRECEDING) AS rows_sum
FROM test_frame;

-- RANGE: logical value-based (same values grouped)
SELECT
    id, val,
    SUM(val) OVER (ORDER BY val RANGE UNBOUNDED PRECEDING) AS range_sum
FROM test_frame;
```

```
Result comparison:
ROWS:                          RANGE:
┌────┬─────┬──────────┐       ┌────┬─────┬───────────┐
│ id │ val │ rows_sum │       │ id │ val │ range_sum │
├────┼─────┼──────────┤       ├────┼─────┼───────────┤
│ 1  │ 100 │ 100      │       │ 1  │ 100 │ 200       │ ← two 100s
│ 2  │ 100 │ 200      │       │ 2  │ 100 │ 200       │ ← same
│ 3  │ 200 │ 400      │       │ 3  │ 200 │ 600       │ ← two 200s
│ 4  │ 200 │ 600      │       │ 4  │ 200 │ 600       │ ← same
│ 5  │ 300 │ 900      │       │ 5  │ 300 │ 900       │
└────┴─────┴──────────┘       └────┴─────┴───────────┘
```

### 5.3 GROUPS (PostgreSQL 11+)

```sql
-- GROUPS: same ORDER BY values as one group
SELECT
    id, val,
    SUM(val) OVER (
        ORDER BY val
        GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW
    ) AS groups_sum
FROM test_frame;
```

### 5.4 EXCLUDE Clause (PostgreSQL 11+)

```sql
-- Exclude specific rows from frame
SELECT
    id, val,
    SUM(val) OVER (
        ORDER BY val
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        EXCLUDE CURRENT ROW  -- exclude current row
    ) AS sum_excluding_current
FROM test_frame;

-- EXCLUDE options:
-- EXCLUDE NO OTHERS (default)
-- EXCLUDE CURRENT ROW
-- EXCLUDE GROUP (current row and same values)
-- EXCLUDE TIES (same values except current row)
```

---

## 6. Practical Use Patterns

### 6.1 Daily Cumulative Sales and Target Achievement

```sql
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
    ) AS cumulative_sales,
    ROUND(
        SUM(amount) OVER (ORDER BY sale_date) * 100.0 / 10000,
        2
    ) AS target_pct  -- target: 10,000
FROM sales
ORDER BY sale_date;
```

### 6.2 Outlier Detection

```sql
-- Data beyond average ± 2 standard deviations
WITH stats AS (
    SELECT
        salesperson,
        amount,
        AVG(amount) OVER (PARTITION BY salesperson) AS avg_amount,
        STDDEV(amount) OVER (PARTITION BY salesperson) AS stddev_amount
    FROM sales
)
SELECT *
FROM stats
WHERE amount > avg_amount + 2 * stddev_amount
   OR amount < avg_amount - 2 * stddev_amount;
```

### 6.3 Consecutive Record Analysis

```sql
-- Calculate consecutive sales days
WITH daily_sales AS (
    SELECT
        salesperson,
        sale_date,
        sale_date - (ROW_NUMBER() OVER (
            PARTITION BY salesperson
            ORDER BY sale_date
        ))::int AS grp
    FROM sales
)
SELECT
    salesperson,
    MIN(sale_date) AS streak_start,
    MAX(sale_date) AS streak_end,
    COUNT(*) AS streak_length
FROM daily_sales
GROUP BY salesperson, grp
ORDER BY salesperson, streak_start;
```

### 6.4 Row/Column Comparison Without Pivot

```sql
-- Current vs previous month vs same month last year
SELECT
    salesperson,
    DATE_TRUNC('month', sale_date) AS month,
    SUM(amount) AS monthly_total,
    LAG(SUM(amount)) OVER (
        PARTITION BY salesperson
        ORDER BY DATE_TRUNC('month', sale_date)
    ) AS prev_month,
    LAG(SUM(amount), 12) OVER (
        PARTITION BY salesperson
        ORDER BY DATE_TRUNC('month', sale_date)
    ) AS same_month_last_year
FROM sales
GROUP BY salesperson, DATE_TRUNC('month', sale_date)
ORDER BY salesperson, month;
```

### 6.5 Sessionization

```sql
-- New session if gap exceeds 30 minutes
WITH events AS (
    SELECT
        user_id,
        event_time,
        LAG(event_time) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS prev_event_time
    FROM user_events
),
session_flags AS (
    SELECT
        user_id,
        event_time,
        CASE
            WHEN prev_event_time IS NULL THEN 1
            WHEN event_time - prev_event_time > INTERVAL '30 minutes' THEN 1
            ELSE 0
        END AS is_new_session
    FROM events
)
SELECT
    user_id,
    event_time,
    SUM(is_new_session) OVER (
        PARTITION BY user_id
        ORDER BY event_time
    ) AS session_id
FROM session_flags;
```

### 6.6 Gap Filling

```sql
-- Generate date series and LEFT JOIN
WITH date_series AS (
    SELECT generate_series(
        '2024-01-01'::date,
        '2024-01-31'::date,
        '1 day'::interval
    )::date AS date
),
daily_totals AS (
    SELECT sale_date, SUM(amount) AS total
    FROM sales
    GROUP BY sale_date
)
SELECT
    ds.date,
    COALESCE(dt.total, 0) AS daily_total,
    SUM(COALESCE(dt.total, 0)) OVER (ORDER BY ds.date) AS running_total
FROM date_series ds
LEFT JOIN daily_totals dt ON ds.date = dt.sale_date
ORDER BY ds.date;
```

### 6.7 Percentile Calculation

```sql
-- Calculate percentiles
SELECT
    salesperson,
    amount,
    PERCENT_RANK() OVER (ORDER BY amount) AS pct_rank,
    CUME_DIST() OVER (ORDER BY amount) AS cume_dist,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) OVER () AS median
FROM sales;

-- Median by group
SELECT DISTINCT
    region,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)
        OVER (PARTITION BY region) AS median_by_region
FROM sales;
```

---

## 7. Practice Problems

### Exercise 1: Sales Performance Analysis
Analyze each salesperson's sales amount and calculate:
- Rank by salesperson
- Percentage of total
- Change from previous sale

```sql
-- Example answer
SELECT
    salesperson,
    sale_date,
    amount,
    RANK() OVER (ORDER BY amount DESC) AS overall_rank,
    RANK() OVER (
        PARTITION BY salesperson
        ORDER BY amount DESC
    ) AS personal_rank,
    ROUND(amount * 100.0 / SUM(amount) OVER (), 2) AS pct_of_total,
    amount - LAG(amount) OVER (
        PARTITION BY salesperson ORDER BY sale_date
    ) AS change_from_prev
FROM sales
ORDER BY salesperson, sale_date;
```

### Exercise 2: Moving Sum
Calculate moving sum for the last 7 days.

```sql
-- Example answer
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    ) AS rolling_7day_sum
FROM sales
ORDER BY sale_date;
```

### Exercise 3: Find Target Achievement Date
Find the first date when cumulative sales reached 5000.

```sql
-- Example answer
SELECT sale_date, cumulative
FROM (
    SELECT
        sale_date,
        SUM(amount) OVER (ORDER BY sale_date) AS cumulative,
        LAG(SUM(amount) OVER (ORDER BY sale_date)) OVER (ORDER BY sale_date) AS prev_cumulative
    FROM sales
) sub
WHERE cumulative >= 5000
  AND (prev_cumulative IS NULL OR prev_cumulative < 5000)
LIMIT 1;
```

---

## References
- [PostgreSQL Window Functions](https://www.postgresql.org/docs/current/functions-window.html)
- [Window Function Tutorial](https://www.postgresql.org/docs/current/tutorial-window.html)
- [SQL Window Functions](https://mode.com/sql-tutorial/sql-window-functions/)

---

**Previous**: [Replication and High Availability](./16_Replication_HA.md) | **Next**: [Table Partitioning](./18_Table_Partitioning.md)
