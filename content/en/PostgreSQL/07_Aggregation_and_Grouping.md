# Aggregation and Grouping

**Previous**: [JOIN](./06_JOIN.md) | **Next**: [Subqueries and CTE](./08_Subqueries_and_CTE.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the five core aggregate functions: COUNT, SUM, AVG, MIN, and MAX
2. Use GROUP BY to partition rows into groups and compute per-group statistics
3. Filter grouped results with HAVING and explain how it differs from WHERE
4. Combine GROUP BY with JOIN to aggregate data across related tables
5. Perform date-based aggregation using DATE_TRUNC and EXTRACT
6. Write conditional aggregations with CASE expressions and the FILTER clause
7. Generate subtotals and grand totals using ROLLUP and CUBE
8. Describe the SQL query execution order (FROM, WHERE, GROUP BY, HAVING, SELECT, ORDER BY, LIMIT)

---

Databases excel at summarizing large volumes of data into concise, actionable numbers. Questions like "What are our total sales by region?" or "Which product category generates the highest average revenue?" require aggregate functions and grouping. Mastering these operations turns a flat table of transactions into the dashboards, reports, and KPIs that drive business decisions.

## 1. Aggregate Functions

Aggregate functions calculate multiple rows into a single result.

| Function | Description |
|----------|-------------|
| `COUNT()` | Row count |
| `SUM()` | Sum |
| `AVG()` | Average |
| `MIN()` | Minimum value |
| `MAX()` | Maximum value |

---

## 2. Practice Table Setup

```sql
CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    product VARCHAR(100),
    category VARCHAR(50),
    amount NUMERIC(10, 2),
    quantity INTEGER,
    sale_date DATE,
    region VARCHAR(50)
);

INSERT INTO sales (product, category, amount, quantity, sale_date, region) VALUES
('Laptop', 'Electronics', 1500000, 2, '2024-01-05', 'Seoul'),
('Mouse', 'Electronics', 50000, 10, '2024-01-05', 'Seoul'),
('Keyboard', 'Electronics', 100000, 5, '2024-01-06', 'Busan'),
('Monitor', 'Electronics', 300000, 3, '2024-01-07', 'Seoul'),
('Desk', 'Furniture', 250000, 2, '2024-01-08', 'Daejeon'),
('Chair', 'Furniture', 150000, 4, '2024-01-08', 'Seoul'),
('Laptop', 'Electronics', 1800000, 1, '2024-01-10', 'Busan'),
('Mouse', 'Electronics', 45000, 20, '2024-01-12', 'Daejeon'),
('Desk', 'Furniture', 280000, 1, '2024-01-15', 'Seoul'),
('Chair', 'Furniture', 180000, 3, '2024-01-15', 'Busan');
```

---

## 3. COUNT - Counting

### Total Row Count

```sql
SELECT COUNT(*) FROM sales;
-- 10
```

### Count Specific Column (Excludes NULL)

```sql
SELECT COUNT(region) FROM sales;
-- Count of non-NULL region
```

### Count Distinct

```sql
SELECT COUNT(DISTINCT category) FROM sales;
-- 2 (Electronics, Furniture)

SELECT COUNT(DISTINCT region) FROM sales;
-- 3 (Seoul, Busan, Daejeon)
```

---

## 4. SUM - Summation

```sql
-- Aggregates collapse millions of rows into a single answer — the database performs
-- the computation server-side, avoiding the cost of transferring all rows to the app
-- Total sales amount
SELECT SUM(amount) FROM sales;
-- 4653000

-- Total quantity sold
SELECT SUM(quantity) FROM sales;
-- 51

-- Conditional sum
SELECT SUM(amount) FROM sales WHERE category = 'Electronics';
```

---

## 5. AVG - Average

```sql
-- Average sales amount
SELECT AVG(amount) FROM sales;
-- 465300

-- Handle decimals
SELECT ROUND(AVG(amount), 2) AS avg_amount FROM sales;

-- Conditional average
SELECT ROUND(AVG(amount), 2)
FROM sales
WHERE region = 'Seoul';
```

---

## 6. MIN / MAX - Minimum/Maximum

```sql
-- Minimum sales amount
SELECT MIN(amount) FROM sales;
-- 45000

-- Maximum sales amount
SELECT MAX(amount) FROM sales;
-- 1800000

-- Most recent sale date
SELECT MAX(sale_date) FROM sales;

-- Oldest sale date
SELECT MIN(sale_date) FROM sales;
```

---

## 7. Using Multiple Aggregate Functions Together

```sql
SELECT
    COUNT(*) AS total_count,
    SUM(amount) AS total_sales,
    ROUND(AVG(amount), 2) AS avg_sales,
    MIN(amount) AS min_sales,
    MAX(amount) AS max_sales,
    SUM(quantity) AS total_quantity
FROM sales;
```

---

## 8. GROUP BY - Grouping

Groups data by specific columns for aggregation.

### Basic GROUP BY

```sql
-- GROUP BY partitions rows into buckets so each aggregate (COUNT, SUM) runs
-- independently per group — this is how you answer "totals per category" in one query
SELECT
    category,
    COUNT(*) AS count,
    SUM(amount) AS total_amount
FROM sales
GROUP BY category;
```

Result:
```
  category   │ count │ total_amount
─────────────┼───────┼──────────────
 Electronics │     6 │      3795000
 Furniture   │     4 │       858000
```

### Sales by Region

```sql
SELECT
    region,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_amount
FROM sales
GROUP BY region
ORDER BY total_amount DESC;
```

### Sales by Product

```sql
SELECT
    product,
    SUM(quantity) AS total_qty,
    SUM(amount) AS total_sales
FROM sales
GROUP BY product
ORDER BY total_sales DESC;
```

---

## 9. Multi-Column GROUP BY

```sql
-- Sales by category + region
SELECT
    category,
    region,
    COUNT(*) AS count,
    SUM(amount) AS total
FROM sales
GROUP BY category, region
ORDER BY category, region;
```

Result:
```
  category   │ region │ count │  total
─────────────┼────────┼───────┼─────────
 Furniture   │ Daejeon│     1 │  250000
 Furniture   │ Busan  │     1 │  180000
 Furniture   │ Seoul  │     2 │  430000
 Electronics │ Daejeon│     1 │   45000
 Electronics │ Busan  │     2 │ 1900000
 Electronics │ Seoul  │     3 │ 1850000
```

---

## 10. HAVING - Group Filtering

WHERE filters individual rows before grouping; HAVING filters entire groups after aggregation.
Use HAVING when the condition involves an aggregate (SUM, COUNT, etc.) that does not exist
until groups are formed -- WHERE cannot reference aggregates because it runs first.

```sql
-- Only categories with total sales >= 500,000
SELECT
    category,
    SUM(amount) AS total_amount
FROM sales
GROUP BY category
HAVING SUM(amount) >= 500000;
```

### WHERE + HAVING

```sql
-- WHERE + HAVING cooperate: WHERE shrinks the dataset first (cheaper), then HAVING
-- filters the aggregated groups — always push filters to WHERE when possible for performance
SELECT
    product,
    SUM(amount) AS total_amount
FROM sales
WHERE region IN ('Seoul', 'Busan')  -- Filter before grouping (row-level)
GROUP BY product
HAVING SUM(amount) >= 1000000       -- Filter after grouping (group-level)
ORDER BY total_amount DESC;
```

### Using Aliases in HAVING (PostgreSQL)

```sql
-- PostgreSQL allows aliases in HAVING
SELECT
    product,
    SUM(amount) AS total
FROM sales
GROUP BY product
HAVING SUM(amount) > 500000;  -- Standard way

-- Or (works in some PostgreSQL versions)
-- HAVING total > 500000;  -- Works only in some versions
```

---

## 11. GROUP BY + JOIN

```sql
-- Setup: Categories table
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    description TEXT
);

INSERT INTO categories (name, description) VALUES
('Electronics', 'Electronic products'),
('Furniture', 'Furniture products');

-- Aggregate with category information
SELECT
    c.name AS category,
    c.description,
    COUNT(s.id) AS sales_count,
    SUM(s.amount) AS total_sales
FROM categories c
LEFT JOIN sales s ON c.name = s.category
GROUP BY c.id, c.name, c.description;
```

---

## 12. Date-Based Aggregation

### Daily Sales

```sql
SELECT
    sale_date,
    COUNT(*) AS count,
    SUM(amount) AS daily_total
FROM sales
GROUP BY sale_date
ORDER BY sale_date;
```

### Monthly Sales

```sql
SELECT
    DATE_TRUNC('month', sale_date) AS month,
    COUNT(*) AS count,
    SUM(amount) AS monthly_total
FROM sales
GROUP BY DATE_TRUNC('month', sale_date)
ORDER BY month;
```

### Yearly Sales

```sql
SELECT
    EXTRACT(YEAR FROM sale_date) AS year,
    SUM(amount) AS yearly_total
FROM sales
GROUP BY EXTRACT(YEAR FROM sale_date);
```

---

## 13. Conditional Aggregation

### CASE + SUM

```sql
SELECT
    SUM(CASE WHEN category = 'Electronics' THEN amount ELSE 0 END) AS electronics,
    SUM(CASE WHEN category = 'Furniture' THEN amount ELSE 0 END) AS furniture
FROM sales;
```

### FILTER (PostgreSQL 9.4+)

```sql
-- FILTER is cleaner than CASE+SUM — it reads as "count only where ..." and the planner
-- can sometimes optimize it better than the equivalent CASE expression
SELECT
    COUNT(*) FILTER (WHERE category = 'Electronics') AS electronics_count,
    COUNT(*) FILTER (WHERE category = 'Furniture') AS furniture_count,
    SUM(amount) FILTER (WHERE region = 'Seoul') AS seoul_sales
FROM sales;
```

---

## 14. ROLLUP and CUBE

### ROLLUP - Add Subtotals

```sql
SELECT
    category,
    region,
    SUM(amount) AS total
FROM sales
GROUP BY ROLLUP (category, region)
ORDER BY category NULLS LAST, region NULLS LAST;
```

Result:
```
  category   │ region │   total
─────────────┼────────┼──────────
 Furniture   │ Daejeon│   250000
 Furniture   │ Busan  │   180000
 Furniture   │ Seoul  │   430000
 Furniture   │ NULL   │   860000  ← Furniture subtotal
 Electronics │ Daejeon│    45000
 Electronics │ Busan  │  1900000
 Electronics │ Seoul  │  1850000
 Electronics │ NULL   │  3795000  ← Electronics subtotal
 NULL        │ NULL   │  4655000  ← Grand total
```

### CUBE - All Combination Subtotals

```sql
SELECT
    category,
    region,
    SUM(amount) AS total
FROM sales
GROUP BY CUBE (category, region)
ORDER BY category NULLS LAST, region NULLS LAST;
```

### GROUPING - Distinguish NULL

```sql
SELECT
    CASE WHEN GROUPING(category) = 1 THEN 'All' ELSE category END AS category,
    CASE WHEN GROUPING(region) = 1 THEN 'All' ELSE region END AS region,
    SUM(amount) AS total
FROM sales
GROUP BY ROLLUP (category, region);
```

---

## 15. Practice Examples

### Practice 1: Basic Aggregation

```sql
-- 1. Overall sales statistics
SELECT
    COUNT(*) AS total_sales,
    SUM(amount) AS total_revenue,
    ROUND(AVG(amount), 0) AS avg_revenue,
    MIN(amount) AS min_revenue,
    MAX(amount) AS max_revenue
FROM sales;

-- 2. Sales statistics by category
SELECT
    category,
    COUNT(*) AS sales_count,
    SUM(quantity) AS total_quantity,
    SUM(amount) AS total_revenue,
    ROUND(AVG(amount), 0) AS avg_revenue
FROM sales
GROUP BY category
ORDER BY total_revenue DESC;
```

### Practice 2: Complex Conditions

```sql
-- 1. Sales by region (500,000+ only)
SELECT
    region,
    SUM(amount) AS total
FROM sales
GROUP BY region
HAVING SUM(amount) >= 500000
ORDER BY total DESC;

-- 2. Product quantity ranking
SELECT
    product,
    SUM(quantity) AS total_qty
FROM sales
GROUP BY product
ORDER BY total_qty DESC
LIMIT 5;
```

### Practice 3: Date Aggregation

```sql
-- 1. Daily sales trend
SELECT
    sale_date,
    SUM(amount) AS daily_sales,
    SUM(SUM(amount)) OVER (ORDER BY sale_date) AS cumulative_sales
FROM sales
GROUP BY sale_date
ORDER BY sale_date;

-- 2. Average daily sales for last 7 days
SELECT
    ROUND(AVG(daily_total), 2) AS avg_daily_sales
FROM (
    SELECT sale_date, SUM(amount) AS daily_total
    FROM sales
    WHERE sale_date >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY sale_date
) daily;
```

### Practice 4: Crosstab (Pivot)

```sql
-- Category × Region sales crosstab
SELECT
    category,
    SUM(amount) FILTER (WHERE region = 'Seoul') AS seoul,
    SUM(amount) FILTER (WHERE region = 'Busan') AS busan,
    SUM(amount) FILTER (WHERE region = 'Daejeon') AS daejeon,
    SUM(amount) AS total
FROM sales
GROUP BY category;
```

Result:
```
  category   │  seoul  │  busan  │ daejeon │   total
─────────────┼─────────┼─────────┼─────────┼──────────
 Furniture   │  430000 │  180000 │  250000 │   860000
 Electronics │ 1850000 │ 1900000 │   45000 │  3795000
```

---

## 16. Query Execution Order

```
FROM / JOIN    ← Specify tables
    ↓
WHERE          ← Filter rows
    ↓
GROUP BY       ← Group
    ↓
HAVING         ← Filter groups
    ↓
SELECT         ← Select columns
    ↓
DISTINCT       ← Remove duplicates
    ↓
ORDER BY       ← Sort
    ↓
LIMIT/OFFSET   ← Limit results
```

---

**Previous**: [JOIN](./06_JOIN.md) | **Next**: [Subqueries and CTE](./08_Subqueries_and_CTE.md)
