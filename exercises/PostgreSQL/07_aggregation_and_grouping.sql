-- Exercises for Lesson 07: Aggregation and Grouping
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- Setup: Create and populate a sales table
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    product VARCHAR(100),
    category VARCHAR(50),
    region VARCHAR(50),
    quantity INTEGER,
    amount INTEGER,
    sale_date DATE
);


-- === Exercise 1: Basic Aggregation ===
-- Problem: Calculate overall and per-category sales statistics.

-- Solution:

-- 1. Overall sales statistics using all five aggregate functions
SELECT
    COUNT(*) AS total_sales,
    SUM(amount) AS total_revenue,
    ROUND(AVG(amount), 0) AS avg_revenue,
    MIN(amount) AS min_revenue,
    MAX(amount) AS max_revenue
FROM sales;

-- 2. Per-category breakdown
-- GROUP BY aggregates rows that share the same category value
SELECT
    category,
    COUNT(*) AS sales_count,
    SUM(quantity) AS total_quantity,
    SUM(amount) AS total_revenue,
    ROUND(AVG(amount), 0) AS avg_revenue
FROM sales
GROUP BY category
ORDER BY total_revenue DESC;


-- === Exercise 2: Complex Conditions ===
-- Problem: Use HAVING to filter grouped results and rank products.

-- Solution:

-- 1. Sales by region, showing only regions with 500,000+ total
-- HAVING filters after GROUP BY (WHERE filters before)
SELECT
    region,
    SUM(amount) AS total
FROM sales
GROUP BY region
HAVING SUM(amount) >= 500000
ORDER BY total DESC;

-- 2. Top 5 products by total quantity sold
SELECT
    product,
    SUM(quantity) AS total_qty
FROM sales
GROUP BY product
ORDER BY total_qty DESC
LIMIT 5;


-- === Exercise 3: Date Aggregation ===
-- Problem: Calculate daily sales trends with cumulative totals.

-- Solution:

-- 1. Daily sales with a running cumulative total
-- The window function SUM() OVER (ORDER BY) computes a running total
-- across the already-grouped daily_sales values
SELECT
    sale_date,
    SUM(amount) AS daily_sales,
    SUM(SUM(amount)) OVER (ORDER BY sale_date) AS cumulative_sales
FROM sales
GROUP BY sale_date
ORDER BY sale_date;

-- 2. Average daily sales for the last 7 days
-- Subquery first computes daily totals, then the outer query averages them
SELECT
    ROUND(AVG(daily_total), 2) AS avg_daily_sales
FROM (
    SELECT sale_date, SUM(amount) AS daily_total
    FROM sales
    WHERE sale_date >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY sale_date
) daily;


-- === Exercise 4: Crosstab (Pivot) ===
-- Problem: Create a category-by-region sales crosstab using FILTER.

-- Solution:

-- FILTER (WHERE ...) is PostgreSQL's clean syntax for conditional aggregation.
-- Each FILTER clause computes a separate SUM for one specific region,
-- producing a pivot table without needing the tablefunc/crosstab extension.
SELECT
    category,
    SUM(amount) FILTER (WHERE region = 'Seoul') AS seoul,
    SUM(amount) FILTER (WHERE region = 'Busan') AS busan,
    SUM(amount) FILTER (WHERE region = 'Daejeon') AS daejeon,
    SUM(amount) AS total
FROM sales
GROUP BY category;
