-- Exercises for Lesson 17: Window Functions
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- Setup: Create a sales table for window function exercises
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    salesperson VARCHAR(50),
    region VARCHAR(50),
    sale_date DATE,
    amount NUMERIC(10,2)
);

TRUNCATE TABLE sales RESTART IDENTITY;

INSERT INTO sales (salesperson, region, sale_date, amount) VALUES
('Alice', 'East', '2024-01-05', 1200.00),
('Alice', 'East', '2024-01-15', 800.00),
('Alice', 'East', '2024-02-10', 1500.00),
('Alice', 'East', '2024-03-01', 950.00),
('Bob',   'West', '2024-01-08', 2000.00),
('Bob',   'West', '2024-01-22', 1100.00),
('Bob',   'West', '2024-02-14', 1800.00),
('Bob',   'West', '2024-03-05', 600.00),
('Carol', 'East', '2024-01-10', 900.00),
('Carol', 'East', '2024-02-20', 1600.00),
('Carol', 'East', '2024-03-15', 2200.00),
('Dave',  'West', '2024-01-18', 750.00),
('Dave',  'West', '2024-02-08', 1300.00),
('Dave',  'West', '2024-03-22', 1100.00);


-- === Exercise 1: Sales Performance Analysis ===
-- Problem: For each sale, show overall rank, personal rank,
-- percentage of total, and change from previous sale.

-- Solution:

SELECT
    salesperson,
    sale_date,
    amount,

    -- Overall rank across all salespeople (descending by amount)
    RANK() OVER (ORDER BY amount DESC) AS overall_rank,

    -- Personal rank within each salesperson's own sales
    RANK() OVER (
        PARTITION BY salesperson
        ORDER BY amount DESC
    ) AS personal_rank,

    -- What percentage of total revenue does this single sale represent?
    -- SUM() OVER () with no frame = grand total across all rows
    ROUND(amount * 100.0 / SUM(amount) OVER (), 2) AS pct_of_total,

    -- Difference from previous sale (chronologically, per salesperson)
    -- LAG looks at the previous row within the same partition
    amount - LAG(amount) OVER (
        PARTITION BY salesperson ORDER BY sale_date
    ) AS change_from_prev

FROM sales
ORDER BY salesperson, sale_date;


-- === Exercise 2: Moving Sum ===
-- Problem: Calculate a rolling 7-day sum for each sale date.

-- Solution:

-- RANGE BETWEEN compares actual date values (not row positions).
-- "6 days PRECEDING AND CURRENT ROW" creates a 7-day window:
-- the current row's date minus 6 days through the current date.
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    ) AS rolling_7day_sum
FROM sales
ORDER BY sale_date;

-- Bonus: compare RANGE vs ROWS behavior
-- ROWS BETWEEN counts physical rows regardless of date gaps
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3row_sum,
    SUM(amount) OVER (
        ORDER BY sale_date
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    ) AS rolling_7day_sum
FROM sales
ORDER BY sale_date;


-- === Exercise 3: Find Target Achievement Date ===
-- Problem: Find the first date when cumulative sales reached 5000.

-- Solution:

-- Strategy: compute a running total, then find the first row where
-- cumulative >= 5000 AND the previous cumulative was < 5000 (or NULL).
-- This identifies the exact "crossing point."
SELECT sale_date, cumulative
FROM (
    SELECT
        sale_date,
        SUM(amount) OVER (ORDER BY sale_date) AS cumulative,
        LAG(SUM(amount) OVER (ORDER BY sale_date))
            OVER (ORDER BY sale_date) AS prev_cumulative
    FROM sales
) sub
WHERE cumulative >= 5000
  AND (prev_cumulative IS NULL OR prev_cumulative < 5000)
LIMIT 1;

-- Alternative approach using a CTE (more readable)
WITH running_totals AS (
    SELECT
        sale_date,
        amount,
        SUM(amount) OVER (ORDER BY sale_date) AS cumulative_total
    FROM sales
)
SELECT
    sale_date AS target_reached_date,
    cumulative_total
FROM running_totals
WHERE cumulative_total >= 5000
ORDER BY sale_date
LIMIT 1;

-- Bonus: Find achievement dates for multiple targets
WITH running_totals AS (
    SELECT
        sale_date,
        SUM(amount) OVER (ORDER BY sale_date) AS cumulative
    FROM sales
),
targets(target) AS (
    VALUES (5000), (10000), (15000)
)
SELECT DISTINCT ON (t.target)
    t.target,
    rt.sale_date AS achieved_date,
    rt.cumulative
FROM targets t
JOIN running_totals rt ON rt.cumulative >= t.target
ORDER BY t.target, rt.sale_date;
