-- ============================================================================
-- PostgreSQL Advanced Window Functions
-- ============================================================================
-- Demonstrates:
--   - ROW_NUMBER, RANK, DENSE_RANK, NTILE
--   - LAG, LEAD for row comparison
--   - Running totals and moving averages
--   - FIRST_VALUE, LAST_VALUE, NTH_VALUE
--   - Window frame specifications (ROWS, RANGE, GROUPS)
--   - FILTER clause with window functions
--   - Named windows (WINDOW clause)
--
-- Prerequisites: PostgreSQL 12+
-- Note: Extends 05_window_functions.sql with advanced patterns
-- Usage: psql -U postgres -d your_database -f 17_window_functions.sql
-- ============================================================================

-- Clean up
DROP TABLE IF EXISTS sales CASCADE;
DROP TABLE IF EXISTS stock_prices CASCADE;

-- ============================================================================
-- Setup
-- ============================================================================

CREATE TABLE sales (
    sale_id SERIAL PRIMARY KEY,
    salesperson TEXT NOT NULL,
    region TEXT NOT NULL,
    product TEXT NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    sale_date DATE NOT NULL
);

INSERT INTO sales (salesperson, region, product, amount, sale_date) VALUES
    ('Alice', 'East', 'Widget', 1200.00, '2025-01-15'),
    ('Alice', 'East', 'Gadget', 800.00, '2025-01-20'),
    ('Alice', 'East', 'Widget', 1500.00, '2025-02-10'),
    ('Alice', 'East', 'Gadget', 950.00, '2025-03-05'),
    ('Bob', 'West', 'Widget', 2000.00, '2025-01-10'),
    ('Bob', 'West', 'Widget', 1800.00, '2025-02-15'),
    ('Bob', 'West', 'Gadget', 600.00, '2025-02-20'),
    ('Bob', 'West', 'Widget', 2200.00, '2025-03-10'),
    ('Charlie', 'East', 'Widget', 900.00, '2025-01-25'),
    ('Charlie', 'East', 'Gadget', 1100.00, '2025-02-05'),
    ('Charlie', 'East', 'Widget', 1300.00, '2025-03-15'),
    ('Diana', 'West', 'Gadget', 1400.00, '2025-01-18'),
    ('Diana', 'West', 'Widget', 1700.00, '2025-02-22'),
    ('Diana', 'West', 'Gadget', 1600.00, '2025-03-08');

CREATE TABLE stock_prices (
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    close_price NUMERIC(10, 2) NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (ticker, trade_date)
);

INSERT INTO stock_prices (ticker, trade_date, close_price, volume) VALUES
    ('AAPL', '2025-03-03', 150.00, 1000000),
    ('AAPL', '2025-03-04', 152.50, 1200000),
    ('AAPL', '2025-03-05', 151.00, 900000),
    ('AAPL', '2025-03-06', 155.00, 1500000),
    ('AAPL', '2025-03-07', 153.50, 1100000),
    ('AAPL', '2025-03-10', 156.00, 1300000),
    ('AAPL', '2025-03-11', 154.50, 1000000),
    ('AAPL', '2025-03-12', 158.00, 1600000),
    ('GOOG', '2025-03-03', 140.00, 800000),
    ('GOOG', '2025-03-04', 142.00, 900000),
    ('GOOG', '2025-03-05', 139.50, 700000),
    ('GOOG', '2025-03-06', 143.00, 1000000),
    ('GOOG', '2025-03-07', 141.50, 850000),
    ('GOOG', '2025-03-10', 145.00, 1100000),
    ('GOOG', '2025-03-11', 144.00, 950000),
    ('GOOG', '2025-03-12', 147.00, 1200000);

-- ============================================================================
-- 1. Ranking Functions
-- ============================================================================

-- Compare ROW_NUMBER, RANK, DENSE_RANK
SELECT
    salesperson,
    SUM(amount) AS total_sales,
    ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS row_num,
    RANK()       OVER (ORDER BY SUM(amount) DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY SUM(amount) DESC) AS dense_rank,
    NTILE(2)     OVER (ORDER BY SUM(amount) DESC) AS quartile
FROM sales
GROUP BY salesperson;

-- Top N per group: top 2 sales per region
SELECT * FROM (
    SELECT
        region,
        salesperson,
        amount,
        sale_date,
        ROW_NUMBER() OVER (
            PARTITION BY region
            ORDER BY amount DESC
        ) AS rn
    FROM sales
) ranked
WHERE rn <= 2;

-- ============================================================================
-- 2. LAG and LEAD: Row Comparison
-- ============================================================================

-- Why: LAG with a named window (WINDOW w) avoids repeating the partition/order
-- clause four times. Computing percentage change from the previous close is a
-- core financial analytics pattern — PARTITION BY ticker ensures each stock
-- is compared only to its own history, not cross-contaminated with other tickers.
SELECT
    ticker,
    trade_date,
    close_price,
    LAG(close_price) OVER w AS prev_close,
    close_price - LAG(close_price) OVER w AS price_change,
    ROUND(
        (close_price - LAG(close_price) OVER w) /
        LAG(close_price) OVER w * 100, 2
    ) AS change_pct
FROM stock_prices
WINDOW w AS (PARTITION BY ticker ORDER BY trade_date);

-- Compare current sale to previous and next
SELECT
    salesperson,
    sale_date,
    amount,
    LAG(amount, 1, 0) OVER w AS prev_sale,
    LEAD(amount, 1, 0) OVER w AS next_sale,
    amount - LAG(amount, 1, amount) OVER w AS diff_from_prev
FROM sales
WINDOW w AS (PARTITION BY salesperson ORDER BY sale_date);

-- ============================================================================
-- 3. Running Totals and Moving Averages
-- ============================================================================

-- Running total of sales per person
SELECT
    salesperson,
    sale_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM sales
ORDER BY salesperson, sale_date;

-- 3-day moving average of stock prices
SELECT
    ticker,
    trade_date,
    close_price,
    ROUND(
        AVG(close_price) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 2
    ) AS moving_avg_3d,
    ROUND(
        AVG(close_price) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ), 2
    ) AS moving_avg_5d
FROM stock_prices
ORDER BY ticker, trade_date;

-- ============================================================================
-- 4. FIRST_VALUE, LAST_VALUE, NTH_VALUE
-- ============================================================================

SELECT
    salesperson,
    sale_date,
    amount,
    FIRST_VALUE(amount) OVER w AS first_sale,
    -- Why: LAST_VALUE needs the full-partition frame (UNBOUNDED FOLLOWING) explicitly;
    -- without it, the default frame ends at CURRENT ROW, returning the current
    -- row's value instead of the actual last row in the partition.
    LAST_VALUE(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_sale,
    NTH_VALUE(amount, 2) OVER w AS second_sale,
    amount - FIRST_VALUE(amount) OVER w AS diff_from_first
FROM sales
WINDOW w AS (
    PARTITION BY salesperson
    ORDER BY sale_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
);

-- ============================================================================
-- 5. Window Frame Specifications
-- ============================================================================

-- ROWS vs RANGE vs GROUPS
-- ROWS: physical rows
-- RANGE: logical range based on ORDER BY value (same values = same frame)
-- GROUPS: groups of peer rows

-- ROWS: exactly 2 preceding physical rows
SELECT
    trade_date,
    close_price,
    AVG(close_price) OVER (
        ORDER BY trade_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rows_avg
FROM stock_prices
WHERE ticker = 'AAPL';

-- Why: RANGE BETWEEN INTERVAL '3 days' PRECEDING gives a calendar-aware window
-- that correctly handles weekends and holidays (market closed days). ROWS BETWEEN
-- would always look at exactly N physical rows regardless of date gaps, which
-- would mix non-adjacent trading days when the market is closed.
SELECT
    trade_date,
    close_price,
    AVG(close_price) OVER (
        ORDER BY trade_date
        RANGE BETWEEN INTERVAL '3 days' PRECEDING AND CURRENT ROW
    ) AS range_avg
FROM stock_prices
WHERE ticker = 'AAPL';

-- Frame bound options:
--   UNBOUNDED PRECEDING  (start of partition)
--   N PRECEDING          (N rows/range before current)
--   CURRENT ROW
--   N FOLLOWING           (N rows/range after current)
--   UNBOUNDED FOLLOWING   (end of partition)

-- ============================================================================
-- 6. Cumulative Distribution
-- ============================================================================

-- Percentile rank of each sale
SELECT
    salesperson,
    amount,
    PERCENT_RANK() OVER (ORDER BY amount) AS pct_rank,
    CUME_DIST() OVER (ORDER BY amount) AS cumulative_dist,
    NTILE(4) OVER (ORDER BY amount) AS quartile
FROM sales;

-- ============================================================================
-- 7. Named Windows (WINDOW clause)
-- ============================================================================

-- Reuse window definitions
SELECT
    salesperson,
    sale_date,
    amount,
    SUM(amount) OVER by_person AS person_total,
    AVG(amount) OVER by_person AS person_avg,
    COUNT(*) OVER by_person AS person_count,
    SUM(amount) OVER by_region AS region_total,
    ROUND(amount / SUM(amount) OVER by_region * 100, 1) AS region_pct
FROM sales
WINDOW
    by_person AS (PARTITION BY salesperson),
    by_region AS (PARTITION BY region);

-- ============================================================================
-- 8. Practical Patterns
-- ============================================================================

-- Why: The "gaps and islands" technique exploits the fact that for consecutive dates,
-- (date - row_number) produces the same constant. When there is a gap, the
-- constant changes, creating a new group. This is a classic SQL pattern for
-- detecting streaks, consecutive sessions, or missing data ranges.
SELECT
    ticker,
    trade_date,
    close_price,
    trade_date - (ROW_NUMBER() OVER (
        PARTITION BY ticker ORDER BY trade_date
    ))::INTEGER AS island_group
FROM stock_prices
ORDER BY ticker, trade_date;

-- 8b. Year-over-year monthly comparison (simulated with months)
SELECT
    salesperson,
    DATE_TRUNC('month', sale_date) AS month,
    SUM(amount) AS monthly_total,
    LAG(SUM(amount)) OVER (
        PARTITION BY salesperson
        ORDER BY DATE_TRUNC('month', sale_date)
    ) AS prev_month_total
FROM sales
GROUP BY salesperson, DATE_TRUNC('month', sale_date)
ORDER BY salesperson, month;

-- 8c. Running distinct count approximation
SELECT DISTINCT
    sale_date,
    COUNT(DISTINCT salesperson) OVER (
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS unique_sellers_so_far,
    SUM(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM sales
ORDER BY sale_date;
