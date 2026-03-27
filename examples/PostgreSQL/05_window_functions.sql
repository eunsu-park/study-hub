-- =============================================================================
-- PostgreSQL Window Function Examples
-- Window Functions (Analytic Functions)
-- =============================================================================

-- First, run the previous example files to create tables and data.

-- =============================================================================
-- 1. Basic Window Function Structure
-- =============================================================================

-- OVER() - Treats the entire table as one window
SELECT
    first_name,
    last_name,
    salary,
    SUM(salary) OVER() AS total_salary,
    ROUND(AVG(salary) OVER(), 2) AS avg_salary,
    COUNT(*) OVER() AS total_count
FROM employees;

-- =============================================================================
-- 2. PARTITION BY - Window per Group
-- =============================================================================

-- Partition by department
SELECT
    e.first_name,
    e.last_name,
    d.dept_name,
    e.salary,
    SUM(e.salary) OVER(PARTITION BY e.dept_id) AS dept_total,
    ROUND(AVG(e.salary) OVER(PARTITION BY e.dept_id), 2) AS dept_avg,
    COUNT(*) OVER(PARTITION BY e.dept_id) AS dept_count
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, e.salary DESC;

-- Salary ratio by department
SELECT
    e.first_name,
    e.last_name,
    d.dept_name,
    e.salary,
    SUM(e.salary) OVER(PARTITION BY e.dept_id) AS dept_total,
    ROUND(100.0 * e.salary / SUM(e.salary) OVER(PARTITION BY e.dept_id), 2) AS salary_pct
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, salary_pct DESC;

-- =============================================================================
-- 3. Ranking Functions
-- =============================================================================

-- ROW_NUMBER: Sequential number (different numbers even for ties)
-- RANK: Same rank for ties, skips next rank
-- DENSE_RANK: Same rank for ties, consecutive next rank

SELECT
    first_name,
    last_name,
    salary,
    ROW_NUMBER() OVER(ORDER BY salary DESC) AS row_num,
    RANK() OVER(ORDER BY salary DESC) AS rank,
    DENSE_RANK() OVER(ORDER BY salary DESC) AS dense_rank
FROM employees;

-- Salary rank within department
SELECT
    e.first_name,
    e.last_name,
    d.dept_name,
    e.salary,
    RANK() OVER(PARTITION BY e.dept_id ORDER BY e.salary DESC) AS dept_salary_rank
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, dept_salary_rank;

-- Why: ROW_NUMBER is used instead of RANK here because we want exactly 2 rows
-- per department even if there are ties. RANK would return more than 2 if
-- multiple employees share the same salary. The subquery wrapper is needed
-- because window functions cannot be used directly in WHERE.
SELECT * FROM (
    SELECT
        e.first_name,
        e.last_name,
        d.dept_name,
        e.salary,
        ROW_NUMBER() OVER(PARTITION BY e.dept_id ORDER BY e.salary DESC) AS rn
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
) ranked
WHERE rn <= 2;

-- NTILE: Divide into N buckets
SELECT
    first_name,
    last_name,
    salary,
    NTILE(4) OVER(ORDER BY salary DESC) AS salary_quartile
FROM employees;

-- =============================================================================
-- 4. Offset Functions (LAG, LEAD, FIRST_VALUE, LAST_VALUE)
-- =============================================================================

-- LAG: Previous row value
-- LEAD: Next row value
SELECT
    first_name,
    hire_date,
    LAG(first_name, 1) OVER(ORDER BY hire_date) AS prev_hire,
    LEAD(first_name, 1) OVER(ORDER BY hire_date) AS next_hire
FROM employees
ORDER BY hire_date;

-- Salary change analysis
SELECT
    first_name,
    salary,
    LAG(salary) OVER(ORDER BY emp_id) AS prev_salary,
    salary - LAG(salary) OVER(ORDER BY emp_id) AS salary_diff
FROM employees;

-- Previous/next employee within department
SELECT
    e.first_name,
    d.dept_name,
    e.salary,
    LAG(e.salary) OVER(PARTITION BY e.dept_id ORDER BY e.salary) AS lower_salary,
    LEAD(e.salary) OVER(PARTITION BY e.dept_id ORDER BY e.salary) AS higher_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, e.salary;

-- Why: LAST_VALUE requires an explicit frame (ROWS BETWEEN UNBOUNDED PRECEDING AND
-- UNBOUNDED FOLLOWING) because the default frame only extends to CURRENT ROW,
-- which would make LAST_VALUE return the current row itself — a common gotcha.
-- FIRST_VALUE works correctly with the default frame since it always starts at
-- the partition beginning.
SELECT
    e.first_name,
    d.dept_name,
    e.salary,
    FIRST_VALUE(e.first_name) OVER(
        PARTITION BY e.dept_id ORDER BY e.salary DESC
    ) AS highest_paid,
    LAST_VALUE(e.first_name) OVER(
        PARTITION BY e.dept_id ORDER BY e.salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- =============================================================================
-- 5. Aggregate Window Functions
-- =============================================================================

-- Running Total
SELECT
    first_name,
    hire_date,
    salary,
    SUM(salary) OVER(ORDER BY hire_date) AS running_total
FROM employees
ORDER BY hire_date;

-- Running total by department
SELECT
    e.first_name,
    d.dept_name,
    e.salary,
    SUM(e.salary) OVER(
        PARTITION BY e.dept_id
        ORDER BY e.emp_id
    ) AS dept_running_total
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, e.emp_id;

-- Why: A moving average smooths out noise by averaging over a sliding window.
-- ROWS BETWEEN 2 PRECEDING AND CURRENT ROW creates a 3-row window. For the
-- first two rows the window is smaller (1 or 2 rows), so the average is less
-- smooth — be aware of this edge effect at partition boundaries.
SELECT
    first_name,
    salary,
    ROUND(AVG(salary) OVER(
        ORDER BY emp_id
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS moving_avg_3
FROM employees;

-- =============================================================================
-- 6. Frame Specification
-- =============================================================================

-- ROWS: Based on physical rows
-- RANGE: Based on logical values

-- 1 row before and after the current row
SELECT
    first_name,
    salary,
    AVG(salary) OVER(
        ORDER BY salary
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS avg_neighbors
FROM employees;

-- From the beginning to the current row
SELECT
    first_name,
    salary,
    MAX(salary) OVER(
        ORDER BY emp_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS max_so_far
FROM employees;

-- From the current row to the end
SELECT
    first_name,
    salary,
    COUNT(*) OVER(
        ORDER BY emp_id
        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
    ) AS remaining_count
FROM employees;

-- =============================================================================
-- 7. Percentile and Distribution Functions
-- =============================================================================

-- PERCENT_RANK: Relative rank between 0 and 1
-- CUME_DIST: Cumulative distribution

SELECT
    first_name,
    salary,
    ROUND(PERCENT_RANK() OVER(ORDER BY salary)::numeric, 4) AS pct_rank,
    ROUND(CUME_DIST() OVER(ORDER BY salary)::numeric, 4) AS cumulative_dist
FROM employees;

-- Percentile within department
SELECT
    e.first_name,
    d.dept_name,
    e.salary,
    ROUND(PERCENT_RANK() OVER(
        PARTITION BY e.dept_id ORDER BY e.salary
    )::numeric, 4) AS dept_percentile
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- =============================================================================
-- 8. Practical Examples
-- =============================================================================

-- Comprehensive salary statistics
SELECT
    e.first_name,
    e.last_name,
    d.dept_name,
    e.salary,
    -- Department statistics
    ROUND(AVG(e.salary) OVER(PARTITION BY e.dept_id), 2) AS dept_avg,
    MIN(e.salary) OVER(PARTITION BY e.dept_id) AS dept_min,
    MAX(e.salary) OVER(PARTITION BY e.dept_id) AS dept_max,
    -- Company-wide statistics
    ROUND(AVG(e.salary) OVER(), 2) AS company_avg,
    -- Ranking
    RANK() OVER(PARTITION BY e.dept_id ORDER BY e.salary DESC) AS dept_rank,
    RANK() OVER(ORDER BY e.salary DESC) AS company_rank,
    -- Percentage of company total
    ROUND(100.0 * e.salary / SUM(e.salary) OVER(), 2) AS company_pct
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, e.salary DESC;

-- Consecutive hiring analysis (interval between hire dates)
SELECT
    first_name,
    hire_date,
    LAG(hire_date) OVER(ORDER BY hire_date) AS prev_hire_date,
    hire_date - LAG(hire_date) OVER(ORDER BY hire_date) AS days_since_last_hire
FROM employees
ORDER BY hire_date;

-- Salary band classification
SELECT
    first_name,
    salary,
    CASE NTILE(3) OVER(ORDER BY salary)
        WHEN 1 THEN 'Low'
        WHEN 2 THEN 'Medium'
        WHEN 3 THEN 'High'
    END AS salary_band
FROM employees;

-- =============================================================================
-- 9. Window Function Alias (WINDOW Clause)
-- =============================================================================

-- Why: The WINDOW clause avoids repeating the same window definition multiple
-- times. Without it, each OVER(...) would duplicate the partition/order spec,
-- making the query harder to read and error-prone to maintain.
SELECT
    first_name,
    salary,
    SUM(salary) OVER w AS running_total,
    AVG(salary) OVER w AS running_avg,
    COUNT(*) OVER w AS running_count
FROM employees
WINDOW w AS (ORDER BY emp_id)
ORDER BY emp_id;

-- =============================================================================
-- Window Function Summary
-- =============================================================================
/*
Ranking Functions:
- ROW_NUMBER(): Sequential number
- RANK(): Same rank for ties, skips next
- DENSE_RANK(): Same rank for ties, consecutive
- NTILE(n): Divide into n buckets

Offset Functions:
- LAG(col, n, default): Value n rows before
- LEAD(col, n, default): Value n rows after
- FIRST_VALUE(col): First value in the frame
- LAST_VALUE(col): Last value in the frame
- NTH_VALUE(col, n): Nth value in the frame

Aggregate Functions:
- SUM(), AVG(), COUNT(), MIN(), MAX() - with OVER()

Distribution Functions:
- PERCENT_RANK(): Percentile rank (0~1)
- CUME_DIST(): Cumulative distribution

Frame Clause:
- ROWS BETWEEN ... AND ...
- RANGE BETWEEN ... AND ...
- UNBOUNDED PRECEDING / FOLLOWING
- CURRENT ROW
- n PRECEDING / FOLLOWING

Syntax:
window_function() OVER(
    [PARTITION BY col, ...]
    [ORDER BY col [ASC|DESC], ...]
    [frame_clause]
)
*/
