-- =============================================================================
-- PostgreSQL Aggregation Function Examples
-- Aggregation Functions and GROUP BY
-- =============================================================================

-- First, run 01_basic_crud.sql and 02_joins.sql to create tables and data.

-- =============================================================================
-- 1. Basic Aggregation Functions
-- =============================================================================

-- COUNT - Row count
SELECT COUNT(*) AS total_employees FROM employees;

SELECT COUNT(email) AS employees_with_email FROM employees;  -- Excludes NULL

SELECT COUNT(DISTINCT dept_id) AS unique_departments FROM employees;

-- SUM - Total
SELECT SUM(salary) AS total_salary FROM employees;

-- AVG - Average
SELECT AVG(salary) AS average_salary FROM employees;

SELECT ROUND(AVG(salary), 2) AS avg_salary_rounded FROM employees;

-- MIN, MAX - Minimum, Maximum
SELECT
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary
FROM employees;

SELECT
    MIN(hire_date) AS first_hire,
    MAX(hire_date) AS last_hire
FROM employees;

-- Using all aggregation functions together
SELECT
    COUNT(*) AS employee_count,
    SUM(salary) AS total_salary,
    ROUND(AVG(salary), 2) AS avg_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary,
    MAX(salary) - MIN(salary) AS salary_range
FROM employees;

-- =============================================================================
-- 2. GROUP BY
-- =============================================================================

-- Aggregate by department
SELECT
    dept_id,
    COUNT(*) AS employee_count,
    ROUND(AVG(salary), 2) AS avg_salary
FROM employees
WHERE dept_id IS NOT NULL
GROUP BY dept_id
ORDER BY employee_count DESC;

-- Used with JOIN
SELECT
    d.dept_name,
    COUNT(e.emp_id) AS employee_count,
    SUM(e.salary) AS total_salary,
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
ORDER BY total_salary DESC NULLS LAST;

-- Group by multiple columns
SELECT
    dept_id,
    is_active,
    COUNT(*) AS employee_count
FROM employees
GROUP BY dept_id, is_active
ORDER BY dept_id, is_active;

-- Group by expression
SELECT
    EXTRACT(YEAR FROM hire_date) AS hire_year,
    COUNT(*) AS hire_count
FROM employees
GROUP BY EXTRACT(YEAR FROM hire_date)
ORDER BY hire_year;

-- =============================================================================
-- 3. HAVING - Group Filtering
-- =============================================================================

-- Departments with 2 or more employees
SELECT
    d.dept_name,
    COUNT(e.emp_id) AS employee_count
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
HAVING COUNT(e.emp_id) >= 2
ORDER BY employee_count DESC;

-- Departments with average salary >= 50000
SELECT
    d.dept_name,
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
HAVING AVG(e.salary) >= 50000
ORDER BY avg_salary DESC;

-- Why: WHERE filters individual rows BEFORE grouping (less data to aggregate),
-- while HAVING filters AFTER grouping (on aggregated results). Putting conditions
-- in WHERE whenever possible is more efficient because it reduces the input to
-- the GROUP BY operation.
SELECT
    d.dept_name,
    COUNT(e.emp_id) AS employee_count,
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
WHERE e.is_active = TRUE  -- Row filtering (before grouping)
GROUP BY d.dept_id, d.dept_name
HAVING COUNT(e.emp_id) >= 1  -- Group filtering (after grouping)
ORDER BY avg_salary DESC;

-- =============================================================================
-- 4. Advanced Aggregation Functions
-- =============================================================================

-- STRING_AGG - String concatenation
SELECT
    d.dept_name,
    STRING_AGG(e.first_name || ' ' || e.last_name, ', ' ORDER BY e.first_name) AS employee_names
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- ARRAY_AGG - Aggregate into array
SELECT
    d.dept_name,
    ARRAY_AGG(e.first_name ORDER BY e.first_name) AS employee_names_array
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- JSON_AGG - Aggregate into JSON array
SELECT
    d.dept_name,
    JSON_AGG(
        JSON_BUILD_OBJECT(
            'name', e.first_name || ' ' || e.last_name,
            'salary', e.salary
        )
    ) AS employees_json
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- =============================================================================
-- 5. ROLLUP, CUBE, GROUPING SETS
-- =============================================================================

-- Why: ROLLUP generates hierarchical subtotals by progressively removing columns
-- from right to left: (dept, year), (dept), then grand total. This is much
-- more efficient than running separate GROUP BY queries for each subtotal level
-- and UNION-ing them together.
SELECT
    d.dept_name,
    EXTRACT(YEAR FROM e.hire_date) AS hire_year,
    COUNT(*) AS employee_count,
    SUM(e.salary) AS total_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY ROLLUP(d.dept_name, EXTRACT(YEAR FROM e.hire_date))
ORDER BY d.dept_name NULLS LAST, hire_year NULLS LAST;

-- CUBE - Subtotals for all possible combinations
SELECT
    d.dept_name,
    e.is_active,
    COUNT(*) AS employee_count
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY CUBE(d.dept_name, e.is_active)
ORDER BY d.dept_name NULLS LAST, e.is_active NULLS LAST;

-- GROUPING SETS - Subtotals for specific combinations only
SELECT
    d.dept_name,
    EXTRACT(YEAR FROM e.hire_date) AS hire_year,
    COUNT(*) AS employee_count
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY GROUPING SETS (
    (d.dept_name),
    (EXTRACT(YEAR FROM e.hire_date)),
    ()
)
ORDER BY d.dept_name NULLS LAST, hire_year NULLS LAST;

-- GROUPING() function to identify subtotal rows
SELECT
    CASE WHEN GROUPING(d.dept_name) = 1 THEN 'All Departments' ELSE d.dept_name END AS dept_name,
    COUNT(*) AS employee_count,
    SUM(e.salary) AS total_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY ROLLUP(d.dept_name)
ORDER BY GROUPING(d.dept_name), d.dept_name;

-- =============================================================================
-- 6. FILTER Clause
-- =============================================================================

-- Why: FILTER clause is PostgreSQL's cleaner alternative to CASE-based conditional
-- aggregation. It reads more naturally and the optimizer can sometimes handle it
-- more efficiently than SUM(CASE WHEN ... END) because the intent is explicit.
SELECT
    d.dept_name,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE e.salary > 50000) AS high_salary_count,
    COUNT(*) FILTER (WHERE e.salary <= 50000) AS low_salary_count,
    ROUND(AVG(e.salary) FILTER (WHERE e.is_active = TRUE), 2) AS active_avg_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- =============================================================================
-- 7. Statistical Functions
-- =============================================================================

-- Standard deviation and variance
SELECT
    d.dept_name,
    ROUND(AVG(e.salary), 2) AS avg_salary,
    ROUND(STDDEV(e.salary), 2) AS stddev_salary,
    ROUND(VARIANCE(e.salary), 2) AS variance_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
-- Why: STDDEV of a single value is NULL (undefined), so we filter out
-- departments with only one employee to avoid meaningless results.
HAVING COUNT(*) > 1;

-- Why: PERCENTILE_CONT interpolates between values for continuous distributions,
-- making it ideal for salary medians. PERCENTILE_DISC (discrete) would return an
-- actual row value instead. The WITHIN GROUP syntax is needed because percentile
-- computation requires a specific sort order.
SELECT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) AS q1_salary,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS q3_salary
FROM employees;

-- Mode
SELECT
    MODE() WITHIN GROUP (ORDER BY dept_id) AS most_common_dept
FROM employees;

-- =============================================================================
-- 8. Conditional Aggregation (with CASE)
-- =============================================================================

SELECT
    d.dept_name,
    COUNT(*) AS total,
    SUM(CASE WHEN e.salary >= 55000 THEN 1 ELSE 0 END) AS high_earners,
    SUM(CASE WHEN e.salary < 55000 THEN 1 ELSE 0 END) AS others,
    ROUND(
        100.0 * SUM(CASE WHEN e.salary >= 55000 THEN 1 ELSE 0 END) / COUNT(*),
        1
    ) AS high_earner_pct
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- =============================================================================
-- Aggregation Function Summary
-- =============================================================================
/*
Basic Aggregation:
- COUNT(*), COUNT(col), COUNT(DISTINCT col)
- SUM(col), AVG(col)
- MIN(col), MAX(col)

String/Array Aggregation:
- STRING_AGG(col, delimiter)
- ARRAY_AGG(col)
- JSON_AGG(value)

Grouping:
- GROUP BY: Basic grouping
- HAVING: Group filtering (after aggregation)
- ROLLUP: Hierarchical subtotals
- CUBE: All combination subtotals
- GROUPING SETS: Specific combination subtotals

Advanced:
- FILTER (WHERE ...): Conditional aggregation
- PERCENTILE_CONT(): Percentiles
- STDDEV(), VARIANCE(): Statistics

Execution Order:
SELECT -> FROM -> WHERE -> GROUP BY -> HAVING -> ORDER BY -> LIMIT
*/
