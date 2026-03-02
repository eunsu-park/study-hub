-- =============================================================================
-- PostgreSQL Subquery and CTE Examples
-- Subqueries and Common Table Expressions (CTE)
-- =============================================================================

-- First, run the previous example files to create tables and data.

-- =============================================================================
-- 1. Scalar Subquery (Returns a Single Value)
-- =============================================================================

-- Used in SELECT clause
SELECT
    first_name,
    last_name,
    salary,
    (SELECT AVG(salary) FROM employees) AS company_avg,
    salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees;

-- Used in WHERE clause
SELECT first_name, last_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Employees earning more than their department average
SELECT e.first_name, e.last_name, e.salary, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE dept_id = e.dept_id
);

-- =============================================================================
-- 2. Inline View (FROM Clause Subquery)
-- =============================================================================

-- Get department statistics via subquery, then join
SELECT
    d.dept_name,
    ds.employee_count,
    ds.avg_salary,
    ds.total_salary
FROM departments d
JOIN (
    SELECT
        dept_id,
        COUNT(*) AS employee_count,
        ROUND(AVG(salary), 2) AS avg_salary,
        SUM(salary) AS total_salary
    FROM employees
    WHERE dept_id IS NOT NULL
    GROUP BY dept_id
) ds ON d.dept_id = ds.dept_id;

-- Query with salary ranking
SELECT *
FROM (
    SELECT
        first_name,
        last_name,
        salary,
        RANK() OVER (ORDER BY salary DESC) AS salary_rank
    FROM employees
) ranked
WHERE salary_rank <= 5;

-- =============================================================================
-- 3. EXISTS / NOT EXISTS
-- =============================================================================

-- Why: EXISTS is preferred over IN for correlated existence checks because it
-- short-circuits (stops scanning as soon as one match is found) and handles NULL
-- values safely. SELECT 1 is a convention — the actual value doesn't matter,
-- only whether any row is returned.
SELECT e.first_name, e.last_name
FROM employees e
WHERE EXISTS (
    SELECT 1
    FROM employee_projects ep
    WHERE ep.emp_id = e.emp_id
);

-- Employees not participating in any project
SELECT e.first_name, e.last_name
FROM employees e
WHERE NOT EXISTS (
    SELECT 1
    FROM employee_projects ep
    WHERE ep.emp_id = e.emp_id
);

-- Departments that have employees
SELECT d.dept_name
FROM departments d
WHERE EXISTS (
    SELECT 1
    FROM employees e
    WHERE e.dept_id = d.dept_id
);

-- =============================================================================
-- 4. IN / NOT IN
-- =============================================================================

-- Engineering department employees
SELECT first_name, last_name
FROM employees
WHERE dept_id IN (
    SELECT dept_id
    FROM departments
    WHERE dept_name = 'Engineering'
);

-- Why: NOT IN has a dangerous gotcha — if the subquery returns ANY NULL value,
-- the entire NOT IN evaluates to NULL (not TRUE), returning zero rows. Prefer
-- NOT EXISTS for safety, or ensure the subquery column is NOT NULL.
SELECT first_name, last_name
FROM employees
WHERE emp_id NOT IN (
    SELECT emp_id FROM employee_projects
);

-- =============================================================================
-- 5. ANY / ALL
-- =============================================================================

-- Employees earning more than any employee in the Engineering department
SELECT first_name, last_name, salary
FROM employees
WHERE salary > ANY (
    SELECT salary
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE d.dept_name = 'Engineering'
);

-- Employees earning more than all employees in the Engineering department
SELECT first_name, last_name, salary
FROM employees
WHERE salary > ALL (
    SELECT salary
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE d.dept_name = 'Engineering'
);

-- =============================================================================
-- 6. Correlated Subquery
-- =============================================================================

-- Compare each employee's salary with their department average
SELECT
    e.first_name,
    e.last_name,
    e.salary,
    (
        SELECT ROUND(AVG(e2.salary), 2)
        FROM employees e2
        WHERE e2.dept_id = e.dept_id
    ) AS dept_avg_salary
FROM employees e
WHERE e.dept_id IS NOT NULL;

-- Highest paid employee in each department
SELECT e.first_name, e.last_name, e.salary, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary = (
    SELECT MAX(e2.salary)
    FROM employees e2
    WHERE e2.dept_id = e.dept_id
);

-- =============================================================================
-- 7. CTE (Common Table Expression) - WITH Clause
-- =============================================================================

-- Basic CTE
WITH dept_stats AS (
    SELECT
        dept_id,
        COUNT(*) AS employee_count,
        ROUND(AVG(salary), 2) AS avg_salary,
        SUM(salary) AS total_salary
    FROM employees
    WHERE dept_id IS NOT NULL
    GROUP BY dept_id
)
SELECT
    d.dept_name,
    ds.employee_count,
    ds.avg_salary,
    ds.total_salary
FROM departments d
JOIN dept_stats ds ON d.dept_id = ds.dept_id
ORDER BY ds.total_salary DESC;

-- Multiple CTEs
WITH
high_earners AS (
    SELECT emp_id, first_name, last_name, salary
    FROM employees
    WHERE salary > 50000
),
dept_names AS (
    SELECT e.emp_id, d.dept_name
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
)
SELECT
    h.first_name,
    h.last_name,
    h.salary,
    dn.dept_name
FROM high_earners h
LEFT JOIN dept_names dn ON h.emp_id = dn.emp_id;

-- Why: A CTE can be referenced multiple times in the main query, avoiding
-- duplicate subquery logic. PostgreSQL 12+ inlines CTEs by default (treating
-- them like subqueries), so there is usually no performance penalty.
WITH emp_summary AS (
    SELECT
        dept_id,
        COUNT(*) AS emp_count,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY dept_id
)
SELECT
    'Total Departments' AS metric,
    COUNT(*) AS value
FROM emp_summary
UNION ALL
SELECT
    'Avg Employees per Dept',
    ROUND(AVG(emp_count), 2)
FROM emp_summary
UNION ALL
SELECT
    'Overall Avg Salary',
    ROUND(AVG(avg_salary), 2)
FROM emp_summary;

-- =============================================================================
-- 8. Recursive CTE
-- =============================================================================

-- Generate number sequence
WITH RECURSIVE numbers AS (
    -- Base case
    SELECT 1 AS n
    UNION ALL
    -- Recursive case
    SELECT n + 1
    FROM numbers
    WHERE n < 10
)
SELECT n FROM numbers;

-- Why: Recursive CTEs are the SQL-standard way to traverse hierarchical data
-- (trees/graphs) without knowing the depth upfront. The ARRAY path column
-- prevents infinite loops and enables proper ordering of the hierarchy.
WITH RECURSIVE org_chart AS (
    -- Base case: Top-level managers (manager_id is NULL)
    SELECT
        emp_id,
        first_name || ' ' || last_name AS name,
        manager_id,
        1 AS level,
        ARRAY[emp_id] AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: Subordinate employees
    SELECT
        e.emp_id,
        e.first_name || ' ' || e.last_name,
        e.manager_id,
        oc.level + 1,
        oc.path || e.emp_id
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.emp_id
)
SELECT
    REPEAT('  ', level - 1) || name AS employee_hierarchy,
    level
FROM org_chart
ORDER BY path;

-- Generate date series
WITH RECURSIVE date_series AS (
    SELECT DATE '2024-01-01' AS date
    UNION ALL
    SELECT date + INTERVAL '1 day'
    FROM date_series
    WHERE date < DATE '2024-01-10'
)
SELECT date FROM date_series;

-- =============================================================================
-- 9. LATERAL JOIN (Alternative to Correlated Subqueries)
-- =============================================================================

-- Why: LATERAL is the key enabler for "top-N per group" queries. Unlike a regular
-- subquery, LATERAL can reference columns from preceding tables (here d.dept_id),
-- making the LIMIT apply independently for each department. This is far more
-- efficient than window functions when N is small.
SELECT d.dept_name, top_employees.*
FROM departments d
CROSS JOIN LATERAL (
    SELECT first_name, last_name, salary
    FROM employees e
    WHERE e.dept_id = d.dept_id
    ORDER BY salary DESC
    LIMIT 2
) AS top_employees;

-- =============================================================================
-- 10. Subquery vs CTE vs LATERAL Comparison
-- =============================================================================

-- Method 1: Subquery (inline view)
SELECT *
FROM (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) sub
WHERE avg_sal > 50000;

-- Method 2: CTE (more readable)
WITH dept_avg AS (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
)
SELECT * FROM dept_avg WHERE avg_sal > 50000;

-- Method 3: LATERAL (when per-row subquery is needed)
SELECT d.dept_name, stats.avg_salary
FROM departments d
CROSS JOIN LATERAL (
    SELECT ROUND(AVG(salary), 2) AS avg_salary
    FROM employees e
    WHERE e.dept_id = d.dept_id
) stats
WHERE stats.avg_salary > 50000;

-- =============================================================================
-- Subquery and CTE Summary
-- =============================================================================
/*
Subquery Placement:
- SELECT: Scalar subquery (single value)
- FROM: Inline view (used as a table)
- WHERE: Used in conditions

Subquery Operators:
- =, <, >: Scalar comparison
- IN, NOT IN: List membership
- EXISTS, NOT EXISTS: Existence check
- ANY, ALL: Conditional comparison

CTE Advantages:
- Improved readability
- Reusable within the query
- Supports recursive queries
- Execution plan optimization hint (MATERIALIZED)

LATERAL:
- Executes correlated subquery per row
- Useful for TOP-N per group problems
*/
