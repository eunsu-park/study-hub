-- Exercises for Lesson 08: Subqueries and CTE
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- Setup: Create departments and employees tables
CREATE TABLE IF NOT EXISTS departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    salary NUMERIC(10, 2),
    hire_date DATE
);

TRUNCATE TABLE employees RESTART IDENTITY CASCADE;
TRUNCATE TABLE departments RESTART IDENTITY CASCADE;

INSERT INTO departments (name) VALUES
('Development'), ('Marketing'), ('HR'), ('Finance');

INSERT INTO employees (name, department_id, salary, hire_date) VALUES
('Kim Dev', 1, 5000000, '2020-03-15'),
('Lee Dev', 1, 4500000, '2021-06-20'),
('Park Marketing', 2, 4000000, '2019-11-10'),
('Choi Marketing', 2, 3800000, '2022-01-05'),
('Jung HR', 3, 3500000, '2020-08-25'),
('Han Finance', 4, 4200000, '2021-03-10'),
('Oh Finance', 4, 3900000, '2022-07-15');


-- === Exercise 1: WHERE Subqueries ===
-- Problem: Use scalar and IN subqueries for filtering.

-- Solution:

-- 1. Employees with salary higher than the overall average
-- The subquery returns a single scalar value used for comparison
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- 2. Most recently hired employee
SELECT * FROM employees
WHERE hire_date = (SELECT MAX(hire_date) FROM employees);

-- 3. Employees in Development or Marketing departments
-- IN subquery returns a set of department IDs to match against
SELECT * FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE name IN ('Development', 'Marketing')
);


-- === Exercise 2: Correlated Subqueries ===
-- Problem: Use correlated subqueries that reference the outer query.

-- Solution:

-- 1. Employees earning more than their department's average
-- The inner query re-executes for each row of the outer query,
-- filtering by the current employee's department_id
SELECT
    e.name,
    e.salary,
    d.name AS department
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department_id = e.department_id
);

-- 2. Highest paid employee in each department
-- For each employee, check if their salary equals the max in their dept
SELECT * FROM employees e
WHERE salary = (
    SELECT MAX(salary)
    FROM employees
    WHERE department_id = e.department_id
);


-- === Exercise 3: CTE Usage ===
-- Problem: Use CTEs to build readable, multi-step queries.

-- Solution:

-- 1. Employee info enriched with department-level statistics
-- The CTE pre-computes stats once; the main query joins to it
WITH dept_stats AS (
    SELECT
        department_id,
        AVG(salary) AS avg_salary,
        COUNT(*) AS emp_count
    FROM employees
    GROUP BY department_id
)
SELECT
    e.name,
    e.salary,
    d.name AS department,
    ROUND(ds.avg_salary, 0) AS dept_avg,
    ds.emp_count AS dept_count
FROM employees e
JOIN departments d ON e.department_id = d.id
JOIN dept_stats ds ON e.department_id = ds.department_id;

-- 2. Salary ranking (overall and within department)
-- Window functions inside a CTE keep the main SELECT clean
WITH ranked_employees AS (
    SELECT
        *,
        RANK() OVER (ORDER BY salary DESC) AS salary_rank,
        RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
    FROM employees
)
SELECT
    name,
    salary,
    salary_rank AS overall_rank,
    dept_rank
FROM ranked_employees
ORDER BY salary_rank;


-- === Exercise 4: Complex Multi-CTE Usage ===
-- Problem: Chain multiple CTEs to find employees above their department
-- average and show the difference.

-- Solution:

-- Step 1 CTE: compute each department's average salary
-- Step 2 CTE: join employees with their dept average, filter above-average only
-- Main query: add department name and format output
WITH
dept_avg AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
),
above_avg AS (
    SELECT
        e.*,
        da.avg_salary,
        e.salary - da.avg_salary AS diff
    FROM employees e
    JOIN dept_avg da ON e.department_id = da.department_id
    WHERE e.salary >= da.avg_salary
)
SELECT
    aa.name,
    d.name AS department,
    aa.salary,
    ROUND(aa.avg_salary, 0) AS dept_avg,
    ROUND(aa.diff, 0) AS above_avg_by
FROM above_avg aa
JOIN departments d ON aa.department_id = d.id
ORDER BY aa.diff DESC;
