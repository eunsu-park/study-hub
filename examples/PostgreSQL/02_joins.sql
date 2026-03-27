-- =============================================================================
-- PostgreSQL JOIN Examples
-- Various Types of JOINs in PostgreSQL
-- =============================================================================

-- First, run 01_basic_crud.sql to create tables and data.

-- =============================================================================
-- Additional Test Data
-- =============================================================================

-- Add an employee without a department (NULL dept_id)
INSERT INTO employees (first_name, last_name, email, salary, dept_id)
VALUES ('Unassigned', 'Employee', 'nodept@company.com', 40000, NULL);

-- Add a department with no employees
INSERT INTO departments (dept_name, location)
VALUES ('Finance', 'Seoul');

-- =============================================================================
-- 1. INNER JOIN
-- =============================================================================
-- Returns only matching rows from both tables

SELECT
    e.emp_id,
    e.first_name,
    e.last_name,
    e.salary,
    d.dept_name,
    d.location
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- Without table aliases
SELECT
    employees.first_name,
    employees.last_name,
    departments.dept_name
FROM employees
INNER JOIN departments ON employees.dept_id = departments.dept_id;

-- =============================================================================
-- 2. LEFT JOIN (LEFT OUTER JOIN)
-- =============================================================================
-- All rows from the left table + matching rows from the right table

SELECT
    e.emp_id,
    e.first_name,
    e.last_name,
    e.salary,
    d.dept_name,
    d.location
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- Why: This is the "anti-join" pattern — LEFT JOIN + WHERE IS NULL finds rows
-- in the left table that have NO match in the right table. It is typically
-- more readable (and sometimes faster) than NOT EXISTS or NOT IN.
SELECT
    e.first_name,
    e.last_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
WHERE d.dept_id IS NULL;

-- =============================================================================
-- 3. RIGHT JOIN (RIGHT OUTER JOIN)
-- =============================================================================
-- All rows from the right table + matching rows from the left table

SELECT
    e.emp_id,
    e.first_name,
    e.last_name,
    d.dept_name,
    d.location
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- Only departments with no employees
SELECT
    d.dept_name,
    d.location
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id
WHERE e.emp_id IS NULL;

-- =============================================================================
-- 4. FULL OUTER JOIN
-- =============================================================================
-- All rows from both tables (NULL where no match)

SELECT
    e.emp_id,
    e.first_name,
    e.last_name,
    d.dept_name,
    d.location
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;

-- Only unmatched rows
SELECT
    e.emp_id,
    e.first_name,
    d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id
WHERE e.emp_id IS NULL OR d.dept_id IS NULL;

-- =============================================================================
-- 5. CROSS JOIN (Cartesian Product)
-- =============================================================================
-- All possible combinations

SELECT
    e.first_name,
    d.dept_name
FROM employees e
CROSS JOIN departments d
LIMIT 20;

-- CROSS JOIN can also be expressed with a comma (no ON clause)
SELECT
    e.first_name,
    d.dept_name
FROM employees e, departments d
WHERE e.dept_id IS NOT NULL
LIMIT 20;

-- =============================================================================
-- 6. SELF JOIN
-- =============================================================================
-- Joining a table with itself

-- Add a manager column for the example
ALTER TABLE employees ADD COLUMN IF NOT EXISTS manager_id INTEGER REFERENCES employees(emp_id);

-- Assign managers to some employees
UPDATE employees SET manager_id = 4 WHERE emp_id IN (1, 2);
UPDATE employees SET manager_id = 1 WHERE emp_id IN (5, 6);

-- Why: A self-join is the only way to resolve hierarchical relationships stored
-- within a single table. LEFT JOIN (not INNER) ensures top-level employees
-- with no manager (manager_id IS NULL) still appear in the result.
SELECT
    e.emp_id,
    e.first_name || ' ' || e.last_name AS employee_name,
    m.first_name || ' ' || m.last_name AS manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;

-- =============================================================================
-- 7. Multi-Table JOIN
-- =============================================================================

-- Create projects table
DROP TABLE IF EXISTS projects CASCADE;
DROP TABLE IF EXISTS employee_projects CASCADE;

CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    budget NUMERIC(12, 2)
);

-- Why: A junction (bridge) table is the standard way to model many-to-many
-- relationships in relational databases. The composite primary key (emp_id, project_id)
-- enforces that each employee-project pair is unique, and doubles as an index
-- for lookups from either direction.
CREATE TABLE employee_projects (
    emp_id INTEGER REFERENCES employees(emp_id),
    project_id INTEGER REFERENCES projects(project_id),
    role VARCHAR(50),
    PRIMARY KEY (emp_id, project_id)
);

-- Insert data
INSERT INTO projects (project_name, start_date, end_date, budget)
VALUES
    ('Website Redesign', '2024-01-01', '2024-06-30', 100000),
    ('Mobile App Development', '2024-03-01', '2024-12-31', 200000),
    ('Data Analytics Platform', '2024-02-01', '2024-08-31', 150000);

INSERT INTO employee_projects (emp_id, project_id, role)
VALUES
    (1, 1, 'Lead'),
    (2, 1, 'Developer'),
    (1, 2, 'Developer'),
    (3, 2, 'Lead'),
    (4, 3, 'Lead'),
    (2, 3, 'Analyst');

-- Join 3 tables: employees + departments + projects
SELECT
    e.first_name || ' ' || e.last_name AS employee_name,
    d.dept_name,
    p.project_name,
    ep.role
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
JOIN employee_projects ep ON e.emp_id = ep.emp_id
JOIN projects p ON ep.project_id = p.project_id
ORDER BY e.first_name, p.project_name;

-- =============================================================================
-- 8. NATURAL JOIN
-- =============================================================================
-- Automatically joins on columns with the same name (not recommended - use explicit ON clause)

-- dept_id has the same name, so it auto-matches
-- SELECT * FROM employees NATURAL JOIN departments;

-- =============================================================================
-- 9. USING Clause
-- =============================================================================
-- Used instead of ON when columns have the same name

SELECT
    e.first_name,
    e.last_name,
    d.dept_name
FROM employees e
JOIN departments d USING (dept_id);

-- =============================================================================
-- 10. JOIN + Aggregation
-- =============================================================================

-- Why: LEFT JOIN from departments ensures departments with zero employees still
-- appear (with count=0). Using COUNT(e.emp_id) instead of COUNT(*) correctly
-- returns 0 for empty departments because COUNT of NULL is 0.
SELECT
    d.dept_name,
    COUNT(e.emp_id) AS employee_count,
    ROUND(AVG(e.salary), 2) AS avg_salary,
    MIN(e.salary) AS min_salary,
    MAX(e.salary) AS max_salary
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
ORDER BY employee_count DESC;

-- Number of employees per project
SELECT
    p.project_name,
    COUNT(ep.emp_id) AS member_count,
    p.budget
FROM projects p
LEFT JOIN employee_projects ep ON p.project_id = ep.project_id
GROUP BY p.project_id, p.project_name, p.budget
ORDER BY member_count DESC;

-- =============================================================================
-- 11. JOIN Performance Tips
-- =============================================================================

-- Ensure indexes exist on columns used in joins
-- CREATE INDEX idx_employees_dept ON employees(dept_id);
-- CREATE INDEX idx_employee_projects_emp ON employee_projects(emp_id);
-- CREATE INDEX idx_employee_projects_proj ON employee_projects(project_id);

-- Why: EXPLAIN ANALYZE runs the actual query (not just planning) to show real
-- execution times and row counts. This reveals whether PostgreSQL chose a nested
-- loop, hash join, or merge join — helping you decide if indexes are needed.
EXPLAIN ANALYZE
SELECT
    e.first_name,
    d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- =============================================================================
-- JOIN Type Summary
-- =============================================================================
/*
| JOIN Type       | Description                                       |
|-----------------|---------------------------------------------------|
| INNER JOIN      | Only matching rows from both tables                |
| LEFT JOIN       | All rows from left table + matching from right     |
| RIGHT JOIN      | All rows from right table + matching from left     |
| FULL OUTER JOIN | All rows from both tables                          |
| CROSS JOIN      | All possible combinations (Cartesian product)      |
| SELF JOIN       | Join a table with itself                           |

Tips:
- Always write ON clauses explicitly
- Create indexes on join columns
- Check execution plans with EXPLAIN
- Avoid unnecessary CROSS JOINs
*/
