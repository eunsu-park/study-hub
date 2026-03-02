-- =============================================================================
-- PostgreSQL CRUD Basic Examples
-- Basic CRUD Operations (Create, Read, Update, Delete)
-- =============================================================================

-- This file demonstrates basic CRUD operations in PostgreSQL.
-- Connect to your database before running: psql -U postgres -d your_database

-- =============================================================================
-- 1. CREATE - Table Creation and Data Insertion
-- =============================================================================

-- Create tables
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS departments CASCADE;

-- Departments table
CREATE TABLE departments (
    dept_id SERIAL PRIMARY KEY,
    dept_name VARCHAR(50) NOT NULL UNIQUE,
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Employees table
CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE DEFAULT CURRENT_DATE,
    salary NUMERIC(10, 2) CHECK (salary > 0),
    dept_id INTEGER REFERENCES departments(dept_id),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Why: Creating indexes on FK columns (dept_id) and frequently-queried columns (email)
-- upfront because JOIN and WHERE lookups on these columns would otherwise trigger
-- full sequential scans as the table grows.
CREATE INDEX idx_employees_dept ON employees(dept_id);
CREATE INDEX idx_employees_email ON employees(email);

-- =============================================================================
-- INSERT - Data Insertion
-- =============================================================================

-- Single row insert
INSERT INTO departments (dept_name, location)
VALUES ('Engineering', 'Seoul');

INSERT INTO departments (dept_name, location)
VALUES ('Marketing', 'Busan');

INSERT INTO departments (dept_name, location)
VALUES ('Sales', 'Daegu');

INSERT INTO departments (dept_name, location)
VALUES ('HR', 'Seoul');

-- Insert multiple rows at once
INSERT INTO employees (first_name, last_name, email, hire_date, salary, dept_id)
VALUES
    ('John', 'Kim', 'kim.cs@company.com', '2020-01-15', 50000, 1),
    ('Sarah', 'Lee', 'lee.yh@company.com', '2019-06-20', 55000, 1),
    ('Mike', 'Park', 'park.ms@company.com', '2021-03-10', 48000, 2),
    ('Emily', 'Jung', 'jung.sj@company.com', '2018-11-05', 62000, 1),
    ('David', 'Choi', 'choi.dw@company.com', '2022-08-01', 45000, 3),
    ('Laura', 'Kang', 'kang.my@company.com', '2020-05-15', 52000, 2),
    ('James', 'Cho', 'cho.jh@company.com', '2019-09-20', 58000, 3),
    ('Anna', 'Yoon', 'yoon.sy@company.com', '2021-12-01', 47000, 4);

-- Why: RETURNING avoids a separate SELECT after INSERT to get the auto-generated
-- emp_id. This is a single round-trip instead of two, which is especially valuable
-- in application code where you need the new ID immediately.
INSERT INTO employees (first_name, last_name, email, salary, dept_id)
VALUES ('New', 'Employee', 'test@company.com', 40000, 1)
RETURNING emp_id, first_name, last_name;

-- =============================================================================
-- 2. READ - Data Retrieval
-- =============================================================================

-- Select all
SELECT * FROM employees;

-- Select specific columns
SELECT first_name, last_name, email, salary
FROM employees;

-- Conditional query (WHERE)
SELECT first_name, last_name, salary
FROM employees
WHERE salary > 50000;

-- Multiple conditions (AND, OR)
SELECT *
FROM employees
WHERE dept_id = 1 AND salary > 50000;

SELECT *
FROM employees
WHERE dept_id = 1 OR salary > 55000;

-- BETWEEN, IN, LIKE
SELECT first_name, last_name, salary
FROM employees
WHERE salary BETWEEN 45000 AND 55000;

SELECT first_name, last_name, dept_id
FROM employees
WHERE dept_id IN (1, 2);

SELECT first_name, last_name, email
FROM employees
WHERE email LIKE '%@company.com';

-- NULL check
SELECT *
FROM employees
WHERE email IS NOT NULL;

-- Sorting (ORDER BY)
SELECT first_name, last_name, salary
FROM employees
ORDER BY salary DESC;

SELECT first_name, last_name, dept_id, salary
FROM employees
ORDER BY dept_id ASC, salary DESC;

-- Limit (LIMIT, OFFSET)
SELECT first_name, last_name, salary
FROM employees
ORDER BY salary DESC
LIMIT 5;

-- Why: OFFSET-based pagination is simple but gets slower as OFFSET grows (PostgreSQL
-- must scan and discard all skipped rows). For deep pages, consider keyset pagination
-- (WHERE emp_id > last_seen_id ORDER BY emp_id LIMIT 3) instead.
SELECT first_name, last_name, salary
FROM employees
ORDER BY emp_id
LIMIT 3 OFFSET 3;

-- DISTINCT - Remove duplicates
SELECT DISTINCT dept_id
FROM employees;

-- Aliases (AS)
SELECT
    first_name AS "First Name",
    last_name AS "Last Name",
    salary AS "Salary"
FROM employees;

-- Computed columns
SELECT
    first_name,
    last_name,
    salary,
    salary * 12 AS annual_salary,
    salary * 1.1 AS after_raise
FROM employees;

-- =============================================================================
-- 3. UPDATE - Data Modification
-- =============================================================================

-- Single row update
UPDATE employees
SET salary = 52000
WHERE emp_id = 1;

-- Multiple column update
UPDATE employees
SET salary = 55000, updated_at = CURRENT_TIMESTAMP
WHERE emp_id = 1;

-- Conditional bulk update
UPDATE employees
SET salary = salary * 1.1
WHERE dept_id = 1;

-- Why: Using a subquery to resolve dept_name->dept_id keeps the update decoupled
-- from magic numbers. If department IDs change, the query still works correctly
-- as long as the name is unique.
UPDATE employees
SET salary = salary * 1.05
WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = 'Engineering');

-- Verify modified data with RETURNING
UPDATE employees
SET salary = salary * 1.02
WHERE emp_id = 3
RETURNING emp_id, first_name, salary;

-- =============================================================================
-- 4. DELETE - Data Deletion
-- =============================================================================

-- Conditional delete
DELETE FROM employees
WHERE emp_id = 9;

-- Delete all (use with caution!)
-- DELETE FROM employees;

-- Why: TRUNCATE is O(1) regardless of table size (it deallocates data pages directly),
-- while DELETE scans every row and writes WAL for each. Use TRUNCATE for bulk
-- cleanup, but note it acquires ACCESS EXCLUSIVE lock and cannot be rolled back
-- in some contexts.
-- TRUNCATE TABLE employees RESTART IDENTITY;

-- Verify deleted data with RETURNING
DELETE FROM employees
WHERE email = 'test@company.com'
RETURNING *;

-- =============================================================================
-- 5. Transaction
-- =============================================================================

-- Begin transaction
BEGIN;

INSERT INTO employees (first_name, last_name, email, salary, dept_id)
VALUES ('Transaction', 'Test', 'trans@company.com', 45000, 1);

UPDATE employees
SET salary = salary * 1.05
WHERE email = 'trans@company.com';

-- Verify
SELECT * FROM employees WHERE email = 'trans@company.com';

-- Commit or rollback
COMMIT;
-- or ROLLBACK;

-- =============================================================================
-- 6. UPSERT (INSERT ... ON CONFLICT)
-- =============================================================================

-- Ignore on duplicate
INSERT INTO departments (dept_name, location)
VALUES ('Engineering', 'Incheon')
ON CONFLICT (dept_name) DO NOTHING;

-- Why: UPSERT (INSERT ON CONFLICT DO UPDATE) is atomic — it avoids the race condition
-- of checking existence then inserting separately. EXCLUDED refers to the row that
-- would have been inserted, letting us merge new values cleanly.
INSERT INTO departments (dept_name, location)
VALUES ('Engineering', 'Incheon')
ON CONFLICT (dept_name)
DO UPDATE SET location = EXCLUDED.location;

-- =============================================================================
-- 7. Table Inspection and Information
-- =============================================================================

-- View table structure
\d employees

-- List tables
\dt

-- View indexes
\di

-- =============================================================================
-- Cleanup (if needed)
-- =============================================================================
-- DROP TABLE IF EXISTS employees CASCADE;
-- DROP TABLE IF EXISTS departments CASCADE;
