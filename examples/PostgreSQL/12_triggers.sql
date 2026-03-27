-- ============================================================================
-- PostgreSQL Triggers
-- ============================================================================
-- Demonstrates:
--   - BEFORE / AFTER triggers
--   - Row-level vs statement-level triggers
--   - INSERT / UPDATE / DELETE trigger events
--   - Trigger functions (NEW, OLD, TG_OP)
--   - Audit logging with triggers
--   - Conditional triggers (WHEN clause)
--   - Event triggers (DDL)
--
-- Prerequisites: PostgreSQL 12+
-- Usage: psql -U postgres -d your_database -f 12_triggers.sql
-- ============================================================================

-- Clean up
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS employee_audit CASCADE;
DROP TABLE IF EXISTS salary_changes CASCADE;
DROP TABLE IF EXISTS inventory CASCADE;
DROP TABLE IF EXISTS inventory_snapshots CASCADE;

-- ============================================================================
-- Setup
-- ============================================================================

CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary NUMERIC(10, 2) NOT NULL CHECK (salary > 0),
    email TEXT UNIQUE,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE employee_audit (
    audit_id SERIAL PRIMARY KEY,
    emp_id INTEGER,
    operation TEXT NOT NULL,     -- INSERT, UPDATE, DELETE
    old_values JSONB,
    new_values JSONB,
    changed_by TEXT DEFAULT CURRENT_USER,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE salary_changes (
    change_id SERIAL PRIMARY KEY,
    emp_id INTEGER NOT NULL,
    old_salary NUMERIC(10, 2),
    new_salary NUMERIC(10, 2),
    change_pct NUMERIC(5, 2),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO employees (name, department, salary, email) VALUES
    ('Alice Johnson', 'Engineering', 95000, 'alice@example.com'),
    ('Bob Smith', 'Engineering', 88000, 'bob@example.com'),
    ('Charlie Brown', 'Marketing', 72000, 'charlie@example.com'),
    ('Diana Prince', 'Engineering', 105000, 'diana@example.com'),
    ('Eve Davis', 'Marketing', 68000, 'eve@example.com');

-- ============================================================================
-- 1. BEFORE UPDATE Trigger: Auto-update timestamp
-- ============================================================================

CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Why: A BEFORE trigger can modify the NEW row before it is written to disk.
    -- This is the right place for auto-timestamps because AFTER triggers cannot
    -- change the row. Returning NULL from a BEFORE trigger would silently cancel
    -- the entire INSERT/UPDATE operation.
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_employee_timestamp
    BEFORE UPDATE ON employees
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp();

-- Test: updated_at changes automatically
UPDATE employees SET salary = 97000 WHERE emp_id = 1;
SELECT name, salary, updated_at FROM employees WHERE emp_id = 1;

-- ============================================================================
-- 2. AFTER Trigger: Audit Logging
-- ============================================================================

CREATE OR REPLACE FUNCTION audit_employee_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO employee_audit (emp_id, operation, new_values)
        VALUES (NEW.emp_id, 'INSERT', to_jsonb(NEW));

    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO employee_audit (emp_id, operation, old_values, new_values)
        VALUES (NEW.emp_id, 'UPDATE', to_jsonb(OLD), to_jsonb(NEW));

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO employee_audit (emp_id, operation, old_values)
        VALUES (OLD.emp_id, 'DELETE', to_jsonb(OLD));
    END IF;

    -- Why: AFTER triggers fire only after the row is committed to the table,
    -- making them ideal for audit logging (we know the change succeeded).
    -- The return value is ignored for AFTER triggers, but we return NULL by
    -- convention to signal that we aren't modifying anything.
    RETURN NULL;
END;
$$;

CREATE TRIGGER trg_employee_audit
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW
    EXECUTE FUNCTION audit_employee_changes();

-- Test: make changes and check audit log
INSERT INTO employees (name, department, salary, email)
VALUES ('Frank Wilson', 'Sales', 65000, 'frank@example.com');

UPDATE employees SET department = 'Engineering' WHERE name = 'Frank Wilson';

DELETE FROM employees WHERE name = 'Frank Wilson';

SELECT operation, emp_id, changed_at
FROM employee_audit
ORDER BY audit_id;

-- ============================================================================
-- 3. Conditional Trigger (WHEN clause): Salary Change Tracking
-- ============================================================================

CREATE OR REPLACE FUNCTION track_salary_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO salary_changes (emp_id, old_salary, new_salary, change_pct)
    VALUES (
        NEW.emp_id,
        OLD.salary,
        NEW.salary,
        ROUND(((NEW.salary - OLD.salary) / OLD.salary) * 100, 2)
    );
    RETURN NEW;
END;
$$;

-- Why: The WHEN clause prevents the trigger from firing on non-salary updates
-- (e.g., department changes). IS DISTINCT FROM handles NULLs correctly — unlike
-- "!=", it treats NULL = NULL as true, avoiding false positives when both
-- old and new values are NULL.
CREATE TRIGGER trg_salary_change
    AFTER UPDATE ON employees
    FOR EACH ROW
    WHEN (OLD.salary IS DISTINCT FROM NEW.salary)
    EXECUTE FUNCTION track_salary_change();

-- Test: salary change fires trigger
UPDATE employees SET salary = 100000 WHERE emp_id = 1;

-- Non-salary change does NOT fire trigger
UPDATE employees SET department = 'Senior Engineering' WHERE emp_id = 1;

SELECT * FROM salary_changes;

-- ============================================================================
-- 4. BEFORE INSERT Trigger: Data Validation & Normalization
-- ============================================================================

CREATE OR REPLACE FUNCTION normalize_employee()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Normalize email to lowercase
    NEW.email = LOWER(TRIM(NEW.email));

    -- Capitalize name
    NEW.name = INITCAP(TRIM(NEW.name));

    -- Enforce minimum salary
    IF NEW.salary < 30000 THEN
        RAISE EXCEPTION 'Salary % is below minimum (30000)', NEW.salary;
    END IF;

    -- Enforce maximum salary raise (for updates)
    IF TG_OP = 'UPDATE' AND NEW.salary > OLD.salary * 1.5 THEN
        RAISE WARNING 'Large salary increase: % → % (>50%%)',
                      OLD.salary, NEW.salary;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_normalize_employee
    BEFORE INSERT OR UPDATE ON employees
    FOR EACH ROW
    EXECUTE FUNCTION normalize_employee();

-- Test: email normalized, name capitalized
INSERT INTO employees (name, department, salary, email)
VALUES ('  grace hopper  ', 'Engineering', 95000, '  GRACE@EXAMPLE.COM  ');

SELECT name, email FROM employees WHERE email = 'grace@example.com';

-- ============================================================================
-- 5. Statement-Level Trigger
-- ============================================================================

CREATE TABLE inventory (
    item_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE inventory_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    operation TEXT NOT NULL,
    row_count INTEGER,
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION log_inventory_statement()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_count FROM inventory;
    INSERT INTO inventory_snapshots (operation, row_count)
    VALUES (TG_OP, v_count);
    RETURN NULL;
END;
$$;

-- Why: FOR EACH STATEMENT fires exactly once regardless of how many rows are
-- affected, making it efficient for bulk operations. Use it for summary logging
-- or notification (e.g., "an INSERT happened") where per-row detail is unnecessary
-- and per-row firing would be wasteful.
CREATE TRIGGER trg_inventory_statement
    AFTER INSERT OR UPDATE OR DELETE ON inventory
    FOR EACH STATEMENT
    EXECUTE FUNCTION log_inventory_statement();

-- Test: one trigger fire for multiple rows
INSERT INTO inventory (name, quantity) VALUES
    ('Widget A', 100),
    ('Widget B', 200),
    ('Widget C', 50);

SELECT * FROM inventory_snapshots;  -- One entry despite 3 rows

-- ============================================================================
-- 6. Trigger Execution Order
-- ============================================================================
-- When multiple triggers exist on the same table:
-- 1. BEFORE statement triggers
-- 2. BEFORE row triggers (alphabetical by trigger name)
-- 3. The actual DML operation
-- 4. AFTER row triggers (alphabetical by trigger name)
-- 5. AFTER statement triggers

-- ============================================================================
-- 7. Useful Trigger Variables
-- ============================================================================
-- Inside trigger functions, these special variables are available:
--
-- NEW:          New row (INSERT/UPDATE). NULL for DELETE.
-- OLD:          Old row (UPDATE/DELETE). NULL for INSERT.
-- TG_OP:        'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE'
-- TG_NAME:      Trigger name
-- TG_TABLE_NAME: Table name
-- TG_WHEN:      'BEFORE', 'AFTER', 'INSTEAD OF'
-- TG_LEVEL:     'ROW', 'STATEMENT'
-- TG_NARGS:     Number of trigger arguments
-- TG_ARGV[]:    Trigger argument array

-- ============================================================================
-- 8. Managing Triggers
-- ============================================================================

-- Why: Disabling triggers is essential during bulk data loads (e.g., migrations)
-- where per-row trigger overhead would make the operation orders of magnitude
-- slower. Always re-enable immediately after to avoid missing audit records.
ALTER TABLE employees DISABLE TRIGGER trg_employee_audit;

-- Re-enable
ALTER TABLE employees ENABLE TRIGGER trg_employee_audit;

-- Disable ALL triggers on a table
-- ALTER TABLE employees DISABLE TRIGGER ALL;

-- List all triggers on a table
SELECT
    tgname AS trigger_name,
    CASE tgtype & 2  WHEN 2 THEN 'BEFORE' ELSE 'AFTER' END AS timing,
    CASE tgtype & 28
        WHEN 4  THEN 'INSERT'
        WHEN 8  THEN 'DELETE'
        WHEN 16 THEN 'UPDATE'
        WHEN 20 THEN 'INSERT OR UPDATE'
        WHEN 28 THEN 'INSERT OR UPDATE OR DELETE'
        ELSE 'OTHER'
    END AS events,
    CASE tgtype & 1 WHEN 1 THEN 'ROW' ELSE 'STATEMENT' END AS level
FROM pg_trigger
WHERE tgrelid = 'employees'::regclass
  AND NOT tgisinternal
ORDER BY tgname;
