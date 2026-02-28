-- Exercises for Lesson 20: Security and Access Control
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: Multi-Tenant Application ===
-- Problem: Set up Row-Level Security (RLS) for a SaaS application
-- with tenant isolation and admin bypass.

-- Solution:

-- Multi-tenant table: all tenants share one table, but each can
-- only see/modify their own rows via RLS policies.
CREATE TABLE tenant_orders (
    id SERIAL PRIMARY KEY,
    tenant_id INT NOT NULL,
    product TEXT NOT NULL,
    amount NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Enable RLS on the table.
-- FORCE ensures RLS applies even to the table owner (important for testing).
ALTER TABLE tenant_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_orders FORCE ROW LEVEL SECURITY;

-- Tenant isolation policy: each tenant can only see/modify their own rows.
-- The app sets 'app.tenant_id' via SET before each request.
-- USING controls which rows are visible (SELECT, UPDATE, DELETE).
-- WITH CHECK controls which rows can be written (INSERT, UPDATE).
CREATE POLICY tenant_orders_isolation ON tenant_orders
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::INT)
    WITH CHECK (tenant_id = current_setting('app.tenant_id')::INT);

-- Admin bypass: members of admin_group can see all rows unconditionally.
-- USING (TRUE) means no row filtering is applied.
CREATE POLICY tenant_orders_admin ON tenant_orders
    FOR ALL
    TO admin_group
    USING (TRUE);

-- Test tenant isolation
SET app.tenant_id = '1';
INSERT INTO tenant_orders (tenant_id, product, amount) VALUES
    (1, 'Widget A', 29.99),
    (1, 'Widget B', 49.99);

-- This should only show tenant 1's orders
SELECT * FROM tenant_orders;

-- Attempt to insert for a different tenant — WITH CHECK blocks this
-- INSERT INTO tenant_orders (tenant_id, product, amount) VALUES (2, 'Gadget', 19.99);
-- ERROR: new row violates row-level security policy

-- Switch to tenant 2
SET app.tenant_id = '2';
INSERT INTO tenant_orders (tenant_id, product, amount) VALUES (2, 'Gadget', 19.99);

-- Only sees tenant 2's orders
SELECT * FROM tenant_orders;


-- === Exercise 2: Audit System ===
-- Problem: Create a comprehensive audit system for the employees table.

-- Solution:

-- Audit table captures who changed what, when, and from where.
-- old_values/new_values as JSONB preserve the complete row snapshots.
CREATE TABLE employee_audit (
    id BIGSERIAL PRIMARY KEY,
    operation TEXT NOT NULL,
    employee_id INT,
    old_values JSONB,
    new_values JSONB,
    changed_fields TEXT[],
    user_name TEXT DEFAULT current_user,
    client_ip INET DEFAULT inet_client_addr(),
    occurred_at TIMESTAMP DEFAULT NOW()
);

-- The trigger function handles INSERT, UPDATE, and DELETE.
-- For UPDATE operations, it identifies which specific columns changed
-- by comparing OLD and NEW values field by field.
CREATE OR REPLACE FUNCTION employee_audit_trigger()
RETURNS TRIGGER AS $$
DECLARE
    changed TEXT[] := '{}';
    old_val TEXT;
    new_val TEXT;
    col RECORD;
BEGIN
    -- For UPDATE: identify which columns actually changed
    IF TG_OP = 'UPDATE' THEN
        FOR col IN
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = TG_TABLE_NAME
              AND table_schema = TG_TABLE_SCHEMA
        LOOP
            EXECUTE format('SELECT ($1).%I::text', col.column_name) INTO old_val USING OLD;
            EXECUTE format('SELECT ($1).%I::text', col.column_name) INTO new_val USING NEW;
            IF old_val IS DISTINCT FROM new_val THEN
                changed := array_append(changed, col.column_name);
            END IF;
        END LOOP;
    END IF;

    INSERT INTO employee_audit (
        operation, employee_id, old_values, new_values, changed_fields
    )
    VALUES (
        TG_OP,
        COALESCE(NEW.id, OLD.id),
        CASE WHEN TG_OP != 'INSERT' THEN to_jsonb(OLD) END,
        CASE WHEN TG_OP != 'DELETE' THEN to_jsonb(NEW) END,
        CASE WHEN TG_OP = 'UPDATE' THEN changed ELSE NULL END
    );

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to employees table
CREATE TRIGGER trg_employee_audit
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW EXECUTE FUNCTION employee_audit_trigger();

-- Test the audit trail
-- INSERT INTO employees (name, department_id, salary) VALUES ('Test User', 1, 50000);
-- UPDATE employees SET salary = 55000 WHERE name = 'Test User';
-- DELETE FROM employees WHERE name = 'Test User';

-- Query the audit log
SELECT
    operation,
    employee_id,
    changed_fields,
    old_values->>'salary' AS old_salary,
    new_values->>'salary' AS new_salary,
    user_name,
    occurred_at
FROM employee_audit
ORDER BY occurred_at DESC;


-- === Exercise 3: Role Hierarchy ===
-- Problem: Create a role hierarchy for a web application with three access tiers.

-- Solution:

-- Step 1: Create base roles (NOLOGIN = group roles, not user accounts)
-- These define permission sets, not actual login credentials.
CREATE ROLE web_anonymous NOLOGIN;
CREATE ROLE web_user NOLOGIN;
CREATE ROLE web_admin NOLOGIN;

-- Step 2: Build inheritance chain
-- web_user inherits all web_anonymous privileges (read public data)
-- web_admin inherits all web_user privileges (read + write)
-- This creates a cumulative permission model: admin > user > anonymous
GRANT web_anonymous TO web_user;
GRANT web_user TO web_admin;

-- Step 3: Assign privileges to each tier

-- Anonymous (public visitors): can browse products and categories
GRANT USAGE ON SCHEMA public TO web_anonymous;
GRANT SELECT ON public.products, public.categories TO web_anonymous;

-- User (logged-in customers): can place orders and write reviews
-- Also needs sequence access for serial columns to auto-increment
GRANT SELECT, INSERT, UPDATE ON public.orders TO web_user;
GRANT SELECT, INSERT ON public.reviews TO web_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO web_user;

-- Admin: full access to all objects
GRANT ALL ON ALL TABLES IN SCHEMA public TO web_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO web_admin;

-- Step 4: Create actual login roles that inherit from the group roles
-- IN ROLE assigns the specified group role at creation time
CREATE ROLE api_anon LOGIN PASSWORD 'anon_secure_password' IN ROLE web_anonymous;
CREATE ROLE api_user LOGIN PASSWORD 'user_secure_password' IN ROLE web_user;
CREATE ROLE api_admin LOGIN PASSWORD 'admin_secure_password' IN ROLE web_admin;

-- Step 5: Future-proof with default privileges
-- Ensures new tables/sequences automatically get the right grants
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO web_anonymous;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE ON TABLES TO web_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE ON SEQUENCES TO web_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON TABLES TO web_admin;

-- Verify the hierarchy
SELECT
    r.rolname,
    r.rolcanlogin,
    ARRAY(
        SELECT b.rolname
        FROM pg_auth_members m
        JOIN pg_roles b ON m.roleid = b.oid
        WHERE m.member = r.oid
    ) AS member_of
FROM pg_roles r
WHERE r.rolname LIKE 'web_%' OR r.rolname LIKE 'api_%'
ORDER BY r.rolname;
