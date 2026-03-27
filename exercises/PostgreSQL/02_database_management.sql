-- Exercises for Lesson 02: Database Management
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- === Exercise 1: Project Database Setup ===
-- Problem: Create a project database with app and read-only users,
-- schemas, and appropriate permissions following least privilege.

-- Solution:

-- 1. Create database
CREATE DATABASE project_db;

-- 2. Switch database
-- \c project_db

-- 3. Create application user (full access to app schema)
CREATE USER app_user WITH PASSWORD 'app_password';

-- 4. Create read-only user (SELECT only)
CREATE USER readonly_user WITH PASSWORD 'readonly_password';

-- 5. Create schemas to logically separate application and reporting objects
CREATE SCHEMA app_schema;
CREATE SCHEMA report_schema;

-- 6. Set permissions
-- app_user: full privileges on the database and app schema
GRANT ALL PRIVILEGES ON DATABASE project_db TO app_user;
GRANT ALL PRIVILEGES ON SCHEMA app_schema TO app_user;

-- readonly_user: can connect and read from app_schema only
GRANT CONNECT ON DATABASE project_db TO readonly_user;
GRANT USAGE ON SCHEMA app_schema TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA app_schema TO readonly_user;

-- 7. Ensure future tables in app_schema are also readable by readonly_user
-- Without this, every new table would require a manual GRANT
ALTER DEFAULT PRIVILEGES IN SCHEMA app_schema
GRANT SELECT ON TABLES TO readonly_user;


-- === Exercise 2: Test User Permissions ===
-- Problem: Create a table as admin, verify readonly_user can SELECT
-- but not INSERT.

-- Solution:

-- Create table in app_schema (run as postgres/admin)
CREATE TABLE app_schema.products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    price NUMERIC(10,2)
);

INSERT INTO app_schema.products (name, price) VALUES
('Laptop', 1500.00),
('Mouse', 35.00);

-- As readonly_user (connect: psql -U readonly_user -d project_db):
-- This should succeed:
SELECT * FROM app_schema.products;

-- This should fail with "permission denied for table products":
-- INSERT INTO app_schema.products (name, price) VALUES ('Keyboard', 80.00);


-- === Exercise 3: Query Database Information ===
-- Problem: Query all database sizes, current connections, and role summaries.

-- Solution:

-- All database sizes (excluding templates)
SELECT
    datname AS database,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
WHERE datistemplate = false
ORDER BY pg_database_size(datname) DESC;

-- Current connection info for this database
SELECT
    pid,
    usename,
    datname,
    client_addr,
    state,
    LEFT(query, 80) AS query_preview
FROM pg_stat_activity
WHERE datname = current_database();

-- Role permissions summary (exclude internal pg_ roles)
SELECT
    r.rolname,
    r.rolsuper AS superuser,
    r.rolcreatedb AS can_create_db,
    r.rolcreaterole AS can_create_role,
    r.rolcanlogin AS can_login
FROM pg_roles r
WHERE r.rolname NOT LIKE 'pg_%'
ORDER BY r.rolname;
