# Database Management

**Previous**: [PostgreSQL Basics](./01_PostgreSQL_Basics.md) | **Next**: [Tables and Data Types](./03_Tables_and_Data_Types.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the hierarchical structure of a PostgreSQL server (server, database, schema, table)
2. Create, list, rename, and drop databases with appropriate options
3. Explain the role of template databases (`template0`, `template1`)
4. Create and manage roles (users and groups) with specific privileges
5. Apply the GRANT/REVOKE system to control access at the database, schema, and table level
6. Configure schemas to logically organize objects within a database
7. Apply the principle of least privilege when designing a permission model

---

In any production environment, properly managing databases, users, and permissions is just as important as writing correct SQL. A misconfigured role or an overly permissive grant can expose sensitive data or allow accidental deletions. This lesson covers the administrative commands you need to set up a secure, well-organized PostgreSQL environment from day one.

## 1. Database Basic Concepts

In PostgreSQL, a database is the top-level container that holds tables, views, functions, and more.

```
┌─────────────────────────────────────────────────────┐
│                PostgreSQL Server                     │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │   DB 1   │  │   DB 2   │  │   DB 3   │          │
│  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │          │
│  │ │Schema│ │  │ │Schema│ │  │ │Schema│ │          │
│  │ │┌────┐│ │  │ │┌────┐│ │  │ │┌────┐│ │          │
│  │ ││Table│ │  │ ││Table│ │  │ ││Table│ │          │
│  │ │└────┘│ │  │ │└────┘│ │  │ │└────┘│ │          │
│  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

---

## 2. Database Creation

### Basic Creation

```sql
CREATE DATABASE mydb;
```

### Creation with Options

```sql
CREATE DATABASE mydb
    WITH
    OWNER = myuser
    ENCODING = 'UTF8'
    LC_COLLATE = 'ko_KR.UTF-8'
    LC_CTYPE = 'ko_KR.UTF-8'
    TEMPLATE = template0
    CONNECTION LIMIT = 100;
```

### Main Options

| Option | Description |
|--------|-------------|
| `OWNER` | Database owner |
| `ENCODING` | Character encoding (UTF8 recommended) |
| `LC_COLLATE` | Sorting locale |
| `LC_CTYPE` | Character classification locale |
| `TEMPLATE` | Template database |
| `CONNECTION LIMIT` | Maximum concurrent connections (-1 for unlimited) |

### Template Databases

```sql
-- template1 is the default template — any extensions or objects you add to template1
-- will appear in every new database. Customize it for org-wide defaults.
CREATE DATABASE mydb TEMPLATE template1;

-- template0 is the pristine, unmodified template — use it when you need a different
-- encoding or locale, since template1 inherits the cluster's original settings
CREATE DATABASE mydb TEMPLATE template0 ENCODING 'UTF8';
```

---

## 3. Database List and Information

### List Databases

```sql
-- psql meta command
\l

-- Detailed info
\l+

-- SQL query
SELECT datname, datdba, encoding, datcollate
FROM pg_database;
```

### Check Current Database

```sql
SELECT current_database();
```

### Check Database Size

```sql
-- Specific database size
SELECT pg_size_pretty(pg_database_size('mydb'));

-- All database sizes
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;
```

---

## 4. Database Switch and Modification

### Switch Database

```sql
-- psql only
\c mydb

-- Or
\connect mydb
```

### Rename Database

```sql
-- No sessions connected to DB
ALTER DATABASE oldname RENAME TO newname;
```

### Change Database Owner

```sql
ALTER DATABASE mydb OWNER TO newowner;
```

### Change Connection Limit

```sql
ALTER DATABASE mydb CONNECTION LIMIT 50;
```

---

## 5. Database Deletion

```sql
-- Basic deletion
DROP DATABASE mydb;

-- Delete only if exists
DROP DATABASE IF EXISTS mydb;

-- Force deletion (terminate connected sessions)
DROP DATABASE mydb WITH (FORCE);  -- PostgreSQL 13+
```

### Check and Terminate Connected Sessions

```sql
-- Check connected sessions
SELECT pid, usename, application_name, client_addr
FROM pg_stat_activity
WHERE datname = 'mydb';

-- Terminate specific session
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'mydb' AND pid <> pg_backend_pid();
```

---

## 6. User (Role) Management

In PostgreSQL, both users and groups are called "Roles".

### Create Role

```sql
-- Create basic user
CREATE ROLE myuser LOGIN PASSWORD 'mypassword';

-- CREATE USER includes LOGIN by default
CREATE USER myuser WITH PASSWORD 'mypassword';

-- With various options
CREATE ROLE admin_user WITH
    LOGIN
    PASSWORD 'securepassword'
    CREATEDB
    CREATEROLE
    VALID UNTIL '2025-12-31';
```

### Role Options

| Option | Description |
|--------|-------------|
| `LOGIN` | Can login |
| `SUPERUSER` | Superuser privileges |
| `CREATEDB` | Can create databases |
| `CREATEROLE` | Can create roles |
| `INHERIT` | Inherit group privileges |
| `REPLICATION` | Replication privileges |
| `PASSWORD 'xxx'` | Set password |
| `VALID UNTIL 'timestamp'` | Account expiration date |
| `CONNECTION LIMIT n` | Maximum connections |

### List Roles

```sql
-- psql meta command
\du

-- Detailed info
\du+

-- SQL query
SELECT rolname, rolsuper, rolcreatedb, rolcreaterole, rolcanlogin
FROM pg_roles;
```

### Modify Role

```sql
-- Change password
ALTER ROLE myuser WITH PASSWORD 'newpassword';

-- Add privilege
ALTER ROLE myuser CREATEDB;

-- Remove privilege
ALTER ROLE myuser NOCREATEDB;

-- Rename
ALTER ROLE oldname RENAME TO newname;
```

### Delete Role

```sql
DROP ROLE myuser;

-- Delete only if exists
DROP ROLE IF EXISTS myuser;
```

---

## 7. Permission Management

### Database Permissions

```sql
-- Grant connect permission to database
GRANT CONNECT ON DATABASE mydb TO myuser;

-- Grant all privileges on database
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;

-- Revoke permissions
REVOKE CONNECT ON DATABASE mydb FROM myuser;
```

### Schema Permissions

```sql
-- Schema usage permission
GRANT USAGE ON SCHEMA public TO myuser;

-- Permission to create objects in schema
GRANT CREATE ON SCHEMA public TO myuser;
```

### Table Permissions

```sql
-- SELECT permission on specific table
GRANT SELECT ON TABLE users TO myuser;

-- All privileges on specific table
GRANT ALL PRIVILEGES ON TABLE users TO myuser;

-- Permissions on all tables in schema
GRANT SELECT ON ALL TABLES IN SCHEMA public TO myuser;

-- Without DEFAULT PRIVILEGES, new tables require manual GRANT every time.
-- This sets up automatic grants so newly created tables inherit the permission.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO myuser;
```

### Permission Types

| Permission | Applied To | Description |
|------------|------------|-------------|
| `SELECT` | Tables, views | Query data |
| `INSERT` | Tables | Insert data |
| `UPDATE` | Tables | Update data |
| `DELETE` | Tables | Delete data |
| `TRUNCATE` | Tables | Empty table |
| `REFERENCES` | Tables | Create foreign keys |
| `TRIGGER` | Tables | Create triggers |
| `CREATE` | DB, schema | Create objects |
| `CONNECT` | DB | Connect |
| `USAGE` | Schema, sequences | Use |
| `EXECUTE` | Functions | Execute |

### Check Permissions

```sql
-- Check table permissions
\dp users

-- Or
SELECT grantee, privilege_type
FROM information_schema.table_privileges
WHERE table_name = 'users';
```

---

## 8. Schema Management

Schemas logically group tables within a database.

### Create Schema

```sql
-- Basic creation
CREATE SCHEMA myschema;

-- Specify owner
CREATE SCHEMA myschema AUTHORIZATION myuser;
```

### List Schemas

```sql
-- psql meta command
\dn

-- SQL query
SELECT schema_name FROM information_schema.schemata;
```

### Use Schema

```sql
-- Specify schema when creating table
CREATE TABLE myschema.users (
    id SERIAL PRIMARY KEY,
    name TEXT
);

-- Set search path
SET search_path TO myschema, public;

-- Check search path
SHOW search_path;
```

### Delete Schema

```sql
-- Delete empty schema
DROP SCHEMA myschema;

-- Delete with contents
DROP SCHEMA myschema CASCADE;
```

---

## 9. Practice Examples

### Practice 1: Project Database Setup

```sql
-- 1. Create database
CREATE DATABASE project_db;

-- 2. Switch database
\c project_db

-- 3. Create application user
CREATE USER app_user WITH PASSWORD 'app_password';

-- 4. Create read-only user
CREATE USER readonly_user WITH PASSWORD 'readonly_password';

-- 5. Create schemas
CREATE SCHEMA app_schema;
CREATE SCHEMA report_schema;

-- 6. Set permissions
-- app_user: full privileges
GRANT ALL PRIVILEGES ON DATABASE project_db TO app_user;
GRANT ALL PRIVILEGES ON SCHEMA app_schema TO app_user;

-- readonly_user: read-only
GRANT CONNECT ON DATABASE project_db TO readonly_user;
GRANT USAGE ON SCHEMA app_schema TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA app_schema TO readonly_user;

-- 7. Apply permissions to future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA app_schema
GRANT SELECT ON TABLES TO readonly_user;
```

### Practice 2: Test User Permissions

```sql
-- Create table as postgres user
CREATE TABLE app_schema.products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    price NUMERIC(10,2)
);

INSERT INTO app_schema.products (name, price) VALUES
('Laptop', 1500.00),
('Mouse', 35.00);

-- Connect as readonly_user to test
-- psql -U readonly_user -d project_db

-- SELECT succeeds
SELECT * FROM app_schema.products;

-- INSERT fails (no permission)
INSERT INTO app_schema.products (name, price) VALUES ('Keyboard', 80.00);
-- ERROR: permission denied for table products
```

### Practice 3: Query Database Information

```sql
-- All database sizes
SELECT
    datname AS database,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
WHERE datistemplate = false
ORDER BY pg_database_size(datname) DESC;

-- Current connection info
SELECT
    pid,
    usename,
    datname,
    client_addr,
    state,
    query
FROM pg_stat_activity
WHERE datname = current_database();

-- Role permissions summary
SELECT
    r.rolname,
    r.rolsuper AS superuser,
    r.rolcreatedb AS can_create_db,
    r.rolcreaterole AS can_create_role,
    r.rolcanlogin AS can_login
FROM pg_roles r
WHERE r.rolname NOT LIKE 'pg_%'
ORDER BY r.rolname;
```

---

## 10. Security Best Practices

### Principle of Least Privilege

```sql
-- Least privilege: grant only the operations the app actually performs.
-- If a compromised app_user has DELETE or DROP, an attacker inherits that power.
GRANT SELECT, INSERT, UPDATE ON users TO app_user;

-- Avoid ALL PRIVILEGES when possible — it includes TRUNCATE, REFERENCES, TRIGGER
-- which most application users never need
-- GRANT ALL PRIVILEGES ON ... -- Not recommended
```

### Minimize Superuser Usage

```sql
-- Use regular users for routine tasks
-- Use superuser only for administrative tasks
```

### Password Policy

```sql
-- Use strong passwords
CREATE USER myuser WITH PASSWORD 'C0mplex!P@ssw0rd';

-- Set account expiration
ALTER ROLE myuser VALID UNTIL '2025-12-31';
```

---

**Previous**: [PostgreSQL Basics](./01_PostgreSQL_Basics.md) | **Next**: [Tables and Data Types](./03_Tables_and_Data_Types.md)
