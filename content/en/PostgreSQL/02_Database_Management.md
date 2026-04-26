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

Before the management commands, read [**Theory & Principles**](#theory--principles) — what a "database" actually is on disk, how `pg_catalog` stores its own metadata, the role of tablespaces, and why every write goes through the WAL first.

---

## Theory & Principles

A `CREATE DATABASE` command is not just a name in a config file. It allocates a physical directory tree, populates that tree with a copy of the template database's files, inserts a row into the cluster-wide `pg_database` catalog, and reserves a slot in the WAL stream so every later modification of that database is durable. Understanding the four pieces below — cluster vs database vs schema, the catalog's structure, the tablespace storage layer, and the WAL — turns "administrative commands" into observable changes you can predict and verify.

This section covers:

- **(A)** The cluster → database → schema → relation hierarchy and how it maps to the filesystem.
- **(B)** How `pg_catalog` stores cluster metadata, and which catalog backs which `\` command.
- **(C)** Tablespaces: the physical layer that decouples a database's location from the data directory.
- **(D)** The WAL: why every modification (including DDL like `CREATE DATABASE`) is logged before it is applied.

### A. Cluster, Database, Schema, Relation

These four words are not synonyms. Each is a distinct level in PostgreSQL's namespace hierarchy.

| Level | What it is | How to list |
|-------|-----------|-------------|
| **Cluster** | One running postmaster + one data directory (`PGDATA`). Holds many databases. | `pg_lsclusters` (Debian) or `SHOW data_directory;` |
| **Database** | An isolated namespace with its own catalog tables, encoding, and collation. | `\l` (queries `pg_database`) |
| **Schema** | A logical grouping of relations *inside* one database. Default is `public`. | `\dn` (queries `pg_namespace`) |
| **Relation** | A table, index, view, sequence, or materialized view. | `\dt` (queries `pg_class WHERE relkind = 'r'`) |

#### A.1 The filesystem mapping

Inside `PGDATA/base/`, every database is a directory whose name is the OID of its `pg_database` row. Inside that directory, every relation is one or more files (split into 1 GB segments) whose names are the OID of their `pg_class` row. So:

```
PGDATA/
├── base/
│   ├── 16384/        ← database OID
│   │   ├── 24576     ← relation OID (heap)
│   │   ├── 24576.1   ← second 1 GB segment
│   │   ├── 24579     ← associated index
│   │   └── ...
│   └── 1/            ← template1 database
└── pg_wal/           ← Write-Ahead Log segments
```

When you run `DROP DATABASE`, PostgreSQL does two things: it removes the row from `pg_database` and it `unlink()`s the directory. There is nothing magical about a database — it is fundamentally a directory plus a catalog row.

#### A.2 Why schemas exist

If everything you create lives in `public`, name collisions become a problem fast. Schemas give you sub-namespaces: `app.users` and `analytics.users` are two different tables. The `search_path` setting controls which schemas are searched when a name is unqualified. `SET search_path = app, public;` means `users` resolves to `app.users` if it exists, otherwise `public.users`.

### B. Inside `pg_catalog`

Every administrative `\` meta-command in `psql` is a wrapper around a SQL query against `pg_catalog`. You can see the query with `\set ECHO_HIDDEN on`. The relevant catalogs for this lesson:

- **`pg_database`** — one row per database. Columns: `oid`, `datname`, `datdba` (owner), `encoding`, `datcollate`, `dattablespace`, `datallowconn`, `datistemplate`. Backs `\l`.
- **`pg_authid`** — one row per role, cluster-wide. Columns: `oid`, `rolname`, `rolsuper`, `rolcanlogin`, `rolpassword` (hashed). Backs `\du`.
- **`pg_roles`** — a view over `pg_authid` that hides the password column. Safe for non-superusers.
- **`pg_tablespace`** — one row per tablespace. Columns: `oid`, `spcname`, `spcowner`, `spcacl`. Backs `\db`.
- **`pg_namespace`** — one row per schema. Backs `\dn`.

Catalog tables are themselves regular tables — they just happen to be the ones the engine reads to plan every other query. They obey ACID. A `CREATE ROLE alice;` is exactly an `INSERT INTO pg_authid` (plus a few side effects) and is rolled back if the surrounding transaction aborts.

#### B.1 Cluster-wide vs database-local

A subtle point: some catalogs are *shared across the entire cluster*, others are *per-database*. Roles, databases, and tablespaces are cluster-wide (you log in as the same `alice` no matter which database you connect to). Tables, schemas, functions, and indexes are per-database (each database has its own `pg_class`). This is why you cannot `JOIN` two databases together — they do not share a catalog.

### C. Tablespaces

By default, every database lives under `PGDATA/base/`. A **tablespace** lets you say "store this database (or this table, or this index) under a different filesystem path". Internally, a tablespace is just a symbolic link from `PGDATA/pg_tblspc/<oid>` to the configured directory.

#### C.1 Why you might use one

1. **Hot data on SSD, cold data on HDD.** Put the partition table on a fast tablespace and the historical archive on a slow one.
2. **Per-tenant disk quotas.** Each tenant gets a tablespace on their own filesystem with its own size limit.
3. **Spreading I/O across volumes** when one disk is the bottleneck.

The `CREATE TABLESPACE archive LOCATION '/mnt/cold/pg';` command does three things: validates that the directory exists and is empty, inserts a row into `pg_tablespace`, and creates the symlink. From then on, `CREATE TABLE foo (...) TABLESPACE archive;` puts `foo`'s files under that path.

#### C.2 What it does not do

A tablespace is *not* a security boundary, *not* a separate database, and *not* a way to share files between clusters. Two different clusters cannot share a tablespace directory — the OID space would collide.

### D. WAL — Write-Ahead Log

The fundamental durability rule: **no modification is applied to a heap page until the corresponding log record is on disk in `pg_wal/`**. This is "write-ahead" — the log is written first, the data file is updated lazily.

The reason: if the server crashes, restart can replay the WAL from the last checkpoint to reconstruct any committed change that had not yet reached the heap file. Without the log, you would have to `fsync()` every dirty data page on every commit, which is far slower because data pages are scattered across the disk while the log is sequential.

#### D.1 What is in a WAL record

A WAL record describes a *physical* change: "set bytes 32-40 of page 17 in relation 24576 to these values", plus the transaction ID that made the change. There are also logical records for things like commit, abort, prepared-transaction state, and full-page images (after a checkpoint, the first modification to a page writes the entire page so corruption from torn writes is recoverable).

#### D.2 Checkpoints

Periodically (every `checkpoint_timeout`, default 5 minutes, or after `max_wal_size` of WAL is generated), the checkpointer process flushes all dirty buffers to their data files and writes a *checkpoint record* to the WAL. After a crash, recovery only needs to replay WAL from the last checkpoint forward — older WAL can be recycled or archived.

#### D.3 Why DDL and `CREATE DATABASE` go through WAL too

Every catalog change is also a heap modification (it writes to `pg_class`, `pg_database`, etc.) and therefore generates WAL. This is why `CREATE DATABASE` can be replicated to a standby — the standby simply replays the WAL records, including the directory creation and template-copy operations.

### From Theory to the Commands Below

Each of the following sections is one of these ideas made concrete:

- **`CREATE DATABASE`, `DROP DATABASE`** — manipulates `pg_database` (§B) and creates/removes a directory under `base/` (§A.1), generating WAL records along the way (§D.3).
- **`CREATE ROLE`, `GRANT`** — modifies the cluster-wide `pg_authid` and per-object ACL columns in `pg_class` (§B.1).
- **`CREATE SCHEMA`, `search_path`** — operates on `pg_namespace` and the per-session resolution rule (§A.2).
- **`CREATE TABLESPACE`** — adds a row to `pg_tablespace` and a symlink under `pg_tblspc/` (§C).
- **`pg_dumpall`, `pg_dump`** — reads `pg_catalog` to reconstruct DDL; `pg_dumpall` includes the cluster-wide catalogs (roles, tablespaces) that `pg_dump` skips (§B.1).

---

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
