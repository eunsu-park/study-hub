# PostgreSQL Basics

**Next**: [Database Management](./02_Database_Management.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what PostgreSQL is and describe its key features (ACID, MVCC, extensibility)
2. Compare PostgreSQL with other popular databases (MySQL, SQLite)
3. Install PostgreSQL using Docker, Homebrew, or a Linux package manager
4. Connect to a PostgreSQL server using the `psql` command-line client
5. Identify and use essential psql meta-commands (`\l`, `\dt`, `\d`, `\c`)
6. Write and execute basic SQL statements (SELECT, CREATE DATABASE)
7. Apply fundamental SQL syntax rules (case sensitivity, comments, semicolons)
8. Troubleshoot common connection and startup errors

---

PostgreSQL is one of the most advanced open-source relational databases available today. Whether you are building a small web application or designing a large-scale analytics platform, PostgreSQL provides the reliability, extensibility, and standards compliance that professional developers and data engineers rely on. This lesson walks you through installation, first connection, and the essential commands that form the starting point of your PostgreSQL journey.

## 1. What is PostgreSQL?

PostgreSQL is an open-source relational database management system (RDBMS).

### Features

- **Open Source**: Free to use
- **SQL Standards Compliance**: Follows ANSI SQL standards well
- **Extensibility**: Supports JSON, arrays, user-defined types
- **ACID Compliance**: Guarantees transaction reliability
- **Concurrency Control**: MVCC (Multi-Version Concurrency Control)

### Why Use PostgreSQL?

```
┌─────────────────────────────────────────────────────────────┐
│                PostgreSQL Advantages                         │
├─────────────────────────────────────────────────────────────┤
│  • Excellent performance for complex queries                 │
│  • Can be used like NoSQL with JSON/JSONB types             │
│  • Built-in full-text search                                │
│  • Geographic data support (PostGIS)                         │
│  • Suitable for large-scale data processing                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Comparison with Other Databases

| Feature | PostgreSQL | MySQL | SQLite |
|---------|------------|-------|--------|
| License | PostgreSQL License | GPL | Public Domain |
| JSON Support | JSONB (high performance) | JSON | JSON (limited) |
| Concurrency | MVCC | InnoDB MVCC | File locking |
| Scalability | Very high | High | Low |
| Use Case | Enterprise, analytics | Web applications | Embedded, testing |

---

## 3. Installation Methods

### Docker (Recommended)

The fastest way to get started.

```bash
# Docker isolates PostgreSQL from your host system — you can experiment freely,
# break things, and "docker rm" to start fresh without affecting other installations.
# It also guarantees identical setups across machines (reproducibility).
docker run --name postgres-study \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_DB=mydb \
  -p 5432:5432 \
  -d postgres:16

# Check if running
docker ps

# Connect to psql inside container
docker exec -it postgres-study psql -U myuser -d mydb
```

### macOS (Homebrew)

```bash
# Install PostgreSQL
brew install postgresql@16

# Start service
brew services start postgresql@16

# Connect to default database
psql postgres
```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Check service status
sudo systemctl status postgresql

# Connect as postgres user
sudo -u postgres psql
```

### Linux (CentOS/RHEL)

```bash
# Add PostgreSQL repository
sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-8-x86_64/pgdg-redhat-repo-latest.noarch.rpm

# Install PostgreSQL
sudo dnf install -y postgresql16-server

# Initialize database
sudo /usr/pgsql-16/bin/postgresql-16-setup initdb

# Start service
sudo systemctl start postgresql-16
sudo systemctl enable postgresql-16
```

### Windows

1. Download installer from [official download page](https://www.postgresql.org/download/windows/)
2. Run installation wizard
3. Set password
4. Use default port 5432
5. Install pgAdmin together (GUI tool)

---

## 4. Verify Installation

```bash
# Check PostgreSQL version
psql --version
# or
postgres --version
```

Example output:
```
psql (PostgreSQL) 16.1
```

---

## 5. psql Client

psql is PostgreSQL's interactive terminal client.

### Connection Methods

```bash
# Default connection (local, current user)
psql

# Connect to specific database
psql -d mydb

# Connect with specific user
psql -U username -d dbname

# Connect with host/port
psql -h localhost -p 5432 -U username -d dbname

# Connect to Docker container
docker exec -it postgres-study psql -U myuser -d mydb
```

### Meta Commands (Backslash Commands)

Commands in psql that start with `\`.

| Command | Description |
|---------|-------------|
| `\l` | List databases |
| `\c dbname` | Connect to database |
| `\dt` | List tables in current DB |
| `\dt+` | List tables (detailed) |
| `\d tablename` | Describe table structure |
| `\d+ tablename` | Describe table (detailed) |
| `\du` | List users (roles) |
| `\dn` | List schemas |
| `\df` | List functions |
| `\di` | List indexes |
| `\x` | Toggle expanded output mode |
| `\timing` | Toggle query execution time display |
| `\i filename` | Execute SQL file |
| `\o filename` | Save output to file |
| `\q` | Quit psql |
| `\?` | Help for meta commands |
| `\h` | Help for SQL commands |
| `\h SELECT` | Help for SELECT syntax |

### Practice: Basic Commands

```sql
-- After connecting to psql

-- List databases
\l

-- Check current connection info
\conninfo

-- List tables (initially empty)
\dt

-- View help
\?
```

---

## 6. Execute First Query

### Simple Calculation

```sql
-- Use like a calculator
SELECT 1 + 1;
```

Output:
```
 ?column?
----------
        2
(1 row)
```

### Print String

```sql
SELECT 'Hello, PostgreSQL!';
```

Output:
```
      ?column?
--------------------
 Hello, PostgreSQL!
(1 row)
```

### Check Current Time

```sql
SELECT NOW();
```

Output:
```
              now
-------------------------------
 2024-01-15 10:30:45.123456+09
(1 row)
```

### Check Version

```sql
SELECT version();
```

---

## 7. Basic SQL Syntax

### Case Sensitivity

- SQL keywords: Case insensitive (`SELECT` = `select`)
- Table/column names: Stored as lowercase by default
- Strings: Use single quotes (`'Hello'`)

```sql
-- These three queries are identical
SELECT * FROM users;
select * from users;
Select * From Users;
```

### Comments

```sql
-- Single line comment

/* Multi-line
   comment */

SELECT 1; -- Inline comment
```

### Statement Termination

- End statements with semicolon (`;`)
- In psql, can input multiple lines and execute with `;`

```sql
SELECT
    id,
    name,
    email
FROM users
WHERE active = true;
```

---

## 8. Database Creation and Deletion

### Create Database

```sql
-- Each database is an isolated namespace — tables in one DB cannot see tables in another.
-- Create separate databases for dev, test, and prod to prevent accidental cross-contamination.
CREATE DATABASE mydb;

-- Specify encoding and locale upfront — they are immutable after creation.
-- UTF8 supports all languages; locale affects sort order and string comparison.
CREATE DATABASE mydb
    ENCODING 'UTF8'
    LC_COLLATE 'ko_KR.UTF-8'
    LC_CTYPE 'ko_KR.UTF-8';
```

### Switch Database

```sql
-- psql meta command
\c mydb
```

Output:
```
You are now connected to database "mydb" as user "postgres".
```

### Delete Database

```sql
DROP DATABASE mydb;

-- Delete only if exists
DROP DATABASE IF EXISTS mydb;
```

---

## 9. Practice Examples

### Practice 1: Verify Environment

```sql
-- 1. Check PostgreSQL version
SELECT version();

-- 2. Check current user
SELECT current_user;

-- 3. Check current database
SELECT current_database();

-- 4. Check current time
SELECT NOW();

-- 5. Check server configuration
SHOW server_version;
SHOW data_directory;
```

### Practice 2: Create First Database

```sql
-- 1. Create study database
CREATE DATABASE study_db;

-- 2. List databases
\l

-- 3. Switch to new database
\c study_db

-- 4. Check connection info
\conninfo
```

### Practice 3: Create Simple Table

```sql
-- 1. Create table
CREATE TABLE hello (
    id SERIAL PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. Insert data
INSERT INTO hello (message) VALUES ('Hello, PostgreSQL!');
INSERT INTO hello (message) VALUES ('My first table!');

-- 3. Query data
SELECT * FROM hello;

-- 4. Check table structure
\d hello
```

Example output:
```
 id |      message       |         created_at
----+--------------------+----------------------------
  1 | Hello, PostgreSQL! | 2024-01-15 10:30:45.123456
  2 | My first table!    | 2024-01-15 10:30:50.654321
(2 rows)
```

---

## 10. Troubleshooting

### Connection Errors

**Error**: `psql: error: connection refused`
```bash
# Check service status
sudo systemctl status postgresql

# Start service
sudo systemctl start postgresql
```

**Error**: `FATAL: password authentication failed`
```bash
# Need to check and modify pg_hba.conf
# Or use correct password
```

**Error**: `FATAL: database "username" does not exist`
```bash
# Connect specifying database
psql -d postgres
```

### Docker Related

```bash
# Check container status
docker ps -a

# Check container logs
docker logs postgres-study

# Restart container
docker restart postgres-study
```

---

**Next**: [Database Management](./02_Database_Management.md)
