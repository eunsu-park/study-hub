# CRUD Basics

**Previous**: [Tables and Data Types](./03_Tables_and_Data_Types.md) | **Next**: [Conditions and Sorting](./05_Conditions_and_Sorting.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what CRUD stands for and why these four operations form the foundation of data manipulation
2. Write INSERT statements to add single and multiple rows, using DEFAULT values and RETURNING
3. Write SELECT statements with column aliases, DISTINCT, and simple expressions
4. Write UPDATE statements with WHERE clauses and verify changes with RETURNING
5. Write DELETE statements safely and distinguish DELETE from TRUNCATE
6. Implement UPSERT logic using ON CONFLICT (DO NOTHING / DO UPDATE)
7. Apply best practices for safe data modification (SELECT-first verification, transactions)

---

Almost every interaction between an application and its database boils down to one of four operations: creating new records, reading existing ones, updating values, or deleting rows. Mastering CRUD in SQL is like learning the four basic arithmetic operations in math -- everything more advanced builds on top of them.

---

## 0. Practice Setup

```sql
-- Create practice table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER,
    city VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 1. INSERT - Data Insertion

### Theory: INSERT — Append-Only at Heart

When you `INSERT INTO t (...) VALUES (...)`, PostgreSQL goes through this sequence:

1. **Find a page with enough free space** by consulting the **Free Space Map** (FSM) — a small auxiliary file (`<oid>_fsm`) that summarizes how much free space each page has.
2. **Acquire a content lock** on that page.
3. **Construct the tuple** in memory: 23-byte header with `xmin = current_xid`, `xmax = 0`; the column data laid out per §B.3 of lesson 03.
4. **Place the tuple** at `pd_upper - tuple_size` and the line pointer at `pd_lower`. Update both pointers.
5. **Write a WAL record** describing the change (`XLOG_HEAP_INSERT`).
6. **Mark the page dirty** in `shared_buffers`. The actual disk write happens later via the background writer or checkpointer.
7. **Update every index** on the table to point at the new row's `(page_no, line_pointer_index)`.

If no page has space, PostgreSQL extends the file by 8 KB (or more — `extend_table_with_multiple_blocks` was introduced in PG 16).

#### A.1 Why INSERT is fast

Steps 3-6 are all in-memory or sequential WAL writes. Step 7 (index updates) is the dominant cost — a table with 5 indexes pays for 6 page updates, not 1. This is why "fewer indexes" is not just a storage rule but a write-throughput rule.

### Insert Single Row

```sql
-- Specify all columns
INSERT INTO users (name, email, age, city)
VALUES ('John Kim', 'kim@email.com', 30, 'Seoul');

-- Specify only some columns (others will be DEFAULT or NULL)
INSERT INTO users (name, email)
VALUES ('Jane Lee', 'lee@email.com');
```

### Insert Multiple Rows

```sql
INSERT INTO users (name, email, age, city) VALUES
('Michael Park', 'park@email.com', 25, 'Busan'),
('Sarah Choi', 'choi@email.com', 28, 'Daejeon'),
('Emma Jung', 'jung@email.com', 35, 'Seoul');
```

### Using DEFAULT Values

```sql
-- Use DEFAULT for specific column
INSERT INTO users (name, email, age, city, created_at)
VALUES ('David Hong', 'hong@email.com', 40, 'Incheon', DEFAULT);

-- All columns DEFAULT (id auto-generated only)
INSERT INTO users DEFAULT VALUES;  -- Error: NOT NULL columns
```

### RETURNING - Return Inserted Data

```sql
-- RETURNING avoids a separate SELECT after INSERT — the database returns the generated
-- values (id, timestamps) in the same round-trip, reducing latency by 50%
INSERT INTO users (name, email, age, city)
VALUES ('Tommy Shin', 'shin@email.com', 5, 'Springfield')
RETURNING id;

-- Return multiple columns
INSERT INTO users (name, email, age, city)
VALUES ('Mary Kim', 'mikim@email.com', 32, 'Seoul')
RETURNING id, name, created_at;

-- Return all columns
INSERT INTO users (name, email)
VALUES ('Test User', 'test@email.com')
RETURNING *;
```

---

## 2. SELECT - Data Querying

### Query All Data

```sql
-- All columns
SELECT * FROM users;

-- Specific columns only
SELECT name, email FROM users;
```

### Column Aliases

```sql
SELECT
    name AS user_name,
    email AS user_email,
    age AS user_age
FROM users;

-- AS can be omitted
SELECT name user_name, email user_email FROM users;
```

### Remove Duplicates (DISTINCT)

```sql
-- Remove duplicate cities
SELECT DISTINCT city FROM users;

-- Remove duplicates of column combinations
SELECT DISTINCT city, age FROM users;
```

### Calculations and Expressions

```sql
-- Calculations
SELECT name, age, age + 10 AS age_after_10_years FROM users;

-- String concatenation
SELECT name || ' (' || email || ')' AS user_info FROM users;

-- CONCAT function
SELECT CONCAT(name, ' - ', city) AS name_city FROM users;
```

### Conditional Queries (Brief)

```sql
-- WHERE clause (details in next chapter)
SELECT * FROM users WHERE city = 'Seoul';
SELECT * FROM users WHERE age >= 30;
```

---

## 3. UPDATE - Data Modification

### Theory: UPDATE — Insert + Mark Old as Superseded

PostgreSQL never overwrites a live row. `UPDATE t SET x = 5 WHERE id = 1;` runs:

1. **Find the existing row** (using an index or sequential scan).
2. **Lock the row** (set `t_infomask` bits).
3. **Construct a new tuple** with the new column values, `xmin = current_xid`, `xmax = 0`.
4. **Place the new tuple** in this page if there is room, or another page (consulting the FSM).
5. **Set the old tuple's `xmax`** to `current_xid` and its `ctid` to point at the new tuple.
6. **Write WAL** (`XLOG_HEAP_UPDATE`).
7. **Insert new index entries** for every changed indexed column — and *also* for unchanged indexed columns if the new tuple lives on a different page (because indexes point at `ctid`, not at primary key).

#### B.1 The cost of two versions

After the UPDATE, *both* row versions exist on disk. Both have line pointers. The old version is invisible to new transactions (because their snapshots see `xmax = current_xid` as committed-and-deleted) but cannot be removed yet because some long-running transaction might still need it. **VACUUM** reclaims the space later by setting line pointer status to `LP_UNUSED`.

This is **table bloat** — a table with frequent UPDATEs grows in disk size faster than its live row count would suggest. Heavy-update workloads need autovacuum tuning to keep this in check.

### Theory: HOT — Heap-Only Tuple

If the UPDATE meets two conditions, PostgreSQL takes a fast path called **HOT**:

1. **No indexed column changed.**
2. **The new tuple fits on the same page** as the old tuple.

In that case, step 7 from §B (update every index) is **skipped entirely**. The new tuple is placed on the same page, and the old line pointer is converted to a "redirect" pointing at the new line pointer. Indexes still point at the old line pointer, but reads transparently follow the redirect chain.

```
Before HOT update:
  index → LP[3] → tuple v1 (id=1, x=4)

After HOT update of x=4 → x=5:
  index → LP[3] → (redirect to LP[4])
                  LP[4] → tuple v2 (id=1, x=5)
```

The benefits stack up:

- **Zero index modifications.** A table with 8 indexes still pays only 1 page update.
- **The dead tuple can be reclaimed by `HOT pruning`** during ordinary page reads, without VACUUM running. The line pointer is freed immediately on page-level cleanup.
- **Reduced WAL volume** because no index updates are logged.

#### C.1 The fillfactor knob

HOT only works if the new tuple fits on the same page. PostgreSQL exposes a per-table `fillfactor` setting (`CREATE TABLE ... WITH (fillfactor = 80);`) that says "leave 20% of each page empty for future updates". For update-heavy tables, setting fillfactor below 100 dramatically increases the HOT hit rate.

### Basic UPDATE

```sql
-- Update specific row
UPDATE users
SET age = 31
WHERE name = 'John Kim';

-- Update multiple columns
UPDATE users
SET age = 26, city = 'Daegu'
WHERE email = 'park@email.com';
```

### UPDATE Without Condition (Caution!)

```sql
-- All rows will be updated!
UPDATE users SET city = 'Seoul';  -- Dangerous!

-- Always check WHERE clause
```

### UPDATE with Calculations

```sql
-- Increment all users' age by 1
UPDATE users SET age = age + 1;

-- Only specific condition users
UPDATE users SET age = age + 1 WHERE city = 'Seoul';
```

### RETURNING to Check Updated Data

```sql
UPDATE users
SET age = 32
WHERE name = 'Jane Lee'
RETURNING *;

UPDATE users
SET city = 'Gwangju'
WHERE age < 30
RETURNING id, name, city;
```

### Set to NULL

```sql
UPDATE users
SET city = NULL
WHERE name = 'Test User';
```

---

## 4. DELETE - Data Deletion

### Theory: DELETE — Just a Tombstone

`DELETE FROM t WHERE id = 1;` does *not* free disk space. It runs:

1. **Find the row.**
2. **Set `xmax = current_xid`** on the tuple header.
3. **Write WAL** (`XLOG_HEAP_DELETE`).

That is it. The tuple body and line pointer remain in place. New transactions will skip the row (its `xmax` is committed by the time their snapshots check), but the bytes are still on disk.

#### D.1 What VACUUM does

When VACUUM runs over the page, it checks each tuple's visibility against the oldest active transaction in the cluster (`OldestXmin`). If `xmax` is older than `OldestXmin`, the tuple is invisible to *every* current and future transaction — safe to remove. VACUUM:

1. **Frees the tuple body** by compacting the page.
2. **Marks the line pointer as `LP_UNUSED`** so it can be reused by a future INSERT on this page.
3. **Records the page in the FSM** so future INSERTs find it.
4. **Updates indexes** to remove entries pointing at line pointers that have been reclaimed (this is the expensive part — VACUUM has to scan each index).

`VACUUM FULL` is more aggressive: it rewrites the entire table into a new file with no dead tuples, then atomically swaps. It takes an `ACCESS EXCLUSIVE` lock and can be slow, but it actually returns disk to the OS.

#### D.2 TRUNCATE — the bypass

`TRUNCATE TABLE t;` is conceptually `DELETE FROM t;` but does not produce dead tuples. It is implemented by allocating a new empty file and atomically swapping. It is O(1) regardless of table size, but it cannot be `WHERE`-filtered and it takes a strong lock.

### Basic DELETE

```sql
-- Delete specific row
DELETE FROM users WHERE name = 'Test User';

-- Multiple conditions
DELETE FROM users WHERE city IS NULL AND age IS NULL;
```

### DELETE Without Condition (Caution!)

```sql
-- Delete all data!
DELETE FROM users;  -- Dangerous!

-- Table remains
```

### RETURNING to Check Deleted Data

```sql
DELETE FROM users
WHERE email = 'test@email.com'
RETURNING *;
```

### TRUNCATE - Empty Table

```sql
-- TRUNCATE bypasses row-level WAL logging — it deallocates pages directly,
-- making it orders of magnitude faster than DELETE for clearing large tables.
-- Trade-off: no per-row triggers fire, and RETURNING is not available.
TRUNCATE TABLE users;

-- Restart SERIAL
TRUNCATE TABLE users RESTART IDENTITY;

-- With related tables (foreign keys)
TRUNCATE TABLE users CASCADE;
```

### DELETE vs TRUNCATE

| Feature | DELETE | TRUNCATE |
|---------|--------|----------|
| WHERE condition | Possible | Not possible |
| Speed | Slow | Fast |
| Transaction rollback | Possible | Limited |
| RETURNING | Possible | Not possible |
| Trigger execution | Executes | Doesn't execute |
| SERIAL reset | No | Optional |

---

## 5. UPSERT (ON CONFLICT)

Insert or update if conflict occurs.

### Ignore on Conflict

```sql
-- ON CONFLICT DO NOTHING is ideal for idempotent inserts — retrying the same request
-- (e.g., from a message queue) won't produce duplicate rows or raise an error
INSERT INTO users (name, email, age, city)
VALUES ('John Kim', 'kim@email.com', 35, 'Busan')
ON CONFLICT (email) DO NOTHING;
```

### Update on Conflict

```sql
-- Update if already exists
INSERT INTO users (name, email, age, city)
VALUES ('John Kim', 'kim@email.com', 35, 'Busan')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,
    city = EXCLUDED.city;
```

### EXCLUDED Keyword

`EXCLUDED` references the data that was attempted to be inserted.

```sql
INSERT INTO users (name, email, age, city)
VALUES ('John Kim', 'kim@email.com', 35, 'Busan')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,           -- New value (35)
    city = users.city,            -- Keep existing value
    name = EXCLUDED.name;         -- New value (John Kim)
```

### Conditional UPSERT

```sql
INSERT INTO users (name, email, age, city)
VALUES ('John Kim', 'kim@email.com', 35, 'Busan')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,
    city = EXCLUDED.city
WHERE users.age < EXCLUDED.age;  -- Only update if new age is greater
```

---

## 6. INSERT with Subquery

### Insert SELECT Results

```sql
-- Copy from another table
CREATE TABLE users_backup AS SELECT * FROM users;

-- Or
INSERT INTO users_backup SELECT * FROM users;

-- Conditional copy
INSERT INTO users_backup
SELECT * FROM users WHERE city = 'Seoul';
```

### Insert Calculated Values

```sql
INSERT INTO statistics (city, user_count)
SELECT city, COUNT(*) FROM users GROUP BY city;
```

---

## 7. Practice Examples

### Prepare Practice Data

```sql
-- Reset table
TRUNCATE TABLE users RESTART IDENTITY;

-- Insert sample data
INSERT INTO users (name, email, age, city) VALUES
('John Kim', 'kim@email.com', 30, 'Seoul'),
('Jane Lee', 'lee@email.com', 25, 'Busan'),
('Michael Park', 'park@email.com', 35, 'Seoul'),
('Sarah Choi', 'choi@email.com', 28, 'Daejeon'),
('Emma Jung', 'jung@email.com', 32, 'Seoul'),
('David Hong', 'hong@email.com', 40, 'Incheon'),
('Kevin Kang', 'kang@email.com', 27, 'Busan'),
('Lisa Son', 'son@email.com', 33, 'Seoul');
```

### Practice 1: Basic CRUD

```sql
-- 1. Add new user
INSERT INTO users (name, email, age, city)
VALUES ('New User', 'new@email.com', 22, 'Gwangju')
RETURNING *;

-- 2. Query Seoul users
SELECT * FROM users WHERE city = 'Seoul';

-- 3. Change city to 'Metropolitan' for users age 30+
UPDATE users
SET city = 'Metropolitan'
WHERE age >= 30
RETURNING name, age, city;

-- 4. Delete Gwangju users
DELETE FROM users
WHERE city = 'Gwangju'
RETURNING *;
```

### Practice 2: UPSERT

```sql
-- Update age and city if email already exists
INSERT INTO users (name, email, age, city)
VALUES ('John Kim', 'kim@email.com', 31, 'Gyeonggi')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,
    city = EXCLUDED.city
RETURNING *;

-- Insert if email doesn't exist
INSERT INTO users (name, email, age, city)
VALUES ('New Member', 'newuser@email.com', 29, 'Jeju')
ON CONFLICT (email)
DO UPDATE SET age = EXCLUDED.age, city = EXCLUDED.city
RETURNING *;
```

### Practice 3: Bulk Data Processing

```sql
-- Create backup table and copy data
CREATE TABLE users_backup AS
SELECT * FROM users WHERE 1=0;  -- Copy structure only

INSERT INTO users_backup
SELECT * FROM users;

-- Backup only specific condition users
INSERT INTO users_backup
SELECT * FROM users WHERE city IN ('Seoul', 'Busan');

-- Check backup
SELECT COUNT(*) FROM users_backup;
```

---

## 8. Precautions and Tips

### Prevent SQL Injection

```sql
-- Bad example (direct string concatenation)
-- "SELECT * FROM users WHERE name = '" + userInput + "'"

-- Good example (use parameter binding - in application)
-- "SELECT * FROM users WHERE name = $1"
```

### Verify Before UPDATE/DELETE

```sql
-- 1. First check with SELECT
SELECT * FROM users WHERE city = 'Seoul';

-- 2. Execute UPDATE/DELETE after confirmation
UPDATE users SET age = age + 1 WHERE city = 'Seoul';
```

### Use Transactions

```sql
-- Use transactions for important operations
BEGIN;
UPDATE users SET age = age + 1 WHERE city = 'Seoul';
-- Check results then
COMMIT;  -- or ROLLBACK;
```

---

**Previous**: [Tables and Data Types](./03_Tables_and_Data_Types.md) | **Next**: [Conditions and Sorting](./05_Conditions_and_Sorting.md)
