# Tables and Data Types

**Previous**: [Database Management](./02_Database_Management.md) | **Next**: [CRUD Basics](./04_CRUD_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create tables using `CREATE TABLE` with appropriate column definitions
2. Distinguish among PostgreSQL numeric types (INTEGER, NUMERIC, SERIAL) and choose the right one
3. Compare character types (CHAR, VARCHAR, TEXT) and date/time types (DATE, TIMESTAMP, TIMESTAMPTZ)
4. Apply special data types including BOOLEAN, JSONB, UUID, arrays, and ENUM
5. Implement constraints (PRIMARY KEY, NOT NULL, UNIQUE, CHECK, FOREIGN KEY) to enforce data integrity
6. Modify existing tables with ALTER TABLE (add/drop columns, change types, manage constraints)
7. Design a multi-table schema with proper foreign key relationships

---

Tables are the fundamental building blocks of any relational database. Every piece of data your application stores -- user profiles, product catalogs, financial transactions -- ultimately lives inside a table with carefully chosen columns, data types, and constraints. Getting the schema right at design time prevents countless headaches later, from subtle data corruption to slow queries.

Before the `CREATE TABLE` syntax, read [**Theory & Principles**](#theory--principles) — how PostgreSQL physically lays out a row inside an 8 KB page, how big values overflow into the TOAST mechanism, and how every data type's storage size flows from these two facts.

---

## Theory & Principles

A `CREATE TABLE` statement is a contract about more than the column names. The choice of `INTEGER` vs `BIGINT`, `VARCHAR(255)` vs `TEXT`, or `TIMESTAMP` vs `TIMESTAMPTZ` translates directly into bytes per row, alignment padding, page utilization, and whether a value can be stored inline or has to be pushed out to a separate file. Once you understand the page layout, the TOAST mechanism, and the per-type storage cost, schema decisions stop being tribal knowledge and become arithmetic.

This section covers:

- **(A)** The PostgreSQL page: 8 KB pages, page header, line pointers, tuple bodies.
- **(B)** A tuple from byte 0: header, null bitmap, alignment, and the column data area.
- **(C)** TOAST — how values larger than ~2 KB are sliced, optionally compressed, and stored out of line.
- **(D)** Per-type storage costs and the alignment trap that wastes space if columns are ordered carelessly.

### A. The 8 KB Page

PostgreSQL reads and writes the heap in fixed-size **pages** (also called blocks). The default size is **8 KB** and is fixed at compile time — every table file is an integer number of pages, every buffer in `shared_buffers` is one page, every WAL update tracks pages.

```
┌────────────────────────────────────────────────┐  byte 0
│ PageHeader (24 bytes)                          │
│  pd_lsn, pd_checksum, pd_lower, pd_upper, ...  │
├────────────────────────────────────────────────┤  pd_lower
│ ItemIdData[]  (4 bytes each, "line pointers")  │
│  ↓ grows downward                              │
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤  free space
│                                                │
│ ↑ grows upward                                 │
│ Tuples (rows)                                  │
├────────────────────────────────────────────────┤  pd_upper
│ Special space (used by indexes, not heap)      │
└────────────────────────────────────────────────┘  byte 8191
```

Two pointers (`pd_lower` and `pd_upper`) describe the boundary of free space. Inserting a row writes the tuple at `pd_upper - tuple_size` and the line pointer at `pd_lower`. Both pointers move toward each other; when they meet, the page is full and PostgreSQL allocates the next page.

#### A.1 Why line pointers exist

Indexes need stable references to rows, but tuples can be deleted, updated (creating new versions), or moved during VACUUM. A line pointer (`ItemId`) is a 4-byte indirection: index entries point to `(page_number, line_pointer_index)`, and the line pointer points to the actual offset within the page. When a tuple is moved, only the line pointer is updated — every index continues to work.

#### A.2 The `ctid` system column

Every row has a `ctid` you can `SELECT`: it is exactly `(page_number, line_pointer_index)`. `ctid` is *not* stable across updates — an UPDATE that changes a row may relocate it, giving it a new ctid (the old line pointer becomes a "redirect" to the new one).

### B. Tuple Layout

Each row is a **tuple** structured like this:

```
┌────────────────────────────────────────┐
│ HeapTupleHeader  (23 bytes minimum)    │
│  xmin, xmax, cmin, cmax, ctid,         │
│  t_infomask, t_hoff                    │
├────────────────────────────────────────┤
│ Null bitmap (optional, 1 bit/column)   │
├────────────────────────────────────────┤
│ Alignment padding to 8-byte boundary   │
├────────────────────────────────────────┤
│ Column 1 data                          │
│ Alignment padding                      │
│ Column 2 data                          │
│   ...                                  │
└────────────────────────────────────────┘
```

#### B.1 Header — what makes MVCC and visibility possible

The 23-byte header carries the visibility metadata that MVCC needs:

- **`xmin`** — the transaction id that created this row version.
- **`xmax`** — the transaction id that deleted/superseded it (0 if still live).
- **`cmin`/`cmax`** — command-id within the transaction (for self-visibility).
- **`ctid`** — physical location of this tuple (or, if updated, of the next version).
- **`t_infomask`** — flag bits: row has nulls? was it frozen by VACUUM? is `xmax` actually a multixact?

The header is the same size for a 1-column table and a 100-column table. Many narrow rows have proportionally more overhead than a few wide rows.

#### B.2 Null bitmap — only allocated if at least one column is null

The bitmap has one bit per column, padded to 8 bytes. PostgreSQL does *not* allocate it if every column in the row is non-null (the `HEAP_HASNULL` bit in `t_infomask` is the flag). For tables with many nullable columns where most rows have nulls, this is a noticeable space saver.

#### B.3 Alignment

Every PostgreSQL data type has an **alignment requirement** — `int2` aligns to 2 bytes, `int4` to 4, `int8` and `timestamp` to 8, `text` to 4 (the length word). The tuple builder inserts padding bytes between columns to satisfy alignment. This is the source of one of the most common storage gotchas — see §D.

### C. TOAST — The Oversized-Attribute Storage Technique

A row cannot exceed one page (8 KB). But a `text` or `bytea` value can easily be megabytes. PostgreSQL resolves this with **TOAST** (The Oversized-Attribute Storage Technique).

#### C.1 The TOAST decision tree

When a tuple would exceed the **TOAST threshold** (`TOAST_TUPLE_THRESHOLD`, default ~2 KB, i.e. ~1/4 of the page), the planner runs this loop on the largest TOAST-able column:

1. **Compress** the value (PGLZ or LZ4 in modern versions). If it now fits, write it inline.
2. If still too large, **slice it into ~2 KB chunks** and write the chunks to the table's TOAST table (a separate relation auto-created at `CREATE TABLE` time, named `pg_toast.pg_toast_<oid>`).
3. The main row stores a small **TOAST pointer** (18 bytes) referencing the chunks by OID and total length.

Each TOAST-able column has a per-column **storage strategy** you can change with `ALTER TABLE ... SET STORAGE`:

| Strategy | Compress? | Out-of-line? |
|----------|-----------|--------------|
| `PLAIN`  | no        | no (only for non-TOAST-able types) |
| `EXTENDED` (default for TEXT/BYTEA) | yes | yes |
| `EXTERNAL` | no | yes (faster for `substring` calls) |
| `MAIN`   | yes | only if still too big after compression |

#### C.2 Why TOAST matters in practice

A `SELECT id FROM big_log_table` is fast even if every row has a 1 MB body, because the body lives in the TOAST table and is not read unless explicitly projected. Conversely, `SELECT body` triggers a join to the TOAST table — invisibly, but it adds I/O. This is the database-level reason why "select only what you need" is not just code style.

### D. Per-Type Storage and the Alignment Trap

A representative subset of PostgreSQL types and their storage:

| Type | Bytes | Alignment | Notes |
|------|-------|-----------|-------|
| `boolean` | 1 | 1 | |
| `smallint` | 2 | 2 | |
| `integer` | 4 | 4 | |
| `bigint` | 8 | 8 | |
| `numeric(p, s)` | variable (~5–8 + 2/digit) | 4 | Arbitrary precision; slower than int |
| `real` (float4) | 4 | 4 | |
| `double precision` (float8) | 8 | 8 | |
| `date` | 4 | 4 | |
| `time` | 8 | 8 | |
| `timestamp`/`timestamptz` | 8 | 8 | Both are 8 bytes; tz info is per-session, not stored |
| `interval` | 16 | 8 | |
| `uuid` | 16 | 4 | |
| `text`/`varchar(n)`/`bytea` | 1 byte length header + payload (TOAST-able) | 4 | `varchar(n)` enforces length; storage is identical to `text` |
| `char(n)` | n bytes (space-padded) | 4 | Almost never the right choice; use `text` |
| `json` | text + length header | 4 | Stored as raw text |
| `jsonb` | binary parse tree + length header | 4 | TOAST-able; supports GIN |

#### D.1 The column ordering trap

Because of alignment padding, the *order* of columns in `CREATE TABLE` affects row size. Consider:

```sql
CREATE TABLE bad  (a int2, b int8, c int2);  -- 2 + 6 pad + 8 + 2 + 6 pad = 24 bytes
CREATE TABLE good (b int8, a int2, c int2);  -- 8 + 2 + 2 + 4 pad         = 16 bytes
```

The "bad" version wastes 8 bytes per row to satisfy `int8`'s 8-byte alignment after a 2-byte field. **Order columns from largest alignment to smallest** to minimize padding. For wide tables this can shrink the heap by 10-20% with no schema change beyond column order.

#### D.2 Variable-length types and the 1-byte short header

`text`, `bytea`, and `varchar` use a length prefix. PostgreSQL has a clever optimization — for values up to 126 bytes, it uses a **1-byte short header** instead of the standard 4 bytes. So a column of mostly-short strings is much cheaper than the 4-byte overhead would suggest.

### From Theory to the SQL Below

Each of the following sections is one of these ideas made concrete:

- **`CREATE TABLE` column list** — declares column types; PostgreSQL computes alignment, padding, and storage strategy from this list (§B.3, §D).
- **Choosing `TEXT` vs `VARCHAR(n)`** — both use the same TOAST-able storage; `varchar(n)` adds a length check (§C, §D).
- **`TIMESTAMP` vs `TIMESTAMPTZ`** — both 8 bytes; the difference is interpretation, not storage (§D).
- **`PRIMARY KEY`, `UNIQUE`, `NOT NULL`** — `NOT NULL` lets PostgreSQL skip the null bitmap (§B.2); `PRIMARY KEY` builds an index whose pages have the same 8 KB layout (§A).
- **`ALTER TABLE ... SET STORAGE`** — changes the TOAST strategy of one column (§C.1).

---

## 1. Table Basic Concepts

A table is a structure that stores data organized into rows and columns.

```
┌──────────────────────────────────────────────────────┐
│                    users table                        │
├────────┬──────────┬─────────────────┬───────────────┤
│   id   │   name   │      email      │  created_at   │
├────────┼──────────┼─────────────────┼───────────────┤
│   1    │  김철수  │ kim@email.com   │ 2024-01-15    │
│   2    │  이영희  │ lee@email.com   │ 2024-01-16    │
│   3    │  박민수  │ park@email.com  │ 2024-01-17    │
└────────┴──────────┴─────────────────┴───────────────┘
  Column                 ↑ each row is one record
```

---

## 2. Table Creation

### Basic Syntax

```sql
CREATE TABLE table_name (
    column1 data_type [constraints],
    column2 data_type [constraints],
    ...
);
```

### Basic Example

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Create Only If Not Exists

```sql
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);
```

---

## 3. Numeric Data Types

### Integer Types

| Type | Size | Range |
|------|------|-------|
| `SMALLINT` | 2 bytes | -32,768 ~ 32,767 |
| `INTEGER` (INT) | 4 bytes | -2,147,483,648 ~ 2,147,483,647 |
| `BIGINT` | 8 bytes | -9 quintillion ~ 9 quintillion |

```sql
CREATE TABLE products (
    id INTEGER,
    quantity SMALLINT,
    total_sold BIGINT
);
```

### Auto-Increment (Serial)

| Type | Range |
|------|-------|
| `SMALLSERIAL` | 1 ~ 32,767 |
| `SERIAL` | 1 ~ 2,147,483,647 |
| `BIGSERIAL` | 1 ~ 9 quintillion |

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,  -- automatically increments: 1, 2, 3, ...
    order_date DATE
);

-- IDENTITY (SQL standard) is preferred over SERIAL in PG 10+ because SERIAL creates
-- a separate sequence with loose coupling — IDENTITY ties the sequence to the column
-- lifecycle and prevents accidental manual inserts that break the sequence
CREATE TABLE orders (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_date DATE
);
```

### Floating-Point Types

| Type | Description |
|------|-------------|
| `REAL` | 4 bytes, 6-digit precision |
| `DOUBLE PRECISION` | 8 bytes, 15-digit precision |
| `NUMERIC(p, s)` | Exact number (p: total digits, s: decimal places) |
| `DECIMAL(p, s)` | Identical to NUMERIC |

```sql
-- Use NUMERIC for money/financial data — it is exact (no rounding errors).
-- REAL/DOUBLE PRECISION are faster but approximate; 0.1 + 0.2 ≠ 0.3 in float.
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    price NUMERIC(10, 2),      -- up to 10 digits, 2 decimal places (e.g. 99999999.99)
    weight REAL,               -- floating point (use for measurements where rounding is OK)
    rating DOUBLE PRECISION    -- higher-precision floating point
);

INSERT INTO products (price, weight, rating) VALUES
(19900.00, 1.5, 4.7);
```

---

## 4. Character Data Types

| Type | Description |
|------|-------------|
| `CHAR(n)` | Fixed-length string (padded with spaces to fill remaining space) |
| `VARCHAR(n)` | Variable-length string (up to n characters) |
| `TEXT` | Unlimited-length string |

```sql
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    code CHAR(10),           -- always 10 characters (used for codes, etc.)
    title VARCHAR(200),      -- up to 200 characters
    content TEXT             -- no length limit
);
```

### VARCHAR vs TEXT

```sql
-- No significant difference in practice. TEXT is often preferred in PostgreSQL
CREATE TABLE posts (
    title VARCHAR(255),  -- when a length limit is needed
    body TEXT            -- when no length limit is needed
);
```

---

## 5. Date/Time Data Types

| Type | Description | Example |
|------|-------------|---------|
| `DATE` | Date only | 2024-01-15 |
| `TIME` | Time only | 14:30:00 |
| `TIMESTAMP` | Date + time | 2024-01-15 14:30:00 |
| `TIMESTAMPTZ` | Date + time + timezone | 2024-01-15 14:30:00+09 |
| `INTERVAL` | Time interval | 2 days 3 hours |

```sql
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    event_name VARCHAR(100),
    event_date DATE,
    start_time TIME,
    created_at TIMESTAMP DEFAULT NOW(),
    scheduled_at TIMESTAMPTZ,
    duration INTERVAL
);

INSERT INTO events (event_name, event_date, start_time, duration) VALUES
('Meeting', '2024-01-20', '14:00:00', '2 hours'),
('Workshop', '2024-01-25', '09:00:00', '1 day');
```

### Date/Time Functions

```sql
-- Current time
SELECT NOW();                    -- 2024-01-15 14:30:00.123456+09
SELECT CURRENT_DATE;             -- 2024-01-15
SELECT CURRENT_TIME;             -- 14:30:00.123456+09
SELECT CURRENT_TIMESTAMP;        -- same as NOW()

-- Date arithmetic
SELECT NOW() + INTERVAL '1 day';
SELECT NOW() - INTERVAL '2 hours';
SELECT '2024-01-20'::DATE - '2024-01-15'::DATE;  -- 5 (number of days)

-- Date extraction
SELECT EXTRACT(YEAR FROM NOW());
SELECT EXTRACT(MONTH FROM NOW());
SELECT EXTRACT(DOW FROM NOW());  -- day of week (0 = Sunday)
```

---

## 6. Boolean Data Type

| Value | TRUE | FALSE | NULL |
|-------|------|-------|------|
| Input | true, 't', 'yes', 'y', '1' | false, 'f', 'no', 'n', '0' | null |

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false
);

INSERT INTO users (name, is_active, is_admin) VALUES
('김철수', true, false),
('Admin', true, true);

SELECT * FROM users WHERE is_active = true;
SELECT * FROM users WHERE NOT is_admin;
```

---

## 7. JSON Data Types

| Type | Description |
|------|-------------|
| `JSON` | Stores JSON as text (parsed on every access) |
| `JSONB` | Stores JSON in binary format (indexable, recommended) |

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    attributes JSONB
);

INSERT INTO products (name, attributes) VALUES
('Laptop', '{"brand": "Samsung", "ram": 16, "storage": "512GB"}'),
('Mouse', '{"brand": "Logitech", "wireless": true, "color": "black"}');

-- Query JSON data
SELECT name, attributes->>'brand' AS brand FROM products;
SELECT name, attributes->'ram' AS ram FROM products;

-- JSON conditional search
SELECT * FROM products WHERE attributes->>'brand' = 'Samsung';
SELECT * FROM products WHERE (attributes->>'ram')::int >= 16;

-- JSON array
INSERT INTO products (name, attributes) VALUES
('Keyboard', '{"brand": "Keychron", "colors": ["white", "black", "gray"]}');

SELECT attributes->'colors'->0 FROM products WHERE name = 'Keyboard';  -- "white"
```

---

## 8. Other Data Types

### UUID

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO sessions (user_id) VALUES (1);
-- id: 550e8400-e29b-41d4-a716-446655440000
```

### Arrays

```sql
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    tags TEXT[]
);

INSERT INTO posts (title, tags) VALUES
('PostgreSQL Basics', ARRAY['database', 'postgresql', 'sql']),
('Getting Started with Docker', '{"docker", "container", "devops"}');

-- Array query
SELECT title, tags[1] FROM posts;  -- first element

-- Array containment check
SELECT * FROM posts WHERE 'docker' = ANY(tags);
SELECT * FROM posts WHERE tags @> ARRAY['sql'];
```

### ENUM

```sql
CREATE TYPE mood AS ENUM ('happy', 'sad', 'neutral');

CREATE TABLE user_moods (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    current_mood mood
);

INSERT INTO user_moods (user_id, current_mood) VALUES (1, 'happy');
```

---

## 9. Constraints

### PRIMARY KEY

```sql
-- Single column primary key
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- Composite primary key
CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);
```

### NOT NULL

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,  -- NULL not allowed
    email VARCHAR(255) NOT NULL
);
```

### UNIQUE

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,  -- no duplicates allowed
    phone VARCHAR(20) UNIQUE             -- no duplicates (multiple NULLs are allowed)
);

-- Composite unique
CREATE TABLE memberships (
    user_id INTEGER,
    group_id INTEGER,
    UNIQUE (user_id, group_id)
);
```

### DEFAULT

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'pending',
    quantity INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO orders DEFAULT VALUES;  -- use default values for all columns
```

### CHECK

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price NUMERIC(10, 2) CHECK (price > 0),
    quantity INTEGER CHECK (quantity >= 0),
    discount NUMERIC(3, 2) CHECK (discount >= 0 AND discount <= 1)
);

-- Named constraints
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    salary NUMERIC(10, 2),
    CONSTRAINT valid_age CHECK (age >= 18 AND age <= 100),
    CONSTRAINT positive_salary CHECK (salary > 0)
);
```

### FOREIGN KEY

```sql
-- Parent table
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- Child table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category_id INTEGER REFERENCES categories(id)
);

-- Choose ON DELETE action based on business rules:
-- CASCADE: child data is meaningless without parent (e.g., order_items without order)
-- SET NULL: child can exist independently (e.g., products when category is removed)
-- RESTRICT: deletion should be blocked if children exist (safest default)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(id)
        ON DELETE CASCADE      -- delete children when parent is deleted
        ON UPDATE CASCADE      -- update children when parent is updated
);
```

### ON DELETE / ON UPDATE Options

| Option | Description |
|--------|-------------|
| `CASCADE` | Delete/update along with parent |
| `SET NULL` | Set to NULL |
| `SET DEFAULT` | Set to default value |
| `RESTRICT` | Prevent delete/update (default) |
| `NO ACTION` | Similar to RESTRICT |

---

## 10. Table Modification

### Add Column

```sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT false;
```

### Drop Column

```sql
ALTER TABLE users DROP COLUMN phone;
ALTER TABLE users DROP COLUMN IF EXISTS phone;
```

### Change Column Type

```sql
ALTER TABLE users ALTER COLUMN name TYPE VARCHAR(200);
ALTER TABLE users ALTER COLUMN age TYPE SMALLINT;

-- When data conversion is needed
ALTER TABLE users ALTER COLUMN price TYPE INTEGER USING price::INTEGER;
```

### Rename Column

```sql
ALTER TABLE users RENAME COLUMN name TO full_name;
```

### Add/Drop Constraints

```sql
-- Add NOT NULL
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- Drop NOT NULL
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;

-- Set DEFAULT
ALTER TABLE users ALTER COLUMN status SET DEFAULT 'active';

-- Drop DEFAULT
ALTER TABLE users ALTER COLUMN status DROP DEFAULT;

-- Add constraint
ALTER TABLE users ADD CONSTRAINT users_email_unique UNIQUE (email);
ALTER TABLE users ADD CONSTRAINT valid_age CHECK (age >= 0);

-- Drop constraint
ALTER TABLE users DROP CONSTRAINT users_email_unique;
```

### Rename Table

```sql
ALTER TABLE users RENAME TO members;
```

---

## 11. Table Deletion

```sql
-- Basic drop
DROP TABLE users;

-- Drop only if it exists
DROP TABLE IF EXISTS users;

-- Drop with dependent objects
DROP TABLE users CASCADE;
```

---

## 12. Table Information

```sql
-- List tables
\dt

-- Table structure
\d users

-- Detailed information
\d+ users

-- Check via SQL query
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'users';
```

---

## 13. Practice Examples

### Practice: Online Shopping Mall Table Design

```sql
-- 1. Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. Categories table
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES categories(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. Products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price NUMERIC(12, 2) NOT NULL CHECK (price >= 0),
    stock INTEGER DEFAULT 0 CHECK (stock >= 0),
    attributes JSONB,
    is_available BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 4. Orders table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    status VARCHAR(20) DEFAULT 'pending' CHECK (
        status IN ('pending', 'paid', 'shipped', 'delivered', 'cancelled')
    ),
    total_amount NUMERIC(12, 2) NOT NULL,
    shipping_address TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 5. Order items table
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Check table structure
\dt
\d products
```

---

**Previous**: [Database Management](./02_Database_Management.md) | **Next**: [CRUD Basics](./04_CRUD_Basics.md)
