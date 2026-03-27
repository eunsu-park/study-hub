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
