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
