# Conditions and Sorting

**Previous**: [CRUD Basics](./04_CRUD_Basics.md) | **Next**: [JOIN](./06_JOIN.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write WHERE clauses using comparison operators (`=`, `<>`, `<`, `>`, `<=`, `>=`)
2. Combine conditions with logical operators (AND, OR, NOT) and apply correct precedence
3. Use BETWEEN, IN, and LIKE/ILIKE for range checks, set membership, and pattern matching
4. Handle NULL values correctly with IS NULL, IS NOT NULL, COALESCE, and NULLIF
5. Sort query results with ORDER BY including multi-column and expression-based ordering
6. Implement pagination using LIMIT, OFFSET, and the SQL-standard FETCH syntax
7. Apply DISTINCT and DISTINCT ON to eliminate duplicate rows from result sets

---

Raw data in a table is only useful when you can filter, sort, and page through it efficiently. In practice, almost every query you write will include a WHERE clause to narrow down results and an ORDER BY to present them in a meaningful sequence. These filtering and sorting skills are the bridge between storing data and extracting actionable information from it.

## 1. WHERE Clause Basics

The WHERE clause selects only rows that match a condition.

```sql
SELECT * FROM users WHERE condition;
UPDATE users SET ... WHERE condition;
DELETE FROM users WHERE condition;
```

---

## 2. Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Equal | `age = 30` |
| `<>` or `!=` | Not equal | `city <> 'Seoul'` |
| `<` | Less than | `age < 30` |
| `>` | Greater than | `age > 30` |
| `<=` | Less than or equal | `age <= 30` |
| `>=` | Greater than or equal | `age >= 30` |

```sql
-- Users with age 30
SELECT * FROM users WHERE age = 30;

-- Users not age 30
SELECT * FROM users WHERE age <> 30;
SELECT * FROM users WHERE age != 30;

-- Age between 25 and 35
SELECT * FROM users WHERE age >= 25 AND age <= 35;
```

---

## 3. Logical Operators

### AND

All conditions must be true.

```sql
-- People in Seoul in their 30s
SELECT * FROM users
WHERE city = 'Seoul' AND age >= 30 AND age < 40;
```

### OR

At least one condition must be true.

```sql
-- Users in Seoul or Busan
SELECT * FROM users
WHERE city = 'Seoul' OR city = 'Busan';
```

### NOT

Negates a condition.

```sql
-- Users not in Seoul
SELECT * FROM users WHERE NOT city = 'Seoul';
SELECT * FROM users WHERE city <> 'Seoul';  -- Same

-- Users not 30 or older
SELECT * FROM users WHERE NOT age >= 30;
SELECT * FROM users WHERE age < 30;  -- Same
```

### Operator Precedence

Processed in order: `NOT` > `AND` > `OR`. Use parentheses for clarity.

```sql
-- May not work as intended
SELECT * FROM users WHERE city = 'Seoul' OR city = 'Busan' AND age >= 30;
-- Actually: All of Seoul OR (Busan AND 30+)

-- Clear with parentheses
SELECT * FROM users WHERE (city = 'Seoul' OR city = 'Busan') AND age >= 30;
```

---

## 4. BETWEEN

Simplifies range conditions.

```sql
-- Age between 25 and 35
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
-- Same as: WHERE age >= 25 AND age <= 35

-- NOT BETWEEN
SELECT * FROM users WHERE age NOT BETWEEN 25 AND 35;

-- Date range
SELECT * FROM orders
WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31';
```

---

## 5. IN

Checks if value matches any in a list.

```sql
-- One of Seoul, Busan, Daejeon
SELECT * FROM users WHERE city IN ('Seoul', 'Busan', 'Daejeon');
-- Same as: WHERE city = 'Seoul' OR city = 'Busan' OR city = 'Daejeon'

-- NOT IN
SELECT * FROM users WHERE city NOT IN ('Seoul', 'Busan');

-- Can use with numbers too
SELECT * FROM users WHERE age IN (25, 30, 35);

-- With subquery
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);
```

---

## 6. LIKE - Pattern Matching

### Wildcards

| Symbol | Meaning |
|--------|---------|
| `%` | Zero or more characters |
| `_` | Exactly one character |

```sql
-- Names starting with 'Kim'
SELECT * FROM users WHERE name LIKE 'Kim%';

-- Names ending with 'su'
SELECT * FROM users WHERE name LIKE '%su';

-- Names containing 'young'
SELECT * FROM users WHERE name LIKE '%young%';

-- Exactly 3 character names
SELECT * FROM users WHERE name LIKE '___';

-- 2 character names starting with 'Kim'
SELECT * FROM users WHERE name LIKE 'Kim_';
```

### ILIKE - Case Insensitive

```sql
-- Case insensitive search (PostgreSQL specific)
SELECT * FROM users WHERE email ILIKE '%KIM%';
SELECT * FROM users WHERE email ILIKE 'kim@%';
```

### NOT LIKE

```sql
SELECT * FROM users WHERE name NOT LIKE 'Kim%';
```

### Escape

```sql
-- When searching for actual % or _
SELECT * FROM products WHERE name LIKE '%50\%%' ESCAPE '\';  -- Contains 50%
```

---

## 7. NULL Handling

NULL is an "unknown value" and cannot be compared with regular comparison operators.

### IS NULL / IS NOT NULL

```sql
-- Users with NULL city
SELECT * FROM users WHERE city IS NULL;

-- Users with non-NULL city
SELECT * FROM users WHERE city IS NOT NULL;

-- Wrong example (always false)
SELECT * FROM users WHERE city = NULL;  -- Doesn't work!
```

### COALESCE - NULL Replacement

```sql
-- COALESCE returns the first non-NULL argument — essential for user-facing output
-- where NULL would display as blank or cause downstream errors in application code
SELECT name, COALESCE(city, 'Unspecified') AS city FROM users;

-- Chain multiple fallbacks: try phone first, then email, then a literal default
SELECT COALESCE(phone, email, 'No contact') AS contact FROM users;
```

### NULLIF

```sql
-- Return NULL if two values are equal
SELECT NULLIF(age, 0) FROM users;  -- NULL if age is 0

-- Prevent division by zero
SELECT total / NULLIF(count, 0) FROM stats;
```

---

## 8. ORDER BY - Sorting

### Basic Sorting

```sql
-- Ascending (default)
SELECT * FROM users ORDER BY age;
SELECT * FROM users ORDER BY age ASC;

-- Descending
SELECT * FROM users ORDER BY age DESC;

-- String sorting
SELECT * FROM users ORDER BY name;  -- Alphabetical
SELECT * FROM users ORDER BY name DESC;
```

### Multiple Column Sorting

```sql
-- Sort by city first, then by age
SELECT * FROM users ORDER BY city, age;

-- City ascending, age descending
SELECT * FROM users ORDER BY city ASC, age DESC;
```

### NULL Sorting Order

```sql
-- NULL last (default: NULL last in ASC)
SELECT * FROM users ORDER BY city NULLS LAST;

-- NULL first
SELECT * FROM users ORDER BY city NULLS FIRST;

-- NULL handling in DESC
SELECT * FROM users ORDER BY city DESC NULLS LAST;
```

### Sort by Expression

```sql
-- Sort by name length
SELECT * FROM users ORDER BY LENGTH(name);

-- Sort by calculated result
SELECT name, age, age * 12 AS months FROM users ORDER BY months DESC;

-- Sort by column position (1-based)
SELECT name, email, age FROM users ORDER BY 3 DESC;  -- Sort by age
```

---

## 9. LIMIT / OFFSET - Result Limiting

### LIMIT

```sql
-- Top 5 only
SELECT * FROM users LIMIT 5;

-- Top 3 oldest users
SELECT * FROM users ORDER BY age DESC LIMIT 3;
```

### OFFSET

```sql
-- Skip first 5, then continue
SELECT * FROM users ORDER BY id OFFSET 5;

-- Pagination: 5 rows starting from 6th
SELECT * FROM users ORDER BY id LIMIT 5 OFFSET 5;
```

### Pagination Calculation

```sql
-- Page 1 (rows 1-10)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 0;

-- Page 2 (rows 11-20)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 10;

-- Page N (calculation: OFFSET = (N-1) * page_size)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 20;  -- Page 3
```

### FETCH (SQL Standard)

```sql
-- Same as LIMIT
SELECT * FROM users
ORDER BY age DESC
FETCH FIRST 5 ROWS ONLY;

-- With OFFSET
SELECT * FROM users
ORDER BY id
OFFSET 10 ROWS
FETCH NEXT 5 ROWS ONLY;
```

---

## 10. DISTINCT - Remove Duplicates

```sql
-- Remove duplicate cities
SELECT DISTINCT city FROM users;

-- Remove duplicates of column combinations
SELECT DISTINCT city, age FROM users;

-- With COUNT
SELECT COUNT(DISTINCT city) FROM users;
```

### DISTINCT ON (PostgreSQL Specific)

```sql
-- First user per city
SELECT DISTINCT ON (city) * FROM users ORDER BY city, created_at;

-- Oldest user per city
SELECT DISTINCT ON (city) * FROM users ORDER BY city, age DESC;
```

---

## 11. Practice Examples

### Sample Data

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    price NUMERIC(10, 2),
    stock INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO products (name, category, price, stock) VALUES
('MacBook Pro 14', 'Laptop', 2490000, 50),
('MacBook Air M2', 'Laptop', 1590000, 100),
('Galaxy Book Pro', 'Laptop', 1790000, 30),
('iPad Pro', 'Tablet', 1290000, 80),
('Galaxy Tab S9', 'Tablet', 1190000, 60),
('AirPods Pro', 'Earbuds', 329000, 200),
('Galaxy Buds2', 'Earbuds', 179000, 150),
('Apple Watch 9', 'Smartwatch', 599000, 70),
('Galaxy Watch6', 'Smartwatch', 399000, 90),
('iPhone 15', 'Smartphone', 1250000, 120),
('Galaxy S24', 'Smartphone', 1150000, NULL);
```

### Practice 1: Basic Conditional Searches

```sql
-- 1. Laptop category products
SELECT * FROM products WHERE category = 'Laptop';

-- 2. Products priced 1,000,000 or more
SELECT * FROM products WHERE price >= 1000000;

-- 3. Products with stock 100+
SELECT * FROM products WHERE stock >= 100;

-- 4. Laptops priced 2,000,000 or less
SELECT * FROM products
WHERE category = 'Laptop' AND price <= 2000000;
```

### Practice 2: Complex Conditions

```sql
-- 1. Laptops or tablets
SELECT * FROM products
WHERE category IN ('Laptop', 'Tablet')
ORDER BY price DESC;

-- 2. Price between 500,000-1,500,000
SELECT * FROM products
WHERE price BETWEEN 500000 AND 1500000
ORDER BY price;

-- 3. Products with 'Pro' in name
SELECT * FROM products WHERE name LIKE '%Pro%';

-- 4. Products with NULL or 0 stock
SELECT * FROM products
WHERE stock IS NULL OR stock = 0;
```

### Practice 3: Sorting and Pagination

```sql
-- 1. Top 5 most expensive products
SELECT * FROM products ORDER BY price DESC LIMIT 5;

-- 2. By category, then price (ascending)
SELECT * FROM products ORDER BY category, price;

-- 3. Page 2 (6th-10th products)
SELECT * FROM products ORDER BY id LIMIT 5 OFFSET 5;

-- 4. Most expensive product per category
SELECT DISTINCT ON (category) *
FROM products
ORDER BY category, price DESC;
```

### Practice 4: NULL Handling

```sql
-- 1. Products with no stock or NULL
SELECT name, COALESCE(stock, 0) AS stock FROM products
WHERE stock IS NULL OR stock = 0;

-- 2. Display NULL as 'Checking stock'
SELECT name, COALESCE(stock::TEXT, 'Checking stock') AS stock_status
FROM products;

-- 3. Sort with NULL last
SELECT * FROM products ORDER BY stock NULLS LAST;
```

---

## 12. Performance Tips

### Use Indexes

```sql
-- Indexes turn O(n) sequential scans into O(log n) B-tree lookups.
-- Only create indexes on columns that appear in WHERE, JOIN, or ORDER BY clauses;
-- each index adds write overhead (INSERT/UPDATE must maintain the index).
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_price ON products(price);

-- Composite index
CREATE INDEX idx_products_cat_price ON products(category, price);
```

### LIKE Pattern Optimization

```sql
-- Prefix patterns anchor to the start — B-tree can binary-search the sorted values
WHERE name LIKE 'MacBook%'

-- Leading wildcard requires scanning every row — consider pg_trgm GIN index
-- or full-text search if this pattern is common in your workload
WHERE name LIKE '%MacBook%'
```

### Apply LIMIT First

```sql
-- LIMIT after sorting (may be inefficient)
SELECT * FROM products ORDER BY price DESC LIMIT 10;

-- Efficient with index
CREATE INDEX idx_products_price_desc ON products(price DESC);
```

---

**Previous**: [CRUD Basics](./04_CRUD_Basics.md) | **Next**: [JOIN](./06_JOIN.md)
