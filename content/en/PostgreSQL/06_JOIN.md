# JOIN

**Previous**: [Conditions and Sorting](./05_Conditions_and_Sorting.md) | **Next**: [Aggregation and Grouping](./07_Aggregation_and_Grouping.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of JOIN and how it connects rows from two or more tables
2. Distinguish among INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, and CROSS JOIN
3. Write multi-table JOIN queries using table aliases and explicit ON conditions
4. Apply SELF JOIN to model hierarchical relationships (e.g., employee-manager)
5. Identify the difference between filtering in the ON clause versus the WHERE clause for outer joins
6. Use the USING clause and understand when NATURAL JOIN is appropriate (and when it is not)
7. Create indexes on foreign key columns to optimize JOIN performance

---

Real-world data is rarely stored in a single table. Customers, orders, and products each live in their own table, and the power of a relational database comes from connecting them on the fly. JOIN is the SQL mechanism that reassembles related data from multiple tables into a single, meaningful result set -- making it one of the most important operations you will use every day.

## 1. JOIN Concept

JOIN is a method to connect two or more tables to query data.

```
┌─────────────────┐     ┌─────────────────┐
│     users       │     │     orders      │
├─────────────────┤     ├─────────────────┤
│ id │ name       │     │ id │ user_id    │
├────┼────────────┤     ├────┼────────────┤
│ 1  │ John Kim   │◄────│ 1  │ 1          │
│ 2  │ Jane Lee   │◄────│ 2  │ 1          │
│ 3  │ Mike Park  │     │ 3  │ 2          │
└────┴────────────┘     └────┴────────────┘
         ↑ users.id = orders.user_id
```

---

## 2. Practice Table Setup

```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255)
);

-- Orders table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    product_name VARCHAR(200),
    amount NUMERIC(10, 2),
    order_date DATE DEFAULT CURRENT_DATE
);

-- Sample data
INSERT INTO users (name, email) VALUES
('John Kim', 'kim@email.com'),
('Jane Lee', 'lee@email.com'),
('Mike Park', 'park@email.com'),
('Sarah Choi', 'choi@email.com');  -- User with no orders

INSERT INTO orders (user_id, product_name, amount) VALUES
(1, 'Laptop', 1500000),
(1, 'Mouse', 50000),
(2, 'Keyboard', 100000),
(2, 'Monitor', 300000),
(3, 'Headset', 150000),
(NULL, 'Gift Set', 80000);  -- Order without user
```

---

> **Analogy -- Venn Diagrams for JOINs**: Think of each table as a circle in a Venn diagram. An INNER JOIN returns the overlapping region -- rows that match in both tables. A LEFT JOIN returns the entire left circle plus the overlap, filling in NULL where the right circle has no match. Understanding JOINs as set operations makes it easy to predict which rows appear in your result.

## 3. INNER JOIN

Returns only data that matches in both tables.

```sql
-- Basic syntax
SELECT columns
FROM table1
INNER JOIN table2 ON table1.column = table2.column;

-- Query user and order information
SELECT
    users.name,
    users.email,
    orders.product_name,
    orders.amount
FROM users
INNER JOIN orders ON users.id = orders.user_id;
```

Result:
```
  name    │      email       │ product_name │  amount
──────────┼──────────────────┼──────────────┼──────────
 John Kim │ kim@email.com    │ Laptop       │ 1500000
 John Kim │ kim@email.com    │ Mouse        │   50000
 Jane Lee │ lee@email.com    │ Keyboard     │  100000
 Jane Lee │ lee@email.com    │ Monitor      │  300000
 Mike Park│ park@email.com   │ Headset      │  150000
```

### Use Table Aliases

```sql
SELECT u.name, o.product_name, o.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;
```

### JOIN Implies INNER JOIN

```sql
-- INNER can be omitted
SELECT u.name, o.product_name
FROM users u
JOIN orders o ON u.id = o.user_id;
```

---

## 4. LEFT (OUTER) JOIN

Returns all rows from left table + matching rows from right table.
Unmatched rows are filled with NULL.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
```

Result:
```
   name     │ product_name │  amount
────────────┼──────────────┼──────────
 John Kim   │ Laptop       │ 1500000
 John Kim   │ Mouse        │   50000
 Jane Lee   │ Keyboard     │  100000
 Jane Lee   │ Monitor      │  300000
 Mike Park  │ Headset      │  150000
 Sarah Choi │ NULL         │ NULL      ← User with no orders included
```

### Find Users Without Orders

```sql
SELECT u.name, u.email
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;
```

---

## 5. RIGHT (OUTER) JOIN

Returns all rows from right table + matching rows from left table.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;
```

Result:
```
   name    │ product_name │  amount
───────────┼──────────────┼──────────
 John Kim  │ Laptop       │ 1500000
 John Kim  │ Mouse        │   50000
 Jane Lee  │ Keyboard     │  100000
 Jane Lee  │ Monitor      │  300000
 Mike Park │ Headset      │  150000
 NULL      │ Gift Set     │   80000   ← Order without user included
```

---

## 6. FULL (OUTER) JOIN

Returns all rows from both tables. Unmatched rows are filled with NULL.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
FULL JOIN orders o ON u.id = o.user_id;
```

Result:
```
   name     │ product_name │  amount
────────────┼──────────────┼──────────
 John Kim   │ Laptop       │ 1500000
 John Kim   │ Mouse        │   50000
 Jane Lee   │ Keyboard     │  100000
 Jane Lee   │ Monitor      │  300000
 Mike Park  │ Headset      │  150000
 Sarah Choi │ NULL         │ NULL      ← User without orders
 NULL       │ Gift Set     │   80000   ← Order without user
```

---

## 7. CROSS JOIN

Returns all possible combinations (Cartesian product).

```sql
-- Color and size tables
CREATE TABLE colors (name VARCHAR(20));
CREATE TABLE sizes (name VARCHAR(10));

INSERT INTO colors VALUES ('Red'), ('Blue'), ('Black');
INSERT INTO sizes VALUES ('S'), ('M'), ('L');

-- All combinations
SELECT c.name AS color, s.name AS size
FROM colors c
CROSS JOIN sizes s;
```

Result:
```
 color │ size
───────┼──────
 Red   │ S
 Red   │ M
 Red   │ L
 Blue  │ S
 Blue  │ M
 Blue  │ L
 Black │ S
 Black │ M
 Black │ L
```

---

## 8. SELF JOIN

Joins a table with itself.

```sql
-- Employee-Manager relationship
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    manager_id INTEGER REFERENCES employees(id)
);

INSERT INTO employees (name, manager_id) VALUES
('CEO', NULL),
('VP', 1),
('Manager A', 2),
('Manager B', 2),
('Employee', 3);

-- Query employee and manager names
SELECT
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

Result:
```
  employee  │ manager
────────────┼─────────
 CEO        │ NULL
 VP         │ CEO
 Manager A  │ VP
 Manager B  │ VP
 Employee   │ Manager A
```

---

## 9. Multiple Table JOIN

Connect 3 or more tables.

```sql
-- Add category table
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- Products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name VARCHAR(200),
    price NUMERIC(10, 2)
);

-- Order items table
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER
);

-- JOIN 3 tables
SELECT
    u.name AS user_name,
    p.name AS product_name,
    c.name AS category_name,
    oi.quantity
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
JOIN categories c ON p.category_id = c.id;
```

---

## 10. JOIN Conditions and WHERE

### ON vs WHERE

```sql
-- ON: Table join condition
-- WHERE: Result filtering

-- LEFT JOIN + WHERE
SELECT u.name, o.product_name, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.amount > 100000;  -- NULL rows removed

-- LEFT JOIN + Additional condition in ON
SELECT u.name, o.product_name, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id AND o.amount > 100000;
-- All users retained, only matching orders connected
```

### Composite JOIN Conditions

```sql
SELECT *
FROM table1 t1
JOIN table2 t2 ON t1.col1 = t2.col1 AND t1.col2 = t2.col2;
```

---

## 11. USING Clause

Simplifies joins when column names are the same.

```sql
-- Using ON
SELECT * FROM orders o
JOIN users u ON o.user_id = u.id;

-- Using USING (when column names match)
-- If orders.user_id and users.user_id are the same:
SELECT * FROM orders
JOIN users USING (user_id);
```

---

## 12. NATURAL JOIN

Automatically joins on all columns with the same name. (Not recommended)

```sql
-- Joins on all columns with same name
SELECT * FROM orders
NATURAL JOIN users;

-- May produce unintended results, explicit ON recommended
```

---

## 13. JOIN Visualization

```
INNER JOIN:         LEFT JOIN:          RIGHT JOIN:         FULL JOIN:
    ┌───┐              ┌───┐              ┌───┐              ┌───┐
   ┌┼───┼┐            ┌┼───┼┐            ┌┼───┼┐            ┌┼───┼┐
  ┌┼│███│┼┐          ┌┼│███│ │          │ │███│┼┐          ┌┼│███│┼┐
  │ │███│ │          ││████│ │          │ │████││          ││█████││
  └┼│███│┼┘          └┼│███│ │          │ │███│┼┘          └┼│███│┼┘
   └┼───┼┘            └┼───┘ │          │ └───┼┘            └─────┼┘
    └───┘              └─────┘          └─────┘              └─────┘
   A ∩ B               All A            All B              A ∪ B
```

---

## 14. Practice Examples

### Practice 1: Basic JOIN

```sql
-- 1. Users who have ordered and their order info
SELECT u.name, o.product_name, o.amount, o.order_date
FROM users u
INNER JOIN orders o ON u.id = o.user_id
ORDER BY o.order_date DESC;

-- 2. Total order amount per user
SELECT u.name, SUM(o.amount) AS total_amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_amount DESC;
```

### Practice 2: OUTER JOIN

```sql
-- 1. All users (regardless of orders)
SELECT
    u.name,
    COALESCE(SUM(o.amount), 0) AS total_amount,
    COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_amount DESC;

-- 2. Find users who haven't ordered
SELECT u.name, u.email
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;

-- 3. Find orders without users
SELECT o.id, o.product_name, o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id
WHERE u.id IS NULL;
```

### Practice 3: Complex Condition JOIN

```sql
-- 1. Users who ordered 1,000,000 or more
SELECT DISTINCT u.name, u.email
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.amount >= 1000000;

-- 2. Users who ordered within last 30 days
SELECT DISTINCT u.name
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days';
```

### Practice 4: Multiple Table JOIN

```sql
-- Connect categories → products → orders
SELECT
    c.name AS category,
    p.name AS product,
    u.name AS customer,
    oi.quantity,
    p.price * oi.quantity AS subtotal
FROM categories c
JOIN products p ON c.id = p.category_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
JOIN users u ON o.user_id = u.id
ORDER BY c.name, p.name;
```

---

## 15. Performance Considerations

### Use Indexes

```sql
-- Create indexes on foreign key columns
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
```

### SELECT Only Needed Columns

```sql
-- Bad example
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

-- Good example
SELECT u.name, o.product_name, o.amount
FROM users u JOIN orders o ON u.id = o.user_id;
```

### Check Execution Plan with EXPLAIN

```sql
EXPLAIN SELECT u.name, o.product_name
FROM users u
JOIN orders o ON u.id = o.user_id;
```

---

**Previous**: [Conditions and Sorting](./05_Conditions_and_Sorting.md) | **Next**: [Aggregation and Grouping](./07_Aggregation_and_Grouping.md)
