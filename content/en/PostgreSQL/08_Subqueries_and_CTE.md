# Subqueries and CTE

**Previous**: [Aggregation and Grouping](./07_Aggregation_and_Grouping.md) | **Next**: [Views and Indexes](./09_Views_and_Indexes.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a subquery is and identify where it can appear (WHERE, FROM, SELECT)
2. Write scalar, multi-row, and correlated subqueries
3. Use EXISTS / NOT EXISTS and compare their behavior to IN / NOT IN
4. Apply FROM-clause subqueries (inline views) to build intermediate result sets
5. Rewrite complex subqueries as CTEs using the WITH clause for improved readability
6. Chain multiple CTEs in a single query
7. Implement recursive CTEs (WITH RECURSIVE) for hierarchical data traversal
8. Choose between subqueries, CTEs, and JOINs based on readability and performance needs

---

As your SQL questions grow more complex, you will often need the answer to one question before you can ask another. Subqueries let you embed one query inside another, while Common Table Expressions (CTEs) let you name and reuse intermediate results. Together, they are the tools that turn a single SQL statement into a structured, multi-step reasoning process.

## 1. What is a Subquery?

A subquery is another query contained within a query.

```sql
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders);  -- Subquery
          ↑
    Query in parentheses
```

---

## 2. WHERE Clause Subqueries

### Scalar Subquery (Single Value)

```sql
-- Products more expensive than average
SELECT * FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- Orders on the latest order date
SELECT * FROM orders
WHERE order_date = (SELECT MAX(order_date) FROM orders);
```

### Multi-Row Subquery

```sql
-- Users who have ordered
SELECT * FROM users
WHERE id IN (SELECT DISTINCT user_id FROM orders);

-- Users who purchased electronics
SELECT * FROM users
WHERE id IN (
    SELECT o.user_id FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    WHERE p.category = 'Electronics'
);
```

### NOT IN

```sql
-- Users who have never ordered
SELECT * FROM users
WHERE id NOT IN (
    SELECT user_id FROM orders WHERE user_id IS NOT NULL
);
-- Caution: NOT IN with NULL may return empty results
```

### ANY / SOME

```sql
-- Furniture more expensive than any electronics item
SELECT * FROM products
WHERE category = 'Furniture'
  AND price > ANY (SELECT price FROM products WHERE category = 'Electronics');
-- = ANY is same as IN
```

### ALL

```sql
-- Products more expensive than all electronics items
SELECT * FROM products
WHERE price > ALL (SELECT price FROM products WHERE category = 'Electronics');
```

---

## 3. EXISTS / NOT EXISTS

Checks only for row existence.

```sql
-- Users with orders
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);

-- Users without orders
SELECT * FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);
```

### IN vs EXISTS

```sql
-- IN: Load subquery results into memory
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders);

-- EXISTS: Check existence for each row
SELECT * FROM users u
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);

-- Generally:
-- - Use IN if subquery results are small
-- - Use EXISTS if subquery results are large
-- - Prefer NOT EXISTS over NOT IN (avoid NULL issues)
```

---

## 4. FROM Clause Subquery (Inline View)

```sql
-- Calculate average price by category, then filter
SELECT *
FROM (
    SELECT category, AVG(price) AS avg_price
    FROM products
    GROUP BY category
) AS category_avg
WHERE avg_price > 100000;

-- Alias required for subquery (AS category_avg)
```

### JOIN After Complex Aggregation

```sql
-- Combine user stats with user info
SELECT
    u.name,
    u.email,
    stats.order_count,
    stats.total_amount
FROM users u
JOIN (
    SELECT
        user_id,
        COUNT(*) AS order_count,
        SUM(amount) AS total_amount
    FROM orders
    GROUP BY user_id
) AS stats ON u.id = stats.user_id;
```

---

## 5. SELECT Clause Subquery (Scalar Subquery)

```sql
-- Display each product with category average price
SELECT
    name,
    price,
    (SELECT AVG(price) FROM products p2 WHERE p2.category = p.category) AS category_avg
FROM products p;

-- Order count for each user
SELECT
    u.name,
    (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS order_count
FROM users u;
```

---

## 6. Correlated Subquery

A subquery that references values from the outer query.

```sql
-- Products more expensive than their category average
SELECT * FROM products p
WHERE price > (
    SELECT AVG(price) FROM products WHERE category = p.category
);
--                                                    ↑ Reference outer query

-- Most expensive product in each category
SELECT * FROM products p
WHERE price = (
    SELECT MAX(price) FROM products WHERE category = p.category
);
```

---

> **Analogy -- SQL Thinks in Sets**: A subquery is like a question within a question: "What is the average salary of employees in departments that have more than 10 people?" You answer the inner question first (which departments?), then use that answer for the outer question. A CTE (Common Table Expression) is like giving that inner answer a name so you can refer to it multiple times -- like writing an intermediate result on a whiteboard before using it in the final calculation.

## 7. CTE (Common Table Expression)

Uses WITH clause to name temporary result sets.

### Basic CTE

```sql
-- Subquery approach
SELECT * FROM (
    SELECT category, AVG(price) AS avg_price
    FROM products
    GROUP BY category
) AS category_stats
WHERE avg_price > 100000;

-- CTE approach (more readable)
WITH category_stats AS (
    SELECT category, AVG(price) AS avg_price
    FROM products
    GROUP BY category
)
SELECT * FROM category_stats
WHERE avg_price > 100000;
```

### Multiple CTEs

```sql
WITH
-- Category statistics
category_stats AS (
    SELECT
        category,
        COUNT(*) AS product_count,
        AVG(price) AS avg_price
    FROM products
    GROUP BY category
),
-- Expensive products (1,000,000+)
expensive_products AS (
    SELECT * FROM products WHERE price >= 1000000
)
SELECT
    cs.category,
    cs.product_count,
    cs.avg_price,
    COUNT(ep.id) AS expensive_count
FROM category_stats cs
LEFT JOIN expensive_products ep ON cs.category = ep.category
GROUP BY cs.category, cs.product_count, cs.avg_price;
```

### CTE with Main Query

```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT
    month,
    total,
    LAG(total) OVER (ORDER BY month) AS prev_month,
    total - LAG(total) OVER (ORDER BY month) AS diff
FROM monthly_sales
ORDER BY month;
```

---

## 8. Recursive CTE (WITH RECURSIVE)

A CTE that references itself.

### Organization Chart Traversal

```sql
-- Employees table
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    manager_id INTEGER REFERENCES employees(id)
);

INSERT INTO employees (name, manager_id) VALUES
('CEO', NULL),
('CTO', 1),
('Dev Manager', 2),
('Developer A', 3),
('Developer B', 3),
('CFO', 1),
('Finance Manager', 6);

-- Query all subordinates from CEO
WITH RECURSIVE org_tree AS (
    -- Base case: CEO
    SELECT id, name, manager_id, 1 AS level, name::TEXT AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: subordinates
    SELECT
        e.id,
        e.name,
        e.manager_id,
        ot.level + 1,
        ot.path || ' > ' || e.name
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT
    REPEAT('  ', level - 1) || name AS org_chart,
    level,
    path
FROM org_tree
ORDER BY path;
```

Result:
```
     org_chart      │ level │           path
────────────────────┼───────┼──────────────────────────
 CEO                │     1 │ CEO
   CFO              │     2 │ CEO > CFO
     Finance Manager│     3 │ CEO > CFO > Finance Manager
   CTO              │     2 │ CEO > CTO
     Dev Manager    │     3 │ CEO > CTO > Dev Manager
       Developer A  │     4 │ CEO > CTO > Dev Manager > Developer A
       Developer B  │     4 │ CEO > CTO > Dev Manager > Developer B
```

### Generate Number Sequence

```sql
-- 1 to 10
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 10
)
SELECT * FROM numbers;
```

### Generate Date Range

```sql
-- Last 7 days
WITH RECURSIVE date_range AS (
    SELECT CURRENT_DATE - INTERVAL '6 days' AS date
    UNION ALL
    SELECT date + INTERVAL '1 day'
    FROM date_range
    WHERE date < CURRENT_DATE
)
SELECT date::DATE FROM date_range;
```

---

## 9. Practice Examples

### Sample Data

```sql
-- Create tables
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department_id INTEGER REFERENCES departments(id),
    salary NUMERIC(10, 2),
    hire_date DATE
);

-- Insert data
INSERT INTO departments (name) VALUES
('Development'), ('Marketing'), ('HR'), ('Finance');

INSERT INTO employees (name, department_id, salary, hire_date) VALUES
('Kim Dev', 1, 5000000, '2020-03-15'),
('Lee Dev', 1, 4500000, '2021-06-20'),
('Park Marketing', 2, 4000000, '2019-11-10'),
('Choi Marketing', 2, 3800000, '2022-01-05'),
('Jung HR', 3, 3500000, '2020-08-25'),
('Han Finance', 4, 4200000, '2021-03-10'),
('Oh Finance', 4, 3900000, '2022-07-15');
```

### Practice 1: WHERE Subqueries

```sql
-- 1. Employees with salary higher than average
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- 2. Most recently hired employee
SELECT * FROM employees
WHERE hire_date = (SELECT MAX(hire_date) FROM employees);

-- 3. Employees in Development or Marketing
SELECT * FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE name IN ('Development', 'Marketing')
);
```

### Practice 2: Correlated Subqueries

```sql
-- 1. Employees with salary higher than their department average
SELECT
    e.name,
    e.salary,
    d.name AS department
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department_id = e.department_id
);

-- 2. Highest paid employee in each department
SELECT * FROM employees e
WHERE salary = (
    SELECT MAX(salary)
    FROM employees
    WHERE department_id = e.department_id
);
```

### Practice 3: CTE Usage

```sql
-- 1. Query employee info with department stats
WITH dept_stats AS (
    SELECT
        department_id,
        AVG(salary) AS avg_salary,
        COUNT(*) AS emp_count
    FROM employees
    GROUP BY department_id
)
SELECT
    e.name,
    e.salary,
    d.name AS department,
    ds.avg_salary AS dept_avg,
    ds.emp_count AS dept_count
FROM employees e
JOIN departments d ON e.department_id = d.id
JOIN dept_stats ds ON e.department_id = ds.department_id;

-- 2. Query with salary ranking
WITH ranked_employees AS (
    SELECT
        *,
        RANK() OVER (ORDER BY salary DESC) AS salary_rank,
        RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
    FROM employees
)
SELECT
    name,
    salary,
    salary_rank AS overall_rank,
    dept_rank AS dept_rank
FROM ranked_employees
ORDER BY salary_rank;
```

### Practice 4: Complex Usage

```sql
-- Employees with above-average salary in their department and the difference
WITH
dept_avg AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
),
above_avg AS (
    SELECT
        e.*,
        da.avg_salary,
        e.salary - da.avg_salary AS diff
    FROM employees e
    JOIN dept_avg da ON e.department_id = da.department_id
    WHERE e.salary >= da.avg_salary
)
SELECT
    aa.name,
    d.name AS department,
    aa.salary,
    ROUND(aa.avg_salary, 0) AS dept_avg,
    ROUND(aa.diff, 0) AS above_avg_by
FROM above_avg aa
JOIN departments d ON aa.department_id = d.id
ORDER BY aa.diff DESC;
```

---

## 10. Subquery vs CTE vs JOIN

| Situation | Recommended |
|-----------|-------------|
| Simple value comparison | Subquery |
| Multiple references | CTE |
| Table connection | JOIN |
| Separate complex logic | CTE |
| Recursive traversal | WITH RECURSIVE |

---

**Previous**: [Aggregation and Grouping](./07_Aggregation_and_Grouping.md) | **Next**: [Views and Indexes](./09_Views_and_Indexes.md)
