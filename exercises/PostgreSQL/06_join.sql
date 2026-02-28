-- Exercises for Lesson 06: JOIN
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- Setup: Create sample tables for JOIN exercises
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    product_name VARCHAR(200),
    amount INTEGER,
    order_date DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name VARCHAR(200) NOT NULL,
    price NUMERIC(12, 2)
);

CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0)
);


-- === Exercise 1: Basic JOIN ===
-- Problem: Show users with their orders and calculate totals per user.

-- Solution:

-- 1. Users who have ordered and their order info
-- INNER JOIN excludes users with zero orders
SELECT u.name, o.product_name, o.amount, o.order_date
FROM users u
INNER JOIN orders o ON u.id = o.user_id
ORDER BY o.order_date DESC;

-- 2. Total order amount per user
-- GROUP BY u.id ensures we aggregate per user even if names are duplicated
SELECT u.name, SUM(o.amount) AS total_amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_amount DESC;


-- === Exercise 2: OUTER JOIN ===
-- Problem: Include all users regardless of orders; find users without orders.

-- Solution:

-- 1. All users with their order summary (including those with no orders)
-- COALESCE converts NULL sum to 0 for users who never ordered
SELECT
    u.name,
    COALESCE(SUM(o.amount), 0) AS total_amount,
    COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_amount DESC;

-- 2. Find users who haven't placed any orders
-- The LEFT JOIN + WHERE o.id IS NULL pattern is an "anti-join":
-- it returns rows from the left table with no match on the right
SELECT u.name, u.email
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;

-- 3. Find orphan orders (orders without valid users)
-- RIGHT JOIN keeps all orders; WHERE u.id IS NULL filters to unmatched ones
SELECT o.id, o.product_name, o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id
WHERE u.id IS NULL;


-- === Exercise 3: Complex Condition JOIN ===
-- Problem: Join with additional WHERE filters.

-- Solution:

-- 1. Users who ordered 1,000,000 or more
-- DISTINCT avoids duplicate rows when a user has multiple high-value orders
SELECT DISTINCT u.name, u.email
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.amount >= 1000000;

-- 2. Users who ordered within the last 30 days
SELECT DISTINCT u.name
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days';


-- === Exercise 4: Multiple Table JOIN ===
-- Problem: Join 5 tables to build a complete order report.

-- Solution:

-- Trace the full path: category -> product -> order_item -> order -> user
-- This gives a denormalized view of each line item with its category and buyer
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
