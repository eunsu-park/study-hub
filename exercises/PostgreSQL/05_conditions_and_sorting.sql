-- Exercises for Lesson 05: Conditions and Sorting
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- Setup: Create and populate the products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50),
    price INTEGER,
    stock INTEGER
);

TRUNCATE TABLE products RESTART IDENTITY;

INSERT INTO products (name, category, price, stock) VALUES
('MacBook Pro', 'Laptop', 2590000, 50),
('MacBook Air', 'Laptop', 1590000, 100),
('Galaxy Book', 'Laptop', 1390000, 70),
('iPad Pro', 'Tablet', 1290000, 80),
('Galaxy Tab S9', 'Tablet', 1190000, 60),
('AirPods Pro', 'Earbuds', 329000, 200),
('Galaxy Buds2', 'Earbuds', 179000, 150),
('Apple Watch 9', 'Smartwatch', 599000, 70),
('Galaxy Watch6', 'Smartwatch', 399000, 90),
('iPhone 15', 'Smartphone', 1250000, 120),
('Galaxy S24', 'Smartphone', 1150000, NULL);


-- === Exercise 1: Basic Conditional Searches ===
-- Problem: Filter products by category, price, and stock thresholds.

-- Solution:

-- 1. Laptop category products
SELECT * FROM products WHERE category = 'Laptop';

-- 2. Products priced 1,000,000 or more
SELECT * FROM products WHERE price >= 1000000;

-- 3. Products with stock 100+
SELECT * FROM products WHERE stock >= 100;

-- 4. Laptops priced 2,000,000 or less
-- Combining two conditions with AND narrows the result set
SELECT * FROM products
WHERE category = 'Laptop' AND price <= 2000000;


-- === Exercise 2: Complex Conditions ===
-- Problem: Use IN, BETWEEN, LIKE, and IS NULL for advanced filtering.

-- Solution:

-- 1. Laptops or tablets, most expensive first
SELECT * FROM products
WHERE category IN ('Laptop', 'Tablet')
ORDER BY price DESC;

-- 2. Price between 500,000 and 1,500,000 (inclusive on both ends)
SELECT * FROM products
WHERE price BETWEEN 500000 AND 1500000
ORDER BY price;

-- 3. Products with 'Pro' anywhere in the name
-- '%Pro%' is a leading-wildcard pattern — cannot use B-tree index efficiently
SELECT * FROM products WHERE name LIKE '%Pro%';

-- 4. Products with NULL or zero stock
-- IS NULL is required because = NULL always evaluates to UNKNOWN (not TRUE)
SELECT * FROM products
WHERE stock IS NULL OR stock = 0;


-- === Exercise 3: Sorting and Pagination ===
-- Problem: Sort and paginate product listings.

-- Solution:

-- 1. Top 5 most expensive products
SELECT * FROM products ORDER BY price DESC LIMIT 5;

-- 2. Sort by category alphabetically, then price ascending within each category
SELECT * FROM products ORDER BY category, price;

-- 3. Page 2 (products 6-10): OFFSET skips the first 5 rows
SELECT * FROM products ORDER BY id LIMIT 5 OFFSET 5;

-- 4. Most expensive product per category
-- DISTINCT ON keeps only the first row for each distinct category value;
-- the ORDER BY must start with the DISTINCT ON column(s)
SELECT DISTINCT ON (category) *
FROM products
ORDER BY category, price DESC;


-- === Exercise 4: NULL Handling ===
-- Problem: Handle NULL stock values gracefully in queries.

-- Solution:

-- 1. Show products with no stock, treating NULL as 0
-- COALESCE returns the first non-NULL argument
SELECT name, COALESCE(stock, 0) AS stock
FROM products
WHERE stock IS NULL OR stock = 0;

-- 2. Display a human-readable status instead of raw NULL
-- Must cast stock to TEXT because COALESCE requires matching types
SELECT
    name,
    COALESCE(stock::TEXT, 'Checking stock') AS stock_status
FROM products;

-- 3. Sort products by stock, but push NULLs to the end
-- By default PostgreSQL sorts NULLs as if they are larger than any value (NULLS LAST
-- is the default for ASC, but being explicit improves readability)
SELECT * FROM products ORDER BY stock NULLS LAST;
