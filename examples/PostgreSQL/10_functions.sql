-- ============================================================================
-- PostgreSQL Functions and Stored Procedures
-- ============================================================================
-- Demonstrates:
--   - SQL functions (LANGUAGE sql)
--   - PL/pgSQL functions with control flow
--   - Stored procedures (CALL syntax)
--   - Function overloading
--   - RETURNS TABLE and SETOF
--   - IMMUTABLE / STABLE / VOLATILE classifications
--
-- Prerequisites: PostgreSQL 11+ (procedures), 14+ (recommended)
-- Usage: psql -U postgres -d your_database -f 10_functions.sql
-- ============================================================================

-- Clean up
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS audit_log CASCADE;
DROP FUNCTION IF EXISTS product_total_value() CASCADE;
DROP FUNCTION IF EXISTS get_product_by_category(TEXT) CASCADE;
DROP FUNCTION IF EXISTS format_price(NUMERIC) CASCADE;
DROP FUNCTION IF EXISTS format_price(NUMERIC, TEXT) CASCADE;
DROP FUNCTION IF EXISTS apply_discount(NUMERIC, NUMERIC) CASCADE;
DROP FUNCTION IF EXISTS search_products(TEXT, NUMERIC, NUMERIC) CASCADE;
DROP FUNCTION IF EXISTS get_price_stats() CASCADE;
DROP PROCEDURE IF EXISTS restock_product(INTEGER, INTEGER) CASCADE;
DROP PROCEDURE IF EXISTS bulk_price_update(TEXT, NUMERIC) CASCADE;

-- ============================================================================
-- Setup: Sample tables
-- ============================================================================

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL CHECK (price >= 0),
    stock INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_log (
    log_id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    details TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO products (name, category, price, stock) VALUES
    ('Laptop Pro', 'Electronics', 1299.99, 50),
    ('Wireless Mouse', 'Electronics', 29.99, 200),
    ('Mechanical Keyboard', 'Electronics', 89.99, 150),
    ('Python Cookbook', 'Books', 49.99, 100),
    ('Database Internals', 'Books', 59.99, 75),
    ('Standing Desk', 'Furniture', 499.99, 30),
    ('Monitor 27"', 'Electronics', 349.99, 80),
    ('Ergonomic Chair', 'Furniture', 699.99, 25);

-- ============================================================================
-- 1. Simple SQL Function (LANGUAGE sql)
-- ============================================================================

-- Why: LANGUAGE sql is preferred over plpgsql when the function body is a single
-- SQL expression, because the optimizer can inline it directly into the calling
-- query. STABLE tells the planner this function reads but never writes data,
-- allowing it to be called once per query snapshot rather than once per row.
CREATE FUNCTION product_total_value()
RETURNS NUMERIC
LANGUAGE sql
STABLE  -- Does not modify DB, depends on DB state
AS $$
    SELECT COALESCE(SUM(price * stock), 0)
    FROM products;
$$;

-- Usage:
SELECT product_total_value() AS total_inventory_value;

-- ============================================================================
-- 2. PL/pgSQL Function with Control Flow
-- ============================================================================

CREATE FUNCTION apply_discount(
    p_product_id INTEGER,
    p_discount_pct NUMERIC
)
RETURNS NUMERIC
LANGUAGE plpgsql
VOLATILE  -- Modifies the database
AS $$
DECLARE
    v_old_price NUMERIC;
    v_new_price NUMERIC;
BEGIN
    -- Get current price
    SELECT price INTO v_old_price
    FROM products
    WHERE product_id = p_product_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Product % not found', p_product_id;
    END IF;

    -- Validate discount
    IF p_discount_pct < 0 OR p_discount_pct > 100 THEN
        RAISE EXCEPTION 'Discount must be between 0 and 100, got %',
                        p_discount_pct;
    END IF;

    -- Calculate and apply
    v_new_price := v_old_price * (1 - p_discount_pct / 100);
    v_new_price := ROUND(v_new_price, 2);

    UPDATE products
    SET price = v_new_price
    WHERE product_id = p_product_id;

    -- Log the change
    INSERT INTO audit_log (table_name, operation, details)
    VALUES ('products', 'DISCOUNT',
            format('Product %s: $%s → $%s (%s%% off)',
                   p_product_id, v_old_price, v_new_price, p_discount_pct));

    RETURN v_new_price;
END;
$$;

-- Usage:
SELECT apply_discount(2, 10);  -- 10% off Wireless Mouse
SELECT * FROM audit_log;

-- ============================================================================
-- 3. Function Overloading
-- ============================================================================

-- Version 1: format_price(amount)
CREATE FUNCTION format_price(p_amount NUMERIC)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE  -- Pure function, no DB access
AS $$
    SELECT '$' || TO_CHAR(p_amount, 'FM999,999,990.00');
$$;

-- Version 2: format_price(amount, currency)
CREATE FUNCTION format_price(p_amount NUMERIC, p_currency TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT CASE p_currency
        WHEN 'USD' THEN '$' || TO_CHAR(p_amount, 'FM999,999,990.00')
        WHEN 'EUR' THEN TO_CHAR(p_amount, 'FM999,999,990.00') || '€'
        WHEN 'GBP' THEN '£' || TO_CHAR(p_amount, 'FM999,999,990.00')
        WHEN 'JPY' THEN '¥' || TO_CHAR(p_amount, 'FM999,999,990')
        ELSE TO_CHAR(p_amount, 'FM999,999,990.00') || ' ' || p_currency
    END;
$$;

-- Usage — PostgreSQL picks the right overload:
SELECT format_price(1299.99);
SELECT format_price(1299.99, 'EUR');
SELECT format_price(1299.99, 'JPY');

-- ============================================================================
-- 4. RETURNS TABLE — set-returning function
-- ============================================================================

-- Why: RETURNS TABLE lets the function act as a virtual table in FROM clauses.
-- The "NULL means no filter" pattern (p_keyword IS NULL OR ...) makes all
-- parameters optional, avoiding the need for multiple overloaded functions
-- for different search combinations.
CREATE FUNCTION search_products(
    p_keyword TEXT DEFAULT NULL,
    p_min_price NUMERIC DEFAULT NULL,
    p_max_price NUMERIC DEFAULT NULL
)
RETURNS TABLE (
    product_id INTEGER,
    name TEXT,
    category TEXT,
    price NUMERIC,
    formatted_price TEXT
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.product_id,
        p.name,
        p.category,
        p.price,
        format_price(p.price) AS formatted_price
    FROM products p
    WHERE (p_keyword IS NULL OR p.name ILIKE '%' || p_keyword || '%')
      AND (p_min_price IS NULL OR p.price >= p_min_price)
      AND (p_max_price IS NULL OR p.price <= p_max_price)
    ORDER BY p.price;
END;
$$;

-- Usage:
SELECT * FROM search_products('key');           -- by keyword
SELECT * FROM search_products(NULL, 50, 500);   -- by price range
SELECT * FROM search_products('o', 0, 100);     -- combined

-- ============================================================================
-- 5. RETURNS SETOF — returning record sets
-- ============================================================================

CREATE FUNCTION get_product_by_category(p_category TEXT)
RETURNS SETOF products
LANGUAGE sql
STABLE
AS $$
    SELECT * FROM products
    WHERE category = p_category
    ORDER BY price DESC;
$$;

-- Usage:
SELECT * FROM get_product_by_category('Electronics');

-- ============================================================================
-- 6. Aggregate Statistics Function
-- ============================================================================

CREATE FUNCTION get_price_stats()
RETURNS TABLE (
    category TEXT,
    product_count BIGINT,
    avg_price NUMERIC,
    min_price NUMERIC,
    max_price NUMERIC,
    total_value NUMERIC
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        p.category,
        COUNT(*),
        ROUND(AVG(p.price), 2),
        MIN(p.price),
        MAX(p.price),
        SUM(p.price * p.stock)
    FROM products p
    GROUP BY p.category
    ORDER BY total_value DESC;
$$;

SELECT * FROM get_price_stats();

-- ============================================================================
-- 7. Stored Procedures (PostgreSQL 11+)
-- ============================================================================
-- Why: Procedures (not functions) are needed when you want COMMIT/ROLLBACK
-- inside the routine body. Functions always run within the caller's transaction,
-- so they cannot commit intermediate work. Use procedures for multi-step
-- operations where partial commit is desirable (e.g., batch processing).
CREATE PROCEDURE restock_product(
    p_product_id INTEGER,
    p_quantity INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_name TEXT;
    v_old_stock INTEGER;
BEGIN
    SELECT name, stock INTO v_name, v_old_stock
    FROM products
    WHERE product_id = p_product_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Product % not found', p_product_id;
    END IF;

    UPDATE products
    SET stock = stock + p_quantity
    WHERE product_id = p_product_id;

    INSERT INTO audit_log (table_name, operation, details)
    VALUES ('products', 'RESTOCK',
            format('%s: %s → %s (+%s)',
                   v_name, v_old_stock, v_old_stock + p_quantity, p_quantity));

    RAISE NOTICE 'Restocked % (% → %)',
                 v_name, v_old_stock, v_old_stock + p_quantity;
END;
$$;

-- Usage:
CALL restock_product(1, 25);  -- Add 25 units to Laptop Pro
SELECT name, stock FROM products WHERE product_id = 1;
SELECT * FROM audit_log ORDER BY log_id DESC LIMIT 1;

-- ============================================================================
-- 8. Procedure with Bulk Operations
-- ============================================================================

CREATE PROCEDURE bulk_price_update(
    p_category TEXT,
    p_adjustment_pct NUMERIC
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    UPDATE products
    SET price = ROUND(price * (1 + p_adjustment_pct / 100), 2)
    WHERE category = p_category;

    -- Why: GET DIAGNOSTICS captures the number of rows affected by the preceding
    -- statement. This is more reliable than running a separate COUNT(*) query,
    -- which could race with concurrent modifications.
    GET DIAGNOSTICS v_count = ROW_COUNT;

    INSERT INTO audit_log (table_name, operation, details)
    VALUES ('products', 'BULK_UPDATE',
            format('Category %s: %s products, %s%% adjustment',
                   p_category, v_count, p_adjustment_pct));

    RAISE NOTICE 'Updated % products in category %', v_count, p_category;
END;
$$;

-- Usage: 5% price increase for Books
CALL bulk_price_update('Books', 5);
SELECT name, price FROM products WHERE category = 'Books';

-- ============================================================================
-- 9. Volatility Classifications
-- ============================================================================
-- IMMUTABLE: result depends only on arguments, never changes
--   Example: format_price(), mathematical functions
--   Optimizer can cache results, use in indexes
--
-- STABLE: result depends on DB state but doesn't modify it
--   Example: get_product_by_category(), lookups
--   Safe within a single query (consistent snapshot)
--
-- VOLATILE (default): may modify DB or return different results each call
--   Example: apply_discount(), random(), now()
--   Never cached, always re-evaluated

-- Check function volatility:
SELECT
    p.proname AS function_name,
    CASE p.provolatile
        WHEN 'i' THEN 'IMMUTABLE'
        WHEN 's' THEN 'STABLE'
        WHEN 'v' THEN 'VOLATILE'
    END AS volatility,
    pg_get_function_arguments(p.oid) AS arguments
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'public'
ORDER BY p.proname;
