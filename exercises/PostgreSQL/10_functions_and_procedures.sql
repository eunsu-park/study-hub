-- Exercises for Lesson 10: Functions and Procedures
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: Utility Functions ===
-- Problem: Create helper functions for email domain extraction,
-- age calculation, and price formatting.

-- Solution:

-- 1. Extract email domain
-- SPLIT_PART splits by delimiter and returns the Nth part
CREATE OR REPLACE FUNCTION get_email_domain(email VARCHAR)
RETURNS VARCHAR
AS $$
BEGIN
    RETURN SPLIT_PART(email, '@', 2);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Test
SELECT get_email_domain('user@gmail.com');       -- gmail.com
SELECT get_email_domain('admin@company.co.kr');   -- company.co.kr

-- 2. Calculate age from birth date
-- AGE() returns an interval; EXTRACT(YEAR FROM ...) pulls out the year component
CREATE OR REPLACE FUNCTION calculate_age(birth_date DATE)
RETURNS INTEGER
AS $$
BEGIN
    RETURN EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date));
END;
$$ LANGUAGE plpgsql STABLE;

-- Test
SELECT calculate_age('1990-05-15');  -- depends on current date

-- 3. Format price with thousands separator and currency suffix
-- TO_CHAR with FM prefix suppresses leading spaces
CREATE OR REPLACE FUNCTION format_price(price NUMERIC)
RETURNS VARCHAR
AS $$
BEGIN
    RETURN TO_CHAR(price, 'FM999,999,999') || ' KRW';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Test
SELECT format_price(1500000);   -- 1,500,000 KRW
SELECT format_price(299.99);    -- 300 KRW (rounds to integer format)


-- === Exercise 2: Business Logic Functions ===
-- Problem: Create functions for order total calculation and user tier assignment.

-- Solution:

-- 1. Calculate order total by joining order_items with products
-- Returns 0 instead of NULL for orders with no items (defensive programming)
CREATE OR REPLACE FUNCTION calculate_order_total(p_order_id INTEGER)
RETURNS NUMERIC
AS $$
DECLARE
    total NUMERIC;
BEGIN
    SELECT SUM(p.price * oi.quantity)
    INTO total
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE oi.order_id = p_order_id;

    RETURN COALESCE(total, 0);
END;
$$ LANGUAGE plpgsql STABLE;

-- 2. Determine user tier based on lifetime spending
-- Business rule: VIP >= 1M, Gold >= 500K, Silver >= 100K, else Bronze
CREATE OR REPLACE FUNCTION get_user_tier(p_user_id INTEGER)
RETURNS VARCHAR
AS $$
DECLARE
    total_spent NUMERIC;
BEGIN
    SELECT COALESCE(SUM(amount), 0)
    INTO total_spent
    FROM orders
    WHERE user_id = p_user_id;

    RETURN CASE
        WHEN total_spent >= 1000000 THEN 'VIP'
        WHEN total_spent >= 500000 THEN 'Gold'
        WHEN total_spent >= 100000 THEN 'Silver'
        ELSE 'Bronze'
    END;
END;
$$ LANGUAGE plpgsql STABLE;


-- === Exercise 3: Data Validation Functions ===
-- Problem: Create email validation and phone number formatting functions.

-- Solution:

-- 1. Email validation using regex
-- The ~* operator performs case-insensitive regex matching
CREATE OR REPLACE FUNCTION is_valid_email(email VARCHAR)
RETURNS BOOLEAN
AS $$
BEGIN
    RETURN email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Test
SELECT is_valid_email('test@email.com');    -- true
SELECT is_valid_email('invalid-email');      -- false
SELECT is_valid_email('user@.com');          -- false

-- 2. Format phone number (strip non-digits, then insert dashes)
-- Handles Korean-style 11-digit phone numbers (010-XXXX-XXXX)
CREATE OR REPLACE FUNCTION format_phone(phone VARCHAR)
RETURNS VARCHAR
AS $$
DECLARE
    cleaned VARCHAR;
BEGIN
    -- Remove all non-digit characters (spaces, dashes, dots, parens)
    cleaned := REGEXP_REPLACE(phone, '[^0-9]', '', 'g');

    IF LENGTH(cleaned) = 11 THEN
        RETURN SUBSTRING(cleaned, 1, 3) || '-' ||
               SUBSTRING(cleaned, 4, 4) || '-' ||
               SUBSTRING(cleaned, 8, 4);
    ELSIF LENGTH(cleaned) = 10 THEN
        -- Handle 10-digit numbers (02-XXXX-XXXX style)
        RETURN SUBSTRING(cleaned, 1, 2) || '-' ||
               SUBSTRING(cleaned, 3, 4) || '-' ||
               SUBSTRING(cleaned, 7, 4);
    ELSE
        -- Return as-is if length doesn't match known patterns
        RETURN phone;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Test
SELECT format_phone('01012345678');    -- 010-1234-5678
SELECT format_phone('010-1234-5678');  -- 010-1234-5678 (already formatted, re-formats cleanly)
SELECT format_phone('010 1234 5678');  -- 010-1234-5678
