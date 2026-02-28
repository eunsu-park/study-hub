-- Exercises for Lesson 12: Triggers
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.


-- === Exercise 1: Auto Timestamp ===
-- Problem: Automatically update the updated_at column on every UPDATE.

-- Solution:

-- Generic trigger function: sets NEW.updated_at to the current time.
-- "Generic" because it works on ANY table that has an updated_at column.
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create the articles table
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Attach the trigger: fires BEFORE UPDATE so we can modify NEW
CREATE TRIGGER set_updated_at
BEFORE UPDATE ON articles
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- Test: insert, then update — observe updated_at changes automatically
INSERT INTO articles (title, content) VALUES ('First Post', 'Hello world');
SELECT id, title, created_at, updated_at FROM articles;

-- Wait a moment, then update
UPDATE articles SET content = 'Hello PostgreSQL triggers!' WHERE id = 1;
SELECT id, title, created_at, updated_at FROM articles;
-- created_at remains the same; updated_at is now later


-- === Exercise 2: Audit Log ===
-- Problem: Track all INSERT, UPDATE, DELETE operations on the users table.

-- Solution:

-- Audit log stores a JSONB snapshot of old and new row data,
-- plus metadata (who changed it, when, and what operation)
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    operation VARCHAR(10),
    old_data JSONB,
    new_data JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT NOW()
);

-- A single trigger function handles all three DML operations.
-- TG_OP tells us which operation fired the trigger.
CREATE OR REPLACE FUNCTION audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_data, changed_by)
        VALUES (TG_TABLE_NAME, 'INSERT', row_to_json(NEW)::JSONB, current_user);
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_data, new_data, changed_by)
        VALUES (TG_TABLE_NAME, 'UPDATE',
                row_to_json(OLD)::JSONB, row_to_json(NEW)::JSONB, current_user);
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_data, changed_by)
        VALUES (TG_TABLE_NAME, 'DELETE', row_to_json(OLD)::JSONB, current_user);
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Attach to users table — AFTER trigger so we log the final state
CREATE TRIGGER users_audit
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION audit_trigger();

-- Test: perform all three operations and check the log
INSERT INTO users (name, email) VALUES ('Audit Test', 'audit@test.com');
UPDATE users SET name = 'Audit Modified' WHERE email = 'audit@test.com';
DELETE FROM users WHERE email = 'audit@test.com';

-- View the audit trail
SELECT operation, old_data, new_data, changed_by, changed_at
FROM audit_log
ORDER BY changed_at DESC;


-- === Exercise 3: Inventory Management ===
-- Problem: Auto-reserve stock when order items are created,
-- and deduct stock when orders are completed.

-- Solution:

-- Inventory table tracks both physical quantity and reserved amount
CREATE TABLE inventory (
    product_id INTEGER PRIMARY KEY,
    quantity INTEGER DEFAULT 0,
    reserved INTEGER DEFAULT 0
);

-- Trigger 1: Reserve stock when an order_item is inserted
-- Checks available = quantity - reserved before allowing the reservation
CREATE OR REPLACE FUNCTION reserve_stock()
RETURNS TRIGGER AS $$
DECLARE
    available INTEGER;
BEGIN
    SELECT quantity - reserved INTO available
    FROM inventory
    WHERE product_id = NEW.product_id;

    IF available IS NULL THEN
        RAISE EXCEPTION 'Product % not in inventory', NEW.product_id;
    END IF;

    IF available < NEW.quantity THEN
        RAISE EXCEPTION 'Insufficient stock: available %, requested %',
            available, NEW.quantity;
    END IF;

    -- Reserve the stock (still physically present, but earmarked)
    UPDATE inventory
    SET reserved = reserved + NEW.quantity
    WHERE product_id = NEW.product_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER before_order_item
BEFORE INSERT ON order_items
FOR EACH ROW
EXECUTE FUNCTION reserve_stock();

-- Trigger 2: Deduct actual stock when order status changes to 'completed'
-- Only fires when status transitions TO 'completed' (not re-saves)
CREATE OR REPLACE FUNCTION complete_stock()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'completed' AND OLD.status <> 'completed' THEN
        UPDATE inventory
        SET quantity = quantity - oi.quantity,
            reserved = reserved - oi.quantity
        FROM order_items oi
        WHERE oi.order_id = NEW.id
          AND inventory.product_id = oi.product_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_order_complete
AFTER UPDATE ON orders
FOR EACH ROW
EXECUTE FUNCTION complete_stock();


-- === Exercise 4: Data Validation ===
-- Problem: Enforce case-insensitive email uniqueness via trigger.

-- Solution:

-- PostgreSQL's UNIQUE constraint is case-sensitive by default.
-- This trigger adds case-insensitive uniqueness checking.
CREATE OR REPLACE FUNCTION check_email_unique()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM users
        WHERE LOWER(email) = LOWER(NEW.email)
          AND id <> COALESCE(NEW.id, -1)
    ) THEN
        RAISE EXCEPTION 'Email already exists (case-insensitive): %', NEW.email;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- BEFORE INSERT OR UPDATE OF email: only fires when email column changes
CREATE TRIGGER before_user_email
BEFORE INSERT OR UPDATE OF email ON users
FOR EACH ROW
EXECUTE FUNCTION check_email_unique();

-- Test: These should conflict
-- INSERT INTO users (name, email) VALUES ('Test1', 'Test@Email.COM');
-- INSERT INTO users (name, email) VALUES ('Test2', 'test@email.com');
-- ERROR: Email already exists (case-insensitive): test@email.com
