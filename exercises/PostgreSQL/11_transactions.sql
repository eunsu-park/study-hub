-- Exercises for Lesson 11: Transactions
-- Topic: PostgreSQL
-- Solutions to practice problems from the lesson.

-- Setup: Create accounts table for transaction exercises
CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    balance NUMERIC(12, 2) NOT NULL DEFAULT 0 CHECK (balance >= 0)
);

TRUNCATE TABLE accounts RESTART IDENTITY;
INSERT INTO accounts (name, balance) VALUES
('Alice', 1000000),
('Bob', 500000),
('Charlie', 200000);


-- === Exercise 1: Basic Transaction (Account Transfer) ===
-- Problem: Implement a safe money transfer procedure with balance checking.

-- Solution:

-- This procedure wraps withdrawal + deposit in a single transaction.
-- If the sender's balance goes negative, an exception is raised and
-- the entire operation is rolled back — neither account is modified.
CREATE OR REPLACE PROCEDURE transfer(
    from_id INTEGER,
    to_id INTEGER,
    amount NUMERIC
)
AS $$
BEGIN
    -- Withdrawal: deduct from sender
    UPDATE accounts SET balance = balance - amount WHERE id = from_id;

    -- Check balance: if negative, abort the entire transfer
    IF (SELECT balance FROM accounts WHERE id = from_id) < 0 THEN
        RAISE EXCEPTION 'Insufficient balance for account %', from_id;
    END IF;

    -- Deposit: add to receiver
    UPDATE accounts SET balance = balance + amount WHERE id = to_id;

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Test: Transfer 100,000 from Alice to Bob
CALL transfer(1, 2, 100000);

-- Verify balances
SELECT * FROM accounts ORDER BY id;


-- === Exercise 2: Using SAVEPOINT ===
-- Problem: Use savepoints to partially rollback within a transaction.

-- Solution:

-- Scenario: Create an order, then attempt to reduce stock.
-- If stock goes negative, rollback only the stock change
-- while keeping the order.
BEGIN;

-- Step 1: Insert the order
INSERT INTO orders (user_id, amount) VALUES (1, 50000);
SAVEPOINT order_created;

-- Step 2: Attempt stock reduction
UPDATE products SET stock = stock - 1 WHERE id = 10;

-- Step 3: Check if stock went negative
-- If yes, ROLLBACK TO SAVEPOINT undoes only the UPDATE,
-- preserving the INSERT above
DO $$
BEGIN
    IF (SELECT stock FROM products WHERE id = 10) < 0 THEN
        ROLLBACK TO SAVEPOINT order_created;
        RAISE NOTICE 'Stock insufficient — order kept, stock change reverted';
    END IF;
END;
$$;

COMMIT;


-- === Exercise 3: Testing Isolation Levels ===
-- Problem: Demonstrate REPEATABLE READ isolation with two concurrent sessions.

-- Solution:

-- This exercise requires two separate psql terminals.
-- The comments show the interleaved execution order.

-- Terminal 1: Start a REPEATABLE READ transaction
-- A snapshot is taken at the first query — all subsequent reads
-- see the database as it was at that moment.
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT * FROM accounts;  -- sees original balances

-- Terminal 2: Modify data and commit
-- UPDATE accounts SET balance = balance + 50000 WHERE id = 1;
-- COMMIT;

-- Terminal 1: Read again — still sees the OLD value (snapshot consistency)
SELECT * FROM accounts;  -- Alice's balance unchanged from Terminal 1's perspective
COMMIT;

-- Terminal 1: Now outside the transaction, reads the committed change
SELECT * FROM accounts;  -- Alice's balance now reflects Terminal 2's update


-- === Exercise 4: FOR UPDATE Lock ===
-- Problem: Safely check and reduce stock under concurrent access.

-- Solution:

-- FOR UPDATE acquires a row-level lock: other transactions that try to
-- SELECT ... FOR UPDATE on the same row will block until this transaction
-- commits or rolls back. This prevents the "double-sell" race condition.
BEGIN;

-- Lock the product row — other sessions wait here
SELECT stock FROM products WHERE id = 1 FOR UPDATE;

-- Safely reduce stock only if positive
-- The WHERE stock > 0 is a safety net, but the lock above
-- guarantees no concurrent modification between SELECT and UPDATE
UPDATE products
SET stock = stock - 1
WHERE id = 1 AND stock > 0;

COMMIT;

-- Bonus: A more robust version using a procedure
CREATE OR REPLACE PROCEDURE safe_reduce_stock(
    p_product_id INTEGER,
    p_quantity INTEGER
)
AS $$
DECLARE
    current_stock INTEGER;
BEGIN
    -- Lock and read in one step
    SELECT stock INTO current_stock
    FROM products
    WHERE id = p_product_id
    FOR UPDATE;

    IF current_stock IS NULL THEN
        RAISE EXCEPTION 'Product % not found', p_product_id;
    END IF;

    IF current_stock < p_quantity THEN
        RAISE EXCEPTION 'Insufficient stock: available %, requested %',
            current_stock, p_quantity;
    END IF;

    UPDATE products
    SET stock = stock - p_quantity
    WHERE id = p_product_id;

    COMMIT;
END;
$$ LANGUAGE plpgsql;
