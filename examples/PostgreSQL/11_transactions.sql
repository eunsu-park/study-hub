-- ============================================================================
-- PostgreSQL Transactions and Isolation Levels
-- ============================================================================
-- Demonstrates:
--   - Transaction basics (BEGIN, COMMIT, ROLLBACK)
--   - Savepoints for partial rollback
--   - Isolation levels (Read Committed, Repeatable Read, Serializable)
--   - Anomalies: dirty read, non-repeatable read, phantom read, serialization
--   - Advisory locks
--
-- Prerequisites: PostgreSQL 12+
-- Usage: psql -U postgres -d your_database -f 11_transactions.sql
-- ============================================================================

-- Clean up
DROP TABLE IF EXISTS accounts CASCADE;
DROP TABLE IF EXISTS transfer_log CASCADE;

-- ============================================================================
-- Setup
-- ============================================================================

CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    owner TEXT NOT NULL,
    balance NUMERIC(12, 2) NOT NULL CHECK (balance >= 0)
);

CREATE TABLE transfer_log (
    log_id SERIAL PRIMARY KEY,
    from_account INTEGER REFERENCES accounts(account_id),
    to_account INTEGER REFERENCES accounts(account_id),
    amount NUMERIC(12, 2) NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO accounts (owner, balance) VALUES
    ('Alice', 10000.00),
    ('Bob', 5000.00),
    ('Charlie', 3000.00),
    ('Diana', 8000.00);

-- ============================================================================
-- 1. Basic Transaction: Fund Transfer
-- ============================================================================

-- Successful transfer
BEGIN;
    UPDATE accounts SET balance = balance - 1000 WHERE account_id = 1;  -- Alice -1000
    UPDATE accounts SET balance = balance + 1000 WHERE account_id = 2;  -- Bob +1000
    INSERT INTO transfer_log (from_account, to_account, amount, status)
    VALUES (1, 2, 1000.00, 'completed');
COMMIT;

SELECT owner, balance FROM accounts ORDER BY account_id;

-- ============================================================================
-- 2. ROLLBACK — Undo on Error
-- ============================================================================

-- This transfer should fail (insufficient funds for Charlie)
BEGIN;
    UPDATE accounts SET balance = balance - 5000 WHERE account_id = 3;  -- Charlie has 3000
    -- Check constraint would fail, but let's show explicit rollback
    -- In practice, the CHECK constraint prevents negative balance
ROLLBACK;

-- Charlie's balance unchanged
SELECT owner, balance FROM accounts WHERE account_id = 3;

-- ============================================================================
-- 3. SAVEPOINT — Partial Rollback
-- ============================================================================

BEGIN;
    -- Why: SAVEPOINT creates a named rollback point within a transaction. This
    -- lets us undo just the bonus credit without losing the debit that already
    -- succeeded. Without savepoints, any error would force rolling back the
    -- entire transaction including the valid debit.
    UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
    SAVEPOINT after_debit;

    -- Step 2: Optional bonus credit (might fail)
    UPDATE accounts SET balance = balance + 600 WHERE account_id = 2;  -- 500 + bonus
    SAVEPOINT after_credit;

    -- Step 3: Oops, bonus was wrong — roll back to after credit
    ROLLBACK TO after_debit;

    -- Step 3 (retry): Normal credit without bonus
    UPDATE accounts SET balance = balance + 500 WHERE account_id = 2;

COMMIT;

SELECT owner, balance FROM accounts WHERE account_id IN (1, 2);

-- ============================================================================
-- 4. Isolation Levels Overview
-- ============================================================================
-- PostgreSQL supports 3 isolation levels (no Read Uncommitted):
--
-- Level              | Dirty Read | Non-Repeatable Read | Phantom Read
-- -------------------|------------|---------------------|-------------
-- Read Committed     | No         | Possible            | Possible
-- Repeatable Read    | No         | No                  | No*
-- Serializable       | No         | No                  | No
--
-- *PostgreSQL's Repeatable Read also prevents phantoms (uses MVCC snapshots)

-- ============================================================================
-- 5. Read Committed (Default)
-- ============================================================================
-- Each statement sees the latest committed data.
-- Different statements in the same transaction may see different snapshots.

-- Session 1:
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
    SELECT balance FROM accounts WHERE account_id = 1;
    -- Returns current balance (e.g., 8500)

    -- Meanwhile, another session commits:
    -- UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;

    -- Same query, same transaction — may see different value!
    SELECT balance FROM accounts WHERE account_id = 1;
    -- Could return 8400 (sees the committed update)
COMMIT;

-- ============================================================================
-- 6. Repeatable Read
-- ============================================================================
-- Transaction sees a snapshot from its start.
-- All queries within the transaction see the same data.

BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
    SELECT balance FROM accounts WHERE account_id = 1;
    -- Returns snapshot value (e.g., 8500)

    -- Even if another session commits changes, this transaction
    -- still sees the original snapshot value.
    SELECT balance FROM accounts WHERE account_id = 1;
    -- Still returns 8500

    -- But if we try to UPDATE a row that was modified by another
    -- committed transaction, PostgreSQL raises a serialization error:
    -- ERROR: could not serialize access due to concurrent update
COMMIT;

-- ============================================================================
-- 7. Serializable
-- ============================================================================
-- Strictest level — transactions behave as if executed sequentially.
-- PostgreSQL uses Serializable Snapshot Isolation (SSI).

-- Classic example: two transactions reading each other's writes
-- Session 1:
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
    SELECT SUM(balance) FROM accounts;
    -- Later: UPDATE accounts SET balance = balance + 100 WHERE account_id = 1;
COMMIT;
-- If another serializable transaction does the same,
-- one will be aborted with serialization_failure.

-- ============================================================================
-- 8. Practical Pattern: Safe Transfer with Retry
-- ============================================================================

-- This function handles serialization failures by design
CREATE OR REPLACE FUNCTION safe_transfer(
    p_from INTEGER,
    p_to INTEGER,
    p_amount NUMERIC
) RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    v_from_balance NUMERIC;
BEGIN
    -- Check source balance
    -- Why: FOR UPDATE acquires a row-level exclusive lock, preventing another
    -- transaction from reading or modifying this balance until we commit.
    -- Without it, two concurrent transfers from the same account could both
    -- see sufficient balance and overdraw the account (lost update anomaly).
    SELECT balance INTO v_from_balance
    FROM accounts
    WHERE account_id = p_from
    FOR UPDATE;  -- Row-level lock

    IF v_from_balance < p_amount THEN
        RETURN 'INSUFFICIENT_FUNDS';
    END IF;

    -- Perform transfer
    UPDATE accounts SET balance = balance - p_amount
    WHERE account_id = p_from;

    UPDATE accounts SET balance = balance + p_amount
    WHERE account_id = p_to;

    INSERT INTO transfer_log (from_account, to_account, amount, status)
    VALUES (p_from, p_to, p_amount, 'completed');

    RETURN 'SUCCESS';
END;
$$;

-- Usage:
SELECT safe_transfer(1, 3, 500);
SELECT owner, balance FROM accounts ORDER BY account_id;

-- ============================================================================
-- 9. Advisory Locks
-- ============================================================================
-- Application-level locks for custom synchronization.
-- Don't lock any table rows — purely advisory.

-- Why: Advisory locks are application-defined locks that don't correspond to any
-- database object. They are useful for coordinating between application instances
-- (e.g., ensuring only one worker processes a job queue item). The integer key (42)
-- is application-defined and must be managed by convention.
SELECT pg_advisory_lock(42);        -- Acquire lock on key 42
-- Do critical work...
SELECT pg_advisory_unlock(42);      -- Release

-- Transaction-level advisory lock (released at COMMIT/ROLLBACK)
BEGIN;
    SELECT pg_advisory_xact_lock(100);  -- Lock held until end of transaction
    -- Do work...
COMMIT;  -- Lock automatically released

-- Why: pg_try_advisory_lock is non-blocking — it returns immediately with false
-- if the lock is held by another session, instead of waiting indefinitely.
-- This is essential for "skip if busy" patterns where blocking would cause
-- throughput degradation or deadlocks.
SELECT pg_try_advisory_lock(42) AS acquired;

-- ============================================================================
-- 10. Transaction Information Functions
-- ============================================================================

-- Current transaction ID
SELECT txid_current();

-- Transaction snapshot
SELECT txid_current_snapshot();

-- Check current isolation level
SHOW transaction_isolation;

-- List active locks
SELECT
    l.locktype,
    l.mode,
    l.granted,
    a.application_name,
    a.query
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE a.pid != pg_backend_pid()
ORDER BY l.locktype;
