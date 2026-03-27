# Transactions

**Previous**: [Functions and Procedures](./10_Functions_and_Procedures.md) | **Next**: [Triggers](./12_Triggers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the concept of a transaction and why atomicity matters for data integrity
2. Describe the four ACID properties and their role in reliable database operations
3. Use BEGIN, COMMIT, and ROLLBACK to control transaction boundaries
4. Create and manage SAVEPOINTs for partial rollback within a transaction
5. Compare the four SQL isolation levels and identify which concurrency anomalies each prevents
6. Diagnose concurrency problems such as dirty reads, non-repeatable reads, and phantom reads
7. Apply row-level and table-level locks to protect critical sections
8. Detect and prevent deadlocks using consistent lock ordering

---

Transactions are the backbone of every reliable database application. Whether you are transferring money between bank accounts, placing an order in an e-commerce system, or recording sensor data from IoT devices, transactions guarantee that a group of operations either fully succeeds or fully fails -- leaving the database in a consistent state. Mastering transactions is essential for building applications that behave correctly under concurrent access and unexpected failures.

## 1. Transaction Concept

A transaction is a collection of operations that constitute a single logical unit of work.

```
┌──────────────────────────────────────────────────────────┐
│                   Account Transfer Transaction           │
├──────────────────────────────────────────────────────────┤
│  1. Deduct 100,000 from Account A                        │
│  2. Add 100,000 to Account B                             │
│  → Both must succeed or both must fail                   │
└──────────────────────────────────────────────────────────┘
```

---

## 2. ACID Properties

| Property | English | Description |
|------|------|------|
| Atomicity | Atomicity | All or nothing |
| Consistency | Consistency | Data consistency maintained before and after transaction |
| Isolation | Isolation | Concurrent transactions don't interfere |
| Durability | Durability | Committed transactions are permanently stored |

---

## 3. Basic Transaction Commands

### BEGIN / COMMIT / ROLLBACK

```sql
-- Start transaction
BEGIN;
-- Or
START TRANSACTION;

-- Perform operations
UPDATE accounts SET balance = balance - 100000 WHERE id = 1;
UPDATE accounts SET balance = balance + 100000 WHERE id = 2;

-- Commit (confirm changes)
COMMIT;

-- Or rollback (cancel changes)
ROLLBACK;
```

### Practice Example

```sql
-- Create table
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    balance NUMERIC(12, 2) DEFAULT 0
);

INSERT INTO accounts (name, balance) VALUES
('Kim', 1000000),
('Lee', 500000);

-- Transfer transaction
BEGIN;

UPDATE accounts SET balance = balance - 100000 WHERE name = 'Kim';
UPDATE accounts SET balance = balance + 100000 WHERE name = 'Lee';

-- Check
SELECT * FROM accounts;

-- Commit or cancel
COMMIT;  -- Or ROLLBACK;
```

---

## 4. Autocommit

psql is in autocommit mode by default.

```sql
-- In autocommit mode, each statement is an individual transaction
INSERT INTO accounts (name, balance) VALUES ('Park', 300000);
-- Immediately committed

-- Disable autocommit
\set AUTOCOMMIT off

-- Now explicit COMMIT required
INSERT INTO accounts (name, balance) VALUES ('Choi', 400000);
COMMIT;

-- Re-enable autocommit
\set AUTOCOMMIT on
```

---

## 5. SAVEPOINT

Create partial rollback points within a transaction.

```sql
BEGIN;

INSERT INTO accounts (name, balance) VALUES ('New1', 100000);
SAVEPOINT sp1;

INSERT INTO accounts (name, balance) VALUES ('New2', 200000);
SAVEPOINT sp2;

INSERT INTO accounts (name, balance) VALUES ('New3', 300000);

-- Rollback to sp2 (cancel only New3)
ROLLBACK TO SAVEPOINT sp2;

-- Rollback to sp1 (cancel New2, New3)
ROLLBACK TO SAVEPOINT sp1;

-- Commit all (save only New1)
COMMIT;
```

### SAVEPOINT Management

```sql
-- Release SAVEPOINT
RELEASE SAVEPOINT sp1;

-- Overwrite SAVEPOINT (recreate with same name)
SAVEPOINT mypoint;
-- ... work ...
SAVEPOINT mypoint;  -- Replace with new point
```

---

## 6. Transaction Isolation Levels

Determines the degree of isolation between concurrently executing transactions.

### Isolation Level Types

| Level | Dirty Read | Non-repeatable Read | Phantom Read |
|------|------------|---------------------|--------------|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible* |
| SERIALIZABLE | Prevented | Prevented | Prevented |

*PostgreSQL's REPEATABLE READ also prevents Phantom Reads

### PostgreSQL Default

PostgreSQL's default isolation level is **READ COMMITTED**.

### Setting Isolation Level

```sql
-- Per-transaction setting
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- Or
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- Session-wide setting
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Check current isolation level
SHOW transaction_isolation;
```

---

## 7. Concurrency Problems

### Dirty Read

Reading uncommitted data → Does not occur in PostgreSQL

### Non-repeatable Read

Reading the same data twice in the same transaction returns different values

```sql
-- Transaction A
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- 1000000

-- Transaction B updates and commits
-- UPDATE accounts SET balance = 900000 WHERE id = 1; COMMIT;

SELECT balance FROM accounts WHERE id = 1;  -- 900000 (different value!)
COMMIT;
```

### Phantom Read

Same query returns different number of rows

```sql
-- Transaction A
BEGIN;
SELECT COUNT(*) FROM accounts WHERE balance > 500000;  -- 2 rows

-- Transaction B inserts new row and commits
-- INSERT INTO accounts VALUES (...); COMMIT;

SELECT COUNT(*) FROM accounts WHERE balance > 500000;  -- 3 rows (phantom row!)
COMMIT;
```

---

## 8. Isolation Level Behavior

### READ COMMITTED (default)

```sql
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- Can see changes committed by other transactions immediately
SELECT * FROM accounts;  -- Latest committed data

COMMIT;
```

### REPEATABLE READ

```sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- Sees snapshot from transaction start time
SELECT * FROM accounts;

-- Same result even if other transactions commit
SELECT * FROM accounts;  -- Same

COMMIT;
```

### SERIALIZABLE

```sql
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Most strict isolation
-- May fail with serialization error
SELECT * FROM accounts WHERE balance > 500000;
UPDATE accounts SET balance = balance + 10000 WHERE id = 1;

COMMIT;
-- ERROR: could not serialize access due to concurrent update
-- (if conflicts with other transactions)
```

---

## 9. Locking

### Row-Level Locks

```sql
-- SELECT FOR UPDATE: Lock while querying
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- Other transactions cannot modify/delete this row

UPDATE accounts SET balance = balance - 100000 WHERE id = 1;
COMMIT;

-- SELECT FOR SHARE: Shared lock (allow reads, prevent writes)
SELECT * FROM accounts WHERE id = 1 FOR SHARE;
```

### Lock Options

```sql
-- Don't wait, fail immediately
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;

-- Wait for specified time
SELECT * FROM accounts WHERE id = 1 FOR UPDATE SKIP LOCKED;
```

### Table-Level Locks

```sql
-- Explicit table lock (rarely used)
LOCK TABLE accounts IN EXCLUSIVE MODE;
```

---

## 10. Deadlock

Two transactions waiting for each other's locks

```sql
-- Transaction A
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- Locks id=1

-- Transaction B
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 2;
-- Locks id=2

-- Transaction A
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
-- Waiting for id=2...

-- Transaction B
UPDATE accounts SET balance = balance + 100 WHERE id = 1;
-- Waiting for id=1... → Deadlock!

-- PostgreSQL automatically aborts one transaction
-- ERROR: deadlock detected
```

### Preventing Deadlocks

```sql
-- Always acquire locks in the same order
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- Always smaller id first
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

---

## 10.5 Deadlock Lab and SERIALIZABLE Isolation

The previous sections introduced deadlocks and isolation levels. This section goes deeper: you will learn how to deliberately create, detect, and debug deadlocks, and understand how PostgreSQL's SERIALIZABLE isolation level uses Serializable Snapshot Isolation (SSI) to prevent anomalies that REPEATABLE READ cannot catch.

### Deadlock Lab: Reproducing a Deadlock Step by Step

Open **two** psql terminals connected to the same database. We will deliberately cause a deadlock.

```sql
-- Setup: create a test table
CREATE TABLE deadlock_lab (
    id INTEGER PRIMARY KEY,
    value TEXT,
    counter INTEGER DEFAULT 0
);

INSERT INTO deadlock_lab VALUES (1, 'Alice', 100), (2, 'Bob', 200);
```

**Terminal 1:**

```sql
-- Step 1: Begin a transaction and lock row id=1
-- Why: this acquires a row-level exclusive lock on id=1
BEGIN;
UPDATE deadlock_lab SET counter = counter + 10 WHERE id = 1;
-- Row id=1 is now locked by Terminal 1
-- DO NOT COMMIT YET — wait for Terminal 2 to lock id=2
```

**Terminal 2:**

```sql
-- Step 2: Begin a transaction and lock row id=2
-- Why: this acquires a row-level exclusive lock on id=2
BEGIN;
UPDATE deadlock_lab SET counter = counter + 20 WHERE id = 2;
-- Row id=2 is now locked by Terminal 2
```

**Terminal 1:**

```sql
-- Step 3: Try to lock row id=2 (held by Terminal 2)
-- Why: Terminal 1 now waits for Terminal 2 to release id=2
UPDATE deadlock_lab SET counter = counter + 10 WHERE id = 2;
-- This will BLOCK — waiting for Terminal 2's lock on id=2
```

**Terminal 2:**

```sql
-- Step 4: Try to lock row id=1 (held by Terminal 1)
-- Why: Terminal 2 now waits for Terminal 1 to release id=1
-- This creates a cycle: T1 waits for T2, T2 waits for T1 → DEADLOCK
UPDATE deadlock_lab SET counter = counter + 20 WHERE id = 1;

-- Within ~1 second, PostgreSQL detects the deadlock:
-- ERROR:  deadlock detected
-- DETAIL: Process 12345 waits for ShareLock on transaction 67890;
--         blocked by process 12346.
--         Process 12346 waits for ShareLock on transaction 67891;
--         blocked by process 12345.
-- HINT:  See server log for query details.

-- PostgreSQL aborts THIS transaction (the victim)
ROLLBACK;  -- Clean up
```

**Terminal 1:**

```sql
-- Terminal 1's UPDATE on id=2 now succeeds (Terminal 2 released its locks)
COMMIT;
```

### PostgreSQL Deadlock Detection Internals

```
┌──────────────────────────────────────────────────────────────────────┐
│              PostgreSQL Deadlock Detection                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PostgreSQL uses a wait-for graph to detect deadlocks:               │
│                                                                      │
│      T1 ──waits for──► T2                                            │
│      ▲                  │                                            │
│      └──waits for───────┘    ← CYCLE = DEADLOCK                     │
│                                                                      │
│  Detection trigger:                                                  │
│  - A process waits for a lock longer than deadlock_timeout (1s)      │
│  - PostgreSQL then builds the wait-for graph                         │
│  - If a cycle is found, one transaction is aborted (the victim)      │
│                                                                      │
│  Key settings:                                                       │
│  - deadlock_timeout = 1s  (default; time before checking for DL)     │
│  - lock_timeout = 0       (0 = wait forever; set to limit waiting)   │
│  - log_lock_waits = off   (set to 'on' to log waits > DL timeout)   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Configuring Deadlock Detection

```sql
-- Check current settings
SHOW deadlock_timeout;     -- Default: 1s
SHOW lock_timeout;         -- Default: 0 (no limit)
SHOW log_lock_waits;       -- Default: off

-- Set lock_timeout to fail fast instead of waiting forever
-- Why: in web applications, waiting 30+ seconds for a lock is worse
-- than failing immediately and retrying
SET lock_timeout = '5s';

-- Set log_lock_waits to monitor lock contention
-- Why: identifies which queries are frequently blocked by locks
ALTER SYSTEM SET log_lock_waits = on;
SELECT pg_reload_conf();

-- Per-session timeout (useful for testing)
BEGIN;
SET LOCAL lock_timeout = '2s';
-- If a lock cannot be acquired within 2 seconds, this fails with:
-- ERROR: canceling statement due to lock timeout
SELECT * FROM deadlock_lab WHERE id = 1 FOR UPDATE;
COMMIT;
```

### Preventing Deadlocks: Consistent Lock Ordering

```sql
-- BAD: inconsistent lock ordering causes deadlocks
-- Transaction A: UPDATE ... WHERE id = 1; UPDATE ... WHERE id = 2;
-- Transaction B: UPDATE ... WHERE id = 2; UPDATE ... WHERE id = 1;

-- GOOD: always lock in the same order (e.g., ascending id)
-- Why: if ALL transactions lock rows in the same order,
-- no cycle can ever form in the wait-for graph

-- Transfer function with consistent lock ordering
CREATE OR REPLACE FUNCTION safe_transfer(
    from_id INTEGER,
    to_id INTEGER,
    amount INTEGER
) RETURNS VOID AS $$
DECLARE
    -- Why: determine lock order before acquiring any locks
    first_id  INTEGER := LEAST(from_id, to_id);
    second_id INTEGER := GREATEST(from_id, to_id);
BEGIN
    -- Always lock the lower id first, regardless of transfer direction
    -- Why: this guarantees all transactions acquire locks in the same order
    PERFORM 1 FROM deadlock_lab WHERE id = first_id FOR UPDATE;
    PERFORM 1 FROM deadlock_lab WHERE id = second_id FOR UPDATE;

    -- Now safely perform the transfer
    UPDATE deadlock_lab SET counter = counter - amount WHERE id = from_id;
    UPDATE deadlock_lab SET counter = counter + amount WHERE id = to_id;
END;
$$ LANGUAGE plpgsql;

-- Both of these calls lock id=1 first, then id=2 — no deadlock possible
SELECT safe_transfer(1, 2, 50);  -- Locks: 1 → 2
SELECT safe_transfer(2, 1, 30);  -- Locks: 1 → 2 (not 2 → 1!)
```

### SERIALIZABLE Isolation: Serializable Snapshot Isolation (SSI)

PostgreSQL implements the SERIALIZABLE isolation level using **Serializable Snapshot Isolation (SSI)**, a technique that detects serialization conflicts without using traditional two-phase locking (2PL).

```
┌──────────────────────────────────────────────────────────────────────┐
│           SSI vs Traditional 2PL                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Traditional 2PL (used by SQL Server, MySQL/InnoDB):                 │
│  - Acquires read locks (shared) and write locks (exclusive)          │
│  - Readers block writers, writers block readers                      │
│  - Guarantees serializability but reduces concurrency                │
│  - High lock contention on read-heavy workloads                      │
│                                                                      │
│  PostgreSQL SSI:                                                     │
│  - Uses MVCC snapshots (readers NEVER block writers)                 │
│  - Tracks read-write dependencies between transactions               │
│  - Detects "dangerous structures" (potential serialization anomalies)│
│  - Aborts one transaction if a conflict is detected                  │
│  - Much better for read-heavy workloads                              │
│                                                                      │
│  Trade-off: SSI may produce false positives (abort transactions      │
│  that would actually have been safe), but never misses a real        │
│  anomaly.                                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Write Skew Anomaly: Why REPEATABLE READ Is Not Enough

Write skew is a concurrency anomaly that REPEATABLE READ cannot prevent, but SERIALIZABLE can. Classic example: a hospital must always have at least one doctor on call.

```sql
-- Setup
CREATE TABLE doctors_on_call (
    id INTEGER PRIMARY KEY,
    name TEXT,
    on_call BOOLEAN DEFAULT true
);

INSERT INTO doctors_on_call VALUES (1, 'Dr. Kim', true), (2, 'Dr. Lee', true);
```

**With REPEATABLE READ (write skew occurs):**

```sql
-- Terminal 1 (Dr. Kim wants to go off-call):
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- Check: is there at least one other doctor on call?
-- Why: at this snapshot, both Dr. Kim and Dr. Lee are on-call
SELECT COUNT(*) FROM doctors_on_call WHERE on_call = true AND id != 1;
-- Returns: 1 (Dr. Lee is on-call) — safe to go off-call

-- Terminal 2 (Dr. Lee wants to go off-call — simultaneously):
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- Check: is there at least one other doctor on call?
-- Why: at this snapshot, both are still on-call
SELECT COUNT(*) FROM doctors_on_call WHERE on_call = true AND id != 2;
-- Returns: 1 (Dr. Kim is on-call) — safe to go off-call

-- Terminal 1:
UPDATE doctors_on_call SET on_call = false WHERE id = 1;
COMMIT;  -- Succeeds!

-- Terminal 2:
UPDATE doctors_on_call SET on_call = false WHERE id = 2;
COMMIT;  -- Also succeeds!

-- PROBLEM: Both doctors went off-call — nobody is covering!
-- Why: Each transaction saw a snapshot where the OTHER doctor was on-call.
-- REPEATABLE READ cannot detect this cross-transaction dependency.
```

**With SERIALIZABLE (write skew prevented):**

```sql
-- Terminal 1:
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM doctors_on_call WHERE on_call = true AND id != 1;
-- Returns: 1

-- Terminal 2:
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM doctors_on_call WHERE on_call = true AND id != 2;
-- Returns: 1

-- Terminal 1:
UPDATE doctors_on_call SET on_call = false WHERE id = 1;
COMMIT;  -- Succeeds

-- Terminal 2:
UPDATE doctors_on_call SET on_call = false WHERE id = 2;
COMMIT;
-- ERROR: could not serialize access due to read/write dependencies
--        among transactions
-- DETAIL: Reason code: Canceled on identification as a pivot, ...

-- Why: PostgreSQL's SSI detected that T2's read (on_call status)
-- was invalidated by T1's write. The system correctly prevented
-- the write skew anomaly.
```

### When to Use SERIALIZABLE

```sql
-- SERIALIZABLE adds overhead. Use it when:
-- 1. Correctness > throughput (financial systems, inventory)
-- 2. Business rules span multiple rows (like the on-call example)
-- 3. You need true serializability guarantees without manual locking

-- For best results with SERIALIZABLE:
-- - Keep transactions short
-- - Be prepared to retry on serialization failures
-- - Use a retry loop in application code:

-- Application pseudo-code:
-- max_retries = 3
-- for attempt in range(max_retries):
--     try:
--         BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE
--         ... your queries ...
--         COMMIT
--         break  # success
--     except SerializationFailure:
--         ROLLBACK
--         # retry with backoff
--         continue
```

### Monitoring Lock Contention and Deadlocks

```sql
-- View all current locks and what they're waiting for
-- Why: essential for debugging lock contention in production
SELECT
    l.pid,
    l.locktype,
    l.mode,
    l.granted,
    a.query,
    a.state,
    age(now(), a.query_start) AS query_age
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted
ORDER BY a.query_start;

-- View deadlock statistics (PostgreSQL 14+)
-- Why: helps understand how frequently deadlocks occur
SELECT
    datname,
    deadlocks,
    conflicts
FROM pg_stat_database
WHERE datname = current_database();

-- Check for long-running transactions (potential lock holders)
-- Why: a forgotten open transaction can hold locks indefinitely
SELECT
    pid,
    now() - xact_start AS transaction_duration,
    state,
    query
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
  AND state != 'idle'
ORDER BY xact_start;
```

---

## 11. Practice Examples

### Practice 1: Basic Transaction

```sql
-- Account transfer
CREATE OR REPLACE PROCEDURE transfer(
    from_id INTEGER,
    to_id INTEGER,
    amount NUMERIC
)
AS $$
BEGIN
    -- Withdrawal
    UPDATE accounts SET balance = balance - amount WHERE id = from_id;

    -- Check balance
    IF (SELECT balance FROM accounts WHERE id = from_id) < 0 THEN
        RAISE EXCEPTION 'Insufficient balance';
    END IF;

    -- Deposit
    UPDATE accounts SET balance = balance + amount WHERE id = to_id;

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Usage
CALL transfer(1, 2, 100000);
```

### Practice 2: Using SAVEPOINT

```sql
BEGIN;

-- Insert base data
INSERT INTO orders (user_id, amount) VALUES (1, 50000);
SAVEPOINT order_created;

-- Attempt to reduce stock
UPDATE products SET stock = stock - 1 WHERE id = 10;

-- Check stock
IF (SELECT stock FROM products WHERE id = 10) < 0 THEN
    ROLLBACK TO SAVEPOINT order_created;
    -- Keep order but cancel stock reduction
END IF;

COMMIT;
```

### Practice 3: Testing Isolation Levels

Test with two terminals.

```sql
-- Terminal 1
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT * FROM accounts;

-- Terminal 2
UPDATE accounts SET balance = balance + 50000 WHERE id = 1;
COMMIT;

-- Terminal 1
SELECT * FROM accounts;  -- Old value (snapshot)
COMMIT;

SELECT * FROM accounts;  -- Now shows changed value
```

### Practice 4: FOR UPDATE Lock

```sql
-- Check and reduce stock (concurrency-safe)
BEGIN;

-- Query with lock
SELECT stock FROM products WHERE id = 1 FOR UPDATE;

-- Check and reduce stock
UPDATE products
SET stock = stock - 1
WHERE id = 1 AND stock > 0;

COMMIT;
```

---

## 12. Transaction Monitoring

```sql
-- Check currently running transactions
SELECT
    pid,
    now() - xact_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE xact_start IS NOT NULL;

-- Check queries waiting for locks
SELECT
    blocked.pid AS blocked_pid,
    blocking.pid AS blocking_pid,
    blocked.query AS blocked_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking ON blocking.pid = ANY(pg_blocking_pids(blocked.pid));
```

---

---

**Previous**: [Functions and Procedures](./10_Functions_and_Procedures.md) | **Next**: [Triggers](./12_Triggers.md)
