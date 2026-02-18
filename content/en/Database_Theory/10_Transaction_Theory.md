# 10. Transaction Theory

**Previous**: [Indexing](./09_Indexing.md) | **Next**: [Concurrency Control](./11_Concurrency_Control.md)

---

## Learning Objectives

- Understand the transaction concept and ACID properties in depth
- Trace through transaction state transitions
- Analyze schedules for conflict serializability using precedence graphs
- Distinguish recoverable, cascadeless, and strict schedules
- Understand SQL isolation levels and the anomalies they permit
- Reason about snapshot isolation and its trade-offs

---

## 1. Transaction Concept

### What is a Transaction?

A **transaction** is a logical unit of work that accesses and possibly modifies the contents of a database. It comprises a sequence of operations (reads and writes) that must be treated as an **indivisible** unit.

**Real-world analogy**: A bank transfer from account A to account B:

```
Transaction T: Transfer $100 from A to B

    read(A)           // A = 1000
    A = A - 100       // A = 900
    write(A)
    read(B)           // B = 500
    B = B + 100       // B = 600
    write(B)
    commit
```

If the system crashes between `write(A)` and `write(B)`, the money would vanish -- $100 deducted from A but never credited to B. Transactions prevent this.

### Why Transactions Matter

Transactions provide two critical guarantees:

1. **Failure atomicity**: If a failure occurs during execution, all partial effects are undone
2. **Isolation**: Concurrent transactions do not interfere with each other

Without transactions, a database cannot ensure **data consistency** in the presence of failures and concurrent access.

---

## 2. ACID Properties

The ACID properties are the four fundamental guarantees that a transaction processing system must provide.

### 2.1 Atomicity (All or Nothing)

A transaction is an **atomic** unit: either all of its operations are reflected in the database, or none are.

```
Transaction T: Transfer $100
┌─────────────────────┐
│ read(A)             │
│ A = A - 100         │
│ write(A)            │
│ ────── crash ────── │ ← If crash here, undo write(A)
│ read(B)             │
│ B = B + 100         │
│ write(B)            │
│ commit              │
└─────────────────────┘

Either BOTH writes persist, or NEITHER does.
```

**Implementation**: The recovery subsystem uses **logs** to undo incomplete transactions and redo committed ones (covered in Lesson 12).

### 2.2 Consistency

A transaction must transform the database from one **consistent state** to another. Consistency is defined by:

- **Integrity constraints**: Primary keys, foreign keys, CHECK constraints, NOT NULL
- **Application-level invariants**: e.g., total money in the bank must be constant

```
Consistent State:
  Account A = 1000, Account B = 500
  Invariant: A + B = 1500

During Transaction:
  After write(A): A = 900, B = 500  → A + B = 1400 ← INCONSISTENT
  After write(B): A = 900, B = 600  → A + B = 1500 ← CONSISTENT again

The intermediate inconsistency is acceptable because it is not visible
to other transactions (ensured by Isolation).
```

**Responsibility**: Consistency is a shared responsibility:
- The **DBMS** enforces declared integrity constraints
- The **application** (programmer) must ensure transaction logic preserves application-level invariants

### 2.3 Isolation

Each transaction must execute as if it were the **only** transaction in the system. Concurrent transactions must not see each other's intermediate states.

```
Without Isolation:

T1: read(A)=1000, A=900, write(A)
                                    T2: read(A)=900 ← sees partial result!
T1: read(B)=500, B=600, write(B)
T1: commit
                                    T2: read(B)=600
                                    T2: A + B = 900 + 600 = 1500 ← correct by luck

But if T2 had read B before T1 wrote it:
                                    T2: read(B)=500
                                    T2: A + B = 900 + 500 = 1400 ← WRONG!
```

**Implementation**: The concurrency control subsystem uses locks, timestamps, or MVCC (covered in Lesson 11).

### 2.4 Durability

Once a transaction has **committed**, its effects must persist even if the system crashes immediately afterward.

```
T: write(A), write(B), commit
                                    ← System crash here

After recovery: A = 900, B = 600   ← Committed changes survive
```

**Implementation**: The recovery subsystem uses **Write-Ahead Logging (WAL)** and **checkpoints** to ensure committed data survives crashes (covered in Lesson 12).

### ACID Summary

| Property | Guarantee | Implemented By |
|---|---|---|
| Atomicity | All or nothing | Recovery system (undo) |
| Consistency | Valid state to valid state | DBMS constraints + app logic |
| Isolation | No interference between concurrent txns | Concurrency control |
| Durability | Committed data survives crashes | Recovery system (redo) + WAL |

---

## 3. Transaction States

A transaction goes through a well-defined set of states during its lifecycle:

```
                    ┌──────────┐
     begin          │          │
    ─────────────→  │  Active  │
                    │          │
                    └────┬─────┘
                         │
                    last statement
                    executed
                         │
                         ▼
                ┌────────────────┐
                │   Partially    │
                │   Committed    │
                └───────┬────────┘
                   ┌────┴────┐
                   │         │
              output has    failure
              been written  detected
              to disk
                   │         │
                   ▼         ▼
            ┌───────────┐  ┌────────┐
            │ Committed │  │ Failed │
            └───────────┘  └───┬────┘
                               │
                          rollback
                          complete
                               │
                               ▼
                          ┌─────────┐
                          │ Aborted │
                          └─────────┘
```

### State Descriptions

| State | Description |
|---|---|
| **Active** | The initial state. The transaction stays in this state while executing read/write operations. |
| **Partially Committed** | After the last statement has been executed. The transaction's effects may still be in volatile memory (buffer). |
| **Committed** | After all changes have been successfully written to stable storage. The transaction is complete and its effects are permanent. |
| **Failed** | After discovery that normal execution can no longer proceed (e.g., constraint violation, deadlock, system error). |
| **Aborted** | After the transaction has been rolled back and the database is restored to the state before the transaction began. |

### After Abort: Restart or Kill?

When a transaction aborts, the system has two options:

1. **Restart**: Re-execute the transaction (appropriate for transient failures like deadlocks)
2. **Kill**: Terminate the transaction entirely (appropriate for logical errors, constraint violations)

```sql
-- Transaction that might be restarted after deadlock:
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
-- If deadlock detected → abort → restart automatically

-- Transaction that should be killed:
BEGIN;
INSERT INTO users(email) VALUES ('duplicate@email.com');
-- If unique constraint violation → abort → do NOT restart
```

---

## 4. Schedules

### 4.1 What is a Schedule?

A **schedule** (or **history**) represents the chronological order in which instructions of concurrent transactions are executed. A schedule must preserve the **internal ordering** of each transaction.

**Notation**: We focus on read and write operations:
- `r_i(X)` = Transaction `T_i` reads data item `X`
- `w_i(X)` = Transaction `T_i` writes data item `X`
- `c_i` = Transaction `T_i` commits
- `a_i` = Transaction `T_i` aborts

### 4.2 Serial Schedule

A **serial schedule** executes transactions one after another, with no interleaving:

```
Serial Schedule S1 (T1 then T2):
┌─────────────────────────────────────┐
│ T1: r₁(A) w₁(A) r₁(B) w₁(B) c₁   │
│                                     │
│         T2: r₂(A) w₂(A) r₂(B) w₂(B) c₂ │
└─────────────────────────────────────┘

Serial Schedule S2 (T2 then T1):
┌─────────────────────────────────────┐
│ T2: r₂(A) w₂(A) r₂(B) w₂(B) c₂   │
│                                     │
│         T1: r₁(A) w₁(A) r₁(B) w₁(B) c₁ │
└─────────────────────────────────────┘
```

**Properties:**
- Always correct (each transaction sees a consistent database)
- No concurrency -- poor performance
- For `n` transactions, there are `n!` possible serial schedules

### 4.3 Serializable Schedule

A **serializable schedule** is a concurrent schedule that produces the same result as **some** serial schedule. It is the gold standard of correctness.

```
Serializable Schedule S3 (interleaved, but equivalent to S1):
┌──────────────────────────────────────────────┐
│ T1: r₁(A) w₁(A)                             │
│                    T2: r₂(A) w₂(A)           │
│ T1:                           r₁(B) w₁(B) c₁│
│                    T2:                r₂(B) w₂(B) c₂│
└──────────────────────────────────────────────┘

This may or may not be serializable — we need formal tests to determine this.
```

---

## 5. Conflict Serializability

### 5.1 Conflicting Operations

Two operations **conflict** if:
1. They belong to **different** transactions
2. They access the **same** data item
3. At least one of them is a **write**

```
Conflict types:

r₁(A) ... r₂(A)    → No conflict (both reads)
r₁(A) ... w₂(A)    → Read-Write conflict (RW)
w₁(A) ... r₂(A)    → Write-Read conflict (WR)
w₁(A) ... w₂(A)    → Write-Write conflict (WW)
```

**Non-conflicting** operations can be **swapped** in a schedule without changing the result.

### 5.2 Conflict Equivalence

Two schedules are **conflict equivalent** if one can be transformed into the other by a series of swaps of **adjacent non-conflicting** operations.

### 5.3 Conflict Serializable

A schedule is **conflict serializable** if it is conflict equivalent to some serial schedule.

**Example: Is this schedule conflict serializable?**

```
Schedule S: r₁(A) r₂(A) w₁(A) w₂(A) r₁(B) r₂(B) w₁(B) w₂(B)

Step-by-step analysis:
1. r₁(A) and r₂(A): no conflict → can swap
2. r₂(A) and w₁(A): RW conflict on A → T₂ reads before T₁ writes (T₂ → T₁? No, T₁ must come after T₂ for A)
3. w₁(A) and w₂(A): WW conflict on A → T₁ writes before T₂ writes → T₁ before T₂
4. r₁(B) and r₂(B): no conflict
5. r₂(B) and w₁(B): RW conflict on B → T₂ reads before T₁ writes → T₂ before T₁
6. w₁(B) and w₂(B): WW conflict on B → T₁ before T₂

For A: from conflict (2), r₂(A) before w₁(A) implies T₂ → T₁
       from conflict (3), w₁(A) before w₂(A) implies T₁ → T₂
For B: from conflict (5), r₂(B) before w₁(B) implies T₂ → T₁
       from conflict (6), w₁(B) before w₂(B) implies T₁ → T₂

We have both T₁ → T₂ AND T₂ → T₁. This is a cycle, so the schedule
is NOT conflict serializable.
```

### 5.4 Precedence Graph (Serialization Graph)

The **precedence graph** provides an efficient algorithm for testing conflict serializability.

**Construction:**
1. Create a node for each transaction `T_i`
2. Add a directed edge `T_i → T_j` if there exists a pair of conflicting operations where `T_i`'s operation comes first

**Theorem**: A schedule is conflict serializable **if and only if** its precedence graph is **acyclic** (has no cycles).

**Example 1: Serializable schedule**

```
Schedule: r₁(A) w₁(A) r₂(A) w₂(A) r₁(B) w₁(B) r₂(B) w₂(B)

Conflicts:
- w₁(A) before r₂(A): T₁ → T₂  (WR on A)
- w₁(A) before w₂(A): T₁ → T₂  (WW on A)
- w₁(B) before r₂(B): T₁ → T₂  (WR on B)
- w₁(B) before w₂(B): T₁ → T₂  (WW on B)

Precedence Graph:
    T₁ ──→ T₂

No cycle → Conflict serializable (equivalent to serial order: T₁, T₂)
```

**Example 2: Non-serializable schedule**

```
Schedule: r₁(A) r₂(B) w₂(A) w₁(B)

Conflicts:
- r₁(A) before w₂(A): T₁ → T₂  (RW on A)
- r₂(B) before w₁(B): T₂ → T₁  (RW on B)

Precedence Graph:
    T₁ ──→ T₂
     ↑       │
     └───────┘

Cycle detected: T₁ → T₂ → T₁
→ NOT conflict serializable
```

**Example 3: Three transactions**

```
Schedule: r₁(A) r₂(B) r₃(C) w₁(B) w₂(C) w₃(A)

Conflicts:
- r₂(B) before w₁(B): T₂ → T₁  (RW on B)
- r₃(C) before w₂(C): T₃ → T₂  (RW on C)
- r₁(A) before w₃(A): T₁ → T₃  (RW on A)

Precedence Graph:
    T₁ ──→ T₃
     ↑       │
     │       ↓
     └── T₂ ←┘

Cycle: T₁ → T₃ → T₂ → T₁
→ NOT conflict serializable
```

### 5.5 Topological Sort for Serial Order

If the precedence graph is acyclic, a **topological sort** gives a valid serial order:

```
Precedence Graph:
    T₁ ──→ T₃
    T₂ ──→ T₃
    T₂ ──→ T₁

Topological sort: T₂, T₁, T₃
This is the equivalent serial schedule.
```

**Algorithm:**
```
TOPOLOGICAL-SORT(graph):
    result = []
    while graph has nodes:
        find a node with no incoming edges
        add it to result
        remove it and its outgoing edges from graph
    if all nodes removed:
        return result  // valid serial order
    else:
        return CYCLE   // not serializable
```

---

## 6. View Serializability

### 6.1 Definition

Two schedules `S` and `S'` are **view equivalent** if:

1. **Initial reads**: If `T_i` reads the initial value of `X` in `S`, then `T_i` reads the initial value of `X` in `S'`
2. **Updated reads**: If `T_i` reads the value of `X` written by `T_j` in `S`, then `T_i` reads the value of `X` written by `T_j` in `S'`
3. **Final writes**: If `T_i` performs the final write of `X` in `S`, then `T_i` performs the final write of `X` in `S'`

A schedule is **view serializable** if it is view equivalent to some serial schedule.

### 6.2 Conflict vs. View Serializability

```
View Serializable ⊇ Conflict Serializable

┌─────────────────────────────────────────┐
│        All Schedules                    │
│  ┌───────────────────────────────────┐  │
│  │    View Serializable              │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │  Conflict Serializable      │  │  │
│  │  │  ┌───────────────────────┐  │  │  │
│  │  │  │   Serial Schedules    │  │  │  │
│  │  │  └───────────────────────┘  │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Key facts:**
- Every conflict serializable schedule is view serializable
- The converse is NOT true (some view serializable schedules are not conflict serializable)
- Testing view serializability is **NP-complete**
- Testing conflict serializability is **polynomial** (cycle detection in precedence graph)
- In practice, DBMS use conflict serializability (or weaker guarantees) due to computational feasibility

### 6.3 Blind Writes

The schedules that are view serializable but NOT conflict serializable involve **blind writes** -- writes that are not preceded by a read of the same item.

```
Schedule: w₁(A) w₂(A) w₂(B) w₁(B)

Precedence graph:
- w₁(A) before w₂(A): T₁ → T₂  (WW on A)
- w₂(B) before w₁(B): T₂ → T₁  (WW on B)
Cycle! → NOT conflict serializable

But this IS view serializable:
- View equivalent to serial order T₁, T₂:
  - No reads to worry about (initial reads or updated reads)
  - Final write of A: T₂ in both schedules ✓
  - Final write of B: T₁ in both schedules ✓
```

---

## 7. Recoverability

Serializability alone is not sufficient for correctness. We also need schedules that allow proper **recovery** from transaction failures.

### 7.1 Recoverable Schedules

A schedule is **recoverable** if, for every pair of transactions `T_i` and `T_j`, if `T_j` reads a value written by `T_i`, then `T_i` commits **before** `T_j` commits.

```
Recoverable:
    T₁: w₁(A) ................... c₁
    T₂:        r₂(A) ................. c₂
    (T₁ commits before T₂ commits ✓)

NOT Recoverable:
    T₁: w₁(A) ........................ a₁ (abort!)
    T₂:        r₂(A) ..... c₂
    (T₂ committed based on T₁'s data, but T₁ aborts later!)
    Problem: T₂ used invalid data but already committed — cannot undo T₂
```

**Why it matters**: If a schedule is not recoverable, a cascading abort might need to undo an already-committed transaction, violating durability.

### 7.2 Cascadeless Schedules (Avoiding Cascading Rollbacks)

A **cascading rollback** occurs when the abort of one transaction forces the abort of other transactions that read its uncommitted data.

```
Cascading Rollback:
    T₁: w₁(A) ................... a₁ (abort!)
    T₂:        r₂(A) w₂(B) ........  ← must abort (read T₁'s data)
    T₃:                    r₃(B) ...  ← must abort (read T₂'s data)

T₁'s abort cascades to T₂, then to T₃.
```

A schedule is **cascadeless** (or **avoids cascading rollbacks**, ACR) if every transaction reads only values written by **committed** transactions.

```
Cascadeless:
    T₁: w₁(A) ............ c₁
    T₂:                         r₂(A) w₂(B)    ← reads only after T₁ committed
```

**Relationship**: Every cascadeless schedule is recoverable, but not vice versa.

### 7.3 Strict Schedules

A schedule is **strict** if no transaction reads or overwrites a value written by an uncommitted transaction.

```
Strict:
    T₁: w₁(A) ............ c₁
    T₂:                         r₂(A) OR w₂(A)  ← only after T₁ committed

Cascadeless but NOT Strict:
    T₁: w₁(A) ............ c₁
    T₂:        w₂(A)                             ← overwrites T₁'s uncommitted data
    (No read of uncommitted data, so cascadeless)
    (But T₂ overwrites before T₁ commits, so NOT strict)
```

**Why strictness matters**: Strict schedules simplify recovery because the "before image" of a write can be used for undo without worrying about other transactions' writes in between.

### 7.4 Hierarchy of Schedule Properties

```
Strict ⊂ Cascadeless ⊂ Recoverable ⊂ All Schedules

┌──────────────────────────────────────────┐
│           All Schedules                  │
│  ┌────────────────────────────────────┐  │
│  │       Recoverable                  │  │
│  │  ┌──────────────────────────────┐  │  │
│  │  │     Cascadeless (ACR)        │  │  │
│  │  │  ┌────────────────────────┐  │  │  │
│  │  │  │      Strict            │  │  │  │
│  │  │  │  ┌──────────────────┐  │  │  │  │
│  │  │  │  │    Serial        │  │  │  │  │
│  │  │  │  └──────────────────┘  │  │  │  │
│  │  │  └────────────────────────┘  │  │  │
│  │  └──────────────────────────────┘  │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

| Property | Condition | Recovery Benefit |
|---|---|---|
| Recoverable | T_j reads from T_i → T_i commits before T_j | Can always undo uncommitted txns |
| Cascadeless | Only read committed data | No cascading aborts |
| Strict | No read/overwrite of uncommitted data | Simple undo (restore before-image) |

---

## 8. Isolation Levels

### 8.1 Motivation

Full serializability provides maximum correctness but at a significant performance cost (more locking, less concurrency). Many applications can tolerate weaker isolation for better throughput.

The **SQL standard** defines four isolation levels, each permitting certain anomalies:

### 8.2 Anomalies (Phenomena)

**Dirty Read**: Reading data written by an uncommitted transaction.

```
T₁: w₁(A=100) ............ a₁ (abort, A reverts to 50)
T₂:            r₂(A)=100       ← T₂ read the "dirty" value 100
```

**Non-Repeatable Read (Fuzzy Read)**: Reading the same item twice yields different values.

```
T₁:                   r₁(A)=50 .............. r₁(A)=100  ← different!
T₂: w₂(A=100) c₂
```

**Phantom Read**: A query returns different sets of rows when executed twice.

```
T₁: SELECT * FROM emp WHERE dept='Eng'  →  {Alice, Bob}
T₂: INSERT INTO emp VALUES('Carol','Eng')  c₂
T₁: SELECT * FROM emp WHERE dept='Eng'  →  {Alice, Bob, Carol}  ← phantom!
```

### 8.3 SQL Isolation Levels

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|---|---|---|---|
| Read Uncommitted | Possible | Possible | Possible |
| Read Committed | Not possible | Possible | Possible |
| Repeatable Read | Not possible | Not possible | Possible |
| Serializable | Not possible | Not possible | Not possible |

```sql
-- Setting isolation level in SQL:
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- or
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

### 8.4 Read Uncommitted

The weakest level. Transactions can read uncommitted ("dirty") data from other transactions.

```
T₁: BEGIN (READ UNCOMMITTED)
    SELECT balance FROM accounts WHERE id = 1;  → 1000
    -- Meanwhile T₂ is updating but hasn't committed:
    -- T₂: UPDATE accounts SET balance = 500 WHERE id = 1;
    SELECT balance FROM accounts WHERE id = 1;  → 500  ← dirty read!
    -- If T₂ rolls back, 500 was never a real value
```

**Use case**: Approximate aggregates, monitoring dashboards where exact values are not critical.

**Implementation**: No read locks acquired. Writers still acquire write locks.

### 8.5 Read Committed

Each read sees only data committed **at the time of the read**. No dirty reads, but the same query may return different results if another transaction commits between reads.

```
T₁: BEGIN (READ COMMITTED)
    SELECT balance FROM accounts WHERE id = 1;  → 1000
    -- T₂: UPDATE accounts SET balance = 500 WHERE id = 1; COMMIT;
    SELECT balance FROM accounts WHERE id = 1;  → 500  ← non-repeatable read
    -- Both reads saw committed data, but different values
```

**Default in**: PostgreSQL, Oracle, SQL Server

**Implementation**: Read locks are acquired and released immediately (not held until commit). Write locks held until commit.

### 8.6 Repeatable Read

Once a transaction reads a data item, it will always see the same value for that item throughout the transaction (no dirty reads, no non-repeatable reads). However, **phantom rows** can still appear.

```
T₁: BEGIN (REPEATABLE READ)
    SELECT * FROM emp WHERE dept = 'Eng';  → {Alice, Bob}
    -- T₂: INSERT INTO emp VALUES('Carol', 'Eng'); COMMIT;
    SELECT * FROM emp WHERE dept = 'Eng';  → {Alice, Bob, Carol}  ← phantom!
    -- Re-read of Alice and Bob's rows gives same data (no fuzzy reads)
    -- But a new row (Carol) appeared — this is a phantom
```

**Implementation**: Read locks held until commit. But new rows matching the predicate are not locked (no predicate locking).

### 8.7 Serializable

The strongest isolation level. Transactions execute as if they were serial. No anomalies of any kind.

```
T₁: BEGIN (SERIALIZABLE)
    SELECT * FROM emp WHERE dept = 'Eng';  → {Alice, Bob}
    -- T₂: INSERT INTO emp VALUES('Carol', 'Eng');
    -- T₂ is BLOCKED (or T₁ would fail on commit in SSI) until T₁ completes
    SELECT * FROM emp WHERE dept = 'Eng';  → {Alice, Bob}  ← same result
COMMIT;
```

**Implementation options:**
- **Strict 2PL with predicate locks** (or index-range locks): traditional approach
- **Serializable Snapshot Isolation (SSI)**: PostgreSQL's approach (optimistic, MVCC-based)

---

## 9. The Phantom Problem

### 9.1 Why Phantoms Are Special

The phantom problem is fundamentally different from dirty reads and non-repeatable reads:

- Dirty/non-repeatable reads involve **existing rows** being modified
- Phantoms involve **new rows** being inserted (or existing rows being modified to match a predicate)

Standard row-level locking cannot prevent phantoms because **you cannot lock rows that do not yet exist**.

### 9.2 Solutions to the Phantom Problem

**Predicate Locking:**
Lock based on the query predicate rather than individual rows.

```
T₁: SELECT * FROM emp WHERE dept = 'Eng'
    → Lock: "no other transaction may insert/update/delete
             any row where dept = 'Eng'"
```

Predicate locking is expensive in general (predicates can be arbitrarily complex).

**Index-Range Locking (Next-Key Locking):**
A practical approximation. Lock a range of index entries, including the "gap" between entries.

```
B+Tree index on dept:
... ──→ [Acctg] ──→ [Eng] ──→ [HR] ──→ [Sales] ──→ ...

Lock the range from 'Eng' to 'HR' (exclusive).
This prevents any INSERT with dept = 'Eng'.

MySQL InnoDB calls this "next-key locking."
```

**Serializable Snapshot Isolation (SSI):**
PostgreSQL's approach. Detect serialization anomalies at commit time using dependency tracking (no predicate locks needed).

---

## 10. Snapshot Isolation

### 10.1 Concept

**Snapshot Isolation (SI)** is a multiversion concurrency control scheme that gives each transaction a consistent **snapshot** of the database as of the transaction's start time.

```
Database state at time 100: A=50, B=100

T₁ starts at time 100: sees snapshot {A=50, B=100}
T₂ starts at time 100: sees snapshot {A=50, B=100}

T₁: write(A=75)
T₂: read(A) → 50  (reads from its snapshot, not T₁'s write)
T₁: commit at time 105

T₂: read(A) → 50  (STILL 50! T₂'s snapshot is from time 100)
T₂: commit at time 110
```

### 10.2 First-Committer-Wins Rule

SI prevents lost updates using the **first-committer-wins** (FCW) rule:

```
T₁ starts at time 100, T₂ starts at time 100

T₁: write(A=75) ............ commit → succeeds (first to commit A)
T₂: write(A=80) ......................... commit → ABORTED!
     (T₂ tried to write A, but A was already modified and
      committed by T₁ since T₂'s snapshot)
```

### 10.3 Snapshot Isolation vs. Serializability

SI prevents dirty reads, non-repeatable reads, and phantoms. But SI is **NOT serializable**! It permits the **write skew** anomaly.

**Write Skew Example:**

```
Constraint: At least one doctor must be on call.
Initially: doctor_A.on_call = true, doctor_B.on_call = true

T₁ (snapshot): sees A.on_call=T, B.on_call=T
T₂ (snapshot): sees A.on_call=T, B.on_call=T

T₁: "The other doctor (B) is on call, so I can go off call"
    UPDATE doctors SET on_call = false WHERE name = 'A';

T₂: "The other doctor (A) is on call, so I can go off call"
    UPDATE doctors SET on_call = false WHERE name = 'B';

T₁: commit  ← succeeds (T₂ hasn't committed yet, no WW conflict on A)
T₂: commit  ← succeeds (T₁ modified A, T₂ modified B, no WW conflict!)

Result: A.on_call = false, B.on_call = false
→ NOBODY is on call! Constraint violated!
```

In a serializable execution, one transaction would see the other's update and maintain the constraint. SI allows this because T₁ and T₂ read from their snapshots and write to **different** items.

### 10.4 Serializable Snapshot Isolation (SSI)

**SSI** extends SI to detect and prevent anomalies like write skew. It tracks **read-write dependencies** between concurrent transactions and aborts transactions that form dangerous structures.

```
SSI detects "dangerous structures":
  T₁ →(rw)→ T₂ →(rw)→ T₃
  (where T₁ reads something T₂ overwrites,
   and T₂ reads something T₃ overwrites)

If this pattern is detected, one of the transactions is aborted.
```

**PostgreSQL**: Uses SSI for `SERIALIZABLE` isolation level since version 9.1. It is an optimistic approach -- transactions proceed without blocking, and conflicts are detected at commit time.

### 10.5 SI in Practice

| Database | Default Level | SI Support | True Serializable |
|---|---|---|---|
| PostgreSQL | Read Committed | Yes (REPEATABLE READ) | Yes (SSI) |
| Oracle | Read Committed | Yes (SERIALIZABLE*) | No (SI only, not true serial) |
| MySQL InnoDB | Repeatable Read | Partial (gap locking) | Yes (lock-based) |
| SQL Server | Read Committed | Yes (SNAPSHOT) | Yes (lock-based) |
| CockroachDB | Serializable | Yes | Yes (SSI) |

*Note: Oracle's "SERIALIZABLE" level actually provides Snapshot Isolation, not true serializability.

---

## 11. Testing for Serializability: Complete Algorithm

### Step-by-Step Algorithm

Given a schedule `S` with transactions `T₁, T₂, ..., Tₙ`:

```
SERIALIZABILITY-TEST(S):

1. Initialize precedence graph G with nodes T₁, ..., Tₙ

2. For each data item X accessed by two or more transactions:
   a. For each pair of operations on X in temporal order:
      - If r_i(X) precedes w_j(X) and i ≠ j:
          add edge T_i → T_j  (RW conflict)
      - If w_i(X) precedes r_j(X) and i ≠ j:
          add edge T_i → T_j  (WR conflict)
      - If w_i(X) precedes w_j(X) and i ≠ j:
          add edge T_i → T_j  (WW conflict)

3. Check if G has a cycle:
   - Use DFS or topological sort
   - If no cycle: S is conflict serializable
     The topological sort gives the equivalent serial order
   - If cycle exists: S is NOT conflict serializable
```

### Detailed Example

```
Schedule S:
  r₁(A) r₂(A) w₁(A) r₃(A) w₃(A) w₂(B) r₃(B) w₁(B) c₁ c₂ c₃

Step 1: Nodes: T₁, T₂, T₃

Step 2: Analyze conflicts by data item:

  Data item A:
    r₁(A) < w₁(A): same transaction, skip
    r₂(A) < w₁(A): RW conflict → T₂ → T₁
    r₂(A) < w₃(A): RW conflict → T₂ → T₃
    w₁(A) < r₃(A): WR conflict → T₁ → T₃
    w₁(A) < w₃(A): WW conflict → T₁ → T₃

  Data item B:
    w₂(B) < r₃(B): WR conflict → T₂ → T₃
    w₂(B) < w₁(B): WW conflict → T₂ → T₁
    r₃(B) < w₁(B): RW conflict → T₃ → T₁

Step 3: Precedence graph:

    T₂ → T₁ (from A: RW and B: WW)
    T₂ → T₃ (from A: RW and B: WR)
    T₁ → T₃ (from A: WR, WW)
    T₃ → T₁ (from B: RW)

         T₂
        ↙  ↘
      T₁ ⇄ T₃

    Cycle: T₁ → T₃ → T₁

    Result: NOT conflict serializable
```

---

## 12. Exercises

### Conceptual Questions

**Exercise 1**: For each ACID property, give a concrete example of what could go wrong if that property were violated. Use a banking scenario with accounts and transfers.

**Exercise 2**: Draw the transaction state diagram and trace the states for the following scenario:
- Transaction T begins and reads data item A
- T computes a new value and writes A
- The system detects a constraint violation
- T is rolled back

**Exercise 3**: Explain why every cascadeless schedule is recoverable, but not every recoverable schedule is cascadeless. Provide example schedules for each case.

### Serializability Analysis

**Exercise 4**: Determine whether each of the following schedules is conflict serializable. If yes, give the equivalent serial order. If no, identify the cycle in the precedence graph.

(a) `r₁(A) r₂(B) w₁(B) w₂(A) c₁ c₂`

(b) `r₁(A) w₂(A) w₁(A) r₂(A) c₁ c₂`

(c) `r₃(B) r₁(A) w₃(A) r₂(B) w₂(A) w₁(B) c₁ c₂ c₃`

(d) `r₁(A) r₂(B) r₃(C) w₁(B) w₂(C) w₃(A) c₁ c₂ c₃`

**Exercise 5**: Given the schedule:
```
r₁(X) r₂(X) w₂(X) r₁(Y) w₁(Y) w₂(Y) c₁ c₂
```
(a) Draw the precedence graph.
(b) Is it conflict serializable?
(c) Is it recoverable?
(d) Is it cascadeless?
(e) Is it strict?

**Exercise 6**: Construct a schedule that is:
(a) View serializable but NOT conflict serializable
(b) Recoverable but NOT cascadeless
(c) Cascadeless but NOT strict

### Isolation Levels

**Exercise 7**: For each scenario, identify the minimum isolation level required to prevent the anomaly:

(a) An inventory system where reading an uncommitted quantity could lead to overselling.

(b) A report that sums account balances, and the sum must be consistent (no partially-applied transfers).

(c) A query that counts employees per department, and the count must not change within the same transaction.

**Exercise 8**: Write a concrete SQL scenario that demonstrates the **write skew** anomaly under Snapshot Isolation. Explain why this cannot happen under true Serializable isolation.

**Exercise 9**: Consider a schedule with three transactions:

```
T₁: reads and writes A
T₂: reads A and writes B
T₃: reads B and writes C
```

(a) If running under Read Committed, what anomalies can occur?
(b) Under Repeatable Read?
(c) Under Serializable?

### Advanced Questions

**Exercise 10**: Prove that the number of possible serial schedules for `n` transactions is `n!`. Then explain why testing all `n!` serial schedules to verify serializability is impractical, and how the precedence graph provides a polynomial-time alternative.

**Exercise 11**: Oracle's `SERIALIZABLE` isolation level actually provides Snapshot Isolation, not true serializability. Design a test case (specific table schema and two concurrent transactions) that would expose this difference. What result would Oracle give vs. a truly serializable system?

**Exercise 12**: A system uses Repeatable Read isolation. Transaction T₁ runs `SELECT COUNT(*) FROM orders WHERE status = 'pending'` twice within the same transaction. Between the two selects, transaction T₂ inserts a new order with status = 'pending' and commits.

(a) What values does T₁ see for the two COUNT queries?
(b) Is this a phantom read?
(c) How would Serializable isolation handle this differently?
(d) How does MySQL InnoDB's "Repeatable Read" with gap locking handle this compared to the SQL standard definition?

---

**Previous**: [Indexing](./09_Indexing.md) | **Next**: [Concurrency Control](./11_Concurrency_Control.md)
