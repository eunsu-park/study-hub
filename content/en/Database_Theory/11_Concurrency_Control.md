# 11. Concurrency Control

**Previous**: [Transaction Theory](./10_Transaction_Theory.md) | **Next**: [Recovery Systems](./12_Recovery_Systems.md)

---

## Learning Objectives

- Understand why concurrency control is essential for database correctness
- Master lock-based protocols: shared/exclusive locks, two-phase locking, and its variants
- Analyze and handle deadlocks: detection, prevention, and timeout strategies
- Understand lock granularity and intention locks for multi-granularity locking
- Learn timestamp-based protocols and Thomas's write rule
- Understand Multiversion Concurrency Control (MVCC) and its implementation
- Compare Optimistic Concurrency Control (OCC) with pessimistic methods

---

## 1. Need for Concurrency Control

### Why Allow Concurrency?

Modern database systems handle thousands of concurrent transactions. Without concurrency:

```
Serial execution of 1000 transactions, each taking 10ms:
  Total time = 1000 × 10ms = 10 seconds

With 100 concurrent transactions:
  Overlap I/O waits with computation
  Total time ≈ 100ms (+ overhead)
```

**Benefits of concurrency:**
- **Improved throughput**: More transactions per second
- **Reduced response time**: Short transactions don't wait behind long ones
- **Better resource utilization**: CPU works while another transaction waits for I/O

### What Can Go Wrong Without Control?

Without concurrency control, interleaved execution can produce incorrect results even if each transaction is individually correct.

**Lost Update Problem:**

```
T₁: read(A) = 100    A = A - 10 = 90    write(A) = 90
T₂:       read(A) = 100    A = A + 20 = 120    write(A) = 120

Timeline:
T₁: read(A)=100  ───────────── write(A)=90
T₂:      read(A)=100 ──────────────── write(A)=120

Result: A = 120
Expected: A = 110 (both updates applied)
T₁'s update is LOST!
```

**Inconsistent Read Problem:**

```
Invariant: A + B = 200 (initially A=100, B=100)

T₁: read(A)=100   A=A-50=50   write(A)=50 ──── read(B)=100   B=B+50=150   write(B)=150
T₂:                                    read(A)=50 read(B)=100 → A+B = 150 ≠ 200!

T₂ reads A after T₁'s update but B before T₁'s update → inconsistent view
```

### The Role of Concurrency Control

The concurrency control subsystem ensures that the concurrent execution of transactions is **equivalent to some serial execution** (serializability), while maximizing the degree of concurrency.

```
                    ┌──────────────────────┐
Transactions ────→  │ Concurrency Control  │ ────→  Serializable Schedule
  T₁, T₂, T₃      │    Subsystem         │        (correct results)
                    │  - Locks             │
                    │  - Timestamps        │
                    │  - MVCC              │
                    └──────────────────────┘
```

---

## 2. Lock-Based Protocols

### 2.1 Lock Types

The most fundamental concurrency control mechanism uses **locks** on data items.

**Shared Lock (S-lock):**
- Acquired for **reading** a data item
- Multiple transactions can hold S-locks on the same item simultaneously
- Also called a **read lock**

**Exclusive Lock (X-lock):**
- Acquired for **writing** (modifying) a data item
- Only one transaction can hold an X-lock on an item at a time
- No other transaction can hold any lock (S or X) on the item simultaneously
- Also called a **write lock**

**Lock Compatibility Matrix:**

```
              Requested Lock
              ┌─────┬─────┐
              │  S  │  X  │
         ┌────┼─────┼─────┤
Held     │ S  │ Yes │ No  │
Lock     ├────┼─────┼─────┤
         │ X  │ No  │ No  │
         └────┴─────┴─────┘

S + S = Compatible (both can proceed)
S + X = Incompatible (must wait)
X + S = Incompatible (must wait)
X + X = Incompatible (must wait)
```

**Lock operations:**
- `lock-S(X)` or `S(X)`: Request shared lock on item X
- `lock-X(X)` or `X(X)`: Request exclusive lock on item X
- `unlock(X)` or `U(X)`: Release lock on item X

### 2.2 Lock Protocol Example

```
T₁ (transfer $50 from A to B):     T₂ (read A + B):
  lock-X(A)                           lock-S(A)     ← BLOCKED (T₁ holds X on A)
  read(A) = 100
  A = A - 50
  write(A) = 50
  lock-X(B)
  read(B) = 200
  B = B + 50
  write(B) = 250
  unlock(A)                           lock-S(A)     ← NOW GRANTED
  unlock(B)                           read(A) = 50
                                      lock-S(B)
                                      read(B) = 250
                                      print(A+B) = 300  ← CORRECT!
                                      unlock(A)
                                      unlock(B)
```

### 2.3 Problems with Basic Locking

Simply acquiring locks before access and releasing after use is **not sufficient** for serializability:

```
Incorrect Protocol (lock, use, unlock immediately):

T₁:  lock-X(A) read(A) write(A) unlock(A) lock-X(B) read(B) write(B) unlock(B)
T₂:                                lock-S(A) read(A) lock-S(B)
                                                          ↑ BLOCKED (T₁ holds X(B))

Timeline:
T₁: X(A) r(A) w(A) U(A) ──── X(B) r(B) w(B) U(B)
T₂:                   S(A) r(A) ────── S(B) r(B)

T₂ reads A AFTER T₁'s update but reads B BEFORE T₁'s update.
T₂ sees an inconsistent state. NOT serializable.

The problem: T₁ released the lock on A too early.
```

This motivates **Two-Phase Locking**.

---

## 3. Two-Phase Locking (2PL)

### 3.1 Basic Two-Phase Locking

A transaction follows the **Two-Phase Locking (2PL) protocol** if all locking operations precede the first unlock operation.

```
Phase 1: GROWING phase          Phase 2: SHRINKING phase
(acquire locks, no releases)    (release locks, no acquisitions)

Number of
Locks Held
    ^
    │        ╱╲
    │       ╱  ╲
    │      ╱    ╲
    │     ╱      ╲
    │    ╱        ╲
    │   ╱          ╲
    │  ╱            ╲
    │ ╱              ╲
    └──────┬──────────┬──→ Time
     Growing  Lock   Shrinking
     phase    point  phase
```

**Rules:**
1. A transaction can acquire locks in the growing phase
2. Once a transaction releases any lock, it enters the shrinking phase
3. No new locks can be acquired in the shrinking phase

**Theorem**: If all transactions in a schedule follow the 2PL protocol, then the schedule is **conflict serializable**.

**Proof sketch**: The "lock point" (the moment a transaction has acquired all its locks) defines the serialization order. If T_i's lock point precedes T_j's lock point, then T_i appears before T_j in the equivalent serial schedule. This ordering is consistent because conflicting operations respect the lock compatibility rules.

### 3.2 Example: 2PL Execution

```
T₁ (2PL):                           T₂ (2PL):
  lock-X(A)
  read(A)                             lock-S(B)
  A = A - 50                          read(B)
  write(A)                            lock-S(A) ← BLOCKED (T₁ holds X(A))
  lock-X(B)
  read(B)          ← Lock point T₁
  B = B + 50
  write(B)
  unlock(A)                           lock-S(A) ← NOW GRANTED (shrinking phase)
  unlock(B)                           read(A)
                                      ← Lock point T₂
                                      unlock(A)
                                      unlock(B)

Serialization order: T₁ before T₂ (T₁'s lock point is earlier)
T₂ sees the database AFTER T₁'s complete transfer. Correct!
```

### 3.3 Problem: Cascading Rollbacks under Basic 2PL

Basic 2PL ensures serializability but **not recoverability**:

```
T₁ (2PL):                           T₂:
  lock-X(A)
  read(A)
  write(A)
  unlock(A)          ← Shrinking phase: released A
                                       lock-S(A)
                                       read(A)   ← reads T₁'s uncommitted data!
  ...
  ABORT              ← T₁ fails!

T₂ read a value written by T₁, which has now aborted.
T₂ must also be rolled back → cascading rollback
```

### 3.4 Strict Two-Phase Locking (Strict 2PL)

**Strict 2PL** adds a rule: all **exclusive (X) locks** are held until the transaction commits or aborts.

```
Strict 2PL:
Number of
Locks Held
    ^
    │    S-locks may       X-locks released
    │    be released       at commit/abort
    │      │                    │
    │      ╱╲                   │
    │     ╱  ╲─── ─── ─── ─── ─┤
    │    ╱         S-locks      │
    │   ╱          released     │
    │  ╱              ╲         │
    │ ╱                ╲        │
    └───────────────────┬──────→ Time
                     commit/abort
```

**Properties:**
- Conflict serializable (inherits from 2PL)
- **Strict** schedules (no cascading rollbacks for X-locked items)
- Most commonly used in practice

### 3.5 Rigorous Two-Phase Locking (Rigorous 2PL)

**Rigorous 2PL** holds **all locks** (both S and X) until commit or abort.

```
Rigorous 2PL:
Number of
Locks Held
    ^
    │
    │    ╱──────────────────────┤
    │   ╱                       │
    │  ╱   All locks held       │
    │ ╱    until commit/abort   │
    │╱                          │
    └───────────────────┬──────→ Time
                     commit/abort
```

**Properties:**
- Conflict serializable
- Strict AND cascadeless
- Serialization order = commit order (easy to reason about)
- Simplest protocol to implement correctly
- The locking protocol used by most database systems

### 3.6 Comparison of 2PL Variants

| Variant | S-Lock Release | X-Lock Release | Guarantees |
|---|---|---|---|
| Basic 2PL | After lock point | After lock point | Conflict serializable |
| Strict 2PL | After lock point | At commit/abort | + Strict (no cascading from writes) |
| Rigorous 2PL | At commit/abort | At commit/abort | + Strict + Cascadeless |

---

## 4. Deadlocks

### 4.1 What is a Deadlock?

A **deadlock** occurs when two or more transactions are waiting for each other to release locks, creating a circular wait.

```
Deadlock Scenario:

T₁: lock-X(A)   ───── lock-X(B) ← BLOCKED (T₂ holds X(B))
T₂: lock-X(B)   ───── lock-X(A) ← BLOCKED (T₁ holds X(A))

T₁ waits for T₂ to release B
T₂ waits for T₁ to release A
Neither can proceed → DEADLOCK
```

```
Wait-For Graph:

T₁ ──waits for──→ T₂
 ↑                  │
 └──waits for──────┘

Cycle → Deadlock!
```

### 4.2 Deadlock Detection

**Wait-For Graph (WFG):**
- Nodes represent transactions
- Edge `T_i → T_j` means `T_i` is waiting for `T_j` to release a lock
- A **cycle** in the WFG indicates a deadlock

```
Detection Algorithm:
DETECT-DEADLOCK():
    Maintain wait-for graph G
    Periodically (or on each lock wait):
        if G contains a cycle:
            Select a victim transaction from the cycle
            Abort the victim
            Release its locks (breaks the cycle)
```

**Victim selection criteria:**
1. **Youngest transaction** (least work to redo)
2. **Transaction with fewest locks** (least disruption)
3. **Transaction closest to completion** (most work already done -- sometimes preferred to avoid wasting it)
4. **Transaction with lowest priority** (application-defined)

**Starvation prevention**: Ensure the same transaction is not repeatedly chosen as a victim. Common approach: increase priority after each abort.

**Example:**

```
Wait-For Graph with 4 transactions:

T₁ → T₂ → T₃ → T₁   (cycle involving T₁, T₂, T₃)
          ↗
     T₄ ─┘             (T₄ waits for T₃ but not in cycle)

Deadlock detected among T₁, T₂, T₃.
Choose victim (e.g., T₁ if it's the youngest).
Abort T₁ → releases locks → T₃ can proceed → T₂ can proceed → T₄ can proceed.
```

### 4.3 Deadlock Prevention

Instead of detecting deadlocks after they occur, we can **prevent** them from ever happening.

**Wait-Die Scheme (non-preemptive):**

```
When T_i requests a lock held by T_j:

  If T_i is OLDER than T_j (T_i has earlier timestamp):
      T_i WAITS (older waits for younger)
  Else:
      T_i DIES (younger is rolled back and restarted)
      (T_i is restarted with its ORIGINAL timestamp to avoid starvation)

Mnemonic: "Old waits, Young dies"
```

```
Example:
T₁ (ts=100, older)  requests lock held by T₂ (ts=200, younger)
  → T₁ WAITS (older waits for younger)

T₂ (ts=200, younger) requests lock held by T₁ (ts=100, older)
  → T₂ DIES (younger dies, gets rolled back)
```

**Wound-Wait Scheme (preemptive):**

```
When T_i requests a lock held by T_j:

  If T_i is OLDER than T_j:
      T_i WOUNDS T_j (forces T_j to roll back, T_i takes the lock)
  Else:
      T_i WAITS (younger waits for older)

Mnemonic: "Old wounds, Young waits"
```

```
Example:
T₁ (ts=100, older)  requests lock held by T₂ (ts=200, younger)
  → T₁ WOUNDS T₂ (T₂ is rolled back, T₁ gets the lock)

T₂ (ts=200, younger) requests lock held by T₁ (ts=100, older)
  → T₂ WAITS (younger waits for older)
```

**Comparison:**

| Scheme | Older requesting from Younger | Younger requesting from Older |
|---|---|---|
| Wait-Die | Older WAITS | Younger DIES (rolled back) |
| Wound-Wait | Older WOUNDS younger (preempts) | Younger WAITS |

Both schemes are **deadlock-free** because they impose a total ordering: in Wait-Die, transactions only wait for younger ones; in Wound-Wait, transactions only wait for older ones. Neither allows circular waits.

**Starvation**: Both schemes restart aborted transactions with their **original timestamp**, so they become increasingly "old" and eventually succeed.

### 4.4 Timeout-Based Approach

A simple, practical approach: if a transaction waits for a lock longer than a **timeout** threshold, assume deadlock and abort it.

```
LOCK-WITH-TIMEOUT(item, lock_type, timeout):
    start_time = current_time()
    while lock is not grantable:
        if current_time() - start_time > timeout:
            ABORT transaction  // assumed deadlock
        wait(short_interval)
    grant lock
```

**Advantages:**
- Simple to implement
- No overhead of maintaining wait-for graphs
- Works well in practice (most deadlocks resolve quickly)

**Disadvantages:**
- May abort transactions that are not actually deadlocked (just slow)
- Timeout too short: unnecessary aborts
- Timeout too long: real deadlocks persist for too long

**In practice**: Most database systems use a combination of timeout (as a first line) and wait-for graph detection (for accuracy).

### 4.5 Deadlock in Practice

```sql
-- PostgreSQL: deadlock detection with wait-for graph
-- Default deadlock_timeout = 1s
SET deadlock_timeout = '1s';

-- Session 1:
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- Then tries:
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
-- BLOCKED if session 2 holds lock on id=2

-- Session 2:
BEGIN;
UPDATE accounts SET balance = balance - 50 WHERE id = 2;
-- Then tries:
UPDATE accounts SET balance = balance + 50 WHERE id = 1;
-- BLOCKED if session 1 holds lock on id=1

-- After deadlock_timeout: PostgreSQL detects the deadlock
-- ERROR: deadlock detected
-- One session is aborted, the other proceeds
```

---

## 5. Lock Granularity

### 5.1 The Granularity Spectrum

Locks can be applied at different levels of the data hierarchy:

```
Granularity Hierarchy:

Database ──→ Coarsest (least concurrency, least overhead)
  │
  Schema
  │
  Table
  │
  Page (block)
  │
  Row (tuple) ──→ Finest (most concurrency, most overhead)
  │
  Field (attribute) ──→ Rarely used in practice
```

**Trade-off:**

| Fine Granularity (Row) | Coarse Granularity (Table) |
|---|---|
| High concurrency | Low concurrency |
| High overhead (many locks) | Low overhead (few locks) |
| Best for OLTP | Best for batch/analytical |

### 5.2 Intention Locks

To support **multi-granularity locking** (MGR), we need **intention locks**. An intention lock on a coarse-grained item indicates that a finer-grained lock exists (or will be requested) on a descendant.

**Lock types with intention locks:**

| Lock | Meaning |
|---|---|
| **IS** (Intention Shared) | Some descendant has or will have an S-lock |
| **IX** (Intention Exclusive) | Some descendant has or will have an X-lock |
| **S** (Shared) | Shared lock on this node and all descendants |
| **X** (Exclusive) | Exclusive lock on this node and all descendants |
| **SIX** (Shared + Intention Exclusive) | S-lock on this node, IX on some descendant |

### 5.3 Compatibility Matrix with Intention Locks

```
              Requested Lock
         ┌─────┬─────┬─────┬─────┬─────┐
         │ IS  │ IX  │  S  │ SIX │  X  │
    ┌────┼─────┼─────┼─────┼─────┼─────┤
    │ IS │ Yes │ Yes │ Yes │ Yes │ No  │
    ├────┼─────┼─────┼─────┼─────┼─────┤
H   │ IX │ Yes │ Yes │ No  │ No  │ No  │
e   ├────┼─────┼─────┼─────┼─────┼─────┤
l   │ S  │ Yes │ No  │ Yes │ No  │ No  │
d   ├────┼─────┼─────┼─────┼─────┼─────┤
    │SIX │ Yes │ No  │ No  │ No  │ No  │
    ├────┼─────┼─────┼─────┼─────┼─────┤
    │ X  │ No  │ No  │ No  │ No  │ No  │
    └────┴─────┴─────┴─────┴─────┴─────┘
```

### 5.4 Multi-Granularity Locking Protocol

To lock a node at any level:

```
LOCK-NODE(node, lock_type):
    // Step 1: Acquire intention locks on all ancestors
    for each ancestor from root down to parent of node:
        if lock_type is S or IS:
            acquire IS lock on ancestor
        if lock_type is X, IX, or SIX:
            acquire IX lock on ancestor

    // Step 2: Acquire the actual lock on the node
    acquire lock_type on node
```

**Example: Transaction T₁ reads row R₁ in table T, Transaction T₂ writes row R₂ in table T:**

```
              Database
             ┌───┴───┐
           IS(T₁)  IX(T₂)
             │       │
             Table T
           ┌───┴───┐
         IS(T₁)  IX(T₂)
           │       │
         Page P1  Page P2
         IS(T₁)  IX(T₂)
           │       │
         Row R1   Row R2
         S(T₁)   X(T₂)

T₁ and T₂ operate on different rows → no conflict at any level → concurrent execution!
```

**Example: Transaction T₃ wants to lock the entire table T for exclusive access:**

```
T₃ requests X-lock on Table T.
Check compatibility with existing locks:
  - IS(T₁) on Table T: IS is compatible with X? → NO!
  - IX(T₂) on Table T: IX is compatible with X? → NO!

T₃ must WAIT until T₁ and T₂ release their intention locks.
The intention locks prevented T₃ from getting a table-level X-lock
while row-level operations are in progress.
```

### 5.5 SIX Lock

The **SIX** (Shared + Intention Exclusive) lock is useful when a transaction needs to **read an entire table** but **write only a few rows**.

```
Without SIX:
  Option 1: S-lock on table (blocks all writers)
  Option 2: IX-lock on table + S-lock on each row (expensive, many locks)

With SIX:
  SIX-lock on table:
  - S component: reads the entire table
  - IX component: allows X-locks on specific rows

Example:
  Transaction: "Update salary for employees earning < 30000"
  SIX-lock on employees table:
    - S: reads all rows to find those with salary < 30000
    - IX → X: writes to the specific rows that match
```

### 5.6 Lock Escalation

When a transaction acquires too many fine-grained locks, the system may **escalate** to a coarser granularity:

```
Lock Escalation:
  T₁ has 5000 row-level S-locks on table T
  → System escalates to a single table-level S-lock
  → Releases all 5000 row locks
  → Reduces memory usage from lock manager

Threshold varies by system:
  - SQL Server: ~5000 locks per table
  - Oracle: does not use lock escalation (uses different approach)
```

**Trade-off**: Escalation reduces lock overhead but decreases concurrency (other transactions may be blocked by the table-level lock).

---

## 6. Timestamp-Based Protocols

### 6.1 Basic Timestamp Ordering (TO)

Instead of locks, each transaction `T_i` is assigned a **unique timestamp** `TS(T_i)` when it begins. The protocol ensures that conflicting operations execute in timestamp order.

**Data item metadata:**
- `W-timestamp(Q)`: Largest timestamp of any transaction that successfully wrote Q
- `R-timestamp(Q)`: Largest timestamp of any transaction that successfully read Q

**Protocol:**

```
Transaction T_i wants to READ data item Q:

  if TS(T_i) < W-timestamp(Q):
      // T_i is trying to read a value that was already overwritten
      // by a younger transaction → T_i is "too late"
      REJECT: Roll back T_i and restart with a new timestamp

  else:
      Allow the read
      R-timestamp(Q) = max(R-timestamp(Q), TS(T_i))


Transaction T_i wants to WRITE data item Q:

  if TS(T_i) < R-timestamp(Q):
      // A younger transaction already read the old value
      // T_i's write would invalidate that read → reject
      REJECT: Roll back T_i and restart with a new timestamp

  if TS(T_i) < W-timestamp(Q):
      // A younger transaction already wrote a newer value
      // T_i's write is "obsolete"
      REJECT: Roll back T_i and restart with a new timestamp

  else:
      Allow the write
      W-timestamp(Q) = TS(T_i)
```

**Properties:**
- Ensures conflict serializability (serialization order = timestamp order)
- **No deadlocks** (transactions never wait; they are rolled back instead)
- May cause **cascading rollbacks** (basic version)
- Higher abort rate than locking (pessimistic vs. optimistic trade-off)

### 6.2 Example: Timestamp Ordering

```
TS(T₁) = 100, TS(T₂) = 200

Initially: W-ts(A) = 0, R-ts(A) = 0, W-ts(B) = 0, R-ts(B) = 0

T₁: read(A)
    TS(T₁)=100 ≥ W-ts(A)=0 → allowed
    R-ts(A) = max(0, 100) = 100

T₂: read(A)
    TS(T₂)=200 ≥ W-ts(A)=0 → allowed
    R-ts(A) = max(100, 200) = 200

T₁: write(A)
    TS(T₁)=100 < R-ts(A)=200 → REJECTED!
    (T₂, which is younger, already read A. T₁'s write would
     invalidate T₂'s read.)
    T₁ is rolled back and restarted with a new timestamp.
```

### 6.3 Thomas's Write Rule

An optimization of basic timestamp ordering for writes:

```
Transaction T_i wants to WRITE data item Q:

  if TS(T_i) < R-timestamp(Q):
      REJECT (same as basic TO)

  if TS(T_i) < W-timestamp(Q):
      // In basic TO: REJECT
      // Thomas's Write Rule: IGNORE the write (skip it!)
      // Why? A younger transaction already wrote a newer value.
      // T_i's write would be overwritten anyway.
      // This is safe because no future transaction will see T_i's value.

  else:
      Allow the write
      W-timestamp(Q) = TS(T_i)
```

**Example:**

```
TS(T₁) = 100, TS(T₂) = 200

T₂: write(A) → W-ts(A) = 200
T₁: write(A) → TS(T₁)=100 < W-ts(A)=200

Basic TO: Roll back T₁
Thomas's Write Rule: Skip T₁'s write (it would be overwritten by T₂ anyway)
                     T₁ continues normally
```

**Caveat**: Thomas's write rule allows schedules that are **view serializable** but NOT necessarily **conflict serializable** (because it effectively reorders writes).

---

## 7. Multiversion Concurrency Control (MVCC)

### 7.1 Concept

**MVCC** maintains multiple versions of each data item. Each write creates a new version rather than overwriting the old value. Reads can access older versions, so readers never block writers and vice versa.

```
MVCC Versions:

Data item A:
  Version 1: value=100, written by T₁ at timestamp 100
  Version 2: value=150, written by T₃ at timestamp 150
  Version 3: value=200, written by T₅ at timestamp 200

T₂ (timestamp 120): reads version 1 (value=100)
  → The latest version with timestamp ≤ 120

T₄ (timestamp 180): reads version 2 (value=150)
  → The latest version with timestamp ≤ 180
```

### 7.2 MVCC Read and Write Rules

```
Transaction T_i reads data item Q:

    Find the version Q_k where:
    - W-timestamp(Q_k) ≤ TS(T_i)
    - W-timestamp(Q_k) is the largest such timestamp
    Return Q_k's value


Transaction T_i writes data item Q:

    Find the version Q_k such that:
    - W-timestamp(Q_k) ≤ TS(T_i) (closest older version)

    Let Q_j be the version with W-timestamp immediately after Q_k

    if TS(T_i) < R-timestamp(Q_k):
        REJECT (a younger transaction read Q_k and expects it unchanged)
    else:
        Create new version Q_i with W-timestamp = TS(T_i)
```

### 7.3 MVCC Benefits and Costs

**Benefits:**
- **Readers never block writers**: Reads always find an appropriate version
- **Writers never block readers**: Old versions remain available for concurrent readers
- **Consistent snapshots**: Each transaction sees a consistent point-in-time view
- **Reduced contention**: Major performance improvement for read-heavy workloads

**Costs:**
- **Storage overhead**: Multiple versions of each data item
- **Garbage collection**: Old versions must be cleaned up (vacuum in PostgreSQL)
- **Version traversal**: Finding the right version takes time (especially with many versions)

### 7.4 MVCC in PostgreSQL

PostgreSQL implements MVCC using **xmin** and **xmax** fields in each tuple (row version):

```
Tuple Header Fields:
┌────────┬────────┬───────────┬──────────┐
│  xmin  │  xmax  │  ctid     │  data    │
│ (creator│(deleter│(physical  │          │
│  txn ID)│ txn ID)│ location) │          │
└────────┴────────┴───────────┴──────────┘

xmin: Transaction ID that created this version
xmax: Transaction ID that deleted/updated this version (0 if live)
ctid: Physical location of the next version (for updates)
```

**How an UPDATE works in PostgreSQL:**

```
Before UPDATE:
┌──────────┬──────────┬───────────────────┐
│ xmin=100 │ xmax=0   │ name='Alice'      │  ← current version
└──────────┴──────────┴───────────────────┘

Transaction 200 executes: UPDATE emp SET name='ALICE' WHERE id=1;

After UPDATE:
┌──────────┬──────────┬───────────────────┐
│ xmin=100 │ xmax=200 │ name='Alice'      │  ← old version (dead tuple)
└──────────┴──────────┴───────────────────┘
┌──────────┬──────────┬───────────────────┐
│ xmin=200 │ xmax=0   │ name='ALICE'      │  ← new version
└──────────┴──────────┴───────────────────┘

Transaction 150 (started before 200): sees name='Alice' (xmin=100, xmax=200>150)
Transaction 250 (started after 200):  sees name='ALICE' (xmin=200≤250, xmax=0)
```

**Visibility rules (simplified):**
A tuple is visible to transaction T if:
1. `xmin` is committed AND `xmin ≤ snapshot_of(T)`
2. Either `xmax = 0` (not deleted) OR `xmax > snapshot_of(T)` (deleted after T's snapshot)

**VACUUM**: PostgreSQL's garbage collector removes dead tuples that are no longer visible to any active transaction.

```sql
-- Manual vacuum
VACUUM employees;

-- Vacuum with analysis (updates statistics too)
VACUUM ANALYZE employees;

-- See dead tuples
SELECT relname, n_dead_tup, n_live_tup
FROM pg_stat_user_tables
WHERE relname = 'employees';
```

### 7.5 MVCC in MySQL InnoDB

InnoDB stores MVCC data differently:

```
InnoDB MVCC:
- Clustered index stores the latest version
- Old versions are stored in the UNDO LOG (rollback segment)
- Each row has hidden columns: DB_TRX_ID, DB_ROLL_PTR

┌────────────────────────────────────────┐
│ Clustered Index (latest version)        │
│ DB_TRX_ID=200 │ DB_ROLL_PTR=ptr1       │
│ name='ALICE'                            │
└────────────────────────┬───────────────┘
                         │ (follow rollback pointer)
                         ▼
┌────────────────────────────────────────┐
│ Undo Log (previous version)             │
│ DB_TRX_ID=100 │ DB_ROLL_PTR=null       │
│ name='Alice'                            │
└────────────────────────────────────────┘
```

**Key difference from PostgreSQL:**
- PostgreSQL: new version in the heap, old version becomes dead tuple
- InnoDB: latest version in the clustered index, old versions in undo log
- InnoDB approach: less heap bloat, but undo log can grow large

---

## 8. Optimistic Concurrency Control (OCC)

### 8.1 Concept

**Optimistic Concurrency Control** (also called **validation-based** protocol) assumes conflicts are rare and lets transactions execute freely, validating correctness only at commit time.

```
Pessimistic (Locking):
  "Prevent conflicts from happening"
  Lock before access → guaranteed safe → may wait/block

Optimistic (OCC):
  "Assume no conflicts, check at the end"
  Execute freely → validate at commit → abort if conflict detected
```

### 8.2 Three Phases

Each transaction goes through three phases:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Phase 1:   │    │   Phase 2:   │    │   Phase 3:   │
│    READ      │ →  │  VALIDATION  │ →  │    WRITE     │
│              │    │              │    │              │
│ Read from DB │    │ Check for    │    │ Apply writes │
│ Write to     │    │ conflicts    │    │ to database  │
│ private      │    │ with other   │    │              │
│ workspace    │    │ transactions │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Phase 1 -- Read Phase:**
- All reads come from the database
- All writes go to a **private workspace** (not the database)
- No locks are acquired

**Phase 2 -- Validation Phase:**
- Check whether the transaction's reads and writes conflict with other concurrent transactions
- If validation succeeds, proceed to write phase
- If validation fails, abort and restart

**Phase 3 -- Write Phase:**
- Apply the changes from the private workspace to the database
- This phase is performed atomically (with brief locks)

### 8.3 Validation Rules

Each transaction `T_i` is assigned a timestamp at the **start of its validation phase**. Let `TS(T_i)` be this timestamp.

For validation of `T_i`, check against every transaction `T_j` where `TS(T_j) < TS(T_i)` (T_j validated earlier):

```
VALIDATE(T_i):
    for each T_j where TS(T_j) < TS(T_i):

        // Condition 1: T_j completed all three phases before T_i started read phase
        if T_j completed write phase before T_i started read phase:
            OK (no overlap at all)

        // Condition 2: T_j completed write phase before T_i started write phase,
        // AND T_j's write set does not intersect T_i's read set
        else if T_j completed write phase before T_i started validation:
            if WriteSet(T_j) ∩ ReadSet(T_i) == ∅:
                OK
            else:
                FAIL → abort T_i

        // Condition 3: T_j has not yet completed write phase
        // AND T_j's write set does not intersect T_i's read or write sets
        else:
            if WriteSet(T_j) ∩ ReadSet(T_i) == ∅
               AND WriteSet(T_j) ∩ WriteSet(T_i) == ∅:
                OK
            else:
                FAIL → abort T_i

    return VALID
```

### 8.4 Example: OCC Validation

```
T₁: ReadSet = {A, B}, WriteSet = {A}
T₂: ReadSet = {B, C}, WriteSet = {C}

Timeline:
T₁: ──Read──┤──Validate(ts=1)──┤──Write──┤
T₂: ────Read────┤──Validate(ts=2)──┤──Write──┤

Validating T₂ against T₁ (TS(T₁)=1 < TS(T₂)=2):
  T₁ completed write before T₂ started validation? → Yes (Condition 2)
  WriteSet(T₁) ∩ ReadSet(T₂) = {A} ∩ {B, C} = ∅ → No conflict!
  T₂ is VALID ✓

If instead WriteSet(T₁) = {B}:
  WriteSet(T₁) ∩ ReadSet(T₂) = {B} ∩ {B, C} = {B} → Conflict!
  T₂ is INVALID → abort and restart
```

### 8.5 When to Use OCC

**OCC is ideal when:**
- Conflicts are **rare** (mostly read-only workloads)
- Transactions are **short** (validation overhead is small relative to work)
- High contention would cause excessive blocking with locks

**OCC is poor when:**
- Conflicts are **frequent** (high abort rate wastes work)
- Transactions are **long** (redo cost is high after abort)
- Write-heavy workloads (many conflicts)

**Real-world use:**
- Google's Spanner uses a variant of OCC for read-write transactions
- Many web applications use "optimistic locking" patterns (version numbers)

```python
# Application-level optimistic locking pattern:
# Step 1: Read the record with its version
row = db.execute("SELECT *, version FROM products WHERE id = ?", [product_id])
old_version = row.version

# Step 2: Compute new values (in application code)
new_price = compute_new_price(row)

# Step 3: Write with version check
result = db.execute(
    "UPDATE products SET price = ?, version = version + 1 "
    "WHERE id = ? AND version = ?",
    [new_price, product_id, old_version]
)

# Step 4: Check if update succeeded
if result.rows_affected == 0:
    # Version changed → someone else modified it → retry
    raise ConflictError("Concurrent modification detected")
```

---

## 9. Comparison of Concurrency Control Methods

### 9.1 Summary Table

| Method | Approach | Deadlock | Cascade | Best For |
|---|---|---|---|---|
| Basic 2PL | Pessimistic (locks) | Possible | Possible | General purpose |
| Strict 2PL | Pessimistic (locks) | Possible | No (writes) | Most OLTP systems |
| Rigorous 2PL | Pessimistic (locks) | Possible | No | When simplicity matters |
| Timestamp Ordering | Optimistic (timestamps) | Impossible | Possible | Low-conflict workloads |
| Thomas's Write Rule | Optimistic (timestamps) | Impossible | Possible | Write-heavy, low conflict |
| MVCC | Multi-version | Depends on impl. | No (via snapshots) | Read-heavy, mixed |
| OCC (Validation) | Optimistic (validate) | Impossible | No | Read-mostly, short txns |

### 9.2 Decision Guide

```
Start
  │
  ├─ Workload is read-heavy?
  │   ├─ Yes → MVCC (PostgreSQL, Oracle style)
  │   └─ No → continue
  │
  ├─ Conflicts are rare?
  │   ├─ Yes → OCC or Timestamp Ordering
  │   └─ No → continue
  │
  ├─ Need range queries / predicate locking?
  │   ├─ Yes → 2PL with index-range locks
  │   └─ No → continue
  │
  ├─ Need strict recoverability?
  │   ├─ Yes → Strict 2PL or Rigorous 2PL
  │   └─ No → Basic 2PL may suffice
  │
  └─ Distributed system?
      ├─ Yes → MVCC + distributed timestamps (Spanner)
      └─ No → Strict 2PL is the safe default
```

### 9.3 What Real Systems Use

| Database | Primary Method |
|---|---|
| PostgreSQL | MVCC (heap-based) + SSI for Serializable |
| MySQL InnoDB | MVCC (undo log) + next-key locking for Serializable |
| Oracle | MVCC (undo tablespace) + row-level locking |
| SQL Server | Lock-based (default) + MVCC (SNAPSHOT isolation opt-in) |
| CockroachDB | MVCC + SSI (serializable by default) |
| Google Spanner | MVCC + TrueTime + 2PL (externally consistent) |

---

## 10. Exercises

### Conceptual Questions

**Exercise 1**: Explain the difference between shared (S) and exclusive (X) locks. Why can multiple S-locks coexist on the same data item, but not multiple X-locks?

**Exercise 2**: A transaction T follows 2PL but acquires all its locks at the very beginning (before any reads or writes) and releases them all at the very end (after the last operation). Is T following strict 2PL? Rigorous 2PL? Both? Explain.

**Exercise 3**: Compare the Wait-Die and Wound-Wait deadlock prevention schemes. For each, explain which transaction gets priority and why starvation is avoided.

### Lock-Based Protocol Problems

**Exercise 4**: Consider three transactions:
```
T₁: read(A), write(A), read(B), write(B)
T₂: read(B), write(B)
T₃: read(A), read(B)
```

(a) Show a possible execution under strict 2PL where T₁ and T₃ can proceed concurrently but T₂ must wait.
(b) Can deadlock occur between T₁ and T₂ under 2PL? If so, show the scenario.

**Exercise 5**: Consider the following lock requests arriving in order:

```
T₁: lock-S(A)
T₂: lock-X(B)
T₃: lock-S(A)
T₁: lock-X(B)     → wait (T₂ holds X(B))
T₂: lock-S(A)     → ?
T₃: lock-X(A)     → ?
```

(a) What happens at each step?
(b) Is there a deadlock? If so, identify the cycle in the wait-for graph.
(c) How would Wait-Die resolve this (assume TS(T₁) < TS(T₂) < TS(T₃))?

**Exercise 6**: In a multi-granularity locking system, transaction T₁ wants to read rows from table A and exclusively update specific rows in table B. Transaction T₂ wants to read the entire table B. Show the intention locks each transaction must acquire and determine whether they can execute concurrently.

### Timestamp and MVCC Problems

**Exercise 7**: Given TS(T₁)=100, TS(T₂)=150, TS(T₃)=200. Initial timestamps: W-ts(A)=0, R-ts(A)=0, W-ts(B)=0, R-ts(B)=0.

Execute the following operations using basic timestamp ordering. For each operation, state whether it is allowed or rejected, and update the timestamps.

```
T₂: read(A)
T₃: read(A)
T₁: write(A)    ← what happens?
T₂: write(A)
T₃: read(B)
T₂: write(B)    ← what happens?
T₁: read(B)     ← what happens?
```

Now repeat the exercise using Thomas's Write Rule. Which operations have different outcomes?

**Exercise 8**: In PostgreSQL's MVCC implementation, explain what happens step by step when:
1. Transaction T₁ (txid=100) inserts a row
2. Transaction T₂ (txid=150) reads the table
3. Transaction T₁ commits
4. Transaction T₃ (txid=200) reads the table
5. Transaction T₂ reads the table again

What does each transaction see at each step? Consider both READ COMMITTED and REPEATABLE READ isolation levels.

### OCC Problem

**Exercise 9**: Three transactions execute under OCC:

```
T₁: ReadSet={A,B}, WriteSet={A}     Start: t=0, Validate: t=5
T₂: ReadSet={B,C}, WriteSet={B}     Start: t=1, Validate: t=6
T₃: ReadSet={A,C}, WriteSet={C}     Start: t=2, Validate: t=7
```

(a) Validate T₂ against T₁. Does it pass?
(b) Validate T₃ against T₁ and T₂. Does it pass?
(c) If T₁'s WriteSet were {B} instead of {A}, which validations would change?

### Analysis and Design

**Exercise 10**: A banking application has these transaction types:
- Balance inquiry (read one account): 60% of transactions
- Transfer (write two accounts): 30% of transactions
- Month-end report (read all accounts): 10% of transactions

Which concurrency control scheme (strict 2PL, MVCC, or OCC) would you recommend? Justify your answer considering throughput, response time, and the characteristics of each transaction type.

**Exercise 11**: Explain why MVCC in PostgreSQL requires VACUUM. What happens if VACUUM is never run? What are the symptoms of a missing or slow VACUUM process (table bloat, transaction ID wraparound)?

**Exercise 12**: Google Spanner uses TrueTime (GPS + atomic clocks) to assign globally consistent timestamps to transactions. Explain why regular clock synchronization (NTP) is insufficient for a globally distributed MVCC system. What property does TrueTime provide that NTP cannot guarantee?

---

**Previous**: [Transaction Theory](./10_Transaction_Theory.md) | **Next**: [Recovery Systems](./12_Recovery_Systems.md)
