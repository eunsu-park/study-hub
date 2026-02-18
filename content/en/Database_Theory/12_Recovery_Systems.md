# 12. Recovery Systems

**Previous**: [Concurrency Control](./11_Concurrency_Control.md) | **Next**: [NoSQL and NewSQL](./13_NoSQL_and_NewSQL.md)

---

## Learning Objectives

- Classify types of failures and understand storage hierarchy
- Master log-based recovery techniques: deferred and immediate modification
- Understand the Write-Ahead Logging (WAL) protocol and why it is essential
- Learn checkpoint mechanisms including fuzzy checkpoints
- Study the ARIES recovery algorithm in detail: Analysis, Redo, Undo phases
- Compare recovery approaches: shadow paging vs. WAL-based recovery
- Understand buffer management policies: force/no-force, steal/no-steal
- Learn media recovery and backup strategies

---

## 1. Failure Classification

Database systems must handle various types of failures gracefully. Understanding failure types is essential for designing appropriate recovery mechanisms.

### 1.1 Transaction Failure

Failures within a single transaction. The rest of the system is unaffected.

**Logical errors:**
- Division by zero
- Constraint violation (e.g., unique key, foreign key)
- Application-level assertion failure
- User-initiated abort (ROLLBACK)

**System errors affecting a transaction:**
- Deadlock victim selection (transaction is aborted to break deadlock)
- Timeout expiration
- Out-of-memory for the transaction's workspace

```
Transaction Failure Recovery:
  T₁: read(A) write(A) read(B) ← constraint violation!
      ↓
  UNDO T₁'s write to A
  Release T₁'s locks
  T₁ enters ABORTED state
```

### 1.2 System Failure (Crash)

The entire system halts unexpectedly. Volatile storage (main memory, buffer pool) is lost, but disk contents survive.

**Causes:**
- Operating system crash
- Power failure (without UPS)
- Hardware failure (CPU, memory)
- Software bug in the DBMS

```
System Failure:
┌──────────────────────────────────┐
│  Main Memory (LOST)              │
│  ┌────────────┐ ┌────────────┐  │
│  │ Buffer Pool │ │ Lock Table │  │
│  │ (dirty     │ │ (all locks │  │
│  │  pages)    │ │  lost)     │  │
│  └────────────┘ └────────────┘  │
└──────────────────────────────────┘
         ↕  CRASH  ↕
┌──────────────────────────────────┐
│  Disk (SURVIVES)                 │
│  ┌────────────┐ ┌────────────┐  │
│  │ Database   │ │ Log File   │  │
│  │ Files      │ │ (WAL)      │  │
│  └────────────┘ └────────────┘  │
└──────────────────────────────────┘

After recovery:
- Committed transactions: REDO (effects must persist)
- Uncommitted transactions: UNDO (effects must be reversed)
```

### 1.3 Disk Failure (Media Failure)

Physical destruction of disk storage. Data on the affected disk is lost.

**Causes:**
- Disk head crash
- Controller failure
- Fire, flood, or other physical damage

```
Disk Failure Recovery:
  Primary disk destroyed
      ↓
  Restore from backup (tape, remote replica)
      ↓
  Replay log from backup point to failure point
      ↓
  Database restored to consistent state
```

### 1.4 Failure Summary

| Failure Type | What is Lost | Recovery Method |
|---|---|---|
| Transaction | Transaction's partial work | Undo (rollback) using log |
| System (crash) | Volatile storage (buffer pool) | Redo committed + Undo uncommitted |
| Disk (media) | Disk contents | Restore from backup + replay log |

---

## 2. Storage Types

### 2.1 Storage Hierarchy

```
Storage Hierarchy:

┌─────────────────────┐
│   Volatile Storage   │  CPU registers, cache, main memory
│   (fast, lost on     │  Access: nanoseconds
│    power failure)    │
└─────────┬───────────┘
          │
┌─────────┴───────────┐
│  Nonvolatile Storage │  SSD, HDD, flash
│  (survives power     │  Access: microseconds (SSD) to milliseconds (HDD)
│   failure, not disk  │
│   failure)           │
└─────────┬───────────┘
          │
┌─────────┴───────────┐
│   Stable Storage     │  Mirrored disks, RAID, remote replicas, tape
│  (ideally survives   │  Access: milliseconds to hours
│   ALL failures)      │
└─────────────────────┘
```

### 2.2 Stable Storage

True stable storage is an idealization -- no real storage survives all possible failures. In practice, we approximate stable storage using:

1. **RAID** (Redundant Array of Independent Disks):
   - RAID 1: Mirroring (two copies of every block)
   - RAID 5/6: Parity-based redundancy

2. **Remote replication**: Write to geographically separate locations

3. **Multiple copies**: Maintain copies on different media (disk + tape + cloud)

**For the log file**, we need the strongest durability guarantee, so the log is typically stored on the most reliable storage available.

---

## 3. Log-Based Recovery

### 3.1 The Log

The **log** (also called **journal** or **write-ahead log**) is a sequential record of all database modifications. It is the foundation of recovery.

```
Log Record Types:

┌──────────────────────────────────────────────────────────┐
│ <T_i, start>           Transaction T_i has started       │
│ <T_i, X, V_old, V_new> T_i changed X from V_old to V_new│
│ <T_i, commit>          Transaction T_i has committed     │
│ <T_i, abort>           Transaction T_i has aborted       │
│ <checkpoint L>          Checkpoint with active txn list L │
└──────────────────────────────────────────────────────────┘
```

**Example log:**

```
LSN  Log Record
───  ─────────────────────────────
1    <T₁, start>
2    <T₁, A, 100, 50>           T₁ changed A from 100 to 50
3    <T₁, B, 200, 250>          T₁ changed B from 200 to 250
4    <T₂, start>
5    <T₂, C, 300, 400>          T₂ changed C from 300 to 400
6    <T₁, commit>
7    <T₂, D, 500, 600>          T₂ changed D from 500 to 600
                                  ← CRASH (T₂ not committed)
```

### 3.2 Log Properties

**Key properties:**
1. **Append-only**: New records are always added at the end
2. **Sequential writes**: Efficient for disk I/O
3. **Write-ahead**: Log records written BEFORE corresponding data modifications (WAL rule)
4. **Force at commit**: Log records for a transaction must be on stable storage before the commit completes

### 3.3 Deferred Database Modification

In the **deferred modification** scheme, all writes are deferred until the transaction commits. During execution, modifications are recorded only in the log.

```
Deferred Modification:

T₁ executes:
  Log: <T₁, start>
  Log: <T₁, A, _, 50>           Record intent to write (old value not needed)
  Log: <T₁, B, _, 250>          Record intent to write
  Log: <T₁, commit>
  ── NOW apply writes to database ──
  Database: A = 50, B = 250

Recovery needs only REDO (no UNDO needed because uncommitted
transactions never wrote to the database).
```

**Recovery after crash:**
- **Committed transactions**: Redo all their writes (from the log)
- **Uncommitted transactions**: Ignore (they never modified the database)

**Advantages:**
- No UNDO needed (simpler recovery)
- No dirty data in the database

**Disadvantages:**
- All writes must fit in memory until commit
- Long transactions require large buffers
- Cannot release locks early (must hold all locks until commit + write)

### 3.4 Immediate Database Modification

In the **immediate modification** scheme, writes are applied to the database as they occur (possibly before commit). This is the approach used by all major database systems.

```
Immediate Modification:

T₁ executes:
  Log: <T₁, start>
  Log: <T₁, A, 100, 50>         Record both old and new values
  Database: A = 50               ← written to database immediately
  Log: <T₁, B, 200, 250>
  Database: B = 250              ← written to database immediately
  Log: <T₁, commit>

Recovery may need both REDO and UNDO.
```

**Recovery after crash:**
- **Committed transactions**: REDO their writes (to ensure changes persist)
- **Uncommitted transactions**: UNDO their writes (to reverse partial changes)

```
Recovery from the example log:

LSN  Log Record
───  ─────────────────────
1    <T₁, start>
2    <T₁, A, 100, 50>
3    <T₁, B, 200, 250>
4    <T₂, start>
5    <T₂, C, 300, 400>
6    <T₁, commit>
7    <T₂, D, 500, 600>
     ← CRASH

Recovery:
  T₁ committed (found <T₁, commit>) → REDO:
    A = 50   (from log record 2)
    B = 250  (from log record 3)

  T₂ not committed (no <T₂, commit>) → UNDO:
    D = 500  (reverse log record 7: restore old value)
    C = 300  (reverse log record 5: restore old value)

  REDO is forward (oldest to newest for committed txns)
  UNDO is backward (newest to oldest for uncommitted txns)
```

**Why REDO is needed even for committed transactions:**
Because a committed transaction's writes might still be in the buffer pool (volatile memory) and not yet flushed to disk when the crash occurred.

**Why UNDO is needed for uncommitted transactions:**
Because an uncommitted transaction's writes might have been flushed to disk (by the buffer manager) before the crash.

---

## 4. Write-Ahead Logging (WAL) Protocol

### 4.1 The WAL Rule

The **Write-Ahead Logging (WAL)** protocol is the fundamental rule that makes log-based recovery work:

**WAL Rule**: Before a modified data page is written from the buffer pool to disk, **all log records pertaining to that page must be flushed to stable storage**.

```
WAL Protocol:

Step 1: Write log record to log buffer (in memory)
Step 2: Flush log record to disk (stable storage)
Step 3: THEN (and only then) write modified data to disk

Correct order:
  Log buffer: <T₁, A, 100, 50> → Log on disk: <T₁, A, 100, 50> → Data on disk: A=50
                                                    ↑
                                          Must happen BEFORE data write

Why? If data is written first and the system crashes before the log record
is written, we cannot undo the change (we don't know the old value).
```

### 4.2 Commit Rule (Force-at-Commit)

When a transaction commits, all its log records must be flushed to stable storage **before** the commit is acknowledged to the user:

```
Commit Protocol:

T₁: ... operations ...
Log: <T₁, commit>

Step 1: Flush ALL of T₁'s log records to disk (including <T₁, commit>)
Step 2: Acknowledge commit to user ("Transaction committed")
Step 3: Modified data pages may be flushed later (lazy)

If crash occurs after Step 1 but before Step 3:
  The log has <T₁, commit> on disk → REDO T₁'s changes during recovery ✓

If crash occurs before Step 1:
  No <T₁, commit> on disk → UNDO T₁'s changes during recovery ✓
```

### 4.3 Group Commit

Flushing the log to disk for every single commit is expensive. **Group commit** batches multiple commits into a single disk write:

```
Without Group Commit:
  T₁ commit → flush → ack
  T₂ commit → flush → ack
  T₃ commit → flush → ack
  = 3 disk flushes (each ~5-10ms for HDD)

With Group Commit:
  T₁ commit → buffer
  T₂ commit → buffer     → single flush → ack T₁, T₂, T₃
  T₃ commit → buffer
  = 1 disk flush

Trade-off: slightly higher latency for individual commits,
but much higher throughput (more commits per second).
```

PostgreSQL uses group commit by default (`commit_delay` and `commit_siblings` settings).

### 4.4 WAL in PostgreSQL

```
PostgreSQL WAL Architecture:

┌──────────────────────────────────────────┐
│  Shared Buffers (Buffer Pool)            │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │Page1│ │Page2│ │Page3│ │Page4│  ...   │
│  │dirty│ │clean│ │dirty│ │clean│        │
│  └─────┘ └─────┘ └─────┘ └─────┘       │
└────────────────┬─────────────────────────┘
                 │
                 │  WAL must be flushed BEFORE
                 │  dirty page can be written
                 ↓
┌──────────────────────────────────────────┐
│  WAL Buffers → WAL Segment Files         │
│  ┌─────────────┐   ┌────────────────────┐│
│  │ WAL buffer  │ → │ 000000010000000042 ││
│  │ (in memory) │   │ 000000010000000043 ││
│  └─────────────┘   │ ... (16MB each)    ││
│                     └────────────────────┘│
└──────────────────────────────────────────┘
```

```sql
-- PostgreSQL WAL configuration
SHOW wal_level;           -- minimal, replica, logical
SHOW max_wal_size;        -- max WAL size before checkpoint (default: 1GB)
SHOW wal_buffers;         -- WAL buffer size (default: ~16MB)
SHOW synchronous_commit;  -- on (default), off (async), remote_write, etc.

-- View WAL statistics
SELECT * FROM pg_stat_wal;
```

---

## 5. Checkpoints

### 5.1 Why Checkpoints?

Without checkpoints, recovery must scan the **entire log** from the beginning to determine which transactions to redo and undo. For a database that has been running for months, this could take hours.

A **checkpoint** establishes a known good state, limiting how far back recovery must scan.

### 5.2 Simple (Quiescent) Checkpoint

The simplest checkpoint protocol:

```
SIMPLE-CHECKPOINT():
    1. Temporarily stop accepting new transactions
    2. Wait for all active transactions to complete (commit or abort)
    3. Flush all dirty pages from buffer pool to disk
    4. Write <checkpoint> record to log
    5. Flush log to disk
    6. Resume accepting transactions
```

```
Log with simple checkpoint:

... <T₁,start> <T₁,A,100,50> <T₁,commit> <checkpoint> <T₂,start> ...
                                     ↑
                            All prior transactions are
                            fully reflected on disk.
                            Recovery starts here.
```

**Problem**: Requires stopping all transactions -- unacceptable for production systems.

### 5.3 Non-Quiescent (Active) Checkpoint

A more practical checkpoint that does not require stopping all transactions:

```
ACTIVE-CHECKPOINT():
    1. Write <checkpoint L> to log, where L = list of active transactions
    2. Flush all dirty pages from buffer pool to disk
       (new transactions and writes can continue during this flush)
    3. Write <end_checkpoint> to log when flush is complete
```

```
Log with active checkpoint:

<T₁,start> <T₁,A,100,50> <checkpoint {T₁,T₂}> <T₂,B,200,300>
                                ↑
                       T₁ and T₂ were active when
                       checkpoint started

Recovery:
  Must scan back to earliest start of transactions in L
  (i.e., <T₁,start> since T₁ is in the active list)
```

### 5.4 Fuzzy Checkpoints

A **fuzzy checkpoint** is the most practical approach, used by ARIES and most modern systems:

```
FUZZY-CHECKPOINT():
    1. Write <begin_checkpoint> to log
    2. Record:
       - Active transaction table (ATT): all active transactions and their status
       - Dirty page table (DPT): all dirty pages in the buffer pool
    3. Write <end_checkpoint ATT, DPT> to log
    4. Flush log to stable storage
    5. Update master record (on disk) to point to <begin_checkpoint>

    Note: Dirty pages are NOT flushed during checkpoint!
    They are flushed asynchronously by the background writer.
```

```
Fuzzy Checkpoint Contents:

<begin_checkpoint>
<end_checkpoint>
  Active Transaction Table:
    ┌─────────┬──────────┬────────────────┐
    │ TxnID   │ Status   │ LastLSN        │
    ├─────────┼──────────┼────────────────┤
    │ T₁      │ Active   │ LSN 150        │
    │ T₂      │ Active   │ LSN 180        │
    └─────────┴──────────┴────────────────┘

  Dirty Page Table:
    ┌─────────┬────────────────┐
    │ PageID  │ RecoveryLSN    │
    ├─────────┼────────────────┤
    │ Page 5  │ LSN 120        │
    │ Page 12 │ LSN 145        │
    │ Page 8  │ LSN 170        │
    └─────────┴────────────────┘
```

**Advantages of fuzzy checkpoints:**
- No transaction stalling
- No forced page flushing during checkpoint
- Very fast checkpoint operation
- Recovery uses the checkpoint tables to determine the starting point

---

## 6. ARIES Recovery Algorithm

### 6.1 Overview

**ARIES** (Algorithm for Recovery and Isolation Exploiting Semantics) is the most widely used recovery algorithm. Developed at IBM Research by C. Mohan et al. in the early 1990s, it is the basis for recovery in DB2, SQL Server, PostgreSQL (adapted), and many other systems.

**Core principles:**
1. **Write-Ahead Logging (WAL)**: Log before data
2. **Repeating history during redo**: Redo all actions (including those of uncommitted transactions) to restore the exact pre-crash state
3. **Logging changes during undo**: Write Compensation Log Records (CLRs) during undo to ensure undo operations are idempotent

### 6.2 Key Data Structures

**Log Sequence Number (LSN):**
Every log record has a unique, monotonically increasing LSN. Each data page stores the LSN of the most recent log record that modified it (**pageLSN**).

```
Log Record:
┌───────┬──────┬────────┬────────────┬──────────┐
│  LSN  │ TxnID│ Type   │ PageID     │ PrevLSN  │
│       │      │(update,│            │(previous │
│       │      │ commit,│            │ log record│
│       │      │ CLR,...│            │ of same  │
│       │      │)       │            │ txn)     │
├───────┴──────┴────────┴────────────┴──────────┤
│  Before Image (old value)                      │
│  After Image (new value)                       │
│  UndoNextLSN (for CLR only)                   │
└────────────────────────────────────────────────┘
```

**Transaction Table (ATT - Active Transaction Table):**
Maintained in memory. For each active transaction:
- TransID
- Status (active, committing, aborting)
- LastLSN: LSN of the most recent log record for this transaction

**Dirty Page Table (DPT):**
Maintained in memory. For each dirty page in the buffer pool:
- PageID
- RecoveryLSN (recLSN): LSN of the **first** log record that dirtied this page since it was last flushed

```
Transaction Table:           Dirty Page Table:
┌─────────┬────────┬───────┐ ┌────────┬──────────┐
│ TransID │ Status │LastLSN│ │ PageID │ recLSN   │
├─────────┼────────┼───────┤ ├────────┼──────────┤
│ T₁      │ active │  45   │ │ P5     │  20      │
│ T₂      │ active │  60   │ │ P3     │  35      │
│ T₃      │commit  │  55   │ │ P8     │  42      │
└─────────┴────────┴───────┘ └────────┴──────────┘
```

### 6.3 Compensation Log Records (CLR)

A **CLR** is a special log record written during the undo phase. It records the undo of a previous update and includes a pointer (**UndoNextLSN**) to the next log record to undo.

```
CLR Structure:

┌──────────────────────────────────────────────┐
│ LSN: 100                                     │
│ TransID: T₁                                  │
│ Type: CLR                                    │
│ UndoNextLSN: 30  (← next record to undo)    │
│ Redo info: "restore A to old value 50"       │
│ PrevLSN: 80                                  │
└──────────────────────────────────────────────┘

Purpose:
- If the system crashes DURING recovery (during undo),
  the CLR ensures we don't undo the same operation twice.
- CLRs are REDO-only (they are never undone themselves).
```

### 6.4 The Three Phases

ARIES recovery proceeds in three phases:

```
Log Timeline:
─────────────────────────────────────────────────────────→
│                │              │                    │
│                │              │                    │
│           Begin Checkpoint    │              CRASH │
│                │              │                    │
│                │         End Checkpoint            │
│                │              │                    │
│    ┌───────────┴──────────────┴────────────────────┤
│    │                                               │
│    │         ← Phase 1: ANALYSIS →                 │
│    │           (scan forward from checkpoint)      │
│    │                                               │
│    │    ← Phase 2: REDO ──────────────────→        │
│    │      (from min recLSN in DPT, forward)        │
│    │                                               │
│    │    ← Phase 3: UNDO ──────────────────→        │
│    │      (from end of log, backward)              │
│    │                                               │
```

### 6.5 Phase 1: Analysis

**Goal**: Determine exactly which transactions were active at crash time and which pages were dirty.

```
ANALYSIS-PHASE():
    // Start from the most recent checkpoint
    Read <end_checkpoint ATT, DPT>
    Initialize ATT and DPT from checkpoint data

    // Scan forward from checkpoint to end of log
    for each log record in forward order:

        if record is <T_i, start>:
            Add T_i to ATT (status = active)

        if record is an UPDATE or CLR:
            if T_i not in ATT:
                Add T_i to ATT
            ATT[T_i].LastLSN = record.LSN

            if record.PageID not in DPT:
                Add page to DPT with recLSN = record.LSN

        if record is <T_i, commit>:
            ATT[T_i].status = committed

        if record is <T_i, abort>:
            ATT[T_i].status = aborting

        if record is <T_i, end>:
            Remove T_i from ATT

    // After analysis:
    // ATT contains all transactions that were active at crash time
    // DPT contains all pages that might have been dirty at crash time
```

### 6.6 Phase 2: Redo (Repeating History)

**Goal**: Restore the database to the exact state it was in at the moment of the crash. This includes redoing ALL modifications, even those of uncommitted transactions.

```
REDO-PHASE():
    // Start from the smallest recLSN in the DPT
    start_LSN = min(recLSN for all pages in DPT)

    // Scan forward from start_LSN to end of log
    for each UPDATE or CLR log record with LSN ≥ start_LSN:

        // Three conditions to SKIP redo:
        if record.PageID not in DPT:
            skip (page is clean)

        if DPT[record.PageID].recLSN > record.LSN:
            skip (page was already flushed after this update)

        // Read the page from disk
        page = read_page(record.PageID)
        if page.pageLSN ≥ record.LSN:
            skip (this update is already reflected on the page)

        // Apply the redo
        Apply the REDO information from the log record to the page
        page.pageLSN = record.LSN

    // After redo:
    // Database is in the exact pre-crash state
    // (including effects of uncommitted transactions)
```

**Why redo uncommitted transactions?**
ARIES "repeats history" to ensure that the undo phase works correctly. The undo phase assumes the database is in the exact pre-crash state, including all effects of all transactions (committed and uncommitted).

### 6.7 Phase 3: Undo

**Goal**: Roll back all transactions that were active at crash time (not committed).

```
UNDO-PHASE():
    // Build the undo list: all uncommitted transactions in ATT
    undo_list = {T_i in ATT where status != committed}

    // Collect the LastLSN for each transaction to undo
    // Process from the end of the log backward
    ToUndo = {ATT[T_i].LastLSN for T_i in undo_list}

    while ToUndo is not empty:
        // Pick the largest LSN (most recent operation)
        lsn = max(ToUndo)
        record = log_record at lsn
        T_i = record.TransID

        if record is a CLR:
            // CLRs are not undone; follow UndoNextLSN
            if record.UndoNextLSN != null:
                replace lsn in ToUndo with record.UndoNextLSN
            else:
                // All of T_i's updates have been undone
                Write <T_i, end> to log
                Remove T_i's entries from ToUndo

        else if record is an UPDATE:
            // Undo this update
            // Step 1: Write a CLR
            CLR = create_CLR(
                TransID = T_i,
                UndoNextLSN = record.PrevLSN,
                Redo_info = "restore to before-image"
            )
            Write CLR to log

            // Step 2: Apply the undo (restore before-image)
            Restore the page to its before-image value

            // Step 3: Follow the chain
            if record.PrevLSN != null:
                replace lsn in ToUndo with record.PrevLSN
            else:
                Write <T_i, end> to log
                Remove T_i's entries from ToUndo
```

### 6.8 Complete ARIES Example

```
Log at time of crash:

LSN  Record                              PrevLSN  UndoNextLSN
───  ────────────────────────────────     ───────  ───────────
10   <T₁, start>                           -
20   <T₁, P5, A: 10→20>                   10
30   <T₂, start>                           -
40   <T₂, P3, B: 30→40>                   30
50   <begin_checkpoint>
55   <end_checkpoint ATT={T₁,T₂}, DPT={P5:20, P3:40}>
60   <T₂, P3, B: 40→50>                   40
70   <T₃, start>                           -
80   <T₁, P5, C: 60→70>                   20
90   <T₃, P8, D: 80→90>                   70
100  <T₁, commit>
110  <T₃, P8, E: 15→25>                   90
     ← CRASH

Phase 1: Analysis (scan from LSN 55 to 110)
  LSN 60: T₂ update → ATT: T₂.LastLSN=60
  LSN 70: T₃ start → ATT adds T₃
  LSN 80: T₁ update → ATT: T₁.LastLSN=80, DPT: P5.recLSN stays 20
  LSN 90: T₃ update → ATT: T₃.LastLSN=90, DPT adds P8.recLSN=90
  LSN 100: T₁ commit → ATT: T₁.status=committed
  LSN 110: T₃ update → ATT: T₃.LastLSN=110

  Result:
    ATT = {T₁(committed, LSN=80), T₂(active, LSN=60), T₃(active, LSN=110)}
    DPT = {P5(recLSN=20), P3(recLSN=40), P8(recLSN=90)}

Phase 2: Redo (from min recLSN = 20)
  LSN 20: P5 in DPT, recLSN=20 ≤ 20, check pageLSN → redo if needed
  LSN 40: P3 in DPT, recLSN=40 ≤ 40, check pageLSN → redo if needed
  LSN 60: P3 in DPT, recLSN=40 ≤ 60, check pageLSN → redo if needed
  LSN 80: P5 in DPT, recLSN=20 ≤ 80, check pageLSN → redo if needed
  LSN 90: P8 in DPT, recLSN=90 ≤ 90, check pageLSN → redo if needed
  LSN 110: P8 in DPT, recLSN=90 ≤ 110, check pageLSN → redo if needed

Phase 3: Undo (uncommitted: T₂, T₃)
  ToUndo = {60 (T₂), 110 (T₃)}

  Step 1: Undo LSN 110 (T₃, P8, E: 15→25)
    Write CLR: <LSN=120, T₃, CLR, UndoNext=90>
    Restore E to 15
    ToUndo = {60, 90}

  Step 2: Undo LSN 90 (T₃, P8, D: 80→90)
    Write CLR: <LSN=130, T₃, CLR, UndoNext=70>
    Restore D to 80
    ToUndo = {60, 70}

  Step 3: Undo LSN 70 → it's <T₃, start>
    Actually LSN 70 is a start record, so T₃ has no PrevLSN from 70.
    We trace via PrevLSN of LSN 90 → 70 is start.
    Write <T₃, end>
    ToUndo = {60}

  Step 4: Undo LSN 60 (T₂, P3, B: 40→50)
    Write CLR: <LSN=140, T₂, CLR, UndoNext=40>
    Restore B to 40
    ToUndo = {40}

  Step 5: Undo LSN 40 (T₂, P3, B: 30→40)
    Write CLR: <LSN=150, T₂, CLR, UndoNext=30>
    Restore B to 30
    ToUndo = {30}

  Step 6: LSN 30 is <T₂, start>
    Write <T₂, end>
    ToUndo = {} → DONE

Final state:
  T₁'s changes persist (committed)
  T₂'s changes undone (B restored to 30)
  T₃'s changes undone (D restored to 80, E restored to 15)
```

### 6.9 Crash During Recovery

One of ARIES's key strengths is handling **crashes during recovery**:

```
Scenario: System crashes during the UNDO phase

Original crash recovery was at Phase 3 (Undo):
  Undo of LSN 110 → wrote CLR (LSN 120)
  Undo of LSN 90  → wrote CLR (LSN 130)
  ← CRASH AGAIN

Second recovery:
  Phase 1 (Analysis): Finds T₂, T₃ still active
  Phase 2 (Redo): Redoes ALL log records including CLRs at LSN 120, 130
                  (This re-applies the undo operations)
  Phase 3 (Undo): For T₃, follows CLR at LSN 130 → UndoNextLSN = 70
                   T₃'s LSNs 110 and 90 are NOT undone again (CLRs skip them)

The CLR chain ensures undo operations are IDEMPOTENT.
No work is repeated, no matter how many times recovery crashes.
```

---

## 7. Shadow Paging

### 7.1 Concept

**Shadow paging** is an alternative to WAL-based recovery. Instead of logging changes, it maintains two page tables:

```
Shadow Paging:

Current Page Table:          Shadow Page Table (on disk):
┌───┬──────┐                 ┌───┬──────┐
│ 1 │ → P1'│ (modified)      │ 1 │ → P1 │ (original)
│ 2 │ → P2 │ (unchanged)     │ 2 │ → P2 │
│ 3 │ → P3'│ (modified)      │ 3 │ → P3 │ (original)
│ 4 │ → P4 │ (unchanged)     │ 4 │ → P4 │
└───┴──────┘                 └───┴──────┘

On commit: Replace shadow page table with current page table
On abort:  Discard current page table, revert to shadow
```

### 7.2 Shadow Paging vs. WAL

| Aspect | Shadow Paging | WAL (ARIES) |
|---|---|---|
| Recovery speed | Fast (just use shadow) | Slower (three phases) |
| Normal operation | Slower (copy-on-write) | Faster (in-place update) |
| Concurrent txns | Difficult to support | Naturally supports |
| Fragmentation | High (scattered pages) | Low |
| Commit overhead | Atomic page table swap | Log flush |
| Used in | SQLite (journal mode), CouchDB | PostgreSQL, MySQL, Oracle, SQL Server |

Shadow paging is rarely used in modern multi-user database systems due to poor support for concurrent transactions and page fragmentation.

---

## 8. Buffer Management Policies

### 8.1 The Four Combinations

Buffer management policies determine when dirty pages are written to disk:

**Steal Policy**: Can the buffer manager flush a dirty page belonging to an uncommitted transaction?
- **Steal**: Yes, dirty pages can be flushed at any time (requires UNDO capability)
- **No-Steal**: No, dirty pages of uncommitted transactions stay in the buffer

**Force Policy**: Must all dirty pages of a transaction be flushed to disk at commit time?
- **Force**: Yes, flush all dirty pages on commit (no REDO needed after crash)
- **No-Force**: No, dirty pages may be flushed later (requires REDO capability)

```
           │ No-Steal          │ Steal
───────────┼───────────────────┼─────────────────────
Force      │ No undo, no redo  │ Undo, no redo
           │ (simplest recovery│ (rare in practice)
           │  but worst perf)  │
───────────┼───────────────────┼─────────────────────
No-Force   │ No undo, redo     │ Undo AND redo
           │ (deferred mod.)   │ (ARIES, most systems)
           │                   │ ← Best performance
```

### 8.2 Why Steal/No-Force is Best

```
Steal advantage:
  Without steal: uncommitted transaction's dirty pages MUST stay in buffer
  → Large transactions can exhaust the buffer pool
  → Limits the number of concurrent transactions

  With steal: buffer manager can evict any page as needed
  → Better buffer utilization
  → Supports larger transactions

No-Force advantage:
  Without no-force: ALL dirty pages flushed at commit
  → Very slow commits (especially if many pages modified)
  → Random I/O pattern (scattered dirty pages)

  With no-force: pages flushed lazily by background writer
  → Fast commits (just flush the log)
  → Sequential I/O for the log
  → Pages flushed in batches (more efficient)
```

### 8.3 Interaction with Recovery

| Policy | UNDO needed? | REDO needed? | Explanation |
|---|---|---|---|
| Steal | Yes | - | Stolen page may have uncommitted data on disk |
| No-Steal | No | - | Uncommitted data never reaches disk |
| Force | - | No | All committed data is on disk at commit |
| No-Force | - | Yes | Committed data may still be only in buffer |

ARIES uses **Steal + No-Force**, which requires both UNDO and REDO capabilities -- but provides the best runtime performance.

---

## 9. Media Recovery and Backup Strategies

### 9.1 The Need for Backups

WAL-based recovery handles transaction failures and system crashes, but NOT media failures (disk destruction). For media recovery, we need **backups**.

```
Media Recovery:

Time: ─────────────────────────────────────────────→
      │         │              │              │
   Full Backup  │          Incremental     Disk Failure
      B₁        │          Backup B₂          ↓
                 │              │           Restore B₁
              Log continues     │           Apply B₂
                 │              │           Replay log from B₂ to failure
                 │              │              ↓
                 │              │           Database recovered!
```

### 9.2 Backup Types

**Full Backup (Base Backup):**
- Complete copy of the entire database
- Self-contained: can restore from this alone
- Large size, time-consuming
- Typically done weekly or monthly

**Incremental Backup:**
- Only pages/blocks changed since the last backup
- Smaller and faster than full backup
- Requires the base backup + all incrementals to restore
- Typically done daily

**Continuous Archiving (WAL Archiving):**
- Archive WAL segments as they are completed
- Combined with a base backup, allows **Point-in-Time Recovery (PITR)**

```
PostgreSQL Backup Strategy:

┌─────────────────────────────────────────────────────┐
│  Sunday: Full base backup (pg_basebackup)           │
│  Mon-Sat: Continuous WAL archiving                  │
│                                                     │
│  To restore to Wednesday 3:00 PM:                   │
│  1. Restore Sunday's base backup                    │
│  2. Replay WAL from Sunday to Wed 3:00 PM           │
│  3. Database is at the exact state of Wed 3:00 PM   │
└─────────────────────────────────────────────────────┘
```

### 9.3 PostgreSQL Backup Commands

```bash
# Full base backup
pg_basebackup -D /backup/base -Ft -z -P

# Configure WAL archiving (in postgresql.conf)
# archive_mode = on
# archive_command = 'cp %p /backup/wal/%f'

# Point-in-Time Recovery
# 1. Stop PostgreSQL
# 2. Restore base backup to data directory
# 3. Create recovery.signal file
# 4. Set restore_command and recovery_target_time in postgresql.conf
# restore_command = 'cp /backup/wal/%f %p'
# recovery_target_time = '2025-03-15 15:00:00'
# 5. Start PostgreSQL → automatic recovery to target time
```

### 9.4 Logical vs. Physical Backup

```
Physical Backup (pg_basebackup):
┌────────────────────────────────────────────┐
│ Copies raw data files (binary)             │
│ + Fast backup and restore                  │
│ + Supports PITR with WAL archiving         │
│ - Same PostgreSQL major version required   │
│ - Same architecture (OS, endianness)       │
└────────────────────────────────────────────┘

Logical Backup (pg_dump):
┌────────────────────────────────────────────┐
│ Exports SQL statements or custom format    │
│ + Version-independent                      │
│ + Can restore to different architecture    │
│ + Can selectively restore tables           │
│ - Slower backup and restore               │
│ - No PITR (point-in-time snapshot only)    │
└────────────────────────────────────────────┘
```

### 9.5 Backup Best Practices

```
The 3-2-1 Rule:
  3 copies of data (original + 2 backups)
  2 different storage media (disk + tape/cloud)
  1 offsite copy (different physical location)

Additional guidelines:
  - Test restores regularly (untested backup = no backup)
  - Monitor WAL archiving (gaps mean data loss risk)
  - Retention policy: keep backups for regulatory period
  - Encrypt backups (especially offsite ones)
  - Document the recovery procedure
```

---

## 10. Recovery in Modern Systems

### 10.1 Recovery and Replication

Modern systems often combine recovery with **replication** for both durability and high availability:

```
Synchronous Replication:

Primary ──WAL──→ Standby
   │                │
   │   Commit only after standby
   │   confirms WAL receipt
   │                │
   ▼                ▼
 Data File       Data File

If primary fails:
  Standby already has all committed WAL
  Promote standby to primary (seconds)
  No data loss (RPO = 0)
```

### 10.2 Performance Considerations

```
Recovery Time Components:

┌────────────────────────────────────────┐
│ Analysis Phase: Fast (scan log once)   │
│ Time: proportional to log since        │
│       last checkpoint                  │
├────────────────────────────────────────┤
│ Redo Phase: Can be slow                │
│ Time: proportional to # of log records │
│       since min(recLSN in DPT)         │
│ Optimization: parallel redo by page    │
├────────────────────────────────────────┤
│ Undo Phase: Usually fast               │
│ Time: proportional to # of updates     │
│       by uncommitted transactions      │
│ Note: Can be deferred (lazy undo)      │
└────────────────────────────────────────┘

To minimize recovery time:
  - Frequent checkpoints (less to redo)
  - Short transactions (less to undo)
  - Sufficient WAL buffer (fewer forced flushes)
```

### 10.3 Parallel Recovery

Modern systems perform redo in parallel:

```
Parallel Redo:

Log records: [P1, P3, P1, P2, P3, P1, P2, ...]

Thread 1 (Page P1): redo [LSN1, LSN3, LSN6, ...]
Thread 2 (Page P2): redo [LSN4, LSN7, ...]
Thread 3 (Page P3): redo [LSN2, LSN5, ...]

Records for the SAME page must be applied in order.
Records for DIFFERENT pages can be applied in parallel.
```

---

## 11. Exercises

### Conceptual Questions

**Exercise 1**: Classify each of the following failures and describe the appropriate recovery action:
(a) A transaction divides by zero
(b) A power outage occurs
(c) A disk head crashes, destroying the data disk
(d) A deadlock is detected
(e) The operating system kernel panics

**Exercise 2**: Explain the difference between volatile, nonvolatile, and stable storage. Why is "true" stable storage impossible to achieve in practice? How do real systems approximate it?

**Exercise 3**: Explain why the WAL (Write-Ahead Logging) protocol requires log records to be flushed to stable storage BEFORE the corresponding data pages. What could go wrong if data pages were flushed first?

### Log-Based Recovery

**Exercise 4**: Given the following log (no checkpoints):

```
LSN  Record
1    <T₁, start>
2    <T₁, A, 10, 20>
3    <T₂, start>
4    <T₂, B, 30, 40>
5    <T₁, C, 50, 60>
6    <T₁, commit>
7    <T₂, D, 70, 80>
8    <T₃, start>
9    <T₃, A, 20, 30>
10   <T₃, commit>
     ← CRASH
```

(a) Which transactions must be redone? Which must be undone?
(b) What are the final values of A, B, C, D after recovery?
(c) Show the complete recovery process step by step.

**Exercise 5**: Repeat Exercise 4, but with a checkpoint `<checkpoint {T₂}>` inserted between LSN 6 and LSN 7. How does the checkpoint change the recovery process?

### ARIES Algorithm

**Exercise 6**: Given the following log with a fuzzy checkpoint:

```
LSN  Record                              PrevLSN
10   <T₁, start>                           -
20   <T₁, P1, X: 5→10>                    10
30   <T₂, start>                           -
40   <T₂, P2, Y: 15→25>                   30
50   <begin_checkpoint>
55   <end_checkpoint ATT={T₁(20),T₂(40)}, DPT={P1:20, P2:40}>
60   <T₁, P3, Z: 35→45>                   20
70   <T₂, commit>
80   <T₃, start>                           -
90   <T₃, P1, X: 10→20>                   80
100  <T₁, P2, W: 50→60>                   60
     ← CRASH
```

(a) Perform the Analysis phase. Show the final ATT and DPT.
(b) Determine the starting LSN for the Redo phase.
(c) Perform the Redo phase. List which log records are redone (assume all pages need redo).
(d) Perform the Undo phase. Show all CLRs written and the final ToUndo set at each step.
(e) What are the final values of X, Y, Z, W after recovery?

**Exercise 7**: Explain why ARIES "repeats history" during the redo phase (i.e., redoes even uncommitted transactions' updates). Why not skip uncommitted transactions' updates and only redo committed ones?

**Exercise 8**: A system crashes during the UNDO phase of ARIES recovery. At the time of the second crash, CLRs have been written for some (but not all) undo operations. Explain step by step what happens during the second recovery attempt. Why is no work repeated?

### Buffer Management

**Exercise 9**: For each combination of steal/no-steal and force/no-force policies, explain:
(a) Whether UNDO capability is needed and why
(b) Whether REDO capability is needed and why
(c) The impact on runtime performance (commit latency, buffer utilization)
(d) Name a system or context where this combination is appropriate

**Exercise 10**: A transaction T modifies 1000 pages. Under a force policy, all 1000 pages must be flushed to disk at commit time. Under a no-force policy, only the log records (~10 KB) need to be flushed. If each page flush takes 5ms (random I/O) and the log flush takes 2ms (sequential I/O):

(a) Calculate the commit latency under force vs. no-force policies
(b) If the system processes 100 such transactions per second, what is the I/O bandwidth requirement under each policy?
(c) Why do virtually all modern database systems use no-force?

### Backup and Media Recovery

**Exercise 11**: A PostgreSQL database has the following backup schedule:
- Full base backup every Sunday at 2:00 AM
- Continuous WAL archiving

On Wednesday at 4:00 PM, a junior DBA accidentally runs `DROP TABLE orders;` and commits. The error is discovered at 5:00 PM. Describe the step-by-step Point-in-Time Recovery process to restore the database to Wednesday 3:59 PM (one minute before the DROP TABLE).

**Exercise 12**: Compare the recovery time objectives for three failure scenarios in a system with:
- Checkpoints every 5 minutes
- Base backup every week
- WAL archived continuously
- Synchronous standby replica

(a) Transaction failure (single transaction aborted): Estimated recovery time?
(b) System failure (server crashes): Estimated recovery time?
(c) Disk failure (primary data disk destroyed): Estimated recovery time? What is the difference between promoting the standby vs. restoring from backup?

---

**Previous**: [Concurrency Control](./11_Concurrency_Control.md) | **Next**: [NoSQL and NewSQL](./13_NoSQL_and_NewSQL.md)
