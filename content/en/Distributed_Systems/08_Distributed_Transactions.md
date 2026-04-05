# Lesson 8: Distributed Transactions and Atomic Commit

[Overview](./00_Overview.md) | [Previous: Byzantine Fault Tolerance](./07_Byzantine_Fault_Tolerance.md) | [Next: Replication Strategies](./09_Replication_Strategies.md)

---

## Learning Objectives

- Analyze why distributed transactions are fundamentally harder than local transactions
- Implement Two-Phase Commit (2PC) with write-ahead logging and understand the blocking problem
- Compare 2PC with Three-Phase Commit (3PC) and explain why 3PC fails under network partitions
- Describe modern distributed transaction protocols: Percolator, Spanner TrueTime, and Calvin
- Design compensating transactions using the Saga pattern for long-lived distributed workflows

---

## 1. Why Distributed Transactions Are Hard

### 1.1 The Fundamental Challenge

A local transaction on a single database has a simple contract: either all operations succeed (commit) or none take effect (abort). The database uses a write-ahead log (WAL) and recovery manager to guarantee this.

In a distributed system, a transaction spans multiple nodes. The challenge is that **each node can independently fail or be partitioned** from the others. Consider:

```
Transaction: Transfer $100 from Account A (Node 1) to Account B (Node 2)

  Node 1: Debit $100 from A   ← succeeds
  --- network partition ---
  Node 2: Credit $100 to B    ← never receives the message

Result: $100 vanishes. Neither atomic commit nor rollback occurred.
```

### 1.2 Partial Failures

Unlike local databases, distributed transactions must handle:

| Failure Mode | Description | Example |
|-------------|-------------|---------|
| Node crash | A participant crashes before/after voting | Coordinator dies in Phase 2 |
| Network partition | Participants cannot communicate | Split-brain between data centers |
| Message loss | Individual messages are dropped | Vote message never arrives |
| Message delay | Messages arrive arbitrarily late | Commit message delayed by minutes |
| Partial completion | Some participants commit, others don't | The core problem |

### 1.3 The Atomic Commit Problem

Formally, the **atomic commit** problem requires:

1. **Agreement**: All participants that decide must decide the same value (commit or abort)
2. **Validity**: If all participants vote "yes" and there are no failures, the decision is "commit"
3. **Termination**: Every non-faulty participant eventually decides
4. **Abort validity**: If any participant votes "abort", the decision must be "abort"

This is related to but distinct from the consensus problem: in atomic commit, **any participant can unilaterally force an abort**, whereas in consensus, the decision must reflect the majority.

---

## 2. Two-Phase Commit (2PC)

### 2.1 Protocol Overview

Two-Phase Commit (Gray, 1978) is the classic solution to atomic commit. It uses a designated **coordinator** that drives the protocol:

```
Phase 1: Voting (Prepare)
  Coordinator → all Participants: "Can you commit?"
  Participants → Coordinator: "Yes" (prepared) or "No" (abort)

Phase 2: Decision (Commit/Abort)
  If all voted "Yes":
    Coordinator → all Participants: "Commit"
  Else:
    Coordinator → all Participants: "Abort"
```

### 2.2 Detailed Message Flow

```
Coordinator          Participant A         Participant B
    │                     │                      │
    │  1. Write BEGIN      │                      │
    │     to WAL           │                      │
    │                     │                      │
    │──── PREPARE ────────▶│                      │
    │──── PREPARE ─────────────────────────────▶│
    │                     │                      │
    │                     │ 2. Acquire locks      │
    │                     │    Write PREPARED     │
    │                     │    to WAL             │
    │                     │                      │ 3. Acquire locks
    │                     │                      │    Write PREPARED
    │                     │                      │    to WAL
    │                     │                      │
    │◀─── VOTE YES ───────│                      │
    │◀─── VOTE YES ────────────────────────────│
    │                     │                      │
    │  4. Write COMMIT    │                      │
    │     to WAL           │                      │  ← POINT OF NO RETURN
    │                     │                      │
    │──── COMMIT ─────────▶│                      │
    │──── COMMIT ──────────────────────────────▶│
    │                     │                      │
    │                     │ 5. Apply changes      │
    │                     │    Release locks      │
    │                     │    Write DONE to WAL  │
    │                     │                      │ 6. Apply changes
    │                     │                      │    Release locks
    │                     │                      │    Write DONE to WAL
    │                     │                      │
    │◀─── ACK ────────────│                      │
    │◀─── ACK ─────────────────────────────────│
    │                     │                      │
    │  7. Write END       │                      │
    │     to WAL           │                      │
```

### 2.3 Implementation

```python
"""
Two-Phase Commit Coordinator with Write-Ahead Logging

Implements the 2PC protocol with:
- Durable WAL for crash recovery
- Timeout-based abort for participant failures
- Presumed-abort optimization
"""

import time
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from enum import Enum
from threading import Timer


class TxnState(Enum):
    INITIATED = "initiated"
    PREPARING = "preparing"
    PREPARED = "prepared"       # all voted yes
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"


@dataclass
class WALEntry:
    """A write-ahead log entry for crash recovery."""
    txn_id: str
    state: TxnState
    participants: List[str]
    timestamp: float


class WriteAheadLog:
    """Durable write-ahead log for 2PC coordinator."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, entry: WALEntry):
        """Append entry and fsync for durability."""
        record = json.dumps({
            'txn_id': entry.txn_id,
            'state': entry.state.value,
            'participants': entry.participants,
            'timestamp': entry.timestamp
        }) + '\n'

        with open(self.path, 'a') as f:
            f.write(record)
            f.flush()
            os.fsync(f.fileno())

    def recover(self) -> Dict[str, WALEntry]:
        """Read all entries and return the latest state per transaction."""
        txns: Dict[str, WALEntry] = {}

        if not os.path.exists(self.path):
            return txns

        with open(self.path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                entry = WALEntry(
                    txn_id=record['txn_id'],
                    state=TxnState(record['state']),
                    participants=record['participants'],
                    timestamp=record['timestamp']
                )
                txns[entry.txn_id] = entry

        return txns


class Participant:
    """A 2PC participant (resource manager)."""

    def __init__(self, name: str):
        self.name = name
        self.prepared_txns: Set[str] = set()
        self.committed_txns: Set[str] = set()
        self.aborted_txns: Set[str] = set()
        self.data: Dict[str, int] = {}       # key-value store
        self.pending: Dict[str, Dict] = {}   # txn_id → {writes}
        self.is_alive = True

    def prepare(self, txn_id: str, writes: Dict[str, int]) -> bool:
        """Phase 1: vote on whether to commit.

        Returns True (vote yes) or False (vote no).
        """
        if not self.is_alive:
            raise ConnectionError(f"{self.name} is down")

        # Check if we can execute the writes (validation)
        for key, value in writes.items():
            if key.startswith("INVALID"):
                return False  # simulate a constraint violation

        # Acquire locks, write PREPARED to local WAL
        self.pending[txn_id] = writes
        self.prepared_txns.add(txn_id)
        return True

    def commit(self, txn_id: str):
        """Phase 2: apply the writes."""
        if not self.is_alive:
            raise ConnectionError(f"{self.name} is down")

        if txn_id in self.pending:
            for key, value in self.pending[txn_id].items():
                self.data[key] = value
            del self.pending[txn_id]
            self.prepared_txns.discard(txn_id)
            self.committed_txns.add(txn_id)

    def abort(self, txn_id: str):
        """Phase 2: discard the pending writes."""
        if not self.is_alive:
            raise ConnectionError(f"{self.name} is down")

        self.pending.pop(txn_id, None)
        self.prepared_txns.discard(txn_id)
        self.aborted_txns.add(txn_id)


class TwoPhaseCommitCoordinator:
    """2PC Coordinator with WAL and timeout-based abort."""

    PREPARE_TIMEOUT = 5.0  # seconds

    def __init__(self, wal_path: str = "/tmp/2pc/coordinator.wal"):
        self.wal = WriteAheadLog(wal_path)
        self.participants: Dict[str, Participant] = {}

    def register_participant(self, participant: Participant):
        """Register a participant (resource manager)."""
        self.participants[participant.name] = participant

    def execute_transaction(self, txn_writes: Dict[str, Dict[str, int]]) -> bool:
        """Execute a distributed transaction.

        Args:
            txn_writes: {participant_name: {key: value}} — writes for each participant

        Returns:
            True if committed, False if aborted.
        """
        txn_id = str(uuid.uuid4())[:8]
        participant_names = list(txn_writes.keys())

        # Log INITIATED
        self.wal.append(WALEntry(
            txn_id, TxnState.INITIATED, participant_names, time.time()
        ))

        # ─── Phase 1: Prepare ───
        self.wal.append(WALEntry(
            txn_id, TxnState.PREPARING, participant_names, time.time()
        ))

        votes = {}
        for name, writes in txn_writes.items():
            participant = self.participants.get(name)
            if participant is None:
                votes[name] = False
                continue

            try:
                vote = participant.prepare(txn_id, writes)
                votes[name] = vote
            except ConnectionError:
                votes[name] = False  # treat unreachable as abort

        all_yes = all(votes.values())

        # ─── Decision Point ───
        if all_yes:
            # COMMIT decision — this WAL write is the point of no return
            self.wal.append(WALEntry(
                txn_id, TxnState.COMMITTED, participant_names, time.time()
            ))
            decision = True
        else:
            self.wal.append(WALEntry(
                txn_id, TxnState.ABORTED, participant_names, time.time()
            ))
            decision = False

        # ─── Phase 2: Notify ───
        for name in participant_names:
            participant = self.participants.get(name)
            if participant is None:
                continue

            try:
                if decision:
                    participant.commit(txn_id)
                else:
                    participant.abort(txn_id)
            except ConnectionError:
                # Participant is down; will recover via WAL
                pass

        return decision

    def recover(self):
        """Recover after coordinator crash.

        Scan WAL to find in-flight transactions and complete them.
        """
        txns = self.wal.recover()

        for txn_id, entry in txns.items():
            if entry.state == TxnState.COMMITTED:
                # Decision was commit; ensure all participants committed
                for name in entry.participants:
                    participant = self.participants.get(name)
                    if participant and txn_id in participant.prepared_txns:
                        participant.commit(txn_id)

            elif entry.state == TxnState.PREPARING:
                # Decision was never made; abort
                for name in entry.participants:
                    participant = self.participants.get(name)
                    if participant:
                        participant.abort(txn_id)

            elif entry.state == TxnState.ABORTED:
                # Decision was abort; ensure all participants aborted
                for name in entry.participants:
                    participant = self.participants.get(name)
                    if participant and txn_id in participant.prepared_txns:
                        participant.abort(txn_id)


def demo_2pc():
    """Demonstrate 2PC with various scenarios."""

    # Setup
    coord = TwoPhaseCommitCoordinator("/tmp/2pc_demo/coordinator.wal")
    db1 = Participant("db1")
    db2 = Participant("db2")
    coord.register_participant(db1)
    coord.register_participant(db2)

    # Scenario 1: Successful transaction
    print("=" * 60)
    print("Scenario 1: Successful distributed transaction")
    print("=" * 60)

    result = coord.execute_transaction({
        "db1": {"account_A": 900},   # debit A by 100
        "db2": {"account_B": 1100},  # credit B by 100
    })
    print(f"Transaction result: {'COMMITTED' if result else 'ABORTED'}")
    print(f"db1 data: {db1.data}")
    print(f"db2 data: {db2.data}")

    # Scenario 2: One participant votes no
    print(f"\n{'=' * 60}")
    print("Scenario 2: Participant votes NO (constraint violation)")
    print("=" * 60)

    result = coord.execute_transaction({
        "db1": {"account_A": 800},
        "db2": {"INVALID_key": 1200},  # will fail validation
    })
    print(f"Transaction result: {'COMMITTED' if result else 'ABORTED'}")
    print(f"db1 data: {db1.data}")  # should be unchanged from scenario 1
    print(f"db2 data: {db2.data}")

    # Scenario 3: Participant crashes during prepare
    print(f"\n{'=' * 60}")
    print("Scenario 3: Participant crash during prepare")
    print("=" * 60)

    db2.is_alive = False  # simulate crash
    result = coord.execute_transaction({
        "db1": {"account_A": 700},
        "db2": {"account_B": 1200},
    })
    print(f"Transaction result: {'COMMITTED' if result else 'ABORTED'}")
    print(f"db1 data: {db1.data}")
    db2.is_alive = True  # recover

    # Scenario 4: Multiple successful transactions
    print(f"\n{'=' * 60}")
    print("Scenario 4: Multiple sequential transactions")
    print("=" * 60)

    for i in range(3):
        result = coord.execute_transaction({
            "db1": {"counter": i + 1},
            "db2": {"counter": i + 1},
        })
        print(f"Transaction {i+1}: {'COMMITTED' if result else 'ABORTED'}")

    print(f"Final db1 data: {db1.data}")
    print(f"Final db2 data: {db2.data}")


if __name__ == "__main__":
    demo_2pc()
```

### 2.4 The Blocking Problem

The fundamental weakness of 2PC: **if the coordinator crashes after sending COMMIT to some participants but not others, the uncommitted participants are stuck** — they cannot abort (because the coordinator may have committed) and cannot commit (because they haven't received the COMMIT message). They are **in doubt**.

```
Coordinator crashes after writing COMMIT to WAL but before sending COMMIT:

Participant A: received COMMIT → committed ✓
Participant B: waiting for decision → BLOCKED ✗

B holds locks and cannot release them.
B cannot abort (A may have committed).
B cannot commit (no coordinator decision received).

B must wait until the coordinator recovers or another participant
tells it the decision.
```

The blocking window is the time between the coordinator writing COMMIT to its WAL and all participants receiving the COMMIT message. During this window:

- Participants that voted YES hold locks and cannot release them
- The system is unavailable for any operation that conflicts with the locked resources
- No amount of timeout logic can safely resolve the situation without the coordinator

### 2.5 Presumed Abort Optimization

The **presumed abort** optimization reduces WAL writes and message complexity:

1. The coordinator does not log the ABORT decision (only COMMIT is logged)
2. If a participant asks about an unknown transaction, the coordinator responds "abort"
3. Participants that voted YES do not need to wait for an ACK to their YES vote

This works because aborting is always safe — it only means the participant undoes its tentative writes. The optimization saves one WAL write and one message round for the common case of aborted transactions.

```python
class PresumedAbortCoordinator:
    """2PC coordinator with presumed-abort optimization."""

    def decide(self, txn_id, votes, participant_names):
        if all(votes.values()):
            # Only log commits (not aborts)
            self.wal.append(WALEntry(
                txn_id, TxnState.COMMITTED, participant_names, time.time()
            ))
            return True
        else:
            # No WAL write for abort (saves I/O)
            # Participants will time out and abort
            return False

    def query_transaction(self, txn_id):
        """Participant asks about a transaction's status.

        If not in WAL → presume abort.
        """
        txns = self.wal.recover()
        if txn_id in txns and txns[txn_id].state == TxnState.COMMITTED:
            return "commit"
        return "abort"  # presumed abort
```

### 2.6 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| WAL writes (coordinator) | 2 (begin + commit/abort) | With presumed abort: 1 for commit |
| WAL writes (participant) | 2 (prepared + done) | Per participant |
| Network round trips | 2 | Prepare + Commit |
| Messages | 4N | N prepares + N votes + N commits + N acks |
| Latency | 2 × RTT + 2 × fsync | Dominated by fsync latency |

---

## 3. Three-Phase Commit (3PC)

### 3.1 Motivation: Eliminating Blocking

3PC (Skeen, 1981) addresses 2PC's blocking problem by adding a **pre-commit** phase that ensures all participants know the decision before committing:

```
Phase 1: Voting (same as 2PC)
  Coordinator → Participants: "Can you commit?"
  Participants → Coordinator: "Yes" or "No"

Phase 2: Pre-commit (NEW)
  If all voted "Yes":
    Coordinator → Participants: "Pre-commit" (prepare to commit)
    Participants → Coordinator: "ACK"
  Else:
    Coordinator → Participants: "Abort"

Phase 3: Commit
  Coordinator → Participants: "Commit"
```

### 3.2 How 3PC Avoids Blocking

The pre-commit phase guarantees a key invariant:

> If **any** participant has committed, then **all** participants have received the pre-commit message.

This means that if the coordinator crashes, the remaining participants can safely decide:
- If any participant has received pre-commit → commit (because the coordinator decided to commit)
- If no participant has received pre-commit → abort (safe because no one committed)

```
2PC blocking scenario (coordinator crashes after commit to A):
  A: committed, B: uncertain → B is STUCK

3PC equivalent scenario:
  A: received pre-commit, B: received pre-commit → BOTH can commit
  OR
  A: received pre-commit, B: did NOT receive pre-commit → IMPOSSIBLE
  (because coordinator sends pre-commit to all before proceeding)
```

### 3.3 Why 3PC Doesn't Work with Network Partitions

3PC's non-blocking property relies on a critical assumption: **no network partitions**. Under partitions, the protocol breaks down:

```
Coordinator sends pre-commit to A and B.
Network partition separates {Coordinator, A} from {B}.

Partition 1: {Coordinator, A}
  Coordinator sends COMMIT to A → A commits

Partition 2: {B}
  B times out waiting for COMMIT
  B runs recovery: "No one in my partition received pre-commit"
  B decides to ABORT

Result: A committed, B aborted → SAFETY VIOLATION
```

The problem: participants in different partitions can make contradictory decisions because they don't see the same set of pre-commit messages.

### 3.4 Comparison: 2PC vs 3PC

| Property | 2PC | 3PC |
|----------|-----|-----|
| Phases | 2 | 3 |
| Messages | 4N | 6N |
| WAL writes | 2 + 2N | 3 + 3N |
| Blocking on coordinator crash | YES | No (without partitions) |
| Safe under network partitions | Yes (blocks) | NO (can violate safety) |
| Practical usage | Ubiquitous | Almost none |
| Latency | 2 RTT | 3 RTT |

**Bottom line**: In practice, 3PC is rarely used because network partitions are unavoidable in real systems, and 3PC's safety violations are worse than 2PC's blocking. Modern systems either tolerate 2PC's blocking or use alternative approaches (Paxos Commit, Percolator, Sagas).

### 3.5 Paxos Commit: The Best of Both Worlds

Lamport and Gray (2006) proposed **Paxos Commit**: replace the coordinator with a Paxos group. This achieves non-blocking atomic commit even under partitions:

```
Instead of one coordinator:
  - Run a Paxos instance to decide "commit or abort" for each participant's vote
  - Participants send their votes to the Paxos acceptors
  - The Paxos group reliably records the decision

Benefit: no single point of failure, no blocking
Cost: higher message complexity (O(N²) for Paxos among coordinators)
```

This is the approach used by Google Spanner for its distributed transactions.

---

## 4. Percolator (Google, 2010)

### 4.1 Overview

Google Percolator (Peng and Dabek, 2010) implements **optimistic distributed transactions** with snapshot isolation over BigTable. It was built for Google's web indexing pipeline, processing updates to a multi-petabyte table.

Key properties:
- **Snapshot isolation** (not serializable)
- **Optimistic concurrency control** — no locks during execution; conflict detection at commit
- **Decentralized** — no central coordinator; each transaction manages its own commit
- **Timestamps** from a centralized **timestamp oracle** (TSO)

### 4.2 Timestamp Oracle

```python
import threading


class TimestampOracle:
    """Centralized timestamp oracle for Percolator transactions.

    In production, this is a highly available service that
    allocates monotonically increasing timestamps.
    """

    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()
        self._batch_size = 1000  # allocate in batches

    def get_timestamp(self) -> int:
        """Return a globally unique, monotonically increasing timestamp."""
        with self._lock:
            self._counter += 1
            return self._counter

    def get_timestamp_batch(self, count: int) -> List[int]:
        """Allocate a batch of timestamps efficiently."""
        with self._lock:
            start = self._counter + 1
            self._counter += count
            return list(range(start, self._counter + 1))
```

### 4.3 Data Model

Percolator stores data in BigTable with three column families per user column:

| Column Family | Purpose | Key | Value |
|--------------|---------|-----|-------|
| `data` | Actual values | `(row, col, start_ts)` | value |
| `lock` | Transaction locks | `(row, col, start_ts)` | `(primary_row, primary_col)` |
| `write` | Commit records | `(row, col, commit_ts)` | `start_ts` |

### 4.4 Write Path (Two-Phase Commit)

```python
class PercolatorTransaction:
    """A Percolator transaction with snapshot isolation."""

    def __init__(self, store, tso: TimestampOracle):
        self.store = store  # BigTable abstraction
        self.tso = tso
        self.start_ts = tso.get_timestamp()
        self.writes: Dict[tuple, Any] = {}  # (row, col) → value

    def get(self, row: str, col: str):
        """Read a value at the transaction's snapshot timestamp.

        Steps:
        1. Check for locks at timestamps ≤ start_ts (if found, wait or clean)
        2. Find the latest write record with commit_ts ≤ start_ts
        3. Use the write record's start_ts to find the data
        """
        # Check for conflicting locks
        lock = self.store.get_lock(row, col, max_ts=self.start_ts)
        if lock is not None:
            # Another transaction has a pending write; wait or clean up
            self._resolve_lock(lock)
            return self.get(row, col)  # retry

        # Find the latest committed write
        write = self.store.get_write(row, col, max_ts=self.start_ts)
        if write is None:
            return None

        data_ts = write.start_ts
        return self.store.get_data(row, col, data_ts)

    def set(self, row: str, col: str, value):
        """Buffer a write (applied at commit time)."""
        self.writes[(row, col)] = value

    def commit(self) -> bool:
        """Commit the transaction using two-phase commit.

        Phase 1 (Prewrite): Lock all cells, starting with the primary
        Phase 2 (Commit): Write commit records and remove locks
        """
        if not self.writes:
            return True

        # Choose a primary lock (first write)
        primary_key = list(self.writes.keys())[0]
        secondaries = list(self.writes.keys())[1:]

        # ─── Phase 1: Prewrite ───
        # Write the primary lock first
        if not self._prewrite(primary_key, is_primary=True):
            return False  # write-write conflict

        # Write secondary locks
        for key in secondaries:
            if not self._prewrite(key, is_primary=False, primary=primary_key):
                # Conflict on secondary; roll back primary
                self._rollback(primary_key)
                return False

        # ─── Phase 2: Commit ───
        commit_ts = self.tso.get_timestamp()

        # Commit primary first (point of no return)
        self._commit_primary(primary_key, commit_ts)

        # Commit secondaries (can be done asynchronously)
        for key in secondaries:
            self._commit_secondary(key, commit_ts)

        return True

    def _prewrite(self, key, is_primary=False, primary=None):
        """Prewrite: write data and lock.

        Checks for:
        1. Write-write conflict: another transaction committed after our start_ts
        2. Lock conflict: another transaction has an active lock
        """
        row, col = key
        value = self.writes[key]

        # Check for write-write conflict
        write = self.store.get_write(row, col, min_ts=self.start_ts)
        if write is not None:
            return False  # another transaction committed in our snapshot

        # Check for lock conflict
        lock = self.store.get_lock(row, col)
        if lock is not None:
            return False  # another transaction has a pending write

        # Write data at start_ts
        self.store.put_data(row, col, self.start_ts, value)

        # Write lock
        if is_primary:
            self.store.put_lock(row, col, self.start_ts, primary_ref=None)
        else:
            self.store.put_lock(row, col, self.start_ts, primary_ref=primary)

        return True

    def _commit_primary(self, key, commit_ts):
        """Commit primary: write record and remove lock atomically."""
        row, col = key
        # This must be atomic (BigTable row transaction)
        self.store.put_write(row, col, commit_ts, start_ts=self.start_ts)
        self.store.delete_lock(row, col, self.start_ts)

    def _commit_secondary(self, key, commit_ts):
        """Commit secondary: write record and remove lock."""
        row, col = key
        self.store.put_write(row, col, commit_ts, start_ts=self.start_ts)
        self.store.delete_lock(row, col, self.start_ts)

    def _rollback(self, key):
        """Roll back a prewritten key."""
        row, col = key
        self.store.delete_data(row, col, self.start_ts)
        self.store.delete_lock(row, col, self.start_ts)

    def _resolve_lock(self, lock):
        """Clean up a lock left by a crashed transaction.

        If the primary lock still exists → the transaction didn't commit → roll back
        If the primary lock is gone → the transaction committed → roll forward
        """
        primary_row, primary_col = lock.primary_ref or (lock.row, lock.col)

        # Check if primary committed
        write = self.store.get_write(primary_row, primary_col,
                                     min_ts=lock.start_ts)
        if write is not None:
            # Primary committed; roll forward this secondary
            self.store.put_write(lock.row, lock.col,
                                write.commit_ts, start_ts=lock.start_ts)
            self.store.delete_lock(lock.row, lock.col, lock.start_ts)
        else:
            # Primary didn't commit; roll back
            self.store.delete_data(lock.row, lock.col, lock.start_ts)
            self.store.delete_lock(lock.row, lock.col, lock.start_ts)
```

### 4.5 Key Insights

1. **No coordinator crash problem**: Each transaction is self-describing. If a transaction crashes after prewriting the primary but before committing, other transactions can detect the stale lock and clean it up.

2. **Primary lock is the decision point**: The primary lock acts as the atomic "commit/abort" decision. If the primary lock is replaced by a write record, the transaction committed.

3. **Snapshot isolation, not serializable**: Percolator provides SI (write-write conflict detection but allows read-write skew). This is weaker than serializable but sufficient for many applications.

---

## 5. Google Spanner TrueTime

### 5.1 External Consistency

Spanner (Corbett et al., 2012) provides **external consistency** (also called strict serializability): if transaction T1 commits before T2 starts, then T1's commit timestamp is less than T2's commit timestamp.

This is stronger than linearizability applied to transactions. It means the transaction order matches real-time order as observed by any external observer.

### 5.2 The Clock Problem

Achieving external consistency requires accurate timestamps. But **clocks are never perfectly synchronized**. If Node A's clock is 5ms ahead of Node B's clock, a transaction on B could get a lower timestamp than an earlier transaction on A.

```
Real time:     T1 commits at t=100ms    T2 starts at t=102ms
Node A clock:  T1 gets timestamp 105    (5ms ahead)
Node B clock:  T2 gets timestamp 100    (2ms behind)

Result: T2.timestamp (100) < T1.timestamp (105)
But T1 happened before T2! External consistency violated.
```

### 5.3 TrueTime API

Spanner's breakthrough: instead of pretending clocks are accurate, **TrueTime explicitly exposes clock uncertainty**.

```python
from dataclasses import dataclass


@dataclass
class TTInterval:
    """A time interval [earliest, latest] representing clock uncertainty."""
    earliest: float  # guaranteed to be ≤ true time
    latest: float    # guaranteed to be ≥ true time

    @property
    def uncertainty(self) -> float:
        """Half the interval width (epsilon)."""
        return (self.latest - self.earliest) / 2.0

    def __str__(self):
        return f"[{self.earliest:.3f}, {self.latest:.3f}] (±{self.uncertainty:.3f})"


class TrueTime:
    """Simulated TrueTime API.

    In Google's production system:
    - GPS receivers and atomic clocks provide time references
    - Typical uncertainty: 1-7ms (average ~4ms)
    - Uncertainty increases between clock synchronizations
    """

    def __init__(self, true_time_func, uncertainty_ms=4.0):
        self.true_time_func = true_time_func
        self.uncertainty = uncertainty_ms / 1000.0  # convert to seconds

    def now(self) -> TTInterval:
        """Return the current time as an interval.

        Guarantee: true time is within [earliest, latest].
        """
        true_time = self.true_time_func()
        return TTInterval(
            earliest=true_time - self.uncertainty,
            latest=true_time + self.uncertainty
        )

    def after(self, t: float) -> bool:
        """True if t has definitely passed."""
        return self.now().earliest > t

    def before(self, t: float) -> bool:
        """True if t has definitely not arrived."""
        return self.now().latest < t
```

### 5.4 Commit-Wait Protocol

Spanner's commit-wait protocol uses TrueTime to ensure external consistency:

```python
class SpannerCommitWait:
    """Spanner's commit-wait protocol for external consistency."""

    def __init__(self, truetime: TrueTime):
        self.tt = truetime

    def commit_transaction(self, txn):
        """Commit a transaction with external consistency guarantee.

        Steps:
        1. Acquire locks (Paxos-based distributed locking)
        2. Choose commit timestamp s = TT.now().latest
        3. Wait until TT.after(s) — the "commit wait"
        4. Release locks and make writes visible at timestamp s
        """
        # Step 1: Locks already acquired by the 2PC prepare phase

        # Step 2: Choose commit timestamp
        s = self.tt.now().latest

        # Step 3: Commit wait — block until we're sure s has passed
        # This ensures no future transaction can get a timestamp ≤ s
        while not self.tt.after(s):
            pass  # busy wait (in practice, sleep for short intervals)

        # Step 4: Apply commit at timestamp s
        txn.commit_at(s)

        return s
```

**Why commit-wait works**:

Let T1 commit with timestamp `s1 = TT.now().latest` at real time `t_abs`.
After waiting, we know `t_abs + wait_time > s1` (TrueTime guarantee).
Any transaction T2 that starts after T1 completes will call `TT.now()` at some `t2 > t_abs + wait_time`.
Therefore `TT.now().latest` for T2 will be `> s1`, giving T2 a higher commit timestamp.

**Cost of commit-wait**: The wait time equals twice the clock uncertainty (~8ms average). This is Spanner's latency tax for external consistency.

### 5.5 GPS and Atomic Clock Synchronization

```
Spanner Timeserver Architecture:

┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Timeserver 1│   │ Timeserver 2│   │ Timeserver 3│
│ GPS receiver│   │ Atomic clock│   │ GPS receiver│
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └─────────┬───────┘─────────┬───────┘
                 │                 │
          ┌──────▼──────┐   ┌──────▼──────┐
          │ Spanserver  │   │ Spanserver  │
          │ (polls all  │   │ (polls all  │
          │ timeservers)│   │ timeservers)│
          └─────────────┘   └─────────────┘

Each spanserver:
1. Polls multiple timeservers
2. Discards outliers (Marzullo's algorithm)
3. Computes uncertainty bounds
4. Typical uncertainty: 1-7ms

GPS provides absolute time (±1μs).
Atomic clocks provide stable frequency (drift ~10⁻¹² s/s).
Using both provides robustness: GPS failure → atomic clock drifts slowly.
```

---

## 6. Calvin (Thomson et al., 2012)

### 6.1 The Deterministic Database Idea

Calvin takes a radically different approach: **if all replicas process the same transactions in the same order, they will arrive at the same state, with no coordination needed during execution**.

This eliminates 2PC entirely — the only coordination is in the **sequencing layer**, which determines the global transaction order.

### 6.2 Architecture

```
┌─────────────────────────────────────────────────┐
│                 Sequencing Layer                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ Sequencer │  │ Sequencer │  │ Sequencer │   │
│  │ (replica) │  │ (replica) │  │ (replica) │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │               │               │        │
│        └───────┬───────┘───────┬───────┘        │
│                │               │                 │
│                ▼               ▼                 │
│  ┌─────────────────────────────────────────┐    │
│  │          Global Transaction Log          │    │
│  │  [T1, T3, T5, T2, T7, T4, T6, T8, ...]  │    │
│  └────────────────────┬────────────────────┘    │
│                       │                         │
│                       ▼                         │
│  ┌──────────────────────────────────────────┐   │
│  │            Scheduling Layer               │   │
│  │   Determines which locks each txn needs   │   │
│  └──────────────────────┬───────────────────┘   │
│                         │                       │
│                         ▼                       │
│  ┌──────────────────────────────────────────┐   │
│  │            Execution Layer                │   │
│  │   Executes transactions deterministically │   │
│  │   No coordination needed!                 │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 6.3 How It Works

```python
class CalvinSequencer:
    """Calvin sequencer that batches and orders transactions."""

    BATCH_INTERVAL_MS = 10  # batch transactions every 10ms

    def __init__(self, sequencer_id, replica_ids):
        self.id = sequencer_id
        self.replica_ids = replica_ids
        self.batch: List[dict] = []
        self.global_sequence: List[dict] = []
        self.epoch = 0

    def receive_transaction(self, txn: dict):
        """Add transaction to current batch."""
        self.batch.append(txn)

    def flush_batch(self):
        """End of epoch: sequence the batch and replicate.

        All sequencer replicas must agree on the same batch ordering.
        This is done via Paxos among sequencers.
        """
        self.epoch += 1
        batch_copy = self.batch[:]
        self.batch = []

        # Replicate batch ordering via Paxos
        # All replicas get the same ordered batch
        self.global_sequence.extend(batch_copy)

        return batch_copy


class CalvinScheduler:
    """Calvin scheduler that determines lock ordering."""

    def __init__(self):
        self.lock_queue: Dict[str, List[str]] = {}  # key → [txn_ids]

    def schedule(self, batch: List[dict]):
        """Analyze read/write sets and create a deterministic lock schedule.

        Because all replicas see the same batch in the same order,
        they all create the same lock schedule.
        """
        for txn in batch:
            for key in txn.get('write_set', []):
                if key not in self.lock_queue:
                    self.lock_queue[key] = []
                self.lock_queue[key].append(txn['id'])

            for key in txn.get('read_set', []):
                if key not in self.lock_queue:
                    self.lock_queue[key] = []
                self.lock_queue[key].append(txn['id'])

        return self.lock_queue


class CalvinExecutor:
    """Calvin executor that runs transactions deterministically."""

    def __init__(self):
        self.state: Dict[str, int] = {}

    def execute(self, txn: dict):
        """Execute a transaction. Because input is deterministic,
        output is deterministic — no coordination needed.

        If a transaction accesses data on a remote partition,
        the remote partition's executor sends the data directly
        (no 2PC — both sides know the transaction will commit).
        """
        for key, value in txn.get('writes', {}).items():
            self.state[key] = value

        return True  # always commits (decided at sequencing time)
```

### 6.4 Calvin Tradeoffs

| Advantage | Disadvantage |
|-----------|-------------|
| No 2PC → lower latency for multi-partition txns | Must know read/write sets before execution |
| High throughput (no coordination during execution) | Sequencer is a potential bottleneck |
| Active replication (all replicas execute all txns) | Cannot handle dependent transactions (result of T1 determines T2's read set) |
| Simple recovery (replay the sequence log) | Batching adds latency (10ms epochs) |

### 6.5 Real-World Impact

Calvin's ideas influenced several production systems:

- **FaunaDB** (now Fauna): commercially implemented Calvin
- **BOHM** (academic): extends Calvin with better support for dependent transactions
- **CockroachDB**: uses some Calvin-inspired ideas for its transaction pipeline

---

## 7. Saga Pattern

### 7.1 Motivation

Traditional distributed transactions (2PC) hold locks for the entire duration of the transaction. For **long-lived transactions** (seconds to hours), this is unacceptable — it blocks other transactions for too long.

The **Saga pattern** (Garcia-Molina and Salem, 1987) decomposes a long-lived transaction into a sequence of **sub-transactions**, each of which commits independently. If a sub-transaction fails, previously committed sub-transactions are compensated (undone) by **compensating transactions**.

### 7.2 Structure

```
Forward path (happy case):
  T1 → T2 → T3 → T4 → T5 → Done ✓

Compensation path (T3 fails):
  T1 → T2 → T3(FAIL) → C2 → C1 → Aborted ✗

Where:
  Ti = sub-transaction i
  Ci = compensating transaction for Ti (semantic undo)
```

### 7.3 Key Concept: Compensating Transactions

Compensating transactions are the **semantic inverse** of the original transaction. They are not rollbacks — they are new forward transactions that reverse the effect.

```python
from abc import ABC, abstractmethod
from typing import Callable, Optional


class SagaStep:
    """A single step in a saga with its compensating action."""

    def __init__(self, name: str, action: Callable, compensation: Callable):
        self.name = name
        self.action = action           # the forward transaction
        self.compensation = compensation  # the compensating transaction

    def execute(self, context: dict) -> bool:
        """Execute the forward action."""
        try:
            return self.action(context)
        except Exception as e:
            print(f"Step '{self.name}' failed: {e}")
            return False

    def compensate(self, context: dict):
        """Execute the compensating transaction."""
        try:
            self.compensation(context)
            print(f"Compensated step '{self.name}'")
        except Exception as e:
            print(f"CRITICAL: Compensation for '{self.name}' failed: {e}")
            # In production, this would trigger alerts and manual intervention
```

### 7.4 Saga Execution Engine

```python
class SagaOrchestrator:
    """Orchestration-based saga execution engine.

    The orchestrator manages the saga's state machine,
    executing steps sequentially and compensating on failure.
    """

    def __init__(self, name: str):
        self.name = name
        self.steps: List[SagaStep] = []
        self.completed_steps: List[SagaStep] = []

    def add_step(self, step: SagaStep):
        """Add a step to the saga."""
        self.steps.append(step)

    def execute(self, context: dict) -> bool:
        """Execute the saga. Returns True if all steps succeed."""
        print(f"\nStarting saga: {self.name}")

        for step in self.steps:
            print(f"  Executing: {step.name}")
            success = step.execute(context)

            if success:
                self.completed_steps.append(step)
            else:
                print(f"  Step '{step.name}' failed. Starting compensation...")
                self._compensate(context)
                return False

        print(f"Saga '{self.name}' completed successfully")
        return True

    def _compensate(self, context: dict):
        """Compensate all completed steps in reverse order."""
        for step in reversed(self.completed_steps):
            step.compensate(context)
        self.completed_steps.clear()


# ─── Example: Travel Booking Saga ───

def book_flight(ctx):
    """Reserve a flight."""
    print(f"    Reserved flight {ctx['flight_id']}")
    ctx['flight_reserved'] = True
    return True

def cancel_flight(ctx):
    """Cancel flight reservation."""
    print(f"    Cancelled flight {ctx['flight_id']}")
    ctx['flight_reserved'] = False

def book_hotel(ctx):
    """Reserve a hotel."""
    print(f"    Reserved hotel {ctx['hotel_id']}")
    ctx['hotel_reserved'] = True
    return True

def cancel_hotel(ctx):
    """Cancel hotel reservation."""
    print(f"    Cancelled hotel {ctx['hotel_id']}")
    ctx['hotel_reserved'] = False

def charge_payment(ctx):
    """Charge the customer's credit card."""
    amount = ctx['total_amount']
    if amount > 10000:
        raise ValueError(f"Amount ${amount} exceeds limit")
    print(f"    Charged ${amount} to card {ctx['card_last4']}")
    ctx['payment_charged'] = True
    return True

def refund_payment(ctx):
    """Refund the customer's credit card."""
    print(f"    Refunded ${ctx['total_amount']} to card {ctx['card_last4']}")
    ctx['payment_charged'] = False

def send_confirmation(ctx):
    """Send confirmation email."""
    print(f"    Sent confirmation to {ctx['email']}")
    ctx['confirmation_sent'] = True
    return True

def send_cancellation(ctx):
    """Send cancellation email."""
    print(f"    Sent cancellation to {ctx['email']}")
    ctx['confirmation_sent'] = False


def demo_sagas():
    """Demonstrate the Saga pattern with travel booking."""

    # Scenario 1: Successful booking
    print("=" * 60)
    print("Scenario 1: Successful travel booking")
    print("=" * 60)

    saga = SagaOrchestrator("Travel Booking")
    saga.add_step(SagaStep("Book Flight", book_flight, cancel_flight))
    saga.add_step(SagaStep("Book Hotel", book_hotel, cancel_hotel))
    saga.add_step(SagaStep("Charge Payment", charge_payment, refund_payment))
    saga.add_step(SagaStep("Send Confirmation", send_confirmation, send_cancellation))

    context = {
        'flight_id': 'UA-123',
        'hotel_id': 'HILTON-456',
        'total_amount': 1500,
        'card_last4': '4242',
        'email': 'user@example.com'
    }

    result = saga.execute(context)
    print(f"Result: {'SUCCESS' if result else 'COMPENSATED'}")
    print(f"Context: { {k:v for k,v in context.items() if 'reserved' in k or 'charged' in k} }")

    # Scenario 2: Payment fails (amount too high)
    print(f"\n{'=' * 60}")
    print("Scenario 2: Payment fails → compensate hotel and flight")
    print("=" * 60)

    saga2 = SagaOrchestrator("Travel Booking (too expensive)")
    saga2.add_step(SagaStep("Book Flight", book_flight, cancel_flight))
    saga2.add_step(SagaStep("Book Hotel", book_hotel, cancel_hotel))
    saga2.add_step(SagaStep("Charge Payment", charge_payment, refund_payment))
    saga2.add_step(SagaStep("Send Confirmation", send_confirmation, send_cancellation))

    context2 = {
        'flight_id': 'BA-789',
        'hotel_id': 'MARRIOTT-012',
        'total_amount': 15000,  # exceeds limit!
        'card_last4': '1234',
        'email': 'user@example.com'
    }

    result2 = saga2.execute(context2)
    print(f"Result: {'SUCCESS' if result2 else 'COMPENSATED'}")
    print(f"Context: { {k:v for k,v in context2.items() if 'reserved' in k or 'charged' in k} }")


if __name__ == "__main__":
    demo_sagas()
```

### 7.5 Choreography vs Orchestration

Two approaches to implementing sagas:

**Orchestration** (centralized): A saga orchestrator (coordinator) tells each participant what to do. The orchestrator maintains the saga's state machine.

```
Orchestrator → Flight Service: "Book flight"
Orchestrator ← Flight Service: "Flight booked"
Orchestrator → Hotel Service: "Book hotel"
Orchestrator ← Hotel Service: "Hotel booked"
Orchestrator → Payment Service: "Charge payment"
Orchestrator ← Payment Service: "Payment failed"
Orchestrator → Hotel Service: "Cancel hotel"
Orchestrator → Flight Service: "Cancel flight"
```

**Choreography** (decentralized): Each service publishes events and listens for events from other services. No central coordinator.

```
Flight Service: books flight → publishes "FlightBooked" event
Hotel Service: hears "FlightBooked" → books hotel → publishes "HotelBooked"
Payment Service: hears "HotelBooked" → charges payment
  If fails: publishes "PaymentFailed"
Hotel Service: hears "PaymentFailed" → cancels hotel → publishes "HotelCancelled"
Flight Service: hears "HotelCancelled" → cancels flight
```

| Aspect | Orchestration | Choreography |
|--------|--------------|--------------|
| Complexity | Centralized logic, easy to understand | Distributed logic, harder to trace |
| Coupling | Services depend on orchestrator | Services depend on events |
| Single point of failure | Orchestrator | None (but event bus is critical) |
| Observability | Easy (orchestrator has full state) | Hard (state is distributed) |
| Scalability | Orchestrator can be bottleneck | Scales independently |
| Testing | Easy to test orchestrator | Requires integration testing |

### 7.6 When to Use Sagas vs 2PC

```
Decision tree:

Transaction duration?
├── Short (< 1 second) → 2PC is acceptable
│   ├── Need serializability? → 2PC (Spanner-style)
│   └── Snapshot isolation OK? → Percolator-style
│
└── Long (seconds to hours) → Saga required
    ├── Simple sequence? → Choreography
    └── Complex workflow with branching? → Orchestration
```

| Property | 2PC | Saga |
|----------|-----|------|
| Isolation | Full (serializable) | Reduced (ACD, not ACID) |
| Locking | Held for transaction duration | Released after each step |
| Latency | Bounded by slowest participant | Sum of all steps |
| Complexity | Protocol complexity | Business logic complexity |
| Rollback | True rollback (undo) | Compensating transactions (semantic undo) |
| Partial visibility | No (all or nothing) | Yes (intermediate states visible) |
| Use cases | Databases, financial transactions | Microservices, booking systems, order processing |

### 7.7 Saga Guarantees: ACD (Not ACID)

Sagas provide:
- **A**tomicity: Via compensating transactions (eventual atomicity)
- **C**onsistency: Maintained by each local transaction
- **D**urability: Each committed sub-transaction is durable

Sagas do NOT provide:
- **I**solation: Intermediate states are visible to other transactions. This requires careful design to handle:
  - **Dirty reads**: Another transaction reads a value that will be compensated
  - **Lost updates**: Two sagas modify the same data without coordination
  - **Countermeasures**: Semantic locks, commutative updates, reread values

---

## 8. Protocol Comparison Summary

| Protocol | Year | Isolation | Blocking | Coordinator | Latency | Use Case |
|----------|------|-----------|----------|-------------|---------|----------|
| 2PC | 1978 | Serializable | Yes | Required | 2 RTT | Traditional databases |
| 3PC | 1981 | Serializable | No* | Required | 3 RTT | Theory only |
| Paxos Commit | 2006 | Serializable | No | Paxos group | 2-3 RTT | Spanner (internal) |
| Percolator | 2010 | Snapshot | No | Decentralized | 2 RTT + TSO | Large-scale KV stores |
| Spanner | 2012 | External consistency | No | Paxos group | 2 RTT + wait | Global databases |
| Calvin | 2012 | Serializable | No | Sequencer | 1 RTT + batch | Deterministic DBs |
| Saga | 1987 | Eventual (ACD) | No | Optional | Sum of steps | Microservices |

\* 3PC is non-blocking only without network partitions

---

## 9. Practical Considerations

### 9.1 Choosing a Transaction Protocol

```
Requirements Analysis:

1. Consistency requirements
   - Serializable → 2PC, Spanner, Calvin
   - Snapshot isolation → Percolator
   - Eventual consistency → Saga

2. Latency budget
   - < 10ms → Single-region 2PC or Calvin
   - < 100ms → Spanner (commit-wait ~8ms)
   - > 100ms → Saga (each step has its own latency)

3. Scale
   - < 10 nodes → 2PC works fine
   - 10-1000 nodes → Percolator or Spanner
   - > 1000 nodes → Calvin or Saga

4. Transaction duration
   - < 100ms → Any protocol
   - 100ms - 1s → 2PC or Percolator
   - > 1s → Saga (lock duration too long for 2PC)

5. Failure model
   - Coordinator can be replicated → 2PC + Paxos
   - No single point of failure → Percolator, Calvin
   - Network partitions expected → Saga (always available)
```

### 9.2 Transaction Processing Pipeline

A modern distributed database typically combines multiple protocols:

```
CockroachDB Transaction Pipeline:

1. Client sends BEGIN
2. SQL parser + optimizer → execution plan
3. For each statement:
   a. KV reads at snapshot timestamp (MVCC)
   b. Buffer writes locally
4. Client sends COMMIT:
   a. Parallel Raft replication of intent writes
   b. Transaction record (commit/abort decision) via Raft
   c. If committed: resolve intents asynchronously
   d. Return success to client

Combines: MVCC + Raft + Parallel Commit (async intent resolution)
```

### 9.3 Idempotency

All distributed transaction participants must handle **duplicate messages** (network retries):

```python
class IdempotentParticipant:
    """Participant that safely handles duplicate messages."""

    def __init__(self):
        self.processed_txns: Dict[str, str] = {}  # txn_id → result

    def prepare(self, txn_id, writes):
        """Idempotent prepare: same input always gives same output."""
        if txn_id in self.processed_txns:
            return self.processed_txns[txn_id]

        result = self._do_prepare(writes)
        self.processed_txns[txn_id] = result
        return result

    def commit(self, txn_id):
        """Idempotent commit: safe to call multiple times."""
        if txn_id in self.committed:
            return  # already committed, no-op
        self._do_commit(txn_id)
```

---

## 10. Summary

Distributed transactions solve the problem of atomic operations across multiple nodes, but each approach makes different tradeoffs between consistency, availability, latency, and complexity.

Two-Phase Commit provides the strongest guarantees (serializability) but blocks when the coordinator fails. Three-Phase Commit eliminates blocking at the cost of safety under network partitions, making it impractical for real systems. These fundamental limitations motivated the development of alternative approaches.

Percolator decentralizes coordination using optimistic concurrency and a timestamp oracle, providing snapshot isolation without a dedicated coordinator. Spanner achieves the strongest possible guarantee — external consistency — by using TrueTime's explicit clock uncertainty bounds and a commit-wait protocol. Calvin eliminates coordination during execution entirely by ensuring all replicas process the same deterministic transaction order.

The Saga pattern takes a fundamentally different approach: it sacrifices isolation to avoid holding locks across long-lived transactions, using compensating transactions to maintain eventual consistency. This pattern has become the dominant approach for distributed transactions in microservice architectures.

The choice between these protocols depends on your consistency requirements, latency budget, transaction duration, and operational complexity tolerance. Understanding the tradeoffs enables informed architectural decisions for distributed systems.

---

[Next: Replication Strategies](./09_Replication_Strategies.md)
