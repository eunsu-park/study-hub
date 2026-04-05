# Lesson 8: 분산 트랜잭션과 Atomic Commit

[Overview](./00_Overview.md) | [Previous: Byzantine Fault Tolerance](./07_Byzantine_Fault_Tolerance.md) | [Next: Replication Strategies](./09_Replication_Strategies.md)

---

## 학습 목표

- 분산 트랜잭션이 로컬 트랜잭션보다 근본적으로 어려운 이유를 분석한다
- Write-ahead logging을 사용하는 Two-Phase Commit (2PC)을 구현하고 blocking 문제를 이해한다
- 2PC와 Three-Phase Commit (3PC)을 비교하고 3PC가 네트워크 파티션에서 실패하는 이유를 설명한다
- 현대적 분산 트랜잭션 프로토콜인 Percolator, Spanner TrueTime, Calvin을 기술한다
- 장기 실행 분산 워크플로우를 위한 Saga 패턴을 사용하여 보상 트랜잭션을 설계한다

---

## 1. 분산 트랜잭션이 어려운 이유

### 1.1 근본적인 도전 과제

단일 데이터베이스에서의 로컬 트랜잭션은 단순한 계약을 가진다: 모든 연산이 성공하거나(commit) 아무것도 적용되지 않는다(abort). 데이터베이스는 write-ahead log (WAL)와 복구 관리자를 사용하여 이를 보장한다.

분산 시스템에서 트랜잭션은 여러 노드에 걸쳐 있다. 문제는 **각 노드가 독립적으로 실패하거나 다른 노드로부터 파티션될 수 있다**는 것이다. 다음을 고려해 보자:

```
Transaction: Transfer $100 from Account A (Node 1) to Account B (Node 2)

  Node 1: Debit $100 from A   ← succeeds
  --- network partition ---
  Node 2: Credit $100 to B    ← never receives the message

Result: $100 vanishes. Neither atomic commit nor rollback occurred.
```

### 1.2 부분 장애

로컬 데이터베이스와 달리, 분산 트랜잭션은 다음을 처리해야 한다:

| 장애 모드 | 설명 | 예시 |
|-------------|-------------|---------|
| 노드 충돌 | participant가 투표 전후에 충돌한다 | Coordinator가 Phase 2에서 죽는다 |
| 네트워크 파티션 | participant들이 통신할 수 없다 | 데이터센터 간 split-brain |
| 메시지 손실 | 개별 메시지가 유실된다 | 투표 메시지가 도착하지 않는다 |
| 메시지 지연 | 메시지가 임의로 늦게 도착한다 | commit 메시지가 몇 분 지연된다 |
| 부분 완료 | 일부 participant는 commit하고 나머지는 안 한다 | 핵심 문제 |

### 1.3 Atomic Commit 문제

공식적으로, **atomic commit** 문제는 다음을 요구한다:

1. **합의(Agreement)**: 결정을 내리는 모든 participant는 동일한 값(commit 또는 abort)을 결정해야 한다
2. **유효성(Validity)**: 모든 participant가 "yes"로 투표하고 장애가 없으면, 결정은 "commit"이다
3. **종료(Termination)**: 결함이 없는 모든 participant는 결국 결정을 내린다
4. **중단 유효성(Abort validity)**: 어떤 participant라도 "abort"로 투표하면, 결정은 "abort"여야 한다

이것은 합의(consensus) 문제와 관련이 있지만 구별된다: atomic commit에서는 **어떤 participant든 일방적으로 abort를 강제할 수 있는** 반면, consensus에서는 결정이 다수를 반영해야 한다.

---

## 2. Two-Phase Commit (2PC)

### 2.1 프로토콜 개요

Two-Phase Commit (Gray, 1978)은 atomic commit에 대한 고전적 해결책이다. 프로토콜을 주도하는 지정된 **coordinator**를 사용한다:

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

### 2.2 상세 메시지 흐름

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

### 2.3 구현

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

### 2.4 Blocking 문제

2PC의 근본적 약점: **coordinator가 일부 participant에게 COMMIT을 보낸 후 나머지에게 보내기 전에 충돌하면, commit을 받지 못한 participant들은 막히게 된다** — coordinator가 이미 commit했을 수 있으므로 abort할 수 없고, COMMIT 메시지를 받지 못했으므로 commit할 수도 없다. 이들은 **불확실(in doubt)** 상태에 있다.

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

blocking 구간은 coordinator가 WAL에 COMMIT을 기록한 시점부터 모든 participant가 COMMIT 메시지를 수신하는 시점까지의 시간이다. 이 구간 동안:

- YES로 투표한 participant들은 잠금을 보유하며 해제할 수 없다
- 잠긴 리소스와 충돌하는 모든 연산에 대해 시스템이 사용 불가능하다
- coordinator 없이는 어떤 타임아웃 로직으로도 상황을 안전하게 해결할 수 없다

### 2.5 Presumed Abort 최적화

**Presumed abort** 최적화는 WAL 기록과 메시지 복잡성을 줄인다:

1. Coordinator는 ABORT 결정을 로깅하지 않는다 (COMMIT만 로깅)
2. Participant가 알 수 없는 트랜잭션에 대해 문의하면, coordinator는 "abort"로 응답한다
3. YES로 투표한 participant는 YES 투표에 대한 ACK을 기다릴 필요가 없다

이것이 동작하는 이유는 abort가 항상 안전하기 때문이다 — participant가 임시 쓰기를 취소하기만 하면 된다. 이 최적화는 중단되는 트랜잭션의 일반적인 경우에 대해 WAL 기록 1회와 메시지 1회를 절약한다.

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

### 2.6 성능 특성

| 지표 | 값 | 비고 |
|--------|-------|-------|
| WAL 기록 (coordinator) | 2 (begin + commit/abort) | Presumed abort 사용 시: commit에 대해 1 |
| WAL 기록 (participant) | 2 (prepared + done) | participant당 |
| 네트워크 왕복 | 2 | Prepare + Commit |
| 메시지 수 | 4N | N prepare + N vote + N commit + N ack |
| 지연 시간 | 2 × RTT + 2 × fsync | fsync 지연에 의해 지배됨 |

---

## 3. Three-Phase Commit (3PC)

### 3.1 동기: Blocking 제거

3PC (Skeen, 1981)는 모든 participant가 commit 전에 결정을 알도록 보장하는 **pre-commit** 단계를 추가하여 2PC의 blocking 문제를 해결한다:

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

### 3.2 3PC가 Blocking을 회피하는 방법

pre-commit 단계는 핵심 불변량을 보장한다:

> **어떤** participant가 commit했다면, **모든** participant가 pre-commit 메시지를 수신한 것이다.

이는 coordinator가 충돌해도 나머지 participant들이 안전하게 결정할 수 있음을 의미한다:
- 어떤 participant가 pre-commit을 수신했다면 → commit (coordinator가 commit을 결정했기 때문)
- 어떤 participant도 pre-commit을 수신하지 않았다면 → abort (아무도 commit하지 않았으므로 안전)

```
2PC blocking scenario (coordinator crashes after commit to A):
  A: committed, B: uncertain → B is STUCK

3PC equivalent scenario:
  A: received pre-commit, B: received pre-commit → BOTH can commit
  OR
  A: received pre-commit, B: did NOT receive pre-commit → IMPOSSIBLE
  (because coordinator sends pre-commit to all before proceeding)
```

### 3.3 3PC가 네트워크 파티션에서 동작하지 않는 이유

3PC의 non-blocking 특성은 중요한 가정에 의존한다: **네트워크 파티션이 없을 것**. 파티션 하에서 프로토콜은 무너진다:

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

문제: 서로 다른 파티션에 있는 participant들이 동일한 pre-commit 메시지 집합을 볼 수 없기 때문에 모순되는 결정을 내릴 수 있다.

### 3.4 비교: 2PC vs 3PC

| 속성 | 2PC | 3PC |
|----------|-----|-----|
| 단계 수 | 2 | 3 |
| 메시지 수 | 4N | 6N |
| WAL 기록 | 2 + 2N | 3 + 3N |
| Coordinator 충돌 시 blocking | YES | 아니오 (파티션 없을 때) |
| 네트워크 파티션에서 안전 | 예 (blocking됨) | NO (안전성 위반 가능) |
| 실제 사용 | 널리 사용됨 | 거의 없음 |
| 지연 시간 | 2 RTT | 3 RTT |

**핵심 요약**: 실제로 3PC는 거의 사용되지 않는다. 실제 시스템에서 네트워크 파티션은 불가피하며, 3PC의 안전성 위반은 2PC의 blocking보다 더 나쁘기 때문이다. 현대 시스템은 2PC의 blocking을 감수하거나 대안적 접근법(Paxos Commit, Percolator, Saga)을 사용한다.

### 3.5 Paxos Commit: 양쪽의 장점 모두

Lamport과 Gray (2006)는 **Paxos Commit**을 제안했다: coordinator를 Paxos 그룹으로 대체하는 것이다. 이를 통해 파티션에서도 non-blocking atomic commit을 달성한다:

```
Instead of one coordinator:
  - Run a Paxos instance to decide "commit or abort" for each participant's vote
  - Participants send their votes to the Paxos acceptors
  - The Paxos group reliably records the decision

Benefit: no single point of failure, no blocking
Cost: higher message complexity (O(N²) for Paxos among coordinators)
```

이것이 Google Spanner가 분산 트랜잭션에 사용하는 접근 방식이다.

---

## 4. Percolator (Google, 2010)

### 4.1 개요

Google Percolator (Peng and Dabek, 2010)는 BigTable 위에서 snapshot isolation을 제공하는 **낙관적 분산 트랜잭션**을 구현한다. 이것은 Google의 웹 인덱싱 파이프라인을 위해 구축되었으며, 수 페타바이트 규모의 테이블에 대한 업데이트를 처리한다.

주요 특성:
- **Snapshot isolation** (serializable 아님)
- **낙관적 동시성 제어** — 실행 중에는 잠금 없음; commit 시 충돌 감지
- **탈중앙화** — 중앙 coordinator 없음; 각 트랜잭션이 자체 commit을 관리
- 중앙 집중식 **timestamp oracle** (TSO)로부터의 **타임스탬프**

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

### 4.3 데이터 모델

Percolator는 사용자 컬럼당 세 개의 column family를 가진 BigTable에 데이터를 저장한다:

| Column Family | 용도 | 키 | 값 |
|--------------|---------|-----|-------|
| `data` | 실제 값 | `(row, col, start_ts)` | value |
| `lock` | 트랜잭션 잠금 | `(row, col, start_ts)` | `(primary_row, primary_col)` |
| `write` | commit 기록 | `(row, col, commit_ts)` | `start_ts` |

### 4.4 쓰기 경로 (Two-Phase Commit)

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

### 4.5 핵심 통찰

1. **Coordinator 충돌 문제 없음**: 각 트랜잭션은 자기 기술적(self-describing)이다. 트랜잭션이 primary를 prewrite한 후 commit하기 전에 충돌하면, 다른 트랜잭션이 오래된 잠금을 감지하고 정리할 수 있다.

2. **Primary lock이 결정 지점**: primary lock이 atomic "commit/abort" 결정 역할을 한다. primary lock이 write 레코드로 대체되면 트랜잭션이 commit된 것이다.

3. **Snapshot isolation, serializable 아님**: Percolator는 SI(write-write 충돌 감지는 하지만 read-write skew를 허용)를 제공한다. 이는 serializable보다 약하지만 많은 애플리케이션에 충분하다.

---

## 5. Google Spanner TrueTime

### 5.1 External Consistency

Spanner (Corbett et al., 2012)는 **external consistency** (strict serializability라고도 함)를 제공한다: 트랜잭션 T1이 T2가 시작되기 전에 commit하면, T1의 commit 타임스탬프가 T2의 commit 타임스탬프보다 작다.

이것은 트랜잭션에 적용된 linearizability보다 강하다. 트랜잭션 순서가 외부 관찰자가 관측한 실시간 순서와 일치함을 의미한다.

### 5.2 시계 문제

External consistency를 달성하려면 정확한 타임스탬프가 필요하다. 하지만 **시계는 절대로 완벽하게 동기화되지 않는다**. 만약 노드 A의 시계가 노드 B보다 5ms 빠르다면, B의 트랜잭션이 A의 더 이전 트랜잭션보다 낮은 타임스탬프를 받을 수 있다.

```
Real time:     T1 commits at t=100ms    T2 starts at t=102ms
Node A clock:  T1 gets timestamp 105    (5ms ahead)
Node B clock:  T2 gets timestamp 100    (2ms behind)

Result: T2.timestamp (100) < T1.timestamp (105)
But T1 happened before T2! External consistency violated.
```

### 5.3 TrueTime API

Spanner의 혁신: 시계가 정확한 척하는 대신, **TrueTime은 시계 불확실성을 명시적으로 노출한다**.

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

### 5.4 Commit-Wait 프로토콜

Spanner의 commit-wait 프로토콜은 TrueTime을 사용하여 external consistency를 보장한다:

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

**Commit-wait가 동작하는 이유**:

T1이 타임스탬프 `s1 = TT.now().latest`로 실시간 `t_abs`에 commit한다고 하자.
대기 후, `t_abs + wait_time > s1`임을 안다 (TrueTime 보장).
T1 완료 후에 시작하는 모든 트랜잭션 T2는 `t2 > t_abs + wait_time`인 어떤 시점에서 `TT.now()`를 호출할 것이다.
따라서 T2의 `TT.now().latest`는 `> s1`이 되어, T2에게 더 높은 commit 타임스탬프를 부여한다.

**Commit-wait의 비용**: 대기 시간은 시계 불확실성의 두 배(평균 ~8ms)이다. 이것이 Spanner가 external consistency를 위해 지불하는 지연 세금이다.

### 5.5 GPS와 원자 시계 동기화

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

### 6.1 결정론적 데이터베이스 아이디어

Calvin은 근본적으로 다른 접근 방식을 취한다: **모든 레플리카가 동일한 트랜잭션을 동일한 순서로 처리하면, 실행 중 조정 없이도 동일한 상태에 도달한다**.

이를 통해 2PC가 완전히 제거된다 — 유일한 조정은 전역 트랜잭션 순서를 결정하는 **sequencing 계층**에서 이루어진다.

### 6.2 아키텍처

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

### 6.3 동작 방식

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

### 6.4 Calvin의 트레이드오프

| 장점 | 단점 |
|-----------|-------------|
| 2PC 없음 → 다중 파티션 트랜잭션의 낮은 지연 시간 | 실행 전에 read/write 집합을 알아야 함 |
| 높은 처리량 (실행 중 조정 없음) | Sequencer가 잠재적 병목이 될 수 있음 |
| 능동적 복제 (모든 레플리카가 모든 트랜잭션 실행) | 종속적 트랜잭션 처리 불가 (T1의 결과가 T2의 read 집합을 결정하는 경우) |
| 간단한 복구 (시퀀스 로그 재생) | 배치 처리가 지연 추가 (10ms epoch) |

### 6.5 실제 영향

Calvin의 아이디어는 여러 프로덕션 시스템에 영향을 미쳤다:

- **FaunaDB** (현재 Fauna): Calvin을 상업적으로 구현
- **BOHM** (학술): 종속적 트랜잭션에 대한 더 나은 지원으로 Calvin 확장
- **CockroachDB**: 트랜잭션 파이프라인에 일부 Calvin 영감의 아이디어 사용

---

## 7. Saga 패턴

### 7.1 동기

전통적인 분산 트랜잭션(2PC)은 트랜잭션 전체 기간 동안 잠금을 보유한다. **장기 실행 트랜잭션**(수 초에서 수 시간)의 경우, 이는 수용할 수 없다 — 다른 트랜잭션을 너무 오래 차단하기 때문이다.

**Saga 패턴** (Garcia-Molina and Salem, 1987)은 장기 실행 트랜잭션을 각각 독립적으로 commit하는 **하위 트랜잭션** 시퀀스로 분해한다. 하위 트랜잭션이 실패하면, 이전에 commit된 하위 트랜잭션들은 **보상 트랜잭션(compensating transaction)**에 의해 보상된다(취소된다).

### 7.2 구조

```
Forward path (happy case):
  T1 → T2 → T3 → T4 → T5 → Done ✓

Compensation path (T3 fails):
  T1 → T2 → T3(FAIL) → C2 → C1 → Aborted ✗

Where:
  Ti = sub-transaction i
  Ci = compensating transaction for Ti (semantic undo)
```

### 7.3 핵심 개념: 보상 트랜잭션

보상 트랜잭션은 원래 트랜잭션의 **의미적 역(semantic inverse)**이다. 롤백이 아니라, 효과를 되돌리는 새로운 정방향 트랜잭션이다.

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

### 7.4 Saga 실행 엔진

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

Saga를 구현하는 두 가지 접근 방식이 있다:

**Orchestration** (중앙 집중형): Saga orchestrator(coordinator)가 각 participant에게 무엇을 할지 지시한다. Orchestrator가 saga의 상태 머신을 유지한다.

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

**Choreography** (탈중앙형): 각 서비스가 이벤트를 발행하고 다른 서비스의 이벤트를 수신한다. 중앙 coordinator가 없다.

```
Flight Service: books flight → publishes "FlightBooked" event
Hotel Service: hears "FlightBooked" → books hotel → publishes "HotelBooked"
Payment Service: hears "HotelBooked" → charges payment
  If fails: publishes "PaymentFailed"
Hotel Service: hears "PaymentFailed" → cancels hotel → publishes "HotelCancelled"
Flight Service: hears "HotelCancelled" → cancels flight
```

| 측면 | Orchestration | Choreography |
|--------|--------------|--------------|
| 복잡성 | 중앙 집중 로직, 이해하기 쉬움 | 분산 로직, 추적하기 어려움 |
| 결합도 | 서비스가 orchestrator에 의존 | 서비스가 이벤트에 의존 |
| 단일 장애점 | Orchestrator | 없음 (단, 이벤트 버스가 중요) |
| 관찰 가능성 | 쉬움 (orchestrator가 전체 상태를 보유) | 어려움 (상태가 분산되어 있음) |
| 확장성 | Orchestrator가 병목이 될 수 있음 | 독립적으로 확장 |
| 테스트 | Orchestrator 테스트가 쉬움 | 통합 테스트 필요 |

### 7.6 Saga vs 2PC 사용 시기

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

| 속성 | 2PC | Saga |
|----------|-----|------|
| 격리성 | 완전함 (serializable) | 감소됨 (ACD, ACID 아님) |
| 잠금 | 트랜잭션 기간 동안 보유 | 각 단계 후 해제 |
| 지연 시간 | 가장 느린 participant에 의해 제한됨 | 모든 단계의 합 |
| 복잡성 | 프로토콜 복잡성 | 비즈니스 로직 복잡성 |
| 롤백 | 진정한 롤백 (undo) | 보상 트랜잭션 (의미적 undo) |
| 부분 가시성 | 없음 (전부 아니면 전무) | 있음 (중간 상태 노출) |
| 사용 사례 | 데이터베이스, 금융 트랜잭션 | 마이크로서비스, 예약 시스템, 주문 처리 |

### 7.7 Saga 보장: ACD (ACID 아님)

Saga가 제공하는 것:
- **A**tomicity: 보상 트랜잭션을 통해 (최종적 원자성)
- **C**onsistency: 각 로컬 트랜잭션에 의해 유지
- **D**urability: commit된 각 하위 트랜잭션은 지속적

Saga가 제공하지 않는 것:
- **I**solation: 중간 상태가 다른 트랜잭션에 노출된다. 다음을 처리하기 위한 신중한 설계가 필요하다:
  - **Dirty read**: 다른 트랜잭션이 보상될 값을 읽는다
  - **Lost update**: 두 saga가 조정 없이 같은 데이터를 수정한다
  - **대응 수단**: 의미적 잠금, 가환적 업데이트, 값 재읽기

---

## 8. 프로토콜 비교 요약

| 프로토콜 | 연도 | 격리성 | Blocking | Coordinator | 지연 시간 | 사용 사례 |
|----------|------|-----------|----------|-------------|---------|----------|
| 2PC | 1978 | Serializable | 예 | 필수 | 2 RTT | 전통적 데이터베이스 |
| 3PC | 1981 | Serializable | 아니오* | 필수 | 3 RTT | 이론만 |
| Paxos Commit | 2006 | Serializable | 아니오 | Paxos 그룹 | 2-3 RTT | Spanner (내부) |
| Percolator | 2010 | Snapshot | 아니오 | 탈중앙 | 2 RTT + TSO | 대규모 KV 저장소 |
| Spanner | 2012 | External consistency | 아니오 | Paxos 그룹 | 2 RTT + wait | 글로벌 데이터베이스 |
| Calvin | 2012 | Serializable | 아니오 | Sequencer | 1 RTT + batch | 결정론적 DB |
| Saga | 1987 | Eventual (ACD) | 아니오 | 선택적 | 단계의 합 | 마이크로서비스 |

\* 3PC는 네트워크 파티션이 없을 때만 non-blocking

---

## 9. 실용적 고려사항

### 9.1 트랜잭션 프로토콜 선택

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

### 9.2 트랜잭션 처리 파이프라인

현대 분산 데이터베이스는 일반적으로 여러 프로토콜을 결합한다:

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

### 9.3 멱등성

모든 분산 트랜잭션 participant는 **중복 메시지**(네트워크 재시도)를 처리해야 한다:

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

## 10. 요약

분산 트랜잭션은 여러 노드에 걸친 원자적 연산 문제를 해결하지만, 각 접근 방식은 일관성, 가용성, 지연 시간, 복잡성 사이에서 서로 다른 트레이드오프를 만든다.

Two-Phase Commit은 가장 강력한 보장(serializability)을 제공하지만 coordinator가 실패하면 blocking된다. Three-Phase Commit은 네트워크 파티션에서의 안전성 비용으로 blocking을 제거하여 실제 시스템에서는 비실용적이다. 이러한 근본적 한계가 대안적 접근법의 개발을 촉진했다.

Percolator는 낙관적 동시성과 timestamp oracle을 사용하여 전용 coordinator 없이 snapshot isolation을 제공하며 조정을 탈중앙화한다. Spanner는 TrueTime의 명시적 시계 불확실성 범위와 commit-wait 프로토콜을 사용하여 가능한 가장 강력한 보장인 external consistency를 달성한다. Calvin은 모든 레플리카가 동일한 결정론적 트랜잭션 순서를 처리하도록 보장하여 실행 중 조정을 완전히 제거한다.

Saga 패턴은 근본적으로 다른 접근 방식을 취한다: 장기 실행 트랜잭션에서 잠금을 보유하지 않기 위해 격리성을 희생하고, 보상 트랜잭션을 사용하여 최종적 일관성을 유지한다. 이 패턴은 마이크로서비스 아키텍처에서 분산 트랜잭션의 지배적인 접근 방식이 되었다.

이러한 프로토콜 간의 선택은 일관성 요구사항, 지연 시간 예산, 트랜잭션 기간, 운영 복잡성 허용 수준에 따라 달라진다. 트레이드오프를 이해하면 분산 시스템에 대한 정보에 기반한 아키텍처 결정을 내릴 수 있다.

---

[Next: Replication Strategies](./09_Replication_Strategies.md)
