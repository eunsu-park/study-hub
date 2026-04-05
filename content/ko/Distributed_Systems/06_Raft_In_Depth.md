# 레슨 6: Raft 심화

[개요](./00_Overview.md) | [이전: Paxos 계열](./05_Paxos_Family.md) | [다음: Byzantine Fault Tolerance](./07_Byzantine_Fault_Tolerance.md)

---

## 학습 목표

- Raft의 이해도 중심 설계 철학과 합의를 어떻게 분해하는지 이해한다
- Pre-vote 확장과 split-vote 해결을 포함한 leader 선출을 구현하고 분석한다
- 충돌 해결, 파이프라이닝, 커밋 진행을 포함한 로그 복제 메커니즘을 추적한다
- 스냅샷 및 InstallSnapshot RPC를 포함한 로그 압축 전략을 설명한다
- joint consensus와 단일 서버 접근법을 사용한 클러스터 멤버십 변경을 설계한다
- linearizable 읽기를 위한 읽기 최적화(ReadIndex, LeaseRead, follower 읽기)를 구현한다

---

## 1. 동기: Paxos를 넘어선 이해도

### 1.1 Raft 설계 철학

Diego Ongaro와 John Ousterhout는 2014년에 명시적인 목표를 가지고 Raft 합의 알고리즘을 발표했다: **이해도**. 사용자 연구에서 학생들이 Paxos보다 Raft를 상당히 더 빠르고 정확하게 학습했음을 보여주었다.

Raft는 두 가지 핵심 기법을 통해 이를 달성한다:

1. **분해**: Raft는 합의를 상대적으로 독립적인 세 가지 하위 문제로 분리한다:
   - Leader 선출
   - 로그 복제
   - 안전성 (leader가 모든 커밋된 항목을 가지고 있음을 보장)

2. **상태 공간 축소**: Raft는 불필요한 복잡성을 추가하는 구성을 허용하지 않는다. 예를 들어, 로그에 구멍이 있을 수 없고(Multi-Paxos와 달리), 완전한 로그를 가진 노드만 leader가 될 수 있다.

### 1.2 Raft vs Multi-Paxos

| 측면 | Multi-Paxos | Raft |
|--------|------------|------|
| 명세 | 불완전 (많은 공백) | 완전 (모든 경우 포함) |
| Leader 선출 | 구현 정의 | 프로토콜에 통합 |
| 로그 구멍 | 허용 (gap 채우기 필요) | 없음 (연속 로그) |
| Leader가 될 수 있는 노드 | 모든 노드 | 최신 로그를 가진 노드만 |
| 재구성 | 미명세 | Joint consensus / 단일 서버 |
| 구현체 | 다양, 종종 미묘하게 오류 | 구현체 간 일관성 |

### 1.3 핵심 Raft 속성

Raft는 다섯 가지 속성을 보장한다:

1. **Election Safety**: 주어진 term에서 최대 하나의 leader만 선출될 수 있다
2. **Leader Append-Only**: leader는 로그 항목을 덮어쓰거나 삭제하지 않고 추가만 한다
3. **Log Matching**: 두 로그가 동일한 인덱스와 term을 가진 항목을 포함하면, 해당 인덱스까지의 로그는 동일하다
4. **Leader Completeness**: 주어진 term에서 로그 항목이 커밋되면, 더 높은 term의 모든 leader 로그에 존재한다
5. **State Machine Safety**: 서버가 주어진 인덱스에서 로그 항목을 적용했다면, 다른 서버는 해당 인덱스에 다른 항목을 적용하지 않는다

---

## 2. Leader 선출 심화

### 2.1 선출 메커니즘

모든 Raft 노드는 세 가지 상태 중 하나에 있다:

```
                 times out,
                 starts election
    ┌───────────────────────────────┐
    │                               │
    ▼                               │
┌────────┐    receives votes    ┌───┴──────┐    discovers current
│Follower│◄─────────────────────│Candidate │    leader or new term
│        │    from new leader   │          │────────────────────┐
└───┬────┘                      └───┬──────┘                    │
    │                               │                           │
    │       discovers                │ receives majority         ▼
    │       higher term              │ of votes              ┌────────┐
    │◄───────────────────────────────┼───────────────────────│ Leader │
    │                               └──────────────────────▶│        │
    │                                                        └────────┘
    │                                                            │
    └────────────────────────────────────────────────────────────┘
                        discovers higher term
```

### 2.2 선출 타임아웃과 랜덤화

각 follower는 **선출 타임아웃** — `T`에서 `2T` 사이의 랜덤 기간(일반적으로 150-300ms)을 유지한다. leader로부터 heartbeat를 받지 못한 채 타임아웃이 만료되면, follower는 candidate가 된다.

랜덤화는 **split vote**를 피하는 데 핵심적이다: 모든 노드가 동일한 타임아웃을 사용하면, 모두 동시에 candidate가 되어 투표를 분할할 것이다.

```python
import random
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set


class State(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    term: int
    index: int
    command: str


@dataclass
class RaftNode:
    """A single Raft node with leader election support."""

    node_id: int
    peers: List[int]
    state: State = State.FOLLOWER

    # Persistent state
    current_term: int = 0
    voted_for: Optional[int] = None
    log: List[LogEntry] = field(default_factory=list)

    # Volatile state
    commit_index: int = 0
    last_applied: int = 0

    # Leader state
    next_index: Dict[int, int] = field(default_factory=dict)
    match_index: Dict[int, int] = field(default_factory=dict)

    # Election state
    votes_received: Set[int] = field(default_factory=set)
    election_timeout: float = 0.0

    # Timing
    HEARTBEAT_INTERVAL: float = 0.05   # 50ms
    MIN_ELECTION_TIMEOUT: float = 0.15  # 150ms
    MAX_ELECTION_TIMEOUT: float = 0.30  # 300ms

    def reset_election_timeout(self):
        """Randomize election timeout to prevent split votes."""
        self.election_timeout = random.uniform(
            self.MIN_ELECTION_TIMEOUT,
            self.MAX_ELECTION_TIMEOUT
        )

    def last_log_index(self) -> int:
        return self.log[-1].index if self.log else 0

    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0

    def start_election(self):
        """Transition to candidate and request votes."""
        self.state = State.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}  # vote for self
        self.reset_election_timeout()

        # Send RequestVote to all peers
        for peer_id in self.peers:
            self.send_request_vote(peer_id)

    def send_request_vote(self, peer_id: int):
        """Send RequestVote RPC (abstract — actual network not shown)."""
        # RequestVote(term, candidateId, lastLogIndex, lastLogTerm)
        pass

    def handle_request_vote(self, candidate_term, candidate_id,
                            last_log_index, last_log_term):
        """Process incoming RequestVote RPC.

        Returns (term, vote_granted).
        """
        # Step down if candidate has higher term
        if candidate_term > self.current_term:
            self.current_term = candidate_term
            self.state = State.FOLLOWER
            self.voted_for = None

        # Reject if candidate's term is stale
        if candidate_term < self.current_term:
            return self.current_term, False

        # Check if we already voted for someone else this term
        if self.voted_for is not None and self.voted_for != candidate_id:
            return self.current_term, False

        # Election restriction: candidate's log must be at least as
        # up-to-date as ours
        if not self._is_log_up_to_date(last_log_index, last_log_term):
            return self.current_term, False

        # Grant vote
        self.voted_for = candidate_id
        self.reset_election_timeout()
        return self.current_term, True

    def _is_log_up_to_date(self, last_index, last_term):
        """Check if candidate's log is at least as up-to-date as ours.

        Raft compares logs by:
        1. Higher last term wins
        2. If same last term, longer log wins
        """
        my_last_term = self.last_log_term()
        my_last_index = self.last_log_index()

        if last_term != my_last_term:
            return last_term > my_last_term
        return last_index >= my_last_index
```

### 2.3 Pre-Vote 확장

Raft의 알려진 문제: leader에서 분리된 노드가 반복적으로 타임아웃하고, term을 증가시키며, 선거를 시작한다. 클러스터에 다시 합류할 때, 높은 term 번호가 현재 leader를 퇴진하도록 강제하여 클러스터를 불필요하게 방해한다.

**Pre-Vote** 확장(Ongaro의 학위 논문, Section 9.6)은 예비 단계를 추가한다:

```
Pre-Vote Protocol:
1. Before starting a real election, a candidate sends PreVote messages
2. PreVote does NOT increment the term
3. Other nodes respond based on whether they would vote for the candidate
4. Only if the candidate receives a majority of PreVote responses does it
   proceed to a real election with term increment

Key check: a node only grants a PreVote if:
  - The candidate's log is up-to-date
  - The voter has NOT received a heartbeat from a leader recently
```

```python
class PreVoteRaftNode(RaftNode):
    """Raft node with Pre-Vote extension."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_votes_received: Set[int] = set()
        self.last_heartbeat_time: float = 0.0

    def start_pre_vote(self):
        """Phase 0: send PreVote without incrementing term."""
        self.pre_votes_received = {self.node_id}

        for peer_id in self.peers:
            self.send_pre_vote(peer_id)

    def send_pre_vote(self, peer_id):
        """Send PreVote RPC (does NOT increment term)."""
        # PreVote(term+1, candidateId, lastLogIndex, lastLogTerm)
        # Note: term+1 to simulate what would happen, without committing
        pass

    def handle_pre_vote(self, candidate_term, candidate_id,
                        last_log_index, last_log_term, current_time):
        """Process incoming PreVote RPC.

        Returns (term, pre_vote_granted).
        """
        # Reject if we recently heard from a leader (prevents unnecessary disruption)
        leader_lease_active = (
            current_time - self.last_heartbeat_time < self.MIN_ELECTION_TIMEOUT
        )
        if leader_lease_active:
            return self.current_term, False

        # Same log up-to-date check as RequestVote
        if candidate_term < self.current_term:
            return self.current_term, False

        if not self._is_log_up_to_date(last_log_index, last_log_term):
            return self.current_term, False

        return self.current_term, True

    def handle_pre_vote_response(self, granted):
        """Process PreVote response."""
        if granted:
            self.pre_votes_received.add(0)  # simplified
            majority = len(self.peers) // 2 + 1
            if len(self.pre_votes_received) >= majority:
                # Pre-vote succeeded; start real election
                self.start_election()

    def on_election_timeout(self, current_time):
        """Called when election timeout fires."""
        if self.state == State.LEADER:
            return

        # Use pre-vote instead of directly starting election
        self.start_pre_vote()
```

### 2.4 Term 혼동과 오래된 Leader

**오래된 leader**는 자신이 여전히 leader라고 믿지만 더 높은 term의 새 leader에 의해 대체된 노드이다. 이는 네트워크 파티션이 복구된 후에 발생할 수 있다.

Raft는 **term 검사**를 통해 이를 처리한다: 모든 RPC는 발신자의 term을 포함한다. 노드가 더 높은 term의 메시지를 받으면, 즉시 follower로 전환된다.

```python
def handle_rpc(self, sender_term, message):
    """Common term-checking logic for all RPCs."""
    if sender_term > self.current_term:
        # Sender is in a higher term; step down
        self.current_term = sender_term
        self.state = State.FOLLOWER
        self.voted_for = None

    if sender_term < self.current_term:
        # Sender is stale; reject
        return False

    return True  # proceed with message handling
```

### 2.5 Split Vote 해결

두 candidate가 동일한 수의 투표를 받으면(split vote), 어느 쪽도 과반수를 달성하지 못한다. 둘 다 타임아웃되고(랜덤 타임아웃으로 인해 서로 다른 시간에) 더 높은 term으로 새 선거를 시작한다.

```
Term 5 Election:
  Node 1 (candidate): gets votes from {1, 2}     → 2 votes (not majority of 5)
  Node 3 (candidate): gets votes from {3, 4}     → 2 votes (not majority of 5)
  Node 5: voted for Node 1                       → already voted

  Result: No leader elected in term 5

Term 6 Election:
  Node 3 times out first (random timeout)
  Node 3 (candidate): gets votes from {3, 4, 5, 1} → 4 votes → LEADER
```

반복적인 split vote의 확률은 타임아웃 범위에 따라 기하급수적으로 감소한다. 150-300ms 범위와 5개 노드에서, 장애 후 leader를 선출하는 예상 시간은 일반적으로 1초 미만이다.

---

## 3. 로그 복제

### 3.1 AppendEntries RPC

Leader는 `AppendEntries` RPC를 사용하여 로그 항목을 follower에 복제하며, 이는 heartbeat 메커니즘(빈 항목 목록으로)으로도 사용된다.

```python
@dataclass
class AppendEntriesRequest:
    term: int               # leader's term
    leader_id: int          # so followers can redirect clients
    prev_log_index: int     # index of log entry immediately preceding new entries
    prev_log_term: int      # term of prev_log_index entry
    entries: List[LogEntry] # log entries to store (empty for heartbeat)
    leader_commit: int      # leader's commit_index


@dataclass
class AppendEntriesResponse:
    term: int               # current term, for leader to update itself
    success: bool           # true if follower contained matching prev entry
    match_index: int        # highest log index known to match (optimization)
```

### 3.2 일관성 검사

`prev_log_index`와 `prev_log_term` 필드는 **Log Matching Property**를 구현한다. 새 항목을 추가하기 전에, follower는 `prev_log_index`에 term이 `prev_log_term`인 일치하는 항목이 있는지 확인한다:

```python
def handle_append_entries(self, req: AppendEntriesRequest):
    """Process AppendEntries from leader."""
    # Term check
    if req.term < self.current_term:
        return AppendEntriesResponse(self.current_term, False, 0)

    self.current_term = req.term
    self.state = State.FOLLOWER
    self.reset_election_timeout()

    # Consistency check: do we have the previous entry?
    if req.prev_log_index > 0:
        if req.prev_log_index > len(self.log):
            # We don't have enough entries
            return AppendEntriesResponse(
                self.current_term, False, len(self.log)
            )

        prev_entry = self.log[req.prev_log_index - 1]
        if prev_entry.term != req.prev_log_term:
            # Term mismatch: delete this entry and all that follow
            self.log = self.log[:req.prev_log_index - 1]
            return AppendEntriesResponse(
                self.current_term, False, len(self.log)
            )

    # Append new entries (overwriting any conflicting entries)
    for entry in req.entries:
        idx = entry.index - 1  # 0-based
        if idx < len(self.log):
            if self.log[idx].term != entry.term:
                self.log = self.log[:idx]  # truncate conflicting entries
                self.log.append(entry)
        else:
            self.log.append(entry)

    # Update commit index
    if req.leader_commit > self.commit_index:
        self.commit_index = min(req.leader_commit, len(self.log))

    return AppendEntriesResponse(
        self.current_term, True, len(self.log)
    )
```

### 3.3 로그 역추적 (충돌 해결)

`AppendEntries`가 일관성 검사에 실패하면, leader는 **역추적**해야 한다 — 로그가 일치하는 지점을 찾을 때까지 이전 `prev_log_index` 값을 시도한다.

**순진한 접근법**: 매번 `next_index`를 1씩 감소시킨다. 이는 충돌 항목당 하나의 RPC가 필요하며, 큰 차이에 대해서는 느리다.

**최적화된 접근법** (Raft 논문에서): follower가 거부 시 충돌 정보를 포함한다:

```python
@dataclass
class ConflictInfo:
    conflict_term: Optional[int] = None   # term of conflicting entry
    conflict_index: int = 0               # first index with conflict_term


def handle_append_entries_rejection(self, follower_id, conflict_info):
    """Leader handles a rejected AppendEntries with conflict info."""
    if conflict_info.conflict_term is not None:
        # Search our log for the last entry with conflict_term
        last_index_of_term = None
        for entry in reversed(self.log):
            if entry.term == conflict_info.conflict_term:
                last_index_of_term = entry.index
                break

        if last_index_of_term is not None:
            # We also have entries with that term; skip past them
            self.next_index[follower_id] = last_index_of_term + 1
        else:
            # We don't have that term; skip to follower's first index of conflict term
            self.next_index[follower_id] = conflict_info.conflict_index
    else:
        # Follower's log is too short
        self.next_index[follower_id] = conflict_info.conflict_index + 1
```

이는 일반적인 경우에 역추적을 O(항목)에서 O(term)으로 줄인다.

### 3.4 파이프라이닝과 배치 최적화

프로덕션 Raft 구현은 처리량을 최대화하기 위해 여러 기법을 사용한다:

**배치**: 여러 클라이언트 요청을 축적하고 단일 `AppendEntries`로 전송한다:

```python
class BatchingLeader:
    """Leader with request batching."""

    def __init__(self, max_batch_size=100, max_batch_delay=0.005):
        self.pending: List[LogEntry] = []
        self.max_batch_size = max_batch_size
        self.max_batch_delay = max_batch_delay  # 5ms max wait
        self.last_flush_time = time.time()

    def submit(self, command: str):
        """Add a command to the batch."""
        entry = LogEntry(term=self.current_term,
                        index=self.last_log_index() + 1,
                        command=command)
        self.pending.append(entry)

        if (len(self.pending) >= self.max_batch_size or
                time.time() - self.last_flush_time >= self.max_batch_delay):
            self.flush_batch()

    def flush_batch(self):
        """Send accumulated entries to all followers."""
        if not self.pending:
            return

        entries = self.pending[:]
        self.pending = []
        self.last_flush_time = time.time()

        for follower_id in self.peers:
            self.send_append_entries(follower_id, entries)
```

**파이프라이닝**: 이전 배치의 확인을 받기 전에 다음 배치를 전송한다:

```
Without pipelining:
  Leader → Follower: AppendEntries(entries=[A])
  Leader ← Follower: OK
  Leader → Follower: AppendEntries(entries=[B])
  Leader ← Follower: OK
  Total: 2 round trips

With pipelining (max_inflight=3):
  Leader → Follower: AppendEntries(entries=[A])
  Leader → Follower: AppendEntries(entries=[B])
  Leader → Follower: AppendEntries(entries=[C])
  Leader ← Follower: OK (A)
  Leader ← Follower: OK (B)
  Leader ← Follower: OK (C)
  Total: ~1 round trip (amortized)
```

### 3.5 커밋 인덱스 진행

Leader는 과반수 노드가 항목을 복제했을 때 `commit_index`를 진행한다. 핵심적인 안전 제약이 있다: **leader는 자신의 term 항목만 커밋한다**.

```python
def try_advance_commit(self):
    """Advance commit_index based on match_index values.

    Safety: only commit entries from current term (Section 5.4.2 of Raft paper).
    """
    for n in range(self.commit_index + 1, len(self.log) + 1):
        if self.log[n - 1].term != self.current_term:
            continue  # cannot commit entries from previous terms directly

        # Count replicas that have this entry
        replication_count = 1  # leader has it
        for peer_id in self.peers:
            if self.match_index.get(peer_id, 0) >= n:
                replication_count += 1

        majority = (len(self.peers) + 1) // 2 + 1
        if replication_count >= majority:
            self.commit_index = n
```

**이 제약이 필요한 이유?** 다음 시나리오를 고려한다(Raft 논문의 Figure 8):

```
Time  S1(L)  S2   S3   S4   S5   Event
─────────────────────────────────────────
t1    [1]    [1]  [1]  [1]  [1]   term 1: entry replicated to all
t2    [1,2]  [1]  [1]  [1]  [1]   term 2: S1 is leader, appends entry (term 2)
t3    [1,2]  [1,2] [1] [1]  [1]   term 2: replicated to S2
      S1 crashes. S5 elected in term 3.
t4    [1,2]  [1,2] [1] [1]  [1,3] term 3: S5 is leader, appends entry (term 3)
      S5 crashes. S1 recovers, elected in term 4.
t5    [1,2]  [1,2] [1,2] [1] [1,3] term 4: S1 replicates entry at index 2

If S1 commits index 2 (term 2) because 3/5 have it... then S1 crashes...
S5 could be elected (its log is as long) and overwrite index 2 with term 3 entry.
VIOLATION of safety!

Solution: S1 must not commit the term-2 entry directly. Instead, it
appends a new entry for term 4 and commits THAT. Committing the term-4
entry indirectly commits all preceding entries (including the term-2 entry).
```

---

## 4. 로그 압축

### 4.1 압축 문제

압축 없이는 로그가 무한히 증가한다. 인덱스 10,000까지 모든 항목을 적용한 노드가 여전히 항목 1-10,000을 저장한다(이미 상태 머신에 반영되었음에도).

### 4.2 스냅샷팅

가장 일반적인 접근법: 주기적으로 상태 머신을 스냅샷하고 스냅샷 인덱스까지의 로그 항목을 삭제한다.

```python
@dataclass
class Snapshot:
    last_included_index: int   # index of last entry in snapshot
    last_included_term: int    # term of last entry in snapshot
    data: bytes                # serialized state machine state


class SnapshotManager:
    """Manages log compaction via snapshots."""

    def __init__(self, threshold_entries=10000):
        self.threshold = threshold_entries
        self.snapshot: Optional[Snapshot] = None

    def maybe_snapshot(self, state_machine, log, last_applied):
        """Take a snapshot if the log is long enough."""
        entries_since_snapshot = last_applied
        if self.snapshot:
            entries_since_snapshot -= self.snapshot.last_included_index

        if entries_since_snapshot < self.threshold:
            return

        # Create snapshot
        self.snapshot = Snapshot(
            last_included_index=last_applied,
            last_included_term=log[last_applied - 1].term,
            data=state_machine.serialize()
        )

        # Discard compacted log entries
        # Keep entries after last_included_index
        del log[:last_applied]

    def restore_from_snapshot(self, state_machine, snapshot: Snapshot):
        """Restore state machine from a snapshot."""
        state_machine.deserialize(snapshot.data)
        return snapshot.last_included_index, snapshot.last_included_term
```

### 4.3 InstallSnapshot RPC

Leader가 크게 뒤처진 follower에게 항목을 복제해야 하는데 이미 압축된 항목인 경우, 대신 스냅샷을 전송한다:

```python
@dataclass
class InstallSnapshotRequest:
    term: int
    leader_id: int
    last_included_index: int
    last_included_term: int
    offset: int         # byte offset within snapshot data
    data: bytes         # raw snapshot chunk
    done: bool          # true if this is the last chunk


def handle_install_snapshot(self, req: InstallSnapshotRequest):
    """Process InstallSnapshot from leader.

    This is called when the leader has compacted entries that
    the follower still needs.
    """
    if req.term < self.current_term:
        return self.current_term

    self.current_term = req.term
    self.state = State.FOLLOWER
    self.reset_election_timeout()

    if req.offset == 0:
        # Start of new snapshot
        self.snapshot_buffer = bytearray()

    # Accumulate chunks
    self.snapshot_buffer.extend(req.data)

    if not req.done:
        return self.current_term  # wait for more chunks

    # Full snapshot received
    snapshot = Snapshot(
        last_included_index=req.last_included_index,
        last_included_term=req.last_included_term,
        data=bytes(self.snapshot_buffer)
    )

    # Discard log entries covered by snapshot
    if (req.last_included_index <= len(self.log) and
            self.log[req.last_included_index - 1].term == req.last_included_term):
        # We have matching entry; keep subsequent entries
        self.log = self.log[req.last_included_index:]
    else:
        # Snapshot is ahead of our log; discard entire log
        self.log = []

    # Apply snapshot to state machine
    self.state_machine.deserialize(snapshot.data)
    self.commit_index = req.last_included_index
    self.last_applied = req.last_included_index

    return self.current_term
```

### 4.4 증분 vs 전체 스냅샷

| 전략 | 장점 | 단점 |
|----------|-----------|---------------|
| 전체 스냅샷 | 간단한 구현; 일관된 시점 | 큰 상태에 비용이 큼 (직렬화를 위한 일시 정지) |
| Copy-on-write 스냅샷 | 일시 정지 없음; OS fork() 사용 | 스냅샷 중 높은 메모리 사용량 |
| 증분 (LSM 스타일) | 압축당 낮은 오버헤드 | 복잡한 구현; 병합 로직 필요 |
| Log-structured merge | 연속적인 압축 | 더 높은 읽기 증폭 |

etcd(가장 널리 배포된 Raft 구현)는 구성 가능한 임계값과 함께 전체 스냅샷을 사용한다. CockroachDB는 RocksDB의 내장 압축 메커니즘을 사용한다.

---

## 5. 클러스터 멤버십 변경

### 5.1 재구성 문제

클러스터 멤버십 변경(노드 추가 또는 제거)은 **서로 다른 노드가 동시에 서로 다른 구성을 관찰할 수 있어** 두 개의 독립적인 과반수를 생성할 수 있기 때문에 위험하다.

```
Old config: {A, B, C}    majority = 2
New config: {A, B, C, D, E}  majority = 3

If half the nodes switch at once:
  {A, B} think old config → majority of old config
  {C, D, E} think new config → majority of new config
  TWO LEADERS!
```

### 5.2 Joint Consensus (2단계 접근법)

Raft 논문의 원래 솔루션은 이전 구성과 새 구성 모두에서 과반수를 필요로 하는 **joint consensus** 단계를 사용한다:

```
Phase 1: C_old → C_{old,new}  (joint configuration)
  - Log entry with C_{old,new} is replicated
  - All decisions need majority of BOTH C_old AND C_new
  - Once committed, cluster is in joint consensus

Phase 2: C_{old,new} → C_new
  - Log entry with C_new is replicated
  - Once committed, old config servers can be shut down
```

```python
@dataclass
class ClusterConfig:
    """Raft cluster configuration."""
    members: Set[int]  # set of node IDs

    def majority_size(self) -> int:
        return len(self.members) // 2 + 1

    def has_majority(self, voters: Set[int]) -> bool:
        return len(voters & self.members) >= self.majority_size()


@dataclass
class JointConfig:
    """Joint configuration for safe reconfiguration."""
    old: ClusterConfig
    new: ClusterConfig

    def has_majority(self, voters: Set[int]) -> bool:
        """Require majority from BOTH old and new configurations."""
        return (self.old.has_majority(voters) and
                self.new.has_majority(voters))


class ReconfigurableRaftNode:
    """Raft node supporting joint consensus reconfiguration."""

    def __init__(self, node_id, initial_members):
        self.config = ClusterConfig(set(initial_members))
        self.joint_config: Optional[JointConfig] = None

    def start_reconfiguration(self, new_members: Set[int]):
        """Begin two-phase reconfiguration."""
        if self.joint_config is not None:
            raise RuntimeError("Reconfiguration already in progress")

        new_config = ClusterConfig(new_members)
        self.joint_config = JointConfig(old=self.config, new=new_config)

        # Replicate C_{old,new} as a special log entry
        self.append_config_entry(self.joint_config)

    def on_joint_config_committed(self):
        """Called when C_{old,new} is committed by joint majority."""
        # Phase 2: transition to C_new
        self.config = self.joint_config.new
        self.joint_config = None
        self.append_config_entry(self.config)

    def current_majority_check(self, voters: Set[int]) -> bool:
        """Check majority against current (possibly joint) config."""
        if self.joint_config is not None:
            return self.joint_config.has_majority(voters)
        return self.config.has_majority(voters)

    def append_config_entry(self, config):
        """Replicate configuration change as a log entry."""
        pass  # implementation omitted
```

### 5.3 단일 서버 변경 (더 간단한 대안)

더 간단한 접근법(역시 Ongaro의 학위 논문에서): **한 번에 하나의 서버만** 추가하거나 제거한다. 이는 최대 하나의 멤버만 다른 구성의 두 과반수가 **항상 겹치기** 때문에 안전하다.

```
Add node D:
  {A, B, C} → {A, B, C, D}
  Old majority: 2, New majority: 3
  Any old majority (2 of 3) and new majority (3 of 4) overlap by at least 1

Remove node C:
  {A, B, C, D} → {A, B, D}
  Old majority: 3, New majority: 2
  Any old majority (3 of 4) and new majority (2 of 3) overlap by at least 1
```

이것은 구현이 훨씬 간단하며 etcd와 대부분의 프로덕션 Raft 시스템에서 사용되는 접근법이다.

### 5.4 재구성 중 안전성

재구성 중 몇 가지 미묘한 문제가 발생한다:

1. **비투표 멤버**: 새 노드는 투표가 허용되기 전에 로그를 따라잡아야 한다. 그렇지 않으면 기여하기 전에 과반수 크기를 증가시켜 일시적으로 가용성을 줄인다.

2. **Leader 이전**: leader가 제거되는 경우, 퇴진하기 전에 리더십을 다른 노드로 이전해야 한다.

3. **실패 시 복구**: 재구성 항목이 커밋되지 않으면(예: leader가 충돌), 새 구성은 되돌려져야 한다.

```python
class NonVotingMember:
    """A node that is catching up but cannot vote yet."""

    def __init__(self, node_id, leader):
        self.node_id = node_id
        self.leader = leader
        self.rounds_needed = 0

    def is_caught_up(self) -> bool:
        """Check if the node has caught up sufficiently.

        The Raft paper suggests checking that the number of rounds
        needed to replicate remaining entries is small.
        """
        remaining = self.leader.last_log_index() - self.match_index
        return remaining < 100  # threshold
```

---

## 6. 읽기 최적화

### 6.1 읽기 문제

기본적으로, Raft에서의 읽기는 leader를 거쳐야 하며 로그 항목(로그에 추가되는 읽기 전용 명령)으로 취급되어야 한다. 이는 linearizability를 위해 필요하지만 불필요한 오버헤드를 추가한다: 읽기는 상태를 수정하지 않으며 지속될 필요가 없다.

### 6.2 ReadIndex

**ReadIndex**는 linearizability를 유지하면서 읽기를 로그에 추가하는 것을 피한다:

```python
def read_index(self, read_callback):
    """Linearizable read without log append.

    Steps:
    1. Record the current commit_index as read_index
    2. Confirm leadership by exchanging heartbeats with a majority
    3. Wait until state machine has applied up to read_index
    4. Execute the read
    """
    if self.state != State.LEADER:
        raise NotLeaderError()

    read_idx = self.commit_index

    # Step 2: confirm we are still leader
    ack_count = 1  # self
    for peer_id in self.peers:
        if self.send_heartbeat_and_wait_ack(peer_id):
            ack_count += 1

    majority = (len(self.peers) + 1) // 2 + 1
    if ack_count < majority:
        raise NotLeaderError("Lost leadership during ReadIndex")

    # Step 3: wait until applied up to read_idx
    while self.last_applied < read_idx:
        self.apply_next()

    # Step 4: execute the read
    return read_callback(self.state_machine)
```

### 6.3 LeaseRead

**LeaseRead**는 더 나아가 **클럭 기반 리스**를 사용하여 heartbeat 왕복을 제거한다. Leader는 heartbeat 확인을 받은 후 리스 기간 동안 leader로 유지된다고 가정한다.

```python
class LeaseBasedReader:
    """Leader-lease-based reads (no heartbeat round trip)."""

    def __init__(self, lease_duration=0.1):
        self.lease_duration = lease_duration  # 100ms
        self.lease_start: float = 0.0
        self.clock_drift_bound: float = 0.001  # max clock drift (1ms)

    def extend_lease(self, ack_time: float):
        """Called when majority of heartbeat ACKs received."""
        self.lease_start = ack_time

    def is_lease_valid(self, current_time: float) -> bool:
        """Check if leader lease is still valid.

        Must subtract clock_drift_bound for safety.
        """
        return current_time < (
            self.lease_start + self.lease_duration - self.clock_drift_bound
        )

    def lease_read(self, current_time, read_callback, state_machine):
        """Execute a read if lease is valid.

        CAUTION: Relies on bounded clock drift. If clocks are
        not well-synchronized, linearizability may be violated.
        """
        if not self.is_lease_valid(current_time):
            raise LeaseExpiredError("Lease expired; use ReadIndex instead")

        return read_callback(state_machine)
```

**중요 주의사항**: LeaseRead는 제한된 클럭 드리프트 가정에 의존한다. 클럭이 한계를 넘어 드리프트하면, 오래된 leader가 새 leader가 선출된 후에도 읽기를 서비스하여 linearizability를 위반할 수 있다.

### 6.4 ReadIndex를 사용한 Follower 읽기

Leader의 부하를 줄이기 위해, follower가 읽기를 서비스할 수 있다:

```python
def follower_read(self, leader_id):
    """Linearizable read from a follower.

    Steps:
    1. Ask the leader for its current commit_index (ReadIndex)
    2. Wait until the follower has applied up to that index
    3. Execute the read locally
    """
    # Step 1: get ReadIndex from leader
    read_idx = self.request_read_index_from_leader(leader_id)

    # Step 2: wait until we've applied up to read_idx
    while self.last_applied < read_idx:
        self.apply_next()

    # Step 3: read from local state machine
    return self.read_from_state_machine()
```

### 6.5 읽기 전략 비교

| 전략 | 지연 시간 | 처리량 | Linearizable | 클럭 의존성 |
|----------|---------|-----------|-------------|-----------------|
| 로그 읽기 (순진) | 1 RT + 디스크 | 낮음 | 예 | 아니오 |
| ReadIndex | 1 RT (heartbeat) | 보통 | 예 | 아니오 |
| LeaseRead | 0 RT (로컬) | 높음 | 예* | 예 (제한된 드리프트) |
| Follower ReadIndex | 1 RT (leader로) | 높음 (분산) | 예 | 아니오 |
| Stale read | 0 RT (로컬) | 최고 | 아니오 | 아니오 |

\* 제한된 클럭 드리프트를 가정할 때 linearizable

---

## 7. Raft에서의 Linearizable 읽기

Linearizable 읽기를 달성하려면 미묘한 문제에 주의를 기울여야 한다: 새 leader가 선출되면, 이전 term의 커밋되지 않은 항목이 있을 수 있으며, 커밋되면 읽기 결과에 영향을 미칠 수 있다.

해결책은 **선출 시 no-op** 기법이다:

```python
def on_become_leader(self):
    """Actions taken immediately after winning election."""
    # Initialize leader state
    for peer_id in self.peers:
        self.next_index[peer_id] = self.last_log_index() + 1
        self.match_index[peer_id] = 0

    # Append a no-op entry for the new term
    # This ensures we can commit entries from previous terms
    noop = LogEntry(
        term=self.current_term,
        index=self.last_log_index() + 1,
        command="NOOP"
    )
    self.log.append(noop)

    # Replicate to followers
    for peer_id in self.peers:
        self.send_append_entries(peer_id, [noop])

    # Cannot serve reads until the no-op is committed
    self.can_serve_reads = False

def on_noop_committed(self):
    """Called when the no-op entry from on_become_leader is committed."""
    self.can_serve_reads = True
```

no-op은 이중 목적을 수행한다:
1. 이전 term의 모든 항목을 커밋한다(커밋 인덱스 진행 규칙에 의해)
2. 클러스터에 대한 새 leader의 권한을 확인한다

---

## 8. 성능 튜닝

### 8.1 핵심 파라미터

| 파라미터 | 일반적인 범위 | 효과 |
|-----------|--------------|--------|
| Heartbeat 간격 | 50-200ms | 낮을수록 → 빠른 장애 감지, 더 많은 네트워크 트래픽 |
| 선출 타임아웃 | heartbeat의 5-10배 | 낮을수록 → 빠른 장애 복구, 더 많은 불필요한 선거 |
| 최대 배치 크기 | 100-10,000 항목 | 높을수록 → 더 나은 처리량, 항목당 더 높은 지연 시간 |
| 파이프라인 깊이 | 1-32 | 높을수록 → 더 나은 활용도, 더 많은 메모리 |
| 스냅샷 임계값 | 10K-100K 항목 | 낮을수록 → 적은 로그 저장소, 더 많은 스냅샷 오버헤드 |

### 8.2 지연 시간 vs 처리량 트레이드오프

```
High Throughput Configuration:
  heartbeat_interval = 200ms
  max_batch_size = 5000
  pipeline_depth = 16
  Result: ~100K ops/sec, ~50ms p99 latency

Low Latency Configuration:
  heartbeat_interval = 50ms
  max_batch_size = 1
  pipeline_depth = 1
  Result: ~5K ops/sec, ~2ms p99 latency
```

### 8.3 디스크 I/O 최적화

Write-ahead log(WAL)가 주요 병목이다. 기법:

```python
class OptimizedWAL:
    """Write-ahead log with batched fsync."""

    def __init__(self, path):
        self.fd = open(path, 'ab', buffering=0)
        self.pending_entries = []

    def append(self, entry: LogEntry):
        """Buffer an entry for batch write."""
        self.pending_entries.append(entry)

    def sync(self):
        """Batch write and fsync all pending entries.

        This is significantly faster than individual fsyncs:
        1 fsync for N entries vs N fsyncs for N entries.
        """
        if not self.pending_entries:
            return

        data = b''.join(
            self._serialize(e) for e in self.pending_entries
        )
        self.fd.write(data)
        os.fsync(self.fd.fileno())
        self.pending_entries.clear()

    def _serialize(self, entry):
        """Serialize a log entry (implementation omitted)."""
        return b''
```

### 8.4 네트워크 최적화

```
Optimization                    Impact
─────────────────────────────────────────
gRPC streaming                  Avoids per-request overhead
Compression (snappy/lz4)        Reduces bandwidth 50-80%
Request pipelining              Better link utilization
Parallel replication            Overlap I/O across followers
```

---

## 9. 코드: Raft Leader 선출 시뮬레이터

```python
"""
Raft Leader Election Simulator with Pre-Vote

Demonstrates:
- Randomized election timeouts
- Pre-vote to prevent disruption from partitioned nodes
- Split vote resolution
- Term progression
"""

import random
import heapq
from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, Tuple
from enum import Enum


class NodeState(Enum):
    FOLLOWER = "follower"
    PRE_CANDIDATE = "pre_candidate"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass(order=True)
class TimerEvent:
    time: float
    node_id: int = field(compare=False)
    event_type: str = field(compare=False)


@dataclass
class VoteRequest:
    term: int
    candidate_id: int
    last_log_term: int
    last_log_index: int
    is_pre_vote: bool = False


@dataclass
class VoteResponse:
    term: int
    granted: bool
    voter_id: int
    is_pre_vote: bool = False


class RaftElectionNode:
    """Raft node focused on leader election with pre-vote."""

    def __init__(self, node_id: int, cluster_size: int):
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.last_log_term = 0
        self.last_log_index = 0

        # Election state
        self.votes: Set[int] = set()
        self.pre_votes: Set[int] = set()
        self.last_heartbeat: float = 0.0

        # Statistics
        self.elections_started = 0
        self.terms_seen: List[int] = []

    @property
    def majority(self):
        return self.cluster_size // 2 + 1

    def election_timeout(self) -> float:
        """Randomized election timeout (150-300ms simulated as 15-30 units)."""
        return random.uniform(15.0, 30.0)

    def handle_vote_request(self, req: VoteRequest, current_time: float) -> VoteResponse:
        """Process a RequestVote or PreVote RPC."""
        if req.is_pre_vote:
            return self._handle_pre_vote(req, current_time)
        return self._handle_real_vote(req)

    def _handle_pre_vote(self, req: VoteRequest, current_time: float) -> VoteResponse:
        """Handle PreVote: don't change state, just indicate willingness to vote."""
        # Deny if we recently heard from a leader
        if current_time - self.last_heartbeat < 15.0:
            return VoteResponse(self.current_term, False, self.node_id, True)

        # Deny if candidate's term is too low
        if req.term < self.current_term:
            return VoteResponse(self.current_term, False, self.node_id, True)

        # Check log freshness
        if not self._is_candidate_log_ok(req.last_log_term, req.last_log_index):
            return VoteResponse(self.current_term, False, self.node_id, True)

        return VoteResponse(self.current_term, True, self.node_id, True)

    def _handle_real_vote(self, req: VoteRequest) -> VoteResponse:
        """Handle real RequestVote."""
        if req.term > self.current_term:
            self.current_term = req.term
            self.state = NodeState.FOLLOWER
            self.voted_for = None

        if req.term < self.current_term:
            return VoteResponse(self.current_term, False, self.node_id)

        if self.voted_for is not None and self.voted_for != req.candidate_id:
            return VoteResponse(self.current_term, False, self.node_id)

        if not self._is_candidate_log_ok(req.last_log_term, req.last_log_index):
            return VoteResponse(self.current_term, False, self.node_id)

        self.voted_for = req.candidate_id
        return VoteResponse(self.current_term, True, self.node_id)

    def _is_candidate_log_ok(self, last_term, last_index):
        if last_term != self.last_log_term:
            return last_term > self.last_log_term
        return last_index >= self.last_log_index

    def start_pre_vote(self):
        """Begin pre-vote phase."""
        self.state = NodeState.PRE_CANDIDATE
        self.pre_votes = {self.node_id}

    def start_election(self):
        """Begin real election (after successful pre-vote)."""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes = {self.node_id}
        self.elections_started += 1
        self.terms_seen.append(self.current_term)

    def become_leader(self):
        """Transition to leader state."""
        self.state = NodeState.LEADER

    def receive_heartbeat(self, leader_term: int, current_time: float):
        """Process heartbeat from leader."""
        if leader_term >= self.current_term:
            self.current_term = leader_term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            self.last_heartbeat = current_time


class ElectionSimulator:
    """Simulate Raft leader elections across a cluster."""

    def __init__(self, cluster_size: int = 5, use_pre_vote: bool = True,
                 seed: int = 42):
        random.seed(seed)
        self.cluster_size = cluster_size
        self.use_pre_vote = use_pre_vote
        self.nodes: Dict[int, RaftElectionNode] = {
            i: RaftElectionNode(i, cluster_size)
            for i in range(cluster_size)
        }
        self.timer_queue: List[TimerEvent] = []
        self.current_time = 0.0
        self.history: List[str] = []
        self.leader_id: Optional[int] = None

        # Schedule initial election timeouts
        for node_id in self.nodes:
            timeout = self.nodes[node_id].election_timeout()
            heapq.heappush(self.timer_queue, TimerEvent(timeout, node_id, "election"))

    def run(self, max_time: float = 200.0) -> Optional[int]:
        """Run simulation until a leader is elected or time runs out."""
        while self.timer_queue and self.current_time < max_time:
            event = heapq.heappop(self.timer_queue)
            self.current_time = event.time
            node = self.nodes[event.node_id]

            if event.event_type == "election":
                if node.state == NodeState.LEADER:
                    continue
                if node.state == NodeState.FOLLOWER and self.leader_id is not None:
                    # Valid leader exists; reset timeout
                    node.receive_heartbeat(
                        self.nodes[self.leader_id].current_term,
                        self.current_time
                    )
                    heapq.heappush(self.timer_queue, TimerEvent(
                        self.current_time + node.election_timeout(),
                        event.node_id, "election"
                    ))
                    continue

                self._run_election(event.node_id)

                if node.state != NodeState.LEADER:
                    # Schedule retry
                    heapq.heappush(self.timer_queue, TimerEvent(
                        self.current_time + node.election_timeout(),
                        event.node_id, "election"
                    ))

            elif event.event_type == "heartbeat":
                if node.state == NodeState.LEADER:
                    for peer_id in self.nodes:
                        if peer_id != event.node_id:
                            self.nodes[peer_id].receive_heartbeat(
                                node.current_term, self.current_time
                            )
                    heapq.heappush(self.timer_queue, TimerEvent(
                        self.current_time + 5.0,  # heartbeat interval
                        event.node_id, "heartbeat"
                    ))

        return self.leader_id

    def _run_election(self, candidate_id: int):
        """Run an election for the given candidate."""
        node = self.nodes[candidate_id]

        if self.use_pre_vote:
            # Phase 0: Pre-vote
            node.start_pre_vote()
            pre_vote_req = VoteRequest(
                term=node.current_term + 1,
                candidate_id=candidate_id,
                last_log_term=node.last_log_term,
                last_log_index=node.last_log_index,
                is_pre_vote=True
            )
            for peer_id, peer in self.nodes.items():
                if peer_id == candidate_id:
                    continue
                resp = peer.handle_vote_request(pre_vote_req, self.current_time)
                if resp.granted:
                    node.pre_votes.add(peer_id)

            if len(node.pre_votes) < node.majority:
                self.history.append(
                    f"t={self.current_time:.1f}: Node {candidate_id} "
                    f"pre-vote failed ({len(node.pre_votes)}/{node.majority})"
                )
                node.state = NodeState.FOLLOWER
                return

            self.history.append(
                f"t={self.current_time:.1f}: Node {candidate_id} "
                f"pre-vote passed ({len(node.pre_votes)}/{node.majority})"
            )

        # Phase 1: Real election
        node.start_election()
        vote_req = VoteRequest(
            term=node.current_term,
            candidate_id=candidate_id,
            last_log_term=node.last_log_term,
            last_log_index=node.last_log_index
        )

        for peer_id, peer in self.nodes.items():
            if peer_id == candidate_id:
                continue
            resp = peer.handle_vote_request(vote_req, self.current_time)
            if resp.granted:
                node.votes.add(peer_id)

        if len(node.votes) >= node.majority:
            node.become_leader()
            self.leader_id = candidate_id
            self.history.append(
                f"t={self.current_time:.1f}: Node {candidate_id} elected leader "
                f"for term {node.current_term} "
                f"(votes: {sorted(node.votes)})"
            )
            # Schedule heartbeats
            heapq.heappush(self.timer_queue, TimerEvent(
                self.current_time + 5.0, candidate_id, "heartbeat"
            ))
        else:
            self.history.append(
                f"t={self.current_time:.1f}: Node {candidate_id} election failed "
                f"in term {node.current_term} "
                f"({len(node.votes)}/{node.majority} votes)"
            )

    def print_report(self):
        """Print simulation results."""
        print(f"\n{'='*60}")
        print(f"Raft Election Simulation ({'pre-vote' if self.use_pre_vote else 'standard'})")
        print(f"Cluster size: {self.cluster_size}")
        print(f"{'='*60}\n")

        print("Event Log:")
        for event in self.history:
            print(f"  {event}")

        print(f"\nFinal State (t={self.current_time:.1f}):")
        for nid, node in sorted(self.nodes.items()):
            print(f"  Node {nid}: state={node.state.value}, "
                  f"term={node.current_term}, "
                  f"elections_started={node.elections_started}")

        if self.leader_id is not None:
            leader = self.nodes[self.leader_id]
            print(f"\nLeader: Node {self.leader_id} (term {leader.current_term})")
        else:
            print("\nNo leader elected!")


def main():
    # Scenario 1: Normal election with pre-vote
    print("\n" + "=" * 60)
    print("Scenario 1: Normal election with pre-vote")
    print("=" * 60)
    sim1 = ElectionSimulator(cluster_size=5, use_pre_vote=True, seed=42)
    sim1.run(max_time=100.0)
    sim1.print_report()

    # Scenario 2: Normal election without pre-vote
    print("\n" + "=" * 60)
    print("Scenario 2: Normal election without pre-vote")
    print("=" * 60)
    sim2 = ElectionSimulator(cluster_size=5, use_pre_vote=False, seed=42)
    sim2.run(max_time=100.0)
    sim2.print_report()

    # Scenario 3: Larger cluster (7 nodes) showing split vote resolution
    print("\n" + "=" * 60)
    print("Scenario 3: 7-node cluster")
    print("=" * 60)
    sim3 = ElectionSimulator(cluster_size=7, use_pre_vote=True, seed=99)
    sim3.run(max_time=100.0)
    sim3.print_report()


if __name__ == "__main__":
    main()
```

### 9.1 핵심 관찰

시뮬레이터를 실행하면 몇 가지 중요한 동작이 드러난다:

1. **첫 타임아웃 승리**: 가장 짧은 랜덤 타임아웃을 가진 노드가 일반적으로 첫 시도에서 leader가 된다.

2. **Pre-vote 필터링**: Pre-vote가 활성화되면, 재합류하는 분리된 노드가 클러스터를 방해할 수 없다. 다른 노드들이 최근에 heartbeat를 받았기 때문에 pre-vote를 거부한다.

3. **Split vote는 드물다**: 2:1 타임아웃 비율(150-300ms)에서, 두 노드가 같은 heartbeat 간격 내에 타임아웃되는 확률은 낮다.

4. **Pre-vote 없이 term 인플레이션**: Pre-vote 없이, 분리된 노드는 실패한 선거마다 term을 증가시킨다. 재합류할 때, 파괴적인 leader 선거를 강제한다.

---

## 10. 요약

Raft는 합의 문제를 leader 선출, 로그 복제, 안전성으로 분해하여, Multi-Paxos와 동일한 정확성 보장을 달성하면서 이해하고 구현하기 상당히 쉽게 만들었다.

Raft를 실용적으로 만드는 핵심 엔지니어링 결정:

- **랜덤 타임아웃**은 명시적 조정 없이 split vote를 방지한다
- **Pre-vote**는 분리된 노드가 클러스터를 방해하는 것을 방지한다
- **Log matching property**는 간단한 이전 항목 검사로 일관성을 보장한다
- **커밋 제약** (현재 term 항목만 커밋)은 Figure 8의 미묘한 안전 위반을 방지한다
- **스냅샷팅**은 정확성을 유지하면서 로그 크기를 제한한다
- **Joint consensus**는 안전한 클러스터 재구성을 가능하게 한다
- **ReadIndex와 LeaseRead**는 로그 오버헤드 없이 빠른 읽기를 가능하게 한다

etcd, CockroachDB, TiKV와 같은 프로덕션 시스템은 전세계적으로 분산된 클러스터에서 초당 수백만 건의 연산을 처리하며 Raft의 실용성을 입증했다.

---

[다음: Byzantine Fault Tolerance](./07_Byzantine_Fault_Tolerance.md)
