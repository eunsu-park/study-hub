# Lesson 6: Raft In Depth

[Overview](./00_Overview.md) | [Previous: Paxos Family](./05_Paxos_Family.md) | [Next: Byzantine Fault Tolerance](./07_Byzantine_Fault_Tolerance.md)

---

## Learning Objectives

- Understand Raft's design philosophy of understandability and how it decomposes consensus
- Implement and analyze leader election with pre-vote extensions and split-vote resolution
- Trace log replication mechanics including conflict resolution, pipelining, and commit advancement
- Explain log compaction strategies including snapshotting and the InstallSnapshot RPC
- Design cluster membership changes using joint consensus and single-server approaches
- Implement read optimizations (ReadIndex, LeaseRead, follower reads) for linearizable reads

---

## 1. Motivation: Understandability over Paxos

### 1.1 The Raft Design Philosophy

Diego Ongaro and John Ousterhout published the Raft consensus algorithm in 2014 with an explicit goal: **understandability**. Their user study showed that students learned Raft significantly faster and more accurately than Paxos.

Raft achieves this through two key techniques:

1. **Decomposition**: Raft separates consensus into three relatively independent sub-problems:
   - Leader election
   - Log replication
   - Safety (ensuring the leader has all committed entries)

2. **State space reduction**: Raft disallows configurations that add unnecessary complexity. For example, logs cannot have holes (unlike Multi-Paxos), and only nodes with complete logs can become leader.

### 1.2 Raft vs Multi-Paxos

| Aspect | Multi-Paxos | Raft |
|--------|------------|------|
| Specification | Incomplete (many gaps) | Complete (covers all cases) |
| Leader election | Implementation-defined | Integrated into protocol |
| Log holes | Allowed (gap filling needed) | Never (contiguous logs) |
| Who can be leader | Any node | Only nodes with up-to-date logs |
| Reconfiguration | Unspecified | Joint consensus / single-server |
| Implementations | Varied, often subtly wrong | Consistent across implementations |

### 1.3 Key Raft Properties

Raft guarantees five properties:

1. **Election Safety**: At most one leader can be elected in a given term
2. **Leader Append-Only**: A leader never overwrites or deletes log entries; it only appends
3. **Log Matching**: If two logs contain an entry with the same index and term, the logs are identical up to that index
4. **Leader Completeness**: If a log entry is committed in a given term, it will be present in the logs of all leaders for higher terms
5. **State Machine Safety**: If a server has applied a log entry at a given index, no other server will ever apply a different entry for that index

---

## 2. Leader Election Deep Dive

### 2.1 Election Mechanism

Every Raft node is in one of three states:

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

### 2.2 Election Timeouts and Randomization

Each follower maintains an **election timeout** — a random duration between `T` and `2T` (typically 150-300ms). When the timeout expires without receiving a heartbeat from the leader, the follower becomes a candidate.

The randomization is critical for avoiding **split votes**: if all nodes used the same timeout, they would all become candidates simultaneously and split the vote.

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

### 2.3 Pre-Vote Extension

A known issue with Raft: a node that is partitioned from the leader will repeatedly time out, increment its term, and start elections. When it rejoins the cluster, its high term number forces the current leader to step down, disrupting the cluster unnecessarily.

The **Pre-Vote** extension (Ongaro's dissertation, Section 9.6) adds a preliminary phase:

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

### 2.4 Term Confusion and Stale Leaders

A **stale leader** is a node that still believes it is leader but has been superseded by a new leader in a higher term. This can happen after a network partition heals.

Raft handles this through **term checking**: every RPC includes the sender's term. If a node receives a message with a higher term, it immediately steps down to follower.

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

### 2.5 Split Vote Resolution

When two candidates receive equal numbers of votes (split vote), neither achieves a majority. Both time out (at different random times due to randomized timeouts) and start new elections with higher terms.

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

The probability of repeated split votes decreases exponentially with the timeout range. With a range of 150-300ms and 5 nodes, the expected time to elect a leader after a failure is typically under 1 second.

---

## 3. Log Replication

### 3.1 AppendEntries RPC

The leader replicates log entries to followers using the `AppendEntries` RPC, which also serves as the heartbeat mechanism (with an empty entries list).

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

### 3.2 Consistency Check

The `prev_log_index` and `prev_log_term` fields implement the **Log Matching Property**. Before appending new entries, the follower checks that it has a matching entry at `prev_log_index` with term `prev_log_term`:

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

### 3.3 Log Backtracking (Conflict Resolution)

When `AppendEntries` fails the consistency check, the leader must **backtrack** — try earlier `prev_log_index` values until it finds a point where the logs agree.

**Naive approach**: Decrement `next_index` by 1 each time. This requires one RPC per conflicting entry, which is slow for large divergences.

**Optimized approach** (from the Raft paper): The follower includes conflict information in its rejection:

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

This reduces backtracking from O(entries) to O(terms) in the common case.

### 3.4 Pipelining and Batching Optimizations

Production Raft implementations use several techniques to maximize throughput:

**Batching**: Accumulate multiple client requests and send them in a single `AppendEntries`:

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

**Pipelining**: Send the next batch before receiving acknowledgment for the previous one:

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

### 3.5 Commit Index Advancement

The leader advances `commit_index` when a majority of nodes have replicated an entry. There is a critical safety constraint: **the leader only commits entries from its own term**.

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

**Why this restriction?** Consider the following scenario (Figure 8 from the Raft paper):

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

## 4. Log Compaction

### 4.1 The Compaction Problem

Without compaction, the log grows indefinitely. A node that has applied all entries up to index 10,000 still stores entries 1-10,000 (even though they are already reflected in the state machine).

### 4.2 Snapshotting

The most common approach: periodically snapshot the state machine and discard log entries up to the snapshot index.

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

When a leader needs to replicate entries to a follower that has fallen far behind (entries already compacted), it sends the snapshot instead:

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

### 4.4 Incremental vs Full Snapshots

| Strategy | Advantages | Disadvantages |
|----------|-----------|---------------|
| Full snapshot | Simple implementation; consistent point-in-time | Expensive for large state (pause for serialization) |
| Copy-on-write snapshot | No pause; uses OS fork() | High memory usage during snapshot |
| Incremental (LSM-style) | Low overhead per compaction | Complex implementation; needs merge logic |
| Log-structured merge | Continuous compaction | Higher read amplification |

etcd (the most widely deployed Raft implementation) uses full snapshots with configurable thresholds. CockroachDB uses RocksDB's built-in compaction mechanism.

---

## 5. Cluster Membership Changes

### 5.1 The Reconfiguration Problem

Changing cluster membership (adding or removing nodes) is dangerous because **different nodes may observe different configurations at the same time**, potentially creating two independent majorities.

```
Old config: {A, B, C}    majority = 2
New config: {A, B, C, D, E}  majority = 3

If half the nodes switch at once:
  {A, B} think old config → majority of old config
  {C, D, E} think new config → majority of new config
  TWO LEADERS!
```

### 5.2 Joint Consensus (Two-Phase Approach)

The Raft paper's original solution uses a **joint consensus** phase where decisions require majorities from both the old and new configurations:

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

### 5.3 Single-Server Changes (Simpler Alternative)

A simpler approach (also from Ongaro's dissertation): only add or remove **one server at a time**. This is safe because any two majorities of configurations that differ by at most one member **always overlap**.

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

This is much simpler to implement and is the approach used by etcd and most production Raft systems.

### 5.4 Safety During Reconfiguration

Several subtle issues arise during reconfiguration:

1. **Non-voting members**: New nodes need to catch up on the log before they should be allowed to vote. Otherwise, they increase the majority size before they can contribute, temporarily reducing availability.

2. **Leader transfer**: If the leader is being removed, it should transfer leadership to another node before stepping down.

3. **Revert on failure**: If a reconfiguration entry is not committed (e.g., the leader crashes), the new configuration should be reverted.

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

## 6. Read Optimizations

### 6.1 The Read Problem

By default, reads in Raft must go through the leader and be treated as log entries (read-only commands appended to the log). This is necessary for linearizability but adds unnecessary overhead: reads don't modify state and don't need to be persisted.

### 6.2 ReadIndex

**ReadIndex** avoids appending reads to the log while maintaining linearizability:

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

**LeaseRead** goes further: it eliminates the heartbeat round trip by using **clock-based leases**. The leader assumes it remains leader for a lease period after receiving heartbeat acknowledgments.

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

**Important caveat**: LeaseRead relies on the assumption of bounded clock drift. If clocks drift beyond the bound, a stale leader may serve reads after a new leader has been elected, violating linearizability.

### 6.4 Follower Reads with ReadIndex

To reduce load on the leader, reads can be served by followers:

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

### 6.5 Comparison of Read Strategies

| Strategy | Latency | Throughput | Linearizable | Clock Dependency |
|----------|---------|-----------|-------------|-----------------|
| Log read (naive) | 1 RT + disk | Low | Yes | No |
| ReadIndex | 1 RT (heartbeat) | Medium | Yes | No |
| LeaseRead | 0 RT (local) | High | Yes* | Yes (bounded drift) |
| Follower ReadIndex | 1 RT (to leader) | High (distributed) | Yes | No |
| Stale read | 0 RT (local) | Highest | No | No |

\* Linearizable assuming bounded clock drift

---

## 7. Linearizable Reads in Raft

Achieving linearizable reads requires careful attention to a subtle issue: when a new leader is elected, there may be uncommitted entries from the previous term that, if committed, would affect read results.

The solution involves the **no-op on election** technique:

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

The no-op serves a dual purpose:
1. It commits all entries from previous terms (by the commit index advancement rule)
2. It confirms the new leader's authority with the cluster

---

## 8. Performance Tuning

### 8.1 Key Parameters

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| Heartbeat interval | 50-200ms | Lower → faster failure detection, more network traffic |
| Election timeout | 5-10× heartbeat | Lower → faster failover, more spurious elections |
| Max batch size | 100-10,000 entries | Higher → better throughput, higher latency per entry |
| Pipeline depth | 1-32 | Higher → better utilization, more memory |
| Snapshot threshold | 10K-100K entries | Lower → less log storage, more snapshot overhead |

### 8.2 Latency vs Throughput Tradeoffs

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

### 8.3 Disk I/O Optimization

The write-ahead log (WAL) is the primary bottleneck. Techniques:

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

### 8.4 Network Optimization

```
Optimization                    Impact
─────────────────────────────────────────
gRPC streaming                  Avoids per-request overhead
Compression (snappy/lz4)        Reduces bandwidth 50-80%
Request pipelining              Better link utilization
Parallel replication            Overlap I/O across followers
```

---

## 9. Code: Raft Leader Election Simulator

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

### 9.1 Key Observations

Running the simulator reveals several important behaviors:

1. **First timeout wins**: The node with the shortest random timeout typically becomes leader in the first attempt.

2. **Pre-vote filtering**: With pre-vote enabled, partitioned nodes that rejoin cannot disrupt the cluster because other nodes reject their pre-vote (they recently received heartbeats).

3. **Split votes are rare**: With the 2:1 timeout ratio (150-300ms), the probability of two nodes timing out within the same heartbeat interval is low.

4. **Term inflation without pre-vote**: Without pre-vote, a partitioned node increments its term with every failed election. When it rejoins, it forces a disruptive leader election.

---

## 10. Summary

Raft decomposes the consensus problem into leader election, log replication, and safety, achieving the same correctness guarantees as Multi-Paxos while being significantly easier to understand and implement.

Key engineering decisions that make Raft practical:

- **Randomized timeouts** prevent split votes without explicit coordination
- **Pre-vote** prevents partitioned nodes from disrupting the cluster
- **Log matching property** ensures consistency with a simple prev-entry check
- **Commit restriction** (only commit entries from the current term) prevents the subtle safety violation in Figure 8
- **Snapshotting** bounds log size while maintaining correctness
- **Joint consensus** enables safe cluster reconfiguration
- **ReadIndex and LeaseRead** enable fast reads without log overhead

Production systems like etcd, CockroachDB, and TiKV have proven Raft's practicality at scale, processing millions of operations per second across globally distributed clusters.

---

[Next: Byzantine Fault Tolerance](./07_Byzantine_Fault_Tolerance.md)
