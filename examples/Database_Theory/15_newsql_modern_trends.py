"""
NewSQL and Modern Trends — Simulations

Demonstrates core NewSQL and modern database concepts using only the standard library:
- Distributed transactions with Two-Phase Commit (2PC)
- Multi-Version Concurrency Control (MVCC) with snapshot isolation
- Raft consensus protocol (leader election + log replication)
- Hybrid Logical Clocks (HLC) for distributed timestamp ordering
- Vector similarity search (brute-force + simple IVF index)
- Time-series data storage with downsampling

Theory:
- NewSQL = ACID + SQL + horizontal scalability
- Key innovations: sharding with distributed transactions, MVCC, Raft consensus
- Google Spanner uses TrueTime (GPS + atomic clocks) for external consistency
- CockroachDB uses Hybrid Logical Clocks (no special hardware)
- Vector databases enable semantic similarity search for AI/ML workloads
- Time-series databases optimize for append-heavy, time-ordered data

All examples use only the Python standard library.
"""

import time
import random
import math
import threading
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 1. TWO-PHASE COMMIT (2PC) — Distributed Transactions
# ============================================================

class VoteResult(Enum):
    COMMIT = "COMMIT"
    ABORT = "ABORT"


class Participant:
    """A node participating in a distributed transaction.

    In NewSQL systems, each shard is a participant that must agree
    to commit or abort a distributed transaction.
    """

    def __init__(self, name: str, fail_on_prepare: bool = False):
        self.name = name
        self.fail_on_prepare = fail_on_prepare
        self.data: Dict[str, Any] = {}
        self._pending: Dict[str, Dict[str, Any]] = {}  # txn_id -> changes
        self.log: List[str] = []

    def prepare(self, txn_id: str, changes: Dict[str, Any]) -> VoteResult:
        """Phase 1: Coordinator asks participant to prepare."""
        self.log.append(f"PREPARE {txn_id}")
        if self.fail_on_prepare:
            self.log.append(f"VOTE ABORT {txn_id} (simulated failure)")
            return VoteResult.ABORT
        self._pending[txn_id] = changes
        self.log.append(f"VOTE COMMIT {txn_id}")
        return VoteResult.COMMIT

    def commit(self, txn_id: str) -> None:
        """Phase 2: Apply the prepared changes."""
        if txn_id in self._pending:
            self.data.update(self._pending.pop(txn_id))
            self.log.append(f"COMMIT {txn_id}")

    def abort(self, txn_id: str) -> None:
        """Phase 2: Discard the prepared changes."""
        self._pending.pop(txn_id, None)
        self.log.append(f"ABORT {txn_id}")


class Coordinator:
    """Transaction coordinator implementing Two-Phase Commit.

    In NewSQL (e.g., CockroachDB), the gateway node acts as coordinator
    for distributed transactions that span multiple ranges/shards.
    """

    def __init__(self, participants: List[Participant]):
        self.participants = participants
        self.log: List[str] = []
        self._txn_counter = 0

    def execute_transaction(self, changes_per_participant: Dict[str, Dict[str, Any]]) -> bool:
        """Execute a distributed transaction across participants.

        Returns True if committed, False if aborted.
        """
        self._txn_counter += 1
        txn_id = f"txn_{self._txn_counter:03d}"
        self.log.append(f"BEGIN {txn_id}")

        # Phase 1: PREPARE
        self.log.append(f"PHASE 1 — Sending PREPARE to all participants")
        votes = {}
        for participant in self.participants:
            changes = changes_per_participant.get(participant.name, {})
            vote = participant.prepare(txn_id, changes)
            votes[participant.name] = vote
            self.log.append(f"  {participant.name} voted {vote.value}")

        # Decision
        all_commit = all(v == VoteResult.COMMIT for v in votes.values())

        # Phase 2: COMMIT or ABORT
        if all_commit:
            self.log.append(f"PHASE 2 — COMMIT (all voted COMMIT)")
            for participant in self.participants:
                participant.commit(txn_id)
            self.log.append(f"COMMITTED {txn_id}")
            return True
        else:
            self.log.append(f"PHASE 2 — ABORT (at least one voted ABORT)")
            for participant in self.participants:
                participant.abort(txn_id)
            self.log.append(f"ABORTED {txn_id}")
            return False


def demonstrate_two_phase_commit():
    """Demonstrate distributed transactions with 2PC."""
    print("=" * 60)
    print("1. TWO-PHASE COMMIT (Distributed Transactions)")
    print("=" * 60)
    print()

    # Scenario 1: Successful distributed transaction
    print("1.1 Successful Transaction (Transfer $100: Account A -> B)")
    print("-" * 60)

    shard_a = Participant("shard_A")
    shard_b = Participant("shard_B")
    shard_a.data = {"account_A": 500}
    shard_b.data = {"account_B": 300}

    coord = Coordinator([shard_a, shard_b])
    result = coord.execute_transaction({
        "shard_A": {"account_A": 400},  # debit 100
        "shard_B": {"account_B": 400},  # credit 100
    })

    for entry in coord.log:
        print(f"  [Coordinator] {entry}")
    print(f"\n  Result: {'COMMITTED' if result else 'ABORTED'}")
    print(f"  shard_A data: {shard_a.data}")
    print(f"  shard_B data: {shard_b.data}")

    # Scenario 2: Failed transaction (one participant aborts)
    print("\n1.2 Failed Transaction (Shard C Fails)")
    print("-" * 60)

    shard_c = Participant("shard_C")
    shard_d = Participant("shard_D", fail_on_prepare=True)  # will fail
    shard_c.data = {"inventory": 10}
    shard_d.data = {"order_status": "pending"}

    coord2 = Coordinator([shard_c, shard_d])
    result = coord2.execute_transaction({
        "shard_C": {"inventory": 9},
        "shard_D": {"order_status": "confirmed"},
    })

    for entry in coord2.log:
        print(f"  [Coordinator] {entry}")
    print(f"\n  Result: {'COMMITTED' if result else 'ABORTED'}")
    print(f"  shard_C data: {shard_c.data}  (unchanged — atomicity!)")
    print(f"  shard_D data: {shard_d.data}  (unchanged)")

    print()


# ============================================================
# 2. MULTI-VERSION CONCURRENCY CONTROL (MVCC)
# ============================================================

@dataclass
class MVCCVersion:
    """A single version of a value, tagged with a timestamp."""
    timestamp: int
    value: Any
    deleted: bool = False


class MVCCStore:
    """MVCC key-value store with snapshot isolation.

    Each write creates a new version instead of overwriting.
    Reads see a consistent snapshot at a given timestamp.

    This is the core mechanism in NewSQL systems:
    - Spanner: globally ordered timestamps via TrueTime
    - CockroachDB: timestamps via Hybrid Logical Clocks
    - TiDB: timestamps via centralized Timestamp Oracle
    """

    def __init__(self):
        self._versions: Dict[str, List[MVCCVersion]] = defaultdict(list)
        self._timestamp = 0

    def _next_ts(self) -> int:
        self._timestamp += 1
        return self._timestamp

    def begin_transaction(self) -> int:
        """Start a transaction, returning its snapshot timestamp."""
        return self._next_ts()

    def read(self, key: str, snapshot_ts: int) -> Optional[Any]:
        """Read the latest version of key visible at snapshot_ts."""
        if key not in self._versions:
            return None
        # Find the latest version with timestamp <= snapshot_ts
        for version in reversed(self._versions[key]):
            if version.timestamp <= snapshot_ts:
                return None if version.deleted else version.value
        return None

    def write(self, key: str, value: Any, txn_ts: int) -> int:
        """Write a new version of key. Returns the commit timestamp."""
        commit_ts = self._next_ts()
        self._versions[key].append(MVCCVersion(
            timestamp=commit_ts, value=value
        ))
        return commit_ts

    def delete(self, key: str, txn_ts: int) -> int:
        """Mark key as deleted (tombstone). Returns commit timestamp."""
        commit_ts = self._next_ts()
        self._versions[key].append(MVCCVersion(
            timestamp=commit_ts, value=None, deleted=True
        ))
        return commit_ts

    def version_count(self, key: str) -> int:
        return len(self._versions.get(key, []))

    def all_versions(self, key: str) -> List[Tuple[int, Any, bool]]:
        """Return all versions of a key: (timestamp, value, deleted)."""
        return [(v.timestamp, v.value, v.deleted)
                for v in self._versions.get(key, [])]


def demonstrate_mvcc():
    """Demonstrate MVCC with snapshot isolation."""
    print("=" * 60)
    print("2. MULTI-VERSION CONCURRENCY CONTROL (MVCC)")
    print("=" * 60)
    print()

    store = MVCCStore()

    print("2.1 Version Chain")
    print("-" * 60)

    # Write multiple versions
    ts1 = store.begin_transaction()
    store.write("balance", 1000, ts1)
    print(f"  T1 (ts={ts1}): write balance=1000")

    ts2 = store.begin_transaction()
    store.write("balance", 900, ts2)
    print(f"  T2 (ts={ts2}): write balance=900")

    ts3 = store.begin_transaction()
    store.write("balance", 750, ts3)
    print(f"  T3 (ts={ts3}): write balance=750")

    print(f"\n  Version chain for 'balance': {store.version_count('balance')} versions")
    for ts, val, deleted in store.all_versions("balance"):
        print(f"    ts={ts}: value={val}, deleted={deleted}")

    # Snapshot reads
    print("\n2.2 Snapshot Isolation (Reads See Consistent Point-in-Time)")
    print("-" * 60)

    # A snapshot at ts=2 should see the value written at ts1
    val_at_ts2 = store.read("balance", snapshot_ts=2)
    val_at_ts4 = store.read("balance", snapshot_ts=4)
    val_at_ts6 = store.read("balance", snapshot_ts=6)
    val_at_ts99 = store.read("balance", snapshot_ts=99)

    print(f"  read('balance', snapshot_ts=2) = {val_at_ts2}   (sees T1's write)")
    print(f"  read('balance', snapshot_ts=4) = {val_at_ts4}   (sees T2's write)")
    print(f"  read('balance', snapshot_ts=6) = {val_at_ts6}   (sees T3's write)")
    print(f"  read('balance', snapshot_ts=99) = {val_at_ts99}  (sees latest)")

    # Concurrent transactions don't interfere
    print("\n2.3 Concurrent Transactions (No Blocking)")
    print("-" * 60)

    # Transaction A starts a snapshot, Transaction B writes, A still sees old value
    ts_a = store.begin_transaction()
    print(f"  Transaction A starts at ts={ts_a}")

    ts_b = store.begin_transaction()
    store.write("balance", 500, ts_b)
    print(f"  Transaction B writes balance=500 at ts={ts_b}")

    val_a = store.read("balance", snapshot_ts=ts_a)
    print(f"  Transaction A reads balance = {val_a}  (still sees old value!)")
    print("  -> A's snapshot is not affected by B's write (isolation)")

    val_latest = store.read("balance", snapshot_ts=99)
    print(f"  New transaction reads balance = {val_latest}  (sees B's write)")

    # Delete with tombstone
    print("\n2.4 Delete via Tombstone")
    print("-" * 60)
    store.write("temp_key", "hello", store.begin_transaction())
    ts_before_delete = store.begin_transaction()
    store.delete("temp_key", store.begin_transaction())
    ts_after_delete = store.begin_transaction()

    print(f"  read('temp_key', before delete) = {store.read('temp_key', ts_before_delete)}")
    print(f"  read('temp_key', after delete)  = {store.read('temp_key', ts_after_delete)}")
    print(f"  Versions of 'temp_key': {store.all_versions('temp_key')}")

    print()


# ============================================================
# 3. RAFT CONSENSUS PROTOCOL
# ============================================================

class RaftState(Enum):
    FOLLOWER = "FOLLOWER"
    CANDIDATE = "CANDIDATE"
    LEADER = "LEADER"


@dataclass
class LogEntry:
    """A Raft log entry (replicated across all nodes)."""
    term: int
    index: int
    command: str
    value: Any


class RaftNode:
    """Simplified Raft consensus node.

    Implements the core Raft concepts:
    - Leader election via randomized timeouts and majority vote
    - Log replication from leader to followers
    - Term-based distributed coordination

    In NewSQL systems, each shard/range is a Raft group:
    - CockroachDB: each Range (~512MB) is a Raft group
    - TiDB/TiKV: each Region (~96MB) is a Raft group
    """

    def __init__(self, node_id: str, cluster_size: int):
        self.node_id = node_id
        self.state = RaftState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = -1
        self.cluster_size = cluster_size
        self.votes_received: set = set()
        self.event_log: List[str] = []

    def start_election(self) -> int:
        """Transition to candidate and start an election."""
        self.current_term += 1
        self.state = RaftState.CANDIDATE
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}  # vote for self
        self.event_log.append(
            f"[Term {self.current_term}] {self.node_id} starts election"
        )
        return self.current_term

    def request_vote(self, candidate_id: str, candidate_term: int) -> bool:
        """Handle a RequestVote RPC from a candidate."""
        if candidate_term > self.current_term:
            self.current_term = candidate_term
            self.state = RaftState.FOLLOWER
            self.voted_for = None

        if candidate_term >= self.current_term and self.voted_for in (None, candidate_id):
            self.voted_for = candidate_id
            self.event_log.append(
                f"[Term {self.current_term}] {self.node_id} votes for {candidate_id}"
            )
            return True
        return False

    def receive_vote(self, voter_id: str) -> bool:
        """Receive a vote. Returns True if majority reached."""
        self.votes_received.add(voter_id)
        majority = self.cluster_size // 2 + 1
        if len(self.votes_received) >= majority and self.state == RaftState.CANDIDATE:
            self.state = RaftState.LEADER
            self.event_log.append(
                f"[Term {self.current_term}] {self.node_id} becomes LEADER "
                f"({len(self.votes_received)}/{self.cluster_size} votes)"
            )
            return True
        return False

    def append_entry(self, command: str, value: Any) -> Optional[LogEntry]:
        """Leader appends a new entry to its log."""
        if self.state != RaftState.LEADER:
            return None
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command=command,
            value=value
        )
        self.log.append(entry)
        self.event_log.append(
            f"[Term {self.current_term}] Leader appends: {command}={value} "
            f"(index={entry.index})"
        )
        return entry

    def replicate_entry(self, entry: LogEntry, leader_term: int) -> bool:
        """Follower receives an AppendEntries RPC from the leader."""
        if leader_term < self.current_term:
            return False
        self.current_term = leader_term
        self.state = RaftState.FOLLOWER
        self.log.append(entry)
        self.event_log.append(
            f"[Term {self.current_term}] {self.node_id} replicates: "
            f"{entry.command}={entry.value} (index={entry.index})"
        )
        return True

    def commit(self, index: int) -> None:
        """Mark log entries up to index as committed."""
        self.commit_index = index
        self.event_log.append(
            f"[Term {self.current_term}] {self.node_id} commits up to index={index}"
        )


def demonstrate_raft_consensus():
    """Demonstrate Raft leader election and log replication."""
    print("=" * 60)
    print("3. RAFT CONSENSUS PROTOCOL")
    print("=" * 60)
    print()

    cluster_size = 5
    nodes = {f"node_{i}": RaftNode(f"node_{i}", cluster_size)
             for i in range(1, cluster_size + 1)}
    node_list = list(nodes.values())

    # Leader election
    print("3.1 Leader Election")
    print("-" * 60)

    candidate = node_list[0]
    term = candidate.start_election()
    print(f"  {candidate.node_id} starts election for term {term}")

    # Collect votes from other nodes
    for node in node_list[1:]:
        granted = node.request_vote(candidate.node_id, term)
        if granted:
            is_leader = candidate.receive_vote(node.node_id)
            print(f"  {node.node_id} votes YES for {candidate.node_id}")
            if is_leader:
                print(f"  -> {candidate.node_id} wins election! "
                      f"(majority = {cluster_size // 2 + 1})")
                break
        else:
            print(f"  {node.node_id} votes NO")

    leader = candidate
    followers = [n for n in node_list if n.node_id != leader.node_id]

    # Log replication
    print(f"\n3.2 Log Replication (Leader: {leader.node_id})")
    print("-" * 60)

    # Leader receives client writes
    entries = [
        ("SET", "balance:1001=500"),
        ("SET", "balance:1002=300"),
        ("SET", "balance:1001=400"),
    ]

    for cmd, val in entries:
        entry = leader.append_entry(cmd, val)
        if entry is None:
            continue
        print(f"  Leader appends: [{entry.index}] {cmd} {val}")

        # Replicate to followers
        ack_count = 1  # leader counts as 1
        for follower in followers:
            success = follower.replicate_entry(entry, leader.current_term)
            if success:
                ack_count += 1
            # Commit once majority replicates
            if ack_count >= cluster_size // 2 + 1:
                leader.commit(entry.index)
                for f in followers:
                    f.commit(entry.index)
                break

        print(f"    Replicated to {ack_count}/{cluster_size} nodes -> committed")

    # Verify all nodes have same log
    print(f"\n3.3 Log Consistency Check")
    print("-" * 60)
    for node in node_list:
        log_entries = [(e.index, e.command, e.value) for e in node.log]
        print(f"  {node.node_id} ({node.state.value:9}): "
              f"commit_index={node.commit_index}, log={log_entries}")

    # Demonstrate leader failure + re-election
    print(f"\n3.4 Leader Failure and Re-Election")
    print("-" * 60)
    print(f"  {leader.node_id} fails (simulated)!")
    leader.state = RaftState.FOLLOWER  # simulate crash

    new_candidate = followers[1]  # node_3 tries to become leader
    new_term = new_candidate.start_election()
    print(f"  {new_candidate.node_id} starts election for term {new_term}")

    vote_count = 0
    for node in node_list:
        if node.node_id == new_candidate.node_id:
            continue
        if node.node_id == leader.node_id:
            print(f"  {node.node_id} is down (no response)")
            continue
        granted = node.request_vote(new_candidate.node_id, new_term)
        if granted:
            new_candidate.receive_vote(node.node_id)
            vote_count += 1
            print(f"  {node.node_id} votes YES")
            if new_candidate.state == RaftState.LEADER:
                print(f"  -> {new_candidate.node_id} becomes new LEADER in term {new_term}!")
                break

    print()


# ============================================================
# 4. HYBRID LOGICAL CLOCKS (HLC)
# ============================================================

@dataclass
class HLCTimestamp:
    """Hybrid Logical Clock timestamp.

    Used by CockroachDB instead of Spanner's TrueTime.
    Combines wall clock time with a logical counter to ensure
    causally consistent ordering without GPS/atomic clocks.
    """
    physical: int   # wall clock component (milliseconds)
    logical: int    # logical counter
    node_id: str    # originating node

    def __lt__(self, other: 'HLCTimestamp') -> bool:
        if self.physical != other.physical:
            return self.physical < other.physical
        return self.logical < other.logical

    def __le__(self, other: 'HLCTimestamp') -> bool:
        return self == other or self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HLCTimestamp):
            return False
        return self.physical == other.physical and self.logical == other.logical

    def __repr__(self) -> str:
        return f"HLC({self.physical}.{self.logical}@{self.node_id})"


class HybridLogicalClock:
    """Hybrid Logical Clock implementation.

    Rules (from the CockroachDB / Lamport paper):
    1. Local event: physical = max(physical, wall_clock); logical++
    2. Send: attach current HLC to message
    3. Receive(msg_hlc): physical = max(physical, msg.physical, wall_clock)
       - if physical unchanged: logical = max(logical, msg.logical) + 1
       - else: logical = 0
    """

    def __init__(self, node_id: str, wall_clock_offset: int = 0):
        self.node_id = node_id
        self._physical = 0
        self._logical = 0
        self._wall_clock_offset = wall_clock_offset  # simulate clock skew

    def _wall_clock(self) -> int:
        """Simulated wall clock (milliseconds)."""
        return int(time.time() * 1000) + self._wall_clock_offset

    def now(self) -> HLCTimestamp:
        """Generate a new HLC timestamp for a local event."""
        wall = self._wall_clock()
        if wall > self._physical:
            self._physical = wall
            self._logical = 0
        else:
            self._logical += 1
        return HLCTimestamp(self._physical, self._logical, self.node_id)

    def receive(self, msg_ts: HLCTimestamp) -> HLCTimestamp:
        """Update HLC upon receiving a message with the sender's timestamp."""
        wall = self._wall_clock()
        old_physical = self._physical

        self._physical = max(self._physical, msg_ts.physical, wall)

        if self._physical == old_physical and self._physical == msg_ts.physical:
            self._logical = max(self._logical, msg_ts.logical) + 1
        elif self._physical == old_physical:
            self._logical += 1
        elif self._physical == msg_ts.physical:
            self._logical = msg_ts.logical + 1
        else:
            self._logical = 0

        return HLCTimestamp(self._physical, self._logical, self.node_id)


def demonstrate_hybrid_logical_clocks():
    """Demonstrate HLC for causal ordering without synchronized clocks."""
    print("=" * 60)
    print("4. HYBRID LOGICAL CLOCKS (HLC)")
    print("=" * 60)
    print()

    print("4.1 Basic HLC Ordering (Single Node)")
    print("-" * 60)

    hlc_a = HybridLogicalClock("node_A")

    timestamps = []
    for i in range(5):
        ts = hlc_a.now()
        timestamps.append(ts)
        print(f"  Event {i+1}: {ts}")

    # Verify monotonically increasing
    is_ordered = all(timestamps[i] < timestamps[i+1]
                     for i in range(len(timestamps) - 1))
    print(f"\n  Monotonically increasing: {is_ordered}")

    print("\n4.2 Cross-Node Causal Ordering")
    print("-" * 60)

    hlc_a = HybridLogicalClock("node_A", wall_clock_offset=0)
    hlc_b = HybridLogicalClock("node_B", wall_clock_offset=50)  # 50ms ahead

    # Node A generates a timestamp
    ts_a1 = hlc_a.now()
    print(f"  node_A event:        {ts_a1}")

    # Node A sends message to Node B
    ts_b1 = hlc_b.receive(ts_a1)
    print(f"  node_B receives msg: {ts_b1}")
    print(f"  Causal order maintained: ts_a1 < ts_b1 = {ts_a1 < ts_b1}")

    # Node B generates a local event
    ts_b2 = hlc_b.now()
    print(f"  node_B local event:  {ts_b2}")

    # Node B sends back to Node A
    ts_a2 = hlc_a.receive(ts_b2)
    print(f"  node_A receives msg: {ts_a2}")
    print(f"  Causal order: ts_a1 < ts_b1 < ts_b2 < ts_a2 = "
          f"{ts_a1 < ts_b1 < ts_b2 < ts_a2}")

    print("\n4.3 HLC vs Spanner TrueTime")
    print("-" * 60)
    print("  Spanner TrueTime:")
    print("    - Uses GPS + atomic clocks for bounded uncertainty")
    print("    - TT.now() returns [earliest, latest] interval")
    print("    - Commit-wait: waits for uncertainty to pass")
    print("    - Guarantees external consistency (linearizability)")
    print("    - Requires specialized hardware")
    print()
    print("  CockroachDB HLC:")
    print("    - Uses NTP-synchronized wall clocks + logical counters")
    print("    - No special hardware needed")
    print("    - Enforces max clock offset (default 500ms)")
    print("    - Uses uncertainty intervals + read restarts")
    print("    - Achieves serializable isolation")

    print()


# ============================================================
# 5. VECTOR SIMILARITY SEARCH
# ============================================================

def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorIndex:
    """Simple vector database with brute-force and IVF search.

    Demonstrates concepts used by vector databases (Pinecone, Weaviate,
    Milvus, pgvector) for semantic similarity search in AI applications.

    Brute-force: O(n) scan, exact results
    IVF (Inverted File Index): approximate, partitions vectors into clusters
    """

    def __init__(self):
        self._vectors: List[Tuple[str, List[float], Dict]] = []  # (id, vec, metadata)
        self._clusters: Dict[int, List[int]] = {}  # cluster_id -> vector indices
        self._centroids: List[List[float]] = []

    def insert(self, vec_id: str, vector: List[float],
               metadata: Optional[Dict] = None) -> None:
        self._vectors.append((vec_id, vector, metadata or {}))

    def search_brute_force(self, query: List[float], top_k: int = 5,
                           metric: str = "cosine") -> List[Tuple[str, float, Dict]]:
        """Exact nearest neighbor search (brute-force)."""
        distances = []
        for vec_id, vec, meta in self._vectors:
            if metric == "cosine":
                sim = cosine_similarity(query, vec)
                distances.append((vec_id, sim, meta))
            else:
                dist = euclidean_distance(query, vec)
                distances.append((vec_id, dist, meta))

        reverse = (metric == "cosine")  # higher similarity = better
        distances.sort(key=lambda x: x[1], reverse=reverse)
        return distances[:top_k]

    def build_ivf_index(self, n_clusters: int = 4, n_iterations: int = 10) -> None:
        """Build an IVF index using simple k-means clustering."""
        if not self._vectors:
            return

        dim = len(self._vectors[0][1])
        # Initialize centroids randomly
        indices = random.sample(range(len(self._vectors)), min(n_clusters, len(self._vectors)))
        self._centroids = [list(self._vectors[i][1]) for i in indices]

        for _ in range(n_iterations):
            # Assign vectors to nearest centroid
            assignments: Dict[int, List[int]] = defaultdict(list)
            for idx, (_, vec, _) in enumerate(self._vectors):
                best_cluster = min(range(len(self._centroids)),
                                   key=lambda c: euclidean_distance(vec, self._centroids[c]))
                assignments[best_cluster].append(idx)

            # Update centroids
            for c_idx in range(len(self._centroids)):
                if c_idx not in assignments:
                    continue
                cluster_vecs = [self._vectors[i][1] for i in assignments[c_idx]]
                self._centroids[c_idx] = [
                    sum(v[d] for v in cluster_vecs) / len(cluster_vecs)
                    for d in range(dim)
                ]

            self._clusters = dict(assignments)

    def search_ivf(self, query: List[float], top_k: int = 5,
                   n_probes: int = 2) -> List[Tuple[str, float, Dict]]:
        """Approximate nearest neighbor search using IVF index.

        Only searches n_probes closest clusters instead of all vectors.
        Trade-off: faster but may miss some true nearest neighbors.
        """
        if not self._centroids:
            return self.search_brute_force(query, top_k)

        # Find closest clusters
        cluster_dists = [(c_idx, euclidean_distance(query, centroid))
                         for c_idx, centroid in enumerate(self._centroids)]
        cluster_dists.sort(key=lambda x: x[1])
        probe_clusters = [c_idx for c_idx, _ in cluster_dists[:n_probes]]

        # Search only within those clusters
        candidates = []
        for c_idx in probe_clusters:
            for vec_idx in self._clusters.get(c_idx, []):
                vec_id, vec, meta = self._vectors[vec_idx]
                sim = cosine_similarity(query, vec)
                candidates.append((vec_id, sim, meta))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]


def demonstrate_vector_search():
    """Demonstrate vector similarity search concepts."""
    print("=" * 60)
    print("5. VECTOR SIMILARITY SEARCH")
    print("=" * 60)
    print()

    random.seed(42)
    index = VectorIndex()

    # Simulate document embeddings (normally from a neural network)
    documents = [
        ("doc_1", "Python programming tutorial",
         [0.9, 0.1, 0.2, 0.0, 0.8, 0.1, 0.3, 0.0]),
        ("doc_2", "Machine learning basics",
         [0.2, 0.9, 0.8, 0.1, 0.3, 0.7, 0.1, 0.0]),
        ("doc_3", "Deep learning neural networks",
         [0.1, 0.8, 0.9, 0.2, 0.2, 0.8, 0.0, 0.1]),
        ("doc_4", "Python data analysis with pandas",
         [0.8, 0.3, 0.3, 0.1, 0.7, 0.2, 0.5, 0.1]),
        ("doc_5", "SQL database queries",
         [0.3, 0.1, 0.1, 0.9, 0.4, 0.0, 0.2, 0.8]),
        ("doc_6", "NoSQL document databases",
         [0.2, 0.2, 0.1, 0.8, 0.3, 0.1, 0.1, 0.9]),
        ("doc_7", "Neural network training techniques",
         [0.1, 0.7, 0.9, 0.1, 0.1, 0.9, 0.0, 0.0]),
        ("doc_8", "Python web development Flask",
         [0.8, 0.0, 0.1, 0.2, 0.9, 0.0, 0.6, 0.1]),
    ]

    print("5.1 Insert Document Embeddings")
    print("-" * 60)
    for doc_id, title, vector in documents:
        index.insert(doc_id, vector, {"title": title})
        print(f"  {doc_id}: {title}")

    # Semantic search
    print("\n5.2 Brute-Force Search (Exact)")
    print("-" * 60)

    query_vec = [0.15, 0.85, 0.9, 0.1, 0.2, 0.8, 0.05, 0.05]
    print("  Query: 'deep learning concepts' (simulated embedding)")
    print("  Top-3 results (cosine similarity):")

    results = index.search_brute_force(query_vec, top_k=3)
    for rank, (doc_id, sim, meta) in enumerate(results, 1):
        print(f"    {rank}. {meta['title']} (sim={sim:.4f})")

    # IVF search
    print("\n5.3 IVF Index Search (Approximate)")
    print("-" * 60)
    index.build_ivf_index(n_clusters=3, n_iterations=5)

    print(f"  Built IVF index with {len(index._centroids)} clusters")
    for c_idx, members in index._clusters.items():
        member_ids = [index._vectors[i][0] for i in members]
        print(f"    Cluster {c_idx}: {member_ids}")

    print(f"\n  Same query with IVF (n_probes=2):")
    ivf_results = index.search_ivf(query_vec, top_k=3, n_probes=2)
    for rank, (doc_id, sim, meta) in enumerate(ivf_results, 1):
        print(f"    {rank}. {meta['title']} (sim={sim:.4f})")

    # Compare brute-force vs IVF
    print("\n5.4 Brute-Force vs IVF Trade-offs")
    print("-" * 60)
    print("  Brute-Force: O(n) — exact results, slow for large datasets")
    print("  IVF:         O(n/k * probes) — approximate, much faster")
    print("  HNSW:        O(log n) — graph-based, best for high recall")
    print()
    print("  Real-world vector databases (Pinecone, Weaviate, Milvus)")
    print("  use HNSW or IVF-PQ for billion-scale vector search")

    print()


# ============================================================
# 6. TIME-SERIES DATA WITH DOWNSAMPLING
# ============================================================

class TimeSeriesStore:
    """Simple time-series store with downsampling.

    Demonstrates concepts from time-series databases (InfluxDB, TimescaleDB):
    - Append-only writes (optimized for sequential inserts)
    - Time-range queries
    - Downsampling (reduce resolution for older data)
    """

    def __init__(self):
        # metric_name -> list of (timestamp, value)
        self._data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def write(self, metric: str, timestamp: float, value: float) -> None:
        """Append a data point (append-only, like time-series DBs)."""
        self._data[metric].append((timestamp, value))

    def query_range(self, metric: str, start: float,
                    end: float) -> List[Tuple[float, float]]:
        """Query data points within a time range."""
        return [(ts, val) for ts, val in self._data.get(metric, [])
                if start <= ts <= end]

    def downsample(self, metric: str, bucket_size: float,
                   agg: str = "avg") -> List[Tuple[float, float]]:
        """Downsample data by averaging within time buckets.

        This is a key optimization in time-series DBs:
        - Keep high-resolution data for recent periods
        - Downsample older data to save storage
        """
        points = self._data.get(metric, [])
        if not points:
            return []

        buckets: Dict[float, List[float]] = defaultdict(list)
        for ts, val in points:
            bucket_key = (ts // bucket_size) * bucket_size
            buckets[bucket_key].append(val)

        result = []
        for bucket_ts in sorted(buckets.keys()):
            values = buckets[bucket_ts]
            if agg == "avg":
                agg_val = sum(values) / len(values)
            elif agg == "max":
                agg_val = max(values)
            elif agg == "min":
                agg_val = min(values)
            elif agg == "sum":
                agg_val = sum(values)
            else:
                agg_val = sum(values) / len(values)
            result.append((bucket_ts, round(agg_val, 2)))

        return result

    def point_count(self, metric: str) -> int:
        return len(self._data.get(metric, []))


def demonstrate_time_series():
    """Demonstrate time-series storage and downsampling."""
    print("=" * 60)
    print("6. TIME-SERIES DATA AND DOWNSAMPLING")
    print("=" * 60)
    print()

    store = TimeSeriesStore()
    random.seed(42)

    # Simulate CPU usage data (1 point per second for 60 seconds)
    print("6.1 Ingesting Time-Series Data (CPU Usage)")
    print("-" * 60)

    base_time = 1700000000.0  # arbitrary epoch
    for i in range(60):
        ts = base_time + i
        cpu = 40 + 20 * math.sin(i / 10.0) + random.uniform(-5, 5)
        store.write("cpu_usage", ts, round(cpu, 2))

    print(f"  Ingested {store.point_count('cpu_usage')} data points (1/sec)")

    # Range query
    print("\n6.2 Range Query (First 10 Seconds)")
    print("-" * 60)
    points = store.query_range("cpu_usage", base_time, base_time + 9)
    for ts, val in points:
        offset = ts - base_time
        bar = "#" * int(val / 2)
        print(f"  t+{offset:4.0f}s: {val:6.2f}% {bar}")

    # Downsampling
    print("\n6.3 Downsampling (10-Second Buckets)")
    print("-" * 60)
    original_count = store.point_count("cpu_usage")
    downsampled = store.downsample("cpu_usage", bucket_size=10.0, agg="avg")

    print(f"  Original:     {original_count} points")
    print(f"  Downsampled:  {len(downsampled)} points (10x reduction)")
    print()

    for ts, val in downsampled:
        offset = ts - base_time
        bar = "#" * int(val / 2)
        print(f"  t+{offset:4.0f}s: avg={val:6.2f}% {bar}")

    # Different aggregations
    print("\n6.4 Different Aggregation Functions")
    print("-" * 60)
    for agg in ["avg", "max", "min"]:
        ds = store.downsample("cpu_usage", bucket_size=20.0, agg=agg)
        values = [f"{val:.1f}" for _, val in ds]
        print(f"  {agg:4}: {values}")

    print("\n6.5 Time-Series DB Optimizations")
    print("-" * 60)
    print("  - Append-only writes (no random updates)")
    print("  - Time-based partitioning (recent data in hot storage)")
    print("  - Columnar compression (delta encoding, gorilla encoding)")
    print("  - Automatic downsampling retention policies")
    print("  - Examples: InfluxDB, TimescaleDB, Prometheus, QuestDB")

    print()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║      NewSQL AND MODERN TRENDS — Simulations                  ║
║  2PC, MVCC, Raft, HLC, Vector Search, Time-Series            ║
╚══════════════════════════════════════════════════════════════╝
""")

    demonstrate_two_phase_commit()
    demonstrate_mvcc()
    demonstrate_raft_consensus()
    demonstrate_hybrid_logical_clocks()
    demonstrate_vector_search()
    demonstrate_time_series()

    print("=" * 60)
    print("SUMMARY: NewSQL AND MODERN DATABASE TRENDS")
    print("=" * 60)
    print("Key takeaways:")
    print("  1. 2PC enables distributed transactions (atomicity across shards)")
    print("  2. MVCC allows concurrent reads/writes without blocking")
    print("  3. Raft consensus provides fault-tolerant replication")
    print("  4. HLC enables causal ordering without specialized hardware")
    print("  5. Vector search powers semantic similarity for AI workloads")
    print("  6. Time-series DBs optimize for append-heavy temporal data")
    print()
    print("NewSQL = ACID + SQL + horizontal scalability")
    print("  - Spanner: TrueTime + Multi-Paxos (external consistency)")
    print("  - CockroachDB: HLC + Raft (serializable isolation)")
    print("  - TiDB: Timestamp Oracle + Raft (HTAP: OLTP + OLAP)")
    print("=" * 60)
