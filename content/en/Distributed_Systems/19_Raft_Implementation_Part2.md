# Lesson 19: Raft Implementation Part 2 — Membership Changes, Log Compaction, and Linearizability

[Overview](./00_Overview.md) | [Previous: Raft Implementation Part 1](./18_Raft_Implementation_Part1.md) | [Next: Distributed Hash Tables](./20_Distributed_Hash_Tables.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement safe cluster membership changes using joint consensus and single-server changes
2. Build log compaction with snapshots to bound memory and disk usage
3. Design the InstallSnapshot RPC for slow followers that have fallen behind
4. Implement linearizable reads using ReadIndex and LeaseRead optimizations
5. Tune Raft performance for throughput, latency, and resource consumption

---

## Table of Contents

1. [Membership Changes Overview](#1-membership-changes-overview)
2. [Single-Server Membership Changes](#2-single-server-membership-changes)
3. [Joint Consensus](#3-joint-consensus)
4. [Log Compaction and Snapshots](#4-log-compaction-and-snapshots)
5. [InstallSnapshot RPC](#5-installsnapshot-rpc)
6. [Linearizable Reads](#6-linearizable-reads)
7. [Performance Tuning](#7-performance-tuning)
8. [Batching and Pipelining](#8-batching-and-pipelining)
9. [Complete Implementation](#9-complete-implementation)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Membership Changes Overview

### 1.1 The Problem

Changing the cluster membership (adding or removing nodes) while the cluster is running is one of the hardest parts of Raft. The fundamental danger is that during a transition, two disjoint majorities could form — one in the old configuration and one in the new — leading to two leaders and a safety violation.

```
Old config: {A, B, C}     Majority = 2
New config: {A, B, C, D, E}  Majority = 3

Danger period:
  {A, B} form a majority in old config → elect leader
  {C, D, E} form a majority in new config → elect leader
  TWO LEADERS simultaneously!
```

### 1.2 Two Approaches

Raft offers two solutions:

| Approach | Complexity | Safety | Availability |
|----------|------------|--------|-------------|
| Single-server changes | Simple | Safe (one at a time) | Brief unavailability |
| Joint consensus | Complex | Safe (arbitrary changes) | No unavailability |

---

## 2. Single-Server Membership Changes

### 2.1 Algorithm

The simplest approach: add or remove one server at a time. This is safe because any two majorities of configurations that differ by at most one server must overlap.

```python
import time
import json
import random
import hashlib
from typing import Optional, Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MembershipChangeType(Enum):
    ADD_SERVER = "add_server"
    REMOVE_SERVER = "remove_server"


@dataclass
class ConfigurationEntry:
    """A configuration change entry in the Raft log."""
    change_type: MembershipChangeType
    server_id: str
    old_config: set
    new_config: set


class RaftMembership:
    """
    Manages Raft cluster membership with single-server changes.

    Safety invariant: Only one membership change can be in-flight
    (uncommitted) at a time. A new change cannot begin until the
    previous one has been committed.
    """

    def __init__(self, initial_members: set[str]):
        self.current_config: set[str] = set(initial_members)
        self.pending_config: Optional[set[str]] = None
        self.change_log: list[ConfigurationEntry] = []
        self.committed_config_index: int = 0

    def propose_add(self, server_id: str) -> Optional[ConfigurationEntry]:
        """
        Propose adding a server to the cluster.

        Returns None if a change is already pending.
        """
        if self.pending_config is not None:
            print(f"Cannot add {server_id}: change already pending")
            return None

        if server_id in self.current_config:
            print(f"Cannot add {server_id}: already a member")
            return None

        new_config = self.current_config | {server_id}
        entry = ConfigurationEntry(
            change_type=MembershipChangeType.ADD_SERVER,
            server_id=server_id,
            old_config=set(self.current_config),
            new_config=new_config,
        )

        self.pending_config = new_config
        self.change_log.append(entry)

        print(f"Proposed: ADD {server_id}")
        print(f"  Old config: {sorted(self.current_config)}")
        print(f"  New config: {sorted(new_config)}")
        print(f"  Old majority: {len(self.current_config) // 2 + 1}")
        print(f"  New majority: {len(new_config) // 2 + 1}")

        return entry

    def propose_remove(self, server_id: str) -> Optional[ConfigurationEntry]:
        """Propose removing a server from the cluster."""
        if self.pending_config is not None:
            print(f"Cannot remove {server_id}: change already pending")
            return None

        if server_id not in self.current_config:
            print(f"Cannot remove {server_id}: not a member")
            return None

        new_config = self.current_config - {server_id}
        entry = ConfigurationEntry(
            change_type=MembershipChangeType.REMOVE_SERVER,
            server_id=server_id,
            old_config=set(self.current_config),
            new_config=new_config,
        )

        self.pending_config = new_config
        self.change_log.append(entry)
        return entry

    def commit_change(self):
        """Commit the pending configuration change."""
        if self.pending_config is None:
            return

        old = self.current_config
        self.current_config = self.pending_config
        self.pending_config = None
        self.committed_config_index += 1

        print(f"Committed config change #{self.committed_config_index}")
        print(f"  {sorted(old)} → {sorted(self.current_config)}")

    def abort_change(self):
        """Abort a pending change (e.g., leadership lost)."""
        self.pending_config = None
        if self.change_log:
            self.change_log.pop()

    def majority_size(self, config: Optional[set] = None) -> int:
        """Calculate majority size for a configuration."""
        c = config or self.current_config
        return len(c) // 2 + 1

    def verify_single_overlap(self) -> bool:
        """
        Verify that old and new configs (differing by 1) always overlap.

        For any two majorities Q_old and Q_new, Q_old ∩ Q_new ≠ ∅.
        """
        if self.pending_config is None:
            return True

        old = self.current_config
        new = self.pending_config
        old_majority = len(old) // 2 + 1
        new_majority = len(new) // 2 + 1

        # The overlap is guaranteed when |old Δ new| = 1
        diff = old.symmetric_difference(new)
        if len(diff) > 1:
            print(f"WARNING: Config differs by {len(diff)} servers, not 1!")
            return False

        # Formal check: enumerate all possible majority pairs
        from itertools import combinations
        for q_old in combinations(old, old_majority):
            for q_new in combinations(new, new_majority):
                if not set(q_old) & set(q_new):
                    print(f"VIOLATION: {set(q_old)} and {set(q_new)} don't overlap!")
                    return False

        return True


def demonstrate_membership_changes():
    """Demonstrate safe single-server membership changes."""
    print("=== Single-Server Membership Changes ===\n")

    membership = RaftMembership({"A", "B", "C"})

    # Add server D
    membership.propose_add("D")
    assert membership.verify_single_overlap()
    membership.commit_change()

    # Add server E
    membership.propose_add("E")
    assert membership.verify_single_overlap()
    membership.commit_change()

    # Remove server A
    membership.propose_remove("A")
    assert membership.verify_single_overlap()
    membership.commit_change()

    print(f"\nFinal config: {sorted(membership.current_config)}")
    print(f"Changes: {membership.committed_config_index}")


demonstrate_membership_changes()
```

### 2.2 Catch-Up Phase

Before adding a new server, it must catch up on the log. Otherwise, it could take a long time for the new server to receive all historical entries, during which commits are slower (the new server cannot contribute to the majority).

```python
class ServerCatchUp:
    """
    Manages the catch-up phase for a new server joining the cluster.

    The new server must replicate all existing log entries before
    it becomes a voting member. During catch-up, it is a non-voting
    learner that receives AppendEntries but does not count toward majority.
    """

    def __init__(self, leader_log_length: int, rounds_threshold: int = 10):
        self.leader_log_length = leader_log_length
        self.rounds_threshold = rounds_threshold
        self.rounds: list[dict] = []
        self.start_time = time.time()

    def record_round(self, entries_behind: int):
        """Record a replication round result."""
        self.rounds.append({
            "entries_behind": entries_behind,
            "timestamp": time.time(),
        })

    def is_caught_up(self) -> bool:
        """
        Determine if the new server is sufficiently caught up.

        The server is ready when the number of entries behind is
        small enough to be replicated within one election timeout.
        """
        if len(self.rounds) < self.rounds_threshold:
            return False

        # Check last few rounds: entries_behind should be decreasing
        recent = self.rounds[-self.rounds_threshold:]
        last_behind = recent[-1]["entries_behind"]

        # Ready if within one heartbeat interval worth of entries
        return last_behind <= 10  # Heuristic: within 10 entries

    def report(self) -> dict:
        """Generate a catch-up progress report."""
        if not self.rounds:
            return {"status": "starting", "entries_behind": self.leader_log_length}

        current_behind = self.rounds[-1]["entries_behind"]
        elapsed = time.time() - self.start_time
        progress = 1.0 - (current_behind / max(self.leader_log_length, 1))

        return {
            "status": "catching_up" if not self.is_caught_up() else "ready",
            "entries_behind": current_behind,
            "rounds_completed": len(self.rounds),
            "progress_pct": round(progress * 100, 1),
            "elapsed_seconds": round(elapsed, 2),
        }


def demonstrate_catch_up():
    """Demonstrate the catch-up process for a new server."""
    print("=== Server Catch-Up Phase ===\n")

    leader_log_length = 10000
    catch_up = ServerCatchUp(leader_log_length)

    # Simulate catch-up rounds
    entries_behind = leader_log_length
    round_num = 0

    while not catch_up.is_caught_up():
        # Each round, the follower catches up some entries
        # but the leader may have added new entries too
        caught_up = min(entries_behind, random.randint(500, 2000))
        new_entries = random.randint(0, 50)  # Leader gets new writes
        entries_behind = entries_behind - caught_up + new_entries
        entries_behind = max(0, entries_behind)

        catch_up.record_round(entries_behind)
        round_num += 1

        if round_num % 5 == 0:
            report = catch_up.report()
            print(f"  Round {round_num}: {report['progress_pct']}% caught up, "
                  f"{report['entries_behind']} behind")

    report = catch_up.report()
    print(f"\n  Catch-up complete after {report['rounds_completed']} rounds")
    print(f"  Server is ready to join as a voting member")


demonstrate_catch_up()
```

---

## 3. Joint Consensus

### 3.1 The Two-Phase Approach

Joint consensus allows arbitrary configuration changes by using a transitional configuration that requires majorities from BOTH the old and new configurations:

```
Phase 1: C_old → C_old,new (joint configuration)
  - Log entry: [C_old,new]
  - Decisions require majority of C_old AND majority of C_new

Phase 2: C_old,new → C_new
  - Log entry: [C_new]
  - Once committed, only C_new matters
```

```python
class JointConsensus:
    """
    Implements joint consensus for arbitrary membership changes.

    During the joint phase, both the old and new configurations
    must independently form majorities for any decision.
    """

    def __init__(self, config: set[str]):
        self.old_config: set[str] = set(config)
        self.new_config: Optional[set[str]] = None
        self.phase: str = "stable"  # stable, joint, transitioning

    def start_change(self, new_config: set[str]):
        """Begin a membership change to new_config."""
        assert self.phase == "stable", "Cannot start change during transition"
        self.new_config = set(new_config)
        self.phase = "joint"

        print(f"Entering joint consensus:")
        print(f"  C_old: {sorted(self.old_config)}")
        print(f"  C_new: {sorted(self.new_config)}")
        print(f"  Need majority of BOTH for any decision")

    def check_majority(self, votes: set[str]) -> bool:
        """
        Check if a set of votes forms a sufficient majority.

        In joint phase: need majority of BOTH old and new configs.
        In stable phase: need majority of current config.
        """
        if self.phase == "stable":
            needed = len(self.old_config) // 2 + 1
            have = len(votes & self.old_config)
            return have >= needed

        elif self.phase == "joint":
            old_needed = len(self.old_config) // 2 + 1
            old_have = len(votes & self.old_config)

            new_needed = len(self.new_config) // 2 + 1
            new_have = len(votes & self.new_config)

            return old_have >= old_needed and new_have >= new_needed

        return False

    def commit_joint(self):
        """
        The joint configuration entry has been committed.
        Now transition to the new configuration.
        """
        assert self.phase == "joint"
        self.phase = "transitioning"
        print(f"Joint config committed. Transitioning to C_new...")

    def commit_new(self):
        """
        The new configuration entry has been committed.
        Transition is complete.
        """
        assert self.phase == "transitioning"
        self.old_config = self.new_config
        self.new_config = None
        self.phase = "stable"
        print(f"Transition complete. Config: {sorted(self.old_config)}")


def demonstrate_joint_consensus():
    """Demonstrate joint consensus for a multi-server change."""
    print("=== Joint Consensus ===\n")

    jc = JointConsensus({"A", "B", "C"})

    # Change from {A,B,C} to {B,C,D,E}
    jc.start_change({"B", "C", "D", "E"})

    # Test majority checks during joint phase
    test_cases = [
        ({"A", "B", "C"}, "old majority only"),
        ({"B", "C", "D"}, "old majority + new majority"),
        ({"C", "D", "E"}, "new majority only"),
        ({"A", "B", "D", "E"}, "old majority + new majority"),
        ({"B", "C", "D", "E"}, "old majority + new majority"),
    ]

    for votes, desc in test_cases:
        result = jc.check_majority(votes)
        print(f"  Votes {sorted(votes):20s} ({desc}): {'PASS' if result else 'FAIL'}")

    # Complete the transition
    jc.commit_joint()
    jc.commit_new()


demonstrate_joint_consensus()
```

---

## 4. Log Compaction and Snapshots

### 4.1 The Problem

Without compaction, the Raft log grows without bound. This wastes disk space, increases startup time (replaying the entire log), and slows down replication for new followers.

```
Without compaction:
  Log: [e1] [e2] [e3] ... [e10000] [e10001] ...
  ← All entries kept forever →

With snapshots:
  Snapshot at index 8000:       Log tail:
  ┌────────────────────┐       [e8001] [e8002] ... [e10001]
  │ state_machine data │       ← Only recent entries kept →
  │ last_included_index│
  │ last_included_term │
  └────────────────────┘
```

### 4.2 Snapshot Implementation

```python
@dataclass
class Snapshot:
    """A point-in-time snapshot of the Raft state machine."""
    last_included_index: int
    last_included_term: int
    data: dict  # Serialized state machine state
    config: set  # Cluster configuration at snapshot time
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self.size_bytes = len(json.dumps(self.data))


class LogCompactor:
    """
    Manages log compaction via snapshots.

    Snapshots are triggered when the log exceeds a configurable size.
    The snapshot captures the state machine state at a committed index,
    and all log entries up to that index are discarded.
    """

    # Snapshot when log exceeds this many entries
    SNAPSHOT_THRESHOLD = 1000
    # Keep at least this many entries after snapshot (for slow followers)
    MIN_LOG_RETENTION = 100

    def __init__(self):
        self.log: list[dict] = []
        self.log_offset: int = 0  # Index of first entry in log
        self.snapshots: list[Snapshot] = []
        self.current_snapshot: Optional[Snapshot] = None
        self.state_machine: dict = {}
        self.commit_index: int = 0

    def append(self, entry: dict):
        """Append an entry to the log."""
        self.log.append(entry)

    def apply_up_to(self, index: int):
        """Apply log entries up to the given index to the state machine."""
        while self.commit_index < index:
            self.commit_index += 1
            relative_idx = self.commit_index - self.log_offset - 1
            if 0 <= relative_idx < len(self.log):
                entry = self.log[relative_idx]
                cmd = entry.get("command", {})
                if cmd.get("op") == "put":
                    self.state_machine[cmd["key"]] = cmd["value"]
                elif cmd.get("op") == "delete":
                    self.state_machine.pop(cmd.get("key"), None)

    def should_snapshot(self) -> bool:
        """Check if the log is large enough to warrant a snapshot."""
        return len(self.log) > self.SNAPSHOT_THRESHOLD

    def create_snapshot(self) -> Snapshot:
        """
        Create a snapshot at the current commit index.

        Steps:
        1. Serialize the state machine state
        2. Record the last included index and term
        3. Discard log entries before the snapshot
        4. Save snapshot to stable storage
        """
        assert self.commit_index > 0, "Cannot snapshot with no committed entries"

        # Find the term of the committed entry
        relative_idx = self.commit_index - self.log_offset - 1
        last_term = self.log[relative_idx]["term"]

        snapshot = Snapshot(
            last_included_index=self.commit_index,
            last_included_term=last_term,
            data=dict(self.state_machine),
            config=set(),
        )

        # Discard log entries that are included in the snapshot
        # Keep MIN_LOG_RETENTION entries for slow followers
        entries_to_discard = max(0, relative_idx + 1 - self.MIN_LOG_RETENTION)
        if entries_to_discard > 0:
            self.log = self.log[entries_to_discard:]
            self.log_offset += entries_to_discard

        self.current_snapshot = snapshot
        self.snapshots.append(snapshot)

        return snapshot

    def restore_from_snapshot(self, snapshot: Snapshot):
        """
        Restore state from a received snapshot.

        Called when a follower receives an InstallSnapshot RPC because
        it has fallen too far behind for log-based replication.
        """
        self.current_snapshot = snapshot
        self.state_machine = dict(snapshot.data)
        self.commit_index = snapshot.last_included_index
        self.log_offset = snapshot.last_included_index

        # Discard all log entries before the snapshot
        self.log = [
            e for e in self.log
            if e.get("index", 0) > snapshot.last_included_index
        ]

    def stats(self) -> dict:
        """Return compaction statistics."""
        return {
            "log_length": len(self.log),
            "log_offset": self.log_offset,
            "commit_index": self.commit_index,
            "snapshots_taken": len(self.snapshots),
            "state_machine_keys": len(self.state_machine),
            "snapshot_size_bytes": self.current_snapshot.size_bytes if self.current_snapshot else 0,
        }


def demonstrate_log_compaction():
    """Demonstrate log compaction with snapshots."""
    print("=== Log Compaction ===\n")

    compactor = LogCompactor()
    compactor.SNAPSHOT_THRESHOLD = 100  # Lower for demo

    # Generate 500 log entries
    for i in range(1, 501):
        compactor.append({
            "term": 1,
            "index": i,
            "command": {"op": "put", "key": f"key_{i % 50}", "value": f"val_{i}"},
        })

        # Apply entries as committed
        compactor.apply_up_to(i)

        # Check if snapshot needed
        if compactor.should_snapshot():
            snapshot = compactor.create_snapshot()
            print(f"  Snapshot at index {snapshot.last_included_index}: "
                  f"log={len(compactor.log)} entries, "
                  f"state={len(compactor.state_machine)} keys, "
                  f"size={snapshot.size_bytes} bytes")

    print(f"\nFinal stats: {compactor.stats()}")


demonstrate_log_compaction()
```

---

## 5. InstallSnapshot RPC

### 5.1 When Snapshots Are Needed

A leader sends an InstallSnapshot RPC when a follower's `nextIndex` points to a log entry that has already been compacted (discarded after snapshotting):

```
Leader:  Snapshot[...index 5000] | [5001] [5002] ... [6000]
                                   ^
Follower needs index 3000 ────────┘ Already compacted!

Leader must send its snapshot instead of log entries.
```

### 5.2 Chunked Transfer

Large snapshots are sent in chunks to avoid blocking the network:

```python
@dataclass
class SnapshotChunk:
    """A chunk of a snapshot being transferred."""
    term: int
    leader_id: str
    last_included_index: int
    last_included_term: int
    offset: int
    data: bytes
    done: bool


class SnapshotTransfer:
    """
    Manages chunked snapshot transfer between leader and follower.

    Large snapshots are split into chunks (default 1MB each) and
    sent sequentially. The follower assembles the chunks and applies
    the snapshot once all chunks are received.
    """

    CHUNK_SIZE = 1024 * 1024  # 1MB chunks

    def __init__(self, snapshot: Snapshot, leader_id: str, term: int):
        self.snapshot = snapshot
        self.leader_id = leader_id
        self.term = term
        self.serialized = json.dumps(snapshot.data).encode()
        self.total_size = len(self.serialized)
        self.offset = 0
        self.chunks_sent = 0

    def next_chunk(self) -> Optional[SnapshotChunk]:
        """Generate the next chunk to send."""
        if self.offset >= self.total_size:
            return None

        end = min(self.offset + self.CHUNK_SIZE, self.total_size)
        chunk = SnapshotChunk(
            term=self.term,
            leader_id=self.leader_id,
            last_included_index=self.snapshot.last_included_index,
            last_included_term=self.snapshot.last_included_term,
            offset=self.offset,
            data=self.serialized[self.offset:end],
            done=(end >= self.total_size),
        )

        self.offset = end
        self.chunks_sent += 1
        return chunk

    def progress(self) -> float:
        """Return transfer progress as a percentage."""
        return (self.offset / self.total_size * 100) if self.total_size > 0 else 100.0


class SnapshotReceiver:
    """Receives and assembles snapshot chunks on the follower side."""

    def __init__(self):
        self.buffer: bytearray = bytearray()
        self.expected_offset: int = 0
        self.last_included_index: int = 0
        self.last_included_term: int = 0
        self.chunks_received: int = 0

    def receive_chunk(self, chunk: SnapshotChunk) -> Optional[Snapshot]:
        """
        Process a received snapshot chunk.

        Returns the complete Snapshot when all chunks are received.
        """
        if chunk.offset != self.expected_offset:
            # Out of order — reset
            self.buffer = bytearray()
            self.expected_offset = 0
            return None

        self.buffer.extend(chunk.data)
        self.expected_offset = chunk.offset + len(chunk.data)
        self.last_included_index = chunk.last_included_index
        self.last_included_term = chunk.last_included_term
        self.chunks_received += 1

        if chunk.done:
            # Assemble the snapshot
            data = json.loads(self.buffer.decode())
            snapshot = Snapshot(
                last_included_index=self.last_included_index,
                last_included_term=self.last_included_term,
                data=data,
                config=set(),
            )
            self._reset()
            return snapshot

        return None

    def _reset(self):
        """Reset receiver state for the next transfer."""
        self.buffer = bytearray()
        self.expected_offset = 0
        self.chunks_received = 0


def demonstrate_snapshot_transfer():
    """Demonstrate chunked snapshot transfer."""
    print("=== Snapshot Transfer ===\n")

    # Create a large-ish snapshot
    data = {f"key_{i}": f"value_{i}" for i in range(1000)}
    snapshot = Snapshot(
        last_included_index=5000,
        last_included_term=3,
        data=data,
        config=set(),
    )

    # Sender
    transfer = SnapshotTransfer(snapshot, "leader", term=3)
    transfer.CHUNK_SIZE = 4096  # Smaller chunks for demo

    # Receiver
    receiver = SnapshotReceiver()

    print(f"Snapshot size: {transfer.total_size} bytes")
    print(f"Chunk size: {transfer.CHUNK_SIZE} bytes")

    result = None
    while result is None:
        chunk = transfer.next_chunk()
        if chunk is None:
            break
        result = receiver.receive_chunk(chunk)
        print(f"  Chunk {transfer.chunks_sent}: offset={chunk.offset}, "
              f"size={len(chunk.data)}, done={chunk.done}, "
              f"progress={transfer.progress():.1f}%")

    if result:
        print(f"\nSnapshot received successfully!")
        print(f"  Index: {result.last_included_index}")
        print(f"  Term: {result.last_included_term}")
        print(f"  Keys: {len(result.data)}")


demonstrate_snapshot_transfer()
```

---

## 6. Linearizable Reads

### 6.1 The Problem with Naive Reads

A naive read from the leader's state machine is NOT linearizable because the leader might be stale (partitioned from the majority). The leader must verify it is still the leader before serving a read.

```
Time ──────────────────────────────────────►
       S1 (old leader)         S2 (new leader)
       ┌──────────┐           ┌──────────┐
       │ x = 1    │           │ x = 2    │  ← Client wrote x=2 to S2
       └──────────┘           └──────────┘
              │
     Client reads x from S1
     Returns 1 ← STALE! Not linearizable.
```

### 6.2 ReadIndex

ReadIndex confirms leadership by checking a majority:

```python
class LinearizableReader:
    """
    Implements linearizable reads for Raft.

    Three approaches:
    1. Log Read: Treat reads as log entries (simple but slow)
    2. ReadIndex: Confirm leadership via heartbeat round
    3. LeaseRead: Use time-based lease to skip heartbeat
    """

    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = peers
        self.commit_index = 0
        self.last_applied = 0
        self.state_machine: dict = {}
        self.pending_reads: list[dict] = []

        # Lease-based
        self.lease_expiry: float = 0.0
        self.LEASE_DURATION: float = 0.1  # 100ms lease

    def read_via_log(self, key: str) -> dict:
        """
        Approach 1: Log Read — treat read as a log entry.

        The read command goes through Raft consensus, ensuring it sees
        all previously committed writes. Correct but adds latency
        of a full consensus round.
        """
        return {
            "method": "log_read",
            "action": "Propose read as log entry → wait for commit → apply",
            "latency": "1 RTT (consensus round)",
            "key": key,
        }

    def read_via_read_index(self, key: str) -> dict:
        """
        Approach 2: ReadIndex — confirm leadership, then read at commitIndex.

        Steps:
        1. Record current commitIndex as readIndex
        2. Send heartbeats to a majority
        3. If majority acknowledges, we are still leader
        4. Wait for state machine to advance to readIndex
        5. Execute the read on the state machine

        Latency: 1 heartbeat RTT (much less than consensus)
        """
        read_index = self.commit_index

        # Simulate heartbeat confirmation
        acks = self._send_heartbeats()
        majority = len(self.peers) // 2 + 1

        if acks >= majority:
            # We are confirmed leader; wait for apply to catch up
            while self.last_applied < read_index:
                pass  # In practice: async wait

            value = self.state_machine.get(key)
            return {
                "method": "read_index",
                "key": key,
                "value": value,
                "read_at_index": read_index,
                "latency": "1 heartbeat RTT",
            }
        else:
            return {"method": "read_index", "error": "Not leader (heartbeat failed)"}

    def read_via_lease(self, key: str) -> dict:
        """
        Approach 3: LeaseRead — use time-based lease to skip heartbeat.

        If the leader's lease has not expired, it can serve reads
        without confirming leadership. This assumes bounded clock drift.

        Warning: LeaseRead depends on clock accuracy. If clocks drift
        beyond bounds, linearizability may be violated.
        """
        now = time.time()

        if now < self.lease_expiry:
            # Lease is valid — serve read immediately
            value = self.state_machine.get(key)
            return {
                "method": "lease_read",
                "key": key,
                "value": value,
                "lease_remaining_ms": round((self.lease_expiry - now) * 1000, 1),
                "latency": "0 RTT (local read)",
            }
        else:
            # Lease expired — fall back to ReadIndex
            return self.read_via_read_index(key)

    def renew_lease(self, heartbeat_acks: int):
        """
        Renew the lease when a majority of heartbeats succeed.

        The lease is set to expire after LEASE_DURATION seconds.
        The leader must receive acks within this window to renew.
        """
        majority = len(self.peers) // 2 + 1
        if heartbeat_acks >= majority:
            self.lease_expiry = time.time() + self.LEASE_DURATION

    def _send_heartbeats(self) -> int:
        """Simulate sending heartbeats and receiving acks."""
        # In production, this is async with timeout
        return len(self.peers)  # Assume all ack for demo


def compare_read_approaches():
    """Compare the three linearizable read approaches."""
    print("=== Linearizable Read Approaches ===\n")

    reader = LinearizableReader("leader", ["f1", "f2", "f3", "f4"])
    reader.commit_index = 100
    reader.last_applied = 100
    reader.state_machine = {"x": "42", "y": "hello"}
    reader.lease_expiry = time.time() + 1.0  # Active lease

    approaches = [
        ("Log Read", reader.read_via_log("x")),
        ("ReadIndex", reader.read_via_read_index("x")),
        ("LeaseRead", reader.read_via_lease("x")),
    ]

    for name, result in approaches:
        print(f"{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print()

    # Comparison table
    print("Comparison:")
    print(f"  {'Approach':<15} {'Latency':<20} {'Safety':<20} {'Requirement'}")
    print(f"  {'Log Read':<15} {'1 consensus RTT':<20} {'Always safe':<20} {'None'}")
    print(f"  {'ReadIndex':<15} {'1 heartbeat RTT':<20} {'Always safe':<20} {'Majority heartbeat'}")
    print(f"  {'LeaseRead':<15} {'0 RTT (local)':<20} {'Clock-dependent':<20} {'Bounded clock drift'}")


compare_read_approaches()
```

---

## 7. Performance Tuning

### 7.1 Key Performance Knobs

```python
@dataclass
class RaftPerformanceConfig:
    """
    Tunable performance parameters for a Raft implementation.

    These parameters trade off between latency, throughput,
    and resource consumption.
    """
    # Election timing
    election_timeout_min_ms: int = 150
    election_timeout_max_ms: int = 300
    heartbeat_interval_ms: int = 50

    # Log batching
    max_entries_per_append: int = 1000  # Max entries in one AppendEntries
    max_batch_size_bytes: int = 1024 * 1024  # 1MB max batch
    batch_wait_ms: int = 1  # Wait up to 1ms to accumulate batch

    # Snapshot
    snapshot_threshold: int = 10000  # Entries before snapshot
    snapshot_chunk_size: int = 1024 * 1024  # 1MB chunks

    # Pipeline
    max_inflight_messages: int = 256  # Max in-flight AppendEntries per peer
    pipeline_enabled: bool = True

    # Disk
    sync_on_apply: bool = False  # fsync state machine changes
    wal_sync_mode: str = "fdatasync"  # "none", "fdatasync", "fsync"

    def validate(self):
        """Validate parameter relationships."""
        errors = []

        # heartbeat << election timeout
        if self.heartbeat_interval_ms >= self.election_timeout_min_ms / 3:
            errors.append(
                f"Heartbeat ({self.heartbeat_interval_ms}ms) should be "
                f"<< election timeout min ({self.election_timeout_min_ms}ms)"
            )

        # Election timeout range
        if self.election_timeout_max_ms <= self.election_timeout_min_ms:
            errors.append("Election timeout max must be > min")

        return errors


def analyze_performance_config():
    """Analyze different performance configurations."""
    print("=== Performance Configuration Analysis ===\n")

    configs = {
        "Low Latency": RaftPerformanceConfig(
            election_timeout_min_ms=100,
            election_timeout_max_ms=200,
            heartbeat_interval_ms=20,
            batch_wait_ms=0,
            max_inflight_messages=512,
        ),
        "High Throughput": RaftPerformanceConfig(
            election_timeout_min_ms=500,
            election_timeout_max_ms=1000,
            heartbeat_interval_ms=100,
            batch_wait_ms=5,
            max_entries_per_append=5000,
            max_batch_size_bytes=4 * 1024 * 1024,
        ),
        "WAN Deployment": RaftPerformanceConfig(
            election_timeout_min_ms=5000,
            election_timeout_max_ms=10000,
            heartbeat_interval_ms=1000,
            batch_wait_ms=10,
            snapshot_chunk_size=256 * 1024,
        ),
    }

    for name, config in configs.items():
        errors = config.validate()
        print(f"{name}:")
        print(f"  Election timeout: [{config.election_timeout_min_ms}, "
              f"{config.election_timeout_max_ms}] ms")
        print(f"  Heartbeat: {config.heartbeat_interval_ms} ms")
        print(f"  Batch wait: {config.batch_wait_ms} ms")
        print(f"  Max batch: {config.max_entries_per_append} entries / "
              f"{config.max_batch_size_bytes / 1024:.0f} KB")
        if errors:
            for e in errors:
                print(f"  WARNING: {e}")
        print()


analyze_performance_config()
```

---

## 8. Batching and Pipelining

### 8.1 Request Batching

Accumulate multiple client requests and replicate them in a single AppendEntries:

```python
class RequestBatcher:
    """
    Batches client requests for efficient replication.

    Instead of replicating one entry at a time, the batcher
    accumulates requests for up to `max_wait` and sends them
    as a single AppendEntries batch.
    """

    def __init__(self, max_size: int = 100, max_wait_ms: float = 1.0):
        self.max_size = max_size
        self.max_wait_ms = max_wait_ms
        self.batch: list[dict] = []
        self.batch_start_time: Optional[float] = None
        self.batches_flushed: int = 0
        self.total_entries: int = 0

    def add(self, entry: dict) -> Optional[list[dict]]:
        """
        Add an entry to the batch.

        Returns the batch when it should be flushed (size or time limit).
        """
        if not self.batch:
            self.batch_start_time = time.time()

        self.batch.append(entry)

        if len(self.batch) >= self.max_size:
            return self._flush()

        return None

    def check_timeout(self) -> Optional[list[dict]]:
        """Check if the batch should be flushed due to timeout."""
        if not self.batch or self.batch_start_time is None:
            return None

        elapsed_ms = (time.time() - self.batch_start_time) * 1000
        if elapsed_ms >= self.max_wait_ms:
            return self._flush()

        return None

    def _flush(self) -> list[dict]:
        """Flush the current batch."""
        batch = self.batch
        self.batch = []
        self.batch_start_time = None
        self.batches_flushed += 1
        self.total_entries += len(batch)
        return batch

    def stats(self) -> dict:
        return {
            "batches_flushed": self.batches_flushed,
            "total_entries": self.total_entries,
            "avg_batch_size": (
                self.total_entries / self.batches_flushed
                if self.batches_flushed > 0 else 0
            ),
            "pending": len(self.batch),
        }


def demonstrate_batching():
    """Demonstrate request batching for throughput improvement."""
    print("=== Request Batching ===\n")

    batcher = RequestBatcher(max_size=10, max_wait_ms=5.0)

    # Simulate a burst of requests
    flushed_batches = []
    for i in range(50):
        entry = {"op": "put", "key": f"k{i}", "value": f"v{i}"}
        batch = batcher.add(entry)
        if batch:
            flushed_batches.append(batch)
            print(f"  Flushed batch of {len(batch)} entries (size limit)")

    # Check for remaining
    remaining = batcher.check_timeout()
    if remaining:
        flushed_batches.append(remaining)

    print(f"\nBatching stats: {batcher.stats()}")
    print(f"  Without batching: {50} AppendEntries RPCs")
    print(f"  With batching: {len(flushed_batches)} AppendEntries RPCs")
    print(f"  Reduction: {(1 - len(flushed_batches)/50)*100:.0f}%")


demonstrate_batching()
```

### 8.2 Pipeline Replication

```python
class PipelinedReplicator:
    """
    Pipelined log replication for reduced latency.

    Instead of waiting for each AppendEntries response before
    sending the next batch, pipeline multiple batches in flight.
    """

    def __init__(self, peer_id: str, max_inflight: int = 8):
        self.peer_id = peer_id
        self.max_inflight = max_inflight
        self.inflight: list[dict] = []
        self.next_index: int = 1
        self.match_index: int = 0
        self.messages_sent: int = 0
        self.messages_acked: int = 0

    def can_send(self) -> bool:
        """Check if we can send another batch."""
        return len(self.inflight) < self.max_inflight

    def send_batch(self, entries: list[dict]) -> dict:
        """Queue a batch for sending."""
        msg = {
            "type": "AppendEntries",
            "to": self.peer_id,
            "prev_log_index": self.next_index - 1,
            "entries": entries,
            "batch_id": self.messages_sent,
        }
        self.inflight.append(msg)
        self.next_index += len(entries)
        self.messages_sent += 1
        return msg

    def ack(self, batch_id: int, success: bool, match_index: int):
        """Process an acknowledgment."""
        self.inflight = [m for m in self.inflight if m["batch_id"] != batch_id]
        self.messages_acked += 1

        if success:
            self.match_index = max(self.match_index, match_index)
        else:
            # Need to retry — reset pipeline
            self.next_index = match_index + 1
            self.inflight.clear()

    def stats(self) -> dict:
        return {
            "peer": self.peer_id,
            "inflight": len(self.inflight),
            "sent": self.messages_sent,
            "acked": self.messages_acked,
            "match_index": self.match_index,
        }


def demonstrate_pipelining():
    """Compare sequential vs pipelined replication."""
    print("=== Pipelined Replication ===\n")

    rtt_ms = 1.0  # 1ms RTT
    num_batches = 20

    # Sequential: one batch at a time
    sequential_time = num_batches * rtt_ms
    print(f"Sequential: {num_batches} batches × {rtt_ms}ms RTT = {sequential_time:.0f}ms")

    # Pipelined: multiple batches in flight
    pipeline_depth = 8
    pipeline_time = rtt_ms + (num_batches - 1) * (rtt_ms / pipeline_depth)
    print(f"Pipelined (depth={pipeline_depth}): ~{pipeline_time:.1f}ms")
    print(f"Speedup: {sequential_time / pipeline_time:.1f}x")


demonstrate_pipelining()
```

---

## 9. Complete Implementation

### 9.1 Integration Test

```python
def integration_test():
    """
    Full integration test: membership change + snapshot + linearizable read.
    """
    print("=== Integration Test ===\n")

    # 1. Start with a 3-node cluster
    membership = RaftMembership({"n1", "n2", "n3"})
    compactor = LogCompactor()
    compactor.SNAPSHOT_THRESHOLD = 50

    reader = LinearizableReader("n1", ["n2", "n3"])

    # 2. Write 200 entries
    for i in range(1, 201):
        compactor.append({
            "term": 1,
            "index": i,
            "command": {"op": "put", "key": f"k{i % 20}", "value": f"v{i}"},
        })
        compactor.apply_up_to(i)

        if compactor.should_snapshot():
            snap = compactor.create_snapshot()
            print(f"  Snapshot at index {snap.last_included_index}, "
                  f"log trimmed to {len(compactor.log)} entries")

    # 3. Add a new node
    print(f"\n  Adding node n4...")
    membership.propose_add("n4")
    membership.commit_change()

    # 4. Linearizable read
    reader.state_machine = dict(compactor.state_machine)
    reader.commit_index = compactor.commit_index
    reader.last_applied = compactor.commit_index
    reader.lease_expiry = time.time() + 1.0

    result = reader.read_via_lease("k5")
    print(f"\n  LeaseRead('k5'): {result}")

    # 5. Summary
    print(f"\n  Cluster: {sorted(membership.current_config)}")
    print(f"  Log: {compactor.stats()}")
    print(f"  State machine: {len(compactor.state_machine)} keys")


integration_test()
```

---

## 10. Summary and Key Takeaways

### Raft Part 2 Checklist

> **RAFT IMPLEMENTATION PART 2 CHECKLIST**
>
> ☐ Membership changes: only one pending at a time
> ☐ New servers catch up before becoming voters
> ☐ Joint consensus for multi-server changes
> ☐ Snapshots triggered at configurable threshold
> ☐ InstallSnapshot RPC with chunked transfer
> ☐ ReadIndex for linearizable reads without log overhead
> ☐ LeaseRead for zero-RTT reads (with clock assumption)
> ☐ Request batching for throughput
> ☐ Pipelined replication for latency

### Key Insights

1. **Membership changes are dangerous**: A bug here creates split-brain. Single-server changes are simpler and safer.
2. **Snapshots are essential**: Without them, log grows unbounded and new nodes take forever to join.
3. **Linearizable reads need work**: Naive leader reads are NOT linearizable. ReadIndex or LeaseRead is required.
4. **Performance tuning is deployment-specific**: LAN vs WAN needs different timeouts. Batching vs latency is a tradeoff.

---

## 11. Practice Problems

### Problem 1: Membership Change Safety

Prove that single-server membership changes cannot create two disjoint majorities. What is the maximum number of servers you can safely add in sequence without waiting for commits?

### Problem 2: Snapshot Optimization

Design a copy-on-write snapshot mechanism that allows the state machine to continue serving reads while the snapshot is being serialized to disk. Consider memory overhead and consistency.

### Problem 3: ReadIndex Latency

A 5-node cluster has 2ms network RTT. Calculate the read latency for:
1. Log Read
2. ReadIndex
3. LeaseRead (with valid lease)
4. LeaseRead (with expired lease)

### Problem 4: Batching Tradeoff

Given a workload of 10,000 requests/second with 1ms network RTT:
- Calculate optimal batch size and wait time
- What is the p99 latency with batching vs without?
- What is the throughput improvement?

### Problem 5: Implementation Challenge

Implement `RaftNode.install_snapshot()` that handles receiving a snapshot from a leader, including:
- Discarding conflicting log entries
- Restoring the state machine
- Updating commit/apply indices

---

## 12. References

1. Ongaro, D. (2014). "Consensus: Bridging Theory and Practice." PhD Dissertation, Stanford University.
2. Ongaro, D. (2015). "Raft Membership Changes." (Raft developer guide)
3. Howard, H. et al. (2015). "Raft Refloated: Do We Have Consensus?" *Operating Systems Review*.
4. etcd documentation: Learner mode and membership changes.
5. TiKV blog: "Raft Optimization" — batching, pipelining, and async apply.
6. CockroachDB blog: "Living Without Atomic Clocks" — LeaseRead in practice.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 9. O'Reilly Media.

---

[Next: Lesson 20 — Distributed Hash Tables](./20_Distributed_Hash_Tables.md)
