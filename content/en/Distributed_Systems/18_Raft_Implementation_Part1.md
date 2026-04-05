# Lesson 18: Raft Implementation Part 1 — Leader Election, Log Replication, and Safety

[Overview](./00_Overview.md) | [Previous: Capstone — Building a Distributed KV Store](./16_Capstone_Building_Distributed_KV_Store.md) | [Next: Raft Implementation Part 2](./19_Raft_Implementation_Part2.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement Raft leader election with randomized timeouts and pre-vote optimization
2. Build a complete log replication pipeline with AppendEntries RPC and commit advancement
3. Prove and enforce the Raft safety properties (Election Safety, Log Matching, Leader Completeness)
4. Construct a deterministic state machine that applies committed entries in order
5. Handle edge cases including split votes, stale leaders, and log divergence

---

## Table of Contents

1. [Introduction to Raft Implementation](#1-introduction-to-raft-implementation)
2. [Node States and Transitions](#2-node-states-and-transitions)
3. [Leader Election Algorithm](#3-leader-election-algorithm)
4. [Log Replication](#4-log-replication)
5. [Safety Properties and Proofs](#5-safety-properties-and-proofs)
6. [State Machine Application](#6-state-machine-application)
7. [Handling Edge Cases](#7-handling-edge-cases)
8. [Complete Implementation](#8-complete-implementation)
9. [Testing Leader Election](#9-testing-leader-election)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Introduction to Raft Implementation

### 1.1 Why Another Raft Lesson?

Lesson 06 introduced Raft concepts. This lesson and the next provide a *production-quality* implementation walkthrough. Where Lesson 06 was "what Raft does," this lesson is "how to build Raft correctly."

The gap between understanding Raft and implementing it correctly is enormous. Subtle bugs hide in:

- **Election timer management**: Off-by-one errors in timeout randomization
- **Log index math**: Confusion between 0-based and 1-based indexing
- **Commit advancement**: Incorrectly advancing `commitIndex` across terms
- **Stale message handling**: Processing messages from old terms

### 1.2 Implementation Scope

```
Part 1 (This Lesson)              Part 2 (Lesson 19)
┌────────────────────────┐       ┌────────────────────────┐
│ Leader Election         │       │ Membership Changes      │
│ Log Replication         │       │ Log Compaction           │
│ Safety Proofs           │       │ Snapshotting             │
│ State Machine           │       │ Linearizable Reads       │
│ Edge Cases              │       │ Performance Tuning       │
└────────────────────────┘       └────────────────────────┘
```

### 1.3 Architecture Overview

```
Client Request
      │
      ▼
┌─────────────┐
│   Leader     │──── AppendEntries RPC ───▶ Followers
│             │◀─── Responses ─────────────┘
│ ┌─────────┐ │
│ │  Log    │ │  ← Uncommitted entries appended
│ └────┬────┘ │
│      │      │
│ ┌────▼────┐ │
│ │ Commit  │ │  ← Majority replicated → committed
│ └────┬────┘ │
│      │      │
│ ┌────▼────┐ │
│ │  State  │ │  ← Applied to key-value state machine
│ │ Machine │ │
│ └─────────┘ │
└─────────────┘
```

---

## 2. Node States and Transitions

### 2.1 The Three States

Every Raft node is in exactly one of three states at any given time:

```
                    ┌──────────┐
         timeout    │          │  receives votes
        ┌──────────▶│ Candidate│──────────────┐
        │           │          │              │
        │           └────┬─────┘              │
        │                │                    │
        │          discovers current          │
        │          leader or new term         │
        │                │                    │
        │                ▼                    ▼
   ┌────┴────┐     ┌──────────┐       ┌──────────┐
   │         │     │          │       │          │
   │ Follower│◀────│ Follower │       │  Leader  │
   │         │     │          │       │          │
   └─────────┘     └──────────┘       └──────────┘
        ▲                                   │
        │          discovers server         │
        │          with higher term         │
        └───────────────────────────────────┘
```

### 2.2 Persistent State

These fields MUST survive crashes (written to stable storage before responding to any RPC):

```python
@dataclass
class PersistentState:
    """State that must be persisted to stable storage before responding to RPCs."""
    current_term: int = 0       # Latest term server has seen
    voted_for: Optional[str] = None  # CandidateId that received vote in current term
    log: list = field(default_factory=lambda: [])  # Log entries (first index is 1)
```

### 2.3 Volatile State

```python
@dataclass
class VolatileState:
    """State that can be reconstructed after a crash."""
    commit_index: int = 0  # Index of highest log entry known to be committed
    last_applied: int = 0  # Index of highest log entry applied to state machine

@dataclass
class LeaderVolatileState:
    """Additional state maintained only on leaders."""
    next_index: dict = field(default_factory=dict)   # For each server: next log index to send
    match_index: dict = field(default_factory=dict)   # For each server: highest log index replicated
```

---

## 3. Leader Election Algorithm

### 3.1 Election Timer

The election timer is the heartbeat of Raft. Getting it right is critical:

```python
import random
import time
import threading
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum


class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class RaftNode:
    """
    Complete Raft node implementation — Part 1: Election + Replication.

    This implements the core Raft protocol as described in the
    Ongaro & Ousterhout paper (2014), with additional production
    hardening from the etcd and TiKV implementations.
    """

    # Election timeout range (milliseconds)
    ELECTION_TIMEOUT_MIN = 150
    ELECTION_TIMEOUT_MAX = 300
    HEARTBEAT_INTERVAL = 50  # ms — must be << ELECTION_TIMEOUT_MIN

    def __init__(self, node_id: str, peers: list[str], storage_path: str = ""):
        self.node_id = node_id
        self.peers = peers
        self.storage_path = storage_path or f"/tmp/raft_{node_id}.json"

        # --- Persistent state (survives crashes) ---
        self.current_term: int = 0
        self.voted_for: Optional[str] = None
        self.log: list[dict] = []  # Each entry: {"term": int, "command": Any, "index": int}

        # --- Volatile state ---
        self.commit_index: int = 0
        self.last_applied: int = 0
        self.state: NodeState = NodeState.FOLLOWER

        # --- Leader-only volatile state ---
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        # --- Timing ---
        self.last_heartbeat_time: float = time.time()
        self.election_timeout: float = self._random_election_timeout()

        # --- State machine ---
        self.state_machine: Dict[str, str] = {}

        # --- Message queue (simulated network) ---
        self.inbox: list[dict] = []
        self.outbox: list[dict] = []

        # --- Statistics ---
        self.elections_started: int = 0
        self.elections_won: int = 0
        self.entries_committed: int = 0

        # Load persisted state
        self._load_persistent_state()

    def _random_election_timeout(self) -> float:
        """
        Generate a random election timeout.

        The randomization is CRITICAL for avoiding split votes.
        Each node picks a timeout uniformly at random from
        [ELECTION_TIMEOUT_MIN, ELECTION_TIMEOUT_MAX].

        The range should be:
        - broadcastTime << electionTimeout << MTBF
        - Where broadcastTime ≈ 0.5ms to 20ms (network RTT)
        - And MTBF is mean time between failures (months/years)
        """
        return random.uniform(
            self.ELECTION_TIMEOUT_MIN / 1000.0,
            self.ELECTION_TIMEOUT_MAX / 1000.0,
        )

    # ─── Persistence ───────────────────────────────────────────

    def _save_persistent_state(self):
        """
        Write persistent state to stable storage.

        CRITICAL: This must complete BEFORE responding to any RPC.
        In production, use fsync() to ensure durability.
        """
        state = {
            "current_term": self.current_term,
            "voted_for": self.voted_for,
            "log": self.log,
        }
        tmp_path = self.storage_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, self.storage_path)  # Atomic rename

    def _load_persistent_state(self):
        """Load persistent state from stable storage on startup."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    state = json.load(f)
                self.current_term = state.get("current_term", 0)
                self.voted_for = state.get("voted_for", None)
                self.log = state.get("log", [])
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted file; start fresh

    # ─── Term Management ───────────────────────────────────────

    def _step_down(self, new_term: int):
        """
        Step down to follower state when a higher term is discovered.

        This is the fundamental mechanism that prevents stale leaders
        from causing inconsistency. ANY message with a higher term
        causes an immediate step-down.
        """
        assert new_term >= self.current_term
        self.current_term = new_term
        self.voted_for = None
        self.state = NodeState.FOLLOWER
        self.last_heartbeat_time = time.time()
        self.election_timeout = self._random_election_timeout()
        self._save_persistent_state()

    # ─── Leader Election ───────────────────────────────────────

    def _start_election(self):
        """
        Start a new election by transitioning to candidate state.

        Steps:
        1. Increment currentTerm
        2. Vote for self
        3. Reset election timer
        4. Send RequestVote RPCs to all peers
        """
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat_time = time.time()
        self.election_timeout = self._random_election_timeout()
        self.elections_started += 1

        self._save_persistent_state()

        # Build RequestVote message
        last_log_index = len(self.log)
        last_log_term = self.log[-1]["term"] if self.log else 0

        for peer in self.peers:
            self.outbox.append({
                "type": "RequestVote",
                "to": peer,
                "from": self.node_id,
                "term": self.current_term,
                "candidate_id": self.node_id,
                "last_log_index": last_log_index,
                "last_log_term": last_log_term,
            })

        # Vote for self counts as one vote
        self._votes_received = {self.node_id}
        self._votes_denied = set()

    def _handle_request_vote(self, msg: dict) -> dict:
        """
        Handle an incoming RequestVote RPC.

        Grant vote if:
        1. Candidate's term ≥ our term
        2. We haven't voted for anyone else in this term
        3. Candidate's log is at least as up-to-date as ours

        The "up-to-date" check is crucial for safety:
        - Compare last log entry terms first
        - If terms are equal, compare log lengths
        """
        candidate_term = msg["term"]
        candidate_id = msg["candidate_id"]
        candidate_last_log_index = msg["last_log_index"]
        candidate_last_log_term = msg["last_log_term"]

        # Step down if candidate has higher term
        if candidate_term > self.current_term:
            self._step_down(candidate_term)

        # Determine if we should grant the vote
        vote_granted = False

        if candidate_term < self.current_term:
            # Candidate is behind — reject
            vote_granted = False
        elif self.voted_for is None or self.voted_for == candidate_id:
            # Haven't voted yet, or already voted for this candidate
            # Check log up-to-date-ness
            our_last_log_index = len(self.log)
            our_last_log_term = self.log[-1]["term"] if self.log else 0

            if candidate_last_log_term > our_last_log_term:
                vote_granted = True
            elif (candidate_last_log_term == our_last_log_term
                  and candidate_last_log_index >= our_last_log_index):
                vote_granted = True

        if vote_granted:
            self.voted_for = candidate_id
            self.last_heartbeat_time = time.time()  # Reset election timer
            self._save_persistent_state()

        return {
            "type": "RequestVoteResponse",
            "to": candidate_id,
            "from": self.node_id,
            "term": self.current_term,
            "vote_granted": vote_granted,
        }

    def _handle_request_vote_response(self, msg: dict):
        """
        Handle a response to our RequestVote RPC.

        If we receive a majority of votes, become leader.
        If we discover a higher term, step down.
        """
        if msg["term"] > self.current_term:
            self._step_down(msg["term"])
            return

        # Only process if we are still a candidate in the same term
        if self.state != NodeState.CANDIDATE or msg["term"] != self.current_term:
            return

        if msg["vote_granted"]:
            self._votes_received.add(msg["from"])
        else:
            self._votes_denied.add(msg["from"])

        # Check if we have a majority
        total_nodes = len(self.peers) + 1  # +1 for self
        if len(self._votes_received) > total_nodes // 2:
            self._become_leader()

    def _become_leader(self):
        """
        Transition to leader state.

        Initialize nextIndex and matchIndex for all peers:
        - nextIndex: initialized to leader's last log index + 1
        - matchIndex: initialized to 0

        Immediately send heartbeats to assert authority.
        """
        self.state = NodeState.LEADER
        self.elections_won += 1

        last_log_index = len(self.log)
        for peer in self.peers:
            self.next_index[peer] = last_log_index + 1
            self.match_index[peer] = 0

        # Send initial heartbeats (empty AppendEntries)
        self._send_heartbeats()

    def _send_heartbeats(self):
        """
        Send AppendEntries RPCs to all peers.

        For each peer, include log entries starting from nextIndex.
        If no new entries, this serves as a heartbeat.
        """
        for peer in self.peers:
            self._send_append_entries(peer)

    # ─── Log Replication ───────────────────────────────────────

    def _send_append_entries(self, peer: str):
        """
        Send AppendEntries RPC to a specific peer.

        The message includes:
        - Leader's term
        - prevLogIndex and prevLogTerm for consistency check
        - New entries (may be empty for heartbeats)
        - Leader's commitIndex
        """
        next_idx = self.next_index.get(peer, len(self.log) + 1)
        prev_log_index = next_idx - 1
        prev_log_term = 0

        if prev_log_index > 0 and prev_log_index <= len(self.log):
            prev_log_term = self.log[prev_log_index - 1]["term"]

        # Entries to send (from nextIndex onwards)
        entries = self.log[next_idx - 1:] if next_idx - 1 < len(self.log) else []

        self.outbox.append({
            "type": "AppendEntries",
            "to": peer,
            "from": self.node_id,
            "term": self.current_term,
            "leader_id": self.node_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": entries,
            "leader_commit": self.commit_index,
        })

    def _handle_append_entries(self, msg: dict) -> dict:
        """
        Handle an incoming AppendEntries RPC (follower side).

        Steps:
        1. Reply false if term < currentTerm
        2. Reply false if log doesn't contain an entry at prevLogIndex
           with matching prevLogTerm
        3. If an existing entry conflicts with a new one, delete the
           existing entry and all that follow it
        4. Append any new entries not already in the log
        5. Advance commitIndex if leaderCommit > commitIndex
        """
        leader_term = msg["term"]

        # Step down if leader has higher term
        if leader_term > self.current_term:
            self._step_down(leader_term)

        # Reject if leader's term is stale
        if leader_term < self.current_term:
            return {
                "type": "AppendEntriesResponse",
                "to": msg["from"],
                "from": self.node_id,
                "term": self.current_term,
                "success": False,
                "match_index": 0,
            }

        # Valid leader — reset election timer
        self.state = NodeState.FOLLOWER
        self.last_heartbeat_time = time.time()

        # Consistency check
        prev_log_index = msg["prev_log_index"]
        prev_log_term = msg["prev_log_term"]

        if prev_log_index > 0:
            if prev_log_index > len(self.log):
                # We don't have an entry at prevLogIndex
                return {
                    "type": "AppendEntriesResponse",
                    "to": msg["from"],
                    "from": self.node_id,
                    "term": self.current_term,
                    "success": False,
                    "match_index": len(self.log),
                    "conflict_term": 0,
                    "conflict_index": len(self.log) + 1,
                }

            if self.log[prev_log_index - 1]["term"] != prev_log_term:
                # Entry exists but term doesn't match
                conflict_term = self.log[prev_log_index - 1]["term"]
                # Find first index with conflict_term for fast backup
                conflict_index = prev_log_index
                for i in range(prev_log_index - 1, -1, -1):
                    if i == 0 or self.log[i - 1]["term"] != conflict_term:
                        conflict_index = i + 1
                        break

                return {
                    "type": "AppendEntriesResponse",
                    "to": msg["from"],
                    "from": self.node_id,
                    "term": self.current_term,
                    "success": False,
                    "match_index": conflict_index - 1,
                    "conflict_term": conflict_term,
                    "conflict_index": conflict_index,
                }

        # Append new entries
        entries = msg.get("entries", [])
        for i, entry in enumerate(entries):
            log_index = prev_log_index + 1 + i
            if log_index <= len(self.log):
                if self.log[log_index - 1]["term"] != entry["term"]:
                    # Conflict: delete this entry and all that follow
                    self.log = self.log[:log_index - 1]
                    self.log.append(entry)
                # else: entry already exists and matches, skip
            else:
                self.log.append(entry)

        self._save_persistent_state()

        # Advance commit index
        if msg["leader_commit"] > self.commit_index:
            last_new_index = prev_log_index + len(entries)
            self.commit_index = min(msg["leader_commit"], last_new_index)

        # Apply committed entries
        self._apply_committed_entries()

        return {
            "type": "AppendEntriesResponse",
            "to": msg["from"],
            "from": self.node_id,
            "term": self.current_term,
            "success": True,
            "match_index": prev_log_index + len(entries),
        }

    def _handle_append_entries_response(self, msg: dict):
        """
        Handle response to AppendEntries RPC (leader side).

        On success: update nextIndex and matchIndex for the peer.
        On failure: decrement nextIndex and retry (with optimization).
        """
        if msg["term"] > self.current_term:
            self._step_down(msg["term"])
            return

        if self.state != NodeState.LEADER:
            return

        peer = msg["from"]

        if msg["success"]:
            # Update tracking for this peer
            new_match_index = msg["match_index"]
            self.match_index[peer] = max(self.match_index.get(peer, 0), new_match_index)
            self.next_index[peer] = self.match_index[peer] + 1

            # Try to advance commit index
            self._advance_commit_index()
        else:
            # Optimization: use conflict information for fast backup
            if "conflict_term" in msg and msg["conflict_term"] > 0:
                # Search our log for conflict_term
                conflict_term = msg["conflict_term"]
                found = False
                for i in range(len(self.log) - 1, -1, -1):
                    if self.log[i]["term"] == conflict_term:
                        self.next_index[peer] = i + 2  # Next entry after our last with that term
                        found = True
                        break
                if not found:
                    self.next_index[peer] = msg.get("conflict_index", 1)
            else:
                # Simple backup: decrement by 1
                self.next_index[peer] = max(1, self.next_index.get(peer, 2) - 1)

            # Retry with updated nextIndex
            self._send_append_entries(peer)

    def _advance_commit_index(self):
        """
        Advance commitIndex based on majority replication.

        Find the highest N such that:
        - A majority of matchIndex[i] ≥ N
        - log[N].term == currentTerm (CRITICAL safety property)

        The term check prevents committing entries from previous terms
        by counting replicas. Entries from previous terms are committed
        indirectly when an entry from the current term is committed.
        """
        for n in range(len(self.log), self.commit_index, -1):
            if self.log[n - 1]["term"] != self.current_term:
                continue

            # Count replicas (including self)
            replicas = 1  # Count self
            for peer in self.peers:
                if self.match_index.get(peer, 0) >= n:
                    replicas += 1

            total_nodes = len(self.peers) + 1
            if replicas > total_nodes // 2:
                self.commit_index = n
                self._apply_committed_entries()
                break

    # ─── State Machine ─────────────────────────────────────────

    def _apply_committed_entries(self):
        """
        Apply committed but not yet applied entries to the state machine.

        Entries are applied in strict log order. This guarantees that
        all nodes apply the same commands in the same order, achieving
        linearizability.
        """
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]
            command = entry.get("command", {})

            op = command.get("op")
            if op == "put":
                self.state_machine[command["key"]] = command["value"]
            elif op == "delete":
                self.state_machine.pop(command["key"], None)
            elif op == "cas":  # Compare-and-swap
                key = command["key"]
                expected = command.get("expected")
                new_value = command["value"]
                if self.state_machine.get(key) == expected:
                    self.state_machine[key] = new_value

            self.entries_committed += 1

    # ─── Client Interface ──────────────────────────────────────

    def propose(self, command: dict) -> bool:
        """
        Propose a new command to the cluster (leader only).

        Returns True if the command was accepted for replication.
        The command is not yet committed — the caller must poll
        for commit status.
        """
        if self.state != NodeState.LEADER:
            return False

        entry = {
            "term": self.current_term,
            "index": len(self.log) + 1,
            "command": command,
        }
        self.log.append(entry)
        self._save_persistent_state()

        # Immediately replicate to all peers
        for peer in self.peers:
            self._send_append_entries(peer)

        return True

    # ─── Tick (Timer) ──────────────────────────────────────────

    def tick(self):
        """
        Called periodically to drive the Raft node.

        - Followers/Candidates: check election timeout
        - Leaders: send heartbeats
        """
        now = time.time()
        elapsed = now - self.last_heartbeat_time

        if self.state == NodeState.LEADER:
            if elapsed >= self.HEARTBEAT_INTERVAL / 1000.0:
                self.last_heartbeat_time = now
                self._send_heartbeats()
        else:
            if elapsed >= self.election_timeout:
                self._start_election()

    # ─── Message Dispatch ──────────────────────────────────────

    def receive(self, msg: dict):
        """Route an incoming message to the appropriate handler."""
        msg_type = msg.get("type")

        if msg_type == "RequestVote":
            response = self._handle_request_vote(msg)
            self.outbox.append(response)
        elif msg_type == "RequestVoteResponse":
            self._handle_request_vote_response(msg)
        elif msg_type == "AppendEntries":
            response = self._handle_append_entries(msg)
            self.outbox.append(response)
        elif msg_type == "AppendEntriesResponse":
            self._handle_append_entries_response(msg)

    def get_status(self) -> dict:
        """Return current node status for debugging."""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "term": self.current_term,
            "voted_for": self.voted_for,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "state_machine_size": len(self.state_machine),
            "elections_started": self.elections_started,
            "elections_won": self.elections_won,
            "entries_committed": self.entries_committed,
        }
```

---

## 4. Log Replication

### 4.1 The Log Matching Property

Raft maintains the **Log Matching Property**:

> If two entries in different logs have the same index and term,
> then they store the same command AND all preceding entries are identical.

This is enforced by the consistency check in AppendEntries:

```
Leader's log:   [1:a] [1:b] [2:c] [3:d] [3:e]
                  ↑ prevLogIndex=4, prevLogTerm=3

Follower A:     [1:a] [1:b] [2:c] [3:d]          ← Match! Append [3:e]
Follower B:     [1:a] [1:b] [2:c]                 ← Fail! No entry at index 4
Follower C:     [1:a] [1:b] [2:c] [2:x]           ← Fail! Term mismatch at index 4
```

### 4.2 Handling Log Divergence

When a follower's log diverges from the leader's, the leader must find the last point of agreement and overwrite everything after it:

```python
def demonstrate_log_divergence():
    """
    Demonstrate how Raft handles log divergence between leader and follower.

    Scenario: A network partition caused two leaders to accept different
    entries. After the partition heals, the surviving leader must bring
    the stale follower's log into alignment.
    """
    # Leader's log after winning election in term 3
    leader_log = [
        {"term": 1, "index": 1, "command": {"op": "put", "key": "x", "value": "1"}},
        {"term": 1, "index": 2, "command": {"op": "put", "key": "y", "value": "2"}},
        {"term": 2, "index": 3, "command": {"op": "put", "key": "x", "value": "3"}},
        {"term": 3, "index": 4, "command": {"op": "put", "key": "z", "value": "4"}},
    ]

    # Follower's log — diverged during partition
    follower_log = [
        {"term": 1, "index": 1, "command": {"op": "put", "key": "x", "value": "1"}},
        {"term": 1, "index": 2, "command": {"op": "put", "key": "y", "value": "2"}},
        {"term": 2, "index": 3, "command": {"op": "put", "key": "x", "value": "3"}},
        {"term": 2, "index": 4, "command": {"op": "put", "key": "w", "value": "9"}},  # DIVERGENT
        {"term": 2, "index": 5, "command": {"op": "put", "key": "v", "value": "8"}},  # DIVERGENT
    ]

    print("=== Log Divergence Resolution ===")
    print(f"Leader log:   {[(e['term'], e['index']) for e in leader_log]}")
    print(f"Follower log: {[(e['term'], e['index']) for e in follower_log]}")

    # Leader tries AppendEntries with prevLogIndex=4, prevLogTerm=3
    prev_idx = len(leader_log)  # 4
    prev_term = leader_log[prev_idx - 1]["term"]  # 3

    # Follower checks: entry at index 4 has term 2, not 3 → CONFLICT
    if follower_log[prev_idx - 1]["term"] != prev_term:
        conflict_term = follower_log[prev_idx - 1]["term"]
        # Find first index with conflict_term
        conflict_start = prev_idx
        for i in range(prev_idx - 1, -1, -1):
            if i == 0 or follower_log[i - 1]["term"] != conflict_term:
                conflict_start = i + 1
                break

        print(f"\nConflict detected at index {prev_idx}")
        print(f"  Leader has term {prev_term}, follower has term {conflict_term}")
        print(f"  Conflict starts at index {conflict_start}")

        # Leader backs up nextIndex to conflict_start
        # and retries, eventually resolving the divergence
        follower_log = follower_log[:conflict_start - 1]  # Truncate
        follower_log.extend(leader_log[conflict_start - 1:])  # Append leader's entries

        print(f"\nAfter resolution:")
        print(f"  Leader log:   {[(e['term'], e['index']) for e in leader_log]}")
        print(f"  Follower log: {[(e['term'], e['index']) for e in follower_log]}")
        print(f"  Logs match: {leader_log == follower_log}")


demonstrate_log_divergence()
```

### 4.3 Fast Backup Optimization

The naive approach decrements `nextIndex` by 1 on each failure, requiring O(n) round trips for n divergent entries. The fast backup optimization includes conflict information in the rejection:

```
Follower → Leader:
  success = False
  conflictTerm = term of the conflicting entry
  conflictIndex = first index with conflictTerm

Leader response:
  If leader has entries with conflictTerm:
    nextIndex = leader's last entry with conflictTerm + 1
  Else:
    nextIndex = conflictIndex
```

This reduces the number of round trips to O(number of distinct conflicting terms).

---

## 5. Safety Properties and Proofs

### 5.1 Election Safety

**Claim**: At most one leader can be elected in a given term.

**Proof**:
1. Each node votes for at most one candidate per term (enforced by `voted_for` persistence).
2. A candidate needs a strict majority (> N/2) of votes to win.
3. Any two majorities overlap by at least one node.
4. That overlapping node voted for at most one candidate.
5. Therefore, at most one candidate can receive a majority. ∎

```python
def verify_election_safety(votes: dict, total_nodes: int) -> bool:
    """
    Verify that at most one candidate received a majority of votes.

    Args:
        votes: {candidate_id: set of voters}
        total_nodes: Total number of nodes in the cluster

    Returns:
        True if election safety is maintained
    """
    majority = total_nodes // 2 + 1
    winners = [c for c, v in votes.items() if len(v) >= majority]

    if len(winners) > 1:
        print(f"SAFETY VIOLATION: Multiple winners: {winners}")
        return False

    # Verify no voter voted for multiple candidates
    all_voters = set()
    for candidate, voters in votes.items():
        for voter in voters:
            if voter in all_voters:
                print(f"SAFETY VIOLATION: {voter} voted for multiple candidates")
                return False
            all_voters.add(voter)

    return True


# Test with 5 nodes
votes_safe = {
    "A": {"A", "B", "C"},  # A wins with 3/5
    "D": {"D", "E"},        # D loses with 2/5
}
print(f"Safe election: {verify_election_safety(votes_safe, 5)}")  # True

votes_unsafe = {
    "A": {"A", "B", "C"},
    "D": {"D", "B", "E"},  # B voted for both! (impossible in correct impl)
}
print(f"Unsafe election: {verify_election_safety(votes_unsafe, 5)}")  # False
```

### 5.2 Leader Completeness

**Claim**: If a log entry is committed in a given term, that entry will be present in the logs of all leaders for all higher-numbered terms.

**Proof sketch**:
1. A committed entry E at index i was replicated to a majority S1.
2. A future leader L must receive votes from a majority S2.
3. S1 ∩ S2 is non-empty (pigeonhole).
4. The voter in the intersection has E in its log.
5. The up-to-date check in RequestVote ensures L's log is at least as up-to-date.
6. Therefore L's log contains E (or an entry at the same index with an equal or higher term, which by Log Matching must agree). ∎

### 5.3 The Commit Rule Subtlety

The most subtle safety issue in Raft is the commit rule for entries from previous terms:

```
Term 1: Leader S1 replicates entry A to S1, S2
         S1 crashes before commit

Term 2: S5 elected (S3, S4, S5 vote; S5 has empty log)
         S5 replicates entry B to S3
         S5 crashes

Term 3: S1 recovers, elected (S1, S2, S3, S4 vote)
         S1 has entry A at index 1

WRONG approach: S1 counts replicas of A (S1, S2 = 2/5 → not enough)
                S1 replicates A to S3, now 3/5 replicas → commit A?

                NO! This is UNSAFE. If S1 crashes now, S5 could be elected
                in term 4 and overwrite A with B.

CORRECT approach: S1 does NOT directly commit A.
                  S1 appends a new entry C in term 3.
                  When C is committed (replicated to majority), A is
                  committed indirectly because it precedes C in the log.
```

```python
def demonstrate_commit_rule():
    """
    Demonstrate why Raft never commits entries from previous terms
    by counting replicas directly.
    """
    print("=== Raft Commit Rule Demonstration ===\n")

    # Scenario setup
    nodes = ["S1", "S2", "S3", "S4", "S5"]

    # After term 1: S1 replicated entry A (term 1) to S1 and S2
    logs = {
        "S1": [{"term": 1, "cmd": "A"}],
        "S2": [{"term": 1, "cmd": "A"}],
        "S3": [],
        "S4": [],
        "S5": [],
    }
    print("After term 1 (S1 crashed after partial replication):")
    for n, log in logs.items():
        print(f"  {n}: {log}")

    # After term 2: S5 elected, replicated entry B to S3
    logs["S5"] = [{"term": 2, "cmd": "B"}]
    logs["S3"] = [{"term": 2, "cmd": "B"}]
    print("\nAfter term 2 (S5 crashed after partial replication):")
    for n, log in logs.items():
        print(f"  {n}: {log}")

    # Term 3: S1 elected
    # WRONG: S1 replicates A to S3, overwriting B
    print("\n--- WRONG approach: commit by counting replicas of old entry ---")
    logs_wrong = {k: list(v) for k, v in logs.items()}
    logs_wrong["S3"] = [{"term": 1, "cmd": "A"}]  # Overwrite B with A
    print("S1 replicates A to S3:")
    replica_count = sum(1 for n in nodes if logs_wrong[n] and logs_wrong[n][0].get("cmd") == "A")
    print(f"  Replicas of A: {replica_count}/5 → {'committed' if replica_count >= 3 else 'not committed'}")
    print("  If S1 crashes now, S5 could win term 4 and overwrite A with B!")
    print("  This would VIOLATE the safety property.")

    # CORRECT: S1 appends entry C in term 3, commits A indirectly
    print("\n--- CORRECT approach: append new entry in current term ---")
    logs_correct = {k: list(v) for k, v in logs.items()}
    logs_correct["S1"].append({"term": 3, "cmd": "C"})
    logs_correct["S2"].append({"term": 3, "cmd": "C"})
    logs_correct["S3"] = [{"term": 1, "cmd": "A"}, {"term": 3, "cmd": "C"}]
    print("S1 replicates A and C to S3:")
    for n in nodes:
        print(f"  {n}: {logs_correct[n]}")

    # Count replicas of C (term 3)
    c_replicas = sum(
        1 for n in nodes
        if len(logs_correct[n]) >= 2 and logs_correct[n][1].get("term") == 3
    )
    print(f"\n  Replicas of C (term 3): {c_replicas}/5")
    print("  C is committed → A is committed indirectly (precedes C in log)")
    print("  S5 cannot win future elections because its log is less up-to-date")


demonstrate_commit_rule()
```

---

## 6. State Machine Application

### 6.1 Deterministic Execution

The state machine MUST be deterministic: given the same log entries in the same order, every node must produce the same state. This means:

- No random number generation in command execution
- No dependency on wall-clock time
- No external I/O during command execution
- No reliance on hash map iteration order (use sorted operations)

```python
class DeterministicKVStateMachine:
    """
    A deterministic key-value state machine for Raft.

    Every operation is a pure function of the current state and the command.
    No side effects, no randomness, no external dependencies.
    """

    def __init__(self):
        self.data: Dict[str, str] = {}
        self.applied_index: int = 0
        self.applied_commands: list = []

    def apply(self, index: int, command: dict) -> dict:
        """
        Apply a command to the state machine.

        Args:
            index: The log index of this command
            command: The command to apply

        Returns:
            Result of the command execution
        """
        assert index == self.applied_index + 1, (
            f"Commands must be applied in order: expected {self.applied_index + 1}, got {index}"
        )

        op = command.get("op")
        result = {"ok": True, "index": index}

        if op == "put":
            self.data[command["key"]] = command["value"]
            result["op"] = "put"

        elif op == "get":
            value = self.data.get(command["key"])
            result["op"] = "get"
            result["value"] = value
            result["found"] = value is not None

        elif op == "delete":
            existed = command["key"] in self.data
            if existed:
                del self.data[command["key"]]
            result["op"] = "delete"
            result["existed"] = existed

        elif op == "cas":
            key = command["key"]
            expected = command.get("expected")
            current = self.data.get(key)
            if current == expected:
                self.data[key] = command["value"]
                result["op"] = "cas"
                result["swapped"] = True
            else:
                result["op"] = "cas"
                result["swapped"] = False
                result["current"] = current
                result["ok"] = False

        elif op == "noop":
            result["op"] = "noop"

        else:
            result["ok"] = False
            result["error"] = f"Unknown operation: {op}"

        self.applied_index = index
        self.applied_commands.append((index, command, result))
        return result

    def snapshot(self) -> dict:
        """Create a snapshot of the current state."""
        return {
            "data": dict(self.data),
            "applied_index": self.applied_index,
        }

    def restore(self, snapshot: dict):
        """Restore state from a snapshot."""
        self.data = dict(snapshot["data"])
        self.applied_index = snapshot["applied_index"]
```

### 6.2 Verifying Determinism

```python
def verify_state_machine_determinism():
    """
    Verify that two state machines produce identical state
    when given the same commands in the same order.
    """
    commands = [
        {"op": "put", "key": "x", "value": "1"},
        {"op": "put", "key": "y", "value": "2"},
        {"op": "cas", "key": "x", "expected": "1", "value": "10"},
        {"op": "delete", "key": "y"},
        {"op": "put", "key": "z", "value": "3"},
        {"op": "cas", "key": "x", "expected": "999", "value": "bad"},
        {"op": "get", "key": "x"},
    ]

    sm1 = DeterministicKVStateMachine()
    sm2 = DeterministicKVStateMachine()

    for i, cmd in enumerate(commands, 1):
        r1 = sm1.apply(i, cmd)
        r2 = sm2.apply(i, cmd)
        assert r1 == r2, f"Divergence at index {i}: {r1} != {r2}"

    assert sm1.data == sm2.data
    print("Determinism verified: both state machines produced identical state")
    print(f"  Final state: {sm1.data}")
    print(f"  Applied {len(commands)} commands")


verify_state_machine_determinism()
```

---

## 7. Handling Edge Cases

### 7.1 Split Vote

A split vote occurs when no candidate receives a majority. Raft handles this through random timeouts:

```python
def simulate_split_vote():
    """
    Simulate a split vote scenario and demonstrate how
    Raft's randomized timeouts resolve it.
    """
    import random

    num_nodes = 5
    num_trials = 10000
    split_votes = 0
    rounds_to_elect = []

    for _ in range(num_trials):
        rounds = 0
        elected = False

        while not elected:
            rounds += 1
            # Each node picks a random timeout
            timeouts = [
                random.uniform(150, 300) for _ in range(num_nodes)
            ]

            # The node with the shortest timeout starts election first
            sorted_nodes = sorted(range(num_nodes), key=lambda i: timeouts[i])
            first = sorted_nodes[0]
            second = sorted_nodes[1]

            # If two nodes start within 10ms of each other, split vote likely
            if timeouts[second] - timeouts[first] < 10:
                split_votes += 1
                # Both start elections, split the votes
                continue
            else:
                # First node starts election, wins
                elected = True
                rounds_to_elect.append(rounds)

    avg_rounds = sum(rounds_to_elect) / len(rounds_to_elect) if rounds_to_elect else 0
    print(f"Split vote simulation ({num_trials} trials):")
    print(f"  Split votes: {split_votes}")
    print(f"  Avg rounds to elect: {avg_rounds:.2f}")
    print(f"  Max rounds: {max(rounds_to_elect) if rounds_to_elect else 0}")


simulate_split_vote()
```

### 7.2 Stale Leader Detection

A leader may be partitioned from the cluster but not know it:

```python
def demonstrate_stale_leader():
    """
    Show how a stale leader is detected and neutralized.

    When a network partition heals, the stale leader discovers
    a higher term and immediately steps down.
    """
    print("=== Stale Leader Detection ===\n")

    # Initial state: S1 is leader in term 1
    print("Phase 1: S1 is leader in term 1")
    print("  Cluster: [S1(leader,t=1), S2(follower,t=1), S3(follower,t=1)]")

    # Network partition: S1 is isolated
    print("\nPhase 2: Network partition isolates S1")
    print("  Partition A: [S1(leader,t=1)] ← thinks it's still leader")
    print("  Partition B: [S2, S3] ← elect new leader")

    # S2 wins election in partition B
    print("\nPhase 3: S2 wins election in term 2")
    print("  Partition A: [S1(leader,t=1)] ← stale leader")
    print("  Partition B: [S2(leader,t=2), S3(follower,t=2)]")

    # S1 tries to replicate — followers in partition B reject
    print("\nPhase 4: Partition heals")
    print("  S1 sends AppendEntries(term=1) to S2")
    print("  S2 replies with term=2 > 1")
    print("  S1 discovers higher term → steps down to follower")
    print("  S1 updates: term=2, state=follower, votedFor=None")
    print("\n  Final: [S1(follower,t=2), S2(leader,t=2), S3(follower,t=2)]")

    # Client impact
    print("\n--- Client Impact ---")
    print("  Writes to S1 during partition: NOT committed (no majority)")
    print("  Reads from S1 during partition: May return stale data!")
    print("  Solution: ReadIndex or LeaseRead (covered in Lesson 19)")


demonstrate_stale_leader()
```

### 7.3 Pre-Vote Extension

Pre-Vote prevents disruption from partitioned nodes that repeatedly increment their term:

```python
def demonstrate_prevote():
    """
    Demonstrate the Pre-Vote extension to Raft.

    Without Pre-Vote: A partitioned node increments its term on each
    election timeout. When the partition heals, its high term forces
    the stable leader to step down, disrupting the cluster.

    With Pre-Vote: A node first checks if it CAN win an election
    before incrementing its term.
    """
    print("=== Pre-Vote Extension ===\n")

    print("--- Without Pre-Vote ---")
    leader_term = 1
    partitioned_term = 1

    # Simulate 10 election timeouts while partitioned
    for i in range(10):
        partitioned_term += 1  # Each timeout increments term

    print(f"  After partition (10 timeouts):")
    print(f"    Leader term: {leader_term}")
    print(f"    Partitioned node term: {partitioned_term}")
    print(f"  Partition heals → partitioned node sends messages with term {partitioned_term}")
    print(f"  Leader steps down! (term {leader_term} < {partitioned_term})")
    print(f"  Result: Unnecessary leader disruption\n")

    print("--- With Pre-Vote ---")
    leader_term = 1
    partitioned_term = 1

    # With pre-vote, the partitioned node never increments its term
    # because it can't get pre-votes from a majority
    for i in range(10):
        # Pre-vote phase: ask peers "would you vote for me?"
        pre_votes = 0  # Can't reach peers → 0 pre-votes
        if pre_votes >= 1:  # Need majority - 1 (self counts)
            partitioned_term += 1  # Would increment, but never reaches here

    print(f"  After partition (10 timeouts):")
    print(f"    Leader term: {leader_term}")
    print(f"    Partitioned node term: {partitioned_term}")
    print(f"  Partition heals → terms are compatible")
    print(f"  Leader continues normally")
    print(f"  Result: No disruption!")


demonstrate_prevote()
```

---

## 8. Complete Implementation

### 8.1 Cluster Simulation

```python
class RaftCluster:
    """
    Simulated Raft cluster for testing.

    Handles message routing, network partitions, and timing.
    """

    def __init__(self, node_ids: list[str]):
        self.nodes: Dict[str, RaftNode] = {}
        self.partitions: list[set[str]] = []  # Empty = no partition
        self.message_queue: list[dict] = []
        self.dropped_messages: int = 0

        for nid in node_ids:
            peers = [p for p in node_ids if p != nid]
            self.nodes[nid] = RaftNode(nid, peers, f"/tmp/raft_test_{nid}.json")

    def tick_all(self):
        """Advance all nodes by one tick."""
        for node in self.nodes.values():
            node.tick()
        self._route_messages()

    def _route_messages(self):
        """Route messages between nodes, respecting partitions."""
        for node in self.nodes.values():
            while node.outbox:
                msg = node.outbox.pop(0)
                dest = msg.get("to")

                if self._can_communicate(msg["from"], dest):
                    if dest in self.nodes:
                        self.nodes[dest].receive(msg)
                else:
                    self.dropped_messages += 1

    def _can_communicate(self, src: str, dst: str) -> bool:
        """Check if two nodes can communicate (no partition between them)."""
        if not self.partitions:
            return True
        for partition in self.partitions:
            if src in partition and dst in partition:
                return True
        return False

    def partition(self, groups: list[list[str]]):
        """Create a network partition."""
        self.partitions = [set(g) for g in groups]

    def heal_partition(self):
        """Remove all network partitions."""
        self.partitions = []

    def get_leader(self) -> Optional[str]:
        """Find the current leader (if any)."""
        leaders = [
            nid for nid, node in self.nodes.items()
            if node.state == NodeState.LEADER
        ]
        return leaders[0] if len(leaders) == 1 else None

    def run_until_leader(self, max_ticks: int = 1000) -> Optional[str]:
        """Run the cluster until a leader is elected."""
        for _ in range(max_ticks):
            self.tick_all()
            leader = self.get_leader()
            if leader:
                return leader
            time.sleep(0.001)
        return None

    def status(self):
        """Print cluster status."""
        print("\n=== Cluster Status ===")
        for nid, node in sorted(self.nodes.items()):
            s = node.get_status()
            print(f"  {nid}: state={s['state']}, term={s['term']}, "
                  f"log={s['log_length']}, commit={s['commit_index']}")
        if self.partitions:
            print(f"  Partitions: {[list(p) for p in self.partitions]}")
        print()


def test_leader_election():
    """Test basic leader election with 5 nodes."""
    print("=== Test: Leader Election ===")

    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])
    leader = cluster.run_until_leader(max_ticks=500)

    if leader:
        print(f"Leader elected: {leader}")
        cluster.status()
    else:
        print("No leader elected within timeout")

    return cluster


def test_log_replication():
    """Test log replication across the cluster."""
    print("\n=== Test: Log Replication ===")

    cluster = RaftCluster(["n1", "n2", "n3"])
    leader_id = cluster.run_until_leader()

    if not leader_id:
        print("Failed to elect leader")
        return

    leader = cluster.nodes[leader_id]

    # Propose some commands
    commands = [
        {"op": "put", "key": "x", "value": "1"},
        {"op": "put", "key": "y", "value": "2"},
        {"op": "put", "key": "z", "value": "3"},
    ]

    for cmd in commands:
        leader.propose(cmd)

    # Run until committed
    for _ in range(200):
        cluster.tick_all()
        time.sleep(0.001)

    cluster.status()

    # Verify all nodes have the same committed state
    for nid, node in cluster.nodes.items():
        print(f"  {nid} state machine: {node.state_machine}")

    return cluster


if __name__ == "__main__":
    test_leader_election()
    test_log_replication()
```

---

## 9. Testing Leader Election

### 9.1 Property-Based Testing

```python
def test_election_safety_property(num_trials: int = 100):
    """
    Verify the Election Safety property across many trials:
    At most one leader per term.

    This is a property-based test that runs many random elections
    and checks the invariant after each one.
    """
    violations = 0

    for trial in range(num_trials):
        cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])

        # Run for a random number of ticks
        ticks = random.randint(50, 300)
        for _ in range(ticks):
            cluster.tick_all()

            # Check: at most one leader per term
            leaders_by_term: Dict[int, list] = {}
            for nid, node in cluster.nodes.items():
                if node.state == NodeState.LEADER:
                    term = node.current_term
                    if term not in leaders_by_term:
                        leaders_by_term[term] = []
                    leaders_by_term[term].append(nid)

            for term, leaders in leaders_by_term.items():
                if len(leaders) > 1:
                    print(f"VIOLATION in trial {trial}: "
                          f"term {term} has leaders {leaders}")
                    violations += 1

    print(f"\nElection Safety Test: {num_trials} trials, {violations} violations")
    return violations == 0
```

### 9.2 Deterministic Testing with Seed

```python
def test_with_seed(seed: int):
    """
    Run a deterministic test with a fixed random seed.

    This enables reproducible testing — if a test fails,
    the seed can be used to reproduce the exact scenario.
    """
    random.seed(seed)
    print(f"\n=== Deterministic Test (seed={seed}) ===")

    cluster = RaftCluster(["n1", "n2", "n3", "n4", "n5"])

    # Phase 1: Elect leader
    leader = cluster.run_until_leader(max_ticks=300)
    print(f"Phase 1: Leader={leader}")

    if not leader:
        print("  No leader elected — seed may produce pathological timing")
        return

    # Phase 2: Replicate entries
    cluster.nodes[leader].propose({"op": "put", "key": "k1", "value": "v1"})
    for _ in range(100):
        cluster.tick_all()

    # Phase 3: Partition the leader
    all_nodes = list(cluster.nodes.keys())
    minority = [leader]
    majority = [n for n in all_nodes if n != leader]
    cluster.partition([minority, majority])
    print(f"Phase 3: Partitioned {leader} from {majority}")

    # Phase 4: New leader in majority partition
    for _ in range(300):
        cluster.tick_all()

    new_leader = None
    for nid in majority:
        if cluster.nodes[nid].state == NodeState.LEADER:
            new_leader = nid
            break
    print(f"Phase 4: New leader={new_leader}")

    # Phase 5: Heal partition
    cluster.heal_partition()
    for _ in range(100):
        cluster.tick_all()

    cluster.status()

    # Verify: old leader stepped down
    old_leader_state = cluster.nodes[leader].state
    print(f"Phase 5: Old leader {leader} is now {old_leader_state.value}")
    assert old_leader_state != NodeState.LEADER or leader == new_leader


# Run with several seeds
for seed in [42, 123, 456, 789, 1000]:
    test_with_seed(seed)
```

---

## 10. Summary and Key Takeaways

### Implementation Checklist

> **RAFT IMPLEMENTATION PART 1 CHECKLIST**
>
> ☐ Persistent state saved to disk before responding to RPCs
> ☐ Election timer uses randomized timeouts
> ☐ RequestVote includes up-to-date log check
> ☐ AppendEntries consistency check with conflict detection
> ☐ Fast backup optimization for log divergence
> ☐ Commit advancement only for current term entries
> ☐ State machine applies entries in strict order
> ☐ Pre-Vote prevents disruption from partitioned nodes
> ☐ Split votes handled by timeout randomization

### Common Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Not persisting `votedFor` | Multiple leaders per term | Always fsync before RPC response |
| Committing old-term entries by counting | Safety violation | Only commit current-term entries directly |
| Not resetting timer on granting vote | Unnecessary elections | Reset timer in `handleRequestVote` when granting |
| 0-based vs 1-based log indexing | Off-by-one everywhere | Pick one convention and be consistent |
| Not checking term in RPC responses | Stale leader continues | Always check and step down on higher term |

---

## 11. Practice Problems

### Problem 1: Log Convergence

Given these logs after a series of partitions and elections, determine the minimum number of AppendEntries rounds needed for full convergence:

```
Leader (term 4): [1:a] [1:b] [2:c] [4:d] [4:e]
Follower A:      [1:a] [1:b] [3:x] [3:y]
Follower B:      [1:a] [1:b] [2:c]
Follower C:      [1:a]
```

### Problem 2: Safety Proof

Prove that the Log Matching Property is maintained by AppendEntries. Specifically, show that if the consistency check passes at index `prevLogIndex`, then all entries before that index must also match.

### Problem 3: Implementation Challenge

Implement a `RaftNode.check_invariants()` method that verifies all Raft safety properties hold at any point:
- `commitIndex <= len(log)`
- `lastApplied <= commitIndex`
- If leader: all entries after commitIndex have `term == currentTerm`
- If follower: `votedFor` is consistent with `currentTerm`

### Problem 4: Timing Analysis

A 5-node cluster has:
- Network RTT: 1ms (within datacenter)
- Election timeout: [150ms, 300ms]
- Heartbeat interval: 50ms

Calculate:
1. Worst-case time to elect a leader from a cold start
2. Maximum number of split vote rounds before convergence (probabilistic)
3. Time window during which a partitioned leader can accept (uncommitted) writes

### Problem 5: Pre-Vote Implementation

Implement the full Pre-Vote extension to the `RaftNode` class. A Pre-Vote RPC is identical to RequestVote but:
- Does not increment the sender's term
- Does not cause the receiver to step down
- Only if a majority of Pre-Votes are received does the node proceed to a real election

---

## 12. References

1. Ongaro, D. & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." *USENIX ATC*. (The Raft paper)
2. Ongaro, D. (2014). "Consensus: Bridging Theory and Practice." PhD Dissertation, Stanford University. (Extended Raft)
3. Howard, H. (2014). "ARC: Analysis of Raft Consensus." Cambridge Technical Report.
4. etcd/raft implementation: https://github.com/etcd-io/raft
5. TiKV/raft-rs implementation: https://github.com/tikv/raft-rs
6. Ongaro, D. (2015). "Bug in Single-Server Membership Changes." (Raft mailing list)
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 9. O'Reilly Media.

---

[Next: Lesson 19 — Raft Implementation Part 2](./19_Raft_Implementation_Part2.md)
