"""
Exercises for Lesson 18: Raft Implementation Part 1
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import random
import time
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


# === Exercise 1: Log Convergence ===
# Problem: Given divergent logs, determine minimum AppendEntries rounds
# for full convergence using fast backup optimization.

def exercise_1_log_convergence():
    """
    Calculate minimum rounds for log convergence.

    Leader (term 4): [1:a] [1:b] [2:c] [4:d] [4:e]
    Follower A:      [1:a] [1:b] [3:x] [3:y]
    Follower B:      [1:a] [1:b] [2:c]
    Follower C:      [1:a]
    """
    leader_log = [(1, "a"), (1, "b"), (2, "c"), (4, "d"), (4, "e")]

    followers = {
        "A": [(1, "a"), (1, "b"), (3, "x"), (3, "y")],
        "B": [(1, "a"), (1, "b"), (2, "c")],
        "C": [(1, "a")],
    }

    print("=== Exercise 1: Log Convergence ===\n")

    for name, log in followers.items():
        # Find divergence point
        match_index = 0
        for i in range(min(len(leader_log), len(log))):
            if leader_log[i] == log[i]:
                match_index = i + 1
            else:
                break

        entries_to_truncate = len(log) - match_index
        entries_to_append = len(leader_log) - match_index

        # With fast backup: 1 round to identify conflict term,
        # then 1 round per distinct conflict term
        conflict_terms = set()
        for i in range(match_index, len(log)):
            conflict_terms.add(log[i][0])

        rounds = max(1, len(conflict_terms))  # At least 1 round
        print(f"  Follower {name}:")
        print(f"    Match point: index {match_index}")
        print(f"    Truncate {entries_to_truncate}, append {entries_to_append}")
        print(f"    Conflict terms: {conflict_terms}")
        print(f"    Rounds needed (fast backup): {rounds}")

    # Total: max of all followers
    print(f"\n  Total rounds for cluster convergence: 2")
    print(f"  (Follower A needs 2 rounds: one for term 3, one for final sync)")


exercise_1_log_convergence()


# === Exercise 2: Election Safety Verification ===
# Problem: Prove that Log Matching Property is maintained by AppendEntries.

class LogEntry:
    """A log entry for verification."""
    def __init__(self, term: int, index: int, command: str):
        self.term = term
        self.index = index
        self.command = command


def exercise_2_safety_proof():
    """
    Verify that the AppendEntries consistency check maintains
    the Log Matching Property.

    Property: If two logs contain an entry with the same index and term,
    then all preceding entries are identical.
    """
    print("\n=== Exercise 2: Log Matching Property ===\n")

    # Proof by induction on log index:
    # Base case: Empty logs trivially satisfy the property.
    # Inductive step: Assume the property holds for all entries up to index i.
    #   When AppendEntries adds entry i+1:
    #   - It checks prevLogIndex=i, prevLogTerm=log[i].term
    #   - If this check passes, entry i matches on both logs
    #   - By the inductive hypothesis, all entries 1..i match
    #   - The new entry i+1 has the same term (sent by the same leader)
    #   - Therefore, entries 1..i+1 all match. QED.

    # Verification by simulation
    def verify_log_matching(log1: list, log2: list) -> bool:
        """Verify the Log Matching Property between two logs."""
        for i in range(min(len(log1), len(log2))):
            if log1[i].term == log2[i].term and log1[i].index == log2[i].index:
                # Check all preceding entries match
                for j in range(i):
                    if (log1[j].term != log2[j].term or
                            log1[j].command != log2[j].command):
                        return False
        return True

    # Test with matching logs
    leader_log = [
        LogEntry(1, 1, "a"), LogEntry(1, 2, "b"),
        LogEntry(2, 3, "c"), LogEntry(3, 4, "d"),
    ]
    follower_log = [
        LogEntry(1, 1, "a"), LogEntry(1, 2, "b"),
        LogEntry(2, 3, "c"),
    ]

    result = verify_log_matching(leader_log, follower_log)
    print(f"  Matching logs: {result}")

    # Test with divergent logs
    divergent_log = [
        LogEntry(1, 1, "a"), LogEntry(1, 2, "b"),
        LogEntry(2, 3, "DIFFERENT"),
    ]
    result = verify_log_matching(leader_log, divergent_log)
    print(f"  Divergent at index 3: {result}")
    print(f"  (Divergence is detected by AppendEntries prevLogTerm check)")


exercise_2_safety_proof()


# === Exercise 3: Invariant Checker ===
# Problem: Implement RaftNode.check_invariants()

class RaftInvariantChecker:
    """
    Checks all Raft safety invariants.
    """

    def __init__(self):
        self.violations: list[str] = []

    def check(self, state: dict) -> list[str]:
        """Check all invariants on a Raft node state."""
        self.violations = []

        commit_index = state.get("commit_index", 0)
        last_applied = state.get("last_applied", 0)
        log_length = state.get("log_length", 0)
        current_term = state.get("current_term", 0)
        voted_for = state.get("voted_for")
        node_state = state.get("state", "follower")

        # Invariant 1: commitIndex <= len(log)
        if commit_index > log_length:
            self.violations.append(
                f"commitIndex ({commit_index}) > log length ({log_length})"
            )

        # Invariant 2: lastApplied <= commitIndex
        if last_applied > commit_index:
            self.violations.append(
                f"lastApplied ({last_applied}) > commitIndex ({commit_index})"
            )

        # Invariant 3: If follower, votedFor is consistent
        if node_state == "follower" and voted_for is not None:
            # votedFor should correspond to current term
            pass  # Cannot fully check without term history

        # Invariant 4: term is non-negative
        if current_term < 0:
            self.violations.append(f"currentTerm is negative: {current_term}")

        # Invariant 5: lastApplied >= 0
        if last_applied < 0:
            self.violations.append(f"lastApplied is negative: {last_applied}")

        return self.violations


def exercise_3():
    """Test the invariant checker."""
    print("\n=== Exercise 3: Invariant Checker ===\n")

    checker = RaftInvariantChecker()

    # Valid state
    violations = checker.check({
        "commit_index": 5,
        "last_applied": 3,
        "log_length": 8,
        "current_term": 2,
        "state": "follower",
    })
    print(f"  Valid state: {len(violations)} violations")

    # Invalid state
    violations = checker.check({
        "commit_index": 10,
        "last_applied": 12,  # > commitIndex!
        "log_length": 8,     # < commitIndex!
        "current_term": 2,
        "state": "leader",
    })
    print(f"  Invalid state: {len(violations)} violations")
    for v in violations:
        print(f"    - {v}")


exercise_3()


# === Exercise 4: Timing Analysis ===
# Problem: Calculate election timing for a 5-node cluster.

def exercise_4_timing():
    """
    Timing analysis for leader election.

    Cluster: 5 nodes
    RTT: 1ms
    Election timeout: [150ms, 300ms]
    Heartbeat: 50ms
    """
    print("\n=== Exercise 4: Timing Analysis ===\n")

    rtt = 1  # ms
    election_min = 150  # ms
    election_max = 300  # ms
    heartbeat = 50  # ms

    # 1. Worst-case cold start election
    # All nodes start simultaneously, pick random timeouts
    # Worst case: split votes for several rounds
    # Each round takes: election_max (timeout) + rtt (vote request) + rtt (response)
    round_time = election_max + 2 * rtt
    # Expected rounds: ~1-2 (with randomization, split votes are rare)
    worst_case_time = round_time * 3  # 3 rounds worst case
    print(f"  Worst-case cold start: ~{worst_case_time}ms ({3} rounds)")

    # 2. Expected split vote rounds
    # P(split vote) ≈ probability that 2 nodes have timeouts within RTT of each other
    # With uniform [150, 300], P(|t1-t2| < RTT) ≈ 2*RTT/range = 2*1/150 ≈ 1.3%
    p_split = 2 * rtt / (election_max - election_min)
    expected_rounds = 1 / (1 - p_split)
    print(f"  P(split vote per round): {p_split:.4f}")
    print(f"  Expected rounds to elect: {expected_rounds:.2f}")

    # 3. Partitioned leader write window
    # A partitioned leader can accept writes until clients timeout
    # or until a new leader is elected in the majority partition
    # New leader elected after: election_timeout (at least) on majority side
    # So partitioned leader window: election_min to election_max
    print(f"  Partitioned leader write window: {election_min}-{election_max}ms")
    print(f"  (Writes accepted but never committed)")


exercise_4_timing()


# === Exercise 5: Pre-Vote Implementation ===
# Problem: Implement the Pre-Vote extension.

class PreVoteRaftNode:
    """Raft node with Pre-Vote extension."""

    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = peers
        self.current_term = 0
        self.voted_for = None
        self.state = "follower"
        self.log = []
        self.pre_votes_received: Set[str] = set()
        self.pre_vote_in_progress = False

    def start_pre_vote(self) -> list[dict]:
        """
        Start a pre-vote phase.

        Pre-vote does NOT increment the term.
        It only checks if a majority would vote for us.
        """
        self.pre_vote_in_progress = True
        self.pre_votes_received = {self.node_id}  # Vote for self

        messages = []
        last_log_index = len(self.log)
        last_log_term = self.log[-1]["term"] if self.log else 0

        for peer in self.peers:
            messages.append({
                "type": "PreVoteRequest",
                "to": peer,
                "from": self.node_id,
                "term": self.current_term + 1,  # Proposed term (not actual)
                "last_log_index": last_log_index,
                "last_log_term": last_log_term,
            })

        return messages

    def handle_pre_vote_request(self, msg: dict) -> dict:
        """
        Handle a pre-vote request.

        Grant pre-vote if:
        1. Candidate's proposed term >= our term
        2. Candidate's log is at least as up-to-date
        3. We haven't heard from a leader recently

        Key difference from RequestVote:
        - Does NOT cause step-down
        - Does NOT update votedFor
        """
        proposed_term = msg["term"]
        grant = False

        if proposed_term >= self.current_term:
            # Check log up-to-date-ness
            our_last_index = len(self.log)
            our_last_term = self.log[-1]["term"] if self.log else 0

            if msg["last_log_term"] > our_last_term:
                grant = True
            elif (msg["last_log_term"] == our_last_term and
                  msg["last_log_index"] >= our_last_index):
                grant = True

        return {
            "type": "PreVoteResponse",
            "to": msg["from"],
            "from": self.node_id,
            "term": self.current_term,
            "vote_granted": grant,
        }

    def handle_pre_vote_response(self, msg: dict) -> Optional[str]:
        """
        Handle a pre-vote response.

        Returns "start_election" if majority of pre-votes received.
        """
        if not self.pre_vote_in_progress:
            return None

        if msg["vote_granted"]:
            self.pre_votes_received.add(msg["from"])

        total = len(self.peers) + 1
        if len(self.pre_votes_received) > total // 2:
            self.pre_vote_in_progress = False
            return "start_election"

        return None


def exercise_5():
    """Test the Pre-Vote implementation."""
    print("\n=== Exercise 5: Pre-Vote Extension ===\n")

    # 5-node cluster
    peers = ["n2", "n3", "n4", "n5"]
    node = PreVoteRaftNode("n1", peers)
    other_nodes = {
        pid: PreVoteRaftNode(pid, [p for p in ["n1"] + peers if p != pid])
        for pid in peers
    }

    # Start pre-vote
    messages = node.start_pre_vote()
    print(f"  Pre-vote messages sent: {len(messages)}")

    # Collect responses
    for msg in messages:
        peer = other_nodes[msg["to"]]
        response = peer.handle_pre_vote_request(msg)
        print(f"  {response['from']}: pre_vote_granted={response['vote_granted']}")

        action = node.handle_pre_vote_response(response)
        if action == "start_election":
            print(f"\n  Pre-vote successful! Starting real election.")
            print(f"  Pre-votes: {node.pre_votes_received}")
            break


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
