"""
Exercises for Lesson 06: Raft in Depth
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# === Exercise 1: Raft Log Compaction and Snapshot Installation ===
# Problem: Implement Raft log compaction via snapshotting. When the
# log grows beyond a threshold, create a snapshot of the state machine
# and discard the compacted log entries. Implement InstallSnapshot RPC
# for slow followers.

@dataclass
class LogEntry:
    """A single entry in the Raft log."""
    term: int
    index: int
    command: str


@dataclass
class Snapshot:
    """A state machine snapshot."""
    last_included_index: int
    last_included_term: int
    state: Dict[str, int]  # the snapshot of the state machine


class RaftNodeWithCompaction:
    """
    Raft node with log compaction support.
    """

    def __init__(self, node_id: str, compaction_threshold: int = 5):
        self.node_id = node_id
        self.log: List[LogEntry] = []
        self.state_machine: Dict[str, int] = {}
        self.snapshot: Optional[Snapshot] = None
        self.compaction_threshold = compaction_threshold
        self.commit_index = 0
        self.last_applied = 0
        self.log_offset = 0  # base index after compaction

    def append_entry(self, term: int, command: str):
        """Append a new log entry."""
        index = self.log_offset + len(self.log) + 1
        self.log.append(LogEntry(term, index, command))
        return index

    def apply_to_state_machine(self, entry: LogEntry):
        """Apply a log entry command (key=value format)."""
        if "=" in entry.command:
            key, val = entry.command.split("=")
            self.state_machine[key.strip()] = int(val.strip())

    def commit_and_apply(self, commit_index: int):
        """Commit entries up to commit_index and apply to state machine."""
        self.commit_index = commit_index
        while self.last_applied < commit_index:
            self.last_applied += 1
            log_idx = self.last_applied - self.log_offset - 1
            if 0 <= log_idx < len(self.log):
                self.apply_to_state_machine(self.log[log_idx])

        # Check if compaction is needed
        if len(self.log) >= self.compaction_threshold:
            self.compact()

    def compact(self):
        """Create snapshot and discard compacted log entries."""
        if not self.log:
            return
        snap_index = self.last_applied
        snap_log_idx = snap_index - self.log_offset - 1
        if snap_log_idx < 0:
            return

        snap_entry = self.log[snap_log_idx]
        self.snapshot = Snapshot(
            last_included_index=snap_index,
            last_included_term=snap_entry.term,
            state=dict(self.state_machine),
        )

        # Discard compacted entries
        self.log = self.log[snap_log_idx + 1:]
        self.log_offset = snap_index

    def install_snapshot(self, snapshot: Snapshot) -> bool:
        """
        InstallSnapshot RPC: replace state with snapshot if it is
        more recent.
        """
        if self.snapshot and snapshot.last_included_index <= self.snapshot.last_included_index:
            return False  # stale snapshot

        self.snapshot = Snapshot(
            last_included_index=snapshot.last_included_index,
            last_included_term=snapshot.last_included_term,
            state=dict(snapshot.state),
        )
        self.state_machine = dict(snapshot.state)
        self.last_applied = snapshot.last_included_index
        self.commit_index = max(self.commit_index, snapshot.last_included_index)

        # Discard all log entries covered by snapshot
        new_log = []
        for entry in self.log:
            if entry.index > snapshot.last_included_index:
                new_log.append(entry)
        self.log = new_log
        self.log_offset = snapshot.last_included_index

        return True


def exercise_1():
    """
    Demonstrate log compaction and snapshot installation.
    """
    print("=== Exercise 1: Raft Log Compaction & Snapshots ===\n")

    leader = RaftNodeWithCompaction("leader", compaction_threshold=5)

    # Append and commit entries
    for i in range(8):
        leader.append_entry(1, f"x={i*10}")

    print(f"Log before compaction: {len(leader.log)} entries")
    leader.commit_and_apply(8)
    print(f"Log after compaction:  {len(leader.log)} entries")
    print(f"Snapshot: index={leader.snapshot.last_included_index}, "
          f"state={leader.snapshot.state}")
    print(f"State machine: {leader.state_machine}")

    # Slow follower needs snapshot
    follower = RaftNodeWithCompaction("follower", compaction_threshold=5)
    print(f"\nFollower log: {len(follower.log)} entries")
    print(f"Installing snapshot on follower...")
    follower.install_snapshot(leader.snapshot)
    print(f"Follower state after snapshot: {follower.state_machine}")
    print(f"Follower last_applied: {follower.last_applied}")
    assert follower.state_machine == leader.state_machine
    print("\nLog compaction and snapshot installation verified.")
    print()


# === Exercise 2: Pre-Vote Protocol ===
# Problem: Implement the Pre-Vote extension to Raft that prevents
# disruption from partitioned nodes. Before starting an election,
# a node sends PreVote requests. It only becomes a candidate if a
# majority responds that they would vote for it.

class RaftState(Enum):
    FOLLOWER = "follower"
    PRE_CANDIDATE = "pre_candidate"
    CANDIDATE = "candidate"
    LEADER = "leader"


class RaftNodeWithPreVote:
    """Raft node with Pre-Vote protocol extension."""

    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.state = RaftState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log_length = 0
        self.last_log_term = 0
        self.leader_id: Optional[str] = None

    def handle_pre_vote_request(
        self, candidate_id: str, candidate_term: int,
        candidate_log_length: int, candidate_last_term: int,
    ) -> Tuple[bool, int]:
        """
        Handle a PreVote request. Grant if:
        1. Candidate's term >= our term
        2. Candidate's log is at least as up-to-date as ours
        3. We haven't heard from a leader recently (simulated)
        """
        # Check term
        if candidate_term < self.current_term:
            return (False, self.current_term)

        # Check log up-to-date
        log_ok = (
            candidate_last_term > self.last_log_term
            or (
                candidate_last_term == self.last_log_term
                and candidate_log_length >= self.log_length
            )
        )

        return (log_ok, self.current_term)

    def start_pre_vote(
        self, peer_responses: Dict[str, Tuple[bool, int]]
    ) -> bool:
        """
        Initiate Pre-Vote phase. Returns True if we should proceed
        to a real election.
        """
        self.state = RaftState.PRE_CANDIDATE
        pre_votes = 1  # vote for self
        max_term = self.current_term

        for peer_id, (granted, peer_term) in peer_responses.items():
            max_term = max(max_term, peer_term)
            if granted:
                pre_votes += 1

        majority = (len(self.peers) + 1) // 2 + 1
        if pre_votes >= majority:
            # Proceed to real election
            self.state = RaftState.CANDIDATE
            self.current_term = max_term + 1
            self.voted_for = self.node_id
            return True
        else:
            self.state = RaftState.FOLLOWER
            return False


def exercise_2():
    """
    Demonstrate Pre-Vote preventing disruption from partitioned nodes.
    """
    print("=== Exercise 2: Pre-Vote Protocol ===\n")

    peers = ["N1", "N2", "N3", "N4", "N5"]

    # Scenario 1: Node in a partition tries pre-vote
    partitioned = RaftNodeWithPreVote("N5", ["N1", "N2", "N3", "N4"])
    partitioned.current_term = 10  # bumped term while partitioned

    # Other nodes are at term 5 with a valid leader
    other_nodes = [RaftNodeWithPreVote(f"N{i}", peers) for i in range(1, 5)]
    for n in other_nodes:
        n.current_term = 5
        n.leader_id = "N1"

    # Partitioned node tries pre-vote (but its log is behind)
    responses = {}
    for n in other_nodes:
        granted, term = n.handle_pre_vote_request(
            "N5", partitioned.current_term + 1,
            partitioned.log_length, partitioned.last_log_term,
        )
        responses[n.node_id] = (granted, term)

    proceed = partitioned.start_pre_vote(responses)
    print(f"Scenario 1: Partitioned node N5 (term={10}) tries pre-vote")
    print(f"  Pre-vote responses: {responses}")
    print(f"  Proceed to election: {proceed}")
    print(f"  N5 state: {partitioned.state.value}")

    # Scenario 2: Valid candidate with up-to-date log
    print()
    candidate = RaftNodeWithPreVote("N2", ["N1", "N3", "N4", "N5"])
    candidate.current_term = 5
    candidate.log_length = 10
    candidate.last_log_term = 5

    for n in other_nodes:
        n.log_length = 10
        n.last_log_term = 5

    responses2 = {}
    for n in other_nodes:
        if n.node_id != candidate.node_id:
            granted, term = n.handle_pre_vote_request(
                "N2", candidate.current_term + 1,
                candidate.log_length, candidate.last_log_term,
            )
            responses2[n.node_id] = (granted, term)

    proceed2 = candidate.start_pre_vote(responses2)
    print(f"Scenario 2: Valid candidate N2 (term=5, log=10) tries pre-vote")
    print(f"  Pre-vote responses: {responses2}")
    print(f"  Proceed to election: {proceed2}")
    print(f"  N2 state: {candidate.state.value}")
    print()


# === Exercise 3: ReadIndex for Linearizable Reads ===
# Problem: Implement the ReadIndex protocol that allows a Raft leader
# to serve linearizable reads without writing a log entry. The leader
# must confirm it is still the leader by exchanging heartbeats with
# a majority before serving the read.

class RaftLeaderWithReadIndex:
    """Raft leader that supports ReadIndex for linearizable reads."""

    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.commit_index = 0
        self.state_machine: Dict[str, int] = {}
        self.is_leader = True
        self.pending_reads: List[Tuple[int, str]] = []  # (read_index, key)

    def handle_read_request(self, key: str) -> Optional[int]:
        """
        Handle a read request using the ReadIndex protocol.

        Steps:
        1. Record the current commit index as the read index.
        2. Send heartbeats to confirm leadership.
        3. Wait until the state machine has applied up to read index.
        4. Return the value.
        """
        if not self.is_leader:
            return None

        read_index = self.commit_index
        # Step 2: Confirm leadership via heartbeats
        if not self._confirm_leadership():
            return None

        # Step 3: State machine must be at least at read_index
        # (In this simulation, it already is)

        # Step 4: Serve the read
        return self.state_machine.get(key)

    def _confirm_leadership(self) -> bool:
        """
        Confirm leadership by checking heartbeat responses from a majority.
        Returns True if a majority responds.
        """
        # Simulate heartbeat responses
        acks = 1  # self
        for peer in self.peers:
            # In real implementation, this would be an actual RPC
            if random.random() < 0.9:  # 90% chance peer responds
                acks += 1

        majority = (len(self.peers) + 1) // 2 + 1
        return acks >= majority

    def apply_command(self, command: str, index: int):
        """Apply a command to the state machine."""
        if "=" in command:
            key, val = command.split("=")
            self.state_machine[key.strip()] = int(val.strip())
        self.commit_index = max(self.commit_index, index)


def exercise_3():
    """
    Demonstrate ReadIndex for linearizable reads.
    """
    print("=== Exercise 3: ReadIndex for Linearizable Reads ===\n")

    random.seed(42)
    leader = RaftLeaderWithReadIndex("L1", ["F1", "F2", "F3", "F4"])

    # Apply some commands
    commands = ["x=10", "y=20", "z=30"]
    for i, cmd in enumerate(commands, 1):
        leader.apply_command(cmd, i)

    print(f"State machine: {leader.state_machine}")
    print(f"Commit index: {leader.commit_index}")

    # ReadIndex reads
    for key in ["x", "y", "z", "w"]:
        result = leader.handle_read_request(key)
        print(f"ReadIndex read({key}) = {result}")

    # Demonstrate what happens if leader loses leadership
    print("\nLeader loses leadership...")
    leader.is_leader = False
    result = leader.handle_read_request("x")
    print(f"ReadIndex read(x) = {result} (rejected: not leader)")
    assert result is None

    print("\nReadIndex protocol avoids log writes for read-only requests.")
    print("It only requires a heartbeat round-trip to confirm leadership.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
