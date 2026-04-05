"""
Raft Consensus Algorithm Simulation

Demonstrates:
- Leader election with term numbers
- Log replication to followers
- Commit rules (majority acknowledgment)
- Leader failure and re-election

Theory:
- Raft is a consensus algorithm for managing a replicated log.
- Roles: Leader, Follower, Candidate.
- Leader election: candidates request votes; majority wins.
  Terms prevent stale leaders.
- Log replication: leader sends AppendEntries to followers.
  Entry committed when majority has it.
- Safety: at most one leader per term; committed entries are durable.

Adapted from System Design Lesson 16.
"""

import random
from enum import Enum
from dataclasses import dataclass, field


class Role(Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"


@dataclass
class LogEntry:
    term: int
    command: str
    index: int


@dataclass
class RaftNode:
    node_id: str
    role: Role = Role.FOLLOWER
    current_term: int = 0
    voted_for: str | None = None
    log: list[LogEntry] = field(default_factory=list)
    commit_index: int = -1
    leader_id: str | None = None
    alive: bool = True

    # Volatile leader state
    next_index: dict[str, int] = field(default_factory=dict)
    match_index: dict[str, int] = field(default_factory=dict)


# Why: This simulation models Raft's core invariants (leader election, log
# replication, majority commit) without real networking. This makes the
# algorithm's guarantees observable in a single-threaded, deterministic setting.
class RaftCluster:
    """Simplified Raft consensus simulation."""

    def __init__(self, node_ids: list[str]):
        self.nodes: dict[str, RaftNode] = {
            nid: RaftNode(node_id=nid) for nid in node_ids
        }
        self.event_log: list[str] = []

    def _log_event(self, msg: str) -> None:
        self.event_log.append(msg)

    @property
    def majority(self) -> int:
        return len(self.nodes) // 2 + 1

    def alive_nodes(self) -> list[RaftNode]:
        return [n for n in self.nodes.values() if n.alive]

    def get_leader(self) -> RaftNode | None:
        for n in self.nodes.values():
            if n.role == Role.LEADER and n.alive:
                return n
        return None

    # ── Leader Election ────────────────────────────────────────────

    def start_election(self, candidate_id: str) -> bool:
        """Node starts an election."""
        candidate = self.nodes[candidate_id]
        if not candidate.alive:
            return False

        # Why: Incrementing the term number before requesting votes ensures that
        # stale leaders from previous terms cannot interfere. Each term has at
        # most one leader, which is Raft's core safety guarantee.
        candidate.current_term += 1
        candidate.role = Role.CANDIDATE
        candidate.voted_for = candidate_id
        votes = 1  # Vote for self

        self._log_event(
            f"  [{candidate_id}] starts election for term {candidate.current_term}"
        )

        # Request votes from all other alive nodes
        for node in self.alive_nodes():
            if node.node_id == candidate_id:
                continue

            vote_granted = self._request_vote(candidate, node)
            if vote_granted:
                votes += 1

        self._log_event(
            f"  [{candidate_id}] received {votes}/{len(self.nodes)} votes"
        )

        if votes >= self.majority:
            self._become_leader(candidate)
            return True
        else:
            candidate.role = Role.FOLLOWER
            candidate.voted_for = None
            return False

    def _request_vote(self, candidate: RaftNode, voter: RaftNode) -> bool:
        """Voter decides whether to grant vote."""
        # Why: Each node votes for at most one candidate per term. This ensures
        # two candidates cannot both win a majority in the same term — preventing
        # the "split brain" problem of having two simultaneous leaders.
        if voter.current_term == candidate.current_term and voter.voted_for is not None:
            return False

        # If candidate's term is lower, deny
        if candidate.current_term < voter.current_term:
            return False

        # Update voter's term
        if candidate.current_term > voter.current_term:
            voter.current_term = candidate.current_term
            voter.voted_for = None
            voter.role = Role.FOLLOWER

        # Why: The "election restriction" — a voter only grants its vote if the
        # candidate's log is at least as up-to-date. This ensures the elected
        # leader already has all committed entries, avoiding data loss.
        candidate_last = candidate.log[-1].term if candidate.log else 0
        voter_last = voter.log[-1].term if voter.log else 0

        if candidate_last >= voter_last:
            voter.voted_for = candidate.node_id
            return True
        return False

    def _become_leader(self, node: RaftNode) -> None:
        node.role = Role.LEADER
        node.leader_id = node.node_id

        # Initialize leader state
        next_idx = len(node.log)
        for n in self.nodes.values():
            node.next_index[n.node_id] = next_idx
            node.match_index[n.node_id] = -1
            if n.node_id != node.node_id and n.alive:
                n.leader_id = node.node_id

        self._log_event(
            f"  [{node.node_id}] becomes LEADER for term {node.current_term}"
        )

    # ── Log Replication ────────────────────────────────────────────

    def client_request(self, command: str) -> bool:
        """Client sends a command to the leader."""
        leader = self.get_leader()
        if not leader:
            self._log_event(f"  No leader available for '{command}'")
            return False

        # Append to leader's log
        entry = LogEntry(
            term=leader.current_term,
            command=command,
            index=len(leader.log),
        )
        leader.log.append(entry)
        leader.match_index[leader.node_id] = entry.index

        self._log_event(
            f"  [{leader.node_id}] appends '{command}' at index {entry.index}"
        )

        # Replicate to followers
        acks = 1  # Leader counts itself
        for node in self.alive_nodes():
            if node.node_id == leader.node_id:
                continue

            success = self._append_entries(leader, node, entry)
            if success:
                acks += 1

        self._log_event(
            f"  [{leader.node_id}] '{command}' acked by {acks}/{len(self.nodes)}"
        )

        # Why: An entry is committed only when a majority of nodes have it.
        # This guarantees that any future leader will have the entry (since
        # majorities always overlap), making committed entries durable.
        if acks >= self.majority:
            leader.commit_index = entry.index
            self._log_event(
                f"  [{leader.node_id}] committed index {entry.index}"
            )
            # Notify followers of commit
            for node in self.alive_nodes():
                if node.node_id != leader.node_id:
                    node.commit_index = min(
                        entry.index, len(node.log) - 1
                    )
            return True
        return False

    def _append_entries(self, leader: RaftNode, follower: RaftNode,
                        entry: LogEntry) -> bool:
        """Leader sends AppendEntries to a follower."""
        if not follower.alive:
            return False

        # Simplified: just append if term matches
        if follower.current_term > leader.current_term:
            return False

        follower.current_term = leader.current_term
        follower.leader_id = leader.node_id
        follower.role = Role.FOLLOWER

        # Append entry
        if entry.index < len(follower.log):
            follower.log[entry.index] = entry
        else:
            follower.log.append(entry)

        leader.match_index[follower.node_id] = entry.index
        leader.next_index[follower.node_id] = entry.index + 1
        return True

    # ── Node Management ────────────────────────────────────────────

    def kill_node(self, node_id: str) -> None:
        self.nodes[node_id].alive = False
        self.nodes[node_id].role = Role.FOLLOWER
        self._log_event(f"  [{node_id}] CRASHED")

    def revive_node(self, node_id: str) -> None:
        self.nodes[node_id].alive = True
        self._log_event(f"  [{node_id}] REVIVED")

    def print_state(self) -> None:
        print(f"\n    {'Node':<8} {'Role':<12} {'Term':>5} {'Log Len':>8} "
              f"{'Commit':>7} {'Alive':>6}")
        print(f"    {'-'*8} {'-'*12} {'-'*5} {'-'*8} {'-'*7} {'-'*6}")
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            print(f"    {nid:<8} {n.role.value:<12} {n.current_term:>5} "
                  f"{len(n.log):>8} {n.commit_index:>7} "
                  f"{'Yes' if n.alive else 'No':>6}")


# ── Demos ──────────────────────────────────────────────────────────────

def demo_election():
    print("=" * 60)
    print("RAFT LEADER ELECTION")
    print("=" * 60)

    cluster = RaftCluster(["N1", "N2", "N3", "N4", "N5"])

    print(f"\n  5-node cluster, majority = {cluster.majority}")

    # N1 starts election
    print(f"\n  --- Election Round 1 ---")
    won = cluster.start_election("N1")
    print(f"  N1 won? {won}")
    cluster.print_state()

    for msg in cluster.event_log:
        print(msg)


def demo_log_replication():
    print("\n" + "=" * 60)
    print("RAFT LOG REPLICATION")
    print("=" * 60)

    cluster = RaftCluster(["N1", "N2", "N3", "N4", "N5"])
    cluster.start_election("N1")
    cluster.event_log.clear()

    # Client requests
    print(f"\n  Sending commands to leader:")
    for cmd in ["SET x=1", "SET y=2", "SET z=3"]:
        committed = cluster.client_request(cmd)
        print(f"    '{cmd}' → {'committed' if committed else 'not committed'}")

    cluster.print_state()

    # Show logs
    print(f"\n  Log contents:")
    for nid in sorted(cluster.nodes):
        n = cluster.nodes[nid]
        entries = [(e.index, e.command) for e in n.log]
        print(f"    {nid}: {entries}")


def demo_leader_failure():
    print("\n" + "=" * 60)
    print("RAFT LEADER FAILURE & RE-ELECTION")
    print("=" * 60)

    cluster = RaftCluster(["N1", "N2", "N3", "N4", "N5"])
    cluster.start_election("N1")
    cluster.event_log.clear()

    # Replicate some entries
    cluster.client_request("SET x=1")
    cluster.client_request("SET y=2")
    cluster.event_log.clear()

    print(f"\n  State with N1 as leader (2 committed entries):")
    cluster.print_state()

    # Kill leader
    print(f"\n  --- N1 crashes! ---")
    cluster.kill_node("N1")

    # N3 starts election
    print(f"\n  --- N3 starts election ---")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    won = cluster.start_election("N3")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    print(f"\n  N3 elected? {won}")
    cluster.print_state()

    # New leader accepts commands
    print(f"\n  --- New leader accepts commands ---")
    committed = cluster.client_request("SET z=3")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    cluster.print_state()


def demo_minority_partition():
    print("\n" + "=" * 60)
    print("RAFT: MINORITY CANNOT COMMIT")
    print("=" * 60)

    cluster = RaftCluster(["N1", "N2", "N3", "N4", "N5"])
    cluster.start_election("N1")
    cluster.event_log.clear()

    # Kill 3 nodes — only N1, N2 remain
    print(f"\n  Killing N3, N4, N5 (leader N1 + N2 remain):")
    cluster.kill_node("N3")
    cluster.kill_node("N4")
    cluster.kill_node("N5")

    committed = cluster.client_request("SET x=1")
    for msg in cluster.event_log:
        print(msg)
    cluster.event_log.clear()

    print(f"\n  Committed? {committed} (need {cluster.majority} acks, only 2 alive)")
    cluster.print_state()

    # Revive nodes
    print(f"\n  --- Reviving N3, N4 ---")
    cluster.revive_node("N3")
    cluster.revive_node("N4")

    committed = cluster.client_request("SET y=2")
    for msg in cluster.event_log:
        print(msg)

    print(f"\n  Now committed? {committed}")
    cluster.print_state()


if __name__ == "__main__":
    demo_election()
    demo_log_replication()
    demo_leader_failure()
    demo_minority_partition()
