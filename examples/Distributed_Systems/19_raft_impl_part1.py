"""
Raft Implementation Part 1: Leader Election State Machine

A detailed implementation of Raft's leader election mechanism including
term management, randomized election timeouts, pre-vote protocol, and
step-down on higher terms. This goes deeper than the basic example in
03_raft_leader_election.py with edge cases and the PreVote extension.

Key concepts:
- Raft state transitions: Follower -> Candidate -> Leader
- Randomized election timeouts to prevent split votes
- Term-based leader precedence
- PreVote extension to prevent disruptive elections
- Split vote scenarios and resolution

Usage:
    python 19_raft_impl_part1.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class RaftRole(Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"


@dataclass
class VoteRequest:
    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int
    is_pre_vote: bool = False


@dataclass
class VoteResponse:
    term: int
    vote_granted: bool
    voter_id: int
    is_pre_vote: bool = False


class RaftNode:
    """Raft node with detailed election logic."""

    def __init__(self, node_id: int, cluster_size: int,
                 election_timeout_range: tuple[int, int] = (150, 300)):
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.role = RaftRole.FOLLOWER
        self.current_term = 0
        self.voted_for: int | None = None
        self.leader_id: int | None = None

        # Log state (simplified)
        self.last_log_index = 0
        self.last_log_term = 0

        # Timing
        self.election_timeout_range = election_timeout_range
        self.election_timeout = 0
        self.elapsed = 0

        # PreVote support
        self.pre_vote_enabled = False

        self.log: list[str] = []
        self._reset_election_timer()

    def _reset_election_timer(self) -> None:
        lo, hi = self.election_timeout_range
        self.election_timeout = random.randint(lo, hi)
        self.elapsed = 0

    def tick(self) -> VoteRequest | None:
        """Advance time by 1ms. May trigger election."""
        if self.role == RaftRole.LEADER:
            return None

        self.elapsed += 1
        if self.elapsed >= self.election_timeout:
            return self._start_election()
        return None

    def _start_election(self) -> VoteRequest:
        """Start an election (or pre-vote)."""
        if self.pre_vote_enabled:
            # PreVote: ask without incrementing term
            self.log.append(
                f"Starting PreVote for term {self.current_term + 1}")
            return VoteRequest(
                term=self.current_term + 1,
                candidate_id=self.node_id,
                last_log_index=self.last_log_index,
                last_log_term=self.last_log_term,
                is_pre_vote=True,
            )

        self.current_term += 1
        self.role = RaftRole.CANDIDATE
        self.voted_for = self.node_id
        self._reset_election_timer()

        self.log.append(f"Starting election for term {self.current_term}")

        return VoteRequest(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=self.last_log_index,
            last_log_term=self.last_log_term,
        )

    def handle_vote_request(self, req: VoteRequest) -> VoteResponse:
        """Handle an incoming VoteRequest."""
        if req.is_pre_vote:
            # PreVote: respond based on whether we WOULD vote, without state changes
            would_grant = (
                req.term > self.current_term
                and (req.last_log_term > self.last_log_term
                     or (req.last_log_term == self.last_log_term
                         and req.last_log_index >= self.last_log_index))
            )
            return VoteResponse(self.current_term, would_grant,
                                self.node_id, is_pre_vote=True)

        # Step down if we see a higher term
        if req.term > self.current_term:
            self.current_term = req.term
            self.role = RaftRole.FOLLOWER
            self.voted_for = None
            self.leader_id = None

        if req.term < self.current_term:
            return VoteResponse(self.current_term, False, self.node_id)

        # Check if we can vote for this candidate
        can_vote = self.voted_for in (None, req.candidate_id)

        # Log completeness check
        log_ok = (
            req.last_log_term > self.last_log_term
            or (req.last_log_term == self.last_log_term
                and req.last_log_index >= self.last_log_index)
        )

        grant = can_vote and log_ok
        if grant:
            self.voted_for = req.candidate_id
            self._reset_election_timer()
            self.log.append(f"Voted for {req.candidate_id} in term {req.term}")

        return VoteResponse(self.current_term, grant, self.node_id)

    def handle_vote_response(self, resp: VoteResponse,
                             votes: dict[int, bool]) -> bool:
        """Handle a VoteResponse. Returns True if elected."""
        if resp.term > self.current_term:
            self.current_term = resp.term
            self.role = RaftRole.FOLLOWER
            self.voted_for = None
            return False

        if resp.is_pre_vote:
            # Just collecting pre-votes, don't change state
            return False

        if self.role != RaftRole.CANDIDATE:
            return False

        votes[resp.voter_id] = resp.vote_granted
        granted = sum(1 for v in votes.values() if v)
        majority = self.cluster_size // 2 + 1

        if granted >= majority:
            self.role = RaftRole.LEADER
            self.leader_id = self.node_id
            self.log.append(f"Elected leader for term {self.current_term}")
            return True

        return False


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_election(nodes: list[RaftNode], candidate_id: int) -> bool:
    """Run one election round for the given candidate."""
    candidate = nodes[candidate_id]
    req = candidate._start_election()

    votes: dict[int, bool] = {candidate_id: True}  # Self-vote

    for node in nodes:
        if node.node_id == candidate_id:
            continue
        resp = node.handle_vote_request(req)
        elected = candidate.handle_vote_response(resp, votes)
        if elected:
            # Notify followers
            for n in nodes:
                if n.node_id != candidate_id:
                    n.leader_id = candidate_id
            return True

    return False


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_basic_election() -> None:
    """Basic election with no contention."""
    print("=" * 70)
    print("Raft Election: Basic (No Contention)")
    print("=" * 70)

    random.seed(42)
    nodes = [RaftNode(i, cluster_size=5) for i in range(5)]

    elected = simulate_election(nodes, candidate_id=0)
    print(f"\n  Node 0 starts election: elected={elected}")
    for node in nodes:
        print(f"    Node {node.node_id}: role={node.role.value}, "
              f"term={node.current_term}, voted_for={node.voted_for}")


def demo_split_vote() -> None:
    """Two candidates split the vote."""
    print("\n" + "=" * 70)
    print("Raft Election: Split Vote Scenario")
    print("=" * 70)

    random.seed(42)
    nodes = [RaftNode(i, cluster_size=5) for i in range(5)]

    # Node 0 and Node 1 start elections simultaneously
    # Node 0 gets votes from 2,3; Node 1 gets vote from 4
    # Neither gets majority (need 3)

    nodes[0].current_term += 1
    nodes[0].role = RaftRole.CANDIDATE
    nodes[0].voted_for = 0

    nodes[1].current_term += 1
    nodes[1].role = RaftRole.CANDIDATE
    nodes[1].voted_for = 1

    # Nodes 2,3 vote for Node 0
    nodes[2].voted_for = 0
    nodes[3].voted_for = 0
    # Node 4 votes for Node 1
    nodes[4].voted_for = 1

    votes_0 = sum(1 for n in nodes if n.voted_for == 0)
    votes_1 = sum(1 for n in nodes if n.voted_for == 1)
    majority = 3

    print(f"\n  Node 0: {votes_0} votes (need {majority})")
    print(f"  Node 1: {votes_1} votes (need {majority})")
    print(f"  Result: SPLIT VOTE — no leader elected")
    print(f"\n  Randomized timeout ensures one will retry first next round")

    # Reset and retry
    for node in nodes:
        node.role = RaftRole.FOLLOWER
        node.voted_for = None

    # Node 1 times out first and wins
    elected = simulate_election(nodes, candidate_id=1)
    print(f"\n  Retry: Node 1 elections first: elected={elected}")
    for node in nodes:
        print(f"    Node {node.node_id}: role={node.role.value}, "
              f"voted_for={node.voted_for}")


def demo_pre_vote() -> None:
    """Demonstrate the PreVote extension."""
    print("\n" + "=" * 70)
    print("Raft PreVote: Preventing Disruptive Elections")
    print("=" * 70)

    print("""
  Problem: A partitioned node's term increases with each failed election.
  When it rejoins, its high term forces all nodes to step down.

  PreVote solution: Before incrementing term, ask peers if they would vote.
  Only start real election if pre-vote succeeds.
""")

    random.seed(42)
    nodes = [RaftNode(i, cluster_size=5) for i in range(5)]

    # Establish Node 0 as leader
    simulate_election(nodes, 0)
    print(f"  Initial: Node 0 is leader at term {nodes[0].current_term}")

    # Node 4 is partitioned — simulate many failed elections
    partitioned = nodes[4]
    partitioned.pre_vote_enabled = False  # Without PreVote
    for _ in range(10):
        partitioned.current_term += 1

    print(f"  Node 4 partitioned, term inflated to {partitioned.current_term}")

    # Without PreVote: Node 4 rejoins and disrupts
    print(f"\n  WITHOUT PreVote:")
    saved_term = nodes[0].current_term
    # Node 4 sends message with high term
    for node in nodes[:4]:
        if partitioned.current_term > node.current_term:
            old_role = node.role.value
            node.current_term = partitioned.current_term
            node.role = RaftRole.FOLLOWER
            node.leader_id = None
            print(f"    Node {node.node_id} steps down: "
                  f"{old_role} -> Follower (term {node.current_term})")

    # Reset for PreVote demo
    for i, node in enumerate(nodes):
        node.current_term = 1
        node.role = RaftRole.FOLLOWER
        node.voted_for = None
    simulate_election(nodes, 0)

    partitioned = nodes[4]
    partitioned.pre_vote_enabled = True
    partitioned.current_term = 1  # PreVote doesn't inflate term

    print(f"\n  WITH PreVote:")
    print(f"    Node 4 sends PreVote (would be term {partitioned.current_term + 1})")

    # PreVote: peers check if they would vote (no: they have a leader)
    req = VoteRequest(
        term=partitioned.current_term + 1,
        candidate_id=4,
        last_log_index=0,
        last_log_term=0,
        is_pre_vote=True,
    )
    pre_votes = 0
    for node in nodes[:4]:
        resp = node.handle_vote_request(req)
        pre_votes += 1 if resp.vote_granted else 0
        print(f"    Node {node.node_id}: pre-vote={'YES' if resp.vote_granted else 'NO'} "
              f"(has leader, won't grant)")

    print(f"    Pre-votes: {pre_votes}/5, real election NOT started")
    print(f"    Leader undisturbed!")


def demo_log_completeness() -> None:
    """Show election restriction based on log completeness."""
    print("\n" + "=" * 70)
    print("Raft Election Restriction: Log Completeness")
    print("=" * 70)

    random.seed(42)
    nodes = [RaftNode(i, cluster_size=5) for i in range(5)]

    # Nodes 0-2 have more log entries
    for i in range(3):
        nodes[i].last_log_index = 10
        nodes[i].last_log_term = 3

    # Nodes 3-4 are behind
    nodes[3].last_log_index = 5
    nodes[3].last_log_term = 2
    nodes[4].last_log_index = 5
    nodes[4].last_log_term = 2

    print(f"\n  Log states:")
    for node in nodes:
        print(f"    Node {node.node_id}: last_index={node.last_log_index}, "
              f"last_term={node.last_log_term}")

    # Node 3 (behind) tries election
    print(f"\n  Node 3 (behind) starts election:")
    elected = simulate_election(nodes, candidate_id=3)
    print(f"    Elected: {elected}")
    print(f"    Nodes 0-2 reject: candidate's log is less complete")

    # Reset votes
    for node in nodes:
        node.role = RaftRole.FOLLOWER
        node.voted_for = None
        node.current_term = 0

    # Node 0 (up-to-date) tries election
    print(f"\n  Node 0 (up-to-date) starts election:")
    elected = simulate_election(nodes, candidate_id=0)
    print(f"    Elected: {elected}")
    print(f"    All nodes accept: candidate's log is at least as complete")


if __name__ == "__main__":
    demo_basic_election()
    demo_split_vote()
    demo_pre_vote()
    demo_log_completeness()
    print("\nDone.")
