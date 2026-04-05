"""
Simplified Raft Leader Election Using asyncio

Implements the leader election component of the Raft consensus algorithm.
Each node starts as a Follower, transitions to Candidate on election timeout,
requests votes from peers, and becomes Leader if it wins a majority.

Key concepts:
- Randomized election timeouts to avoid split votes
- Term-based voting: each node votes at most once per term
- Majority quorum: candidate needs > N/2 votes to win
- Split-brain prevention: at most one leader per term
- Step-down on higher term discovery

Reference: Ongaro & Ousterhout, "In Search of an Understandable Consensus
Algorithm" (ATC 2014)

Usage:
    python 03_raft_leader_election.py
"""

import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum


class State(Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"


@dataclass
class RequestVoteArgs:
    """RPC arguments for RequestVote."""
    term: int
    candidate_id: int


@dataclass
class RequestVoteReply:
    """RPC reply for RequestVote."""
    term: int
    vote_granted: bool


@dataclass
class AppendEntriesArgs:
    """Heartbeat (empty AppendEntries) from leader."""
    term: int
    leader_id: int


@dataclass
class AppendEntriesReply:
    """Reply to heartbeat."""
    term: int
    success: bool


class RaftNode:
    """
    A Raft node that participates in leader election.

    This is a simplified simulation: instead of real network RPCs, nodes
    communicate via in-memory async method calls through a cluster object.
    """

    # Timing constants (in seconds) — shortened for simulation
    HEARTBEAT_INTERVAL = 0.15
    ELECTION_TIMEOUT_MIN = 0.3
    ELECTION_TIMEOUT_MAX = 0.6

    def __init__(self, node_id: int, cluster: "RaftCluster"):
        self.node_id = node_id
        self.cluster = cluster

        # Persistent state
        self.current_term = 0
        self.voted_for: int | None = None

        # Volatile state
        self.state = State.FOLLOWER
        self.leader_id: int | None = None
        self.votes_received = 0

        # Control
        self._election_timer: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._running = True

        # Event log for display
        self.log_entries: list[str] = []

    def _log(self, msg: str) -> None:
        """Record and print a log message."""
        entry = f"[Node {self.node_id} | Term {self.current_term} | {self.state.value}] {msg}"
        self.log_entries.append(entry)
        print(entry)

    def _random_timeout(self) -> float:
        """Generate a randomized election timeout."""
        return random.uniform(self.ELECTION_TIMEOUT_MIN, self.ELECTION_TIMEOUT_MAX)

    async def start(self) -> None:
        """Start the node as a follower with an election timer."""
        self._log("Starting as Follower")
        self._reset_election_timer()

    def stop(self) -> None:
        """Gracefully stop the node."""
        self._running = False
        if self._election_timer and not self._election_timer.done():
            self._election_timer.cancel()
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

    def _reset_election_timer(self) -> None:
        """Reset (restart) the election timeout timer."""
        if self._election_timer and not self._election_timer.done():
            self._election_timer.cancel()
        self._election_timer = asyncio.create_task(self._election_timeout())

    async def _election_timeout(self) -> None:
        """Wait for a random timeout, then start an election."""
        timeout = self._random_timeout()
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return

        if not self._running:
            return

        # Timeout expired without hearing from a leader -> start election
        if self.state != State.LEADER:
            await self._start_election()

    async def _start_election(self) -> None:
        """Transition to Candidate and request votes from all peers."""
        self.current_term += 1
        self.state = State.CANDIDATE
        self.voted_for = self.node_id
        self.votes_received = 1  # Vote for self
        self.leader_id = None

        self._log(f"Election timeout! Starting election for term {self.current_term}")

        # Send RequestVote RPCs to all peers concurrently
        peers = self.cluster.get_peers(self.node_id)
        tasks = []
        for peer_id in peers:
            args = RequestVoteArgs(term=self.current_term, candidate_id=self.node_id)
            tasks.append(asyncio.create_task(
                self._send_request_vote(peer_id, args)
            ))

        # Wait for all vote responses (with a timeout)
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                   timeout=self._random_timeout())
        except asyncio.TimeoutError:
            pass

        # Check if we won (we might have stepped down during the process)
        if (self.state == State.CANDIDATE and
                self.votes_received > len(self.cluster.nodes) // 2):
            await self._become_leader()
        elif self.state == State.CANDIDATE:
            # Election failed (split vote or lost). Restart as follower.
            self._log("Election failed (not enough votes). Reverting to Follower.")
            self.state = State.FOLLOWER
            self.voted_for = None
            self._reset_election_timer()

    async def _send_request_vote(self, peer_id: int,
                                 args: RequestVoteArgs) -> None:
        """Send a RequestVote RPC to a peer and handle the reply."""
        peer = self.cluster.get_node(peer_id)
        if peer is None:
            return

        reply = await peer.handle_request_vote(args)

        # If we're no longer a candidate, ignore the reply
        if self.state != State.CANDIDATE:
            return

        # If reply has a higher term, step down
        if reply.term > self.current_term:
            self._step_down(reply.term)
            return

        if reply.vote_granted:
            self.votes_received += 1
            self._log(f"Received vote from Node {peer_id} "
                      f"({self.votes_received}/{len(self.cluster.nodes)} total)")

    async def handle_request_vote(self, args: RequestVoteArgs) -> RequestVoteReply:
        """Handle an incoming RequestVote RPC."""
        # If the candidate's term is higher, update our term
        if args.term > self.current_term:
            self._step_down(args.term)

        # Grant vote if: same term AND we haven't voted (or voted for this candidate)
        grant = False
        if (args.term == self.current_term and
                (self.voted_for is None or self.voted_for == args.candidate_id)):
            grant = True
            self.voted_for = args.candidate_id
            self._log(f"Granted vote to Node {args.candidate_id} for term {args.term}")
            # Reset election timer since we heard from a valid candidate
            self._reset_election_timer()

        return RequestVoteReply(term=self.current_term, vote_granted=grant)

    async def _become_leader(self) -> None:
        """Transition to Leader state and start sending heartbeats."""
        self.state = State.LEADER
        self.leader_id = self.node_id
        self._log(f"*** Won election! Became LEADER for term {self.current_term} ***")

        # Cancel election timer
        if self._election_timer and not self._election_timer.done():
            self._election_timer.cancel()

        # Start sending periodic heartbeats
        self._heartbeat_task = asyncio.create_task(self._send_heartbeats())

    async def _send_heartbeats(self) -> None:
        """Periodically send empty AppendEntries (heartbeat) to all peers."""
        while self._running and self.state == State.LEADER:
            peers = self.cluster.get_peers(self.node_id)
            for peer_id in peers:
                peer = self.cluster.get_node(peer_id)
                if peer is None:
                    continue
                args = AppendEntriesArgs(term=self.current_term,
                                         leader_id=self.node_id)
                reply = await peer.handle_append_entries(args)
                if reply.term > self.current_term:
                    self._step_down(reply.term)
                    return

            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                return

    async def handle_append_entries(self,
                                    args: AppendEntriesArgs) -> AppendEntriesReply:
        """Handle an incoming AppendEntries (heartbeat) RPC."""
        if args.term < self.current_term:
            return AppendEntriesReply(term=self.current_term, success=False)

        # Valid heartbeat from current or newer leader
        if args.term >= self.current_term:
            if args.term > self.current_term or self.state == State.CANDIDATE:
                self._step_down(args.term)
            self.leader_id = args.leader_id
            self._reset_election_timer()

        return AppendEntriesReply(term=self.current_term, success=True)

    def _step_down(self, new_term: int) -> None:
        """Step down to Follower on discovering a higher term."""
        old_term = self.current_term
        self.current_term = new_term
        self.state = State.FOLLOWER
        self.voted_for = None
        self.leader_id = None

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        if new_term > old_term:
            self._log(f"Discovered higher term {new_term}. Stepping down to Follower.")

        self._reset_election_timer()


class RaftCluster:
    """Manages a cluster of Raft nodes for simulation."""

    def __init__(self, size: int):
        self.nodes: dict[int, RaftNode] = {}
        for i in range(size):
            self.nodes[i] = RaftNode(i, self)

    def get_node(self, node_id: int) -> RaftNode | None:
        return self.nodes.get(node_id)

    def get_peers(self, node_id: int) -> list[int]:
        return [nid for nid in self.nodes if nid != node_id]

    async def run_simulation(self, duration: float) -> None:
        """Run the cluster for a given duration, then stop."""
        print("=" * 70)
        print(f"Raft Leader Election Simulation — {len(self.nodes)}-node cluster")
        print(f"Running for {duration}s with randomized election timeouts")
        print("=" * 70 + "\n")

        # Start all nodes
        for node in self.nodes.values():
            await node.start()

        # Let the simulation run
        await asyncio.sleep(duration)

        # Stop all nodes
        for node in self.nodes.values():
            node.stop()

        # Brief delay for cleanup
        await asyncio.sleep(0.1)

    def print_summary(self) -> None:
        """Print the final state of all nodes."""
        print("\n" + "=" * 70)
        print("Final Cluster State")
        print("=" * 70)

        leaders = []
        for nid, node in sorted(self.nodes.items()):
            leader_str = (f"Leader={node.leader_id}"
                          if node.leader_id is not None else "Leader=unknown")
            print(f"  Node {nid}: state={node.state.value:>10}, "
                  f"term={node.current_term}, voted_for={node.voted_for}, "
                  f"{leader_str}")
            if node.state == State.LEADER:
                leaders.append(nid)

        print(f"\nLeaders in final state: {leaders if leaders else '(none yet)'}")

        # Verify safety: at most one leader per term
        term_leaders: dict[int, list[int]] = {}
        for nid, node in self.nodes.items():
            if node.state == State.LEADER:
                term_leaders.setdefault(node.current_term, []).append(nid)

        print("\nSafety check (at most one leader per term):")
        if not term_leaders:
            print("  No leaders elected yet.")
        for term, leaders_in_term in sorted(term_leaders.items()):
            status = "SAFE" if len(leaders_in_term) == 1 else "VIOLATION!"
            print(f"  Term {term}: leaders={leaders_in_term} — {status}")


async def main() -> None:
    """Run the Raft leader election simulation."""
    random.seed(42)  # For reproducibility

    # Scenario 1: Normal election
    cluster = RaftCluster(size=5)
    await cluster.run_simulation(duration=2.0)
    cluster.print_summary()

    print("\n" + "#" * 70)
    print("# Scenario 2: Observe multiple terms (leader crashes)")
    print("#" * 70 + "\n")

    # Scenario 2: Kill the leader partway through to force re-election
    random.seed(123)
    cluster2 = RaftCluster(size=5)

    print("=" * 70)
    print("Phase 1: Initial election (2s)")
    print("=" * 70 + "\n")

    for node in cluster2.nodes.values():
        await node.start()

    await asyncio.sleep(2.0)

    # Find and kill the leader
    leader_id = None
    for nid, node in cluster2.nodes.items():
        if node.state == State.LEADER:
            leader_id = nid
            break

    if leader_id is not None:
        print(f"\n{'=' * 70}")
        print(f"Phase 2: Killing leader Node {leader_id} — forcing re-election")
        print("=" * 70 + "\n")
        cluster2.nodes[leader_id].stop()
        del cluster2.nodes[leader_id]

        await asyncio.sleep(2.0)
    else:
        print("\nNo leader found to kill. Continuing...\n")
        await asyncio.sleep(1.0)

    for node in cluster2.nodes.values():
        node.stop()

    await asyncio.sleep(0.1)
    cluster2.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
    print("\nDone.")
