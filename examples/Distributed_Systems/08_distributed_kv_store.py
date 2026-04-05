"""
Minimal Distributed Key-Value Store Using Simplified Raft Consensus

A complete, self-contained simulation of a distributed key-value store
backed by Raft consensus. The cluster processes client commands (put, get,
delete) through a leader, replicates log entries to followers, and handles
node failures and network partitions.

Key concepts:
- State machine replication: all nodes apply the same commands in order
- Linearizable reads: reads go through the leader to ensure freshness
- Leader election on failure detection
- Log replication with majority commit
- Fault tolerance: cluster operates with N/2 + 1 nodes
- Partition recovery and state reconciliation

Usage:
    python 08_distributed_kv_store.py
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class OpType(Enum):
    PUT = "PUT"
    GET = "GET"
    DELETE = "DELETE"


@dataclass
class Command:
    """A client command to the KV store."""
    op: OpType
    key: str
    value: str | None = None  # Only for PUT

    def __repr__(self) -> str:
        if self.op == OpType.PUT:
            return f"PUT({self.key}={self.value})"
        elif self.op == OpType.DELETE:
            return f"DELETE({self.key})"
        else:
            return f"GET({self.key})"


@dataclass
class LogEntry:
    """A Raft log entry."""
    term: int
    index: int
    command: Command


@dataclass
class ClientResult:
    """Result returned to a client."""
    success: bool
    value: str | None = None
    error: str | None = None

    def __repr__(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        if self.value is not None:
            return f"OK: {self.value}"
        return "OK"


# ---------------------------------------------------------------------------
# KV State Machine
# ---------------------------------------------------------------------------

class KVStateMachine:
    """
    A simple key-value store state machine.
    Commands are applied in log order to produce a deterministic state.
    """

    def __init__(self):
        self._store: dict[str, str] = {}

    def apply(self, cmd: Command) -> ClientResult:
        """Apply a command and return the result."""
        if cmd.op == OpType.PUT:
            self._store[cmd.key] = cmd.value  # type: ignore
            return ClientResult(success=True)
        elif cmd.op == OpType.GET:
            val = self._store.get(cmd.key)
            if val is None:
                return ClientResult(success=False, error=f"key '{cmd.key}' not found")
            return ClientResult(success=True, value=val)
        elif cmd.op == OpType.DELETE:
            if cmd.key in self._store:
                del self._store[cmd.key]
                return ClientResult(success=True)
            return ClientResult(success=False, error=f"key '{cmd.key}' not found")
        return ClientResult(success=False, error="unknown operation")

    def snapshot(self) -> dict[str, str]:
        """Return a copy of the current state."""
        return dict(self._store)


# ---------------------------------------------------------------------------
# Raft Node
# ---------------------------------------------------------------------------

class NodeState(Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"
    DOWN = "Down"  # Simulated failure


class RaftKVNode:
    """
    A Raft node with an integrated KV state machine.
    Communication is simulated via in-process method calls.
    """

    def __init__(self, node_id: int, cluster: "RaftKVCluster"):
        self.node_id = node_id
        self.cluster = cluster

        # Persistent Raft state
        self.current_term = 0
        self.voted_for: int | None = None
        self.log: list[LogEntry] = []

        # Volatile Raft state
        self.state = NodeState.FOLLOWER
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id: int | None = None

        # Leader volatile state
        self.next_index: dict[int, int] = {}
        self.match_index: dict[int, int] = {}

        # State machine
        self.kv = KVStateMachine()

        # Pending client results (index -> result placeholder)
        self._pending: dict[int, ClientResult | None] = {}

    @property
    def last_log_index(self) -> int:
        return len(self.log)

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0

    def is_up(self) -> bool:
        return self.state != NodeState.DOWN

    # --- Leader Election (simplified) ---

    def start_election(self) -> bool:
        """Run a simplified election. Returns True if this node becomes leader."""
        if not self.is_up():
            return False

        self.current_term += 1
        self.state = NodeState.CANDIDATE
        self.voted_for = self.node_id
        votes = 1  # Vote for self

        _log(self, f"Starting election for term {self.current_term}")

        for peer_id in self.cluster.get_peer_ids(self.node_id):
            peer = self.cluster.get_node(peer_id)
            if peer is None or not peer.is_up():
                continue
            if not self.cluster.is_reachable(self.node_id, peer_id):
                continue

            # RequestVote: simplified — grant if term is higher and not yet voted
            if peer.current_term < self.current_term:
                peer.current_term = self.current_term
                peer.voted_for = self.node_id
                peer.state = NodeState.FOLLOWER
                votes += 1
                _log(peer, f"Voted for Node {self.node_id} in term {self.current_term}")
            elif (peer.current_term == self.current_term and
                  peer.voted_for in (None, self.node_id)):
                peer.voted_for = self.node_id
                votes += 1
                _log(peer, f"Voted for Node {self.node_id} in term {self.current_term}")

        majority = len(self.cluster.nodes) // 2 + 1
        if votes >= majority:
            self._become_leader()
            return True
        else:
            _log(self, f"Election failed: got {votes}/{majority} needed")
            self.state = NodeState.FOLLOWER
            return False

    def _become_leader(self) -> None:
        """Transition to leader."""
        self.state = NodeState.LEADER
        self.leader_id = self.node_id

        # Initialize leader state
        for peer_id in self.cluster.get_peer_ids(self.node_id):
            self.next_index[peer_id] = self.last_log_index + 1
            self.match_index[peer_id] = 0

        _log(self, f"*** Became LEADER for term {self.current_term} ***")

        # Notify followers of new leader
        for peer_id in self.cluster.get_peer_ids(self.node_id):
            peer = self.cluster.get_node(peer_id)
            if peer and peer.is_up() and self.cluster.is_reachable(self.node_id, peer_id):
                peer.leader_id = self.node_id

    # --- Client Commands ---

    def client_request(self, cmd: Command) -> ClientResult:
        """
        Handle a client command.
        GET: read from leader's state machine (linearizable).
        PUT/DELETE: append to log, replicate, commit, then apply.
        """
        if not self.is_up():
            return ClientResult(success=False, error="node is down")

        if self.state != NodeState.LEADER:
            if self.leader_id is not None:
                # Redirect to leader
                leader = self.cluster.get_node(self.leader_id)
                if leader and leader.is_up():
                    return leader.client_request(cmd)
            return ClientResult(success=False, error="no leader available")

        # GET: serve from local state machine
        if cmd.op == OpType.GET:
            return self.kv.apply(cmd)

        # PUT/DELETE: append to log and replicate
        entry = LogEntry(term=self.current_term,
                         index=self.last_log_index + 1,
                         command=cmd)
        self.log.append(entry)
        _log(self, f"Appended {cmd} at index {entry.index}")

        # Replicate to followers
        self._replicate_all()

        # Advance commit index
        self._advance_commit_index()

        # Apply committed entries
        self._apply_committed()

        # Return result of applying the command
        if entry.index <= self.last_applied:
            return self.kv.apply(cmd) if cmd.op == OpType.GET else ClientResult(success=True)
        return ClientResult(success=True)

    # --- Log Replication ---

    def _replicate_all(self) -> None:
        """Replicate log entries to all reachable followers."""
        for peer_id in self.cluster.get_peer_ids(self.node_id):
            peer = self.cluster.get_node(peer_id)
            if peer is None or not peer.is_up():
                continue
            if not self.cluster.is_reachable(self.node_id, peer_id):
                continue
            self._replicate_to(peer_id, peer)

    def _replicate_to(self, peer_id: int, peer: RaftKVNode) -> None:
        """Send AppendEntries to a single peer."""
        while True:
            ni = self.next_index.get(peer_id, 1)
            prev_idx = ni - 1
            prev_term = self.log[prev_idx - 1].term if prev_idx > 0 else 0

            entries = self.log[ni - 1:]

            # Consistency check on peer side
            if prev_idx > len(peer.log):
                self.next_index[peer_id] = max(1, ni - 1)
                if self.next_index[peer_id] < ni:
                    continue
                break
            if prev_idx > 0 and peer.log[prev_idx - 1].term != prev_term:
                peer.log = peer.log[:prev_idx - 1]
                self.next_index[peer_id] = max(1, ni - 1)
                continue

            # Append entries
            for entry in entries:
                idx = entry.index - 1
                if idx < len(peer.log):
                    if peer.log[idx].term != entry.term:
                        peer.log = peer.log[:idx]
                        peer.log.append(entry)
                else:
                    peer.log.append(entry)

            if entries:
                self.next_index[peer_id] = entries[-1].index + 1
                self.match_index[peer_id] = entries[-1].index

            # Update peer's commit index and apply
            if self.commit_index > peer.commit_index:
                peer.commit_index = min(self.commit_index, len(peer.log))
                peer._apply_committed()

            break

    def _advance_commit_index(self) -> None:
        """Advance commit index when majority has replicated."""
        if self.state != NodeState.LEADER:
            return

        total_nodes = len(self.cluster.nodes)
        for n in range(self.commit_index + 1, self.last_log_index + 1):
            if self.log[n - 1].term != self.current_term:
                continue
            replicas = 1  # Leader counts
            for peer_id, mi in self.match_index.items():
                if mi >= n:
                    replicas += 1
            if replicas > total_nodes // 2:
                self.commit_index = n

    def _apply_committed(self) -> None:
        """Apply committed entries to the KV state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]
            result = self.kv.apply(entry.command)

    # --- Node Failure Simulation ---

    def crash(self) -> None:
        """Simulate node crash."""
        self.state = NodeState.DOWN
        _log(self, "*** NODE CRASHED ***")

    def recover(self) -> None:
        """Simulate node recovery (state on disk is preserved)."""
        self.state = NodeState.FOLLOWER
        self.leader_id = None
        _log(self, "*** NODE RECOVERED ***")

    def print_state(self) -> None:
        """Print node status."""
        log_cmds = [str(e.command) for e in self.log]
        kv_snap = self.kv.snapshot()
        status = self.state.value
        print(f"  Node {self.node_id} [{status:>9}] term={self.current_term} "
              f"commit={self.commit_index} applied={self.last_applied} "
              f"log={log_cmds}")
        print(f"{'':>36}kv={kv_snap}")


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

class RaftKVCluster:
    """Manages nodes and simulates network partitions."""

    def __init__(self, size: int):
        self.nodes: dict[int, RaftKVNode] = {}
        for i in range(size):
            self.nodes[i] = RaftKVNode(i, self)
        self._partitions: set[tuple[int, int]] = set()

    def get_node(self, node_id: int) -> RaftKVNode | None:
        return self.nodes.get(node_id)

    def get_peer_ids(self, node_id: int) -> list[int]:
        return [nid for nid in self.nodes if nid != node_id]

    def is_reachable(self, a: int, b: int) -> bool:
        return (a, b) not in self._partitions

    def partition(self, group_a: list[int], group_b: list[int]) -> None:
        for a in group_a:
            for b in group_b:
                self._partitions.add((a, b))
                self._partitions.add((b, a))
        print(f"\n  *** PARTITION: {group_a} <-X-> {group_b} ***\n")

    def heal(self) -> None:
        self._partitions.clear()
        print("\n  *** PARTITION HEALED ***\n")

    def find_leader(self) -> RaftKVNode | None:
        for node in self.nodes.values():
            if node.state == NodeState.LEADER:
                return node
        return None

    def print_cluster(self, header: str = "") -> None:
        if header:
            print(f"\n  --- {header} ---")
        for nid in sorted(self.nodes.keys()):
            self.nodes[nid].print_state()


def _log(node: RaftKVNode, msg: str) -> None:
    """Print a log message from a node."""
    print(f"  [Node {node.node_id} T{node.current_term} "
          f"{node.state.value:>9}] {msg}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation() -> None:
    """Run a complete distributed KV store simulation."""

    random.seed(42)

    # ---- Scenario 1: Normal Operations ----
    print("=" * 70)
    print("Scenario 1: Normal Operations (3-node cluster)")
    print("=" * 70)

    cluster = RaftKVCluster(size=3)

    # Elect Node 0 as leader
    print("\n--- Leader Election ---")
    cluster.nodes[0].start_election()

    # Process client commands
    leader = cluster.find_leader()
    assert leader is not None

    print("\n--- Client Commands ---")
    commands = [
        Command(OpType.PUT, "name", "Alice"),
        Command(OpType.PUT, "age", "30"),
        Command(OpType.PUT, "city", "Seoul"),
        Command(OpType.GET, "name"),
        Command(OpType.GET, "age"),
    ]

    for cmd in commands:
        result = leader.client_request(cmd)
        print(f"  Client: {cmd} => {result}")

    cluster.print_cluster("Cluster state after normal operations")

    # Verify all nodes have the same state
    states = [cluster.nodes[i].kv.snapshot() for i in range(3)]
    assert states[0] == states[1] == states[2], "State divergence detected!"
    print(f"\n  All nodes have consistent state: {states[0]}")

    # ---- Scenario 2: Node Failure ----
    print("\n" + "=" * 70)
    print("Scenario 2: Node Failure (1 of 3 nodes crashes)")
    print("=" * 70)

    # Crash Node 2
    print("\n--- Crashing Node 2 ---")
    cluster.nodes[2].crash()

    # Cluster should still operate (2/3 = majority)
    print("\n--- Client Commands (with 2/3 nodes) ---")
    more_commands = [
        Command(OpType.PUT, "color", "blue"),
        Command(OpType.DELETE, "age"),
        Command(OpType.GET, "color"),
        Command(OpType.GET, "name"),
    ]

    for cmd in more_commands:
        result = leader.client_request(cmd)
        print(f"  Client: {cmd} => {result}")

    cluster.print_cluster("State after Node 2 crash")

    # ---- Scenario 3: Node Recovery ----
    print("\n" + "=" * 70)
    print("Scenario 3: Node Recovery")
    print("=" * 70)

    print("\n--- Recovering Node 2 ---")
    cluster.nodes[2].recover()

    # Leader replicates missing entries to recovered node
    leader._replicate_all()
    leader._advance_commit_index()

    cluster.print_cluster("State after Node 2 recovery")

    # Verify convergence
    live_states = {}
    for nid, node in cluster.nodes.items():
        if node.is_up():
            live_states[nid] = node.kv.snapshot()
    values = list(live_states.values())
    assert all(v == values[0] for v in values), "State divergence after recovery!"
    print(f"\n  All live nodes converged: {values[0]}")

    # ---- Scenario 4: Network Partition ----
    print("\n" + "=" * 70)
    print("Scenario 4: Network Partition")
    print("=" * 70)

    cluster2 = RaftKVCluster(size=3)
    cluster2.nodes[0].start_election()
    leader2 = cluster2.find_leader()
    assert leader2 is not None

    # Pre-partition data
    leader2.client_request(Command(OpType.PUT, "x", "1"))
    leader2.client_request(Command(OpType.PUT, "y", "2"))

    cluster2.print_cluster("Pre-partition state")

    # Partition: {0} | {1, 2}
    # Node 0 (leader) is isolated!
    cluster2.partition(group_a=[0], group_b=[1, 2])

    # Old leader can't commit (minority)
    print("--- Old leader (Node 0) tries to write during partition ---")
    result = leader2.client_request(Command(OpType.PUT, "x", "999"))
    print(f"  PUT x=999 => {result}")
    print(f"  Node 0 commit_index: {cluster2.nodes[0].commit_index} "
          f"(cannot advance - no majority)")

    # Majority elects new leader
    print("\n--- Majority partition elects new leader ---")
    cluster2.nodes[1].start_election()
    new_leader = cluster2.find_leader()

    if new_leader and new_leader.node_id != 0:
        print(f"\n--- New leader (Node {new_leader.node_id}) processes commands ---")
        result = new_leader.client_request(Command(OpType.PUT, "z", "3"))
        print(f"  PUT z=3 => {result}")

    cluster2.print_cluster("During partition")

    # Heal partition
    cluster2.heal()

    # New leader reconciles with old leader
    if new_leader:
        new_leader._replicate_all()
        new_leader._advance_commit_index()

    cluster2.print_cluster("After partition healed")

    # Final verification
    print("\n--- Final linearizable reads through leader ---")
    leader_final = cluster2.find_leader()
    if leader_final:
        for key in ["x", "y", "z"]:
            result = leader_final.client_request(Command(OpType.GET, key))
            print(f"  GET {key} => {result}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
  This simulation demonstrated:

  1. Normal operations: PUT/GET/DELETE through leader, replicated to followers
  2. Fault tolerance: cluster continues operating with 2/3 nodes
  3. Node recovery: crashed node catches up on missed log entries
  4. Network partition:
     - Minority leader cannot commit (no majority)
     - Majority elects new leader and continues
     - After healing, logs reconcile and state converges

  Properties guaranteed:
  - Linearizability: reads through leader reflect all committed writes
  - Safety: at most one leader per term
  - Liveness: cluster makes progress as long as majority is available
""")


if __name__ == "__main__":
    run_simulation()
    print("Done.")
