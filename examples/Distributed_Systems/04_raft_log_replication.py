"""
Raft Log Replication with Simulated Network Partitions

Implements the log replication component of Raft. The leader accepts client
commands, appends them to its log, and replicates them to followers via
AppendEntries RPCs. A command is committed once a majority of nodes have
replicated it.

Key concepts:
- Log entries: (term, index, command)
- AppendEntries consistency check (prev_log_index, prev_log_term)
- Commit index advancement via majority replication
- Network partitions: minority partition stops making progress
- Partition healing: log reconciliation, followers catch up

Reference: Ongaro & Ousterhout, "In Search of an Understandable Consensus
Algorithm" (ATC 2014), Sections 5.3-5.4

Usage:
    python 04_raft_log_replication.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogEntry:
    """A single entry in the Raft log."""
    term: int
    index: int
    command: str

    def __repr__(self) -> str:
        return f"({self.index}:T{self.term} '{self.command}')"


@dataclass
class AppendEntriesArgs:
    """Arguments for the AppendEntries RPC."""
    term: int
    leader_id: int
    prev_log_index: int
    prev_log_term: int
    entries: list[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesReply:
    """Reply from AppendEntries RPC."""
    term: int
    success: bool
    # Optimization: if rejected, hint the last matching index
    match_index: int = 0


class RaftNode:
    """A simplified Raft node focused on log replication."""

    def __init__(self, node_id: int, cluster: "RaftCluster"):
        self.node_id = node_id
        self.cluster = cluster

        # Persistent state
        self.current_term = 1
        self.log: list[LogEntry] = []  # 1-indexed conceptually; stored 0-indexed

        # Volatile state
        self.commit_index = 0
        self.last_applied = 0
        self.is_leader = False

        # Leader-only volatile state (initialized on election)
        self.next_index: dict[int, int] = {}   # peer_id -> next log index to send
        self.match_index: dict[int, int] = {}   # peer_id -> highest replicated index

        # Applied commands (state machine)
        self.applied_commands: list[str] = []

    @property
    def last_log_index(self) -> int:
        return len(self.log)

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0

    def become_leader(self) -> None:
        """Initialize leader state."""
        self.is_leader = True
        peers = self.cluster.get_peers(self.node_id)
        for peer_id in peers:
            self.next_index[peer_id] = self.last_log_index + 1
            self.match_index[peer_id] = 0
        print(f"  [Node {self.node_id}] Became LEADER for term {self.current_term}")

    def client_request(self, command: str) -> bool:
        """
        Handle a client request (leader only).
        Append to local log, then replicate to followers.
        """
        if not self.is_leader:
            print(f"  [Node {self.node_id}] Not leader, rejecting command: {command}")
            return False

        # Append to local log
        entry = LogEntry(term=self.current_term,
                         index=self.last_log_index + 1,
                         command=command)
        self.log.append(entry)
        print(f"  [Node {self.node_id}] Leader appended: {entry}")

        # Replicate to all followers
        self._replicate_to_all()

        # Try to advance commit index
        self._advance_commit_index()

        return True

    def _replicate_to_all(self) -> None:
        """Send AppendEntries to all reachable peers."""
        peers = self.cluster.get_peers(self.node_id)
        for peer_id in peers:
            if not self.cluster.is_reachable(self.node_id, peer_id):
                continue
            self._replicate_to(peer_id)

    def _replicate_to(self, peer_id: int) -> None:
        """Send AppendEntries RPC to a single peer, retrying on mismatch."""
        peer = self.cluster.get_node(peer_id)
        if peer is None:
            return

        while True:
            next_idx = self.next_index.get(peer_id, 1)
            prev_log_index = next_idx - 1
            prev_log_term = self.log[prev_log_index - 1].term if prev_log_index > 0 else 0

            # Entries to send
            entries_to_send = self.log[next_idx - 1:]

            args = AppendEntriesArgs(
                term=self.current_term,
                leader_id=self.node_id,
                prev_log_index=prev_log_index,
                prev_log_term=prev_log_term,
                entries=entries_to_send,
                leader_commit=self.commit_index,
            )

            reply = peer.handle_append_entries(args)

            if reply.success:
                # Update next_index and match_index
                if entries_to_send:
                    self.next_index[peer_id] = entries_to_send[-1].index + 1
                    self.match_index[peer_id] = entries_to_send[-1].index
                break
            else:
                # Decrement next_index and retry (log inconsistency)
                if reply.term > self.current_term:
                    print(f"  [Node {self.node_id}] Discovered higher term from "
                          f"Node {peer_id}. Stepping down.")
                    self.is_leader = False
                    self.current_term = reply.term
                    return
                self.next_index[peer_id] = max(1, self.next_index[peer_id] - 1)
                print(f"  [Node {self.node_id}] AppendEntries to Node {peer_id} "
                      f"failed. Decrementing next_index to {self.next_index[peer_id]}")

    def handle_append_entries(self, args: AppendEntriesArgs) -> AppendEntriesReply:
        """Handle incoming AppendEntries RPC (follower side)."""
        # Reject if leader's term is stale
        if args.term < self.current_term:
            return AppendEntriesReply(term=self.current_term, success=False)

        # Update term if needed
        if args.term > self.current_term:
            self.current_term = args.term
            self.is_leader = False

        # Consistency check: verify prev_log matches
        if args.prev_log_index > 0:
            if args.prev_log_index > len(self.log):
                # We don't have the entry at prev_log_index
                return AppendEntriesReply(term=self.current_term, success=False,
                                          match_index=len(self.log))
            if self.log[args.prev_log_index - 1].term != args.prev_log_term:
                # Term mismatch at prev_log_index — delete conflicting entries
                self.log = self.log[:args.prev_log_index - 1]
                return AppendEntriesReply(term=self.current_term, success=False,
                                          match_index=len(self.log))

        # Append new entries (overwrite any conflicting entries)
        for entry in args.entries:
            idx = entry.index - 1  # 0-based
            if idx < len(self.log):
                if self.log[idx].term != entry.term:
                    # Conflict: delete this and all following entries
                    self.log = self.log[:idx]
                    self.log.append(entry)
                # else: already have this entry, skip
            else:
                self.log.append(entry)

        # Update commit index
        if args.leader_commit > self.commit_index:
            self.commit_index = min(args.leader_commit, len(self.log))
            self._apply_committed()

        if args.entries:
            print(f"  [Node {self.node_id}] Replicated {len(args.entries)} entries "
                  f"from Leader {args.leader_id}. Log length={len(self.log)}, "
                  f"commit={self.commit_index}")

        return AppendEntriesReply(term=self.current_term, success=True,
                                  match_index=len(self.log))

    def _advance_commit_index(self) -> None:
        """
        Leader advances commit_index when a majority has replicated an entry.
        Only entries from the current term are committed by counting replicas.
        """
        if not self.is_leader:
            return

        cluster_size = len(self.cluster.nodes)
        for n in range(self.commit_index + 1, self.last_log_index + 1):
            if self.log[n - 1].term != self.current_term:
                continue
            # Count replicas (leader itself counts as 1)
            replicas = 1
            for peer_id, match_idx in self.match_index.items():
                if match_idx >= n:
                    replicas += 1
            if replicas > cluster_size // 2:
                self.commit_index = n

        self._apply_committed()

    def _apply_committed(self) -> None:
        """Apply committed but not yet applied entries to the state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            cmd = self.log[self.last_applied - 1].command
            self.applied_commands.append(cmd)

    def print_log(self) -> None:
        """Display the node's log."""
        log_str = ", ".join(str(e) for e in self.log)
        role = "LEADER" if self.is_leader else "follower"
        print(f"  Node {self.node_id} ({role}): "
              f"commit={self.commit_index} applied={self.last_applied} "
              f"log=[{log_str}]")


class RaftCluster:
    """Manages nodes and simulates network connectivity."""

    def __init__(self, size: int):
        self.nodes: dict[int, RaftNode] = {}
        for i in range(size):
            self.nodes[i] = RaftNode(i, self)
        # Connectivity matrix: (a, b) -> reachable?
        self._partitions: set[tuple[int, int]] = set()

    def get_node(self, node_id: int) -> RaftNode | None:
        return self.nodes.get(node_id)

    def get_peers(self, node_id: int) -> list[int]:
        return [nid for nid in self.nodes if nid != node_id]

    def is_reachable(self, from_id: int, to_id: int) -> bool:
        """Check if two nodes can communicate (not partitioned)."""
        return (from_id, to_id) not in self._partitions

    def partition(self, group_a: list[int], group_b: list[int]) -> None:
        """Create a network partition between two groups."""
        for a in group_a:
            for b in group_b:
                self._partitions.add((a, b))
                self._partitions.add((b, a))
        print(f"\n  *** NETWORK PARTITION: {group_a} <-X-> {group_b} ***\n")

    def heal_partition(self) -> None:
        """Remove all network partitions."""
        self._partitions.clear()
        print("\n  *** PARTITION HEALED: all nodes can communicate ***\n")

    def print_all_logs(self) -> None:
        """Print logs for all nodes."""
        for nid in sorted(self.nodes.keys()):
            self.nodes[nid].print_log()


def run_simulation() -> None:
    """Run a complete log replication simulation with partition scenarios."""

    # --- Scenario 1: Normal replication ---
    print("=" * 70)
    print("Scenario 1: Normal Log Replication (5-node cluster)")
    print("=" * 70)

    cluster = RaftCluster(size=5)
    leader = cluster.nodes[0]
    leader.become_leader()

    # Client sends commands
    commands = ["SET x=1", "SET y=2", "SET z=3"]
    for cmd in commands:
        print(f"\n--- Client request: {cmd} ---")
        leader.client_request(cmd)

    print("\nCluster state after normal replication:")
    cluster.print_all_logs()

    # --- Scenario 2: Network partition ---
    print("\n" + "=" * 70)
    print("Scenario 2: Network Partition")
    print("=" * 70)

    cluster2 = RaftCluster(size=5)
    leader2 = cluster2.nodes[0]
    leader2.become_leader()

    # Replicate some entries before partition
    print("\n--- Pre-partition: replicate 2 entries ---")
    leader2.client_request("SET a=10")
    leader2.client_request("SET b=20")

    print("\nState before partition:")
    cluster2.print_all_logs()

    # Create partition: {0, 1} vs {2, 3, 4}
    # Node 0 (leader) is in the minority partition!
    cluster2.partition(group_a=[0, 1], group_b=[2, 3, 4])

    # Leader in minority tries to replicate — can't get majority
    print("--- Leader (minority) tries to replicate during partition ---")
    leader2.client_request("SET c=30")  # This will NOT be committed

    print("\nState during partition (leader in minority):")
    cluster2.print_all_logs()
    print(f"\n  Leader commit_index = {leader2.commit_index} "
          f"(cannot advance — no majority!)")
    print(f"  Entry 'SET c=30' is in leader's log but NOT committed.")

    # Meanwhile, majority elects a new leader (Node 2)
    print("\n--- Majority partition elects new leader (Node 2) ---")
    new_leader = cluster2.nodes[2]
    new_leader.current_term = 2
    new_leader.become_leader()

    # New leader replicates within majority
    new_leader.client_request("SET d=40")

    print("\nState after majority leader replicates:")
    cluster2.print_all_logs()

    # Heal partition
    cluster2.heal_partition()

    # Old leader discovers higher term and steps down
    print("--- Partition healed: old leader (Node 0) contacts new leader ---")
    # Simulate: new leader sends AppendEntries to old leader's partition
    new_leader._replicate_to_all()
    new_leader._advance_commit_index()

    print("\nFinal state after partition healed:")
    cluster2.print_all_logs()

    # --- Scenario 3: Log reconciliation ---
    print("\n" + "=" * 70)
    print("Scenario 3: Log Reconciliation After Conflict")
    print("=" * 70)

    cluster3 = RaftCluster(size=3)
    leader3 = cluster3.nodes[0]
    leader3.become_leader()

    # Replicate an entry
    leader3.client_request("SET x=1")

    print("\nInitial state:")
    cluster3.print_all_logs()

    # Partition: leader alone
    cluster3.partition(group_a=[0], group_b=[1, 2])

    # Old leader appends entries it can't commit
    leader3.client_request("SET x=2")  # Uncommitted
    leader3.client_request("SET x=3")  # Uncommitted

    # New leader elected in majority partition
    new_leader3 = cluster3.nodes[1]
    new_leader3.current_term = 2
    new_leader3.become_leader()
    new_leader3.client_request("SET x=99")

    print("\nDuring partition (divergent logs):")
    cluster3.print_all_logs()

    # Heal and reconcile
    cluster3.heal_partition()
    new_leader3._replicate_to_all()
    new_leader3._advance_commit_index()

    print("After reconciliation (Node 0's uncommitted entries overwritten):")
    cluster3.print_all_logs()

    # Summary
    print("\n" + "=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
1. Log entries are committed only when replicated to a majority.
2. A leader in a minority partition cannot commit new entries.
3. After partition heals, the leader with the higher term wins.
4. Uncommitted entries from a stale leader are overwritten during
   log reconciliation (AppendEntries consistency check).
5. The commit index only advances for entries from the current term.
""")


if __name__ == "__main__":
    run_simulation()
    print("Done.")
