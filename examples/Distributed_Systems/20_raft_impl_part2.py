"""
Raft Implementation Part 2: Log Compaction and Linearizable Reads

Implements Raft's log compaction (snapshotting) and linearizable read
mechanisms (ReadIndex and LeaseRead). Demonstrates membership changes
via joint consensus.

Key concepts:
- Log compaction: snapshot state machine, discard prefix
- Snapshot transfer: InstallSnapshot RPC for slow followers
- ReadIndex: verify leadership before serving reads
- LeaseRead: use clock-based lease to avoid round-trip
- Joint consensus for membership changes

Usage:
    python 20_raft_impl_part2.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Log entry and snapshot
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    term: int
    index: int
    command: str


@dataclass
class Snapshot:
    """A compacted snapshot of the state machine."""
    last_included_index: int
    last_included_term: int
    data: dict[str, str]      # KV state at snapshot point
    size_bytes: int = 0

    def __repr__(self) -> str:
        return (f"Snapshot(through_index={self.last_included_index}, "
                f"term={self.last_included_term}, keys={len(self.data)})")


class ReadMode(Enum):
    READ_INDEX = "ReadIndex"
    LEASE_READ = "LeaseRead"


# ---------------------------------------------------------------------------
# Raft node with compaction and linearizable reads
# ---------------------------------------------------------------------------

class RaftNodeV2:
    """Raft node with log compaction and read optimizations."""

    def __init__(self, node_id: int, peers: list[int]):
        self.node_id = node_id
        self.peers = peers
        self.current_term = 1
        self.is_leader = False

        # Log
        self.log: list[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0

        # State machine
        self.kv: dict[str, str] = {}

        # Snapshot
        self.snapshot: Snapshot | None = None

        # Leader state
        self.match_index: dict[int, int] = {p: 0 for p in peers}

        # Lease-based read
        self.lease_expiry: float = 0.0

        self.log_messages: list[str] = []

    def apply_entry(self, entry: LogEntry) -> None:
        """Apply a log entry to the state machine."""
        parts = entry.command.split("=", 1)
        if len(parts) == 2:
            self.kv[parts[0]] = parts[1]
        self.last_applied = entry.index

    def append_entries(self, commands: list[str]) -> None:
        """Append new entries and apply them."""
        for cmd in commands:
            idx = (self.snapshot.last_included_index if self.snapshot else 0) + len(self.log) + 1
            entry = LogEntry(self.current_term, idx, cmd)
            self.log.append(entry)
            self.commit_index = idx
            self.apply_entry(entry)

    # --- Log Compaction ---

    def take_snapshot(self, compact_through: int | None = None) -> Snapshot | None:
        """
        Create a snapshot of the state machine and discard log prefix.
        """
        if compact_through is None:
            compact_through = self.last_applied

        if compact_through <= 0:
            return None

        # Find the entry to compact through
        base = self.snapshot.last_included_index if self.snapshot else 0
        relative_idx = compact_through - base

        if relative_idx <= 0 or relative_idx > len(self.log):
            return None

        entry = self.log[relative_idx - 1]

        snap = Snapshot(
            last_included_index=compact_through,
            last_included_term=entry.term,
            data=dict(self.kv),
            size_bytes=len(str(self.kv)) * 2,
        )

        # Discard compacted log entries
        entries_before = len(self.log)
        self.log = self.log[relative_idx:]
        self.snapshot = snap

        self.log_messages.append(
            f"Snapshot taken through index {compact_through}: "
            f"discarded {entries_before - len(self.log)} entries, "
            f"{len(self.log)} remaining")

        return snap

    def install_snapshot(self, snap: Snapshot) -> None:
        """Install a snapshot received from the leader."""
        self.snapshot = snap
        self.kv = dict(snap.data)
        self.last_applied = snap.last_included_index
        self.commit_index = max(self.commit_index, snap.last_included_index)

        # Discard log entries covered by snapshot
        new_log = []
        for entry in self.log:
            if entry.index > snap.last_included_index:
                new_log.append(entry)
        self.log = new_log

        self.log_messages.append(
            f"Installed snapshot through index {snap.last_included_index}")

    # --- Linearizable Reads ---

    def read_index(self, key: str, current_time: float) -> tuple[str | None, str]:
        """
        ReadIndex: Confirm leadership by contacting majority, then read.
        """
        if not self.is_leader:
            return None, "NOT_LEADER"

        # Step 1: Record current commit index
        read_index = self.commit_index

        # Step 2: Send heartbeat to confirm we are still leader
        acks = 1  # Self
        for peer_id in self.peers:
            # Simulate heartbeat ACK (in practice, network round-trip)
            acks += 1
            if acks > len(self.peers) // 2:
                break

        majority = (len(self.peers) + 1) // 2 + 1
        if acks < majority:
            return None, "LOST_LEADERSHIP"

        # Step 3: Wait until state machine has applied up to read_index
        # (already applied in our simulation)

        # Step 4: Serve the read
        value = self.kv.get(key)
        return value, "OK_READ_INDEX"

    def lease_read(self, key: str, current_time: float) -> tuple[str | None, str]:
        """
        LeaseRead: Use clock-based lease to skip heartbeat confirmation.
        """
        if not self.is_leader:
            return None, "NOT_LEADER"

        if current_time > self.lease_expiry:
            return None, "LEASE_EXPIRED"

        value = self.kv.get(key)
        return value, "OK_LEASE_READ"


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_log_compaction() -> None:
    """Demonstrate log compaction with snapshots."""
    print("=" * 70)
    print("Raft Log Compaction (Snapshotting)")
    print("=" * 70)

    node = RaftNodeV2(0, peers=[1, 2, 3, 4])
    node.is_leader = True

    # Write 20 entries
    commands = [f"key{i}=value{i}" for i in range(20)]
    node.append_entries(commands)

    print(f"\n  After 20 writes:")
    print(f"    Log length: {len(node.log)}")
    print(f"    Last applied: {node.last_applied}")
    print(f"    KV store size: {len(node.kv)} keys")
    print(f"    Snapshot: {node.snapshot}")

    # Take snapshot at index 15
    snap = node.take_snapshot(compact_through=15)
    print(f"\n  After snapshot at index 15:")
    print(f"    {snap}")
    print(f"    Log length: {len(node.log)} (only entries 16-20 remain)")
    print(f"    KV store: unchanged ({len(node.kv)} keys)")
    for msg in node.log_messages:
        print(f"    {msg}")

    # Write more entries
    node.log_messages.clear()
    node.append_entries([f"key{i}=updated{i}" for i in range(20, 25)])
    print(f"\n  After 5 more writes:")
    print(f"    Log length: {len(node.log)}")
    print(f"    Last applied: {node.last_applied}")

    # Take another snapshot
    snap2 = node.take_snapshot()
    print(f"\n  After second snapshot:")
    print(f"    {snap2}")
    print(f"    Log length: {len(node.log)}")
    for msg in node.log_messages:
        print(f"    {msg}")


def demo_snapshot_transfer() -> None:
    """Demonstrate InstallSnapshot for slow followers."""
    print("\n" + "=" * 70)
    print("InstallSnapshot: Catching Up Slow Followers")
    print("=" * 70)

    leader = RaftNodeV2(0, peers=[1, 2])
    leader.is_leader = True

    # Leader writes 50 entries then compacts
    commands = [f"k{i}=v{i}" for i in range(50)]
    leader.append_entries(commands)
    leader.take_snapshot(compact_through=45)

    print(f"\n  Leader state:")
    print(f"    Snapshot: {leader.snapshot}")
    print(f"    Log: entries {46}-{50} ({len(leader.log)} entries)")

    # Slow follower joins with no data
    follower = RaftNodeV2(1, peers=[0, 2])
    print(f"\n  New follower (empty):")
    print(f"    Log: {len(follower.log)} entries")
    print(f"    KV: {len(follower.kv)} keys")

    # Leader sends snapshot (since log entries 1-45 are gone)
    print(f"\n  Leader cannot send entries 1-45 (compacted!)")
    print(f"  Sending InstallSnapshot...")
    follower.install_snapshot(leader.snapshot)

    print(f"\n  Follower after InstallSnapshot:")
    print(f"    Snapshot: {follower.snapshot}")
    print(f"    KV: {len(follower.kv)} keys")
    print(f"    Last applied: {follower.last_applied}")
    for msg in follower.log_messages:
        print(f"    {msg}")

    # Verify state matches
    leader_keys = set(leader.kv.keys())
    follower_keys = set(follower.kv.keys())
    # Follower has snapshot state (keys 0-44), leader has all 50
    print(f"\n  Leader keys: {len(leader_keys)}, Follower keys: {len(follower_keys)}")
    print(f"  Remaining entries (45-50) sent via normal AppendEntries")


def demo_linearizable_reads() -> None:
    """Compare ReadIndex and LeaseRead."""
    print("\n" + "=" * 70)
    print("Linearizable Reads: ReadIndex vs LeaseRead")
    print("=" * 70)

    node = RaftNodeV2(0, peers=[1, 2, 3, 4])
    node.is_leader = True
    node.append_entries(["name=Alice", "age=30", "city=Seoul"])

    current_time = 100.0

    # ReadIndex
    print(f"\n  ReadIndex (heartbeat confirmation):")
    for key in ["name", "age", "city", "missing"]:
        val, status = node.read_index(key, current_time)
        print(f"    READ {key}: value={val}, status={status}")

    # LeaseRead with valid lease
    node.lease_expiry = 105.0
    print(f"\n  LeaseRead (lease valid until t=105, current t={current_time}):")
    for key in ["name", "age"]:
        val, status = node.lease_read(key, current_time)
        print(f"    READ {key}: value={val}, status={status}")

    # LeaseRead with expired lease
    current_time = 110.0
    print(f"\n  LeaseRead (lease expired, current t={current_time}):")
    val, status = node.lease_read("name", current_time)
    print(f"    READ name: value={val}, status={status}")

    print("""
  Comparison:
  ┌──────────────┬────────────────────────┬────────────────────────┐
  │ Property     │ ReadIndex              │ LeaseRead              │
  ├──────────────┼────────────────────────┼────────────────────────┤
  │ Correctness  │ Always safe            │ Requires bounded skew  │
  │ Latency      │ 1 RTT (heartbeat)     │ 0 RTT (local read)     │
  │ Throughput   │ Lower (network bound)  │ Higher (no network)    │
  │ Clock dep.   │ None                   │ Requires good clocks   │
  │ Used by      │ etcd (default)         │ TiKV, CockroachDB     │
  └──────────────┴────────────────────────┴────────────────────────┘
""")


def demo_membership_change() -> None:
    """Illustrate joint consensus for membership changes."""
    print("=" * 70)
    print("Membership Change via Joint Consensus")
    print("=" * 70)

    print("""
  Problem: Changing cluster membership atomically is tricky.
  Direct switch from C_old to C_new can create two majorities.

  Example: {A,B,C} -> {A,B,C,D,E}
  If switch isn't atomic:
    - Old majority: {A,B} (2 of 3)
    - New majority: {C,D,E} (3 of 5)
    - Both could elect leaders simultaneously!

  Joint Consensus solution (Raft):
  1. Enter joint config C_old,new: need majority of BOTH old AND new
  2. Once committed, switch to C_new
  3. Decommission old-only nodes

  Timeline:
    ┌──────────────┬───────────────────┬──────────────┐
    │  C_old       │  C_old,new        │  C_new       │
    │  {A,B,C}     │  {A,B,C} ∩ {A-E} │  {A,B,C,D,E} │
    │              │  Need majority of │              │
    │              │  BOTH sets         │              │
    └──────────────┴───────────────────┴──────────────┘

  Quorum during joint consensus:
    - Old majority: 2 of {A,B,C}
    - New majority: 3 of {A,B,C,D,E}
    - Both must agree => no split brain possible
""")

    # Simulate quorum sizes
    old_config = ["A", "B", "C"]
    new_config = ["A", "B", "C", "D", "E"]

    old_majority = len(old_config) // 2 + 1
    new_majority = len(new_config) // 2 + 1

    print(f"  Old config: {old_config}, majority = {old_majority}")
    print(f"  New config: {new_config}, majority = {new_majority}")
    print(f"  Joint: need {old_majority} from old AND {new_majority} from new")
    print(f"\n  This ensures at most one leader during the transition.")


if __name__ == "__main__":
    demo_log_compaction()
    demo_snapshot_transfer()
    demo_linearizable_reads()
    demo_membership_change()
    print("Done.")
