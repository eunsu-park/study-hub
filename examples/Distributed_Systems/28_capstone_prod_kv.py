"""
Capstone: Production Distributed KV Store (Raft + Sharding + Replication)

Extends the basic distributed KV store (example 08) with sharding across
multiple Raft groups, shard rebalancing, client routing, and a production
readiness checklist. This is a high-level architectural simulation.

Key concepts:
- Multi-Raft: one Raft group per shard (like CockroachDB ranges)
- Hash-based sharding with configurable shard count
- Shard-to-group mapping and client-side routing
- Shard splitting when a shard grows too large
- Health monitoring and automatic failover
- Production readiness: observability, backups, upgrades

Usage:
    python 28_capstone_prod_kv.py
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Shard and Routing
# ---------------------------------------------------------------------------

NUM_SHARDS = 8


def key_to_shard(key: str, num_shards: int = NUM_SHARDS) -> int:
    """Hash a key to a shard ID."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return h % num_shards


@dataclass
class ShardInfo:
    """Metadata about a shard."""
    shard_id: int
    raft_group_id: int
    key_count: int = 0
    size_bytes: int = 0
    leader_node: str = ""


class ShardRouter:
    """Routes client requests to the correct Raft group."""

    def __init__(self, num_shards: int = NUM_SHARDS):
        self.num_shards = num_shards
        self.shard_map: dict[int, ShardInfo] = {}
        self.log: list[str] = []

    def assign_shard(self, shard_id: int, raft_group_id: int,
                     leader_node: str) -> None:
        self.shard_map[shard_id] = ShardInfo(
            shard_id=shard_id,
            raft_group_id=raft_group_id,
            leader_node=leader_node,
        )

    def route(self, key: str) -> ShardInfo | None:
        shard_id = key_to_shard(key, self.num_shards)
        return self.shard_map.get(shard_id)


# ---------------------------------------------------------------------------
# Simplified Multi-Raft
# ---------------------------------------------------------------------------

class NodeRole(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    DOWN = "down"


@dataclass
class RaftGroupNode:
    """A node participating in a Raft group."""
    node_id: str
    role: NodeRole = NodeRole.FOLLOWER
    term: int = 0


@dataclass
class RaftGroup:
    """A Raft consensus group managing one or more shards."""
    group_id: int
    nodes: dict[str, RaftGroupNode] = field(default_factory=dict)
    kv_store: dict[str, str] = field(default_factory=dict)
    commit_index: int = 0
    log_entries: list[tuple[int, str, str]] = field(default_factory=list)  # (term, key, val)

    @property
    def leader(self) -> str | None:
        for nid, node in self.nodes.items():
            if node.role == NodeRole.LEADER:
                return nid
        return None

    def elect_leader(self, node_id: str) -> None:
        for nid, node in self.nodes.items():
            node.role = NodeRole.FOLLOWER
        if node_id in self.nodes:
            self.nodes[node_id].role = NodeRole.LEADER
            self.nodes[node_id].term += 1

    def put(self, key: str, value: str) -> bool:
        leader = self.leader
        if leader is None:
            return False
        self.log_entries.append((self.nodes[leader].term, key, value))
        self.kv_store[key] = value
        self.commit_index += 1
        return True

    def get(self, key: str) -> str | None:
        return self.kv_store.get(key)


# ---------------------------------------------------------------------------
# Distributed KV Store
# ---------------------------------------------------------------------------

class ProductionKVStore:
    """
    Production-grade distributed KV store architecture.
    Multiple Raft groups, each owning a subset of shards.
    """

    def __init__(self, num_shards: int = NUM_SHARDS,
                 num_groups: int = 4, nodes_per_group: int = 3):
        self.router = ShardRouter(num_shards)
        self.groups: dict[int, RaftGroup] = {}
        self.metrics: dict[str, int] = defaultdict(int)
        self.log: list[str] = []

        # Create Raft groups
        for g in range(num_groups):
            group = RaftGroup(group_id=g)
            for n in range(nodes_per_group):
                node_id = f"node-{g}-{n}"
                group.nodes[node_id] = RaftGroupNode(node_id)
            # Elect first node as leader
            leader = f"node-{g}-0"
            group.elect_leader(leader)
            self.groups[g] = group

        # Assign shards to groups (round-robin)
        for shard_id in range(num_shards):
            group_id = shard_id % num_groups
            leader = self.groups[group_id].leader or f"node-{group_id}-0"
            self.router.assign_shard(shard_id, group_id, leader)

    def put(self, key: str, value: str) -> bool:
        """Write a key-value pair."""
        shard_info = self.router.route(key)
        if shard_info is None:
            self.metrics["errors"] += 1
            return False

        group = self.groups[shard_info.raft_group_id]
        ok = group.put(key, value)
        if ok:
            shard_info.key_count += 1
            shard_info.size_bytes += len(key) + len(value)
            self.metrics["writes"] += 1
        else:
            self.metrics["errors"] += 1
        return ok

    def get(self, key: str) -> str | None:
        """Read a value."""
        shard_info = self.router.route(key)
        if shard_info is None:
            self.metrics["errors"] += 1
            return None

        group = self.groups[shard_info.raft_group_id]
        self.metrics["reads"] += 1
        return group.get(key)

    def status(self) -> dict:
        """Get cluster status."""
        return {
            "num_shards": self.router.num_shards,
            "num_groups": len(self.groups),
            "total_keys": sum(
                len(g.kv_store) for g in self.groups.values()),
            "metrics": dict(self.metrics),
        }


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_sharded_kv() -> None:
    """Demonstrate the sharded KV store."""
    print("=" * 70)
    print("Production KV Store: Multi-Raft with Sharding")
    print("=" * 70)

    store = ProductionKVStore(num_shards=8, num_groups=4, nodes_per_group=3)

    # Show shard-to-group mapping
    print(f"\n  Shard -> Raft Group mapping:")
    for sid, info in sorted(store.router.shard_map.items()):
        print(f"    Shard {sid} -> Group {info.raft_group_id} "
              f"(leader: {info.leader_node})")

    # Write data
    data = {f"user:{i}": f"data-{i}" for i in range(100)}
    for key, value in data.items():
        store.put(key, value)

    # Show distribution
    print(f"\n  After 100 writes:")
    for sid, info in sorted(store.router.shard_map.items()):
        print(f"    Shard {sid}: {info.key_count} keys, "
              f"{info.size_bytes} bytes")

    # Read data
    for i in range(0, 100, 25):
        key = f"user:{i}"
        val = store.get(key)
        shard = key_to_shard(key)
        print(f"\n    GET {key} => {val} (shard {shard})")

    status = store.status()
    print(f"\n  Cluster status: {status}")


def demo_failover() -> None:
    """Demonstrate automatic leader failover."""
    print("\n" + "=" * 70)
    print("Automatic Failover")
    print("=" * 70)

    store = ProductionKVStore(num_shards=4, num_groups=2, nodes_per_group=3)
    store.put("x", "1")

    group = store.groups[0]
    leader = group.leader
    print(f"\n  Group 0 leader: {leader}")

    # Crash leader
    print(f"  Crashing {leader}...")
    group.nodes[leader].role = NodeRole.DOWN

    # Try write — fails because no leader
    ok = store.put("x", "2")
    print(f"  PUT x=2: {'OK' if ok else 'FAILED (no leader)'}")

    # Elect new leader
    new_leader = [nid for nid, n in group.nodes.items()
                  if n.role != NodeRole.DOWN][0]
    group.elect_leader(new_leader)
    print(f"  New leader elected: {new_leader}")

    # Update router
    for sid, info in store.router.shard_map.items():
        if info.raft_group_id == 0:
            info.leader_node = new_leader

    # Retry write
    ok = store.put("x", "2")
    print(f"  PUT x=2: {'OK' if ok else 'FAILED'}")
    print(f"  GET x => {store.get('x')}")


def demo_shard_rebalancing() -> None:
    """Demonstrate shard rebalancing when adding a new group."""
    print("\n" + "=" * 70)
    print("Shard Rebalancing: Adding a New Raft Group")
    print("=" * 70)

    store = ProductionKVStore(num_shards=8, num_groups=2, nodes_per_group=3)

    # Write data
    for i in range(50):
        store.put(f"key-{i}", f"val-{i}")

    print(f"\n  Before rebalancing (2 groups, 8 shards):")
    for gid, group in store.groups.items():
        shards = [sid for sid, info in store.router.shard_map.items()
                  if info.raft_group_id == gid]
        print(f"    Group {gid}: shards {shards}, keys: {len(group.kv_store)}")

    # Add new group
    new_group = RaftGroup(group_id=2)
    for n in range(3):
        nid = f"node-2-{n}"
        new_group.nodes[nid] = RaftGroupNode(nid)
    new_group.elect_leader("node-2-0")
    store.groups[2] = new_group

    # Rebalance: move some shards to the new group
    # Move shards 5, 6, 7 to group 2
    for sid in [5, 6, 7]:
        old_group_id = store.router.shard_map[sid].raft_group_id
        old_group = store.groups[old_group_id]

        # Transfer keys belonging to this shard
        keys_to_move = [k for k in old_group.kv_store
                        if key_to_shard(k) == sid]
        for key in keys_to_move:
            new_group.kv_store[key] = old_group.kv_store.pop(key)

        store.router.assign_shard(sid, 2, "node-2-0")

    print(f"\n  After rebalancing (3 groups, 8 shards):")
    for gid, group in sorted(store.groups.items()):
        shards = [sid for sid, info in store.router.shard_map.items()
                  if info.raft_group_id == gid]
        print(f"    Group {gid}: shards {shards}, keys: {len(group.kv_store)}")


def demo_production_checklist() -> None:
    """Print production readiness checklist."""
    print("\n" + "=" * 70)
    print("Production Readiness Checklist")
    print("=" * 70)

    print("""
  ┌───┬──────────────────────┬─────────────────────────────────────┐
  │ # │ Category             │ Requirements                        │
  ├───┼──────────────────────┼─────────────────────────────────────┤
  │ 1 │ Consensus            │ Raft with pre-vote, learner nodes   │
  │ 2 │ Sharding             │ Range or hash sharding, auto-split  │
  │ 3 │ Replication          │ 3 or 5 replicas per shard           │
  │ 4 │ Failover             │ Automatic leader election <10s      │
  │ 5 │ Rebalancing          │ Online shard migration              │
  │ 6 │ Compaction           │ Log compaction with snapshots       │
  │ 7 │ Linearizable reads   │ ReadIndex or LeaseRead              │
  │ 8 │ Observability        │ Metrics, tracing, structured logs   │
  │ 9 │ Backup/Restore       │ Consistent snapshots, point-in-time │
  │10 │ Rolling Upgrades     │ Version-compatible Raft protocol     │
  │11 │ Rate Limiting        │ Per-client and per-shard limits      │
  │12 │ Authentication       │ mTLS between nodes, client auth      │
  │13 │ Testing              │ Jepsen, chaos engineering, sim tests │
  │14 │ Capacity Planning    │ Auto-scaling triggers, load testing  │
  └───┴──────────────────────┴─────────────────────────────────────┘

  Architecture diagram:
  ┌─────────┐     ┌─────────────────────────────────────┐
  │ Client  │────▶│ Router / Load Balancer               │
  └─────────┘     └──────────┬──────────┬───────────────┘
                             │          │
                   ┌─────────▼──┐  ┌────▼──────────┐
                   │ Raft Group 0│  │ Raft Group 1  │ ...
                   │ [Shards 0,1]│  │ [Shards 2,3]  │
                   │ L F F       │  │ L F F         │
                   └─────────────┘  └───────────────┘

  L = Leader, F = Follower
  Each Raft group is an independent consensus instance.
""")


if __name__ == "__main__":
    demo_sharded_kv()
    demo_failover()
    demo_shard_rebalancing()
    demo_production_checklist()
    print("Done.")
