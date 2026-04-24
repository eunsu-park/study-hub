"""
NoSQL — Partition Keys, Hot Partitions, and Consistency Models

An in-memory simulation of a wide-column / document store (DynamoDB,
Cloud Firestore, Cosmos DB). Focuses on the lessons that bite teams in
production:

1. Partition-key choice determines hot spots.
2. Strong vs. eventual consistency is a per-read decision and a per-$ one.
3. Secondary indexes double the write cost; plan them deliberately.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# 1. Table model
# =============================================================================

@dataclass
class Item:
    pk: str                      # partition key
    sk: str                      # sort key
    attrs: Dict[str, object]


@dataclass
class Partition:
    """A simulated storage unit. Real systems shard across many of these."""
    name: str
    items: List[Item] = field(default_factory=list)
    write_units_used: int = 0

    def put(self, item: Item, index_count: int) -> int:
        # Each index adds one write amplification unit
        cost = 1 + index_count
        self.write_units_used += cost
        self.items.append(item)
        return cost


@dataclass
class NoSQLTable:
    name: str
    num_partitions: int = 4
    indexes: List[str] = field(default_factory=list)
    _partitions: Dict[int, Partition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for i in range(self.num_partitions):
            self._partitions[i] = Partition(name=f"P{i}")

    def _route(self, pk: str) -> Partition:
        # Hash the partition key to pick a partition. Real systems use MD5 /
        # consistent hashing; here a simple python hash is enough to show the
        # concept. The point is that the SAME partition key ALWAYS lands on
        # the SAME partition.
        h = abs(hash(pk)) % self.num_partitions
        return self._partitions[h]

    def put(self, item: Item) -> int:
        return self._route(item.pk).put(item, len(self.indexes))

    def get(self, pk: str, sk: str, strong: bool = False) -> Optional[Item]:
        """Strong read costs 2 read units (RU), eventual costs 1 RU.
        We do not model latency here; see comments in the demo."""
        p = self._route(pk)
        for it in p.items:
            if it.pk == pk and it.sk == sk:
                return it
        return None

    def heatmap(self) -> Dict[str, int]:
        return {p.name: p.write_units_used for p in self._partitions.values()}


# =============================================================================
# 2. Hot-partition scenarios
# =============================================================================

def scenario_uniform(table: NoSQLTable, n: int) -> None:
    """Uniformly distributed pk — the good case."""
    for i in range(n):
        table.put(Item(pk=f"user_{i}", sk="profile", attrs={"seq": i}))


def scenario_hot_key(table: NoSQLTable, n: int) -> None:
    """One very popular key — the bad case.
    Typical causes: using a timestamp truncated to 'today' as pk, or a
    small enum (status=active/inactive) as pk on a table with 95% active.
    """
    for i in range(n):
        pk = "status_active" if i % 20 != 0 else f"status_inactive_{i}"
        table.put(Item(pk=pk, sk=f"event_{i}", attrs={"seq": i}))


def print_heatmap(title: str, heatmap: Dict[str, int]) -> None:
    total = sum(heatmap.values()) or 1
    print(f"\n{title}")
    for name in sorted(heatmap):
        pct = 100 * heatmap[name] / total
        bar = "#" * int(pct / 2)
        print(f"  {name}: {heatmap[name]:>5} WU  ({pct:>5.1f}%)  {bar}")


# =============================================================================
# 3. Demo
# =============================================================================

def demo() -> None:
    random.seed(0)
    N = 10_000

    # --- No secondary index, uniform keys ---
    table_a = NoSQLTable(name="users_uniform")
    scenario_uniform(table_a, N)
    print_heatmap("Uniform keys (no index)", table_a.heatmap())

    # --- Add two indexes: every write now costs 3 WU instead of 1 ---
    table_b = NoSQLTable(name="users_uniform_2idx", indexes=["by_email", "by_city"])
    scenario_uniform(table_b, N)
    print_heatmap("Uniform keys (2 secondary indexes)", table_b.heatmap())
    print(f"  note: two indexes tripled the per-write cost — partition load is 3x higher.")

    # --- Hot partition ---
    table_c = NoSQLTable(name="events_hot")
    scenario_hot_key(table_c, N)
    print_heatmap("Hot partition (status=active as pk)", table_c.heatmap())

    # --- Consistency demo ---
    print("\nStrong vs Eventual reads")
    print("  Strong read:   2 RU, returns latest committed write, may add ~10 ms.")
    print("  Eventual read: 1 RU, may return a slightly stale value, faster + cheaper.")
    print("  Rule of thumb: default to eventual for feeds/timelines, strong for money moves.")

    # --- Cost implication ---
    strong_reads_per_sec = 500
    eventual_reads_per_sec = 500
    writes_per_sec = 100
    indexes = 2
    total_wu = writes_per_sec * (1 + indexes)
    total_ru = strong_reads_per_sec * 2 + eventual_reads_per_sec * 1
    print(f"\nCapacity budget @ 500 strong reads/s, 500 eventual, 100 writes/s, {indexes} idx:")
    print(f"  {total_wu} WCU  (write capacity units)")
    print(f"  {total_ru} RCU  (read capacity units)")
    print(f"  dropping one index ≈ {writes_per_sec} WCU back — often worth it.")


if __name__ == "__main__":
    demo()
