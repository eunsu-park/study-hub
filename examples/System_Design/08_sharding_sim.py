"""
Database Sharding Simulation

Demonstrates:
- Hash-based sharding
- Range-based sharding
- Directory-based sharding
- Shard rebalancing on node addition
- Hot-spot analysis

Theory:
- Sharding partitions data across multiple database instances.
- Hash-based: key → hash(key) % N. Uniform distribution but hard to
  range-query and rebalancing is expensive.
- Range-based: key ranges mapped to shards. Good for range queries
  but can create hot spots.
- Directory-based: lookup table maps keys to shards. Flexible but
  the directory is a single point of failure.

Adapted from System Design Lesson 08.
"""

import hashlib
from collections import defaultdict


# ── Hash-Based Sharding ───────────────────────────────────────────────

# Why: Hash-based sharding distributes data uniformly regardless of key patterns.
# This prevents hot spots from sequential IDs (e.g., user_1001, user_1002 all
# landing on the same shard), but sacrifices the ability to do efficient range queries.
class HashShard:
    """Hash-based sharding."""

    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        self.shards: dict[int, dict[str, str]] = {
            i: {} for i in range(num_shards)
        }

    def _get_shard(self, key: str) -> int:
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return h % self.num_shards

    def put(self, key: str, value: str) -> int:
        shard_id = self._get_shard(key)
        self.shards[shard_id][key] = value
        return shard_id

    def get(self, key: str) -> tuple[int, str | None]:
        shard_id = self._get_shard(key)
        return shard_id, self.shards[shard_id].get(key)

    def distribution(self) -> dict[int, int]:
        return {i: len(self.shards[i]) for i in range(self.num_shards)}


# ── Range-Based Sharding ──────────────────────────────────────────────

# Why: Range-based sharding enables efficient range queries (e.g., "all orders
# from January") since contiguous keys live on the same shard. The downside is
# hot spots when traffic concentrates on recent data (e.g., latest timestamps).
class RangeShard:
    """Range-based sharding on integer keys."""

    def __init__(self, boundaries: list[int]):
        """boundaries: sorted list of split points.
        E.g. [100, 200, 300] → shard0: [0,100), shard1: [100,200),
        shard2: [200,300), shard3: [300,∞)
        """
        self.boundaries = sorted(boundaries)
        self.num_shards = len(boundaries) + 1
        self.shards: dict[int, dict[int, str]] = {
            i: {} for i in range(self.num_shards)
        }

    def _get_shard(self, key: int) -> int:
        for i, boundary in enumerate(self.boundaries):
            if key < boundary:
                return i
        return self.num_shards - 1

    def put(self, key: int, value: str) -> int:
        shard_id = self._get_shard(key)
        self.shards[shard_id][key] = value
        return shard_id

    def get(self, key: int) -> tuple[int, str | None]:
        shard_id = self._get_shard(key)
        return shard_id, self.shards[shard_id].get(key)

    def range_query(self, low: int, high: int) -> list[tuple[int, str]]:
        """Range query: return all (key, value) where low <= key < high."""
        # Why: Range queries are the main advantage of range sharding. In a real
        # system, only the shards covering [low, high) would be contacted,
        # avoiding a scatter-gather across all shards.
        results = []
        for shard_id in range(self.num_shards):
            for key, value in self.shards[shard_id].items():
                if low <= key < high:
                    results.append((key, value))
        return sorted(results)

    def distribution(self) -> dict[int, int]:
        return {i: len(self.shards[i]) for i in range(self.num_shards)}


# ── Directory-Based Sharding ──────────────────────────────────────────

# Why: Directory-based sharding decouples key placement from any formula, making
# rebalancing trivial (just update the directory). The trade-off: the directory
# itself becomes a single point of failure and a potential bottleneck.
class DirectoryShard:
    """Directory-based sharding with explicit key→shard mapping."""

    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        self.directory: dict[str, int] = {}
        self.shards: dict[int, dict[str, str]] = {
            i: {} for i in range(num_shards)
        }
        self.next_shard = 0  # round-robin assignment

    def _assign_shard(self, key: str) -> int:
        if key not in self.directory:
            shard_id = self.next_shard
            self.directory[key] = shard_id
            self.next_shard = (self.next_shard + 1) % self.num_shards
        return self.directory[key]

    def put(self, key: str, value: str) -> int:
        shard_id = self._assign_shard(key)
        self.shards[shard_id][key] = value
        return shard_id

    def get(self, key: str) -> tuple[int, str | None]:
        shard_id = self.directory.get(key, -1)
        if shard_id == -1:
            return -1, None
        return shard_id, self.shards[shard_id].get(key)

    def migrate(self, key: str, new_shard: int) -> bool:
        """Move a key to a different shard."""
        # Why: Directory sharding makes per-key migration trivial — just move
        # the data and update the directory entry. Hash/range sharding cannot
        # do this without resharding entire partitions.
        if key not in self.directory:
            return False
        old_shard = self.directory[key]
        if old_shard == new_shard:
            return False
        value = self.shards[old_shard].pop(key, None)
        if value is not None:
            self.shards[new_shard][key] = value
            self.directory[key] = new_shard
            return True
        return False

    def distribution(self) -> dict[int, int]:
        return {i: len(self.shards[i]) for i in range(self.num_shards)}


# ── Demos ──────────────────────────────────────────────────────────────

def print_dist(dist: dict[int, int], label: str) -> None:
    total = sum(dist.values())
    print(f"\n  {label}:")
    print(f"    {'Shard':>8}  {'Count':>6}  {'Share':>6}  Distribution")
    print(f"    {'-'*8}  {'-'*6}  {'-'*6}  {'-'*20}")
    for shard in sorted(dist):
        count = dist[shard]
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    Shard-{shard:>2}  {count:>6}  {pct:>5.1f}%  {bar}")


def demo_hash_sharding():
    print("=" * 60)
    print("HASH-BASED SHARDING")
    print("=" * 60)

    hs = HashShard(num_shards=4)

    # Insert 1000 keys
    for i in range(1000):
        hs.put(f"user:{i}", f"data-{i}")

    print_dist(hs.distribution(), "1000 keys across 4 shards")

    # Lookup
    print(f"\n  Sample lookups:")
    for key in ["user:42", "user:500", "user:999"]:
        shard, val = hs.get(key)
        print(f"    {key} → shard {shard}")

    # Rebalancing problem
    print(f"\n  Rebalancing problem (adding shard 5):")
    hs5 = HashShard(num_shards=5)
    moved = 0
    for i in range(1000):
        key = f"user:{i}"
        old_shard = hs._get_shard(key)
        new_shard = hs5._get_shard(key)
        if old_shard != new_shard:
            moved += 1
    print(f"    Keys that change shard: {moved}/1000 ({moved/10:.0f}%)")
    print(f"    Hash-based sharding requires moving ~{moved/10:.0f}% of data!")


def demo_range_sharding():
    print("\n" + "=" * 60)
    print("RANGE-BASED SHARDING")
    print("=" * 60)

    rs = RangeShard(boundaries=[250, 500, 750])
    print(f"\n  Boundaries: [250, 500, 750]")
    print(f"  Shard 0: [0, 250), Shard 1: [250, 500)")
    print(f"  Shard 2: [500, 750), Shard 3: [750, ∞)")

    # Uniform distribution
    print(f"\n  Uniform keys (0-999):")
    for i in range(1000):
        rs.put(i, f"data-{i}")
    print_dist(rs.distribution(), "Distribution")

    # Range query
    results = rs.range_query(100, 200)
    print(f"\n  Range query [100, 200): {len(results)} results")
    print(f"  First 5: {results[:5]}")

    # Hot spot scenario
    print(f"\n  Hot-spot scenario (skewed keys):")
    rs_hot = RangeShard(boundaries=[250, 500, 750])
    import random
    random.seed(42)
    for _ in range(1000):
        # 70% of keys in range [0, 250)
        if random.random() < 0.7:
            key = random.randint(0, 249)
        else:
            key = random.randint(250, 999)
        rs_hot.put(key, "data")
    print_dist(rs_hot.distribution(), "Skewed distribution")
    print(f"    Shard 0 is a hot spot!")


def demo_directory_sharding():
    print("\n" + "=" * 60)
    print("DIRECTORY-BASED SHARDING")
    print("=" * 60)

    ds = DirectoryShard(num_shards=3)

    # Insert keys (round-robin assignment)
    for i in range(12):
        shard = ds.put(f"key-{i}", f"val-{i}")
        print(f"    key-{i} → shard {shard}")

    print_dist(ds.distribution(), "Initial distribution")

    # Migration
    print(f"\n  Migrating key-0 from shard 0 to shard 2:")
    ds.migrate("key-0", 2)
    shard, val = ds.get("key-0")
    print(f"    key-0 now at shard {shard}: {val}")
    print_dist(ds.distribution(), "After migration")


def demo_comparison():
    print("\n" + "=" * 60)
    print("SHARDING STRATEGY COMPARISON")
    print("=" * 60)

    print(f"""
  {'Feature':<25} {'Hash':>10} {'Range':>10} {'Directory':>10}
  {'-'*25} {'-'*10} {'-'*10} {'-'*10}
  {'Even distribution':<25} {'Good':>10} {'Depends':>10} {'Manual':>10}
  {'Range queries':<25} {'Hard':>10} {'Easy':>10} {'Hard':>10}
  {'Rebalancing':<25} {'Hard':>10} {'Easy':>10} {'Easy':>10}
  {'Hot-spot risk':<25} {'Low':>10} {'High':>10} {'Low':>10}
  {'Complexity':<25} {'Low':>10} {'Medium':>10} {'High':>10}
  {'Lookup overhead':<25} {'O(1)':>10} {'O(log N)':>10} {'O(1)+net':>10}""")


if __name__ == "__main__":
    demo_hash_sharding()
    demo_range_sharding()
    demo_directory_sharding()
    demo_comparison()
