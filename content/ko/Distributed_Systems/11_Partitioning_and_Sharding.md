# 레슨 11: 파티셔닝과 샤딩

[개요](./00_Overview.md) | [이전: CRDT와 최종 일관성](./10_CRDTs_and_Eventual_Consistency.md) | [다음: 분산 스토리지 사례 연구](./12_Distributed_Storage_Case_Studies.md)

---

## 학습 목표

- 파티셔닝이 쓰기 확장과 대규모 데이터셋 관리에 필수적인 이유 이해
- Key-range, hash, compound 파티셔닝 전략의 비교 및 트레이드오프 분석
- 가상 노드를 사용한 consistent hashing 구현 및 키 분산 균일성 분석
- 보조 인덱스 파티셔닝 접근법(local vs global)과 쿼리에 미치는 영향 평가
- 프로덕션 시스템을 위한 리밸런싱 전략 및 핫스팟 완화 기법 설계

---

## 1. 왜 데이터를 파티셔닝하는가?

복제(레슨 9)는 동일한 데이터를 여러 노드에 복사하여 장애 허용과 읽기 처리량을 향상시킨다. 하지만 복제는 다음의 경우에 도움이 **되지 않는다**:

- 데이터셋이 단일 머신에 맞지 않을 정도로 큰 경우
- 쓰기 처리량이 단일 머신이 처리할 수 있는 것을 초과하는 경우
- 규제나 지연 시간 이유로 쿼리를 지리적으로 가까운 노드에서 처리해야 하는 경우

**파티셔닝** (**샤딩**이라고도 함)은 데이터를 여러 노드에 분할하여, 각 노드가 데이터의 일부를 담당한다.

```
Single Node (before partitioning):
┌─────────────────────────────────────────┐
│           All Data (10 TB)              │
│     All Writes (50,000 QPS)             │
└─────────────────────────────────────────┘
     ⇓ Cannot scale further ⇓

Partitioned (4 nodes):
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Shard 1  │ │ Shard 2  │ │ Shard 3  │ │ Shard 4  │
│ 2.5 TB   │ │ 2.5 TB   │ │ 2.5 TB   │ │ 2.5 TB   │
│ ~12.5K   │ │ ~12.5K   │ │ ~12.5K   │ │ ~12.5K   │
│  QPS     │ │  QPS     │ │  QPS     │ │  QPS     │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 1.1 좋은 파티셔닝 방식의 목표

| 목표 | 설명 |
|---|---|
| **균형 잡힌 부하** | 각 파티션이 대략 동일한 데이터 볼륨과 쿼리 부하를 처리 |
| **최소한의 교차 파티션 쿼리** | 대부분의 쿼리가 하나의 파티션만 접근 |
| **효율적인 리밸런싱** | 노드 추가/제거 시 최소한의 데이터 이동 |
| **핫스팟 방지** | 단일 파티션이 불균형적인 트래픽을 받지 않음 |

### 1.2 파티셔닝 + 복제

실제로 파티셔닝과 복제는 결합된다. 각 파티션은 장애 허용을 위해 여러 노드에 복제된다.

```
Partition scheme with replication factor 3:

Node 1: [P1-leader] [P2-follower] [P4-follower]
Node 2: [P2-leader] [P3-follower] [P1-follower]
Node 3: [P3-leader] [P4-follower] [P2-follower]
Node 4: [P4-leader] [P1-follower] [P3-follower]
```

---

## 2. 파티셔닝 전략

### 2.1 Key-Range 파티셔닝

각 파티션에 사전순 또는 숫자순으로 정렬된 연속 키 범위를 할당한다.

```
Partition boundaries for user IDs (sorted):
  P1: [0000, 2499]
  P2: [2500, 4999]
  P3: [5000, 7499]
  P4: [7500, 9999]

Query: WHERE user_id = 3500  →  routes to P2
Range: WHERE user_id BETWEEN 3000 AND 4000  →  routes to P2 (single partition!)
```

**장점**:
- 효율적인 범위 쿼리: 연속 키가 동일 파티션에 위치
- 효율적인 정렬 스캔: 데이터가 각 파티션 내에서 이미 정렬됨
- 이해하고 구현하기 쉬움

**단점**:
- **핫스팟**: 키가 순차적이면(자동 증가 ID, 타임스탬프) 모든 최근 쓰기가 동일 파티션에 집중
- 경계를 설정하려면 키 분포를 알아야 함

```python
from dataclasses import dataclass, field
from typing import Any, Optional
import bisect


@dataclass
class KeyRangePartition:
    """A partition covering a contiguous key range."""
    partition_id: int
    start_key: str  # inclusive
    end_key: str    # exclusive
    data: dict[str, Any] = field(default_factory=dict)

    def contains(self, key: str) -> bool:
        return self.start_key <= key < self.end_key

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def range_scan(self, start: str, end: str) -> list[tuple[str, Any]]:
        """Efficient range scan within this partition."""
        return [
            (k, v) for k, v in sorted(self.data.items())
            if start <= k < end
        ]

    @property
    def size(self) -> int:
        return len(self.data)


class KeyRangePartitioner:
    """
    Key-range partitioning with configurable boundaries.
    Supports efficient range queries but is prone to hot spots.
    """

    def __init__(self, boundaries: list[str]):
        """
        Create partitions from boundary list.
        boundaries = ["", "d", "h", "p", ""] creates 4 partitions:
          P0: [""..."d"), P1: ["d"..."h"), P2: ["h"..."p"), P3: ["p"..."")
        """
        self.partitions: list[KeyRangePartition] = []
        for i in range(len(boundaries) - 1):
            self.partitions.append(KeyRangePartition(
                partition_id=i,
                start_key=boundaries[i],
                end_key=boundaries[i + 1] if boundaries[i + 1] else chr(0x10FFFF),
            ))

    def get_partition(self, key: str) -> KeyRangePartition:
        """Route a key to its partition."""
        for partition in self.partitions:
            if partition.contains(key):
                return partition
        raise ValueError(f"No partition found for key: {key}")

    def put(self, key: str, value: Any) -> int:
        """Write a key-value pair. Returns the partition ID."""
        partition = self.get_partition(key)
        partition.put(key, value)
        return partition.partition_id

    def get(self, key: str) -> Optional[Any]:
        """Read a single key."""
        partition = self.get_partition(key)
        return partition.get(key)

    def range_query(self, start: str, end: str) -> list[tuple[str, Any]]:
        """
        Range query across potentially multiple partitions.
        Key-range partitioning makes this efficient:
        only touch partitions that overlap the range.
        """
        results = []
        for partition in self.partitions:
            # Check if partition overlaps with query range
            if partition.start_key < end and partition.end_key > start:
                results.extend(partition.range_scan(start, end))
        return sorted(results)

    def print_distribution(self) -> None:
        """Print data distribution across partitions."""
        total = sum(p.size for p in self.partitions)
        for p in self.partitions:
            pct = (p.size / total * 100) if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  P{p.partition_id} [{p.start_key!r:>4}..{p.end_key!r:<4}]: "
                  f"{p.size:>6} keys ({pct:5.1f}%) {bar}")


def demo_key_range():
    """Demonstrate key-range partitioning with hot spot problem."""
    import random
    import string

    # Create 4 partitions based on first letter
    partitioner = KeyRangePartitioner(["", "d", "h", "p", ""])

    # Uniform-ish distribution: random keys
    for _ in range(10000):
        key = "".join(random.choices(string.ascii_lowercase, k=8))
        partitioner.put(key, "value")

    print("Uniform key distribution:")
    partitioner.print_distribution()

    # Skewed distribution: keys starting with 'a' (hot spot)
    skewed = KeyRangePartitioner(["", "d", "h", "p", ""])
    for i in range(10000):
        # 80% of keys start with 'a' or 'b'
        if random.random() < 0.8:
            key = random.choice("ab") + f"{i:07d}"
        else:
            key = random.choice(string.ascii_lowercase) + f"{i:07d}"
        skewed.put(key, "value")

    print("\nSkewed key distribution (hot spot on P0):")
    skewed.print_distribution()
```

### 2.2 Hash 파티셔닝

키에 해시 함수를 적용하고 해시 값을 사용하여 파티션을 결정한다. 이는 키 패턴에 관계없이 키를 균일하게 분산한다.

```
hash("user:1") = 0x3A7F  →  partition = 0x3A7F % 4 = 3
hash("user:2") = 0x9C12  →  partition = 0x9C12 % 4 = 2
hash("user:3") = 0x1E45  →  partition = 0x1E45 % 4 = 1
```

**장점**:
- 균일 분산으로 키 패턴에 의한 핫스팟 제거
- 미리 키 분포를 알 필요 없음

**단점**:
- **범위 쿼리 불가**: 키 공간에서 인접한 키들이 파티션에 흩어짐
- 노드 추가/제거 시 대부분의 키 재해싱 필요 (consistent hashing 없이)

```python
import hashlib
from typing import Any, Optional


class HashPartitioner:
    """
    Hash-based partitioning.
    Uniform distribution but loses range query efficiency.
    """

    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
        self.partitions: list[dict[str, Any]] = [
            {} for _ in range(num_partitions)
        ]

    def _hash(self, key: str) -> int:
        """Deterministic hash function."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _get_partition_id(self, key: str) -> int:
        """Map key to partition via hash modulo."""
        return self._hash(key) % self.num_partitions

    def put(self, key: str, value: Any) -> int:
        """Write key-value pair. Returns partition ID."""
        pid = self._get_partition_id(key)
        self.partitions[pid][key] = value
        return pid

    def get(self, key: str) -> Optional[Any]:
        """Read by key."""
        pid = self._get_partition_id(key)
        return self.partitions[pid].get(key)

    def range_query(self, start: str, end: str) -> list[tuple[str, Any]]:
        """
        Range query: MUST scan ALL partitions (scatter-gather).
        This is the key disadvantage of hash partitioning.
        """
        results = []
        for partition in self.partitions:
            for key, value in partition.items():
                if start <= key < end:
                    results.append((key, value))
        return sorted(results)

    def print_distribution(self) -> None:
        total = sum(len(p) for p in self.partitions)
        for i, p in enumerate(self.partitions):
            pct = (len(p) / total * 100) if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  P{i}: {len(p):>6} keys ({pct:5.1f}%) {bar}")

    def rebalance_cost(self, new_num_partitions: int) -> float:
        """
        Calculate the fraction of keys that would move
        if we changed the number of partitions.
        (With simple modulo hashing, almost all keys move!)
        """
        moved = 0
        total = 0
        for pid, partition in enumerate(self.partitions):
            for key in partition:
                total += 1
                new_pid = self._hash(key) % new_num_partitions
                if new_pid != pid:
                    moved += 1
        return moved / max(total, 1)
```

### 2.3 복합(Compound) 파티셔닝

하나의 컬럼에는 hash 파티셔닝을, 다른 컬럼에는 key-range 파티셔닝을 사용한다. Cassandra와 DynamoDB에서 사용된다.

```
Compound key: (user_id, timestamp)

Partition by: hash(user_id)          → uniform distribution across partitions
Sort within partition by: timestamp  → efficient range queries on time

Query: "All posts by user X in March 2024"
  → hash(user_X) routes to single partition
  → range scan on timestamp within that partition
```

```python
@dataclass
class CompoundKey:
    """Compound key with partition and clustering components."""
    partition_key: str   # hashed for partition routing
    clustering_key: str  # sorted within partition for range queries

    def __lt__(self, other: 'CompoundKey') -> bool:
        return self.clustering_key < other.clustering_key


class CompoundPartitioner:
    """
    Compound partitioning (Cassandra-style).
    Hash on partition_key for distribution.
    Sort by clustering_key within each partition for range queries.
    """

    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
        # Each partition stores: {partition_key: sorted_list_of_(clustering_key, value)}
        self.partitions: list[dict[str, list[tuple[str, Any]]]] = [
            {} for _ in range(num_partitions)
        ]

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _get_partition_id(self, partition_key: str) -> int:
        return self._hash(partition_key) % self.num_partitions

    def put(self, partition_key: str, clustering_key: str, value: Any) -> int:
        """Insert with compound key."""
        pid = self._get_partition_id(partition_key)
        partition = self.partitions[pid]
        if partition_key not in partition:
            partition[partition_key] = []

        # Insert in sorted order by clustering key
        entries = partition[partition_key]
        insert_pos = bisect.bisect_left(
            [ck for ck, _ in entries], clustering_key
        )
        entries.insert(insert_pos, (clustering_key, value))
        return pid

    def get(self, partition_key: str, clustering_key: str) -> Optional[Any]:
        """Point lookup by full compound key."""
        pid = self._get_partition_id(partition_key)
        entries = self.partitions[pid].get(partition_key, [])
        for ck, val in entries:
            if ck == clustering_key:
                return val
        return None

    def range_within_partition(self, partition_key: str,
                                start_ck: str, end_ck: str) -> list[tuple[str, Any]]:
        """
        Efficient range query on clustering key within a single partition.
        Only touches ONE partition (the one owning partition_key).
        """
        pid = self._get_partition_id(partition_key)
        entries = self.partitions[pid].get(partition_key, [])
        return [(ck, val) for ck, val in entries if start_ck <= ck < end_ck]


def demo_compound():
    """Demonstrate compound partitioning for time-series data."""
    partitioner = CompoundPartitioner(num_partitions=4)

    # Insert user activity log: partition by user, sort by timestamp
    events = [
        ("user:alice", "2024-03-01T10:00", "login"),
        ("user:alice", "2024-03-01T10:05", "view_page"),
        ("user:alice", "2024-03-01T10:15", "purchase"),
        ("user:alice", "2024-03-02T09:00", "login"),
        ("user:bob",   "2024-03-01T11:00", "login"),
        ("user:bob",   "2024-03-01T11:30", "view_page"),
    ]

    for user, ts, action in events:
        partitioner.put(user, ts, action)

    # Efficient range query: Alice's activity on March 1st
    results = partitioner.range_within_partition(
        "user:alice", "2024-03-01", "2024-03-02"
    )
    print("Alice's March 1st activity:")
    for ts, action in results:
        print(f"  {ts}: {action}")
```

### 2.4 전략 비교

| 전략 | 분산 | 범위 쿼리 | 핫스팟 | 리밸런싱 |
|---|---|---|---|---|
| **Key-range** | 잠재적 편향 | 효율적 (단일 파티션) | 높은 위험 (순차 키) | 경계 이동 |
| **Hash** | 균일 | 비효율적 (scatter-gather) | 낮은 위험 | 높은 비용 (modulo 변경) |
| **Compound** | 균일 (파티션 키) | 파티션 내에서 효율적 | 파티션 키에서 낮음 | Hash와 동일 |

---

## 3. Consistent Hashing

단순 hash 파티셔닝(`hash(key) % N`)의 근본적 문제는 N을 변경하면 **대부분의 키가 재매핑**된다는 것이다. Consistent hashing은 데이터 이동을 최소화한다.

### 3.1 해시 링

키와 노드 모두 순환 링(0에서 2^m - 1)에 해시된다. 각 키는 키의 위치에서 시계 방향으로 걸을 때 처음 만나는 노드에 할당된다.

```
Hash Ring (0 to 360 degrees for illustration):

          0°
          │
     330° │  30°
          │
 300° ────●────  60°     Node A at 45°
          │               Node B at 150°
 270° ────┼──── 90°      Node C at 270°
          │
 240° ────●──── 120°
          │
     210° │  150°
          │
         180°

Key "user:1" hashes to 80°  → walks clockwise → reaches Node B at 150°
Key "user:2" hashes to 200° → walks clockwise → reaches Node C at 270°
Key "user:3" hashes to 300° → walks clockwise → reaches Node A at 45°
```

노드가 추가되거나 제거되면, 링에서 새/제거된 노드와 그 전임자 사이의 키만 영향을 받는다. 평균적으로 **K/N**개의 키만 이동한다(K는 전체 키, N은 노드 수).

### 3.2 구현: 기본 Consistent Hashing

```python
import hashlib
import bisect
from collections import defaultdict
from typing import Any, Optional


class ConsistentHashRing:
    """
    Consistent hashing ring.
    Maps keys to nodes with minimal disruption when nodes change.
    """

    def __init__(self, hash_bits: int = 160):
        self.hash_bits = hash_bits
        self.ring_size = 2 ** hash_bits
        self._ring: list[int] = []           # sorted positions on the ring
        self._ring_to_node: dict[int, str] = {}  # position -> node_id
        self._nodes: set[str] = set()

    def _hash(self, key: str) -> int:
        """Hash a key to a position on the ring."""
        h = hashlib.sha1(key.encode()).hexdigest()
        return int(h, 16) % self.ring_size

    def add_node(self, node_id: str) -> None:
        """Add a node to the ring."""
        if node_id in self._nodes:
            return
        self._nodes.add(node_id)
        position = self._hash(node_id)
        bisect.insort(self._ring, position)
        self._ring_to_node[position] = node_id

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring."""
        if node_id not in self._nodes:
            return
        self._nodes.discard(node_id)
        position = self._hash(node_id)
        self._ring.remove(position)
        del self._ring_to_node[position]

    def get_node(self, key: str) -> Optional[str]:
        """Find the node responsible for a key."""
        if not self._ring:
            return None
        position = self._hash(key)
        # Find the first ring position >= key position (clockwise walk)
        idx = bisect.bisect_left(self._ring, position)
        if idx == len(self._ring):
            idx = 0  # wrap around
        return self._ring_to_node[self._ring[idx]]

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """Count how many keys are assigned to each node."""
        distribution: dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                distribution[node] += 1
        return dict(distribution)
```

### 3.3 가상 노드 (VNode)

물리 노드가 적으면 링이 제대로 균형 잡히지 않는다 — 일부 노드가 다른 노드보다 훨씬 많은 키를 받는다. **가상 노드**는 각 물리 노드를 링의 여러 위치에 배치하여 이를 해결한다.

```
Without virtual nodes (3 nodes):
  Node A: 1 position → owns ~33% of ring (in theory)
  Actual: could be 10%-60% due to hash distribution

With 100 virtual nodes each (3 nodes):
  Node A: 100 positions → owns ~33% of ring (much closer to ideal)
  Standard deviation decreases as √(num_vnodes)
```

```python
class ConsistentHashRingWithVNodes:
    """
    Consistent hashing ring with virtual nodes for better balance.
    Each physical node is mapped to multiple positions on the ring.
    """

    def __init__(self, num_virtual_nodes: int = 150):
        self.num_virtual_nodes = num_virtual_nodes
        self._ring: list[int] = []
        self._ring_to_node: dict[int, str] = {}
        self._nodes: set[str] = set()

    def _hash(self, key: str) -> int:
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str) -> set[str]:
        """
        Add a physical node with virtual nodes.
        Returns the set of keys that would need to be moved to this node.
        """
        if node_id in self._nodes:
            return set()
        self._nodes.add(node_id)

        for i in range(self.num_virtual_nodes):
            vnode_key = f"{node_id}:vnode:{i}"
            position = self._hash(vnode_key)
            bisect.insort(self._ring, position)
            self._ring_to_node[position] = node_id

        return set()  # in practice, would track affected key ranges

    def remove_node(self, node_id: str) -> None:
        """Remove a physical node and all its virtual nodes."""
        if node_id not in self._nodes:
            return
        self._nodes.discard(node_id)

        positions_to_remove = [
            pos for pos, nid in self._ring_to_node.items()
            if nid == node_id
        ]
        for pos in positions_to_remove:
            self._ring.remove(pos)
            del self._ring_to_node[pos]

    def get_node(self, key: str) -> Optional[str]:
        """Find the responsible physical node for a key."""
        if not self._ring:
            return None
        position = self._hash(key)
        idx = bisect.bisect_left(self._ring, position)
        if idx == len(self._ring):
            idx = 0
        return self._ring_to_node[self._ring[idx]]

    def get_n_nodes(self, key: str, n: int) -> list[str]:
        """
        Get N distinct physical nodes for a key (for replication).
        Walk clockwise, skipping virtual nodes of already-selected physical nodes.
        """
        if not self._ring or n > len(self._nodes):
            return list(self._nodes)

        position = self._hash(key)
        idx = bisect.bisect_left(self._ring, position)
        selected: list[str] = []
        seen: set[str] = set()

        for offset in range(len(self._ring)):
            ring_idx = (idx + offset) % len(self._ring)
            node = self._ring_to_node[self._ring[ring_idx]]
            if node not in seen:
                selected.append(node)
                seen.add(node)
                if len(selected) == n:
                    break

        return selected

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """Measure key distribution across physical nodes."""
        dist: dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                dist[node] += 1
        return dict(dist)

    def measure_balance(self, keys: list[str]) -> dict[str, float]:
        """
        Measure distribution balance.
        Returns statistics: min%, max%, stddev%, coefficient of variation.
        """
        import statistics

        dist = self.get_distribution(keys)
        total = sum(dist.values())
        if total == 0 or len(dist) == 0:
            return {"min_pct": 0, "max_pct": 0, "stddev_pct": 0, "cv": 0}

        percentages = [count / total * 100 for count in dist.values()]
        ideal = 100 / len(dist)

        return {
            "ideal_pct": round(ideal, 2),
            "min_pct": round(min(percentages), 2),
            "max_pct": round(max(percentages), 2),
            "stddev_pct": round(statistics.stdev(percentages), 2) if len(percentages) > 1 else 0,
            "cv": round(statistics.stdev(percentages) / statistics.mean(percentages), 4) if len(percentages) > 1 else 0,
        }


def demo_virtual_nodes():
    """Compare balance with different numbers of virtual nodes."""
    import random
    import string

    # Generate 100,000 random keys
    keys = [
        "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
        for _ in range(100_000)
    ]

    configs = [1, 10, 50, 150, 500]
    print(f"{'VNodes':>8} {'Min%':>8} {'Max%':>8} {'StdDev%':>9} {'CV':>8}")
    print("-" * 43)

    for num_vnodes in configs:
        ring = ConsistentHashRingWithVNodes(num_virtual_nodes=num_vnodes)
        for i in range(5):
            ring.add_node(f"node-{i}")

        stats = ring.measure_balance(keys)
        print(f"{num_vnodes:>8} {stats['min_pct']:>7.1f}% {stats['max_pct']:>7.1f}% "
              f"{stats['stddev_pct']:>8.2f}% {stats['cv']:>7.4f}")
```

### 3.4 부하 제한 Consistent Hashing

Google(2017)은 단일 노드가 평균 부하의 (1 + ε) 배 이상 과부하되는 것을 방지하는 부하 제한 consistent hashing을 도입했다.

**알고리즘**: 키의 주요 노드가 "가득 찬" 경우(부하 한도 초과), 키는 링에서 가용 용량이 있는 다음 노드로 리디렉션된다.

```python
class BoundedLoadHashRing(ConsistentHashRingWithVNodes):
    """
    Bounded-load consistent hashing (Google, 2017).
    Ensures no node exceeds (1 + epsilon) * average_load.
    """

    def __init__(self, num_virtual_nodes: int = 150, epsilon: float = 0.25):
        super().__init__(num_virtual_nodes)
        self.epsilon = epsilon
        self._load: dict[str, int] = defaultdict(int)
        self._total_keys = 0

    def _load_cap(self) -> float:
        """Maximum load per node = ceil((1 + epsilon) * average_load)."""
        if not self._nodes:
            return float("inf")
        avg = self._total_keys / len(self._nodes)
        return max(1, int((1 + self.epsilon) * avg) + 1)

    def assign_key(self, key: str) -> Optional[str]:
        """
        Assign a key to a node, respecting load bounds.
        If the primary node is overloaded, walk clockwise to find
        the next node with available capacity.
        """
        if not self._ring:
            return None

        position = self._hash(key)
        idx = bisect.bisect_left(self._ring, position)
        cap = self._load_cap()

        for offset in range(len(self._ring)):
            ring_idx = (idx + offset) % len(self._ring)
            node = self._ring_to_node[self._ring[ring_idx]]
            if self._load[node] < cap:
                self._load[node] += 1
                self._total_keys += 1
                return node

        # All nodes at capacity (shouldn't happen if cap is calculated correctly)
        return None

    def release_key(self, node_id: str) -> None:
        """Release a key from a node's load count."""
        if self._load[node_id] > 0:
            self._load[node_id] -= 1
            self._total_keys -= 1
```

### 3.5 Jump Consistent Hashing

Jump consistent hashing(Google, 2014)은 O(1) 메모리와 O(ln N) 시간을 사용하는 더 단순한 알고리즘이다. 노드가 0부터 N-1까지 번호가 매겨진 경우(이름 없는 노드)에 이상적이다.

```python
def jump_consistent_hash(key: int, num_buckets: int) -> int:
    """
    Jump consistent hashing (Lamping & Veach, 2014).

    O(ln n) time, O(1) space.
    Maps key to bucket in [0, num_buckets).
    When num_buckets changes by 1, only 1/num_buckets fraction of keys move.

    Based on the mathematical property:
    ch(key, n+1) = ch(key, n) with probability n/(n+1)
                 = n           with probability 1/(n+1)
    """
    b: int = -1  # "bucket" — tracks the jump destination
    j: int = 0   # "jump" — tracks current candidate

    # Use key as seed for a linear congruential generator
    seed = key
    while j < num_buckets:
        b = j
        seed = ((seed * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
        j = int((b + 1) * (1 << 31) / ((seed >> 33) + 1))

    return b


def demo_jump_hash():
    """Demonstrate jump consistent hash properties."""
    # Distribution test
    num_buckets = 5
    bucket_counts = defaultdict(int)
    for key in range(100_000):
        bucket = jump_consistent_hash(key, num_buckets)
        bucket_counts[bucket] += 1

    print("Jump hash distribution (5 buckets, 100K keys):")
    for bucket in sorted(bucket_counts):
        count = bucket_counts[bucket]
        print(f"  Bucket {bucket}: {count:>6} ({count/1000:.1f}%)")

    # Movement test: how many keys move when adding a bucket?
    moved = 0
    total = 100_000
    for key in range(total):
        old_bucket = jump_consistent_hash(key, num_buckets)
        new_bucket = jump_consistent_hash(key, num_buckets + 1)
        if old_bucket != new_bucket:
            moved += 1

    print(f"\nKeys moved when adding 1 bucket: {moved} ({moved/total*100:.1f}%)")
    print(f"Theoretical minimum: {total//(num_buckets+1)} ({100/(num_buckets+1):.1f}%)")
```

---

## 4. 보조 인덱스 파티셔닝

지금까지 기본 키에 의한 파티셔닝을 논의했다. 하지만 많은 쿼리는 보조 인덱스를 사용한다(예: "city=Seattle인 모든 사용자 찾기"). 보조 인덱스를 어떻게 파티셔닝하는가?

### 4.1 문서 파티셔닝(로컬) 인덱스

각 파티션이 해당 파티션의 데이터만 포함하는 자체 보조 인덱스를 유지한다.

```
Partition 1                  Partition 2
┌──────────────────────┐    ┌──────────────────────┐
│ Data:                │    │ Data:                │
│   user:1 {city:SEA}  │    │   user:3 {city:SEA}  │
│   user:2 {city:NYC}  │    │   user:4 {city:LAX}  │
│                      │    │                      │
│ Local index:         │    │ Local index:         │
│   city:SEA → [1]     │    │   city:SEA → [3]     │
│   city:NYC → [2]     │    │   city:LAX → [4]     │
└──────────────────────┘    └──────────────────────┘

Query: WHERE city='SEA'
  → Must scatter to ALL partitions (scatter-gather)
  → Partition 1 returns [user:1]
  → Partition 2 returns [user:3]
  → Merge results: [user:1, user:3]
```

```python
class LocalIndexPartition:
    """
    Document-partitioned (local) secondary index.
    Each partition maintains its own index.
    """

    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.data: dict[str, dict] = {}
        # Secondary indexes: {field_name: {field_value: [primary_keys]}}
        self.indexes: dict[str, dict[Any, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def put(self, key: str, doc: dict, indexed_fields: list[str]) -> None:
        """Insert a document and update local indexes."""
        self.data[key] = doc
        for field in indexed_fields:
            if field in doc:
                self.indexes[field][doc[field]].append(key)

    def query_index(self, field: str, value: Any) -> list[str]:
        """Query the local index. Returns keys from THIS partition only."""
        return self.indexes.get(field, {}).get(value, [])


class ScatterGatherQueryEngine:
    """
    Query engine for document-partitioned indexes.
    Must scatter queries to ALL partitions and gather results.
    """

    def __init__(self, partitions: list[LocalIndexPartition]):
        self.partitions = partitions

    def query(self, field: str, value: Any) -> list[tuple[int, str]]:
        """
        Scatter-gather query across all partitions.
        Returns list of (partition_id, key) tuples.
        Cost: O(num_partitions) network round trips.
        """
        results = []
        for partition in self.partitions:
            local_results = partition.query_index(field, value)
            for key in local_results:
                results.append((partition.partition_id, key))
        return results

    @property
    def query_amplification(self) -> int:
        """Number of partitions that must be queried for a single index lookup."""
        return len(self.partitions)
```

### 4.2 용어 파티셔닝(글로벌) 인덱스

글로벌 인덱스는 모든 파티션의 모든 데이터를 포함하지만, 인덱스 자체가 용어(인덱스 값)별로 파티셔닝된다.

```
Partition 1 (data)           Partition 2 (data)
┌──────────────────────┐    ┌──────────────────────┐
│ user:1 {city:SEA}    │    │ user:3 {city:SEA}    │
│ user:2 {city:NYC}    │    │ user:4 {city:LAX}    │
└──────────────────────┘    └──────────────────────┘

Index Partition A (cities A-M)    Index Partition B (cities N-Z)
┌──────────────────────────┐    ┌──────────────────────────┐
│ city:LAX → [P2:user:4]  │    │ city:NYC → [P1:user:2]   │
│                          │    │ city:SEA → [P1:user:1,   │
│                          │    │            P2:user:3]     │
└──────────────────────────┘    └──────────────────────────┘

Query: WHERE city='SEA'
  → Route to Index Partition B (because 'S' is in N-Z range)
  → Single-partition read: [user:1, user:3]
  → No scatter-gather!
```

```python
class GlobalIndexPartition:
    """
    Term-partitioned (global) secondary index partition.
    Each index partition covers a range of index terms.
    """

    def __init__(self, partition_id: int, term_range: tuple[str, str]):
        self.partition_id = partition_id
        self.term_start = term_range[0]  # inclusive
        self.term_end = term_range[1]    # exclusive
        # Index: {field: {value: [(data_partition_id, primary_key)]}}
        self.index: dict[str, dict[Any, list[tuple[int, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def covers_term(self, term_value: str) -> bool:
        """Check if this index partition covers a given term value."""
        return self.term_start <= str(term_value) < self.term_end

    def add_entry(self, field: str, value: Any,
                  data_partition_id: int, primary_key: str) -> None:
        """Add an index entry."""
        self.index[field][value].append((data_partition_id, primary_key))

    def query(self, field: str, value: Any) -> list[tuple[int, str]]:
        """Query this index partition. Returns (data_partition, key) pairs."""
        return self.index.get(field, {}).get(value, [])
```

### 4.3 로컬 vs 글로벌 인덱스 비교

| 측면 | 로컬 (문서 파티셔닝) | 글로벌 (용어 파티셔닝) |
|---|---|---|
| **읽기 (인덱스 쿼리)** | Scatter-gather: O(파티션 수) | 단일 파티션: O(1) |
| **쓰기** | 단일 파티션 (로컬 인덱스 업데이트) | 교차 파티션 (원격 인덱스 파티션 업데이트) |
| **인덱스 최신성** | 항상 최신 | 지연될 수 있음 (비동기 업데이트) |
| **쓰기 증폭** | 낮음 | 높음 (글로벌 인덱스 업데이트 필요) |
| **일관성** | 강함 (데이터에 로컬) | 최종적 일관성 (일반적) |
| **사용 시스템** | MongoDB, Elasticsearch | DynamoDB (GSI), Riak |

**경험 법칙**: 워크로드가 보조 인덱스에서 읽기 중심이면 글로벌을, 쓰기 중심이면 로컬을 선호한다.

---

## 5. 리밸런싱 전략

클러스터가 성장하거나 축소될 때, 데이터를 노드 간에 재분배해야 한다. 목표는 최소한의 중단으로 리밸런싱하는 것이다.

### 5.1 고정 파티션 수

시작 시 노드보다 훨씬 많은 파티션을 생성한다. 노드가 추가되면 기존 파티션이 이동(분할이 아님)된다.

```
Initial: 64 partitions, 4 nodes
  Node A: P0-P15  (16 partitions)
  Node B: P16-P31 (16 partitions)
  Node C: P32-P47 (16 partitions)
  Node D: P48-P63 (16 partitions)

Add Node E:
  Node A: P0-P12  (13 partitions, gave 3 to E)
  Node B: P16-P28 (13 partitions, gave 3 to E)
  Node C: P32-P44 (13 partitions, gave 3 to E)
  Node D: P48-P60 (13 partitions, gave 3 to E)
  Node E: P13-P15, P29-P31, P45-P47, P61-P63 (12 partitions)
```

**사용 시스템**: Riak, Elasticsearch, Couchbase, Voldemort

**트레이드오프**: 파티션 수를 미리 선택해야 한다. 너무 적으면 → 균형이 어려움. 너무 많으면 → 파티션당 오버헤드.

```python
class FixedPartitionRebalancer:
    """
    Fixed-partition rebalancing strategy.
    Partitions are pre-created; nodes own subsets of partitions.
    """

    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
        self.partition_to_node: dict[int, str] = {}
        self.nodes: list[str] = []

    def add_node(self, node_id: str) -> dict[int, tuple[str, str]]:
        """
        Add a node and rebalance.
        Returns: {partition_id: (old_node, new_node)} for moved partitions.
        """
        self.nodes.append(node_id)
        return self._rebalance()

    def remove_node(self, node_id: str) -> dict[int, tuple[str, str]]:
        """Remove a node and rebalance."""
        self.nodes.remove(node_id)
        return self._rebalance()

    def _rebalance(self) -> dict[int, tuple[str, str]]:
        """Redistribute partitions evenly across nodes."""
        if not self.nodes:
            return {}

        moves: dict[int, tuple[str, str]] = {}
        target_per_node = self.num_partitions // len(self.nodes)
        remainder = self.num_partitions % len(self.nodes)

        # Calculate ideal assignment
        ideal: dict[int, str] = {}
        partition_idx = 0
        for i, node in enumerate(self.nodes):
            count = target_per_node + (1 if i < remainder else 0)
            for _ in range(count):
                if partition_idx < self.num_partitions:
                    ideal[partition_idx] = node
                    partition_idx += 1

        # Determine moves
        for pid, new_node in ideal.items():
            old_node = self.partition_to_node.get(pid)
            if old_node and old_node != new_node:
                moves[pid] = (old_node, new_node)

        self.partition_to_node = ideal
        return moves

    def print_assignment(self) -> None:
        """Print current partition assignment."""
        node_partitions: dict[str, list[int]] = defaultdict(list)
        for pid, node in sorted(self.partition_to_node.items()):
            node_partitions[node].append(pid)
        for node in sorted(node_partitions):
            pids = node_partitions[node]
            print(f"  {node}: {len(pids)} partitions "
                  f"[{pids[0]}..{pids[-1]}]")
```

### 5.2 동적 파티셔닝

하나의 파티션으로 시작한다. 파티션이 너무 커지면 반으로 분할한다. 축소되면 인접 파티션을 병합한다.

**사용 시스템**: HBase, MongoDB, RethinkDB

```python
class DynamicPartition:
    """A dynamically splitting partition."""

    def __init__(self, partition_id: str, start_key: str, end_key: str,
                 max_size: int = 10000):
        self.partition_id = partition_id
        self.start_key = start_key
        self.end_key = end_key
        self.max_size = max_size
        self.data: dict[str, Any] = {}

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def should_split(self) -> bool:
        return self.size >= self.max_size

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value

    def split(self) -> tuple['DynamicPartition', 'DynamicPartition']:
        """Split this partition into two halves by key range."""
        sorted_keys = sorted(self.data.keys())
        mid_idx = len(sorted_keys) // 2
        mid_key = sorted_keys[mid_idx]

        left = DynamicPartition(
            f"{self.partition_id}_L", self.start_key, mid_key, self.max_size
        )
        right = DynamicPartition(
            f"{self.partition_id}_R", mid_key, self.end_key, self.max_size
        )

        for key, value in self.data.items():
            if key < mid_key:
                left.data[key] = value
            else:
                right.data[key] = value

        return left, right
```

### 5.3 노드에 비례 (Cassandra)

각 노드가 고정 수의 파티션을 소유한다. 새 노드가 합류하면 기존 파티션을 무작위로 분할한다.

| 전략 | 사전 구성? | 데이터 증가 적응? | 노드 변경 적응? | 데이터 이동 |
|---|---|---|---|---|
| **고정** | 예 (N을 미리 설정) | 아니오 (고정 파티션 수) | 예 (파티션 이동) | 최소 |
| **동적** | 아니오 (1로 시작) | 예 (분할/병합) | 부분적 | 보통 |
| **비례** | 아니오 | 예 | 예 | 보통 |

---

## 6. 교차 파티션 연산

### 6.1 Scatter-Gather 쿼리

쿼리를 단일 파티션으로 라우팅할 수 없을 때, 코디네이터는 쿼리를 모든(또는 많은) 파티션에 분산하고 결과를 수집해야 한다.

```python
import time
import concurrent.futures
from typing import Any, Callable


class ScatterGatherCoordinator:
    """
    Coordinator for scatter-gather queries across partitions.
    Demonstrates parallel execution and result merging.
    """

    def __init__(self, num_partitions: int, max_workers: int = 10):
        self.num_partitions = num_partitions
        self.max_workers = max_workers

    def scatter_gather(self, query_func: Callable[[int], list[Any]],
                       merge_func: Callable[[list[list[Any]]], list[Any]],
                       target_partitions: list[int] | None = None) -> dict:
        """
        Execute a query across partitions in parallel.

        Args:
            query_func: Function(partition_id) -> list of results
            merge_func: Function(list of result lists) -> merged results
            target_partitions: Specific partitions to query (None = all)
        """
        targets = target_partitions or list(range(self.num_partitions))

        start_time = time.time()
        all_results = []

        # Execute queries in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(targets))
        ) as executor:
            futures = {
                executor.submit(query_func, pid): pid
                for pid in targets
            }
            for future in concurrent.futures.as_completed(futures):
                partition_id = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"  [SCATTER] Partition {partition_id} failed: {e}")

        # Merge results
        merged = merge_func(all_results)
        elapsed = time.time() - start_time

        return {
            "results": merged,
            "partitions_queried": len(targets),
            "elapsed_ms": round(elapsed * 1000, 2),
        }
```

### 6.2 파티션 인식 라우팅

스마트 클라이언트(또는 라우팅 티어)가 파티션 맵을 유지하고 쿼리를 올바른 파티션으로 직접 라우팅하여, 가능한 경우 scatter-gather를 피한다.

```
Three approaches to request routing:

1. Client-side routing:
   Client ──► Partition (direct)
   Pro: lowest latency
   Con: client must maintain partition map

2. Routing tier:
   Client ──► Router ──► Partition
   Pro: clients are simple
   Con: extra hop

3. Any-node routing:
   Client ──► Any Node ──► Correct Partition
   Pro: simple client
   Con: extra hop, all nodes must maintain routing info
```

```python
class PartitionRouter:
    """
    Partition-aware request router.
    Maintains a partition map and routes requests to the correct node.
    """

    def __init__(self):
        # partition_id -> node_address
        self.partition_map: dict[int, str] = {}
        # Partitioning function
        self._hash = lambda key: int(
            hashlib.md5(key.encode()).hexdigest(), 16
        )
        self._num_partitions = 0

    def update_map(self, new_map: dict[int, str]) -> None:
        """Update the partition map (received from coordinator/ZooKeeper)."""
        self.partition_map = new_map
        self._num_partitions = len(new_map)

    def route(self, key: str) -> tuple[int, str]:
        """Route a key to (partition_id, node_address)."""
        if self._num_partitions == 0:
            raise RuntimeError("Partition map not initialized")
        pid = self._hash(key) % self._num_partitions
        node = self.partition_map.get(pid)
        if node is None:
            raise RuntimeError(f"No node for partition {pid}")
        return pid, node

    def route_range(self, start_key: str, end_key: str) -> list[tuple[int, str]]:
        """
        Determine which partitions a range query must touch.
        With hash partitioning: ALL partitions (worst case).
        With range partitioning: only overlapping partitions.
        """
        # Hash partitioning: we can't determine range overlap from hashes
        # Must touch all partitions
        return [(pid, node) for pid, node in self.partition_map.items()]
```

---

## 7. 핫스팟 완화

Hash 파티셔닝을 사용해도, 특정 키가 불균형적인 트래픽을 받으면 핫스팟이 발생할 수 있다(예: 유명인의 소셜 미디어 게시물).

### 7.1 키 솔팅(Key Salting)

핫 키에 랜덤 접미사를 추가하여 여러 파티션에 분산시킨다.

```python
import random


class SaltedKeyPartitioner:
    """
    Mitigate hot spots by salting keys with random suffixes.
    Distributes a single hot key across multiple partitions.
    """

    def __init__(self, num_partitions: int, num_salts: int = 10):
        self.num_partitions = num_partitions
        self.num_salts = num_salts
        self.hot_keys: set[str] = set()
        self.partitions: list[dict[str, Any]] = [
            {} for _ in range(num_partitions)
        ]

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def mark_hot(self, key: str) -> None:
        """Mark a key as hot (requiring salting)."""
        self.hot_keys.add(key)

    def write(self, key: str, value: Any) -> int:
        """
        Write with salting for hot keys.
        Hot keys are spread across num_salts partitions.
        """
        if key in self.hot_keys:
            salt = random.randint(0, self.num_salts - 1)
            salted_key = f"{key}:salt:{salt}"
        else:
            salted_key = key

        pid = self._hash(salted_key) % self.num_partitions
        self.partitions[pid][salted_key] = value
        return pid

    def read(self, key: str) -> list[Any]:
        """
        Read a potentially salted key.
        Must check all salt variants (scatter to num_salts partitions).
        """
        if key in self.hot_keys:
            results = []
            for salt in range(self.num_salts):
                salted_key = f"{key}:salt:{salt}"
                pid = self._hash(salted_key) % self.num_partitions
                val = self.partitions[pid].get(salted_key)
                if val is not None:
                    results.append(val)
            return results
        else:
            pid = self._hash(key) % self.num_partitions
            val = self.partitions[pid].get(key)
            return [val] if val is not None else []


def demo_hot_spot_salting():
    """Demonstrate hot spot mitigation via key salting."""
    partitioner = SaltedKeyPartitioner(num_partitions=8, num_salts=10)
    partitioner.mark_hot("celebrity:post:12345")

    # Write 1000 times to the hot key (simulating viral content)
    partition_counts = defaultdict(int)
    for i in range(1000):
        pid = partitioner.write("celebrity:post:12345", f"comment_{i}")
        partition_counts[pid] += 1

    print("Hot key distribution across partitions (with salting):")
    for pid in sorted(partition_counts):
        count = partition_counts[pid]
        bar = "#" * (count // 10)
        print(f"  P{pid}: {count:>5} writes {bar}")

    # Without salting: all 1000 would go to one partition
    unsalted = SaltedKeyPartitioner(num_partitions=8, num_salts=10)
    unsalted_counts = defaultdict(int)
    for i in range(1000):
        pid = unsalted.write("celebrity:post:12345", f"comment_{i}")
        unsalted_counts[pid] += 1

    print("\nWithout salting (same key, same partition):")
    for pid in sorted(unsalted_counts):
        count = unsalted_counts[pid]
        bar = "#" * (count // 10)
        print(f"  P{pid}: {count:>5} writes {bar}")
```

### 7.2 핫 파티션 분할

파티션이 (크기가 아닌) 트래픽으로 너무 뜨거워지면, 동적으로 분할한다.

### 7.3 핫 키 캐싱

라우팅 계층에서 핫 키를 캐싱하여 파티션을 치지 않고 읽기를 흡수한다.

```python
from collections import OrderedDict
from typing import Optional


class PartitionCache:
    """
    LRU cache at the routing layer for hot keys.
    Absorbs reads for frequently accessed keys.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 60.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get from cache. Returns None on miss or expiry."""
        if key in self._cache:
            value, insert_time = self._cache[key]
            if time.time() - insert_time < self.ttl_seconds:
                self._cache.move_to_end(key)
                self._hits += 1
                return value
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Insert into cache, evicting LRU if full."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time())
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
```

---

## 8. 종합 Consistent Hashing 구현

```python
"""
Complete consistent hashing implementation with:
- Virtual nodes
- Key distribution analysis
- Rebalancing cost measurement
- Node addition/removal simulation
"""

import hashlib
import bisect
import statistics
from collections import defaultdict
from typing import Optional


class ProductionConsistentHashRing:
    """
    Production-grade consistent hashing with virtual nodes,
    distribution analysis, and rebalancing cost tracking.
    """

    def __init__(self, num_virtual_nodes: int = 150,
                 replication_factor: int = 3):
        self.num_virtual_nodes = num_virtual_nodes
        self.replication_factor = replication_factor
        self._ring: list[int] = []
        self._ring_to_node: dict[int, str] = {}
        self._nodes: set[str] = set()
        self._key_assignments: dict[str, str] = {}  # key -> node

    def _hash(self, key: str) -> int:
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str) -> None:
        """Add a node with virtual nodes."""
        self._nodes.add(node_id)
        for i in range(self.num_virtual_nodes):
            vnode_key = f"{node_id}#vn{i}"
            pos = self._hash(vnode_key)
            bisect.insort(self._ring, pos)
            self._ring_to_node[pos] = node_id

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its virtual nodes."""
        self._nodes.discard(node_id)
        to_remove = [
            pos for pos, nid in self._ring_to_node.items()
            if nid == node_id
        ]
        for pos in to_remove:
            self._ring.remove(pos)
            del self._ring_to_node[pos]

    def get_node(self, key: str) -> Optional[str]:
        """Get the primary node for a key."""
        if not self._ring:
            return None
        pos = self._hash(key)
        idx = bisect.bisect_left(self._ring, pos)
        if idx == len(self._ring):
            idx = 0
        return self._ring_to_node[self._ring[idx]]

    def get_replica_nodes(self, key: str) -> list[str]:
        """Get N distinct nodes for replication (walk clockwise)."""
        if not self._ring:
            return []
        pos = self._hash(key)
        idx = bisect.bisect_left(self._ring, pos)
        selected = []
        seen = set()
        for offset in range(len(self._ring)):
            ring_idx = (idx + offset) % len(self._ring)
            node = self._ring_to_node[self._ring[ring_idx]]
            if node not in seen:
                selected.append(node)
                seen.add(node)
                if len(selected) == min(self.replication_factor, len(self._nodes)):
                    break
        return selected

    def measure_distribution(self, keys: list[str]) -> dict:
        """Comprehensive distribution analysis."""
        node_counts: dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                node_counts[node] += 1

        total = sum(node_counts.values())
        if not node_counts or total == 0:
            return {"error": "No data"}

        counts = list(node_counts.values())
        ideal = total / len(self._nodes)

        return {
            "total_keys": total,
            "num_nodes": len(self._nodes),
            "ideal_per_node": round(ideal, 1),
            "min_keys": min(counts),
            "max_keys": max(counts),
            "stddev": round(statistics.stdev(counts), 1) if len(counts) > 1 else 0,
            "max_deviation_pct": round(
                abs(max(counts) - ideal) / ideal * 100, 1
            ),
            "distribution": dict(sorted(node_counts.items())),
        }

    def measure_rebalance_cost(self, keys: list[str],
                                operation: str,
                                node_id: str) -> dict:
        """
        Measure how many keys would move if a node is added or removed.
        """
        # Record current assignments
        old_assignments = {key: self.get_node(key) for key in keys}

        # Perform the operation
        if operation == "add":
            self.add_node(node_id)
        elif operation == "remove":
            self.remove_node(node_id)

        # Record new assignments
        new_assignments = {key: self.get_node(key) for key in keys}

        # Undo the operation
        if operation == "add":
            self.remove_node(node_id)
        elif operation == "remove":
            self.add_node(node_id)

        # Count moves
        moved = sum(
            1 for key in keys
            if old_assignments[key] != new_assignments[key]
        )

        return {
            "total_keys": len(keys),
            "keys_moved": moved,
            "movement_pct": round(moved / len(keys) * 100, 2),
            "ideal_movement_pct": round(100 / (len(self._nodes) + 1), 2)
            if operation == "add"
            else round(100 / len(self._nodes), 2),
        }


def run_comprehensive_demo():
    """Full demonstration of consistent hashing with analysis."""
    import random
    import string

    print("=" * 70)
    print("CONSISTENT HASHING COMPREHENSIVE DEMO")
    print("=" * 70)

    # Generate test keys
    keys = [
        "".join(random.choices(string.ascii_lowercase + string.digits, k=16))
        for _ in range(100_000)
    ]

    # Test different vnode counts
    print("\n1. Impact of Virtual Node Count on Distribution Balance")
    print("-" * 70)
    print(f"{'VNodes':>8} {'Min':>8} {'Max':>8} {'StdDev':>8} {'MaxDev%':>9}")
    print("-" * 43)

    for vnodes in [1, 5, 20, 50, 100, 200, 500]:
        ring = ProductionConsistentHashRing(num_virtual_nodes=vnodes)
        for i in range(5):
            ring.add_node(f"node-{i}")
        stats = ring.measure_distribution(keys)
        print(f"{vnodes:>8} {stats['min_keys']:>8} {stats['max_keys']:>8} "
              f"{stats['stddev']:>8} {stats['max_deviation_pct']:>8.1f}%")

    # Test rebalancing cost
    print("\n2. Rebalancing Cost (adding node-5 to 5-node cluster)")
    print("-" * 70)

    ring = ProductionConsistentHashRing(num_virtual_nodes=150)
    for i in range(5):
        ring.add_node(f"node-{i}")

    cost = ring.measure_rebalance_cost(keys, "add", "node-5")
    print(f"  Keys moved: {cost['keys_moved']:,} / {cost['total_keys']:,} "
          f"({cost['movement_pct']:.1f}%)")
    print(f"  Ideal:      {cost['ideal_movement_pct']:.1f}%")

    # Test replication
    print("\n3. Replica Node Selection")
    print("-" * 70)
    ring2 = ProductionConsistentHashRing(num_virtual_nodes=150, replication_factor=3)
    for i in range(5):
        ring2.add_node(f"node-{i}")

    for key in ["user:1", "user:2", "user:3"]:
        replicas = ring2.get_replica_nodes(key)
        print(f"  Key '{key}' → replicas: {replicas}")


if __name__ == "__main__":
    print("--- Key-Range Partitioning ---")
    demo_key_range()

    print("\n--- Compound Partitioning ---")
    demo_compound()

    print("\n--- Virtual Nodes Balance ---")
    demo_virtual_nodes()

    print("\n--- Jump Consistent Hash ---")
    demo_jump_hash()

    print("\n--- Hot Spot Salting ---")
    demo_hot_spot_salting()

    print("\n--- Comprehensive Demo ---")
    run_comprehensive_demo()
```

---

## 9. 요약

| 개념 | 핵심 내용 |
|---|---|
| **Key-range 파티셔닝** | 범위 쿼리에 적합; 순차 키의 핫스팟 위험 |
| **Hash 파티셔닝** | 균일 분산; 범위 쿼리 불가 |
| **Compound 파티셔닝** | 분산을 위한 해시 + 파티션 내 범위 쿼리를 위한 정렬 |
| **Consistent hashing** | 노드 변경 시 데이터 이동 최소화; 가상 노드가 균형 향상 |
| **Jump consistent hash** | O(1) 메모리, O(ln N) 시간; 번호가 매겨진 버킷에 이상적 |
| **로컬 보조 인덱스** | 데이터 쓰기와 함께 업데이트; 읽기에 scatter-gather 필요 |
| **글로벌 보조 인덱스** | 효율적인 읽기; 교차 파티션 쓰기 필요 |
| **고정 파티션** | 단순한 리밸런싱; 파티션 수가 미리 고정 |
| **동적 파티션** | 데이터 성장에 적응; 더 복잡한 관리 |
| **핫스팟 완화** | 키 솔팅, 파티션 분할, 캐싱 |

### 의사 결정 프레임워크

```
Do you need range queries?
  ├── Yes → Key-range or Compound partitioning
  └── No  → Hash partitioning

How many nodes will you have?
  ├── Fixed, known → Jump consistent hash
  └── Dynamic       → Consistent hashing with virtual nodes

Secondary index query pattern?
  ├── Read-heavy → Global (term-partitioned) index
  └── Write-heavy → Local (document-partitioned) index

Expected hot spots?
  ├── Predictable → Salt known hot keys
  └── Unpredictable → Bounded-load consistent hashing + monitoring
```

---

[다음: 분산 스토리지 사례 연구](./12_Distributed_Storage_Case_Studies.md)
