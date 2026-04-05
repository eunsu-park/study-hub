# 레슨 12: 분산 스토리지 사례 연구

[개요](./00_Overview.md) | [이전: 파티셔닝과 샤딩](./11_Partitioning_and_Sharding.md) | [다음: 장애 감지와 멤버십](./13_Failure_Detection_and_Membership.md)

---

## 학습 목표

- Google Spanner의 아키텍처, TrueTime API, external consistency 달성 방법 분석
- Amazon Dynamo의 설계 철학과 일관성을 희생한 항상 쓰기 가능(always-writable) 접근 방식 이해
- Apache Kafka의 분산 커밋 로그 아키텍처와 exactly-once 시맨틱 평가
- CockroachDB의 Raft와 하이브리드 논리 시계를 사용한 직렬화 가능 분산 SQL 구현 검토
- 이러한 시스템들을 일관성, 파티셔닝, 복제, 트랜잭션 모델 측면에서 비교 대조

---

## 1. Google Spanner

Spanner는 Google의 전 세계적으로 분산된 강한 일관성 데이터베이스이다. Paxos 복제와 **TrueTime**이라는 새로운 시간 API의 조합을 사용하여, 글로벌 규모에서 **external consistency**(linearizability보다 강한)를 제공하는 최초의 시스템이다.

### 1.1 아키텍처 개요

Spanner의 아키텍처는 여러 추상화 계층을 가진다:

```
Universe (one Spanner deployment, e.g., "production")
  │
  ├── Zone 1 (≈ datacenter)
  │   ├── Zone Master (assigns data to spanservers)
  │   ├── Location Proxy (clients use to find data)
  │   ├── Spanserver 1
  │   │   ├── Tablet 1 (key range: [A..F))
  │   │   │   ├── Colossus (distributed file system) for storage
  │   │   │   └── Paxos state machine (replicates this tablet)
  │   │   ├── Tablet 2 (key range: [F..K))
  │   │   └── ...
  │   └── Spanserver 2
  │       └── ...
  │
  ├── Zone 2 (another datacenter, possibly another continent)
  │   └── (same structure)
  │
  └── Zone 3
      └── (same structure)

Each tablet is replicated via Paxos across zones.
One Paxos replica is the leader; it handles reads and writes.
```

| 컴포넌트 | 역할 |
|---|---|
| **Universe** | 단일 Spanner 배포 (예: 프로덕션용, 테스트용 각각) |
| **Zone** | 물리적 격리 단위 (≈ 데이터센터). Paxos를 위해 최소 3개 zone 필요. |
| **Spanserver** | 데이터를 서빙. 각각 100-1000개의 tablet을 관리. |
| **Tablet** | 연속된 행 범위, Colossus에 저장. 복제의 기본 단위. |
| **Directory** | 공통 키 접두사를 가진 행의 논리적 그룹. 데이터 배치 및 이동의 단위. |

### 1.2 TrueTime

Spanner의 가장 혁신적인 기여는 **TrueTime API**로, 시계 불확실성을 명시적으로 노출한다.

대부분의 시스템은 시계가 정확하다고 가정한다(그리고 틀린다). Spanner는 시계가 제한된 불확실성을 가진다는 것을 받아들이고, 커밋하기 전에 **불확실성을 기다린다**.

```
Standard clock API:
  now() → timestamp        (no uncertainty information)

TrueTime API:
  TT.now() → TTinterval    (earliest, latest)
  TT.after(t) → bool       ("is it definitely after time t?")
  TT.before(t) → bool      ("is it definitely before time t?")

TTinterval example:
  TT.now() = [t - ε, t + ε]
  where ε is typically 1-7 ms (GPS + atomic clock synchronization)
```

```python
import time
from dataclasses import dataclass


@dataclass
class TTInterval:
    """TrueTime interval: [earliest, latest]."""
    earliest: float
    latest: float

    @property
    def uncertainty(self) -> float:
        return self.latest - self.earliest

    def definitely_after(self, t: float) -> bool:
        """Is the current time definitely after t?"""
        return self.earliest > t

    def definitely_before(self, t: float) -> bool:
        """Is the current time definitely before t?"""
        return self.latest < t


class TrueTime:
    """
    Simulated TrueTime API.
    In production, Google uses GPS receivers + atomic clocks
    in each datacenter to bound uncertainty to ~1-7ms.
    """

    def __init__(self, epsilon_ms: float = 4.0):
        self.epsilon_ms = epsilon_ms  # half-width of uncertainty

    def now(self) -> TTInterval:
        """Return a time interval guaranteed to contain the true time."""
        t = time.time()
        epsilon = self.epsilon_ms / 1000.0
        return TTInterval(earliest=t - epsilon, latest=t + epsilon)

    def after(self, t: float) -> bool:
        """Is the true time definitely after t?"""
        return self.now().earliest > t

    def before(self, t: float) -> bool:
        """Is the true time definitely before t?"""
        return self.now().latest < t

    def wait_until_after(self, t: float) -> None:
        """
        Block until the true time is definitely after t.
        This is the 'commit wait' that ensures external consistency.
        """
        while not self.after(t):
            time.sleep(0.001)  # poll every 1ms
```

### 1.3 커밋 대기를 통한 External Consistency

External consistency는 다음을 의미한다: 트랜잭션 T1이 트랜잭션 T2가 시작하기 전에(실제 벽시계 시간으로) 커밋되면, T1의 커밋 타임스탬프가 T2의 커밋 타임스탬프보다 작다.

**커밋 프로토콜**:

```
1. Leader picks commit timestamp s = TT.now().latest
   (guaranteed to be ≥ true time)

2. Leader WAITS until TT.after(s) is true
   (commit wait: ensures no future transaction can pick a timestamp ≤ s)

3. Leader applies the write and responds to client

Wait time ≈ 2ε (twice the clock uncertainty)
With ε ≈ 4ms, commit wait ≈ 8ms
```

```python
class SpannerTransaction:
    """
    Simplified Spanner read-write transaction with commit wait.
    Demonstrates how TrueTime ensures external consistency.
    """

    def __init__(self, tt: TrueTime, transaction_id: str):
        self.tt = tt
        self.transaction_id = transaction_id
        self.read_set: dict[str, tuple] = {}    # key -> (value, timestamp)
        self.write_set: dict[str, any] = {}      # key -> value
        self.commit_timestamp: float | None = None
        self.state = "active"

    def read(self, store: dict, key: str) -> any:
        """Read a key (acquires read lock in real Spanner)."""
        if key in self.write_set:
            return self.write_set[key]
        entry = store.get(key)
        if entry:
            value, ts = entry
            self.read_set[key] = (value, ts)
            return value
        return None

    def write(self, key: str, value: any) -> None:
        """Buffer a write (applied at commit time)."""
        self.write_set[key] = value

    def commit(self, store: dict) -> bool:
        """
        Commit with TrueTime-based external consistency.

        Steps:
        1. Acquire write locks (simplified)
        2. Pick commit timestamp = TT.now().latest
        3. COMMIT WAIT: wait until TT.after(commit_timestamp)
        4. Apply writes with commit timestamp
        5. Release locks
        """
        if self.state != "active":
            return False

        # Step 2: Pick commit timestamp (guaranteed ≥ true time)
        interval = self.tt.now()
        self.commit_timestamp = interval.latest

        # Step 3: Commit wait — ensure no future tx can get a lower timestamp
        self.tt.wait_until_after(self.commit_timestamp)

        # Step 4: Apply writes
        for key, value in self.write_set.items():
            store[key] = (value, self.commit_timestamp)

        self.state = "committed"
        return True


class SpannerReadOnlyTransaction:
    """
    Spanner read-only transaction.
    Lock-free: picks a timestamp and reads a consistent snapshot.
    Can execute on ANY replica (not just the leader).
    """

    def __init__(self, tt: TrueTime):
        # Pick a read timestamp that is guaranteed to be in the past
        # (all writes with timestamp ≤ this are committed)
        interval = tt.now()
        self.read_timestamp = interval.latest
        # Wait to ensure this timestamp is in the past
        tt.wait_until_after(self.read_timestamp)

    def read(self, store: dict, key: str) -> any:
        """
        Read the value of key at self.read_timestamp.
        Uses multi-version storage: reads the latest version ≤ read_timestamp.
        """
        entry = store.get(key)
        if entry:
            value, write_ts = entry
            if write_ts <= self.read_timestamp:
                return value
        return None
```

### 1.4 스키마 인터리빙

Spanner는 **인터리브 테이블**을 지원한다: 자식 테이블의 행이 부모 행과 물리적으로 함께 배치된다. 이는 분산 데이터베이스에서 성능에 매우 중요하다.

```sql
-- Parent table
CREATE TABLE Users (
  user_id INT64 NOT NULL,
  name    STRING(100),
) PRIMARY KEY (user_id);

-- Child table interleaved in parent
CREATE TABLE Orders (
  user_id  INT64 NOT NULL,
  order_id INT64 NOT NULL,
  total    FLOAT64,
) PRIMARY KEY (user_id, order_id),
  INTERLEAVE IN PARENT Users ON DELETE CASCADE;
```

```
Physical layout (interleaved):

Tablet for key range [user_id 1000..2000]:
  ┌─────────────────────────────────────────┐
  │ Users(1000, "Alice")                    │
  │   Orders(1000, 1, 59.99)               │
  │   Orders(1000, 2, 124.50)              │
  │ Users(1001, "Bob")                      │
  │   Orders(1001, 1, 29.99)               │
  │ Users(1002, "Carol")                    │
  │   (no orders)                           │
  └─────────────────────────────────────────┘

Query: "SELECT * FROM Orders WHERE user_id = 1000"
→ Hits exactly ONE tablet, reads co-located data
→ No distributed join needed!
```

### 1.5 Spanner 성능 특성

| 연산 | 지연 시간 | 참고 |
|---|---|---|
| 읽기-쓰기 트랜잭션 | ~10-15ms | Paxos + 커밋 대기 (~2ε ≈ 8ms) |
| 읽기 전용 트랜잭션 (단일 지역) | ~1-5ms | 로컬 복제본에서 잠금 없는 스냅샷 읽기 |
| 읽기 전용 트랜잭션 (지역 간) | ~5-50ms | 안전한 스냅샷 타임스탬프를 기다려야 함 |
| 강한 읽기 | ~5-10ms | 최신 타임스탬프에서 읽기 |
| 오래된 읽기 (제한된 staleness) | ~1-2ms | 약간 이전 타임스탬프에서 읽기, 대기 회피 |

### 1.6 Spanner SLA

Google Cloud Spanner는 다음을 제공한다:
- **99.999% 가용성** (멀티 리전) — 연간 ~5분 다운타임
- **99.99% 가용성** (단일 리전) — 연간 ~52분 다운타임
- **External consistency** — 상용 데이터베이스가 제공하는 가장 강한 일관성 보장

---

## 2. Amazon Dynamo

Amazon Dynamo(2007년 논문)는 많은 현대 분산 키-값 저장소의 기초 설계이다. 설계 철학이 Spanner와 근본적으로 다르다.

### 2.1 설계 철학

| 원칙 | Spanner | Dynamo |
|---|---|---|
| **일관성** | External consistency (가장 강함) | 최종 일관성 (가장 약한 실용적 수준) |
| **가용성** | 일관성을 위해 가용성 희생 | **항상 쓰기 가능** — 쓰기를 절대 거부하지 않음 |
| **설계 목표** | 강한 보장 | SLA 기반 지연 시간 (99.9번째 백분위수) |
| **충돌 해결** | 방지 (잠금 + Paxos) | 감지 + 해결 (vector clock, LWW) |

Dynamo는 Amazon의 장바구니를 위해 설계되었으며, 쓰기를 거부하는 것보다 (충돌이 발생하더라도) 항상 수락하는 것이 낫다. 고객이 장바구니에 항목을 추가할 때 절대 오류를 보면 안 된다.

### 2.2 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Dynamo Cluster                               │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Node A   │  │ Node B   │  │ Node C   │  │ Node D   │  ...     │
│  │          │  │          │  │          │  │          │          │
│  │ Key range│  │ Key range│  │ Key range│  │ Key range│          │
│  │ [0, 90)  │  │ [60, 180)│  │[120, 270)│  │[210, 360)│          │
│  │          │  │          │  │          │  │          │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
│       │              │              │              │               │
│       └──── Consistent Hashing Ring (with virtual nodes) ─────────│
│       └──── Gossip Protocol (membership + failure detection) ─────│
│       └──── Sloppy Quorums + Hinted Handoff ──────────────────────│
│       └──── Vector Clocks (conflict detection) ───────────────────│
│       └──── Merkle Trees (anti-entropy) ──────────────────────────│
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 핵심 기술

다음 각 기술은 레슨 9-11에서 자세히 다루었다. 여기서는 Dynamo가 이들을 어떻게 결합하는지 본다:

#### 가상 노드를 사용한 Consistent Hashing

Dynamo는 키를 분산하기 위해 가상 노드가 있는 consistent hashing(레슨 11)을 사용한다. 각 물리 노드가 해시 링에서 여러 위치를 담당한다.

```python
class DynamoNode:
    """
    Simplified Dynamo node with vector clocks and quorum operations.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.store: dict[str, list[tuple[any, dict[str, int]]]] = {}
        # key -> list of (value, vector_clock) for conflict tracking
        self.hint_store: list[tuple[str, str, any, dict[str, int]]] = []
        # (target_node_id, key, value, vector_clock)

    def put(self, key: str, value: any,
            context: dict[str, int] | None = None) -> dict[str, int]:
        """
        Write a value with vector clock versioning.
        context: the vector clock from a previous read (None for new key).
        """
        # Increment this node's entry in the vector clock
        vc = dict(context) if context else {}
        vc[self.node_id] = vc.get(self.node_id, 0) + 1

        if key not in self.store:
            self.store[key] = []

        # Remove versions dominated by the new vector clock
        surviving = []
        for old_val, old_vc in self.store[key]:
            if not self._vc_dominates(vc, old_vc):
                surviving.append((old_val, old_vc))
        surviving.append((value, vc))
        self.store[key] = surviving

        return vc

    def get(self, key: str) -> list[tuple[any, dict[str, int]]]:
        """
        Read all versions of a key.
        Returns list of (value, vector_clock) — may have multiple
        concurrent versions (conflicts).
        """
        return self.store.get(key, [])

    @staticmethod
    def _vc_dominates(vc_a: dict[str, int], vc_b: dict[str, int]) -> bool:
        """Check if vector clock A dominates (causally after) B."""
        # A dominates B iff A[i] >= B[i] for all i and A[j] > B[j] for some j
        all_nodes = set(vc_a.keys()) | set(vc_b.keys())
        at_least_one_greater = False
        for node in all_nodes:
            a_val = vc_a.get(node, 0)
            b_val = vc_b.get(node, 0)
            if a_val < b_val:
                return False
            if a_val > b_val:
                at_least_one_greater = True
        return at_least_one_greater


class DynamoCoordinator:
    """
    Coordinator for Dynamo-style operations.
    Handles quorum reads/writes with conflict detection.
    """

    def __init__(self, nodes: list[DynamoNode], n: int = 3, w: int = 2, r: int = 2):
        self.nodes = nodes
        self.n = n
        self.w = w
        self.r = r

    def put(self, key: str, value: any,
            context: dict[str, int] | None = None) -> dict:
        """
        Quorum write with vector clock.
        Sends to N nodes, waits for W acknowledgments.
        """
        # Select N nodes (simplified: first N)
        target_nodes = self.nodes[:self.n]
        acks = 0
        final_vc = None

        for node in target_nodes:
            try:
                vc = node.put(key, value, context)
                acks += 1
                final_vc = vc
            except Exception:
                pass

        return {
            "success": acks >= self.w,
            "acks": acks,
            "vector_clock": final_vc,
        }

    def get(self, key: str) -> dict:
        """
        Quorum read.
        Reads from N nodes, waits for R responses.
        Returns all concurrent versions (for client-side resolution).
        """
        target_nodes = self.nodes[:self.n]
        all_versions: list[tuple[any, dict[str, int]]] = []
        responses = 0

        for node in target_nodes:
            versions = node.get(key)
            if versions:
                all_versions.extend(versions)
                responses += 1

        # Merge: keep only non-dominated versions
        merged = self._merge_versions(all_versions)

        return {
            "success": responses >= self.r,
            "responses": responses,
            "versions": merged,
            "has_conflict": len(merged) > 1,
        }

    @staticmethod
    def _merge_versions(
            versions: list[tuple[any, dict[str, int]]]
    ) -> list[tuple[any, dict[str, int]]]:
        """Keep only versions not dominated by any other version."""
        if not versions:
            return []
        result = []
        for val, vc in versions:
            dominated = any(
                DynamoNode._vc_dominates(other_vc, vc)
                for other_val, other_vc in versions
                if other_vc != vc
            )
            if not dominated and (val, vc) not in result:
                result.append((val, vc))
        return result


def demo_dynamo_conflict():
    """Demonstrate Dynamo-style conflict detection with vector clocks."""
    nodes = [DynamoNode(f"N{i}") for i in range(3)]
    coord = DynamoCoordinator(nodes, n=3, w=2, r=2)

    # Initial write
    result = coord.put("cart:user1", {"items": ["book"]})
    print(f"Initial write: {result}")

    # Simulate partition: two nodes independently update
    # Node 0 adds "pen"
    vc0 = nodes[0].put("cart:user1", {"items": ["book", "pen"]},
                        context={"N0": 1})

    # Node 1 (didn't see the pen addition) adds "notebook"
    vc1 = nodes[1].put("cart:user1", {"items": ["book", "notebook"]},
                        context={"N1": 1})

    # Read detects conflict: two concurrent versions
    read_result = coord.get("cart:user1")
    print(f"\nConflict detected: {read_result['has_conflict']}")
    print(f"Concurrent versions:")
    for val, vc in read_result["versions"]:
        print(f"  {val} (vc={vc})")

    # Client resolves by merging (union of items)
    if read_result["has_conflict"]:
        all_items = set()
        merged_vc: dict[str, int] = {}
        for val, vc in read_result["versions"]:
            all_items.update(val.get("items", []))
            for node, count in vc.items():
                merged_vc[node] = max(merged_vc.get(node, 0), count)

        resolved = {"items": sorted(all_items)}
        coord.put("cart:user1", resolved, context=merged_vc)
        print(f"\nResolved: {resolved}")

        # Verify: no more conflicts
        final = coord.get("cart:user1")
        print(f"After resolution, conflict: {final['has_conflict']}")
        print(f"Final value: {final['versions'][0][0]}")
```

### 2.4 Dynamo vs DynamoDB

Amazon Dynamo(2007년 논문)는 *내부* 시스템이다. **Amazon DynamoDB**(관리형 서비스, 2012년 출시)는 Dynamo에서 영감을 받았지만 상당히 다르다:

| 측면 | Dynamo (2007년 논문) | DynamoDB (관리형 서비스) |
|---|---|---|
| **충돌 해결** | Vector clock + 클라이언트 해결 | Last-writer-wins (단순화) |
| **파티셔닝** | Consistent hashing | Hash + range (compound) |
| **멤버십** | Gossip 프로토콜 | 중앙화된 메타데이터 서비스 |
| **스토리지** | 인메모리 + 디스크 | SSD 기반 자동 티어링 |
| **트랜잭션** | 없음 | TransactWriteItems / TransactGetItems (2018) |
| **강한 일관성** | 미지원 | 읽기별 선택 가능 (ConsistentRead=true) |
| **글로벌 테이블** | 수동 멀티 DC | 내장 멀티 리전 복제 |

---

## 3. Apache Kafka

Kafka는 이벤트 스트리밍, pub/sub 메시징, 이벤트 소싱에 사용되는 분산 커밋 로그 플랫폼이다. 전통적인 의미의 데이터베이스는 아니지만, 고유한 일관성과 내구성 특성을 가진 분산 스토리지 시스템이다.

### 3.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kafka Cluster                               │
│                                                                  │
│  Topic: "orders" (3 partitions, replication factor 2)           │
│                                                                  │
│  Broker 1           Broker 2           Broker 3                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ P0 (Leader)  │  │ P0 (Follower)│  │ P1 (Follower)│         │
│  │ P1 (Leader)  │  │ P2 (Leader)  │  │ P2 (Follower)│         │
│  │ P2 (Follower)│  │ P1 (Follower)│  │ P0 (spare)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  Each partition is an append-only log:                           │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐                    │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │  ← offsets       │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                    │
│  │◄─── committed ────►│◄─── uncommitted ──►│                    │
│            (HW)                 (LEO)                            │
└─────────────────────────────────────────────────────────────────┘

HW  = High Watermark (all ISR replicas have this offset)
LEO = Log End Offset (latest offset on the leader)
```

### 3.2 ISR (In-Sync Replicas)와 리더 선출

Kafka는 **ISR**이라는 개념을 사용한다 — leader와 "동기화 상태"인 복제본의 집합이다. 복제본이 설정 가능한 시간 창 내에 leader의 로그 끝까지 모든 메시지를 가져왔으면 ISR에 포함된다.

```
Leader (Broker 1):  [0, 1, 2, 3, 4, 5, 6]  ← LEO = 7
ISR Replica (Br 2): [0, 1, 2, 3, 4, 5]      ← caught up within timeout → IN ISR
Lagging (Br 3):     [0, 1, 2, 3]             ← too far behind → OUT of ISR

High Watermark (HW) = min(LEO of all ISR replicas) = 6
Messages 0-5 are "committed" (safe to consume)
Message 6 is on Br 2 but not committed (Br 3 doesn't have it — but Br 3 is out of ISR)
```

**리더 선출**: leader가 장애가 발생하면, ISR에서 새 leader가 선택된다. ISR 멤버만 후보가 된다(`unclean.leader.election.enable=true`가 아닌 한, 이는 데이터 손실 위험이 있다).

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AckMode(Enum):
    FIRE_AND_FORGET = 0    # acks=0: don't wait
    LEADER_ACK = 1          # acks=1: leader confirms write to its log
    ALL_ISR_ACK = -1        # acks=all: all ISR replicas confirm


@dataclass
class LogEntry:
    """A single entry in the Kafka log."""
    offset: int
    key: Optional[str]
    value: str
    timestamp: float
    producer_id: Optional[int] = None
    sequence_num: Optional[int] = None  # for idempotent producers


@dataclass
class KafkaPartition:
    """Simulated Kafka partition with ISR tracking."""
    topic: str
    partition_id: int
    leader_id: int
    isr: list[int]  # broker IDs in ISR
    log: list[LogEntry] = field(default_factory=list)
    high_watermark: int = 0

    @property
    def leo(self) -> int:
        """Log End Offset."""
        return len(self.log)

    def append(self, key: Optional[str], value: str,
               ack_mode: AckMode = AckMode.ALL_ISR_ACK,
               producer_id: Optional[int] = None,
               sequence_num: Optional[int] = None) -> dict:
        """
        Append a message to the partition log.
        Acknowledgment depends on ack_mode.
        """
        # Idempotent producer: reject duplicate sequence numbers
        if producer_id is not None and sequence_num is not None:
            for entry in reversed(self.log):
                if (entry.producer_id == producer_id and
                        entry.sequence_num == sequence_num):
                    return {
                        "status": "duplicate",
                        "offset": entry.offset,
                    }

        import time
        entry = LogEntry(
            offset=self.leo,
            key=key,
            value=value,
            timestamp=time.time(),
            producer_id=producer_id,
            sequence_num=sequence_num,
        )
        self.log.append(entry)

        if ack_mode == AckMode.ALL_ISR_ACK:
            # In production: wait for all ISR replicas to fetch this offset
            self.high_watermark = self.leo  # simplified: assume instant replication
        elif ack_mode == AckMode.LEADER_ACK:
            pass  # leader-only ack, HW advances when replicas catch up
        elif ack_mode == AckMode.FIRE_AND_FORGET:
            pass  # no waiting at all

        return {
            "status": "ok",
            "offset": entry.offset,
            "partition": self.partition_id,
        }

    def read(self, offset: int, max_records: int = 10,
             read_committed: bool = True) -> list[LogEntry]:
        """
        Read records starting from offset.
        read_committed=True: only return up to high watermark.
        """
        end = self.high_watermark if read_committed else self.leo
        return self.log[offset:min(offset + max_records, end)]
```

### 3.3 로그 컴팩션

Kafka는 **로그 컴팩션**을 지원한다: 시간별로 오래된 세그먼트를 삭제하는 대신, 각 키의 최신 값만 유지한다. 이는 Kafka를 변경 로그 스타일의 저장소로 전환한다.

```
Before compaction:
  offset: 0   key=A  value="v1"
  offset: 1   key=B  value="v1"
  offset: 2   key=A  value="v2"    ← supersedes offset 0
  offset: 3   key=C  value="v1"
  offset: 4   key=B  value="v2"    ← supersedes offset 1
  offset: 5   key=A  value=null    ← tombstone: delete key A

After compaction:
  offset: 3   key=C  value="v1"    (latest for C)
  offset: 4   key=B  value="v2"    (latest for B)
  (key A deleted: tombstone removes it entirely after grace period)
```

```python
def compact_log(log: list[LogEntry]) -> list[LogEntry]:
    """
    Simulate Kafka log compaction.
    Keep only the latest entry for each key.
    Tombstones (value=None) delete the key entirely.
    """
    # Track the latest entry for each key
    latest: dict[str, LogEntry] = {}
    for entry in log:
        if entry.key is not None:
            latest[entry.key] = entry

    # Build compacted log, preserving order
    compacted = []
    seen_keys: set[str] = set()
    for entry in reversed(log):
        if entry.key is None:
            continue
        if entry.key not in seen_keys:
            seen_keys.add(entry.key)
            if entry.value is not None:  # skip tombstones
                compacted.append(entry)

    compacted.reverse()
    return compacted
```

### 3.4 Exactly-Once 시맨틱 (EOS)

Kafka는 두 가지 메커니즘을 통해 exactly-once 처리를 달성한다:

1. **멱등 프로듀서**: 각 프로듀서가 고유 ID와 시퀀스 번호를 받는다. 브로커가 메시지를 중복 제거한다.

2. **트랜잭션**: 여러 파티션에 걸친 원자적 쓰기. 모든 메시지가 커밋되거나 아무것도 커밋되지 않는다.

```python
class KafkaTransaction:
    """
    Simplified Kafka transaction.
    Provides atomic writes across multiple partitions.
    """

    def __init__(self, transaction_id: str, partitions: list[KafkaPartition]):
        self.transaction_id = transaction_id
        self.partitions = {p.partition_id: p for p in partitions}
        self.pending: list[tuple[int, str, str]] = []  # (partition_id, key, value)
        self.state = "active"  # active → preparing → committed / aborted

    def send(self, partition_id: int, key: str, value: str) -> None:
        """Buffer a message for transactional send."""
        if self.state != "active":
            raise RuntimeError(f"Transaction is {self.state}")
        self.pending.append((partition_id, key, value))

    def commit(self) -> bool:
        """
        Commit the transaction atomically.
        Uses a two-phase protocol with a transaction coordinator.
        """
        self.state = "preparing"

        # Phase 1: Write all messages to partition logs (not yet visible)
        offsets = []
        for pid, key, value in self.pending:
            partition = self.partitions[pid]
            result = partition.append(key, value)
            offsets.append(result)

        # Phase 2: Write commit marker to transaction log
        # (In real Kafka: __transaction_state internal topic)
        self.state = "committed"

        # Messages become visible to consumers with read_committed=True
        return True

    def abort(self) -> None:
        """Abort the transaction. Pending messages are discarded."""
        self.state = "aborted"
        self.pending.clear()
```

### 3.5 컨슈머 그룹과 파티션 할당

```
Topic "orders" with 4 partitions:

Consumer Group "order-processing":
  Consumer A: assigned P0, P1
  Consumer B: assigned P2, P3

  Each partition is consumed by exactly ONE consumer in the group.
  Adding Consumer C triggers rebalance:
    Consumer A: P0
    Consumer B: P2, P3
    Consumer C: P1

  Max parallelism = number of partitions
```

### 3.6 KRaft: ZooKeeper 제거

Kafka 3.3(2022)부터 Kafka는 메타데이터 관리를 위해 ZooKeeper를 대체하는 **KRaft** (Kafka Raft)를 사용한다:

| 측면 | ZooKeeper 기반 | KRaft |
|---|---|---|
| **메타데이터 저장** | 외부 ZooKeeper 앙상블 | 내부 Raft 기반 메타데이터 quorum |
| **컨트롤러** | 단일 활성 컨트롤러 (ZK를 통해 선출) | Raft를 통한 활성 컨트롤러 + 대기 컨트롤러 |
| **확장성** | ~200K 파티션에서 ZK 병목 | 수백만 파티션 |
| **운영** | 두 시스템 관리 (Kafka + ZK) | 단일 시스템 |
| **시작** | ZK를 먼저 시작해야 함 | 자체 포함 |

---

## 4. CockroachDB

CockroachDB는 전용 하드웨어(GPS/원자 시계 등) 없이 Spanner의 일관성 보장을 제공하도록 설계된 오픈 소스 분산 SQL 데이터베이스이다.

### 4.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    CockroachDB Cluster                          │
│                                                                  │
│  SQL Layer:     Parse → Plan → Optimize → Distribute → Execute  │
│                            │                                     │
│  Distribution Layer:       │                                     │
│    ┌────────────────────────┼──────────────────────┐            │
│    │                        ▼                      │            │
│    │  Range 1            Range 2         Range N   │            │
│    │  [A..F)             [F..K)          [X..Z)    │            │
│    │  ┌────────────┐    ┌────────────┐   ┌──────┐  │            │
│    │  │Raft Group 1│    │Raft Group 2│   │Raft N│  │            │
│    │  │ L F F      │    │ F L F      │   │F F L │  │            │
│    │  └────────────┘    └────────────┘   └──────┘  │            │
│    │  (L=Leader, F=Follower)                       │            │
│    └───────────────────────────────────────────────┘            │
│                                                                  │
│  Storage Layer:  Pebble (LSM-tree key-value engine, Go)         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 자동 분할이 있는 Range 기반 파티셔닝

CockroachDB는 데이터를 **range**(기본 512 MB)로 파티셔닝한다. 각 range는 키 공간의 연속 구간이다. range가 임계값을 초과하면 자동으로 분할된다.

```
Initial state:
  Range 1: ["", "") — one range covering everything

After data growth:
  Range 1: ["", "customer:5000")
  Range 2: ["customer:5000", "order:3000")
  Range 3: ["order:3000", "")

Each range is independently replicated via its own Raft group.
```

### 4.3 트랜잭션 모델: Write Intent와 Parallel Commit

CockroachDB는 TrueTime이 필요 없는 고유한 트랜잭션 프로토콜을 사용한다.

#### Write Intent

잠금을 획득하는 대신, 트랜잭션은 **intent** — 다른 트랜잭션에 "커밋되지 않음"으로 보이는 잠정적 값 — 을 쓴다.

```
Transaction T1 writes key "x":

Before T1:
  key "x" → value "old" (committed)

During T1 (before commit):
  key "x" → INTENT(txn=T1, value="new", status=PENDING)
  (other transactions see the intent and check T1's status)

After T1 commits:
  key "x" → value "new" (committed, intent cleaned up)

If T1 aborts:
  key "x" → value "old" (intent cleaned up)
```

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import time


class TxnStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class WriteIntent:
    """A provisional write that may or may not be committed."""
    txn_id: str
    key: str
    value: Any
    timestamp: float
    status: TxnStatus = TxnStatus.PENDING


@dataclass
class CockroachTransaction:
    """
    Simplified CockroachDB transaction with write intents.
    """
    txn_id: str
    timestamp: float = field(default_factory=time.time)
    status: TxnStatus = TxnStatus.PENDING
    intents: list[WriteIntent] = field(default_factory=list)

    def write(self, store: dict, key: str, value: Any) -> WriteIntent:
        """
        Write an intent (provisional value).
        The intent is visible to other transactions as PENDING.
        """
        intent = WriteIntent(
            txn_id=self.txn_id,
            key=key,
            value=value,
            timestamp=self.timestamp,
        )
        store[key] = intent
        self.intents.append(intent)
        return intent

    def read(self, store: dict, key: str) -> Optional[Any]:
        """
        Read a key. If there's a pending intent from another txn,
        we must check that txn's status (push or wait).
        """
        entry = store.get(key)
        if entry is None:
            return None

        if isinstance(entry, WriteIntent):
            if entry.txn_id == self.txn_id:
                # Our own intent — read the provisional value
                return entry.value
            elif entry.status == TxnStatus.COMMITTED:
                return entry.value
            elif entry.status == TxnStatus.PENDING:
                # Conflict! In real CockroachDB:
                # 1. Push the other txn's timestamp forward, OR
                # 2. Wait for the other txn to complete, OR
                # 3. Abort the other txn if it's expired
                raise Exception(
                    f"Write-read conflict: {self.txn_id} blocked by "
                    f"{entry.txn_id} on key {key}"
                )
            else:
                # Aborted intent — ignore it
                return None
        else:
            return entry

    def commit(self, txn_record: dict) -> bool:
        """
        Commit the transaction.
        1. Write COMMITTED status to transaction record
        2. Asynchronously clean up intents
        """
        self.status = TxnStatus.COMMITTED
        txn_record[self.txn_id] = self.status

        # Mark all intents as committed
        for intent in self.intents:
            intent.status = TxnStatus.COMMITTED

        return True

    def abort(self, txn_record: dict) -> None:
        """Abort the transaction and clean up intents."""
        self.status = TxnStatus.ABORTED
        txn_record[self.txn_id] = self.status
        for intent in self.intents:
            intent.status = TxnStatus.ABORTED
```

#### Parallel Commit

CockroachDB의 **parallel commit** 최적화는 트랜잭션 레코드와 intent를 순차적이 아닌 병렬로 쓰기하여 커밋 지연을 줄인다.

```
Standard 2-phase commit:
  1. Write intents to all ranges                    (1 round trip)
  2. Write COMMITTED to transaction record           (1 round trip)
  3. Clean up intents asynchronously
  Total: 2 round trips before acknowledging client

Parallel commits:
  1. Write intents + STAGING status in parallel      (1 round trip)
  2. Transaction is implicitly committed if all
     intents are present (verified lazily)
  Total: 1 round trip before acknowledging client
```

### 4.4 시계 동기화: 하이브리드 논리 시계

CockroachDB에는 TrueTime이 없다. 대신 **하이브리드 논리 시계(HLC)** — 물리적 시간과 논리적 카운터의 조합(레슨 2에서 설명)을 사용한다.

```python
@dataclass
class HybridLogicalClock:
    """
    Hybrid Logical Clock (HLC).
    Combines physical time with a logical counter.
    Guarantees causality without perfect clock synchronization.
    """
    physical: int = 0    # wall clock time in nanoseconds
    logical: int = 0     # logical counter for same-physical-time ordering
    node_id: str = ""

    def now(self) -> 'HybridLogicalClock':
        """Generate a new HLC timestamp."""
        import time
        wall = int(time.time() * 1e9)

        if wall > self.physical:
            self.physical = wall
            self.logical = 0
        else:
            self.logical += 1

        return HybridLogicalClock(self.physical, self.logical, self.node_id)

    def update(self, remote: 'HybridLogicalClock') -> 'HybridLogicalClock':
        """Update HLC upon receiving a message from another node."""
        import time
        wall = int(time.time() * 1e9)

        if wall > self.physical and wall > remote.physical:
            self.physical = wall
            self.logical = 0
        elif remote.physical > self.physical:
            self.physical = remote.physical
            self.logical = remote.logical + 1
        elif self.physical > remote.physical:
            self.logical += 1
        else:
            # Same physical time
            self.logical = max(self.logical, remote.logical) + 1

        return HybridLogicalClock(self.physical, self.logical, self.node_id)

    def __lt__(self, other: 'HybridLogicalClock') -> bool:
        if self.physical != other.physical:
            return self.physical < other.physical
        return self.logical < other.logical

    def __repr__(self) -> str:
        return f"HLC({self.physical}, {self.logical})"
```

**불확실성 창 문제**: TrueTime 없이 CockroachDB는 클럭 스큐를 정확히 제한할 수 없다. 설정 가능한 `max_offset`(기본 500ms)을 사용한다. 노드의 시계가 `max_offset` 이상 벗어난 것이 감지되면, 클러스터에서 제거된다.

읽기의 경우, CockroachDB가 **불확실성 재시작**을 수행해야 할 수 있다: 읽기가 불확실성 창 내의 타임스탬프를 가진 값을 만나면, 더 높은 타임스탬프에서 트랜잭션을 재시작한다.

### 4.5 멀티 리전 기능

CockroachDB는 여러 멀티 리전 토폴로지를 지원한다:

```
1. REGIONAL BY ROW:
   Each row specifies its home region. Reads are fast from that region.
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ US-East │    │ EU-West │    │ AP-SE   │
   │ US rows │    │ EU rows │    │ AP rows │
   │ (leader)│    │ (leader)│    │ (leader)│
   └─────────┘    └─────────┘    └─────────┘

2. GLOBAL TABLES:
   All regions can read locally (stale follower reads).
   Writes go to the leaseholder in any region.

3. REGIONAL TABLES:
   Entire table is homed in one region.
   All reads and writes go to that region.
```

---

## 5. 종합 비교

### 5.1 기능 비교 표

| 기능 | Spanner | Dynamo/DynamoDB | Kafka | CockroachDB |
|---|---|---|---|---|
| **유형** | 관계형 (SQL) | 키-값 | 분산 로그 | 관계형 (SQL) |
| **일관성** | External (가장 강함) | 최종적 (설정 가능) | 파티션별 순서 | Serializable |
| **트랜잭션** | 완전한 ACID (2PC + Paxos) | 제한적 (DDB: 단일 항목 또는 transact API) | 프로듀서 트랜잭션 | 완전한 ACID (parallel commit) |
| **파티셔닝** | Range (directory 기반) | Hash (consistent hashing) | Hash (키별 또는 round-robin) | Range (자동 분할) |
| **복제** | 분할별 Paxos | Sloppy quorum (N, W, R) | ISR (leader + 동기화 복제본) | Range별 Raft |
| **충돌 해결** | 방지 (잠금) | 감지 (vector clock / LWW) | 해당 없음 (추가 전용) | 방지 (write intent) |
| **시계 메커니즘** | TrueTime (GPS + 원자시계) | Vector clock → LWW | 해당 없음 (오프셋 기반 순서) | 하이브리드 논리 시계 |
| **스키마** | 인터리브 테이블의 SQL | 키-값 (DDB: 문서) | Schema Registry (Avro/Protobuf) | PostgreSQL 호환 SQL |
| **오픈 소스** | 아니오 (Cloud Spanner) | 아니오 (DynamoDB) | 예 (Apache 2.0) | 예 (BSL → Apache 2.0) |

### 5.2 일관성과 가용성 트레이드오프

```
                        Consistency
                            ▲
                            │
          Spanner ●         │
                            │
   CockroachDB ●           │
                            │
                            │
                            │
                ────────────┼────────────► Availability
                            │
              Kafka ●       │
     (per-partition)        │
                            │
               DynamoDB ●   │  (configurable)
                            │
          Dynamo ●          │
     (always writable)      │
```

### 5.3 각 시스템을 사용해야 할 때

| 시스템 | 적합한 경우 | 적합하지 않은 경우 |
|---|---|---|
| **Spanner** | 글로벌 트랜잭션, 금융 시스템, 재고 관리 | 비용에 민감한 워크로드, 단순 키-값 접근 |
| **DynamoDB** | 고처리량 키-값, 서버리스 백엔드, 장바구니 | 복잡한 쿼리, 조인, 강한 일관성 요구 |
| **Kafka** | 이벤트 스트리밍, 로그 집계, 이벤트 소싱, CDC | 낮은 지연 포인트 쿼리, OLTP 워크로드 |
| **CockroachDB** | 멀티 리전 SQL, PostgreSQL 마이그레이션, 규제 준수 | 초저지연 단일 리전, 매우 높은 쓰기 처리량 |

---

## 6. 학습한 교훈: 공통 패턴

이 네 시스템을 연구하면 분산 스토리지 설계에서 반복되는 패턴이 드러난다.

### 6.1 패턴: 파티션 + 복제

모든 시스템이 파티셔닝(확장을 위해)과 복제(장애 허용을 위해)를 결합한다.

```python
def summarize_partition_replicate():
    """Common pattern: how each system partitions and replicates."""
    systems = {
        "Spanner": {
            "partition": "Range-based (directories, key-order preserving)",
            "replicate": "Paxos per split (synchronous, cross-zone)",
            "unit": "Split (tablet)",
        },
        "Dynamo": {
            "partition": "Consistent hashing with virtual nodes",
            "replicate": "Sloppy quorum (N copies, W write acks, R read acks)",
            "unit": "Virtual node range",
        },
        "Kafka": {
            "partition": "Hash or key-based (producer determines partition)",
            "replicate": "ISR-based (leader + in-sync followers)",
            "unit": "Partition (append-only log segment)",
        },
        "CockroachDB": {
            "partition": "Range-based with automatic splitting at 512MB",
            "replicate": "Raft per range (typically 3 or 5 replicas)",
            "unit": "Range",
        },
    }

    for name, details in systems.items():
        print(f"\n{name}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
```

### 6.2 패턴: 파티션별 합의

Spanner, CockroachDB, Kafka 모두 클러스터 전체 합의가 아닌 파티션별 합의(Paxos, Raft, 또는 ISR)를 사용한다. 이는 확장성에 매우 중요하다 — 합의 오버헤드가 전역이 아닌 파티션별이다.

### 6.3 패턴: 타임스탬프 기반 MVCC

네 시스템 모두 어떤 형태의 타임스탬프 순서를 사용한다:

| 시스템 | 타임스탬프 소스 | MVCC? |
|---|---|---|
| Spanner | TrueTime (물리적 + 불확실성) | 예 |
| Dynamo | Vector clock (논리적) | 키당 다중 버전 |
| Kafka | 오프셋 (논리적, 파티션별) | 추가 전용 (자연적 버전 관리) |
| CockroachDB | HLC (물리적 + 논리적) | 예 |

### 6.4 패턴: 지연 시간과 일관성 간의 트레이드오프

```
Higher Consistency ─────────────────────────► Lower Consistency
Higher Latency                                Lower Latency

Spanner          CockroachDB       Kafka ISR        Dynamo
(commit wait     (uncertainty       (acks=all)      (sloppy quorum
 ~2ε ≈ 8ms)      restart)                           W=1)
```

### 6.5 안티패턴: 시계 의존성

| 시스템 | 시계 의존성 | 위험 |
|---|---|---|
| Spanner | 높음 (TrueTime이 핵심) | GPS + 원자 시계로 완화 |
| Dynamo | 낮음 (vector clock은 논리적) | 시계 위험 없음, 하지만 메타데이터 오버헤드 |
| CockroachDB | 중간 (HLC + max_offset) | max_offset 초과 시계 스큐 → 노드 제거 |
| Kafka | 없음 (오프셋 기반) | 시계 위험 없음 |

---

## 7. 아키텍처 결정 시뮬레이터

```python
"""
Decision framework for choosing a distributed storage system.
Scores each system based on workload requirements.
"""

from dataclasses import dataclass


@dataclass
class WorkloadRequirements:
    """Describe the workload to get system recommendations."""
    needs_sql: bool = False
    needs_transactions: bool = False
    consistency: str = "eventual"  # "strong", "serializable", "eventual"
    read_write_ratio: float = 1.0  # >1 = read-heavy, <1 = write-heavy
    latency_p99_ms: float = 100.0
    multi_region: bool = False
    throughput_qps: int = 10000
    data_model: str = "key_value"  # "key_value", "relational", "event_log"
    open_source_required: bool = False
    budget: str = "medium"  # "low", "medium", "high"


def recommend_system(req: WorkloadRequirements) -> list[tuple[str, float, str]]:
    """
    Score each system (0-100) based on workload fit.
    Returns sorted list of (system, score, rationale).
    """
    scores: dict[str, tuple[float, list[str]]] = {
        "Spanner": (50.0, []),
        "DynamoDB": (50.0, []),
        "Kafka": (50.0, []),
        "CockroachDB": (50.0, []),
    }

    # SQL support
    if req.needs_sql:
        scores["Spanner"] = (scores["Spanner"][0] + 15, scores["Spanner"][1] + ["SQL support"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 15, scores["CockroachDB"][1] + ["PostgreSQL-compatible SQL"])
        scores["DynamoDB"] = (scores["DynamoDB"][0] - 10, scores["DynamoDB"][1] + ["No SQL (PartiQL is limited)"])
        scores["Kafka"] = (scores["Kafka"][0] - 20, scores["Kafka"][1] + ["Not a query engine"])

    # Transactions
    if req.needs_transactions:
        scores["Spanner"] = (scores["Spanner"][0] + 15, scores["Spanner"][1] + ["Full ACID transactions"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 15, scores["CockroachDB"][1] + ["Serializable transactions"])
        scores["DynamoDB"] = (scores["DynamoDB"][0] - 5, scores["DynamoDB"][1] + ["Limited transactions"])
        scores["Kafka"] = (scores["Kafka"][0] + 5, scores["Kafka"][1] + ["Producer transactions (EOS)"])

    # Consistency
    if req.consistency == "strong":
        scores["Spanner"] = (scores["Spanner"][0] + 20, scores["Spanner"][1] + ["External consistency"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 15, scores["CockroachDB"][1] + ["Serializable"])
        scores["DynamoDB"] = (scores["DynamoDB"][0] - 10, scores["DynamoDB"][1] + ["Eventual by default"])
    elif req.consistency == "eventual":
        scores["DynamoDB"] = (scores["DynamoDB"][0] + 10, scores["DynamoDB"][1] + ["Optimized for eventual"])
        scores["Kafka"] = (scores["Kafka"][0] + 10, scores["Kafka"][1] + ["Per-partition ordering"])

    # Data model
    if req.data_model == "event_log":
        scores["Kafka"] = (scores["Kafka"][0] + 25, scores["Kafka"][1] + ["Purpose-built for event logs"])
    elif req.data_model == "key_value":
        scores["DynamoDB"] = (scores["DynamoDB"][0] + 15, scores["DynamoDB"][1] + ["Purpose-built key-value"])

    # Multi-region
    if req.multi_region:
        scores["Spanner"] = (scores["Spanner"][0] + 15, scores["Spanner"][1] + ["Best-in-class multi-region"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 10, scores["CockroachDB"][1] + ["Multi-region support"])

    # Open source
    if req.open_source_required:
        scores["Kafka"] = (scores["Kafka"][0] + 10, scores["Kafka"][1] + ["Apache 2.0"])
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 10, scores["CockroachDB"][1] + ["Open source core"])
        scores["Spanner"] = (scores["Spanner"][0] - 15, scores["Spanner"][1] + ["Proprietary (Cloud only)"])
        scores["DynamoDB"] = (scores["DynamoDB"][0] - 15, scores["DynamoDB"][1] + ["Proprietary (AWS only)"])

    # Latency
    if req.latency_p99_ms < 10:
        scores["DynamoDB"] = (scores["DynamoDB"][0] + 10, scores["DynamoDB"][1] + ["Single-digit ms P99"])
        scores["Spanner"] = (scores["Spanner"][0] - 5, scores["Spanner"][1] + ["Commit wait adds latency"])

    # Budget
    if req.budget == "low":
        scores["CockroachDB"] = (scores["CockroachDB"][0] + 5, scores["CockroachDB"][1] + ["Self-hosted option"])
        scores["Kafka"] = (scores["Kafka"][0] + 5, scores["Kafka"][1] + ["Self-hosted option"])
        scores["Spanner"] = (scores["Spanner"][0] - 10, scores["Spanner"][1] + ["Expensive"])

    # Format results
    results = [
        (name, max(0, min(100, score)), "; ".join(reasons))
        for name, (score, reasons) in scores.items()
    ]
    results.sort(key=lambda x: -x[1])
    return results


def demo_decision_framework():
    """Demonstrate the decision framework with sample workloads."""

    workloads = [
        ("E-commerce order management", WorkloadRequirements(
            needs_sql=True, needs_transactions=True,
            consistency="strong", multi_region=True,
            data_model="relational", budget="high",
        )),
        ("Real-time event streaming", WorkloadRequirements(
            needs_sql=False, needs_transactions=False,
            consistency="eventual", throughput_qps=500000,
            data_model="event_log", open_source_required=True,
        )),
        ("User session store", WorkloadRequirements(
            needs_sql=False, needs_transactions=False,
            consistency="eventual", latency_p99_ms=5,
            data_model="key_value", budget="medium",
        )),
        ("Multi-region financial ledger", WorkloadRequirements(
            needs_sql=True, needs_transactions=True,
            consistency="strong", multi_region=True,
            data_model="relational", open_source_required=True,
        )),
    ]

    for name, req in workloads:
        print(f"\nWorkload: {name}")
        print("-" * 60)
        recommendations = recommend_system(req)
        for system, score, rationale in recommendations:
            print(f"  {system:<15} Score: {score:>5.0f}/100")
            if rationale:
                print(f"    Reasons: {rationale}")


if __name__ == "__main__":
    print("=== Dynamo Conflict Detection Demo ===")
    demo_dynamo_conflict()

    print("\n=== System Decision Framework ===")
    demo_decision_framework()
```

---

## 8. 요약

| 시스템 | 핵심 혁신 | 주요 트레이드오프 |
|---|---|---|
| **Spanner** | TrueTime이 글로벌 규모에서 external consistency를 가능하게 함 | 전용 하드웨어(GPS + 원자 시계) 필요; 높은 쓰기 지연 |
| **Dynamo** | Sloppy quorum과 충돌 감지로 항상 쓰기 가능 | 애플리케이션이 충돌을 처리해야 함; 약한 일관성 |
| **Kafka** | ISR 복제를 가진 분산 추가 전용 커밋 로그 | 범용 데이터베이스가 아님; 포인트 쿼리 없음 |
| **CockroachDB** | 전용 하드웨어 없이 직렬화 가능 SQL | 불확실성 재시작; 최종적 일관성 시스템보다 높은 지연 |

### 모든 시스템에 걸친 핵심 인사이트

1. **공짜 점심은 없다**: 모든 시스템은 무언가를 교환한다. Spanner는 일관성을 위해 비용과 지연을 교환한다. Dynamo는 가용성과 지연을 위해 일관성을 교환한다.

2. **파티션별 합의가 확장된다**: 강한 보장을 제공하는 모든 시스템은 전역 합의가 아닌 파티션별 합의를 사용한다.

3. **시간이 가장 어려운 문제이다**: 각 시스템의 시간 접근법(TrueTime, vector clock, HLC, 오프셋)이 근본적 특성을 정의한다.

4. **물리적 co-location이 중요하다**: Spanner의 인터리브 테이블, CockroachDB의 range, Kafka의 파티션 — 모두 데이터 지역성을 활용한다.

5. **운영 단순성은 가치가 있다**: Kafka의 ZooKeeper에서 KRaft로의 이전, DynamoDB의 서버리스 모델 — 운영 복잡성 감소가 기능이다.

---

[다음: 장애 감지와 멤버십](./13_Failure_Detection_and_Membership.md)
