# 레슨 9: 복제 전략

[개요](./00_Overview.md) | [이전: 분산 트랜잭션](./08_Distributed_Transactions.md) | [다음: CRDT와 최종 일관성](./10_CRDTs_and_Eventual_Consistency.md)

---

## 학습 목표

- 데이터 복제의 근본적인 동기 이해: 장애 허용, 지연 시간 감소, 처리량 확장
- Single-leader, multi-leader, leaderless 복제 아키텍처의 비교 및 트레이드오프 분석
- Quorum 기반 일관성 보장 분석 및 실패 시나리오 파악
- Multi-leader 및 leaderless 시스템의 충돌 감지 및 해결 전략 구현
- Chain replication을 quorum 기반 접근법의 대안으로 평가

---

## 1. 왜 데이터를 복제하는가?

복제(replication) — 동일한 데이터의 사본을 여러 머신에 유지하는 것 — 는 분산 시스템에서 가장 기본적인 기술 중 하나이다. 구체적인 전략을 살펴보기 전에, *왜* 복제하는지를 명확히 이해해야 한다.

### 1.1 장애 허용(Fault Tolerance)

하드웨어는 고장난다. 디스크가 손상되고, 서버가 충돌하고, 네트워크 링크가 끊기고, 데이터센터 전체가 정전된다. 복제 없이는 단일 디스크 장애로 데이터가 영구적으로 손실될 수 있다.

상용 디스크의 **평균 고장 간격(MTBF)**은 대략 3~5년이다. 10,000개 디스크로 구성된 클러스터에서는 *매일* 여러 건의 디스크 장애가 발생한다는 의미이다.

```
Single copy:
  P(data loss in 1 year) = P(disk failure) ≈ 20-33%

Three replicas on independent disks:
  P(data loss in 1 year) = P(all three fail) ≈ (0.25)^3 ≈ 1.5%
  (Much lower with prompt replacement: ~0.0001%)
```

산술은 명확하다: 복제 없이는 대규모 환경에서 데이터 손실은 불가피하다.

### 1.2 지연 시간 감소

물리학은 엄격한 제한을 부과한다. 광섬유에서 빛의 속도는 약 200,000 km/s이므로, 뉴욕에서 도쿄까지(편도 ~10,800 km) 왕복은 최소 ~108 ms가 소요된다. 여러 지리적 지역에 복제본을 배치하면 사용자가 가까운 사본에서 읽을 수 있다.

```
┌──────────┐         ┌──────────┐         ┌──────────┐
│ US-East  │◄───────►│ EU-West  │◄───────►│ AP-Tokyo │
│ Replica  │  ~70ms  │ Replica  │  ~120ms │ Replica  │
└──────────┘         └──────────┘         └──────────┘
     ▲                                         ▲
     │ <5ms                              <5ms  │
  US User                              JP User
```

### 1.3 처리량 확장(읽기 스케일링)

단일 서버가 초당 처리할 수 있는 읽기 쿼리 수는 유한하다. 읽기를 복제본으로 분산하면 읽기 처리량을 거의 선형적으로 수평 확장할 수 있다.

| 구성 | 읽기 처리량 | 쓰기 처리량 |
|---|---|---|
| 단일 노드 | 10,000 QPS | 10,000 QPS |
| 1 leader + 4 복제본 | ~50,000 QPS (읽기) | 10,000 QPS (쓰기) |
| 1 leader + 9 복제본 | ~100,000 QPS (읽기) | 10,000 QPS (쓰기) |

**핵심 트레이드오프**: 복제는 읽기 처리량을 향상시키지만 쓰기 처리량은 향상시키지 *않는다* (모든 쓰기는 모든 복제본에 적용되어야 한다). 쓰기 확장에는 *파티셔닝*이 필요하다 (레슨 11).

### 1.4 근본적인 트레이드오프

복제는 **복제본 간 동기화 유지**라는 문제를 만든다. 모든 복제 전략은 다음에 답해야 한다:

1. **쓰기가 어떻게 전파되는가?** 동기식인가 비동기식인가?
2. **충돌은 어떻게 처리되는가?** 두 복제본이 충돌하는 쓰기를 수락하면?
3. **클라이언트가 관찰하는 일관성 보장은 무엇인가?** 클라이언트가 오래된 데이터를 읽을 수 있는가?

이 레슨의 나머지 부분에서는 이러한 질문에 답하기 위한 주요 전략을 탐구한다.

---

## 2. Single-Leader 복제

Single-leader(primary-backup 또는 master-slave라고도 함) 복제는 가장 일반적인 전략이다. 하나의 지정된 노드 — **leader** — 가 모든 쓰기를 수락한다. Leader는 복제 스트림(write-ahead log, logical log, 또는 statement 기반 로그)을 **follower**(복제본)에 전송한다.

```
  Writes                 Reads
    │                   ┌──► Follower 1
    ▼                   │
┌────────┐   Replication│   ┌──────────┐
│ Leader │──────────────┼──►│Follower 2│
└────────┘              │   └──────────┘
    │                   │
    │                   └──► Follower 3
    ▼
  Reads
```

### 2.1 동기식 vs 비동기식 복제

핵심 설계 선택은 leader가 클라이언트에게 쓰기를 확인하기 전에 follower의 확인을 기다리는지 여부이다.

#### 동기식 복제

```
Client        Leader        Follower
  │───write───►│               │
  │            │───replicate──►│
  │            │◄──────ack─────│
  │◄───ack─────│               │
```

**보장**: leader가 쓰기를 확인하면, follower가 내구성 있는 사본을 가지고 있다.

**문제**: follower가 느리거나 접근 불가능하면, leader가 *차단*된다. 쓰기 지연 시간이 *가장 느린* follower의 지연 시간이 된다.

```python
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReplicaState:
    """Simulates a follower replica with variable latency."""
    name: str
    is_available: bool = True
    latency_ms: float = 5.0

    def apply_write(self, key: str, value: str) -> bool:
        """Simulate applying a write. Returns success/failure."""
        if not self.is_available:
            return False
        time.sleep(self.latency_ms / 1000.0)
        return True


def synchronous_write(leader_data: dict, replicas: list[ReplicaState],
                      key: str, value: str, timeout_s: float = 5.0) -> bool:
    """
    Synchronous replication: wait for ALL followers before acknowledging.
    Blocked if any follower is slow or down.
    """
    # Step 1: Apply to leader
    leader_data[key] = value

    # Step 2: Wait for ALL followers
    for replica in replicas:
        start = time.time()
        success = replica.apply_write(key, value)
        elapsed = time.time() - start

        if not success or elapsed > timeout_s:
            # Synchronous replication fails if any follower is unreachable
            print(f"  [SYNC] Follower {replica.name} failed — write BLOCKED")
            return False
        print(f"  [SYNC] Follower {replica.name} acked in {elapsed*1000:.1f}ms")

    return True
```

#### 비동기식 복제

```
Client        Leader        Follower
  │───write───►│               │
  │◄───ack─────│               │
  │            │───replicate──►│  (happens later)
  │            │◄──────ack─────│
```

**보장**: leader가 즉시 확인한다. 쓰기가 빠르다.

**문제**: 복제가 완료되기 전에 leader가 충돌하면, 확인된 쓰기가 **손실**된다. Follower가 오래된 데이터를 제공할 수 있다.

```python
import threading
import queue
from typing import Any


class AsyncReplicationLog:
    """Asynchronous replication via a background thread."""

    def __init__(self, replicas: list[ReplicaState]):
        self.replicas = replicas
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._worker = threading.Thread(target=self._replication_loop, daemon=True)
        self._worker.start()

    def enqueue_write(self, key: str, value: str) -> None:
        """Enqueue a write for asynchronous replication. Returns immediately."""
        self._queue.put((key, value))

    def _replication_loop(self) -> None:
        """Background loop: drain the queue and replicate to followers."""
        while True:
            key, value = self._queue.get()
            for replica in self.replicas:
                try:
                    replica.apply_write(key, value)
                except Exception as e:
                    # In practice: retry with exponential backoff
                    print(f"  [ASYNC] Replication to {replica.name} failed: {e}")
            self._queue.task_done()


def async_write(leader_data: dict, repl_log: AsyncReplicationLog,
                key: str, value: str) -> bool:
    """
    Asynchronous replication: leader acks immediately.
    Replication happens in background — fast but risks data loss.
    """
    leader_data[key] = value
    repl_log.enqueue_write(key, value)  # non-blocking
    return True  # ack to client before followers confirm
```

### 2.2 반동기식 복제(Semi-Synchronous Replication)

MySQL, PostgreSQL 등에서 사용하는 실용적인 절충안: 하나의 follower는 동기식이고, 나머지는 비동기식이다. 이는 확인 전에 쓰기가 최소 두 노드에 있음을 보장한다.

```
Client        Leader       Sync Follower    Async Followers
  │───write───►│               │                  │
  │            │───replicate──►│                  │
  │            │◄──────ack─────│                  │
  │◄───ack─────│               │                  │
  │            │───replicate──────────────────────►│ (background)
```

동기식 follower를 사용할 수 없게 되면, 다른 follower가 동기식 역할로 승격된다. 이를 때때로 **반동기식(semi-synchronous)** 복제라고 한다.

```python
def semi_sync_write(leader_data: dict, sync_replica: ReplicaState,
                    async_replicas: list[ReplicaState],
                    repl_log: AsyncReplicationLog,
                    key: str, value: str) -> bool:
    """
    Semi-synchronous: wait for ONE sync follower, then ack.
    Remaining followers replicate asynchronously.
    """
    # Step 1: Apply to leader
    leader_data[key] = value

    # Step 2: Synchronously replicate to the designated sync follower
    success = sync_replica.apply_write(key, value)
    if not success:
        # Failover: promote an async follower to sync role
        for candidate in async_replicas:
            if candidate.apply_write(key, value):
                print(f"  [SEMI] Promoted {candidate.name} to sync role")
                break
        else:
            return False  # all followers unreachable

    # Step 3: Asynchronously replicate to remaining followers
    for replica in async_replicas:
        repl_log.enqueue_write(key, value)

    return True
```

### 2.3 복제 로그 형식

Leader는 *무엇이 변경되었는지*를 follower에게 전달해야 한다. 세 가지 접근 방식이 있다:

| 형식 | 설명 | 장점 | 단점 |
|---|---|---|---|
| **Statement 기반** | SQL/명령어 전송 (예: `UPDATE users SET name='X' WHERE id=1`) | 단순하고 작은 크기 | 비결정적 함수 (NOW(), RAND()), auto-increment, 트리거 |
| **Write-ahead log (WAL) 전송** | 물리적 WAL 바이트 전송 | 정확한 바이트 수준 복제 | 스토리지 엔진 버전에 결합; 교차 버전 복제 불가 |
| **논리적 (row 기반) 로그** | 논리적 변경 전송: "행 X가 A에서 B로 변경됨" | 버전 독립적, 외부 시스템에서 소비 가능 | 대량 작업 시 WAL보다 큰 크기 |

현대 시스템(PostgreSQL 논리적 복제, MySQL binlog row 형식, MongoDB oplog)은 압도적으로 논리적 로그를 사용한다.

### 2.4 복제 지연과 그 영향

비동기식 복제에서 follower는 leader보다 뒤처질 수 있다. 이는 클라이언트에게 보이는 여러 일관성 이상 현상을 만든다.

#### Read-Your-Writes 위반

사용자가 데이터를 쓰고, 아직 업데이트를 받지 못한 follower에서 다시 읽는다.

```
Time ──────────────────────────────────────►

Client:    WRITE x=42 ──────── READ x ──► sees x=OLD!
                │                   │
Leader:    x=42 applied             │
                │                   │
Follower:  ─────────── lag ────── x=OLD still
```

**해결**: Read-your-writes 일관성. 사용자 자신의 읽기를 leader로 라우팅하거나, 사용자의 최신 쓰기 타임스탬프를 추적하여 따라잡은 follower에서만 읽는다.

#### 단조 읽기(Monotonic Read) 위반

사용자가 follower A(따라잡음)에서 읽은 다음, follower B(지연 중)에서 읽으면 데이터가 *시간 역행*한다.

```
Time ──────────────────────────────────────►

Client:    READ x ──► x=42     READ x ──► x=17 (older!)
                │                   │
Follower A: x=42 (caught up)       │
Follower B: ──────────────────── x=17 (lagging)
```

**해결**: 단조 읽기(Monotonic reads). 각 사용자의 읽기를 단일 follower에 고정(세션 스티키니스)하거나, 사용자의 최근 관찰 버전을 추적한다.

#### 인과적 순서(Causal Ordering) 위반

사용자 A가 질문을 쓰고, 사용자 B가 답변을 쓴다. Follower가 질문보다 답변을 먼저 받으면 의미가 통하지 않는다.

```
Leader:    Q posted at t=1    A posted at t=2
           │                  │
Follower:  A posted at t=2   Q posted at t=1  ← wrong order!
```

**해결**: 인과적 일관성(Causal consistency). 이벤트 B가 이벤트 A에 의존하면, 모든 복제본이 A를 B보다 먼저 보도록 보장한다. 이를 위해 인과적 의존성 추적(vector clock, version vector)이 필요하다.

### 2.5 Leader 장애: 페일오버

Leader가 충돌하면 follower가 승격되어야 한다. 이 과정 — **페일오버(failover)** — 는 분산 시스템에서 가장 위험한 작업 중 하나이다.

#### 페일오버 단계

```
1. Detect leader failure (timeout-based heartbeat)
2. Choose a new leader (most up-to-date follower, or by election)
3. Reconfigure clients to send writes to the new leader
4. Old leader must recognize it is no longer leader when it recovers
```

#### Split-Brain 문제

이전 leader가 교체된 사실을 모른 채 다시 온라인에 복귀하면, *두 노드가 동시에 쓰기를 수락*한다. 이를 **split-brain**이라 하며 데이터 분기를 초래한다.

```
                  Network partition
                       │
┌───────────┐     ┌────┼────────┐
│ Old Leader│     │    │        │
│ (thinks   │     │ New Leader  │
│  it's     │     │ (elected)   │
│  leader)  │     │             │
└───────────┘     └─────────────┘
  Writes ▲              ▲ Writes
         │              │
   Client A         Client B
         (DATA DIVERGES!)
```

**방지 메커니즘**:

| 메커니즘 | 작동 방식 |
|---|---|
| **Fencing token** | 새 leader가 단조 증가 토큰을 받는다. 이전 토큰의 쓰기는 거부된다. |
| **STONITH** (Shoot The Other Node In The Head) | 새 leader를 승격하기 전에 이전 leader의 전원을 물리적으로 차단한다. |
| **Epoch 번호** | 각 leader 선출 시 epoch가 증가한다. Follower는 오래된 epoch의 쓰기를 거부한다. |
| **Lease 기반 리더십** | Leader가 주기적으로 시간 제한 lease를 갱신해야 한다. 실패하면 다른 노드가 lease를 획득한다. |

```python
import time
from dataclasses import dataclass


@dataclass
class FencingToken:
    """Monotonically increasing token to prevent split-brain."""
    epoch: int
    leader_id: str


class FencedLeaderNode:
    """A leader node that uses fencing tokens to prevent split-brain."""

    def __init__(self, node_id: str, epoch: int):
        self.node_id = node_id
        self.epoch = epoch
        self.data: dict[str, str] = {}
        self.is_leader = True

    def write(self, key: str, value: str, token: FencingToken) -> bool:
        """Accept write only if the fencing token matches current epoch."""
        if token.epoch < self.epoch:
            print(f"  [FENCE] Rejected write from epoch {token.epoch} "
                  f"(current: {self.epoch})")
            return False
        if token.leader_id != self.node_id:
            print(f"  [FENCE] Rejected write from {token.leader_id} "
                  f"(current leader: {self.node_id})")
            return False
        self.data[key] = value
        return True

    def step_down(self) -> None:
        """Recognize that this node is no longer the leader."""
        self.is_leader = False
        print(f"  [FENCE] Node {self.node_id} stepped down from epoch {self.epoch}")


def simulate_failover():
    """Demonstrate fencing tokens preventing split-brain."""
    old_leader = FencedLeaderNode("node-1", epoch=1)
    old_token = FencingToken(epoch=1, leader_id="node-1")

    # Old leader accepts writes in epoch 1
    assert old_leader.write("x", "100", old_token)

    # Failover: new leader elected with epoch 2
    new_leader = FencedLeaderNode("node-2", epoch=2)
    new_token = FencingToken(epoch=2, leader_id="node-2")

    # New leader accepts writes
    assert new_leader.write("x", "200", new_token)

    # Old leader comes back — tries to write with old token
    # A properly fenced follower/storage rejects it
    assert not new_leader.write("x", "300", old_token)  # REJECTED

    old_leader.step_down()
```

---

## 3. Multi-Leader 복제

Multi-leader(multi-master 또는 active-active라고도 함) 복제에서는 둘 이상의 노드가 쓰기를 수락한다. 각 leader는 자신의 쓰기를 다른 모든 leader에 복제한다.

### 3.1 사용 사례

Multi-leader 복제가 적합한 경우:

| 사용 사례 | Multi-Leader를 사용하는 이유 |
|---|---|
| **멀티 데이터센터** | 각 DC가 로컬 leader를 보유한다. 쓰기가 빠르고(로컬), DC 간 비동기 복제된다. |
| **오프라인 가능 클라이언트** | 각 장치가 오프라인일 때 "leader"이다. 연결 복구 시 동기화한다 (CouchDB, PouchDB). |
| **실시간 협업 편집** | 각 사용자의 클라이언트가 leader로 동작한다. 변경 사항이 비동기적으로 병합된다 (레슨 10의 CRDT가 선호되는 경우가 많다). |

### 3.2 복제 토폴로지

Leader들이 서로에게 쓰기를 어떻게 전파하는가?

```
All-to-All              Circular              Star (Hub-and-Spoke)

  A ◄──► B              A ──► B              A ──► H ◄── B
  ▲  ╲╱  ▲              ▲     │                    │
  │  ╱╲  │              │     ▼                    ▼
  C ◄──► D              D ◄── C              C ◄── H ──► D
```

| 토폴로지 | 장애 허용 | 지연 시간 | 복잡도 |
|---|---|---|---|
| **All-to-all** | 높음 (모든 노드가 장애 가능) | 가장 낮음 (직접 경로) | 순서 문제 (추월) |
| **Circular** | 낮음 (단일 장애로 링 파손) | 높음 (홉별) | 단순하지만 취약 |
| **Star** | 중간 (허브가 SPOF) | 중간 | 단순한 라우팅 |

All-to-all이 실제로 가장 일반적이지만, 순서 문제를 감지하고 해결하기 위해 version vector 또는 Lamport 타임스탬프가 필요하다.

### 3.3 충돌 감지

여러 leader가 동시에 쓰기를 수락하면 충돌은 불가피하다. 충돌은 두 leader가 독립적으로 동일한 데이터를 수정할 때 발생한다.

```
Time ──────────────────────────────────────►

Leader A:  SET x = "foo"   ──── replicate ──► Leader B
Leader B:  SET x = "bar"   ──── replicate ──► Leader A
                                                  │
           x = ??? (conflict!)                    │
```

**충돌은 언제 감지되는가?**

| 시점 | 메커니즘 | 의미 |
|---|---|---|
| **동기식** (쓰기 시점) | 모든 leader의 합의를 기다림 | Multi-leader의 목적에 어긋남 (WAN 지연에 차단) |
| **비동기식** (복제 시점) | 복제된 쓰기 도착 시 감지 | 사후 해결 필요; 두 쓰기 모두 이미 확인됨 |

실제로는 비동기식 감지가 거의 항상 사용된다. 핵심 질문은: 충돌을 어떻게 *해결*하는가?

### 3.4 충돌 해결 전략

#### Last-Writer-Wins (LWW)

각 쓰기에 타임스탬프를 할당한다. 가장 높은 타임스탬프를 가진 쓰기가 승리하고, 나머지는 조용히 버려진다.

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class TimestampedValue:
    value: Any
    timestamp: float  # wall clock or logical clock
    writer_id: str


def lww_resolve(existing: TimestampedValue, incoming: TimestampedValue) -> TimestampedValue:
    """Last-Writer-Wins: higher timestamp wins. Tie-break on writer_id."""
    if incoming.timestamp > existing.timestamp:
        return incoming
    if incoming.timestamp == existing.timestamp:
        # Deterministic tie-breaking: higher writer_id wins
        return incoming if incoming.writer_id > existing.writer_id else existing
    return existing
```

**문제**: LWW는 *쓰기를 조용히 버린다*. 두 사용자가 동일한 레코드의 서로 다른 필드를 동시에 업데이트하면, 하나의 업데이트가 손실된다. 이것은 많은 프로덕션 버그의 원인이다.

#### 병합 함수

하나의 쓰기를 버리는 대신 두 쓰기를 병합한다. 이는 애플리케이션에 따라 다르다.

```python
def merge_shopping_carts(cart_a: set[str], cart_b: set[str]) -> set[str]:
    """Merge two shopping carts by taking the union of items."""
    return cart_a | cart_b


def merge_counters(count_a: int, origin_a: int,
                   count_b: int, origin_b: int) -> int:
    """
    Merge two counters that diverged from a common origin.
    Each counter records its starting point (origin).
    Merged value = origin + (delta_a) + (delta_b).
    """
    delta_a = count_a - origin_a
    delta_b = count_b - origin_b
    return origin_a + delta_a + delta_b
```

#### 커스텀 애플리케이션 수준 해결

모든 충돌 버전을 보존하고 애플리케이션(또는 사용자)이 해결하게 한다. Amazon의 장바구니가 유명한 이 접근법을 사용했다.

```python
@dataclass
class ConflictRecord:
    key: str
    conflicting_values: list[TimestampedValue]
    resolved: bool = False
    resolution: Any = None

    def resolve(self, strategy: str = "manual") -> Any:
        if strategy == "union":
            # For set-like values: merge via union
            merged = set()
            for v in self.conflicting_values:
                if isinstance(v.value, (set, list)):
                    merged.update(v.value)
            self.resolution = merged
        elif strategy == "lww":
            self.resolution = max(
                self.conflicting_values, key=lambda v: v.timestamp
            ).value
        elif strategy == "manual":
            # Present all versions to user
            print(f"Conflict on key '{self.key}':")
            for i, v in enumerate(self.conflicting_values):
                print(f"  [{i}] {v.value} (from {v.writer_id} at {v.timestamp})")
            # In production: return all versions to client for resolution
            self.resolution = self.conflicting_values
        self.resolved = True
        return self.resolution
```

### 3.5 Multi-Leader 복제 구현

```python
import time
import threading
from collections import defaultdict
from typing import Optional


class MultiLeaderNode:
    """A node in a multi-leader replication system."""

    def __init__(self, node_id: str, peers: Optional[list] = None):
        self.node_id = node_id
        self.data: dict[str, TimestampedValue] = {}
        self.peers: list['MultiLeaderNode'] = peers or []
        self.replication_log: list[tuple[str, TimestampedValue]] = []
        self._lock = threading.Lock()

    def local_write(self, key: str, value: Any) -> TimestampedValue:
        """Accept a write locally and enqueue for replication."""
        ts_value = TimestampedValue(
            value=value,
            timestamp=time.time(),
            writer_id=self.node_id,
        )
        with self._lock:
            self.data[key] = ts_value
            self.replication_log.append((key, ts_value))
        return ts_value

    def receive_replication(self, key: str, incoming: TimestampedValue) -> None:
        """Receive a replicated write from a peer. Resolve conflicts via LWW."""
        with self._lock:
            existing = self.data.get(key)
            if existing is None:
                self.data[key] = incoming
            else:
                self.data[key] = lww_resolve(existing, incoming)

    def replicate_to_peers(self) -> None:
        """Send all pending replication log entries to all peers."""
        with self._lock:
            pending = list(self.replication_log)
            self.replication_log.clear()
        for key, ts_value in pending:
            for peer in self.peers:
                peer.receive_replication(key, ts_value)

    def read(self, key: str) -> Optional[Any]:
        with self._lock:
            tv = self.data.get(key)
            return tv.value if tv else None


def demo_multi_leader():
    """Demonstrate multi-leader replication with conflict resolution."""
    node_a = MultiLeaderNode("A")
    node_b = MultiLeaderNode("B")
    node_a.peers = [node_b]
    node_b.peers = [node_a]

    # Concurrent writes to the same key
    node_a.local_write("user:1:name", "Alice")
    node_b.local_write("user:1:name", "Alicia")

    # Before replication: each node has its own version
    print(f"Before sync — A: {node_a.read('user:1:name')}, "
          f"B: {node_b.read('user:1:name')}")

    # Replicate (LWW resolution)
    node_a.replicate_to_peers()
    node_b.replicate_to_peers()

    # After replication: both converge (LWW picks higher timestamp)
    print(f"After sync  — A: {node_a.read('user:1:name')}, "
          f"B: {node_b.read('user:1:name')}")
```

---

## 4. Leaderless 복제 (Dynamo 스타일)

Leaderless 복제에서는 지정된 leader가 없다. 모든 노드가 읽기와 쓰기를 수락할 수 있다. 일관성은 **quorum** 프로토콜을 통해 달성된다.

이 접근법은 Amazon의 Dynamo 논문(2007)에 의해 대중화되었으며, Cassandra, Riak, Voldemort에서 사용된다.

### 4.1 읽기 및 쓰기 Quorum

**N**개의 복제본이 주어졌을 때:
- **W**: 쓰기를 확인해야 하는 복제본 수
- **R**: 읽기에 응답해야 하는 복제본 수

**Quorum 조건**: **W + R > N**이면, 모든 읽기가 최신 쓰기를 가진 복제본을 최소 하나 포함하는 것이 보장된다.

```
    N = 3 replicas

    Write quorum W = 2:          Read quorum R = 2:
    ┌─────────────┐              ┌─────────────┐
    │ Replica 1 ✓ │              │ Replica 1 ✓ │
    │ Replica 2 ✓ │              │ Replica 2   │
    │ Replica 3   │              │ Replica 3 ✓ │
    └─────────────┘              └─────────────┘

    W + R = 2 + 2 = 4 > 3 = N  ✓
    Overlap guaranteed: at least one replica has the latest write
```

#### 수학적 기반

Quorum 조건은 쓰기 집합과 읽기 집합 간의 **교집합**이 비어있지 않음을 보장한다:

$$|W \cap R| \geq W + R - N > 0 \quad \text{when} \quad W + R > N$$

일반적인 구성:

| N | W | R | 특성 |
|---|---|---|---|
| 3 | 2 | 2 | 표준 quorum. 읽기와 쓰기 모두에서 1개 장애 허용. |
| 3 | 3 | 1 | 쓰기 집중: 빠른 읽기, 하지만 어떤 장애든 쓰기를 차단. |
| 3 | 1 | 3 | 읽기 집중: 빠른 쓰기, 하지만 어떤 장애든 읽기를 차단. |
| 5 | 3 | 3 | 2개 장애 허용. 지연 시간 비용으로 더 높은 가용성. |

### 4.2 Quorum 읽기와 쓰기: 구현

```python
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VersionedValue:
    """A value tagged with a version (timestamp) for comparison."""
    value: Any
    version: int  # logical clock / version number
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other: 'VersionedValue') -> bool:
        return self.version < other.version


class DynamoStyleNode:
    """Simulates a node in a leaderless (Dynamo-style) system."""

    def __init__(self, node_id: str, is_available: bool = True):
        self.node_id = node_id
        self.store: dict[str, VersionedValue] = {}
        self.is_available = is_available
        self.hinted_handoff: list[tuple[str, str, VersionedValue]] = []

    def put(self, key: str, value: VersionedValue) -> bool:
        """Write a value to local storage."""
        if not self.is_available:
            return False
        existing = self.store.get(key)
        if existing is None or value.version > existing.version:
            self.store[key] = value
        return True

    def get(self, key: str) -> Optional[VersionedValue]:
        """Read a value from local storage."""
        if not self.is_available:
            return None
        return self.store.get(key)


class QuorumCoordinator:
    """
    Coordinator for quorum-based reads and writes.
    Implements W + R > N consistency with read repair.
    """

    def __init__(self, nodes: list[DynamoStyleNode], n: int, w: int, r: int):
        self.nodes = nodes
        self.n = n
        self.w = w
        self.r = r
        self._version_counter = 0

        assert w + r > n, f"Quorum condition violated: W({w}) + R({r}) <= N({n})"
        assert len(nodes) >= n, f"Need at least N={n} nodes"

    def _select_replicas(self, key: str) -> list[DynamoStyleNode]:
        """
        Select N replica nodes for a key using consistent hashing.
        (Simplified: use hash to pick starting position, then take N consecutive.)
        """
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        start = h % len(self.nodes)
        replicas = []
        for i in range(self.n):
            replicas.append(self.nodes[(start + i) % len(self.nodes)])
        return replicas

    def write(self, key: str, value: Any) -> bool:
        """
        Quorum write: send to N replicas, wait for W acknowledgments.
        Returns True if quorum is met.
        """
        self._version_counter += 1
        versioned = VersionedValue(value=value, version=self._version_counter)

        replicas = self._select_replicas(key)
        acks = 0
        for replica in replicas:
            if replica.put(key, versioned):
                acks += 1

        if acks >= self.w:
            print(f"  [WRITE] key={key}, value={value}, version={versioned.version}, "
                  f"acks={acks}/{self.n} (need {self.w})")
            return True
        else:
            print(f"  [WRITE FAILED] key={key}, acks={acks}/{self.n} "
                  f"(need {self.w})")
            return False

    def read(self, key: str) -> Optional[Any]:
        """
        Quorum read: read from N replicas, wait for R responses.
        Return the value with the highest version. Trigger read repair.
        """
        replicas = self._select_replicas(key)
        responses: list[tuple[DynamoStyleNode, Optional[VersionedValue]]] = []

        for replica in replicas:
            val = replica.get(key)
            responses.append((replica, val))

        # Filter successful responses
        successful = [(node, val) for node, val in responses if val is not None]

        if len(successful) < self.r:
            print(f"  [READ FAILED] key={key}, responses={len(successful)}/{self.n} "
                  f"(need {self.r})")
            return None

        # Find the most recent value
        best_node, best_val = max(successful, key=lambda x: x[1].version)

        # Read repair: send the latest value to nodes with stale data
        for node, val in responses:
            if val is None or val.version < best_val.version:
                node.put(key, best_val)  # read repair
                print(f"  [READ REPAIR] Updating {node.node_id} "
                      f"from v{val.version if val else 0} to v{best_val.version}")

        print(f"  [READ] key={key}, value={best_val.value}, "
              f"version={best_val.version}")
        return best_val.value


def demo_quorum():
    """Demonstrate quorum reads and writes with read repair."""
    nodes = [DynamoStyleNode(f"node-{i}") for i in range(5)]
    coord = QuorumCoordinator(nodes, n=3, w=2, r=2)

    # Normal write and read
    coord.write("user:42", {"name": "Alice", "email": "alice@example.com"})
    coord.read("user:42")

    # Simulate a node failure during write, then read repair
    print("\n--- Simulating node failure ---")
    nodes[1].is_available = False
    coord.write("user:42", {"name": "Alice Updated", "email": "alice@new.com"})

    # Node comes back — read repair will update it
    nodes[1].is_available = True
    coord.read("user:42")  # triggers read repair on node-1
```

### 4.3 Sloppy Quorum과 Hinted Handoff

**엄격한 quorum(strict quorum)**은 키에 대해 지정된 N개 복제본에서 W와 R 확인을 요구한다. 하지만 해당 N개 노드 중 여러 개가 다운되면? 클러스터에 다른 건강한 노드가 있음에도 쓰기가 실패한다.

**Sloppy quorum**은 지정된 N개만이 아닌 클러스터의 *아무* W개 노드에서 쓰기를 수락할 수 있게 한다. 지정된 노드가 복구되면, 임시 보유자가 데이터를 전달한다 — 이것이 **hinted handoff**이다.

```
Normal:
  Key "x" → Replicas {A, B, C}
  Write to A, B, C — W=2 → OK if 2 ack

Sloppy (C is down):
  Key "x" → Replicas {A, B, C}  but C is down
  Write to A, B, D — W=2 → OK (D holds hint for C)

  When C recovers:
    D → C: "Here's the write I held for you"  (hinted handoff)
    D deletes its temporary copy
```

```python
@dataclass
class HintedWrite:
    """A write held temporarily for an unavailable node."""
    target_node_id: str
    key: str
    value: VersionedValue
    hint_holder_id: str


class SloppyQuorumCoordinator(QuorumCoordinator):
    """Extends quorum coordinator with sloppy quorum and hinted handoff."""

    def __init__(self, nodes: list[DynamoStyleNode], n: int, w: int, r: int):
        super().__init__(nodes, n, w, r)
        self.hints: list[HintedWrite] = []

    def write_sloppy(self, key: str, value: Any) -> bool:
        """
        Sloppy quorum write: if designated replicas are down,
        write to other nodes and create hints for handoff.
        """
        self._version_counter += 1
        versioned = VersionedValue(value=value, version=self._version_counter)

        designated = self._select_replicas(key)
        all_nodes = list(self.nodes)
        random.shuffle(all_nodes)

        acks = 0
        for replica in designated:
            if replica.put(key, versioned):
                acks += 1

        # If we haven't reached quorum from designated nodes,
        # try other nodes (sloppy quorum)
        if acks < self.w:
            for node in all_nodes:
                if node not in designated and node.is_available:
                    node.put(key, versioned)
                    acks += 1
                    # Record hint for the unavailable designated node
                    down_nodes = [n for n in designated if not n.is_available]
                    if down_nodes:
                        self.hints.append(HintedWrite(
                            target_node_id=down_nodes[0].node_id,
                            key=key,
                            value=versioned,
                            hint_holder_id=node.node_id,
                        ))
                    if acks >= self.w:
                        break

        return acks >= self.w

    def process_hinted_handoff(self) -> int:
        """Deliver hinted writes to nodes that have recovered."""
        delivered = 0
        remaining = []
        for hint in self.hints:
            target = next(
                (n for n in self.nodes if n.node_id == hint.target_node_id), None
            )
            if target and target.is_available:
                target.put(hint.key, hint.value)
                delivered += 1
                print(f"  [HANDOFF] Delivered {hint.key} v{hint.value.version} "
                      f"from {hint.hint_holder_id} to {hint.target_node_id}")
            else:
                remaining.append(hint)
        self.hints = remaining
        return delivered
```

**중요한 주의사항**: Sloppy quorum은 W + R > N 겹침 속성을 **보장하지 않는다**. Sloppy quorum 쓰기가 지정된 집합 외부의 노드로 갈 수 있으므로, 이후의 엄격한 quorum 읽기가 이를 놓칠 수 있다. Sloppy quorum은 일관성보다 가용성을 우선시한다.

### 4.4 Merkle 트리를 이용한 Anti-Entropy

Read repair는 키가 읽힐 때만 오래된 데이터를 수정한다. 드물게 읽히는 데이터의 경우, 백그라운드 프로세스 — **anti-entropy** — 가 능동적으로 복제본을 동기화한다.

단순한 접근법(모든 키-값 쌍 비교)은 O(전체 데이터)이며, 대용량 데이터셋에는 비실용적이다. **Merkle 트리**(해시 트리)는 효율적인 비교를 가능하게 한다.

```
Level 0 (root):      H(H12 + H34)
                      ╱         ╲
Level 1:          H12            H34
                 ╱   ╲          ╱   ╲
Level 2:       H1     H2     H3     H4
               │      │      │      │
Data:        key1   key2   key3   key4
```

두 노드가 Merkle 트리를 상향식으로 비교한다:
1. 루트 해시가 일치하면 → 모든 데이터가 동일 → 완료
2. 루트가 다르면 → 자식으로 내려감
3. 차이나는 리프 범위가 식별될 때까지 반복
4. 차이나는 범위만 동기화

**복잡도**: 차이를 식별하는 데 O(log N) 비교, 무차별 대입은 O(N).

```python
import hashlib
from typing import Optional


class MerkleNode:
    """Node in a Merkle tree for anti-entropy synchronization."""

    def __init__(self, range_start: str, range_end: str):
        self.range_start = range_start
        self.range_end = range_end
        self.hash_value: str = ""
        self.left: Optional['MerkleNode'] = None
        self.right: Optional['MerkleNode'] = None
        self.keys: list[tuple[str, str]] = []  # leaf nodes store (key, value) pairs

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def compute_hash(self) -> str:
        """Compute hash bottom-up."""
        if self.is_leaf():
            data = "".join(f"{k}:{v}" for k, v in sorted(self.keys))
            self.hash_value = hashlib.sha256(data.encode()).hexdigest()[:16]
        else:
            left_hash = self.left.compute_hash() if self.left else ""
            right_hash = self.right.compute_hash() if self.right else ""
            combined = left_hash + right_hash
            self.hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return self.hash_value


def find_differences(node_a: MerkleNode, node_b: MerkleNode) -> list[str]:
    """
    Compare two Merkle trees and return the key ranges that differ.
    Only descends into subtrees where hashes don't match.
    """
    if node_a.hash_value == node_b.hash_value:
        return []  # subtrees are identical

    if node_a.is_leaf() or node_b.is_leaf():
        return [f"[{node_a.range_start}, {node_a.range_end}]"]

    diffs = []
    if node_a.left and node_b.left:
        diffs.extend(find_differences(node_a.left, node_b.left))
    if node_a.right and node_b.right:
        diffs.extend(find_differences(node_a.right, node_b.right))
    return diffs
```

### 4.5 Quorum 일관성이 실패하는 경우

W + R > N이라도, quorum 일관성은 실제로 위반될 수 있다:

| 시나리오 | 실패 이유 |
|---|---|
| **Sloppy quorum** | 쓰기가 비지정 노드로 감; 읽기가 이를 보지 못함 |
| **동시 쓰기** | 유사한 타임스탬프의 두 쓰기가 동일 키에; 복제본마다 순서가 다를 수 있음 |
| **동시 읽기와 쓰기** | 어떤 복제본이 응답하느냐에 따라 읽기가 이전 또는 새 값을 볼 수 있음 |
| **실패한 쓰기 롤백** | 쓰기가 W-1개 복제본에 도달한 후 코디네이터 장애; 일부 복제본에 부분 쓰기 지속 |
| **LWW에서의 클럭 스큐** | 버전 관리에 벽시계 타임스탬프를 사용하면, 클럭 스큐로 새로운 쓰기가 덮어쓰일 수 있음 |

**핵심 인사이트**: Dynamo 스타일 quorum은 **확률적** 일관성을 제공하며, 강한 일관성이 아니다. 강한 일관성을 위해서는 합의 프로토콜이 필요하다 (레슨 5-6).

---

## 5. Chain Replication

Chain replication은 quorum 기반 접근법의 대안으로, 읽기 집중 워크로드에 대해 높은 처리량과 **강한 일관성**을 제공한다.

### 5.1 기본 Chain Replication

노드는 체인으로 구성된다. 쓰기는 **head**에서 들어와 체인을 따라 전파된다. 읽기는 **tail**이 처리한다.

```
            Write                                         Read
              │                                            ▲
              ▼                                            │
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ Head │───►│Node 2│───►│Node 3│───►│Node 4│───►│ Tail │
└──────┘    └──────┘    └──────┘    └──────┘    └──────┘
              │           │           │           │
         Write is confirmed only when it reaches the tail
```

**특성**:
- **강한 일관성**: tail이 모든 커밋된 쓰기의 전체 순서를 가짐
- **쓰기 지연**: 모든 노드 간 지연의 합 (quorum처럼 병렬이 아님)
- **읽기 지연**: 단일 노드 (tail)
- **읽기 처리량**: 단일 노드(tail)로 제한

```python
from collections import OrderedDict
from typing import Optional, Any


class ChainNode:
    """A node in a chain replication system."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.data: dict[str, Any] = {}
        self.successor: Optional['ChainNode'] = None
        self.pending_writes: OrderedDict[int, tuple[str, Any]] = OrderedDict()
        self._write_seq = 0

    @property
    def is_head(self) -> bool:
        return True  # set by chain manager

    @property
    def is_tail(self) -> bool:
        return self.successor is None

    def handle_write(self, key: str, value: Any, seq: Optional[int] = None) -> bool:
        """
        Handle a write request.
        Head: create sequence number, forward to successor.
        Middle: apply and forward.
        Tail: apply and acknowledge.
        """
        if seq is None:
            # Head node: assign sequence number
            self._write_seq += 1
            seq = self._write_seq

        # Apply locally
        self.data[key] = value

        if self.is_tail:
            # Tail: write is committed
            return True
        else:
            # Forward to successor
            return self.successor.handle_write(key, value, seq)

    def handle_read(self, key: str) -> Optional[Any]:
        """
        Only the tail serves reads (guarantees strong consistency).
        """
        if not self.is_tail:
            raise RuntimeError("Reads must be served by the tail node")
        return self.data.get(key)


def build_chain(node_ids: list[str]) -> tuple[ChainNode, ChainNode]:
    """Build a chain of nodes. Returns (head, tail)."""
    nodes = [ChainNode(nid) for nid in node_ids]
    for i in range(len(nodes) - 1):
        nodes[i].successor = nodes[i + 1]
    return nodes[0], nodes[-1]


def demo_chain_replication():
    """Demonstrate basic chain replication."""
    head, tail = build_chain(["H", "M1", "M2", "T"])

    # Writes enter at head, propagate through chain
    head.handle_write("account:1", {"balance": 1000})
    head.handle_write("account:2", {"balance": 500})

    # Reads served only by tail (strong consistency)
    print(f"Read account:1 = {tail.handle_read('account:1')}")
    print(f"Read account:2 = {tail.handle_read('account:2')}")
```

### 5.2 CRAQ: 분산 쿼리가 가능한 Chain Replication

기본 chain replication의 병목은 tail이 모든 읽기를 처리한다는 것이다. **CRAQ** (Chain Replication with Apportioned Queries)는 체인의 *아무* 노드에서도 읽기를 허용하여, 체인 길이에 비례해 읽기 처리량을 선형적으로 향상시킨다.

**핵심 아이디어**: 각 노드가 각 키의 여러 버전을 저장한다. 노드의 최신 버전이 커밋된(tail) 버전과 같으면 키는 **clean**이다. Clean 키는 아무 노드에서나 제공할 수 있다. Dirty 키는 tail 확인이 필요하다.

```
CRAQ Node State for key "x":

  Node has versions:  [v1: "old", v2: "new"]
  Tail committed:     v1

  → Key is DIRTY (v2 not yet committed)
  → Read must check with tail

  After tail commits v2:
  → Key is CLEAN
  → Any node can serve the read
```

```python
class CRAQNode:
    """A node in a CRAQ (Chain Replication with Apportioned Queries) system."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.versions: dict[str, list[tuple[int, Any]]] = {}  # key -> [(version, value)]
        self.committed_version: dict[str, int] = {}  # key -> latest committed version
        self.successor: Optional['CRAQNode'] = None
        self.predecessor: Optional['CRAQNode'] = None
        self._seq = 0

    @property
    def is_tail(self) -> bool:
        return self.successor is None

    @property
    def is_head(self) -> bool:
        return self.predecessor is None

    def handle_write(self, key: str, value: Any, version: Optional[int] = None) -> bool:
        """Write propagates head to tail. Tail sends commit back up the chain."""
        if version is None:
            self._seq += 1
            version = self._seq

        # Store new version
        if key not in self.versions:
            self.versions[key] = []
        self.versions[key].append((version, value))

        if self.is_tail:
            # Tail: commit and propagate commit backward
            self.committed_version[key] = version
            self._clean_old_versions(key, version)
            self._propagate_commit_backward(key, version)
            return True
        else:
            return self.successor.handle_write(key, value, version)

    def _propagate_commit_backward(self, key: str, version: int) -> None:
        """Send commit notification up the chain."""
        if self.predecessor:
            self.predecessor._receive_commit(key, version)

    def _receive_commit(self, key: str, version: int) -> None:
        """Process commit notification: clean old versions, propagate backward."""
        self.committed_version[key] = version
        self._clean_old_versions(key, version)
        if self.predecessor:
            self.predecessor._receive_commit(key, version)

    def _clean_old_versions(self, key: str, committed: int) -> None:
        """Remove versions older than the committed version."""
        if key in self.versions:
            self.versions[key] = [
                (v, val) for v, val in self.versions[key] if v >= committed
            ]

    def read(self, key: str) -> Optional[Any]:
        """
        CRAQ read: serve locally if key is clean, otherwise check tail.
        A key is clean if only one version exists (the committed one).
        """
        if key not in self.versions:
            return None

        versions = self.versions[key]
        if len(versions) == 1:
            # Clean: only committed version exists — serve locally
            return versions[0][1]
        else:
            # Dirty: multiple versions — must check tail for committed version
            return self._check_tail_version(key)

    def _check_tail_version(self, key: str) -> Optional[Any]:
        """Ask the tail for the committed version of a key."""
        node = self
        while not node.is_tail:
            node = node.successor
        committed_ver = node.committed_version.get(key)
        if committed_ver is None:
            return None
        # Return the committed version from local store
        for v, val in self.versions.get(key, []):
            if v == committed_ver:
                return val
        return None
```

### 5.3 복제 접근법 비교

| 특성 | Single-Leader | Multi-Leader | Leaderless (Quorum) | Chain Replication |
|---|---|---|---|---|
| **일관성** | 강함 (동기) 또는 최종적 (비동기) | 최종적 (충돌) | 확률적 | 강함 |
| **쓰기 지연** | 낮음 (leader만) | 낮음 (로컬 leader) | W 복제본 병렬 | 체인 홉의 합 |
| **읽기 지연** | Leader 또는 follower | 아무 leader | R 복제본 병렬 | Tail만 (또는 CRAQ) |
| **읽기 처리량** | Follower로 확장 | Leader로 확장 | N으로 확장 | CRAQ: 체인으로 확장 |
| **쓰기 처리량** | 단일 leader 병목 | 다중 leader | 아무 노드 | Head 병목 |
| **장애 허용** | Leader 장애 시 페일오버 필요 | 개별 leader 장애 허용 | N-W 쓰기 또는 N-R 읽기 장애 허용 | Head/tail 장애 시 재구성 필요 |
| **충돌 처리** | 충돌 없음 (단일 writer) | 충돌 해결 필요 | 버전 충돌 가능 | 충돌 없음 (tail에서 전체 순서) |
| **복잡도** | 낮음 | 높음 (충돌) | 중간 | 낮음-중간 |

---

## 6. 종합: 포괄적 Quorum 시뮬레이터

다음 구현은 이 레슨의 많은 개념을 구성 가능한 시뮬레이터로 통합한다.

```python
"""
Comprehensive quorum-based replication simulator.
Demonstrates configurable W, R, N with failure injection,
read repair, and consistency measurement.
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ConsistencyLevel(Enum):
    ONE = 1
    QUORUM = 2
    ALL = 3


@dataclass
class WriteResult:
    success: bool
    acks: int
    required: int
    version: int
    key: str
    value: Any


@dataclass
class ReadResult:
    success: bool
    value: Any
    version: int
    responses: int
    required: int
    stale_replicas_repaired: int


@dataclass
class Replica:
    """A storage replica with failure simulation."""
    replica_id: str
    store: dict[str, tuple[Any, int]] = field(default_factory=dict)
    is_available: bool = True
    latency_ms: float = 1.0
    failure_rate: float = 0.0  # probability of transient failure per operation

    def put(self, key: str, value: Any, version: int) -> bool:
        if not self.is_available:
            return False
        if random.random() < self.failure_rate:
            return False
        existing = self.store.get(key)
        if existing is None or version > existing[1]:
            self.store[key] = (value, version)
        return True

    def get(self, key: str) -> Optional[tuple[Any, int]]:
        if not self.is_available:
            return None
        if random.random() < self.failure_rate:
            return None
        return self.store.get(key)


class ReplicationSimulator:
    """
    Configurable quorum-based replication simulator.
    Supports strict and sloppy quorums, read repair, and consistency measurement.
    """

    def __init__(self, num_replicas: int = 5, n: int = 3, w: int = 2, r: int = 2,
                 sloppy_quorum: bool = False):
        self.replicas = [Replica(f"R{i}") for i in range(num_replicas)]
        self.n = n
        self.w = w
        self.r = r
        self.sloppy_quorum = sloppy_quorum
        self._version = 0
        self._write_log: list[WriteResult] = []
        self._read_log: list[ReadResult] = []

    def _get_preference_list(self, key: str) -> list[Replica]:
        """Determine the N preferred replicas for a key."""
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        start = h % len(self.replicas)
        result = []
        for i in range(len(self.replicas)):
            idx = (start + i) % len(self.replicas)
            result.append(self.replicas[idx])
            if len(result) == self.n:
                break
        return result

    def write(self, key: str, value: Any,
              consistency: ConsistencyLevel = ConsistencyLevel.QUORUM) -> WriteResult:
        """Write with configurable consistency level."""
        self._version += 1
        version = self._version

        required = {
            ConsistencyLevel.ONE: 1,
            ConsistencyLevel.QUORUM: self.w,
            ConsistencyLevel.ALL: self.n,
        }[consistency]

        preferred = self._get_preference_list(key)
        acks = 0

        # Try preferred replicas first
        for replica in preferred:
            if replica.put(key, value, version):
                acks += 1

        # Sloppy quorum: try non-preferred replicas if needed
        if self.sloppy_quorum and acks < required:
            others = [r for r in self.replicas if r not in preferred]
            for replica in others:
                if replica.put(key, value, version):
                    acks += 1
                    if acks >= required:
                        break

        result = WriteResult(
            success=acks >= required,
            acks=acks,
            required=required,
            version=version,
            key=key,
            value=value,
        )
        self._write_log.append(result)
        return result

    def read(self, key: str,
             consistency: ConsistencyLevel = ConsistencyLevel.QUORUM) -> ReadResult:
        """Read with configurable consistency level and read repair."""
        required = {
            ConsistencyLevel.ONE: 1,
            ConsistencyLevel.QUORUM: self.r,
            ConsistencyLevel.ALL: self.n,
        }[consistency]

        preferred = self._get_preference_list(key)
        responses: list[tuple[Replica, Any, int]] = []

        for replica in preferred:
            result = replica.get(key)
            if result is not None:
                responses.append((replica, result[0], result[1]))

        if len(responses) < required:
            return ReadResult(False, None, -1, len(responses), required, 0)

        # Find the latest version
        best_replica, best_value, best_version = max(responses, key=lambda x: x[2])

        # Read repair: update stale replicas
        repaired = 0
        for replica, value, version in responses:
            if version < best_version:
                replica.put(key, best_value, best_version)
                repaired += 1

        result = ReadResult(
            success=True,
            value=best_value,
            version=best_version,
            responses=len(responses),
            required=required,
            stale_replicas_repaired=repaired,
        )
        self._read_log.append(result)
        return result

    def measure_consistency(self, num_operations: int = 1000) -> dict:
        """
        Measure consistency by performing concurrent writes and reads.
        Returns statistics on stale reads and read repair frequency.
        """
        stale_reads = 0
        total_reads = 0
        repairs = 0

        for i in range(num_operations):
            key = f"key-{random.randint(0, 99)}"

            if random.random() < 0.5:
                # Write
                self.write(key, f"value-{i}")
            else:
                # Read
                result = self.read(key)
                if result.success:
                    total_reads += 1
                    repairs += result.stale_replicas_repaired
                    # Check if we got the latest version
                    latest = self._get_latest_version(key)
                    if result.version < latest:
                        stale_reads += 1

        return {
            "total_reads": total_reads,
            "stale_reads": stale_reads,
            "stale_read_rate": stale_reads / max(total_reads, 1),
            "total_repairs": repairs,
            "repair_rate": repairs / max(total_reads, 1),
        }

    def _get_latest_version(self, key: str) -> int:
        """Get the maximum version of a key across all replicas."""
        max_version = 0
        for replica in self.replicas:
            result = replica.store.get(key)
            if result and result[1] > max_version:
                max_version = result[1]
        return max_version


def run_consistency_experiment():
    """Compare consistency under different configurations."""
    configs = [
        {"n": 3, "w": 2, "r": 2, "sloppy": False, "label": "Strict W=2 R=2"},
        {"n": 3, "w": 1, "r": 1, "sloppy": False, "label": "Strict W=1 R=1"},
        {"n": 3, "w": 3, "r": 1, "sloppy": False, "label": "Strict W=3 R=1"},
        {"n": 3, "w": 2, "r": 2, "sloppy": True,  "label": "Sloppy W=2 R=2"},
    ]

    print(f"{'Configuration':<25} {'Stale Rate':>12} {'Repair Rate':>12}")
    print("-" * 51)

    for cfg in configs:
        sim = ReplicationSimulator(
            num_replicas=5, n=cfg["n"], w=cfg["w"], r=cfg["r"],
            sloppy_quorum=cfg["sloppy"],
        )
        # Inject some failures
        sim.replicas[2].failure_rate = 0.1
        stats = sim.measure_consistency(2000)
        print(f"{cfg['label']:<25} {stats['stale_read_rate']:>11.2%} "
              f"{stats['repair_rate']:>11.2%}")


if __name__ == "__main__":
    print("=== Quorum Demo ===")
    demo_quorum()

    print("\n=== Chain Replication Demo ===")
    demo_chain_replication()

    print("\n=== Consistency Experiment ===")
    run_consistency_experiment()
```

---

## 7. 요약

| 개념 | 핵심 내용 |
|---|---|
| **Single-leader** | 단순하고 강한 일관성 가능하지만, leader가 병목이며 페일오버가 위험함 |
| **Multi-leader** | 지역 간 낮은 쓰기 지연, 하지만 충돌이 불가피하고 해결이 어려움 |
| **Leaderless (quorum)** | W + R > N을 통한 높은 가용성, 하지만 확률적 일관성만 제공 |
| **Chain replication** | CRAQ를 통한 높은 읽기 처리량의 강한 일관성, 하지만 쓰기 지연은 체인의 합 |
| **복제 지연** | Read-your-writes, 단조 읽기, 인과적 순서 위반을 초래 |
| **충돌 해결** | LWW는 단순하지만 데이터를 손실; 병합 함수와 애플리케이션 수준 해결이 더 안전 |
| **Anti-entropy** | Merkle 트리가 백그라운드 동기화를 위해 분기된 데이터를 효율적으로 식별 |

### 복제 전략 선택 시 핵심 질문

1. **애플리케이션에 어떤 일관성이 필요한가?** 강함 → single-leader 또는 chain. 최종적 → multi-leader 또는 leaderless.
2. **읽기:쓰기 비율은?** 읽기 집중 → chain 또는 leaderless. 쓰기 집중 → multi-leader.
3. **지리적 분산이 필요한가?** 예 → multi-leader 또는 leaderless.
4. **장애 허용 요구 사항은?** 높음 → leaderless (조절 가능한 W, R, N).
5. **애플리케이션이 충돌을 해결할 수 있는가?** 예 → multi-leader 또는 커스텀 병합의 leaderless. 아니오 → single-leader.

---

[다음: CRDT와 최종 일관성](./10_CRDTs_and_Eventual_Consistency.md)
