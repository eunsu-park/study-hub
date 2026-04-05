# Lesson 14: 분산 조정 프리미티브

[Overview](./00_Overview.md) | [이전: 장애 감지와 멤버십](./13_Failure_Detection_and_Membership.md) | [다음: TLA+를 이용한 형식 검증](./15_Formal_Verification_TLAplus.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. fencing token을 활용한 분산 잠금 메커니즘을 설계하고 정확성을 평가한다
2. Chubby, ZooKeeper, Redlock 접근 방식과 각각의 트레이드오프를 분석한다
3. 여러 전략(bully, ring, consensus 기반)을 사용하여 leader election을 구현한다
4. Snowflake 알고리즘을 사용하여 전역적으로 고유하고 시간 정렬 가능한 식별자를 생성한다
5. 서비스 디스커버리 아키텍처를 비교하고 주어진 시스템에 적합한 패턴을 선택한다

---

## 목차

1. [조정 프리미티브가 중요한 이유](#1-조정-프리미티브가-중요한-이유)
2. [분산 잠금](#2-분산-잠금)
3. [Fencing Token](#3-fencing-token)
4. [분산 배리어](#4-분산-배리어)
5. [Leader Election 패턴](#5-leader-election-패턴)
6. [시퀀스 번호와 순서 지정](#6-시퀀스-번호와-순서-지정)
7. [서비스 디스커버리](#7-서비스-디스커버리)
8. [구현: Fencing이 있는 분산 잠금](#8-구현-fencing이-있는-분산-잠금)
9. [구현: Snowflake ID 생성기](#9-구현-snowflake-id-생성기)
10. [요약 및 추가 읽을거리](#10-요약-및-추가-읽을거리)

---

## 1. 조정 프리미티브가 중요한 이유

분산 시스템은 협력해야 하는 독립적인 프로세스들로 구성됩니다. 조정 프리미티브 없이는 상호 배제, 순서화된 연산, 또는 일관된 설정을 달성하는 것이 불가능하거나, 에지 케이스에서 필연적으로 실패하는 임시방편적 솔루션을 필요로 합니다.

### 1.1 조정 프리미티브 구성 요소

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Coordination Primitives                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Mutual       │  │ Ordering     │  │ Discovery    │              │
│  │ Exclusion    │  │              │  │              │              │
│  │              │  │              │  │              │              │
│  │ • Locks      │  │ • Sequence   │  │ • Service    │              │
│  │ • Barriers   │  │   numbers    │  │   registry   │              │
│  │ • Leader     │  │ • Barriers   │  │ • Config     │              │
│  │   election   │  │ • Total      │  │   management │              │
│  │              │  │   ordering   │  │ • DNS        │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  Built on: Consensus (Paxos/Raft/ZAB) or Probabilistic Guarantees  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 올바른 조정을 위한 요구사항

모든 조정 프리미티브는 다음을 다루어야 합니다:

| 요구사항 | 설명 |
|------------|-------------|
| **Safety** | 나쁜 일은 절대 발생하지 않음 (예: 두 클라이언트가 동일한 잠금을 보유) |
| **Liveness** | 좋은 일은 결국 발생함 (예: 잠금이 결국 부여됨) |
| **장애 허용** | 노드/네트워크 장애에도 프리미티브가 작동함 |
| **성능** | 사용 사례에 대해 오버헤드가 허용 범위 내임 |

**관찰**: FLP 불가능성(Lesson 3)에 의해 비동기 시스템에서 이 모든 것을 완벽하게 가질 수는 없습니다. 모든 조정 프리미티브는 트레이드오프를 만듭니다.

---

## 2. 분산 잠금

### 2.1 요구사항

분산 잠금은 다음을 만족해야 합니다:

1. **상호 배제**: 어느 시점에서든 최대 하나의 클라이언트만 잠금을 보유
2. **데드락 방지**: 잠금을 보유한 채 클라이언트가 크래시하면 잠금이 결국 해제됨
3. **장애 허용**: 장애에도 잠금 서비스가 계속 기능함

### 2.2 Chubby (Google)

Chubby는 Paxos consensus 위에 구축된 Google의 분산 잠금 서비스입니다. Mike Burrows가 2006년에 발표했습니다.

**아키텍처**:

```
┌──────────────────────────────────────────────┐
│                Chubby Cell                    │
│                                              │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│  │Server│  │Server│  │Server│  │Server│  │Server│
│  │  1   │  │  2   │  │  3   │  │  4   │  │  5   │
│  │      │  │(MASTER)│ │      │  │      │  │      │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
│     │         │         │         │         │     │
│     └─────────┴────┬────┴─────────┴─────────┘     │
│                    │ Paxos                         │
│                    │ Consensus                     │
└────────────────────┼──────────────────────────────┘
                     │
            ┌────────┴────────┐
            │  Client Library │
            │  (Caching,      │
            │   KeepAlive)    │
            └─────────────────┘
```

**주요 설계 결정**:

| 기능 | 설계 선택 | 근거 |
|---------|--------------|-----------|
| 잠금 세분성 | 거친 단위(Coarse-grained) | 잠금이 수시간/수일 동안 유지됨 (예: master election) |
| 잠금 지연 | 잠금 해제 후 1분 지연 | 크래시된 클라이언트의 진행 중인 작업이 완료되도록 허용 |
| Sequencer | 불투명한 바이트 문자열 | 클라이언트가 fencing을 위해 리소스 서버에 sequencer를 전달 |
| 캐싱 | 클라이언트 라이브러리가 파일 데이터를 캐시 | Chubby master가 KeepAlive를 통해 무효화 |
| KeepAlive | 주기적 하트비트 (기본 12초) | 세션 유지, 클라이언트 장애 감지 |
| 세션 | 임시 리스(Ephemeral lease) | 세션 타임아웃 시 모든 잠금 해제 |

**잠금 지연 메커니즘**:

```
Timeline:
  Client A acquires lock ─────────── Client A crashes ──── Lock delay (60s) ──── Client B acquires
       │                                    │                                          │
       └── Operations using lock ───────────┘                                          │
                                                                                       │
  During lock delay:                                                                   │
  • No new client can acquire                                                          │
  • Client A's in-flight operations complete                                           │
  • Resource servers reject stale sequencers                                            │
```

### 2.3 ZooKeeper 레시피

ZooKeeper는 잠금, 배리어, election으로 구성할 수 있는 저수준 프리미티브를 제공합니다.

**잠금을 위한 Ephemeral Sequential Znode**:

```
Lock path: /locks/my-resource

Client A creates: /locks/my-resource/lock-0000000001 (ephemeral, sequential)
Client B creates: /locks/my-resource/lock-0000000002 (ephemeral, sequential)
Client C creates: /locks/my-resource/lock-0000000003 (ephemeral, sequential)

Rule: Client with the lowest sequence number holds the lock.
      Others watch the znode just before them.

Client A holds lock (lowest: 0001)
Client B watches 0001 (will be notified if A's znode deleted)
Client C watches 0002 (will be notified if B's znode deleted)
```

**왜 잠금 znode가 아닌 이전 노드를 감시하는가?**

이는 **herd effect**를 방지합니다: N개의 클라이언트가 모두 잠금 znode를 감시하면, 잠금이 해제될 때 N개의 클라이언트 모두가 동시에 깨어나지만 하나만 획득할 수 있습니다. 이전 노드 감시를 사용하면 하나의 클라이언트만 깨어납니다 — 다음 순서에 있는 클라이언트입니다.

```python
# Pseudocode for ZooKeeper distributed lock
class ZooKeeperLock:
    def __init__(self, zk_client, lock_path):
        self.zk = zk_client
        self.lock_path = lock_path
        self.my_znode = None

    def acquire(self):
        # Create ephemeral sequential znode
        self.my_znode = self.zk.create(
            f"{self.lock_path}/lock-",
            ephemeral=True,
            sequential=True,
        )

        while True:
            # Get all children sorted by sequence number
            children = sorted(self.zk.get_children(self.lock_path))

            my_seq = self.my_znode.split("-")[-1]
            my_index = next(
                i for i, c in enumerate(children)
                if c.split("-")[-1] == my_seq
            )

            if my_index == 0:
                # We have the lowest sequence number: lock acquired!
                return True

            # Watch the predecessor
            predecessor = children[my_index - 1]
            predecessor_path = f"{self.lock_path}/{predecessor}"

            # Block until predecessor is deleted
            event = self.zk.exists(predecessor_path, watch=True)
            if event is not None:
                # Predecessor still exists: wait for watch notification
                self._wait_for_event()

    def release(self):
        if self.my_znode:
            self.zk.delete(self.my_znode)
            self.my_znode = None
```

**ZooKeeper를 이용한 읽기-쓰기 잠금**:

```
Write lock: same as exclusive lock above
Read lock:
  1. Create ephemeral sequential znode: /locks/resource/read-NNNN
  2. Get all children
  3. If no WRITE znode has a lower sequence number → read lock acquired
  4. Otherwise, watch the highest WRITE znode with lower sequence number

This allows:
  - Multiple concurrent readers
  - Writers block until all prior readers complete
  - Readers block until prior writer completes
```

### 2.4 Redlock (Redis)

Redlock은 Salvatore Sanfilippo(antirez)가 여러 독립적인 Redis 인스턴스를 사용하여 분산 잠금을 수행하기 위해 제안한 알고리즘입니다.

**알고리즘**:

```
Setup: N Redis masters (typically N=5), no replication

Step 1: Client gets current time T1

Step 2: Client tries to acquire lock on all N instances:
   SET resource_name my_random_value NX PX 30000
   (set if not exists, with 30 second expiry)

Step 3: Client gets current time T2

Step 4: Lock is acquired if and only if:
   a) Lock was acquired on at least N/2 + 1 instances (majority)
   b) Total elapsed time (T2 - T1) < lock validity time
   c) Remaining validity = lock_ttl - (T2 - T1)

Step 5: If lock not acquired, release on all instances

┌──────────┐
│ Client   │
│          │─── SET NX ──▶ Redis 1  ✓
│          │─── SET NX ──▶ Redis 2  ✓
│          │─── SET NX ──▶ Redis 3  ✓  ← majority (3/5)
│          │─── SET NX ──▶ Redis 4  ✗  (already locked)
│          │─── SET NX ──▶ Redis 5  ✓
│          │
│ Lock acquired (4/5 > 3 = N/2+1)
└──────────┘
```

### 2.5 Redlock 논쟁

**Martin Kleppmann의 비판** ("How to do distributed locking", 2016):

```
Problem 1: GC Pauses

Client A                    Redis            Resource Server
   │                          │                      │
   │── acquire lock ──────▶  │                      │
   │◀── lock granted ──────  │                      │
   │                          │                      │
   │  ┌── GC PAUSE ──────┐  │                      │
   │  │  (30+ seconds)    │  │── lock expires ────▶│
   │  │                    │  │                      │
   │  │  Client B          │  │◀── acquires lock ──│
   │  │  acquires lock     │  │── writes data ─────▶│
   │  └───────────────────┘  │                      │
   │                          │                      │
   │── writes data ─────────────────────────────────▶│  ← UNSAFE!
   │  (believes it still holds the lock)             │
   │                                                  │

Two clients write concurrently → mutual exclusion violated!
```

```
Problem 2: Clock Skew

Redis 1:  time = 10:00:00 ──── lock expires at 10:00:30
Redis 2:  time = 10:00:00 ──── lock expires at 10:00:30
Redis 3:  time = 10:00:05 ──── lock expires at 10:00:35  ← clock ahead
Redis 4:  time = 09:59:55 ──── lock expires at 10:00:25  ← clock behind
Redis 5:  time = 10:00:00 ──── lock expires at 10:00:30

Locks expire at different real times due to clock skew.
A second client could acquire a majority before the first client's
locks have all expired.
```

**Kleppmann의 해결책**: fencing token 사용 (섹션 3 참조).

**Antirez의 반론**:

| Kleppmann의 주장 | Antirez의 반박 |
|-------------------|-------------------|
| GC 일시정지가 safety를 깨뜨림 | 모든 잠금 알고리즘(ZooKeeper 포함)에 이 문제가 있음. Redlock의 TTL이 문제를 제한함. |
| 클럭 스큐는 무한함 | 제한된 클럭 드리프트는 합리적인 가정임 (NTP + CLOCK_MONOTONIC) |
| Fencing token이 모든 것을 해결함 | fencing token이 있으면 잠금 자체도 필요 없음 |

**논쟁으로부터의 교훈**:

1. **분산 잠금만으로는 safety를 보장할 수 없음** — 항상 fencing이나 멱등 연산이 필요
2. **클럭 가정이 중요함** — 시스템이 클럭에 대해 무엇을 가정하는지 명시적으로 기술
3. **오류의 비용** — 잠금 위반이 데이터 손상을 일으키면, Redlock이 아닌 consensus 기반 잠금(ZooKeeper, etcd) 사용
4. **효율성 vs 정확성** — Redlock은 효율성(중복 작업 방지)에는 괜찮지만, 정확성(데이터 손상 방지)에는 부적합

---

## 3. Fencing Token

### 3.1 문제: 만료된 잠금 보유자

완벽한 잠금 서비스가 있더라도, 긴 일시정지(GC, 네트워크 지연, 페이지 폴트)를 경험한 클라이언트는 잠금이 만료되어 다른 클라이언트에게 부여된 후에도 계속 작업을 수행할 수 있습니다.

### 3.2 해결책: 단조 증가 토큰

```
Lock Service              Client A                Resource Server
     │                       │                           │
     │◀── acquire lock ─────│                           │
     │── token=33 ──────────▶│                           │
     │                       │── write(data, token=33) ─▶│
     │                       │                           │ Accepts: 33 > last_token(0)
     │                       │                           │ last_token = 33
     │                       │                           │
     │   (A pauses, lock expires)                        │
     │                       │                           │
     │◀── acquire lock ──── Client B                     │
     │── token=34 ──────────▶│                           │
     │                       │── write(data, token=34) ─▶│
     │                       │                           │ Accepts: 34 > 33
     │                       │                           │ last_token = 34
     │                       │                           │
     │            (A resumes from pause)                  │
     │                       │                           │
     │                 Client A                          │
     │                       │── write(data, token=33) ─▶│
     │                       │                           │ REJECTS: 33 < 34
     │                       │◀── error: stale token ───│
```

### 3.3 Fencing Token의 요구사항

1. **단조성**: 각 새 토큰은 모든 이전 토큰보다 엄격히 커야 함
2. **고유성**: 두 잠금 부여가 같은 토큰을 생성해서는 안 됨
3. **내구성**: 토큰 시퀀스가 잠금 서비스 크래시에서도 살아남아야 함

**다양한 시스템이 fencing token을 제공하는 방법**:

| 시스템 | Fencing Token 소스 | 단조성 보장 |
|--------|---------------------|----------------------|
| ZooKeeper | `czxid` (create transaction ID) | ZAB를 통한 전역 순서 보장 |
| etcd | Revision number | Raft를 통한 전역 순서 보장 |
| Chubby | Sequencer (불투명 문자열) | Paxos 로그 인덱스로 순서 보장 |
| Redlock | 내장되어 있지 않음 | 별도로 추가해야 함 (접근 방식의 약점) |

### 3.4 리소스 서버에서 Fencing 구현

리소스 서버는 다음을 해야 합니다:

1. 모든 요청과 함께 fencing token을 수신
2. 가장 높은 것보다 낮거나 같은 토큰의 요청을 거부
3. 가장 높은 토큰을 영속화 (크래시에서 살아남도록)

```python
class FencedResourceServer:
    """Resource server that enforces fencing tokens."""

    def __init__(self):
        self.data = {}
        self.highest_token = 0
        self._token_lock = threading.Lock()

    def write(self, key: str, value: str, fencing_token: int) -> bool:
        """
        Write a value, but only if the fencing token is valid.

        Args:
            key: The key to write
            value: The value to write
            fencing_token: Must be > highest previously seen token

        Returns:
            True if write succeeded, False if rejected (stale token)
        """
        with self._token_lock:
            if fencing_token <= self.highest_token:
                return False  # Stale token — reject
            self.highest_token = fencing_token
            self.data[key] = value
            return True

    def read(self, key: str) -> Optional[str]:
        """Read a value (no fencing needed for reads in simple case)."""
        return self.data.get(key)
```

---

## 4. 분산 배리어

### 4.1 단일 배리어

배리어는 조건이 충족될 때까지(예: 모든 프로세스가 도착할 때까지) 모든 프로세스를 차단합니다.

```
Process 1: ─────────── arrive ──── WAIT ──── proceed ─────▶
Process 2: ── arrive ──────────── WAIT ──── proceed ─────▶
Process 3: ────────────── arrive ─ WAIT ──── proceed ─────▶
                                    │
                            All 3 arrived:
                            barrier opens
```

### 4.2 이중 배리어

이중 배리어는 두 개의 동기화 지점을 가집니다:
1. **진입 배리어**: N개의 프로세스가 모두 진입할 때까지 차단
2. **종료 배리어**: N개의 프로세스가 모두 완료할 때까지 차단

```
                 Entry Barrier              Exit Barrier
Process 1: ──── enter ─── WAIT ── work ── done ─── WAIT ── leave ──▶
Process 2: ── enter ───── WAIT ── work ── done ─── WAIT ── leave ──▶
Process 3: ────── enter ─ WAIT ── work ── done ─── WAIT ── leave ──▶
                            │                        │
                    All entered                All finished
```

### 4.3 ZooKeeper 이중 배리어 구현

```python
class ZooKeeperDoubleBarrier:
    """
    Double barrier using ZooKeeper.

    Pseudocode for the ZooKeeper recipe.
    """

    def __init__(self, zk_client, barrier_path: str, num_processes: int):
        self.zk = zk_client
        self.barrier_path = barrier_path
        self.num_processes = num_processes
        self.my_node = None

    def enter(self, process_id: str) -> None:
        """
        Enter the barrier. Blocks until all processes have entered.
        """
        # Create ephemeral child node
        self.my_node = self.zk.create(
            f"{self.barrier_path}/{process_id}",
            ephemeral=True,
        )

        while True:
            children = self.zk.get_children(self.barrier_path)
            if len(children) >= self.num_processes:
                # Create "ready" node to signal all have arrived
                return
            else:
                # Watch for new children
                self.zk.get_children(self.barrier_path, watch=True)
                self._wait_for_event()

    def leave(self, process_id: str) -> None:
        """
        Leave the barrier. Blocks until all processes have left.
        """
        while True:
            children = sorted(self.zk.get_children(self.barrier_path))

            if len(children) == 0:
                return  # All have left

            if len(children) == 1 and children[0] == process_id:
                # We are the last one: delete and exit
                self.zk.delete(self.my_node)
                return

            if children[-1] == process_id:
                # We have the highest sequence: delete self, watch lowest
                self.zk.delete(self.my_node)
                lowest = f"{self.barrier_path}/{children[0]}"
                self.zk.exists(lowest, watch=True)
                self._wait_for_event()
            else:
                # Watch the highest sequence node
                highest = f"{self.barrier_path}/{children[-1]}"
                self.zk.exists(highest, watch=True)
                self._wait_for_event()

    def _wait_for_event(self):
        """Block until a watch fires (simplified)."""
        pass  # In real code, use threading.Event or asyncio
```

---

## 5. Leader Election 패턴

### 5.1 왜 Leader Election이 필요한가

많은 분산 알고리즘은 단일 leader를 갖는 것이 유리합니다:
- **Consensus 프로토콜**: Leader가 제안을 주도함 (Raft, Multi-Paxos)
- **데이터베이스 복제**: Primary가 쓰기를 수신함
- **조정**: 하나의 프로세스가 예약된 작업을 수행함
- **부하 분산**: 하나의 프로세스가 작업을 분배함

### 5.2 Bully 알고리즘 (Garcia-Molina, 1982)

```
Assumption: Each process has a unique numeric ID. Higher ID = higher priority.

Algorithm (process P detects coordinator failure):

1. P sends ELECTION to all processes with higher IDs
2. If no response within timeout → P declares itself coordinator
3. If P receives OK from any higher-ID process → P waits
4. The highest-ID process that responds eventually sends COORDINATOR to all

Example with processes {1, 2, 3, 4, 5}, process 5 is current coordinator:

Process 5 crashes.
Process 2 detects crash.

  2 ──ELECTION──▶ 3    ✓ OK
  2 ──ELECTION──▶ 4    ✓ OK
  2 ──ELECTION──▶ 5    ✗ no response

  3 ──ELECTION──▶ 4    ✓ OK
  3 ──ELECTION──▶ 5    ✗ no response

  4 ──ELECTION──▶ 5    ✗ no response

  4 sends COORDINATOR to {1, 2, 3}
  Process 4 is the new coordinator.
```

**속성**:

| 속성 | 값 |
|----------|-------|
| 메시지 복잡도 | 최악의 경우 O(n²) |
| 시간 복잡도 | 최악의 경우 O(n) 타임아웃 |
| 장애 허용 | 크래시 장애를 처리함 |
| 가정 | 프로세스 ID가 전체 순서를 가짐 |
| 약점 | 가장 높은 ID의 프로세스가 계속 크래시하고 복구되면 불안정 |

### 5.3 Ring 기반 Election (Chang-Roberts, 1979)

```
Processes arranged in a logical ring.

Process P starts election:
1. P sends ELECTION(P.id) to its successor
2. Each process forwards ELECTION(id) if id > own_id
3. If id < own_id, replace with ELECTION(own_id)
4. If id == own_id, this process is the leader: send COORDINATOR

Example: ring [3] → [1] → [4] → [2] → [3]

  3 starts election:
  3 ──ELECT(3)──▶ 1 ──ELECT(3)──▶ 4    (3 > 1, so forward 3)
                                   4 ──ELECT(4)──▶ 2   (4 > 3, so replace with 4)
                                                    2 ──ELECT(4)──▶ 3  (4 > 2)
                                                                     3 ──ELECT(4)──▶ 1 (4 > 3)
                                                                                      1 ──ELECT(4)──▶ 4
  4 receives ELECT(4): ID matches → 4 is leader
  4 sends COORDINATOR(4) around ring
```

**속성**:

| 속성 | 값 |
|----------|-------|
| 메시지 복잡도 | 최선의 경우 O(n), 최악의 경우 O(n²) |
| 가정 | 논리적 ring 토폴로지 |
| 장애 허용 | ring이 유지되어야 함; election 중 장애에 강건하지 않음 |

### 5.4 Consensus 기반 Election

현대 시스템은 강력한 보장을 제공하기 때문에 consensus 프로토콜을 leader election에 사용합니다:

**Raft leader election** (Lesson 6에서):

```
Terms:  ─── term 1 (leader: A) ───│─── term 2 (election) ───│─── term 3 (leader: C) ──

Node A: Leader ─── heartbeats ───── CRASH
Node B: Follower ─────────────────── timeout ── Candidate(term 2) ── loses ── Follower
Node C: Follower ─────────────────── timeout ── Candidate(term 2) ── WINS ─── Leader
Node D: Follower ─────────────────── votes for C ────────────────────────── Follower
Node E: Follower ─────────────────── votes for C ────────────────────────── Follower
```

**ZooKeeper ephemeral sequential znode를 이용한 election**:

```python
class ZooKeeperLeaderElection:
    """Leader election using ZooKeeper ephemeral sequential znodes."""

    def __init__(self, zk_client, election_path: str, on_elected, on_revoked):
        self.zk = zk_client
        self.election_path = election_path
        self.on_elected = on_elected      # Callback when elected
        self.on_revoked = on_revoked      # Callback when leadership lost
        self.my_znode = None
        self.is_leader = False

    def run_election(self) -> None:
        """Participate in leader election."""
        # Create ephemeral sequential znode
        self.my_znode = self.zk.create(
            f"{self.election_path}/candidate-",
            ephemeral=True,
            sequential=True,
        )
        self._check_leadership()

    def _check_leadership(self) -> None:
        """Check if this node is the leader."""
        children = sorted(self.zk.get_children(self.election_path))
        my_seq = self.my_znode.split("-")[-1]

        if children[0].split("-")[-1] == my_seq:
            # We have the lowest sequence number: we are leader
            if not self.is_leader:
                self.is_leader = True
                self.on_elected()
        else:
            # Watch predecessor (avoid herd effect)
            my_index = next(
                i for i, c in enumerate(children)
                if c.split("-")[-1] == my_seq
            )
            predecessor = children[my_index - 1]
            predecessor_path = f"{self.election_path}/{predecessor}"

            # Watch predecessor: when it goes away, re-check
            exists = self.zk.exists(predecessor_path, watch=True)
            if exists is None:
                # Predecessor already gone: re-check immediately
                self._check_leadership()
            # Otherwise, wait for watch notification, then re-check
```

### 5.5 Election 알고리즘 비교

| 알고리즘 | 메시지 수 | 장애 모델 | 보장 | 사용처 |
|-----------|----------|-------------|------------|---------|
| Bully | O(n²) | Crash-stop | 고유한 leader | 학술 |
| Ring | O(n)-O(n²) | election 중 장애 없음 | 고유한 leader | Token ring |
| Raft | O(n) | Crash-recovery | term당 최대 하나의 leader | etcd, CockroachDB |
| ZAB | O(n) | Crash-recovery | 고유한 leader | ZooKeeper |
| ZK ephemeral | O(n) | Crash-stop | 고유한 leader | 애플리케이션 레벨 |

---

## 6. 시퀀스 번호와 순서 지정

### 6.1 전역 순서의 필요성

많은 시스템은 전역적으로 고유하고 대략적으로 시간 순서가 지정된 식별자를 필요로 합니다:
- 데이터베이스 기본 키
- 분산 로그에서의 이벤트 순서 지정
- 분산 추적 (trace ID, span ID)
- 트랜잭션 ID

### 6.2 ID 생성 접근 방식

| 접근 방식 | 순서 | 고유성 | 조정 | 처리량 |
|----------|----------|------------|--------------|-----------|
| Auto-increment (단일 DB) | 전체 순서 | 보장됨 | 단일 지점 | 낮음 (~10K/s) |
| UUID v4 | 없음 | 확률적 | 없음 | 매우 높음 |
| UUID v7 | 대략적 시간 | 확률적 | 없음 | 매우 높음 |
| Snowflake | 시간 + 부분적 | 보장됨 | 머신 ID 할당 | 높음 (~4M/s/노드) |
| ULID | 시간 + 랜덤 | 확률적 | 없음 | 매우 높음 |
| Timestamp + counter | 시간 | 조정 필요 | 노드별 카운터 | 높음 |

### 6.3 Twitter Snowflake ID

Snowflake는 생성기 간 조정 없이 대규모로 대략적인 시간 순서의 고유 ID를 생성하기 위해 Twitter에서 설계되었습니다.

**비트 레이아웃** (총 64비트):

```
┌─────────────────────────────────────────────────────────────────┐
│ 0 │      41 bits: timestamp        │ 5 │ 5 │  12 bits:        │
│   │      (milliseconds since       │ DC│ W │  sequence         │
│   │       custom epoch)            │ ID│ ID│  number           │
└─────────────────────────────────────────────────────────────────┘

Bit 63:    Unused (sign bit, always 0)
Bits 62-22: Timestamp (41 bits → 2^41 ms ≈ 69.7 years)
Bits 21-17: Datacenter ID (5 bits → 32 datacenters)
Bits 16-12: Worker ID (5 bits → 32 workers per DC)
Bits 11-0:  Sequence number (12 bits → 4096 IDs per ms per worker)
```

**속성**:
- **시간 정렬 가능**: 나중에 생성된 ID가 더 높은 값을 가짐 (클럭 스큐 허용 범위 내에서)
- **조정 불필요**: 각 worker가 독립적으로 생성
- **높은 처리량**: 4096 IDs/ms/worker = ~4M IDs/s/worker
- **K-sortable**: 서로 몇 밀리초 이내의 ID는 시간 순서대로 정렬됨

**클럭 스큐 처리**: 시스템 클럭이 뒤로 이동하면, Snowflake는 클럭이 따라잡을 때까지 ID 생성을 거부해야 합니다 (중복 ID 방지).

### 6.4 ULID (Universally Unique Lexicographically Sortable Identifier)

```
 01AN4Z07BY      79KA1307SR9X4MV3

|----------|    |----------------|
 Timestamp       Randomness
  48 bits         80 bits
  (ms)

Total: 128 bits, encoded as 26-character Crockford Base32

Properties:
  - Lexicographically sortable (string comparison works)
  - 1.21e+24 unique ULIDs per millisecond
  - No coordination needed
  - Compatible with UUID storage (128 bits)
  - Monotonic option: within same ms, increment LSB of random part
```

### 6.5 UUID v7 (RFC 9562, 2024)

데이터베이스에 친화적인 시간 순서를 위해 설계된 최신 표준 UUID 형식입니다:

```
┌──────────────────────────────────────────────────┐
│ 48 bits: Unix timestamp (ms) │ 4 bits: version  │
│ 12 bits: rand_a              │ 2 bits: variant   │
│ 62 bits: rand_b                                  │
└──────────────────────────────────────────────────┘

Total: 128 bits
Monotonic within millisecond: optional (implementation-defined)
Database friendly: indexes well due to time-ordering
```

---

## 7. 서비스 디스커버리

### 7.1 서비스 디스커버리 문제

동적인 분산 시스템에서 서비스는 시작, 중지, 이동, 확장됩니다. 클라이언트는 통신이 필요한 서비스의 현재 주소를 어떻게 찾을까요?

```
Without service discovery:           With service discovery:

Client → hardcoded IP:port          Client → registry → dynamic IP:port
(breaks when service moves)          (resilient to changes)
```

### 7.2 DNS 기반 디스커버리

```
┌────────┐   DNS query: api.service.consul
│ Client │ ──────────────────────────────────▶ ┌────────────┐
│        │ ◀────────────────────────────────── │ DNS Server │
│        │   A records: 10.0.1.5, 10.0.1.6    │ (Consul)   │
└────────┘                                     └────────────┘

Advantages:
  - Universal: every language/platform supports DNS
  - No client library needed
  - Caching built into DNS protocol

Disadvantages:
  - TTL-based caching: stale results during transitions
  - No health checking at query time (relies on DNS updater)
  - Limited metadata (just IP + port)
  - DNS caching in OS/language runtimes hard to control
```

**시스템**: Consul DNS 인터페이스, AWS Route 53 (health check 포함), Kubernetes CoreDNS

### 7.3 KV 기반 디스커버리

```
┌────────┐   GET /v1/kv/services/api
│ Client │ ──────────────────────────────────▶ ┌──────────┐
│        │ ◀────────────────────────────────── │ etcd /   │
│        │   {"host": "10.0.1.5", "port": 8080,│ Consul / │
│        │    "version": "2.1", "weight": 100}  │ ZooKeeper│
└────────┘                                     └──────────┘

Advantages:
  - Rich metadata (version, weight, tags, health status)
  - Watch/subscribe for real-time updates
  - Strong consistency (consensus-backed)
  - Health checking with TTL-based sessions

Disadvantages:
  - Requires client library
  - Additional infrastructure to operate
  - More complex than DNS
```

### 7.4 Client-Side vs Server-Side 디스커버리

```
Client-Side Discovery:
┌────────┐     query     ┌──────────┐
│ Client │──────────────▶│ Registry │
│        │◀──────────────│          │
│        │  service list  └──────────┘
│        │
│        │──────────────▶ Service Instance A
│        │  (direct call,  (10.0.1.5:8080)
│        │   client picks)
└────────┘

  + No extra hop for requests
  + Client controls load balancing strategy
  - Client must implement discovery logic
  - Different languages need different implementations
  Examples: Netflix Eureka (client-side), gRPC name resolution


Server-Side Discovery:
┌────────┐              ┌───────────┐              ┌──────────┐
│ Client │─── request ─▶│ Load      │─── forward ─▶│ Service  │
│        │◀── response ──│ Balancer  │◀── response ──│ Instance │
└────────┘              │ / Gateway │              └──────────┘
                        └─────┬─────┘
                              │ query
                        ┌─────▼─────┐
                        │ Registry  │
                        └───────────┘

  + Client is simple (just call the load balancer)
  + Language-agnostic
  - Extra network hop
  - Load balancer can become bottleneck
  Examples: AWS ALB, Kubernetes kube-proxy, Nginx
```

### 7.5 서비스 메시

서비스 메시는 디스커버리와 통신 로직을 사이드카 프록시로 이동합니다:

```
┌────────────────────────────────┐   ┌────────────────────────────────┐
│ Pod A                          │   │ Pod B                          │
│ ┌──────────┐  ┌──────────────┐│   │┌──────────────┐  ┌──────────┐│
│ │ Service  │──│ Envoy Proxy  ││───▶││ Envoy Proxy  │──│ Service  ││
│ │ (app)    │  │ (sidecar)    ││   │ │ (sidecar)    │  │ (app)    ││
│ └──────────┘  └──────────────┘│   │└──────────────┘  └──────────┘│
└────────────────────────────────┘   └────────────────────────────────┘
                     │                          │
                     └──────────┬───────────────┘
                         ┌──────▼──────┐
                         │ Control     │
                         │ Plane       │
                         │ (Istio/     │
                         │  Linkerd)   │
                         └─────────────┘

The sidecar handles:
  - Service discovery
  - Load balancing
  - TLS termination
  - Retries, circuit breaking
  - Observability (metrics, tracing)
```

### 7.6 서비스 디스커버리 접근 방식 비교

| 측면 | DNS | KV Store | Client-Side | Server-Side | 서비스 메시 |
|--------|-----|----------|-------------|-------------|-------------|
| 복잡도 | 낮음 | 중간 | 중간 | 낮음 (클라이언트) | 높음 |
| 지연 | 캐시 의존 | 낮음 | 가장 낮음 | +1 홉 | +1 홉 |
| 메타데이터 | 제한적 | 풍부 | 풍부 | 제한적 | 풍부 |
| 헬스 체크 | 외부 | TTL 세션 | 클라이언트 관리 | LB 헬스 체크 | 사이드카 헬스 |
| 언어 지원 | 범용 | 클라이언트 라이브러리 | 언어별 | 범용 | 범용 |
| 일관성 | Eventual | Strong | Eventual | LB에 의존 | Eventual |
| 예시 | Consul DNS, Route 53 | etcd, Consul KV | Eureka, gRPC | AWS ALB, k8s | Istio, Linkerd |

---

## 8. 구현: Fencing이 있는 분산 잠금

```python
import time
import threading
import hashlib
import os
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class LockInfo:
    """Information about a held lock."""
    owner: str
    fencing_token: int
    acquired_at: float
    ttl: float  # Time-to-live in seconds

    @property
    def is_expired(self) -> bool:
        return time.monotonic() - self.acquired_at > self.ttl


class DistributedLockManager:
    """
    A distributed lock manager with fencing token support.

    This implementation simulates a consensus-backed lock service
    (similar to ZooKeeper or etcd). In production, the lock state
    would be replicated via Raft or Paxos.

    Features:
      - Mutual exclusion with TTL-based expiration
      - Monotonically increasing fencing tokens
      - Lock delay (Chubby-style) to drain in-flight operations
      - Deadlock freedom via TTL expiration
    """

    def __init__(self, lock_delay: float = 5.0):
        self._locks: Dict[str, LockInfo] = {}
        self._next_token: int = 1
        self._lock_delay_until: Dict[str, float] = {}
        self._lock_delay_duration = lock_delay
        self._mutex = threading.Lock()

    def acquire(
        self,
        resource: str,
        owner: str,
        ttl: float = 30.0,
        timeout: float = 10.0,
    ) -> Optional[int]:
        """
        Attempt to acquire a lock on a resource.

        Args:
            resource: The resource to lock
            owner: Unique identifier for the lock owner
            ttl: Lock time-to-live in seconds
            timeout: Maximum time to wait for lock acquisition

        Returns:
            Fencing token if acquired, None if timeout
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            with self._mutex:
                # Check lock delay
                if resource in self._lock_delay_until:
                    if time.monotonic() < self._lock_delay_until[resource]:
                        # Lock delay active: cannot acquire yet
                        time.sleep(0.1)
                        continue
                    else:
                        del self._lock_delay_until[resource]

                # Check if lock is free or expired
                current = self._locks.get(resource)
                if current is None or current.is_expired:
                    # Clean up expired lock (with lock delay)
                    if current is not None and current.is_expired:
                        self._lock_delay_until[resource] = (
                            time.monotonic() + self._lock_delay_duration
                        )
                        del self._locks[resource]
                        continue  # Must wait for lock delay

                    # Grant the lock
                    token = self._next_token
                    self._next_token += 1
                    self._locks[resource] = LockInfo(
                        owner=owner,
                        fencing_token=token,
                        acquired_at=time.monotonic(),
                        ttl=ttl,
                    )
                    return token

                # Lock held by someone else
                if current.owner == owner:
                    # Re-entrant: return existing token
                    return current.fencing_token

            time.sleep(0.1)  # Backoff before retry

        return None  # Timeout

    def release(self, resource: str, owner: str) -> bool:
        """
        Release a lock.

        Args:
            resource: The resource to unlock
            owner: Must match the lock owner

        Returns:
            True if released, False if not the owner
        """
        with self._mutex:
            current = self._locks.get(resource)
            if current is None:
                return False
            if current.owner != owner:
                return False

            # Activate lock delay
            self._lock_delay_until[resource] = (
                time.monotonic() + self._lock_delay_duration
            )
            del self._locks[resource]
            return True

    def get_lock_info(self, resource: str) -> Optional[LockInfo]:
        """Get information about a lock."""
        with self._mutex:
            lock = self._locks.get(resource)
            if lock is not None and lock.is_expired:
                return None
            return lock


class FencedKeyValueStore:
    """
    A key-value store that enforces fencing tokens.

    Every write operation must include a fencing token.
    Writes with tokens lower than or equal to the highest
    seen token for that key are rejected.
    """

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._highest_token: Dict[str, int] = {}
        self._lock = threading.Lock()

    def write(self, key: str, value: str, fencing_token: int) -> Tuple[bool, str]:
        """
        Write a value with fencing token enforcement.

        Returns:
            (success, message) tuple
        """
        with self._lock:
            highest = self._highest_token.get(key, 0)
            if fencing_token <= highest:
                return (
                    False,
                    f"Stale token: {fencing_token} <= {highest}. "
                    f"Write rejected."
                )
            self._highest_token[key] = fencing_token
            self._data[key] = value
            return (True, f"Write accepted with token {fencing_token}")

    def read(self, key: str) -> Optional[str]:
        """Read a value (no fencing needed)."""
        with self._lock:
            return self._data.get(key)


def demo_fenced_locking():
    """Demonstrate distributed locking with fencing tokens."""
    lock_mgr = DistributedLockManager(lock_delay=0.5)
    kv_store = FencedKeyValueStore()

    print("=== Distributed Lock with Fencing Tokens Demo ===\n")

    # Client A acquires lock
    token_a = lock_mgr.acquire("my-resource", "client-A", ttl=2.0)
    print(f"Client A acquired lock with token: {token_a}")

    # Client A writes with its token
    ok, msg = kv_store.write("config", "value-from-A", token_a)
    print(f"Client A writes: {msg}")

    # Client B tries to acquire (should wait/fail)
    token_b = lock_mgr.acquire("my-resource", "client-B", ttl=2.0, timeout=0.5)
    print(f"Client B acquire attempt: token={token_b}")  # None (timeout)

    # Client A releases lock
    lock_mgr.release("my-resource", "client-A")
    print("Client A released lock")

    # Wait for lock delay to expire
    time.sleep(0.6)

    # Client B acquires lock (gets higher token)
    token_b = lock_mgr.acquire("my-resource", "client-B", ttl=2.0, timeout=5.0)
    print(f"\nClient B acquired lock with token: {token_b}")

    # Client B writes with its token
    ok, msg = kv_store.write("config", "value-from-B", token_b)
    print(f"Client B writes: {msg}")

    # Simulate: Client A (stale) tries to write with old token
    ok, msg = kv_store.write("config", "stale-value-from-A", token_a)
    print(f"\nClient A (stale) tries to write: {msg}")  # REJECTED

    # Client B writes again (still valid)
    ok, msg = kv_store.write("config", "value-from-B-v2", token_b)
    print(f"Client B writes again: {msg}")

    print(f"\nFinal value of 'config': {kv_store.read('config')}")


if __name__ == "__main__":
    demo_fenced_locking()
```

---

## 9. 구현: Snowflake ID 생성기

```python
import time
import threading
from typing import Optional

class SnowflakeIDGenerator:
    """
    Twitter Snowflake ID generator.

    Generates 64-bit, roughly time-ordered, unique IDs
    without coordination between generators.

    Bit layout:
      - 1 bit:  unused (sign bit)
      - 41 bits: timestamp (ms since custom epoch)
      - 5 bits:  datacenter ID (0-31)
      - 5 bits:  worker ID (0-31)
      - 12 bits: sequence number (0-4095)

    Supports up to 4096 IDs per millisecond per worker.
    Timestamp space: ~69.7 years from epoch.
    """

    # Bit allocation
    TIMESTAMP_BITS = 41
    DATACENTER_BITS = 5
    WORKER_BITS = 5
    SEQUENCE_BITS = 12

    # Maximum values
    MAX_DATACENTER_ID = (1 << DATACENTER_BITS) - 1   # 31
    MAX_WORKER_ID = (1 << WORKER_BITS) - 1            # 31
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1            # 4095

    # Bit shifts
    WORKER_SHIFT = SEQUENCE_BITS                       # 12
    DATACENTER_SHIFT = SEQUENCE_BITS + WORKER_BITS     # 17
    TIMESTAMP_SHIFT = (
        SEQUENCE_BITS + WORKER_BITS + DATACENTER_BITS  # 22
    )

    # Custom epoch: 2020-01-01 00:00:00 UTC (in ms)
    CUSTOM_EPOCH = 1577836800000

    def __init__(self, datacenter_id: int, worker_id: int):
        """
        Initialize the Snowflake generator.

        Args:
            datacenter_id: Datacenter identifier (0-31)
            worker_id: Worker identifier within datacenter (0-31)

        Raises:
            ValueError: If IDs are out of range
        """
        if not 0 <= datacenter_id <= self.MAX_DATACENTER_ID:
            raise ValueError(
                f"datacenter_id must be 0-{self.MAX_DATACENTER_ID}, "
                f"got {datacenter_id}"
            )
        if not 0 <= worker_id <= self.MAX_WORKER_ID:
            raise ValueError(
                f"worker_id must be 0-{self.MAX_WORKER_ID}, "
                f"got {worker_id}"
            )

        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self._sequence = 0
        self._last_timestamp = -1
        self._lock = threading.Lock()

    def _current_time_ms(self) -> int:
        """Get current time in milliseconds."""
        return int(time.time() * 1000)

    def _wait_next_ms(self, last_ts: int) -> int:
        """Block until the clock advances to the next millisecond."""
        ts = self._current_time_ms()
        while ts <= last_ts:
            ts = self._current_time_ms()
        return ts

    def generate(self) -> int:
        """
        Generate a unique Snowflake ID.

        Returns:
            64-bit integer ID

        Raises:
            RuntimeError: If clock moves backward
        """
        with self._lock:
            timestamp = self._current_time_ms()

            if timestamp < self._last_timestamp:
                # Clock moved backward: refuse to generate
                delta = self._last_timestamp - timestamp
                raise RuntimeError(
                    f"Clock moved backward by {delta}ms. "
                    f"Refusing to generate ID to prevent duplicates."
                )

            if timestamp == self._last_timestamp:
                # Same millisecond: increment sequence
                self._sequence = (self._sequence + 1) & self.MAX_SEQUENCE
                if self._sequence == 0:
                    # Sequence exhausted: wait for next millisecond
                    timestamp = self._wait_next_ms(self._last_timestamp)
            else:
                # New millisecond: reset sequence
                self._sequence = 0

            self._last_timestamp = timestamp

            # Compose the ID
            snowflake_id = (
                ((timestamp - self.CUSTOM_EPOCH) << self.TIMESTAMP_SHIFT)
                | (self.datacenter_id << self.DATACENTER_SHIFT)
                | (self.worker_id << self.WORKER_SHIFT)
                | self._sequence
            )

            return snowflake_id

    @classmethod
    def parse(cls, snowflake_id: int) -> dict:
        """
        Decompose a Snowflake ID into its components.

        Args:
            snowflake_id: The ID to parse

        Returns:
            Dict with timestamp_ms, datacenter_id, worker_id, sequence
        """
        sequence = snowflake_id & cls.MAX_SEQUENCE
        worker_id = (snowflake_id >> cls.WORKER_SHIFT) & cls.MAX_WORKER_ID
        datacenter_id = (
            (snowflake_id >> cls.DATACENTER_SHIFT) & cls.MAX_DATACENTER_ID
        )
        timestamp_ms = (
            (snowflake_id >> cls.TIMESTAMP_SHIFT) + cls.CUSTOM_EPOCH
        )

        return {
            "id": snowflake_id,
            "timestamp_ms": timestamp_ms,
            "timestamp_iso": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.gmtime(timestamp_ms / 1000),
            ),
            "datacenter_id": datacenter_id,
            "worker_id": worker_id,
            "sequence": sequence,
            "binary": format(snowflake_id, "064b"),
        }

    @classmethod
    def print_bit_layout(cls, snowflake_id: int) -> None:
        """Print a visual breakdown of the ID's bit layout."""
        binary = format(snowflake_id, "064b")
        print(f"ID: {snowflake_id}")
        print(f"Binary: {binary}")
        print(f"  [0]  Unused:    {binary[0]}")
        print(f"  [1-41]  Timestamp:  {binary[1:42]} "
              f"({int(binary[1:42], 2)})")
        print(f"  [42-46] Datacenter: {binary[42:47]} "
              f"({int(binary[42:47], 2)})")
        print(f"  [47-51] Worker:     {binary[47:52]} "
              f"({int(binary[47:52], 2)})")
        print(f"  [52-63] Sequence:   {binary[52:64]} "
              f"({int(binary[52:64], 2)})")


def demo_snowflake():
    """Demonstrate Snowflake ID generation."""
    print("=== Snowflake ID Generator Demo ===\n")

    # Create generators for different workers
    gen_dc0_w0 = SnowflakeIDGenerator(datacenter_id=0, worker_id=0)
    gen_dc0_w1 = SnowflakeIDGenerator(datacenter_id=0, worker_id=1)
    gen_dc1_w0 = SnowflakeIDGenerator(datacenter_id=1, worker_id=0)

    # Generate IDs from different workers
    print("IDs from datacenter 0, worker 0:")
    for i in range(5):
        sid = gen_dc0_w0.generate()
        parsed = SnowflakeIDGenerator.parse(sid)
        print(f"  {sid:>20d}  seq={parsed['sequence']}  "
              f"time={parsed['timestamp_iso']}")

    print("\nIDs from datacenter 0, worker 1:")
    for i in range(3):
        sid = gen_dc0_w1.generate()
        parsed = SnowflakeIDGenerator.parse(sid)
        print(f"  {sid:>20d}  seq={parsed['sequence']}  "
              f"dc={parsed['datacenter_id']} w={parsed['worker_id']}")

    print("\nIDs from datacenter 1, worker 0:")
    for i in range(3):
        sid = gen_dc1_w0.generate()
        parsed = SnowflakeIDGenerator.parse(sid)
        print(f"  {sid:>20d}  seq={parsed['sequence']}  "
              f"dc={parsed['datacenter_id']} w={parsed['worker_id']}")

    # Show bit layout
    print("\n=== Bit Layout ===")
    sid = gen_dc0_w0.generate()
    SnowflakeIDGenerator.print_bit_layout(sid)

    # Demonstrate ordering
    print("\n=== Ordering Guarantee ===")
    ids = [gen_dc0_w0.generate() for _ in range(10)]
    print(f"All IDs strictly increasing: {all(ids[i] < ids[i+1] for i in range(len(ids)-1))}")
    print(f"First: {ids[0]}")
    print(f"Last:  {ids[-1]}")

    # Multi-threaded generation
    print("\n=== Multi-threaded Generation ===")
    all_ids = []
    lock = threading.Lock()

    def generate_ids(generator, count):
        local_ids = [generator.generate() for _ in range(count)]
        with lock:
            all_ids.extend(local_ids)

    threads = [
        threading.Thread(target=generate_ids, args=(gen_dc0_w0, 1000)),
        threading.Thread(target=generate_ids, args=(gen_dc0_w1, 1000)),
        threading.Thread(target=generate_ids, args=(gen_dc1_w0, 1000)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    unique_ids = set(all_ids)
    print(f"Generated {len(all_ids)} IDs across 3 workers")
    print(f"Unique IDs: {len(unique_ids)}")
    print(f"Duplicates: {len(all_ids) - len(unique_ids)}")


if __name__ == "__main__":
    demo_snowflake()
```

---

## 10. 요약 및 추가 읽을거리

### 핵심 요점

| 프리미티브 | 핵심 통찰 | 프로덕션 시스템 |
|-----------|-------------|-------------------|
| 분산 잠금 | 정확성을 위해 항상 fencing token을 사용 | ZooKeeper, etcd, Chubby |
| Redlock | 효율성에는 OK, 정확성 보장에는 부적합 | Redis |
| Fencing token | 단조 토큰으로 리소스 서버가 만료된 연산을 거부 | 모든 잠금 서비스 |
| 배리어 | 이중 배리어가 배치 병렬 계산을 동기화 | ZooKeeper, MapReduce |
| Leader election | Consensus 기반이 유일한 프로덕션급 접근 방식 | Raft, ZAB, Paxos |
| Snowflake ID | 시간 + 머신 + 시퀀스 = 조정 없이 고유 | Twitter, Discord |
| 서비스 디스커버리 | 시스템 요구사항과 팀 역량에 맞는 접근 방식 선택 | Consul, etcd, k8s |

### 필수 읽을거리

1. **Burrows (2006)** — "The Chubby Lock Service for Loosely-Coupled Distributed Systems"
2. **Hunt et al. (2010)** — "ZooKeeper: Wait-free Coordination for Internet-scale Systems"
3. **Kleppmann (2016)** — "How to do distributed locking" (Redlock 비판 블로그 포스트)
4. **Sanfilippo (2016)** — "Is Redlock safe?" (Kleppmann에 대한 응답)
5. **Garcia-Molina (1982)** — "Elections in a Distributed Computing System" (Bully 알고리즘)

### 다른 레슨과의 연결

- **Lesson 5 (Paxos)** 및 **Lesson 6 (Raft)**: Consensus가 조정 프리미티브의 기반
- **Lesson 8 (분산 트랜잭션)**: 2PC 코디네이터 내에서 잠금 사용
- **Lesson 13 (장애 감지)**: 장애 감지기가 leader 재선출을 트리거
- **Lesson 16 (캡스톤)**: KV 스토어에서 leader election과 fencing 사용

---

[다음: TLA+를 이용한 형식 검증](./15_Formal_Verification_TLAplus.md)
