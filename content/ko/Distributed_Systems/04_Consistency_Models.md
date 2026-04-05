# 레슨 4: 일관성 모델 심층 분석

[개요](./00_Overview.md) | [이전](./03_FLP_Impossibility_and_Bounds.md) | [다음](./05_Paxos_Family.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. history와 실시간 순서를 사용하여 linearizability를 형식적으로 정의하고, 동시 실행에서 linearization point를 식별
2. linearizability, sequential consistency, causal consistency, eventual consistency를 정확한 형식 기준으로 구분
3. CAP 정리를 진술하고 증명하며, 그 함의에 대한 흔한 오해를 설명
4. PACELC 프레임워크를 적용하여 파티션 이외의 일관성-지연 트레이드오프를 추론
5. linearizability checker를 구현하고 코드에서 다양한 일관성 수준을 시뮬레이션

---

## 목차

1. [일관성이 중요한 이유](#1-일관성이-중요한-이유)
2. [Linearizability](#2-linearizability)
3. [Sequential Consistency](#3-sequential-consistency)
4. [Causal Consistency](#4-causal-consistency)
5. [Eventual Consistency](#5-eventual-consistency)
6. [세션 보장](#6-세션-보장)
7. [일관성 계층 구조](#7-일관성-계층-구조)
8. [CAP 정리 재탐구](#8-cap-정리-재탐구)
9. [PACELC: CAP을 넘어서](#9-pacelc-cap을-넘어서)
10. [Jepsen 스타일 일관성 검증](#10-jepsen-스타일-일관성-검증)
11. [코드: Linearizability Checker](#11-코드-linearizability-checker)
12. [코드: 일관성 수준 시뮬레이션](#12-코드-일관성-수준-시뮬레이션)
13. [요약](#13-요약)
14. [연습 문제](#14-연습-문제)
15. [참고 문헌](#15-참고-문헌)

---

## 1. 일관성이 중요한 이유

### 1.1 복제 문제

데이터가 여러 노드에 복제될 때, 근본적인 질문이 발생합니다: **클라이언트가 읽는 데이터에 대해 어떤 보장을 받을 수 있는가?**

간단한 시나리오를 고려해 봅시다:

```
Client A writes x = 1 to Node 1 at time t₁
Client B reads x from Node 2 at time t₂ > t₁

Questions:
  - Must B see x = 1? (strong consistency)
  - Could B see x = null (the old value)? (weak consistency)
  - Is there a middle ground? (yes, many)
```

답은 **일관성 모델** -- 분산 시스템과 클라이언트 간에 어떤 동작이 가능한지에 대한 계약 -- 에 따라 달라집니다.

### 1.2 일관성 스펙트럼

```
Strong ◄─────────────────────────────────────────────────► Weak

Linearizable  Sequential  Causal   PRAM   Eventual   No
                                                   Guarantee
     │            │          │       │        │         │
     │            │          │       │        │         │
  Strongest    Program    Cause-    Per-   Eventually  Chaos
  (real-time   order      effect   client  converge
   ordering)   preserved  tracked  order
```

스펙트럼의 각 지점은 다음 사이의 다른 트레이드오프를 나타냅니다:

- **일관성**: 읽기가 얼마나 "최신"인가
- **가용성**: 장애 중에 시스템이 응답할 수 있는가
- **성능**: 지연 시간과 처리량
- **복잡성**: 구현하고 추론하기 얼마나 어려운가

### 1.3 일관성의 비용

| 일관성 수준 | 필요한 조율 | 일반적 지연 시간 | 파티션 중 가용성 |
|-------------|------------|-----------------|-----------------|
| Linearizable | 완전 (quorum 읽기 + 쓰기) | 높음 | 낮음 (소수 파티션에서 불가용) |
| Sequential | 세션별 순서 지정 | 중간 | 중간 |
| Causal | 메타데이터 추적 | 중간 | 높음 (모든 replica에서 서비스 가능) |
| Eventual | 없음 (비동기 복제) | 낮음 | 높음 (항상 가용) |

---

## 2. Linearizability

### 2.1 비공식 정의

Linearizability (Herlihy & Wing, 1990)는 가장 강력한 단일 객체 일관성 모델입니다. 비공식적으로:

> 모든 연산은 호출과 응답 사이의 어떤 순간에 **원자적으로** 효력을 발휘하는 것으로 나타나며, 모든 연산은 실시간 순서를 존중하는 단일 순차 순서와 일치합니다.

다시 말해, linearizable 시스템은 복제되어 있음에도 불구하고 데이터의 단일 복사본처럼 "동작합니다."

### 2.2 형식적 정의

**History** `H`는 공유 객체에 대한 연산의 호출 및 응답 이벤트의 시퀀스입니다. 각 연산 `op`는 다음으로 구성됩니다:

- `inv(op)`: 호출 이벤트 (클라이언트가 요청을 보낼 때)
- `res(op)`: 응답 이벤트 (클라이언트가 응답을 받을 때)
- `op`의 **실시간 구간**은 `[inv(op), res(op)]`

**실시간 순서**: 연산 `op₁`이 연산 `op₂`에 **선행**한다 (`op₁ <_H op₂`로 표기)는 것은 `res(op₁)`이 `H`에서 `inv(op₂)` 이전에 발생함을 의미합니다.

**동시 연산**: `op₁`과 `op₂`는 어느 쪽도 다른 쪽에 선행하지 않으면 동시적입니다 (실시간 구간이 겹침).

**Linearizability**: History `H`가 linearizable하다는 것은 `H`의 연산들의 **linearization** -- 전순서 `S` -- 가 존재하여 다음을 만족하는 것입니다:

1. **실시간 제약**: `op₁ <_H op₂`이면 `op₁ <_S op₂` (linearization이 실시간 순서를 존중)
2. **순차 명세**: 전순서 `S`를 객체에 적용하면, 객체의 순차 명세와 일치하는 결과를 생성 (예: 레지스터는 마지막으로 쓰인 값을 반환)

### 2.3 Linearization Point

**Linearization point**는 연산의 실시간 구간 내에서 연산이 "효력을 발휘하는" 특정 순간입니다. 각 연산에 linearization point를 할당하여 linearization point 순서대로 연산을 실행했을 때 순차 명세를 만족하면, 해당 history는 linearizable합니다.

```
Example: Two writes and one read on a register (initial value = 0)

Time ────────────────────────────────────────────────────────►

Client A:  |──── write(x, 1) ────|
                      ↑ linearization point

Client B:       |──── write(x, 2) ────|
                            ↑ linearization point

Client C:                |──── read(x) → 2 ────|
                                  ↑ linearization point

Linearization order: write(x,1), write(x,2), read(x)→2  ✓
  read returns 2, which is the last written value.

Alternative linearization: write(x,2), write(x,1), read(x)→2  ✗
  This would require read to return 1 (last written), not 2.
  But the real-time constraint is also violated:
  write(x,1) starts before write(x,2), so their linearization
  points must be compatible with overlap (both are ok here).
  However, read must return 1 if write(x,1) comes last → contradiction.
```

### 2.4 Linearizability 예제

**예제 1: Linearizable history**

```
Time ──────────────────────────────────────►

Client A:  |── w(x,1) ──|     |── r(x)→2 ──|
Client B:        |── w(x,2) ──|

Linearization: w(x,1), w(x,2), r(x)→2  ✓
  w(x,1) completes before r(x) starts → w(x,1) <_S r(x)  ✓
  w(x,2) completes before r(x) starts → w(x,2) <_S r(x)  ✓
  r(x) returns 2, which is the value of the last write   ✓
```

**예제 2: Non-linearizable history**

```
Time ──────────────────────────────────────►

Client A:  |── w(x,1) ──|
Client B:        |── w(x,2) ──|
Client C:                         |── r(x)→1 ──|

Is this linearizable?
  r(x) starts after both writes complete.
  Any linearization must have both writes before the read.
  Possible orders:
    w(x,1), w(x,2), r(x) → read should return 2, but returns 1 ✗
    w(x,2), w(x,1), r(x) → read should return 1 ✓
      But does this violate real-time? w(x,1) starts before w(x,2),
      but since they overlap, either order is allowed.
  So this IS linearizable with order: w(x,2), w(x,1), r(x)→1  ✓
```

**예제 3: 확실히 non-linearizable**

```
Time ──────────────────────────────────────────────────────►

Client A:  |── w(x,1) ──|
Client B:                     |── w(x,2) ──|
Client C:                                       |── r(x)→1 ──|

Is this linearizable?
  w(x,1) completes before w(x,2) starts → w(x,1) <_H w(x,2)
  w(x,2) completes before r(x) starts → w(x,2) <_H r(x)
  Therefore: w(x,1) <_S w(x,2) <_S r(x)
  Last write before r(x) is w(x,2), so read must return 2.
  But read returns 1. CONTRADICTION.
  This history is NOT linearizable.  ✗
```

### 2.5 Linearizability 검증

**정리** (Gibbons & Korach, 1997): history가 linearizable한지 검증하는 것은 일반적으로 NP-완전입니다.

그러나 동시 연산 수가 적은 단일 객체 history에 대해서는 실용적인 알고리즘이 존재합니다:

1. **Wing & Gong 알고리즘**: 가능한 모든 linearization을 열거 (최악의 경우 지수적이지만, 가지치기로 소규모 history에서 실용적)
2. **Knossos** (Jepsen에서 사용): 레지스터 history를 위한 효율적인 linearizability checker
3. **Porcupine** (Go): P-compositionality 최적화를 사용하는 빠른 linearizability checker

### 2.6 지역성과 합성성

**정리** (Herlihy & Wing, 1990): Linearizability는 **지역적** 속성입니다: history `H`가 linearizable한 것은 각 객체 `x`에 대해 sub-history `H|x` (`x`에 대한 연산으로 제한)가 linearizable한 것과 동치입니다.

이것은 강력한 속성입니다: 객체별로 linearizability를 검증할 수 있습니다. Sequential consistency는 이 속성을 갖지 않습니다.

### 2.7 Linearizability의 비용

Linearizability는 조율을 요구하며, 이에는 본질적인 비용이 있습니다:

```
To guarantee linearizable reads:
  Option 1: Read from leader (bottleneck, not fault-tolerant for reads)
  Option 2: Read from quorum (majority must agree)
  Option 3: LeaseRead (leader holds a lease, serves reads locally during lease)
  Option 4: ReadIndex (leader confirms it is still leader before serving read)

All options add latency compared to reading from the nearest replica.
```

**정리** (Attiya & Welch, 1994): 메시지 전달 시스템에서 linearizable read-write 레지스터의 모든 구현은 최소 한 번의 메시지 왕복이 필요합니다 (순수 로컬 처리 불가).

---

## 3. Sequential Consistency

### 3.1 정의

**Sequential consistency** (Lamport, 1979): history가 sequentially consistent하다는 것은 모든 연산의 전순서가 존재하여 다음을 만족하는 것입니다:

1. **프로그램 순서**: 각 프로세스의 연산이 프로세스가 발행한 순서대로 나타남
2. **순차 명세**: 전순서가 객체의 순차 명세와 일치

**Linearizability와의 핵심 차이**: Sequential consistency는 실시간 순서를 존중할 필요가 **없습니다**. 다른 프로세스의 두 연산은, 하나가 다른 하나가 시작하기 전에 완료되었더라도 전순서에서 재배열될 수 있습니다.

### 3.2 Linearizability와의 비교

```
Linearizability:
  Respects: program order + real-time order
  Guarantee: "behaves like a single copy in real time"

Sequential consistency:
  Respects: program order ONLY (not real-time)
  Guarantee: "behaves like a single copy, but may be delayed"
```

### 3.3 예제: Sequentially Consistent이지만 Linearizable이 아닌 경우

```
Time ──────────────────────────────────────────────────────►

Process P1:  |── w(x,1) ──|                |── r(y)→0 ──|
Process P2:                  |── w(y,1) ──|   |── r(x)→0 ──|

Linearizable?
  w(x,1) completes before w(y,1) starts → w(x,1) <_H w(y,1)
  w(y,1) completes before r(x) starts → w(y,1) <_H r(x)
  So: w(x,1) <_S w(y,1) <_S r(x)
  r(x) must return 1 (x was written). But r(x)→0. NOT linearizable. ✗

Sequentially consistent?
  Try order: r(y)→0, r(x)→0, w(x,1), w(y,1)
  Check program order:
    P1: w(x,1) before r(y)→0? In our total order, r(y) is first. VIOLATES P1's order. ✗

  Try order: r(x)→0, w(x,1), r(y)→0, w(y,1)
  Check program order:
    P1: w(x,1) before r(y)→0? In total order, w(x,1) is second, r(y) is third. ✓
    P2: w(y,1) before r(x)→0? In total order, w(y,1) is fourth, r(x) is first. VIOLATES P2's order. ✗

  Try order: w(x,1), r(x)→0 ... no, r(x) returns 0 but x was written to 1. ✗

  Actually, consider: r(x)→0, r(y)→0, w(x,1), w(y,1)
  P1 order: w(x,1) then r(y)→0? In total order, w(x,1) is 3rd, r(y) is 2nd. VIOLATES. ✗

  Hmm, let us try: w(y,1), r(y)→0 ... r(y) should return 1, not 0. ✗

  Actually this history IS sequentially consistent:
  Order: r(y)→0, w(y,1), r(x)→0, w(x,1)
  P1 program order: P1 does w(x,1) then r(y)→0.
    In total order: r(y)→0 is 1st, w(x,1) is 4th. VIOLATES P1's order. ✗

  Let me reconsider. The history r(x)→0, r(y)→0 means both reads return initial value 0.
  For this to be sequentially consistent, both reads must occur before the corresponding writes.

  Total order: r(y)→0, r(x)→0, w(x,1), w(y,1)
  P1 program order: w(x,1) is before r(y). In total order: r(y) is 1st, w(x,1) is 3rd. VIOLATES.

  This history is NEITHER linearizable NOR sequentially consistent.
  Let me fix the example.
```

**수정된 예제**:

```
Time ──────────────────────────────────────────────────────►

Process P1:  |── w(x,1) ──|
Process P2:                               |── r(x)→0 ──|

Linearizable? No.
  w(x,1) completes before r(x) starts → r(x) must return 1. But returns 0. ✗

Sequentially consistent? Yes!
  Total order: r(x)→0, w(x,1)
  P1 program order: only w(x,1) → satisfied trivially.
  P2 program order: only r(x)→0 → satisfied trivially.
  Sequential spec: r(x) happens before w(x,1), so x has initial value 0. ✓

The key: sequential consistency allows reordering across processes,
even if one operation completed before another started in real time.
```

### 3.4 속성

| 속성 | Linearizability | Sequential Consistency |
|------|----------------|----------------------|
| 실시간 존중 | 예 | 아니오 |
| 프로그램 순서 존중 | 예 | 예 |
| 합성 가능 (지역적) | 예 | **아니오** |
| 검증 NP-완전 | 예 | 예 |
| 실제 사용 | 데이터베이스, linearizable KV store | CPU 메모리 모델, Zookeeper |

**Sequential consistency의 비합성성**: 두 객체가 각각 개별적으로 sequentially consistent할 수 있지만, 결합된 history는 그렇지 않을 수 있습니다. 이로 인해 sequentially consistent 컴포넌트로 복잡한 시스템을 구축하기가 훨씬 어렵습니다.

---

## 4. Causal Consistency

### 4.1 정의

**Causal consistency** (Ahamad et al., 1995): history가 causally consistent하다는 것은 모든 프로세스가 인과적으로 관련된 연산을 동일한 순서로 관찰하는 것입니다. 동시 연산 (인과적으로 관련되지 않은)은 다른 프로세스에서 다른 순서로 관찰될 수 있습니다.

형식적으로, 복제된 저장소가 causally consistent한 것은 임의의 두 연산 `a`와 `b`에 대해:

```
If a →(causally precedes) b, then every process sees a before b.
If a ‖ b (concurrent), different processes may see them in different orders.
```

### 4.2 인과적 선행

두 연산이 인과적으로 관련 (`a → b`)되는 경우:

1. **같은 세션**: `a`와 `b`가 같은 클라이언트 세션에서 발행되고, `a`가 `b`에 선행
2. **Reads-from**: `b`가 `a`에 의해 쓰인 값을 읽음
3. **추이성**: `a → c`이고 `c → b`이면 `a → b`

### 4.3 예제

```
Process P1: w(x, 1)
Process P2: r(x) → 1; w(y, 2)     [reads x=1, then writes y=2]
Process P3: r(y) → 2; r(x) → ?

Causal chain: w(x,1) → r(x)→1 → w(y,2) → r(y)→2

Since r(y)→2 causally depends on w(y,2), which causally depends on
r(x)→1, which causally depends on w(x,1):

  r(y)→2 causally depends on w(x,1)

Therefore, P3 MUST see w(x,1) before or by the time it sees w(y,2).
So r(x) at P3 must return 1 (or a later value).

Under causal consistency:  r(x) = 1  ✓
Under eventual consistency: r(x) could be 0 (stale)  — depends on timing
```

### 4.4 Causal Consistency 구현

Causal consistency는 다음을 사용하여 구현할 수 있습니다:

1. **Vector clock / version vector**: 인과적 의존성을 추적하고 의존성이 충족될 때까지 전달을 지연 (레슨 02 참조)
2. **Causal broadcast**: causal broadcast 프로토콜을 사용하여 메시지가 인과적 순서로 전달되도록 보장
3. **의존성 추적**: 각 쓰기가 의존하는 연산 집합을 포함

```python
class CausalStore:
    """
    A causally consistent key-value store using dependency tracking.
    Each write carries a vector clock summarizing its causal past.
    Reads are blocked until the causal dependencies are satisfied.
    """

    def __init__(self, replica_id: str, all_replicas: list[str]):
        self.replica_id = replica_id
        self.all_replicas = sorted(all_replicas)
        self.idx = self.all_replicas.index(replica_id)
        self.n = len(all_replicas)

        # Local state
        self.store: dict[str, tuple[any, list[int]]] = {}  # key -> (value, vector_clock)
        self.vc = [0] * self.n  # local vector clock

        # Pending writes from other replicas (waiting for causal deps)
        self.pending: list[tuple[str, any, list[int]]] = []

    def write(self, key: str, value) -> list[int]:
        """Write a value (locally). Returns the write's vector clock."""
        self.vc[self.idx] += 1
        write_vc = list(self.vc)
        self.store[key] = (value, write_vc)
        return write_vc

    def read(self, key: str) -> tuple[any, list[int]]:
        """Read a value. Returns (value, vector_clock) or (None, [0]*n)."""
        if key in self.store:
            return self.store[key]
        return (None, [0] * self.n)

    def receive_write(self, key: str, value, write_vc: list[int]):
        """Receive a replicated write from another replica."""
        self.pending.append((key, value, write_vc))
        self._try_apply_pending()

    def _can_apply(self, write_vc: list[int]) -> bool:
        """Check if all causal dependencies are satisfied."""
        for i in range(self.n):
            if i == self.idx:
                continue  # skip self
            if write_vc[i] > self.vc[i]:
                return False  # missing a causally preceding write
        return True

    def _try_apply_pending(self):
        """Apply pending writes whose dependencies are satisfied."""
        applied = True
        while applied:
            applied = False
            remaining = []
            for key, value, write_vc in self.pending:
                if self._can_apply(write_vc):
                    # Apply the write
                    existing_vc = self.store.get(key, (None, [0] * self.n))[1]
                    # Only apply if this write is newer
                    if not all(write_vc[i] <= existing_vc[i] for i in range(self.n)):
                        self.store[key] = (value, write_vc)
                    # Update local vector clock
                    for i in range(self.n):
                        self.vc[i] = max(self.vc[i], write_vc[i])
                    applied = True
                else:
                    remaining.append((key, value, write_vc))
            self.pending = remaining
```

### 4.5 실제 시스템에서의 Causal Consistency

| 시스템 | Causal Consistency 구현 |
|--------|------------------------|
| COPS (Lloyd et al., 2011) | 명시적 의존성 추적과 가장 가까운 replica 읽기 |
| Eiger | 의존성 메타데이터를 사용한 다중 키 causal 트랜잭션 |
| MongoDB (causal sessions) | 인과적 순서를 위한 hybrid logical clock |
| Riak (with CRDTs) | Vector clock + CRDT로 충돌 없는 인과적 업데이트 |

---

## 5. Eventual Consistency

### 5.1 정의

**Eventual consistency**: 데이터 항목에 새로운 업데이트가 없으면, 모든 replica가 **결국** 동일한 값으로 수렴합니다.

형식적으로:

```
For any key k, if the last write to k occurs at time t:
  ∃ time T > t such that ∀ replicas r, at time T:
    read(r, k) returns the value of the last write

The convergence time T is unbounded (no guarantee on how long it takes).
```

### 5.2 Eventual Consistency가 보장하지 않는 것

| 비보장 사항 | 결과 |
|------------|------|
| Read-your-writes | x=5를 쓰고, 즉시 x를 읽으면 이전 값 3을 받을 수 있음 |
| Monotonic reads | x=5를 읽고, 다시 읽으면 x=3을 받을 수 있음 (역행) |
| Causal ordering | 원본 메시지를 보지 못한 채 답장을 볼 수 있음 |
| Write ordering | 같은 클라이언트의 쓰기가 재배열될 수 있음 |
| 수렴 시간 | 수렴하는 데 걸리는 시간에 대한 보장 없음 |
| 충돌 해결 | replica가 발산할 때 누가 이기는가? (구현에 따라 다름) |

### 5.3 Anti-Entropy 프로토콜

Eventual consistency는 업데이트를 전파하기 위해 **anti-entropy** 메커니즘에 의존합니다:

```python
class AntiEntropyStore:
    """
    Eventually consistent key-value store using anti-entropy.
    Periodically synchronizes with random peers.
    """

    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = peers
        self.store: dict[str, tuple[any, float]] = {}  # key -> (value, timestamp)

    def write(self, key: str, value):
        """Write with last-writer-wins (LWW) timestamp."""
        self.store[key] = (value, time.time())

    def read(self, key: str):
        """Read local value (may be stale)."""
        if key in self.store:
            return self.store[key][0]
        return None

    def anti_entropy_push(self, peer_store: 'AntiEntropyStore'):
        """Push local state to a peer (epidemic/gossip protocol)."""
        for key, (value, ts) in self.store.items():
            peer_val = peer_store.store.get(key)
            if peer_val is None or ts > peer_val[1]:
                peer_store.store[key] = (value, ts)

    def anti_entropy_pull(self, peer_store: 'AntiEntropyStore'):
        """Pull state from a peer."""
        for key, (value, ts) in peer_store.store.items():
            local_val = self.store.get(key)
            if local_val is None or ts > local_val[1]:
                self.store[key] = (value, ts)

    def anti_entropy_round(self, all_stores: dict[str, 'AntiEntropyStore']):
        """One round of anti-entropy: pick a random peer and sync."""
        peer_id = random.choice(self.peers)
        peer = all_stores[peer_id]
        # Push-pull: bidirectional sync
        self.anti_entropy_push(peer)
        self.anti_entropy_pull(peer)
```

### 5.4 충돌 해결 전략

동시 쓰기가 충돌을 만들 때, eventual consistency 시스템은 다음 전략 중 하나를 사용합니다:

| 전략 | 작동 방식 | 장점 | 단점 |
|------|----------|------|------|
| Last-Writer-Wins (LWW) | 높은 타임스탬프가 승리 | 단순하고 자동적 | 데이터 손실 (패자 폐기) |
| 다중 값 (siblings) | 모든 동시 값을 클라이언트에 반환 | 데이터 손실 없음 | 클라이언트가 해결해야 함 |
| CRDT | 대수적 병합 함수 사용 | 자동적, 충돌 없음 | 제한된 데이터 타입 |
| 애플리케이션 병합 | 커스텀 병합 로직 | 유연함 | 복잡한 애플리케이션 코드 |
| Operational transform | 동시 연산을 변환 | 텍스트 편집에 적합 | 복잡하고 구현이 까다로움 |

---

## 6. 세션 보장

세션 보장 (Terry et al., 1994)은 eventual consistency보다 강하지만 sequential consistency보다 약한 클라이언트별 일관성 속성을 제공합니다.

### 6.1 네 가지 세션 보장

**Read Your Writes**: 클라이언트가 값 `v`를 쓰면, 같은 클라이언트의 후속 읽기는 `v` 또는 이후 값을 반환합니다.

```
Client A: write(x, 5) → OK
Client A: read(x) → must return 5 (or later)
           NOT allowed: read(x) → 3 (stale)
```

**Monotonic Reads**: 클라이언트가 키 `x`에 대해 값 `v`를 읽으면, 같은 클라이언트의 `x`에 대한 후속 읽기는 `v` 또는 이후 값을 반환합니다.

```
Client A: read(x) → 5
Client A: read(x) → must return 5 or later
           NOT allowed: read(x) → 3 (went backward)
```

**Monotonic Writes**: 같은 클라이언트의 쓰기는 모든 replica에서 순서대로 적용됩니다.

```
Client A: write(x, 1) then write(x, 2)
All replicas must apply write(x,1) before write(x,2)
NOT allowed: some replica applies write(x,2) first
```

**Writes Follow Reads**: 클라이언트가 `x`를 읽은 후 `y`를 쓰면, `y`에 대한 쓰기는 모든 replica에서 `x`의 읽기 이후로 순서가 매겨집니다. ("답장이 메시지를 따른다" 패턴을 포착)

```
Client A: read(x) → 5, then write(y, "response to x=5")
All replicas see the causal dependency:
  the state that produced x=5 is visible before write(y) is applied
```

### 6.2 구현

세션 보장은 **version vector** 또는 **세션 토큰**을 사용하여 구현할 수 있습니다:

```python
class SessionToken:
    """
    Track session state for implementing session guarantees.
    The token is sent with each request and updated with each response.
    """

    def __init__(self):
        self.read_vector: dict[str, int] = {}   # key -> version last read
        self.write_vector: dict[str, int] = {}   # key -> version last written

    def after_read(self, key: str, version: int):
        """Update token after a read."""
        self.read_vector[key] = max(self.read_vector.get(key, 0), version)

    def after_write(self, key: str, version: int):
        """Update token after a write."""
        self.write_vector[key] = max(self.write_vector.get(key, 0), version)

    def min_read_version(self, key: str) -> int:
        """Minimum version that a read of key must return (monotonic reads + RYW)."""
        return max(
            self.read_vector.get(key, 0),    # monotonic reads
            self.write_vector.get(key, 0),    # read your writes
        )
```

### 6.3 실제 시스템에서의 세션 보장

| 시스템 | 제공하는 세션 보장 |
|--------|-------------------|
| Azure Cosmos DB | 설정 가능 (session consistency 수준에서 네 가지 모두 제공) |
| MongoDB | Causal session (네 가지 모두 + causal ordering) |
| DynamoDB | Read-your-writes (consistent read 사용 시) |
| Cassandra | Read-your-writes (LOCAL_QUORUM 사용 시) |

---

## 7. 일관성 계층 구조

### 7.1 계층 다이어그램

```
                    Strict Serializability
                    (linearizable + serializable)
                           │
                    Linearizability
                    (single-object, real-time)
                           │
                    Sequential Consistency
                    (single-object, program-order)
                           │
                    Causal Consistency
                    (respects causality)
                      /          \
              PRAM                Writes Follow Reads
          (per-process order)     (read-write causality)
              |        \           /
        Monotonic      Read Your    Monotonic
         Reads         Writes        Writes
              \          |          /
               Eventual Consistency
               (convergence only)
```

### 7.2 비교표

| 모델 | 실시간 | 프로그램 순서 | 인과적 | 수렴 | 연산 |
|------|--------|-------------|--------|------|------|
| Linearizable | 예 | 예 | 예 | 예 | 단일 객체 |
| Sequential | 아니오 | 예 | 부분적 | 예 | 단일 객체 |
| Causal | 아니오 | 예 | 예 | 예 | 다중 객체 가능 |
| PRAM | 아니오 | 예 (프로세스별) | 아니오 | 예 | 단일 객체 |
| Eventual | 아니오 | 아니오 | 아니오 | 예 | 단일 객체 |

### 7.3 다중 객체 일관성

단일 객체 일관성 모델 (linearizability, sequential consistency)은 개별 데이터 항목에 적용됩니다. 여러 객체에 걸친 트랜잭션에는 다음이 필요합니다:

| 모델 | 설명 | 단일 객체 대응 |
|------|------|---------------|
| Strict serializability | 트랜잭션이 실시간 순서대로 직렬화되어 나타남 | Linearizability |
| Serializability | 트랜잭션이 직렬화되어 나타남 (임의의 순서) | Sequential consistency |
| Snapshot isolation | 트랜잭션이 일관된 스냅샷에서 읽음 | N/A (serializable보다 약함) |
| Read committed | 각 쿼리가 커밋된 데이터를 읽음 (다른 스냅샷을 볼 수 있음) | N/A |
| Read uncommitted | 커밋되지 않은 (dirty) 데이터를 읽을 수 있음 | 보장 없음 |

**Strict serializability** = Linearizability + Serializability. 이것은 데이터베이스의 최고 표준이지만 구현 비용이 가장 높습니다.

---

## 8. CAP 정리 재탐구

### 8.1 비공식 진술

CAP 정리 (Brewer, 2000; Gilbert & Lynch, 2002)는 분산 데이터 저장소가 동시에 다음 세 가지 보장 중 최대 **둘만** 제공할 수 있다고 말합니다:

- **Consistency** (C): 모든 읽기가 가장 최근 쓰기를 받음 (linearizability)
- **Availability** (A): 모든 요청이 오류 없는 응답을 받음 (타임아웃 없음)
- **Partition tolerance** (P): 네트워크 파티션에도 불구하고 시스템이 계속 운영됨

### 8.2 형식적 진술

**정리** (Gilbert & Lynch, 2002): 비동기 네트워크에서 다음 모두를 보장하는 read/write 데이터 객체를 구현하는 것은 불가능합니다:

1. **Consistency (Linearizability)**: 모든 연산이 단일 복사본에서 실행되는 것처럼 동작
2. **Availability**: 장애가 없는 노드에 대한 모든 요청이 응답을 받음
3. **Partition tolerance**: 노드 간 임의의 메시지 손실에도 시스템이 올바르게 기능

### 8.3 증명 스케치

증명은 모순에 의합니다. 네트워크로 연결되어 파티션될 수 있는 두 노드 `G₁`과 `G₂`를 가진 시스템을 고려합니다.

```
Setup:
  Node G₁        Node G₂
    │                │
    │    PARTITION    │
    │     ✕✕✕✕✕✕     │
    │                │

Execution:
  1. Client writes x = v₁ to G₁
  2. Network partitions (G₁ and G₂ cannot communicate)
  3. Client reads x from G₂

If Available + Partition-tolerant:
  G₂ must respond to the read (availability).
  G₂ has not received the write (partition).
  G₂ returns stale value → NOT consistent (linearizable). ✗

If Consistent + Partition-tolerant:
  G₂ must return v₁ (consistency).
  G₂ cannot learn v₁ (partition).
  G₂ must not respond → NOT available. ✗

If Consistent + Available:
  Must respond correctly to all requests.
  But if a partition occurs, this is impossible.
  So the system is NOT partition-tolerant. ✗
```

### 8.4 흔한 오해

**오해 1: "세 가지 중 두 가지를 고르라"**

이 프레이밍은 CA, CP, AP가 동등하게 유효한 선택인 것처럼 제시합니다. 현실에서 네트워크 파티션은 모든 분산 시스템에서 **반드시 발생합니다** (케이블이 끊기고, 스위치가 장애나고, 클라우드 AZ가 연결을 잃습니다). 따라서 partition tolerance는 선택 사항이 아닙니다 -- 파티션을 반드시 처리해야 합니다. 실제 선택은:

> **파티션 중에 consistency를 희생할 것인가 (AP) 아니면 availability를 희생할 것인가 (CP)?**

**오해 2: "CAP은 전체 시스템에 적용된다"**

CAP은 **개별 데이터 항목**에 대한 **개별 연산**에 적용됩니다. 시스템은 다른 데이터에 대해 다른 트레이드오프를 할 수 있습니다:

```
Example: E-commerce system
  - Product catalog: AP (eventual consistency, always available)
  - Shopping cart: AP (merge conflicts with CRDTs)
  - Inventory count: CP (linearizable, may be unavailable during partition)
  - Payment processing: CP (must be consistent, may timeout)
```

**오해 3: "C는 아무 일관성이나 의미한다"**

CAP에서 "C"는 구체적으로 **linearizability** -- 가장 강력한 단일 객체 일관성 모델 -- 을 의미합니다. 더 약한 모델 (causal, eventual)은 availability 및 partition tolerance와 함께 달성할 수 있습니다.

**오해 4: "C 또는 A를 영구적으로 잃는다"**

트레이드오프는 **파티션 중에만** 적용됩니다. 네트워크가 정상일 때:

```
No partition:  C + A are both achievable (e.g., Raft with healthy leader)
Partition:     Must choose C or A for each request
After healing: C + A resume
```

### 8.5 실제 시스템에서의 CAP

| 시스템 | 파티션 중 | 정상 운영 | 분류 |
|--------|----------|----------|------|
| Spanner | CP (소수 파티션에서 불가용) | CA (TrueTime 사용) | CP |
| DynamoDB | AP (sloppy quorum) | 조정 가능 (CP 가능) | AP 기본 |
| ZooKeeper | CP (소수 불가용) | CP | CP |
| Cassandra | AP (조정 가능) | AP 또는 CP (조정 가능) | AP 기본 |
| CockroachDB | CP (소수 파티션에서 불가용) | CA | CP |
| Riak | AP (siblings) | AP | AP |
| etcd | CP (Raft 과반수) | CP | CP |

---

## 9. PACELC: CAP을 넘어서

### 9.1 동기

CAP 정리는 **파티션 중** (P)에 무엇이 발생하는지만 알려줍니다. 하지만 파티션은 드뭅니다. **정상 운영** (파티션 없음) 중에는 어떤 트레이드오프가 존재하는가?

### 9.2 PACELC 프레임워크

Daniel Abadi (2012)가 PACELC를 제안했습니다:

```
If Partition (P):
  Choose Availability (A) or Consistency (C)
Else (E):
  Choose Latency (L) or Consistency (C)
```

이것은 파티션이 없더라도 **consistency**와 **latency** 사이에 트레이드오프가 있다는 근본적 통찰을 포착합니다 (consistency는 조율을 요구하고, 조율은 지연을 추가하기 때문입니다).

### 9.3 PACELC 분류

| 시스템 | P 시: A 또는 C? | E 시: L 또는 C? | 전체 분류 |
|--------|----------------|----------------|----------|
| DynamoDB | PA | EL | PA/EL |
| Cassandra | PA | EL | PA/EL (기본) |
| Riak | PA | EL | PA/EL |
| MongoDB | PA | EC | PA/EC (w:majority 사용) |
| Spanner | PC | EC | PC/EC |
| CockroachDB | PC | EC | PC/EC |
| ZooKeeper | PC | EC | PC/EC |
| etcd | PC | EC | PC/EC |
| PNUTS (Yahoo) | PC | EL | PC/EL |
| Megastore (Google) | PC | EC | PC/EC |

### 9.4 Latency-Consistency 트레이드오프

파티션이 없더라도, 동기적 복제 (consistency를 위한)는 지연을 추가합니다:

```
Eventual consistency (EL):
  Client → Nearest Replica → ACK
  Latency: ~1 RTT (local)

Strong consistency (EC):
  Client → Leader → Replicate to Majority → ACK
  Latency: ~2-3 RTTs (may cross data centers)

Numbers (US East to US West):
  Local replica: ~1 ms
  Cross-region replication: ~60-80 ms
  Quorum write (majority across 3 regions): ~120-160 ms
```

### 9.5 조정 가능한 일관성

일부 시스템은 연산별로 일관성을 선택할 수 있게 합니다:

```python
# Cassandra-style tunable consistency
def write(key, value, consistency_level):
    """
    consistency_level determines how many replicas must acknowledge.
    """
    if consistency_level == "ONE":
        # Write to 1 replica, acknowledge immediately
        # Fast but weak consistency (PA/EL)
        replicate_async(key, value)
        return "OK"

    elif consistency_level == "QUORUM":
        # Write to majority of replicas before acknowledging
        # Slower but strong consistency (PC/EC)
        acks = replicate_sync(key, value, count=quorum_size())
        if acks >= quorum_size():
            return "OK"
        else:
            raise TimeoutError("Could not reach quorum")

    elif consistency_level == "ALL":
        # Write to ALL replicas before acknowledging
        # Slowest, strongest, but least available
        acks = replicate_sync(key, value, count=total_replicas())
        if acks == total_replicas():
            return "OK"
        else:
            raise TimeoutError("Could not reach all replicas")


def quorum_size():
    """Quorum = floor(n/2) + 1 for n replicas."""
    return total_replicas() // 2 + 1
```

**Quorum intersection 보장**: `W + R > N` (write quorum + read quorum > 전체 replica 수)이면, 모든 read quorum이 모든 write quorum과 겹쳐서, 읽기가 최신 쓰기를 볼 수 있도록 보장합니다. 이것이 Cassandra와 같은 시스템이 조정 가능한 strong consistency를 제공하는 방식입니다.

---

## 10. Jepsen 스타일 일관성 검증

### 10.1 Jepsen이란?

Jepsen (Kyle Kingsbury)은 분산 시스템의 일관성 주장을 테스트하기 위한 프레임워크입니다. 다음을 수행합니다:

1. 분산 시스템을 설정 (예: 데이터베이스 클러스터)
2. 워크로드를 실행 (동시 읽기와 쓰기)
3. 장애를 주입 (네트워크 파티션, 프로세스 crash, clock skew)
4. 모든 연산의 history를 기록 (호출과 응답)
5. history가 주장된 일관성 모델을 만족하는지 검증

### 10.2 일관성 검증 작동 방식

```
Step 1: Run workload and record history

  Client 1: write(x, 1) [t=100, ok at t=110]
  Client 2: read(x) [t=105, returned 0 at t=108]
  Client 3: write(x, 2) [t=112, ok at t=120]
  Client 2: read(x) [t=115, returned 2 at t=118]
  ...

Step 2: Model the history

  Operation 1: write(x, 1), invoked=100, completed=110
  Operation 2: read(x)→0,   invoked=105, completed=108
  Operation 3: write(x, 2), invoked=112, completed=120
  Operation 4: read(x)→2,   invoked=115, completed=118

Step 3: Check linearizability

  Try to find a linearization (total order respecting real-time):
  Op2 must come before Op1 completes (Op2 finishes at 108, Op1 finishes at 110).
  But Op2 overlaps with Op1 (invoked 105 < 110).
  Both orders are possible for Op1 and Op2.

  If Op2 is linearized before Op1: read(x)→0, then write(x,1). OK (x starts at 0). ✓
  If Op1 is linearized before Op2: write(x,1), then read(x)→0. NOT OK (should return 1). ✗

  So only the first linearization works for Op1 and Op2.
  Continue with Op3 and Op4...
```

### 10.3 주요 Jepsen 발견 사항

| 시스템 | 주장 | 발견 | 문제 |
|--------|------|------|------|
| MongoDB (2013) | Strong consistency | 파티션 중 쓰기 유실 | 커밋된 데이터의 rollback |
| Elasticsearch (2014) | Sequential consistency | Split brain | 적절한 quorum 없음 |
| RabbitMQ (2014) | Durable queue | 메시지 유실 | 파티션 처리 버그 |
| etcd (2020) | Linearizable | Stale read | lease 기반 읽기 오류 |
| Redis Cluster (2013) | Consistent | 쓰기 유실 | 쓰기에 대한 합의 없음 |
| CockroachDB (2020) | Serializable | 올바름 | 잘 구현된 Raft |

---

## 11. 코드: Linearizability Checker

### 11.1 데이터 구조

```python
"""
Linearizability checker for a read-write register.
Uses the Wing & Gong algorithm with pruning.
"""

from dataclasses import dataclass
from itertools import permutations
from typing import Optional


@dataclass
class Operation:
    """An operation on a read-write register."""
    op_id: int
    op_type: str      # "write" or "read"
    key: str
    value: Optional[int]        # value written (write) or value read (read)
    invoke_time: float
    complete_time: float

    def __repr__(self):
        if self.op_type == "write":
            return f"w({self.key},{self.value})[{self.invoke_time}-{self.complete_time}]"
        else:
            return f"r({self.key})→{self.value}[{self.invoke_time}-{self.complete_time}]"


def real_time_precedes(op1: Operation, op2: Operation) -> bool:
    """Check if op1 completes before op2 starts (real-time precedence)."""
    return op1.complete_time < op2.invoke_time


def is_valid_sequential_history(ops: list[Operation], initial_value: int = 0) -> bool:
    """
    Check if a sequence of operations is a valid sequential execution
    of a read-write register.
    """
    store: dict[str, int] = {}

    for op in ops:
        if op.op_type == "write":
            store[op.key] = op.value
        elif op.op_type == "read":
            current = store.get(op.key, initial_value)
            if op.value != current:
                return False
    return True


def check_linearizability_brute(
    history: list[Operation],
    initial_value: int = 0,
) -> tuple[bool, Optional[list[Operation]]]:
    """
    Brute-force linearizability checker.
    Try all permutations and check if any is a valid linearization.

    WARNING: O(n!) complexity. Only suitable for small histories (n ≤ 10).
    """
    n = len(history)

    for perm in permutations(range(n)):
        ordered = [history[i] for i in perm]

        # Check real-time constraint
        valid_rt = True
        for i in range(n):
            for j in range(i + 1, n):
                original_i = perm[i]
                original_j = perm[j]
                # If op at position j precedes op at position i in real time,
                # then this ordering violates real-time constraint
                if real_time_precedes(history[original_j], history[original_i]):
                    valid_rt = False
                    break
            if not valid_rt:
                break

        if not valid_rt:
            continue

        # Check sequential specification
        if is_valid_sequential_history(ordered, initial_value):
            return (True, ordered)

    return (False, None)
```

### 11.2 효율적인 Linearizability Checker

```python
def check_linearizability_wg(
    history: list[Operation],
    initial_value: int = 0,
) -> tuple[bool, Optional[list[Operation]]]:
    """
    Wing & Gong style linearizability checker with pruning.
    Uses DFS with backtracking, pruning invalid prefixes early.

    Much faster than brute force for practical histories.
    """
    n = len(history)
    if n == 0:
        return (True, [])

    # Build precedence graph
    # must_precede[i] = set of op indices that must come before op i
    must_precede: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and real_time_precedes(history[i], history[j]):
                must_precede[j].add(i)

    # DFS with backtracking
    linearization: list[int] = []
    used: set[int] = set()
    store: dict[str, int] = {}  # current register state

    def dfs() -> bool:
        if len(linearization) == n:
            return True

        # Find candidate operations: those whose predecessors are all used
        candidates = []
        for i in range(n):
            if i not in used and must_precede[i].issubset(used):
                candidates.append(i)

        for i in candidates:
            op = history[i]

            # Check if adding this operation is consistent
            if op.op_type == "read":
                current = store.get(op.key, initial_value)
                if op.value != current:
                    continue  # prune: read returns wrong value

            # Apply operation
            old_value = store.get(op.key)
            if op.op_type == "write":
                store[op.key] = op.value

            linearization.append(i)
            used.add(i)

            if dfs():
                return True

            # Backtrack
            linearization.pop()
            used.discard(i)
            if op.op_type == "write":
                if old_value is not None:
                    store[op.key] = old_value
                else:
                    del store[op.key]

        return False

    if dfs():
        return (True, [history[i] for i in linearization])
    return (False, None)
```

### 11.3 Checker 테스트

```python
def test_linearizability_checker():
    """Test the linearizability checker with known examples."""

    print("="*60)
    print("LINEARIZABILITY CHECKER TESTS")
    print("="*60)

    # Test 1: Simple linearizable history
    h1 = [
        Operation(0, "write", "x", 1, 0, 10),
        Operation(1, "read", "x", 1, 15, 20),
    ]
    result, order = check_linearizability_wg(h1)
    print(f"\nTest 1 (simple write-read): {'PASS' if result else 'FAIL'}")
    if order:
        print(f"  Linearization: {order}")

    # Test 2: Non-linearizable (stale read after write completes)
    h2 = [
        Operation(0, "write", "x", 1, 0, 10),
        Operation(1, "write", "x", 2, 15, 25),
        Operation(2, "read", "x", 1, 30, 35),  # reads old value after both writes
    ]
    result, order = check_linearizability_wg(h2)
    print(f"\nTest 2 (stale read): {'PASS' if not result else 'FAIL'}")
    print(f"  Linearizable: {result} (expected False)")

    # Test 3: Concurrent operations (linearizable)
    h3 = [
        Operation(0, "write", "x", 1, 0, 15),
        Operation(1, "write", "x", 2, 5, 20),   # overlaps with write(x,1)
        Operation(2, "read", "x", 2, 25, 30),
    ]
    result, order = check_linearizability_wg(h3)
    print(f"\nTest 3 (concurrent writes, read 2): {'PASS' if result else 'FAIL'}")
    if order:
        print(f"  Linearization: {order}")

    # Test 4: Concurrent reads returning different values (linearizable if writes overlap)
    h4 = [
        Operation(0, "write", "x", 1, 0, 20),
        Operation(1, "write", "x", 2, 10, 30),  # overlaps
        Operation(2, "read", "x", 1, 12, 18),   # during overlap, reads 1
        Operation(3, "read", "x", 2, 15, 25),   # during overlap, reads 2
    ]
    result, order = check_linearizability_wg(h4)
    print(f"\nTest 4 (concurrent everything): {'PASS' if result else 'FAIL'}")
    if order:
        print(f"  Linearization: {order}")

    # Test 5: Clearly non-linearizable (time travel)
    h5 = [
        Operation(0, "write", "x", 1, 0, 10),
        Operation(1, "write", "x", 2, 20, 30),
        Operation(2, "read", "x", 2, 35, 40),
        Operation(3, "read", "x", 1, 45, 50),  # reads 1 after reading 2!
    ]
    result, order = check_linearizability_wg(h5)
    print(f"\nTest 5 (time-travel read): {'PASS' if not result else 'FAIL'}")
    print(f"  Linearizable: {result} (expected False)")

    # Test 6: Multi-key history
    h6 = [
        Operation(0, "write", "x", 1, 0, 10),
        Operation(1, "write", "y", 2, 5, 15),
        Operation(2, "read", "x", 1, 20, 25),
        Operation(3, "read", "y", 2, 22, 28),
    ]
    result, order = check_linearizability_wg(h6)
    print(f"\nTest 6 (multi-key): {'PASS' if result else 'FAIL'}")
    if order:
        print(f"  Linearization: {order}")


test_linearizability_checker()
```

---

## 12. 코드: 일관성 수준 시뮬레이션

### 12.1 다중 수준 일관성 저장소

```python
"""
Simulate a replicated key-value store operating at different
consistency levels. Shows the observable differences between
linearizable, sequentially consistent, causally consistent,
and eventually consistent stores.
"""

import random
import time
import threading
from collections import defaultdict
from enum import Enum


class ConsistencyLevel(Enum):
    LINEARIZABLE = "linearizable"
    SEQUENTIAL = "sequential"
    CAUSAL = "causal"
    EVENTUAL = "eventual"


class ReplicatedStore:
    """
    A simulated replicated store with configurable consistency.
    """

    def __init__(
        self,
        num_replicas: int = 3,
        consistency: ConsistencyLevel = ConsistencyLevel.LINEARIZABLE,
        replication_delay_ms: float = 50,
    ):
        self.num_replicas = num_replicas
        self.consistency = consistency
        self.replication_delay = replication_delay_ms / 1000

        # Each replica has its own copy of the data
        self.replicas: list[dict[str, tuple[int, float]]] = [
            {} for _ in range(num_replicas)
        ]  # key -> (value, write_timestamp)

        # Replication queue (for async replication)
        self.replication_queue: list[tuple[str, int, float, int]] = []
        # (key, value, timestamp, source_replica)

        # Operation history for checking
        self.history: list[dict] = []
        self.lock = threading.Lock()
        self.global_order_counter = 0

    def write(self, key: str, value: int, client_id: str = "default") -> dict:
        """Write a value to the store."""
        with self.lock:
            self.global_order_counter += 1
            ts = time.time()
            op = {
                "type": "write",
                "key": key,
                "value": value,
                "client": client_id,
                "invoke_time": ts,
                "global_order": self.global_order_counter,
            }

            if self.consistency == ConsistencyLevel.LINEARIZABLE:
                # Synchronous replication to majority before acknowledging
                quorum = self.num_replicas // 2 + 1
                for i in range(quorum):
                    self.replicas[i][key] = (value, ts)
                # Async replication to remaining
                for i in range(quorum, self.num_replicas):
                    self.replication_queue.append((key, value, ts, 0))

            elif self.consistency == ConsistencyLevel.SEQUENTIAL:
                # Write to primary (replica 0) only, async to others
                self.replicas[0][key] = (value, ts)
                for i in range(1, self.num_replicas):
                    self.replication_queue.append((key, value, ts, 0))

            elif self.consistency == ConsistencyLevel.CAUSAL:
                # Write to local replica, track causal dependencies
                replica_id = hash(client_id) % self.num_replicas
                self.replicas[replica_id][key] = (value, ts)
                for i in range(self.num_replicas):
                    if i != replica_id:
                        self.replication_queue.append((key, value, ts, replica_id))

            elif self.consistency == ConsistencyLevel.EVENTUAL:
                # Write to random replica, lazy propagation
                replica_id = random.randint(0, self.num_replicas - 1)
                self.replicas[replica_id][key] = (value, ts)
                for i in range(self.num_replicas):
                    if i != replica_id:
                        self.replication_queue.append((key, value, ts, replica_id))

            op["complete_time"] = time.time()
            self.history.append(op)
            return op

    def read(self, key: str, client_id: str = "default") -> dict:
        """Read a value from the store."""
        with self.lock:
            self.global_order_counter += 1
            ts = time.time()
            op = {
                "type": "read",
                "key": key,
                "client": client_id,
                "invoke_time": ts,
                "global_order": self.global_order_counter,
            }

            if self.consistency == ConsistencyLevel.LINEARIZABLE:
                # Read from quorum (majority)
                quorum = self.num_replicas // 2 + 1
                values = []
                for i in range(quorum):
                    if key in self.replicas[i]:
                        values.append(self.replicas[i][key])
                # Return the latest value
                if values:
                    latest = max(values, key=lambda x: x[1])
                    op["value"] = latest[0]
                else:
                    op["value"] = None

            elif self.consistency == ConsistencyLevel.SEQUENTIAL:
                # Read from primary (ensures program order)
                if key in self.replicas[0]:
                    op["value"] = self.replicas[0][key][0]
                else:
                    op["value"] = None

            elif self.consistency == ConsistencyLevel.CAUSAL:
                # Read from local replica (may be stale for non-causal)
                replica_id = hash(client_id) % self.num_replicas
                if key in self.replicas[replica_id]:
                    op["value"] = self.replicas[replica_id][key][0]
                else:
                    op["value"] = None

            elif self.consistency == ConsistencyLevel.EVENTUAL:
                # Read from random replica (may be very stale)
                replica_id = random.randint(0, self.num_replicas - 1)
                if key in self.replicas[replica_id]:
                    op["value"] = self.replicas[replica_id][key][0]
                else:
                    op["value"] = None

            op["complete_time"] = time.time()
            self.history.append(op)
            return op

    def apply_pending_replications(self, fraction: float = 1.0):
        """
        Apply a fraction of pending replications.
        Simulates async replication delay.
        """
        with self.lock:
            n = int(len(self.replication_queue) * fraction)
            to_apply = self.replication_queue[:n]
            self.replication_queue = self.replication_queue[n:]

            for key, value, ts, source in to_apply:
                for i in range(self.num_replicas):
                    if i != source:
                        existing = self.replicas[i].get(key)
                        if existing is None or ts > existing[1]:
                            self.replicas[i][key] = (value, ts)


def demonstrate_consistency_differences():
    """
    Run the same workload under different consistency levels
    and observe the behavioral differences.
    """
    print("="*70)
    print("CONSISTENCY LEVEL COMPARISON")
    print("="*70)

    for level in ConsistencyLevel:
        print(f"\n{'─'*70}")
        print(f"Consistency Level: {level.value.upper()}")
        print(f"{'─'*70}")

        store = ReplicatedStore(
            num_replicas=3,
            consistency=level,
            replication_delay_ms=50,
        )

        # Workload: Client A writes, Client B reads
        store.write("x", 1, client_id="A")
        store.write("x", 2, client_id="A")

        # Apply some replications (simulate partial convergence)
        store.apply_pending_replications(fraction=0.5)

        # Client B reads immediately
        read1 = store.read("x", client_id="B")
        print(f"  Client A writes x=1, then x=2")
        print(f"  Client B reads x → {read1['value']}")

        # Apply remaining replications
        store.apply_pending_replications(fraction=1.0)

        # Client B reads again after convergence
        read2 = store.read("x", client_id="B")
        print(f"  After convergence, Client B reads x → {read2['value']}")

        # Stale read check
        if read1['value'] != 2:
            print(f"  ⚠ Stale read detected! (expected 2, got {read1['value']})")
        else:
            print(f"  ✓ Read returned latest value")

        # Check: do all replicas agree?
        values = set()
        for r in store.replicas:
            if "x" in r:
                values.add(r["x"][0])
        if len(values) <= 1:
            print(f"  ✓ All replicas converged to same value")
        else:
            print(f"  ⚠ Replicas diverged: {values}")


demonstrate_consistency_differences()
```

### 12.2 이상 탐지

```python
def detect_anomalies(history: list[dict]) -> list[str]:
    """
    Detect common consistency anomalies in a recorded history.
    """
    anomalies = []

    # Group operations by client
    by_client: dict[str, list[dict]] = defaultdict(list)
    for op in history:
        by_client[op["client"]].append(op)

    # Check read-your-writes
    for client, ops in by_client.items():
        last_write: dict[str, int] = {}
        for op in sorted(ops, key=lambda x: x["invoke_time"]):
            if op["type"] == "write":
                last_write[op["key"]] = op["value"]
            elif op["type"] == "read":
                key = op["key"]
                if key in last_write and op["value"] != last_write[key]:
                    anomalies.append(
                        f"READ-YOUR-WRITES violation: {client} wrote "
                        f"{key}={last_write[key]} but read {key}={op['value']}"
                    )

    # Check monotonic reads
    for client, ops in by_client.items():
        last_read: dict[str, int] = {}
        for op in sorted(ops, key=lambda x: x["invoke_time"]):
            if op["type"] == "read" and op["value"] is not None:
                key = op["key"]
                if key in last_read:
                    # Check if value went "backward" (crude check using value ordering)
                    if op["value"] < last_read[key]:
                        anomalies.append(
                            f"MONOTONIC-READS violation: {client} read "
                            f"{key}={last_read[key]} then {key}={op['value']}"
                        )
                last_read[key] = op["value"]

    # Check for stale reads (read returning old value after a write completed)
    writes_by_key: dict[str, list[dict]] = defaultdict(list)
    reads_by_key: dict[str, list[dict]] = defaultdict(list)
    for op in history:
        if op["type"] == "write":
            writes_by_key[op["key"]].append(op)
        else:
            reads_by_key[op["key"]].append(op)

    for key in reads_by_key:
        for read_op in reads_by_key[key]:
            for write_op in writes_by_key.get(key, []):
                # If write completed before read started, read should see it (linearizable)
                if (write_op["complete_time"] < read_op["invoke_time"] and
                        read_op["value"] is not None and
                        read_op["value"] != write_op["value"]):
                    # Check if there is a later write that the read might have seen
                    later_writes = [
                        w for w in writes_by_key[key]
                        if w["complete_time"] <= read_op["invoke_time"]
                    ]
                    if later_writes:
                        latest = max(later_writes, key=lambda w: w["complete_time"])
                        if read_op["value"] != latest["value"]:
                            anomalies.append(
                                f"STALE-READ: read({key})→{read_op['value']} "
                                f"but latest completed write was {key}={latest['value']}"
                            )

    return anomalies
```

---

## 13. 요약

### 일관성 모델 결정 가이드

```
                     Need real-time guarantees?
                    /                            \
                 Yes                              No
                  |                                |
           Linearizable                    Need program-order?
           (Spanner, etcd)                /                  \
                                        Yes                   No
                                         |                     |
                                  Sequential            Need causality?
                                  Consistency           /              \
                                  (ZooKeeper)         Yes              No
                                                       |                |
                                                 Causal              Eventual
                                                 Consistency         Consistency
                                                 (MongoDB)           (DynamoDB)
```

### 핵심 요점

1. **Linearizability**는 최고 표준입니다: 연산이 원자적으로 나타나며 실시간 순서를 존중합니다. 하지만 조율 (quorum 읽기/쓰기)이 필요하고 파티션 중 가용성을 희생합니다.

2. **Sequential consistency**는 실시간 순서를 완화하지만 프로그램 순서를 보존합니다. Linearizability보다 저렴하지만 합성 불가능합니다 (비지역적).

3. **Causal consistency**는 원인-결과를 존중하지만 동시 연산이 발산하는 것을 허용합니다. 높은 가용성으로 달성 가능합니다 (읽기에 quorum 불필요).

4. **Eventual consistency**는 수렴만 보장합니다. 가장 저렴하고 가장 가용하지만, 애플리케이션 수준의 충돌 처리가 필요합니다.

5. **CAP 정리**는 실제로 파티션 트레이드오프에 관한 것입니다: 파티션 중에 consistency (CP) 또는 availability (AP)를 선택합니다. 대부분의 실제 시스템은 중요 데이터에 CP, 비중요 데이터에 AP를 선택합니다.

6. **PACELC**는 CAP을 확장하여 정상 운영 중의 latency-consistency 트레이드오프를 포함하며, 이것이 파티션 트레이드오프보다 더 영향력이 큰 경우가 많습니다.

7. **주장을 검증하라**: Jepsen 스타일 테스트를 사용하여 시스템이 실제로 주장하는 일관성을 제공하는지 확인하십시오. 많은 프로덕션 시스템이 명시된 보장을 위반하는 것이 발견되었습니다.

---

## 14. 연습 문제

### 문제 1: Linearizability 검증

각 history가 linearizable한지 판별하시오 (모든 레지스터의 초기값은 0):

```
History A:
  Client 1: w(x,1) [t=0, t=10]
  Client 2: r(x)→1 [t=5, t=15]
  Client 3: w(x,2) [t=12, t=20]
  Client 3: r(x)→2 [t=25, t=30]

History B:
  Client 1: w(x,1) [t=0, t=10]
  Client 2: w(x,2) [t=5, t=15]
  Client 3: r(x)→1 [t=20, t=25]
  Client 4: r(x)→2 [t=22, t=28]

History C:
  Client 1: w(x,1) [t=0, t=10]
  Client 1: w(x,2) [t=15, t=25]
  Client 2: r(x)→2 [t=20, t=30]
  Client 2: r(x)→1 [t=35, t=40]
```

### 문제 2: CAP 분류

각 시나리오에서 시스템이 CP와 AP 중 어느 것을 선택하고 있는지 설명하시오:

1. 네트워크 파티션 중에 데이터베이스가 모든 쓰기 연산에 오류를 반환하지만, 로컬 replica에서 읽기 연산은 계속 서비스합니다.
2. 파티션 중에 키-값 저장소가 읽기와 쓰기를 모두 계속 받아들이지만, 파티션이 치유될 때 쓰기가 유실될 수 있습니다.
3. 시스템이 quorum 읽기와 쓰기 (W + R > N)를 사용합니다. quorum 미만의 노드에만 도달 가능한 파티션 중에는 모든 연산이 실패합니다.

### 문제 3: PACELC 분석

각 시스템을 PACELC로 분류하고 정당화하시오:

1. 파티션 중에 오래된 데이터를 서비스하고 항상 가장 가까운 replica에서 데이터를 반환하는 캐시.
2. 파티션 중에 차단되고 정상 운영 중에도 과반수 확인 응답을 기다리는 금융 데이터베이스.
3. 파티션 중에 오래된 데이터를 서비스하지만 정상 운영 중에는 좋아요 수를 동기적으로 복제하는 소셜 미디어 피드.

### 문제 4: 세션 보장 위반

클라이언트가 복제된 저장소에 대해 다음 연산을 수행합니다:

```
write(x, 5) → OK (to Replica A)
read(x) → 3         (from Replica B, stale)
read(x) → 5         (from Replica A)
read(x) → 3         (from Replica B, still stale)
```

어떤 세션 보장이 위반되었습니까? 각 위반을 어떻게 수정하겠습니까?

### 문제 5: 구현 도전

linearizability checker를 다음과 같이 확장하시오:

1. 읽기와 쓰기 외에 compare-and-swap (CAS) 연산 지원
2. 불완전한 연산 처리 (호출되었지만 응답을 받지 못한 연산 -- 연산이 효력을 발휘했거나 아니거나)
3. linearization의 시각화 생성 (각 연산의 linearization point가 어디에 놓이는지 보여주는 타임라인)

---

## 15. 참고 문헌

1. Herlihy, M. P., & Wing, J. M. (1990). "Linearizability: A Correctness Condition for Concurrent Objects." *ACM Transactions on Programming Languages and Systems*, 12(3), 463-492.
2. Lamport, L. (1979). "How to Make a Multiprocessor Computer That Correctly Executes Multiprocess Programs." *IEEE Transactions on Computers*, C-28(9), 690-691.
3. Gilbert, S., & Lynch, N. (2002). "Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services." *ACM SIGACT News*, 33(2), 51-59.
4. Brewer, E. (2012). "CAP Twelve Years Later: How the 'Rules' Have Changed." *IEEE Computer*, 45(2), 23-29.
5. Abadi, D. (2012). "Consistency Tradeoffs in Modern Distributed Database System Design." *IEEE Computer*, 45(2), 37-42.
6. Ahamad, M., et al. (1995). "Causal Memory: Definitions, Implementation, and Programming." *Distributed Computing*, 9(1), 37-49.
7. Terry, D. B., et al. (1994). "Session Guarantees for Weakly Consistent Replicated Data." *PDIS 1994*.
8. Attiya, H., & Welch, J. (1994). "Sequential Consistency versus Linearizability." *ACM Transactions on Computer Systems*, 12(2), 91-122.
9. Gibbons, P. B., & Korach, E. (1997). "Testing Shared Memories." *SIAM Journal on Computing*, 26(4), 1208-1244.
10. Kingsbury, K. "Jepsen: Analyses of Distributed Systems." https://jepsen.io/analyses.
11. Lloyd, W., et al. (2011). "Don't Settle for Eventual: Scalable Causal Consistency for Wide-Area Storage with COPS." *SOSP 2011*.
12. Viotti, P., & Vukolic, M. (2016). "Consistency in Non-Transactional Distributed Storage Systems." *ACM Computing Surveys*, 49(1), 1-34.

---

[다음: 레슨 05 - Paxos 계열](./05_Paxos_Family.md)
