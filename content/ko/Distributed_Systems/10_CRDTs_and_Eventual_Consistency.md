# 레슨 10: CRDT와 최종 일관성

[개요](./00_Overview.md) | [이전: 복제 전략](./09_Replication_Strategies.md) | [다음: 파티셔닝과 샤딩](./11_Partitioning_and_Sharding.md)

---

## 학습 목표

- 최종 일관성(eventual consistency), 수렴(convergence), 강한 최종 일관성(SEC)을 구분된 보장으로 형식화
- CRDT의 수학적 기반 이해: join-semilattice, 교환법칙, 결합법칙, 멱등성
- 상태 기반 CRDT(G-Counter, PN-Counter, G-Set, 2P-Set, LWW-Register, OR-Set)를 처음부터 구현
- 상태 기반 vs 연산 기반 CRDT 비교 및 트레이드오프 분석
- CRDT의 한계 평가 및 프로덕션 시스템의 실용적 적용 사례 파악

---

## 1. 문제: 조정 없는 동시 업데이트

복수의 writer가 있는 복제 시스템(multi-leader 또는 leaderless)에서 동일 데이터에 대한 동시 업데이트는 불가피하다. 레슨 9에서 이것이 충돌을 만든다는 것을 알았다. 전통적 해결책 — 합의 프로토콜이나 single-leader 직렬화 — 은 지연을 추가하고 가용성을 감소시킨다.

**조정 없이 자동으로 수렴하는 데이터 구조를 설계할 수 있을까?**

이것이 CRDT (Conflict-free Replicated Data Types)가 답하는 핵심 질문이다.

### 1.1 동기 부여 예제

"좋아요 카운터"가 있는 협업 문서를 생각해보자. 세 복제본이 독립적으로 증가를 처리한다:

```
Time ──────────────────────────────────────────────►

Replica A:  counter=0  ──► +1 ──► counter=1
Replica B:  counter=0  ──► +1 ──► +1 ──► counter=2
Replica C:  counter=0  ──► +1 ──► counter=1

Merging: What should the final count be?
  - Simple addition: 1 + 2 + 1 = 4?  (WRONG — double-counts the initial 0)
  - Max: max(1, 2, 1) = 2?           (WRONG — loses A and C increments)
  - ???: The correct answer is 4 total increments
```

단순한 정수 카운터는 올바르게 병합할 수 없다. 업데이트를 잃거나 이중 계산하지 않고 병합할 수 있도록 *충분한 정보를 기록하는* 데이터 구조가 필요하다.

### 1.2 왜 합의를 사용하지 않는가?

| 접근법 | 지연 시간 | 가용성 | 복잡도 |
|---|---|---|---|
| 합의 (Paxos/Raft) | 높음 (다중 왕복) | 과반수 필요 | 높음 |
| Single-leader | 중간 (leader 왕복) | Leader가 SPOF | 중간 |
| CRDT | 없음 (로컬 연산) | 항상 가용 | 중간 (설계 시점) |

CRDT는 복잡성을 **런타임 조정**에서 **데이터 구조 설계**로 이동시킨다. 연산은 네트워크 왕복 없이 로컬에서 적용된다. 복제본은 비동기적으로 동기화하며 *수학적으로 수렴이 보장*된다.

---

## 2. 최종 일관성의 형식화

CRDT를 살펴보기 전에, "최종 일관성"이 정확히 무엇을 의미하는지 정의해야 한다. 이 용어는 종종 느슨하게 사용되며, 여기서는 세 가지 수준을 구분한다.

### 2.1 기본 최종 일관성

**정의**: 새로운 업데이트가 없으면, *결국* 모든 복제본이 동일한 값으로 수렴한다.

이것은 약한 보장이다:
- 수렴에 *얼마나 걸리는지*는 말하지 않는다
- 수렴 중에 복제본이 다른 값을 반환할 수 있다
- *어떤* 값으로 수렴하는지 보장하지 않는다

### 2.2 강한 수렴(Strong Convergence)

**정의**: *동일한 업데이트 집합*을 받은 복제본은 업데이트가 수신된 순서에 관계없이 *동일한 상태*에 있다.

이것은 엄밀히 더 강하다: 수렴이 *결정적*이며 업데이트의 *집합*에만 의존하고 순서에는 의존하지 않음을 보장한다.

### 2.3 강한 최종 일관성 (SEC)

**정의**: 시스템이 SEC를 제공하려면:
1. **최종 전달(Eventual delivery)**: 하나의 올바른 복제본에 적용된 모든 업데이트가 결국 모든 올바른 복제본에 적용됨
2. **강한 수렴**: 동일한 업데이트 집합을 받은 복제본이 동일한 상태에 있음
3. **종료(Termination)**: 모든 연산이 유한 시간 내에 완료됨 (차단 없음)

SEC는 CRDT가 제공하는 보장이다. 기본 최종 일관성보다 강하지만 linearizability나 순차 일관성보다 약하다.

```
Strength hierarchy:

  Linearizability (strongest)
       │
  Sequential Consistency
       │
  Causal Consistency
       │
  Strong Eventual Consistency (SEC)  ← CRDTs guarantee this
       │
  Basic Eventual Consistency
       │
  No consistency (weakest)
```

---

## 3. 수학적 기반

CRDT는 병합 연산을 **join-semilattice**의 **join**으로 제한하여 SEC를 달성한다.

### 3.1 Join-Semilattice

**Join-semilattice**는 다음을 만족하는 이항 연산 ⊔ (join/merge)을 가진 집합 S이다:

| 속성 | 정의 | 직관 |
|---|---|---|
| **교환법칙** | a ⊔ b = b ⊔ a | 병합 순서가 중요하지 않음 |
| **결합법칙** | (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c) | 병합 그룹핑이 중요하지 않음 |
| **멱등성** | a ⊔ a = a | 동일한 상태를 두 번 병합해도 효과 없음 |

이 세 가지 속성이 함께 보장하는 것:
- 병합 연산의 모든 순열이 동일한 결과를 생성 (교환법칙 + 결합법칙)
- 중복 메시지가 무해함 (멱등성)
- 조정이 필요 없음 — 다른 복제본에서 상태를 받을 때마다 병합하면 됨

### 3.2 부분 순서(Partial Order)

Join-semilattice는 다음과 같이 정의되는 부분 순서 ≤를 유도한다:

$$a \leq b \iff a \sqcup b = b$$

직관적으로: a ≤ b는 "b가 a의 모든 정보를 포함함(그리고 아마 더 많이)"을 의미한다. 병합 연산은 이 부분 순서에서 항상 *위로* 이동한다 — 상태가 단조적으로 성장한다.

```
Example: Set union as join-semilattice

      {a,b,c}        ← top (after all merges)
      ╱   │   ╲
  {a,b} {a,c} {b,c}
    ╱╲   ╱╲   ╱╲
  {a}  {b}  {c}
    ╲   │   ╱
       {}              ← bottom (initial state)

Merge = set union: {a,b} ⊔ {b,c} = {a,b,c}
```

### 3.3 왜 이것이 수렴을 보장하는가

**정리** (Shapiro et al., 2011): 각 복제본의 상태가 join-semilattice를 형성하고 병합 연산이 lattice join이면, 시스템은 강한 최종 일관성을 달성한다.

**증명 스케치**:
1. 모든 복제본이 바닥 원소에서 시작
2. 각 로컬 업데이트가 lattice에서 상태를 위로 이동
3. 원격 상태와의 병합도 join이므로 위로 이동
4. 교환법칙 + 결합법칙이 병합 순서에 관계없이 동일한 최종 상태를 보장
5. 멱등성이 중복 메시지를 무해하게 만듦
6. 모든 복제본이 모든 업데이트를 수신하면, 모두 동일한 lattice 원소에 위치

---

## 4. 상태 기반 CRDT (CvRDT)

상태 기반 CRDT (Convergent Replicated Data Types)는 **전체 상태**를 다른 복제본에 전송하고, lattice join을 사용하여 병합하는 방식으로 작동한다.

```
Replica A                    Replica B
    │                            │
    │  local operations          │  local operations
    │  ──────────────            │  ──────────────
    │                            │
    │──── send full state ──────►│
    │                            │  merge(local_state, received_state)
    │◄── send full state ────────│
    │  merge(local_state,        │
    │        received_state)     │
```

### 4.1 G-Counter (증가 전용 카운터)

가장 단순한 CRDT. 각 노드가 벡터에서 자신의 카운터를 유지한다. 전역 카운트는 모든 항목의 합이다.

**상태**: 노드별 카운트 벡터 `{node_id: count}`

**연산**:
- `increment(node_id)`: 이 노드의 항목 증가
- `value()`: 모든 항목의 합
- `merge(other)`: 원소별 최대값

```python
from __future__ import annotations
from typing import Any


class GCounter:
    """
    Grow-Only Counter (G-Counter).

    Each replica has its own slot in a vector. Increment only
    affects the local slot. Merge takes element-wise maximum.

    Lattice: the partial order is component-wise ≤
    Join: component-wise max
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: dict[str, int] = {}

    def increment(self, amount: int = 1) -> None:
        """Increment this node's counter. Only grows (amount must be >= 0)."""
        assert amount >= 0, "G-Counter can only grow"
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount

    def value(self) -> int:
        """The counter value is the sum of all node counts."""
        return sum(self.counts.values())

    def merge(self, other: GCounter) -> GCounter:
        """
        Merge with another G-Counter.
        Takes element-wise maximum — satisfies semilattice properties.
        """
        result = GCounter(self.node_id)
        all_nodes = set(self.counts.keys()) | set(other.counts.keys())
        for node in all_nodes:
            result.counts[node] = max(
                self.counts.get(node, 0),
                other.counts.get(node, 0),
            )
        return result

    def __repr__(self) -> str:
        return f"GCounter(value={self.value()}, counts={self.counts})"


def demo_g_counter():
    """Demonstrate G-Counter convergence."""
    # Three replicas
    a = GCounter("A")
    b = GCounter("B")
    c = GCounter("C")

    # Independent increments
    a.increment(3)  # A sees 3 clicks
    b.increment(5)  # B sees 5 clicks
    c.increment(2)  # C sees 2 clicks

    print(f"Before merge: A={a.value()}, B={b.value()}, C={c.value()}")

    # Merge A and B
    ab = a.merge(b)
    print(f"A merge B = {ab.value()}")  # 3 + 5 = 8

    # Merge result with C
    abc = ab.merge(c)
    print(f"(A merge B) merge C = {abc.value()}")  # 3 + 5 + 2 = 10

    # Verify commutativity: C merge (B merge A)
    ba = b.merge(a)
    cba = c.merge(ba)
    print(f"C merge (B merge A) = {cba.value()}")  # also 10

    # Verify idempotence: merging with self
    abc2 = abc.merge(abc)
    print(f"Idempotent: {abc.value()} == {abc2.value()}")  # True
```

**왜 원소별 최대값인가?** 덧셈을 사용하면 병합 시 이중 계산이 발생한다. 단순 `max`를 사용하면 독립적인 증가를 잃는다. 노드별 카운터에 대한 원소별 max는 올바른 의미를 제공한다: 각 노드의 기여가 정확히 한 번 계산된다.

### 4.2 PN-Counter (양음 카운터)

G-Counter는 증가만 가능하다. 감소를 지원하려면 *두 개의* G-Counter를 사용한다: 증가용(P)과 감소용(N). 값은 P - N이다.

```python
class PNCounter:
    """
    Positive-Negative Counter.

    Two G-Counters: P for increments, N for decrements.
    Value = P.value() - N.value()

    Supports both increment and decrement while maintaining
    CRDT convergence properties.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.p = GCounter(node_id)  # positive (increment) counter
        self.n = GCounter(node_id)  # negative (decrement) counter

    def increment(self, amount: int = 1) -> None:
        """Increment the counter."""
        self.p.increment(amount)

    def decrement(self, amount: int = 1) -> None:
        """Decrement the counter."""
        self.n.increment(amount)

    def value(self) -> int:
        """Net value = increments - decrements."""
        return self.p.value() - self.n.value()

    def merge(self, other: PNCounter) -> PNCounter:
        """Merge two PN-Counters by merging P and N independently."""
        result = PNCounter(self.node_id)
        result.p = self.p.merge(other.p)
        result.n = self.n.merge(other.n)
        return result

    def __repr__(self) -> str:
        return f"PNCounter(value={self.value()}, P={self.p.value()}, N={self.n.value()})"


def demo_pn_counter():
    """Demonstrate PN-Counter with increments and decrements."""
    a = PNCounter("A")
    b = PNCounter("B")

    # A increments 10 times, decrements 3 times
    a.increment(10)
    a.decrement(3)

    # B increments 5 times, decrements 1 time
    b.increment(5)
    b.decrement(1)

    print(f"A = {a.value()} (P={a.p.value()}, N={a.n.value()})")  # 10-3=7
    print(f"B = {b.value()} (P={b.p.value()}, N={b.n.value()})")  # 5-1=4

    # Merge: total value should be (10+5) - (3+1) = 11
    merged = a.merge(b)
    print(f"Merged = {merged.value()}")  # 11
```

### 4.3 G-Set (증가 전용 집합)

원소 추가만 지원하고 제거는 지원하지 않는 집합.

**Lattice**: 집합 합집합이 join이다.

```python
class GSet:
    """
    Grow-Only Set.

    Elements can only be added, never removed.
    Merge = set union (trivially a join-semilattice).
    """

    def __init__(self):
        self.elements: set[Any] = set()

    def add(self, element: Any) -> None:
        """Add an element. Irreversible."""
        self.elements.add(element)

    def lookup(self, element: Any) -> bool:
        """Check membership."""
        return element in self.elements

    def merge(self, other: GSet) -> GSet:
        """Merge = set union."""
        result = GSet()
        result.elements = self.elements | other.elements
        return result

    def value(self) -> set:
        return frozenset(self.elements)

    def __repr__(self) -> str:
        return f"GSet({self.elements})"
```

### 4.4 2P-Set (2단계 집합)

추가와 제거를 모두 지원하지만, 한 번 제거된 원소는 **다시 추가할 수 없다** (tombstone이 영구적).

```python
class TwoPSet:
    """
    Two-Phase Set (2P-Set).

    Two G-Sets: one for additions (A), one for removals (R).
    An element is in the set iff it is in A but not in R.

    Limitation: once removed, an element cannot be re-added.
    """

    def __init__(self):
        self.add_set = GSet()      # additions
        self.remove_set = GSet()   # removals (tombstones)

    def add(self, element: Any) -> None:
        """Add an element (only effective if not previously removed)."""
        self.add_set.add(element)

    def remove(self, element: Any) -> bool:
        """
        Remove an element. Must have been added first.
        Once removed, cannot be re-added (permanent tombstone).
        """
        if not self.add_set.lookup(element):
            return False  # can only remove elements that exist
        self.remove_set.add(element)
        return True

    def lookup(self, element: Any) -> bool:
        """Element is present iff in add_set and NOT in remove_set."""
        return self.add_set.lookup(element) and not self.remove_set.lookup(element)

    def value(self) -> set:
        return self.add_set.elements - self.remove_set.elements

    def merge(self, other: TwoPSet) -> TwoPSet:
        """Merge by independently merging add and remove sets."""
        result = TwoPSet()
        result.add_set = self.add_set.merge(other.add_set)
        result.remove_set = self.remove_set.merge(other.remove_set)
        return result

    def __repr__(self) -> str:
        return f"2PSet(value={self.value()})"
```

### 4.5 LWW-Register (Last-Writer-Wins 레지스터)

타임스탬프가 포함된 단일 값을 저장한다. 병합 시 더 높은 타임스탬프를 가진 값이 승리한다.

```python
import time as _time
from dataclasses import dataclass


@dataclass
class LWWRegister:
    """
    Last-Writer-Wins Register.

    Each update is tagged with a timestamp. Merge keeps the
    value with the highest timestamp.

    Lattice: ordered by timestamp (higher = greater in lattice)
    Join: max by timestamp

    WARNING: Relies on clock accuracy. Clock skew can cause
    "newer" writes to be silently overwritten.
    """

    value: Any = None
    timestamp: float = 0.0
    node_id: str = ""

    def update(self, value: Any, node_id: str,
               timestamp: float | None = None) -> None:
        """Update the register with a new value and timestamp."""
        ts = timestamp if timestamp is not None else _time.time()
        if ts > self.timestamp or (ts == self.timestamp and node_id > self.node_id):
            self.value = value
            self.timestamp = ts
            self.node_id = node_id

    def merge(self, other: LWWRegister) -> LWWRegister:
        """Merge: keep the value with the higher timestamp."""
        if other.timestamp > self.timestamp:
            return LWWRegister(other.value, other.timestamp, other.node_id)
        if other.timestamp == self.timestamp:
            # Deterministic tie-breaking: higher node_id wins
            if other.node_id > self.node_id:
                return LWWRegister(other.value, other.timestamp, other.node_id)
        return LWWRegister(self.value, self.timestamp, self.node_id)

    def __repr__(self) -> str:
        return f"LWWRegister(value={self.value}, ts={self.timestamp:.6f})"


def demo_lww_register():
    """Demonstrate LWW-Register with concurrent writes."""
    reg_a = LWWRegister()
    reg_b = LWWRegister()

    # Concurrent updates with explicit timestamps
    reg_a.update("Alice", "A", timestamp=100.0)
    reg_b.update("Bob", "B", timestamp=100.5)  # later timestamp

    # Merge: Bob wins (higher timestamp)
    merged = reg_a.merge(reg_b)
    print(f"Merged: {merged}")  # Bob

    # The problem: if clocks are skewed, older data can win
    reg_c = LWWRegister()
    reg_c.update("Charlie", "C", timestamp=200.0)  # skewed clock far ahead

    reg_d = LWWRegister()
    reg_d.update("Diana", "D", timestamp=101.0)  # real time, but lower

    merged2 = reg_c.merge(reg_d)
    print(f"Skewed: {merged2}")  # Charlie wins despite being "wrong"
```

### 4.6 MV-Register (다중 값 레지스터)

동시 쓰기를 버리는 대신(LWW), MV-Register는 모든 동시 값을 보존한다. 그러면 애플리케이션이 충돌을 해결할 수 있다.

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VectorClock:
    """Immutable vector clock for causality tracking."""
    clock: tuple[tuple[str, int], ...] = ()

    def _to_dict(self) -> dict[str, int]:
        return dict(self.clock)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> VectorClock:
        return cls(tuple(sorted(d.items())))

    def increment(self, node_id: str) -> VectorClock:
        d = self._to_dict()
        d[node_id] = d.get(node_id, 0) + 1
        return VectorClock.from_dict(d)

    def merge(self, other: VectorClock) -> VectorClock:
        d = self._to_dict()
        for node, count in other._to_dict().items():
            d[node] = max(d.get(node, 0), count)
        return VectorClock.from_dict(d)

    def __le__(self, other: VectorClock) -> bool:
        """a <= b iff all components of a are <= corresponding components of b."""
        d_self = self._to_dict()
        d_other = other._to_dict()
        for node, count in d_self.items():
            if count > d_other.get(node, 0):
                return False
        return True

    def concurrent_with(self, other: VectorClock) -> bool:
        """Two clocks are concurrent iff neither dominates the other."""
        return not (self <= other) and not (other <= self)


class MVRegister:
    """
    Multi-Value Register.

    Preserves all concurrent values instead of arbitrarily picking one.
    Uses vector clocks to determine causality.
    Clients see all concurrent values and can resolve conflicts.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # List of (value, vector_clock) pairs
        self.values: list[tuple[Any, VectorClock]] = []

    def update(self, value: Any) -> None:
        """
        Update with a new value. Supersedes all values causally
        dominated by this node's current knowledge.
        """
        # Merge all current vector clocks and increment
        merged_vc = VectorClock()
        for _, vc in self.values:
            merged_vc = merged_vc.merge(vc)
        new_vc = merged_vc.increment(self.node_id)

        # New value supersedes all existing values
        self.values = [(value, new_vc)]

    def read(self) -> list[Any]:
        """Return all concurrent values."""
        return [v for v, _ in self.values]

    def merge(self, other: MVRegister) -> MVRegister:
        """
        Merge: keep values that are not dominated by any value
        in the other register.
        """
        result = MVRegister(self.node_id)
        all_entries = self.values + other.values

        # Keep only entries not dominated by another entry
        surviving = []
        for val, vc in all_entries:
            dominated = False
            for other_val, other_vc in all_entries:
                if vc <= other_vc and vc != other_vc:
                    dominated = True
                    break
            if not dominated:
                # Deduplicate
                if (val, vc) not in surviving:
                    surviving.append((val, vc))
        result.values = surviving
        return result

    def __repr__(self) -> str:
        return f"MVRegister(values={self.read()})"


def demo_mv_register():
    """Demonstrate MV-Register preserving concurrent writes."""
    reg_a = MVRegister("A")
    reg_b = MVRegister("B")

    # Sequential update on A
    reg_a.update("version-1")
    reg_a.update("version-2")  # supersedes version-1

    # Concurrent update on B (doesn't know about A's updates)
    reg_b.update("version-X")

    print(f"A: {reg_a.read()}")  # ['version-2']
    print(f"B: {reg_b.read()}")  # ['version-X']

    # Merge: both concurrent values preserved
    merged = reg_a.merge(reg_b)
    print(f"Merged: {merged.read()}")  # ['version-2', 'version-X']
    # Application must resolve: e.g., pick one or merge semantically
```

### 4.7 OR-Set (관찰 제거 집합)

OR-Set은 가장 실용적인 CRDT 중 하나이다. 2P-Set과 달리, 제거 후 원소를 다시 추가할 수 있다. 각 추가에 고유 식별자가 태그되며, 제거는 *관찰된* 태그만 영향을 준다.

```python
import uuid
from typing import Any


class ORSet:
    """
    Observed-Remove Set (OR-Set).

    Each add() generates a unique tag. remove() removes all
    currently observed tags for an element. A concurrent add()
    on another replica (with a different tag) survives the remove.

    This is the most practical set CRDT: supports add, remove,
    and re-add with intuitive semantics.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # Map: element -> set of unique tags
        self.elements: dict[Any, set[str]] = {}
        # Tombstones: tags that have been removed
        self.tombstones: set[str] = set()

    def add(self, element: Any) -> str:
        """
        Add an element with a unique tag.
        Returns the tag (useful for testing).
        """
        tag = f"{self.node_id}:{uuid.uuid4().hex[:8]}"
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)
        return tag

    def remove(self, element: Any) -> bool:
        """
        Remove an element by tombstoning all its currently observed tags.
        Concurrent adds (with different tags) will survive this remove.
        """
        if element not in self.elements:
            return False
        tags = self.elements[element]
        if not tags:
            return False
        # Tombstone all observed tags
        self.tombstones.update(tags)
        self.elements[element] = set()
        return True

    def lookup(self, element: Any) -> bool:
        """Element is present iff it has at least one non-tombstoned tag."""
        if element not in self.elements:
            return False
        live_tags = self.elements[element] - self.tombstones
        return len(live_tags) > 0

    def value(self) -> set:
        """Return the set of all present elements."""
        result = set()
        for element, tags in self.elements.items():
            live_tags = tags - self.tombstones
            if live_tags:
                result.add(element)
        return result

    def merge(self, other: ORSet) -> ORSet:
        """
        Merge two OR-Sets.

        For each element:
        - Union all tags from both replicas
        - Union all tombstones from both replicas
        - An element is present iff it has tags not in tombstones
        """
        result = ORSet(self.node_id)

        # Union all elements and their tags
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        for element in all_elements:
            tags_a = self.elements.get(element, set())
            tags_b = other.elements.get(element, set())
            result.elements[element] = tags_a | tags_b

        # Union all tombstones
        result.tombstones = self.tombstones | other.tombstones

        return result

    def __repr__(self) -> str:
        return f"ORSet({self.value()})"


def demo_or_set():
    """Demonstrate OR-Set with concurrent add and remove."""
    set_a = ORSet("A")
    set_b = ORSet("B")

    # A adds "milk"
    set_a.add("milk")
    print(f"A after add milk: {set_a.value()}")  # {'milk'}

    # Sync: B receives A's state
    synced = set_a.merge(set_b)
    set_b = synced  # B now has "milk"
    set_b.node_id = "B"

    # Concurrent operations:
    # A removes "milk"
    set_a.remove("milk")
    print(f"A after remove milk: {set_a.value()}")  # set()

    # B concurrently adds "milk" again (different tag!)
    set_b.add("milk")
    print(f"B after concurrent add: {set_b.value()}")  # {'milk'}

    # Merge: B's add survives A's remove (add wins over concurrent remove)
    merged = set_a.merge(set_b)
    print(f"Merged (add wins): {merged.value()}")  # {'milk'}

    # This is the "add-wins" semantics of OR-Set:
    # A removed the tags it observed, but B added a new tag
    # that A's remove didn't know about.
```

---

## 5. 연산 기반 CRDT (CmRDT)

연산 기반 CRDT (Commutative Replicated Data Types)는 다른 접근 방식을 취한다: 전체 상태를 보내는 대신 **연산**을 다른 복제본에 전송한다.

### 5.1 요구 사항

연산 기반 CRDT가 수렴하려면 다음이 필요하다:

1. **교환 가능한 연산**: 연산이 적용되는 순서에 관계없이 결과가 동일해야 함
2. **신뢰할 수 있는 인과적 브로드캐스트**: 메시징 레이어가 모든 연산을 정확히 한 번, 인과적 순서로 전달해야 함

```
State-based:                    Operation-based:
  Send full state               Send operation (e.g., "increment by 1")
  Receiver merges                Receiver applies operation
  Idempotent (resend OK)         NOT idempotent (must deliver exactly once)
  Larger messages                Smaller messages
  Any network (even lossy)       Requires reliable causal broadcast
```

### 5.2 연산 기반 Counter

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class OpType(Enum):
    INCREMENT = "increment"
    DECREMENT = "decrement"


@dataclass
class Operation:
    """An operation to be broadcast to all replicas."""
    op_type: OpType
    amount: int
    source_node: str
    sequence_num: int  # for causal ordering


class OpBasedCounter:
    """
    Operation-based counter CRDT.

    Operations (increment, decrement) are broadcast to all replicas.
    Operations must be delivered exactly once in causal order.

    Commutativity: increment(a) then increment(b) = increment(b) then increment(a)
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.value_: int = 0
        self._seq = 0
        self._delivered: set[tuple[str, int]] = set()  # dedup: (source, seq)
        self.peers: list[OpBasedCounter] = []

    def increment(self, amount: int = 1) -> None:
        """Local increment + broadcast to peers."""
        self.value_ += amount
        self._seq += 1
        op = Operation(OpType.INCREMENT, amount, self.node_id, self._seq)
        self._broadcast(op)

    def decrement(self, amount: int = 1) -> None:
        """Local decrement + broadcast to peers."""
        self.value_ -= amount
        self._seq += 1
        op = Operation(OpType.DECREMENT, amount, self.node_id, self._seq)
        self._broadcast(op)

    def receive(self, op: Operation) -> None:
        """
        Receive and apply an operation from a peer.
        Must be exactly-once delivery (dedup check).
        """
        key = (op.source_node, op.sequence_num)
        if key in self._delivered:
            return  # duplicate — ignore
        self._delivered.add(key)

        if op.op_type == OpType.INCREMENT:
            self.value_ += op.amount
        elif op.op_type == OpType.DECREMENT:
            self.value_ -= op.amount

    def _broadcast(self, op: Operation) -> None:
        """Reliable broadcast to all peers."""
        for peer in self.peers:
            peer.receive(op)

    def __repr__(self) -> str:
        return f"OpCounter({self.node_id}, value={self.value_})"


def demo_op_counter():
    """Demonstrate operation-based counter."""
    a = OpBasedCounter("A")
    b = OpBasedCounter("B")
    c = OpBasedCounter("C")

    # Wire up peers
    a.peers = [b, c]
    b.peers = [a, c]
    c.peers = [a, b]

    # Operations at different nodes
    a.increment(5)
    b.increment(3)
    c.decrement(1)

    # All replicas converge
    print(f"A={a.value_}, B={b.value_}, C={c.value_}")  # all 7
    assert a.value_ == b.value_ == c.value_ == 7
```

### 5.3 연산 기반 vs 상태 기반: 트레이드오프

| 측면 | 상태 기반 (CvRDT) | 연산 기반 (CmRDT) |
|---|---|---|
| **메시지 크기** | 전체 상태 (클 수 있음) | 단일 연산 (작음) |
| **메시지 전달** | 아무거나 (손실, 순서 변경 OK) | 정확히 한 번, 인과적 순서 필요 |
| **멱등성** | 내장 (병합이 멱등) | 멱등이 아님 (중복 제거 필요) |
| **네트워크 요구** | 비신뢰적 (gossip 가능) | 신뢰할 수 있는 인과적 브로드캐스트 |
| **구현** | 더 단순한 병합 로직 | 더 단순한 연산별 로직 |
| **대역폭** | 높음 (상태 크기에 비례) | 낮음 (연산 수에 비례) |

**실제로는**: 상태 기반 CRDT가 네트워크 요구 사항이 약하기 때문에 더 일반적이다. Delta-state CRDT(섹션 6)는 양쪽의 장점을 결합한다.

---

## 6. Delta-State CRDT

Delta-state CRDT는 상태 기반 CRDT의 주요 약점을 해결한다: **전체 상태** 전송이 대규모 데이터 구조에 비용이 많이 든다.

### 6.1 핵심 아이디어

전체 상태를 보내는 대신, 마지막 동기화 이후의 최소 상태 변경인 **delta**만 전송한다. Delta 자체가 유효한 join-semilattice 원소여야 병합할 수 있다.

```
Full state sync:
  Replica A: {a:5, b:3, c:7} ──── sends 3 entries ────► Replica B

Delta sync:
  Replica A: only changed a:5→6 ── sends {a:6} ──────► Replica B
  Replica B: merge({a:6}) into local state
```

### 6.2 Delta G-Counter

```python
class DeltaGCounter:
    """
    Delta-state G-Counter.

    Tracks changes since last sync and sends only the delta.
    Reduces bandwidth from O(num_nodes) to O(changed_nodes) per sync.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: dict[str, int] = {}
        self._pending_delta: dict[str, int] = {}  # unsent changes

    def increment(self, amount: int = 1) -> None:
        """Increment and record the delta."""
        old = self.counts.get(self.node_id, 0)
        new = old + amount
        self.counts[self.node_id] = new
        self._pending_delta[self.node_id] = new

    def value(self) -> int:
        return sum(self.counts.values())

    def get_delta(self) -> dict[str, int]:
        """Get the pending delta (changes since last sync) and clear it."""
        delta = dict(self._pending_delta)
        self._pending_delta.clear()
        return delta

    def merge_delta(self, delta: dict[str, int]) -> None:
        """Merge a delta from another replica."""
        for node, count in delta.items():
            self.counts[node] = max(self.counts.get(node, 0), count)

    def merge_full(self, other: DeltaGCounter) -> None:
        """Full state merge (fallback for new replicas or recovery)."""
        for node, count in other.counts.items():
            self.counts[node] = max(self.counts.get(node, 0), count)

    def __repr__(self) -> str:
        return f"DeltaGCounter(value={self.value()}, counts={self.counts})"


def demo_delta_counter():
    """Demonstrate delta-state synchronization."""
    a = DeltaGCounter("A")
    b = DeltaGCounter("B")

    # A increments several times
    a.increment(3)
    a.increment(2)

    # B increments
    b.increment(7)

    # Sync via deltas (not full state)
    delta_a = a.get_delta()  # {A: 5}
    delta_b = b.get_delta()  # {B: 7}

    print(f"Delta from A: {delta_a}")  # small!
    print(f"Delta from B: {delta_b}")  # small!

    a.merge_delta(delta_b)
    b.merge_delta(delta_a)

    print(f"A: {a.value()}, B: {b.value()}")  # both 12
```

### 6.3 Delta 전파 전략

| 전략 | 설명 | 사용 사례 |
|---|---|---|
| **쌍별(Pairwise)** | 각 복제본 쌍이 delta를 교환 | 소규모 클러스터 |
| **Gossip** | 피어를 무작위로 선택하여 delta 교환 | 대규모 클러스터 |
| **Anti-entropy** | 주기적 전체 상태 동기화를 폴백으로 | 복구, 새 노드 |
| **인과적 delta** | Delta에 인과적 메타데이터 첨부 | 순서가 중요할 때 |

---

## 7. 실용적 응용

### 7.1 협업 편집

**Automerge**와 **Yjs**는 실시간 협업 편집에 CRDT를 사용한다. 텍스트는 각 문자가 고유 ID와 위치를 가진 시퀀스 CRDT로 표현된다.

```
User A types "Hello":  H-e-l-l-o
User B types "Hi" concurrently:  H-i

CRDT merge produces interleaved but consistent result:
  Both replicas see the same merged document
  (exact result depends on CRDT algorithm and IDs)
```

### 7.2 장바구니 (Amazon)

Amazon의 Dynamo 논문은 장바구니를 OR-Set으로 설명했다:
- 항목 추가 = 고유 태그가 있는 `add(item)`
- 항목 제거 = 관찰된 태그를 tombstone하는 `remove(item)`
- 다른 복제본에서의 동시 추가: 두 항목 모두 유지
- "add-wins" 의미론: 최악의 경우, 제거된 항목이 다시 나타남 (고객이 다시 제거)

```python
class ShoppingCart:
    """Shopping cart built on OR-Set CRDT."""

    def __init__(self, user_id: str, replica_id: str):
        self.user_id = user_id
        self._set = ORSet(replica_id)

    def add_item(self, item: str, quantity: int = 1) -> None:
        """Add an item to the cart."""
        self._set.add((item, quantity))

    def remove_item(self, item: str) -> None:
        """Remove all entries for an item."""
        to_remove = [
            (i, q) for i, q in self._set.value() if i == item
        ]
        for entry in to_remove:
            self._set.remove(entry)

    def contents(self) -> dict[str, int]:
        """Get cart contents as {item: total_quantity}."""
        result: dict[str, int] = {}
        for item, qty in self._set.value():
            result[item] = result.get(item, 0) + qty
        return result

    def merge(self, other: ShoppingCart) -> ShoppingCart:
        """Merge with another replica of the same cart."""
        result = ShoppingCart(self.user_id, self._set.node_id)
        result._set = self._set.merge(other._set)
        return result

    def __repr__(self) -> str:
        return f"Cart({self.contents()})"
```

### 7.3 분산 카운터 (Redis CRDT)

Redis Enterprise는 active-active 지리적 복제에 CRDT를 사용한다. 카운터는 PN-Counter로, 집합은 OR-Set으로, 문자열은 LWW-Register로 구현된다.

```
Redis CRDT Type Mapping:
  Redis String   →  LWW-Register (or op-based with timestamp)
  Redis Counter  →  PN-Counter (per-replica increments)
  Redis Set      →  OR-Set (add-wins semantics)
  Redis Hash     →  Map of LWW-Registers (per-field LWW)
  Redis Sorted Set → OR-Set with LWW scores
```

### 7.4 DNS 헬스 체크 (AWS Route 53)

AWS Route 53는 전 세계에 분산된 여러 헬스 체커 노드의 결과를 집계하기 위해 CRDT를 사용한다. 각 체커가 독립적으로 UP/DOWN을 보고하며, 집계된 결과가 조정 없이 수렴한다.

---

## 8. CRDT의 한계

CRDT는 강력하지만 보편적 해결책은 아니다. 한계를 이해하는 것이 중요하다.

### 8.1 단조적 성장 (Tombstone 오버헤드)

CRDT는 lattice에서 **단조적** 상태 성장을 요구한다. 제거는 tombstone을 통해 구현되며, (조정 없이는) 절대 가비지 컬렉션되지 않는다.

```
OR-Set after 1 million add/remove cycles:

  Live elements: 100
  Tombstones: 999,900  ← dominates memory!
```

**완화 방법**:
- 주기적 가비지 컬렉션 epoch (조정 필요 — CRDT 속성 약화)
- 인과적 안정성: 모든 복제본이 tombstone을 관찰하면 GC 가능
- 애플리케이션 수준 TTL (예: 7일 이상 된 tombstone 정리)

### 8.2 제한된 표현력

모든 데이터 구조가 CRDT로 표현될 수 있는 것은 아니다. CRDT는 교환 가능하게 만들 수 있는 연산으로 근본적으로 제한된다.

| CRDT 가능 | CRDT로 표현하기 어려움 |
|---|---|
| 카운터 | 제한된 카운터 (예: "최대 100") |
| 집합 | 고유 원소 집합 (예: "정확히 X 중 하나") |
| 레지스터 | 이동 연산 (예: "항목을 집합 A에서 집합 B로 원자적으로 이동") |
| 시퀀스 (텍스트) | 여러 CRDT에 걸친 트랜잭션 |
| 플래그 (활성화/비활성화) | 불변성 보존 연산 |

### 8.3 충돌 의미론이 사용자를 놀라게 할 수 있음

- **LWW가 쓰기를 조용히 버림**: 사용자가 자신의 변경이 폐기되었다는 것을 모를 수 있음
- **Add-wins (OR-Set)**: 동시 추가가 있으면 제거된 항목이 다시 나타날 수 있음
- **PN-Counter가 음수가 될 수 있음**: 증가가 전파되지 않은 상태에서 0 아래로 감소 가능

### 8.4 메타데이터 오버헤드

CRDT는 복제본 수와 연산 수에 따라 성장하는 메타데이터(vector clock, 고유 태그, 노드별 카운터)를 가진다.

```
G-Counter metadata: O(number of replicas)
OR-Set metadata: O(number of add operations ever performed)
MV-Register metadata: O(number of concurrent versions × vector clock size)
```

복제본이 많거나 쓰기 빈도가 높은 시스템에서는 이 오버헤드가 상당할 수 있다.

### 8.5 전체 순서 없음

CRDT는 수렴을 제공하지만 전체 순서를 제공하지 않는다. "카운터가 정확히 100에 도달한 후 멈춤"이 필요하다면, CRDT는 조정 없이 이를 강제할 수 없다.

---

## 9. 종합 CRDT 비교

```python
def crdt_comparison_table():
    """Print a comprehensive comparison of all CRDTs covered."""
    headers = ["CRDT", "Type", "Operations", "Merge", "Limitations"]
    rows = [
        ["G-Counter", "State", "increment", "element-wise max",
         "Cannot decrement"],
        ["PN-Counter", "State", "increment, decrement", "merge P and N G-Counters",
         "Can go negative; no bound"],
        ["G-Set", "State", "add", "set union",
         "Cannot remove elements"],
        ["2P-Set", "State", "add, remove (once)", "merge add and remove sets",
         "Cannot re-add removed elements"],
        ["LWW-Register", "State", "update (with timestamp)", "keep higher timestamp",
         "Silent data loss; clock-dependent"],
        ["MV-Register", "State", "update (with vector clock)", "keep concurrent values",
         "Multiple values need app resolution"],
        ["OR-Set", "State", "add, remove, re-add", "union tags + tombstones",
         "Tombstone overhead"],
        ["Op-Counter", "Op", "increment, decrement", "apply operations",
         "Requires reliable causal broadcast"],
    ]

    # Print formatted table
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)))
```

| CRDT | 유형 | 지원 연산 | 병합 전략 | 주요 한계 |
|---|---|---|---|---|
| G-Counter | 상태 | increment | 원소별 max | 감소 불가 |
| PN-Counter | 상태 | increment, decrement | P와 N을 독립적으로 병합 | 음수 가능 |
| G-Set | 상태 | add | 집합 합집합 | 제거 불가 |
| 2P-Set | 상태 | add, remove (1회) | add + remove 집합 병합 | 제거 후 재추가 불가 |
| LWW-Register | 상태 | update (타임스탬프) | 더 높은 타임스탬프 승리 | 데이터 조용한 손실 |
| MV-Register | 상태 | update (vector clock) | 동시 값 유지 | 앱이 충돌 해결 필요 |
| OR-Set | 상태 | add, remove, re-add | 태그 합집합 + tombstone | Tombstone 증가 |
| Delta-State | 상태 | 기본 CRDT와 동일 | delta 병합 | Delta 추적 필요 |
| Op-Counter | 연산 기반 | increment, decrement | 연산 적용 | 신뢰할 수 있는 전달 필요 |

---

## 10. 전체 데모

```python
def full_crdt_demo():
    """Comprehensive demonstration of all CRDTs with merge verification."""

    print("=" * 60)
    print("G-Counter Demo")
    print("=" * 60)
    demo_g_counter()

    print("\n" + "=" * 60)
    print("PN-Counter Demo")
    print("=" * 60)
    demo_pn_counter()

    print("\n" + "=" * 60)
    print("LWW-Register Demo")
    print("=" * 60)
    demo_lww_register()

    print("\n" + "=" * 60)
    print("MV-Register Demo")
    print("=" * 60)
    demo_mv_register()

    print("\n" + "=" * 60)
    print("OR-Set Demo")
    print("=" * 60)
    demo_or_set()

    print("\n" + "=" * 60)
    print("Op-Based Counter Demo")
    print("=" * 60)
    demo_op_counter()

    print("\n" + "=" * 60)
    print("Delta-State Counter Demo")
    print("=" * 60)
    demo_delta_counter()

    print("\n" + "=" * 60)
    print("CRDT Comparison")
    print("=" * 60)
    crdt_comparison_table()

    # Verify semilattice properties for G-Counter
    print("\n" + "=" * 60)
    print("Semilattice Property Verification (G-Counter)")
    print("=" * 60)

    a = GCounter("A")
    b = GCounter("B")
    c = GCounter("C")
    a.increment(3)
    b.increment(5)
    c.increment(7)

    # Commutativity: a ⊔ b = b ⊔ a
    ab = a.merge(b)
    ba = b.merge(a)
    assert ab.counts == ba.counts, "Commutativity violated!"
    print(f"Commutativity: a⊔b = b⊔a = {ab.value()} ✓")

    # Associativity: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
    ab_c = ab.merge(c)
    bc = b.merge(c)
    a_bc = a.merge(bc)
    assert ab_c.counts == a_bc.counts, "Associativity violated!"
    print(f"Associativity: (a⊔b)⊔c = a⊔(b⊔c) = {ab_c.value()} ✓")

    # Idempotence: a ⊔ a = a
    aa = a.merge(a)
    assert aa.counts == a.counts, "Idempotence violated!"
    print(f"Idempotence: a⊔a = a = {aa.value()} ✓")

    print("\nAll semilattice properties verified!")


if __name__ == "__main__":
    full_crdt_demo()
```

---

## 11. 요약

| 개념 | 핵심 내용 |
|---|---|
| **최종 일관성** | 약한 보장 — 복제본이 시간 제한 없이 "결국" 수렴 |
| **강한 최종 일관성** | CRDT가 동일한 업데이트가 전달되면 수렴을 보장 |
| **Join-semilattice** | 교환 + 결합 + 멱등 병합 = 순서 독립적 수렴 |
| **상태 기반 CRDT** | 전체 상태 전송; lattice join으로 병합; 손실 네트워크에 내성 |
| **연산 기반 CRDT** | 연산 전송; 더 작은 메시지; 신뢰할 수 있는 인과적 전달 필요 |
| **Delta-state CRDT** | 양쪽의 장점: semilattice 병합으로 작은 delta |
| **OR-Set** | 가장 실용적인 집합 CRDT: "add-wins" 의미론의 add, remove, re-add |
| **한계** | Tombstone 오버헤드, 제한된 표현력, 불변성 강제 불가 |

### 설계 지침

1. **필요를 충족하는 가장 단순한 CRDT로 시작** (G-Counter → PN-Counter → OR-Set 순)
2. **충돌 의미론 이해** — LWW는 데이터를 조용히 잃음; 사용자에게 설명하거나 add-wins 선택
3. **Tombstone 가비지 컬렉션 계획** — 없으면 메타데이터가 무한 증가
4. **대역폭 제한 환경에서는 delta-state CRDT 고려**
5. **CRDT는 합의를 보완** — 최종 일관성을 허용할 수 있는 데이터에 CRDT를, 강한 일관성이 필요한 데이터에 Paxos/Raft 사용

---

[다음: 파티셔닝과 샤딩](./11_Partitioning_and_Sharding.md)
