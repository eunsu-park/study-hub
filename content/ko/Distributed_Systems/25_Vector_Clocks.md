# 레슨 25: 벡터 클럭과 인과성 추적 (Vector Clocks and Causality Tracking)

[개요](./00_Overview.md) | [이전: 이벤트 소싱과 CQRS](./24_Event_Sourcing_CQRS.md) | [다음: 분산 테스트](./26_Distributed_Testing.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. Lamport 클럭, 벡터 클럭(vector clock), 버전 벡터(version vector)를 처음부터 구현
2. 벡터 타임스탬프를 사용하여 인과적 순서(causal ordering) 결정 및 동시 이벤트(concurrent event) 감지
3. 벡터 클럭을 사용한 충돌 해결 전략 구축 (LWW, 병합, CRDT)
4. 키-값 스토어(key-value store)에서 정확한 인과성 추적을 위한 dotted version vector 구현
5. 다양한 논리 클럭(logical clock) 메커니즘의 공간-시간 트레이드오프 분석

---

## 목차

1. [분산 시스템에서의 인과성](#1-분산-시스템에서의-인과성)
2. [Lamport 클럭](#2-lamport-클럭)
3. [벡터 클럭](#3-벡터-클럭)
4. [벡터 클럭으로 이벤트 비교](#4-벡터-클럭으로-이벤트-비교)
5. [버전 벡터](#5-버전-벡터)
6. [충돌 감지와 해결](#6-충돌-감지와-해결)
7. [Dotted Version Vector](#7-dotted-version-vector)
8. [하이브리드 논리 클럭](#8-하이브리드-논리-클럭)
9. [실제 응용](#9-실제-응용)
10. [요약 및 핵심 정리](#10-요약-및-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 분산 시스템에서의 인과성

### 1.1 Happens-Before 관계

Lamport의 happens-before 관계(→)는 인과적 순서를 캡처한다:

```
프로세스 P1:  a ─────── b ─────── c
                        \
                         msg
                          \
프로세스 P2:  d ─────── e ─ f ─── g

a → b (같은 프로세스, a가 b 전에 발생)
b → f (b가 메시지를 보내고, f가 수신)
a → f (전이성: a → b → f)
d ∥ a (동시적: 둘 사이에 인과적 경로가 없음)
```

```python
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy


@dataclass
class CausalEvent:
    """인과적 메타데이터가 포함된 이벤트."""
    process_id: str
    event_id: str
    event_type: str  # "local", "send", "receive"
    data: dict = field(default_factory=dict)
    timestamp: dict = field(default_factory=dict)  # 논리적 타임스탬프
```

---

## 2. Lamport 클럭

### 2.1 구현

```python
class LamportClock:
    """
    Lamport 논리 클럭(logical clock) 구현.

    규칙:
    1. 각 로컬 이벤트 전: counter += 1
    2. 전송 전: counter += 1; 카운터를 메시지에 첨부
    3. 수신 시: counter = max(local, received) + 1

    속성: a → b이면 L(a) < L(b)
    한계: L(a) < L(b)이 a → b를 의미하지 않음
    """

    def __init__(self, process_id: str):
        self.process_id = process_id
        self.counter: int = 0
        self.history: list[Tuple[str, int]] = []

    def local_event(self, event_name: str = "") -> int:
        """로컬 이벤트를 기록한다."""
        self.counter += 1
        self.history.append((event_name or f"local_{self.counter}", self.counter))
        return self.counter

    def send(self) -> int:
        """전송할 메시지를 준비한다. 첨부할 타임스탬프를 반환한다."""
        self.counter += 1
        self.history.append(("send", self.counter))
        return self.counter

    def receive(self, received_timestamp: int) -> int:
        """수신된 메시지를 처리한다."""
        self.counter = max(self.counter, received_timestamp) + 1
        self.history.append(("receive", self.counter))
        return self.counter

    def get_time(self) -> int:
        return self.counter


def demonstrate_lamport_clocks():
    """Lamport 클럭과 그 한계를 시연한다."""
    print("=== Lamport Clocks ===\n")

    p1 = LamportClock("P1")
    p2 = LamportClock("P2")
    p3 = LamportClock("P3")

    # P1: 로컬 이벤트, 그 다음 P2로 전송
    p1.local_event("write_x")          # L=1
    ts = p1.send()                      # L=2
    p2.receive(ts)                      # L=3

    # P2: 로컬 이벤트, 그 다음 P3로 전송
    p2.local_event("process_x")        # L=4
    ts = p2.send()                      # L=5
    p3.receive(ts)                      # L=6

    # P3: 로컬 이벤트 (P1의 write_x와 동시적)
    p3_local = p3.local_event("write_y")  # L=7

    # P1: 또 다른 로컬 이벤트 (P3와 동시적)
    p1.local_event("read_x")           # L=3

    print("Event histories:")
    for proc in [p1, p2, p3]:
        print(f"  {proc.process_id}: {proc.history}")

    print(f"\nLimitation demo:")
    print(f"  P1's read_x has L=3, P3's write_y has L=7")
    print(f"  L(read_x) < L(write_y) but they are CONCURRENT!")
    print(f"  Lamport clocks cannot detect concurrency")


demonstrate_lamport_clocks()
```

---

## 3. 벡터 클럭

### 3.1 구현

```python
class VectorClock:
    """
    벡터 클럭(vector clock) 구현.

    각 프로세스가 프로세스당 하나의 카운터 벡터를 유지한다.
    벡터 클럭은 인과적 순서를 결정하고 동시성을 감지할 수 있다.

    규칙:
    1. 각 로컬 이벤트 전: VC[self] += 1
    2. 전송 전: VC[self] += 1; VC를 메시지에 첨부
    3. 수신 시: VC[i] = max(VC[i], received[i]) (모든 i에 대해); VC[self] += 1
    """

    def __init__(self, process_id: str, all_processes: list[str]):
        self.process_id = process_id
        self.clock: Dict[str, int] = {p: 0 for p in all_processes}

    def local_event(self) -> Dict[str, int]:
        """로컬 이벤트를 기록한다."""
        self.clock[self.process_id] += 1
        return self.get_timestamp()

    def send(self) -> Dict[str, int]:
        """메시지를 전송할 준비를 한다. 첨부할 타임스탬프를 반환한다."""
        self.clock[self.process_id] += 1
        return self.get_timestamp()

    def receive(self, received_vc: Dict[str, int]) -> Dict[str, int]:
        """벡터 타임스탬프가 포함된 수신 메시지를 처리한다."""
        for proc_id in self.clock:
            self.clock[proc_id] = max(
                self.clock.get(proc_id, 0),
                received_vc.get(proc_id, 0),
            )
        self.clock[self.process_id] += 1
        return self.get_timestamp()

    def get_timestamp(self) -> Dict[str, int]:
        """현재 벡터 타임스탬프의 복사본을 반환한다."""
        return dict(self.clock)

    @staticmethod
    def happens_before(vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """vc1 → vc2 (vc1이 vc2 전에 발생)인지 확인한다."""
        all_keys = set(vc1.keys()) | set(vc2.keys())
        at_least_one_less = False
        for key in all_keys:
            v1 = vc1.get(key, 0)
            v2 = vc2.get(key, 0)
            if v1 > v2:
                return False
            if v1 < v2:
                at_least_one_less = True
        return at_least_one_less

    @staticmethod
    def concurrent(vc1: Dict[str, int], vc2: Dict[str, int]) -> bool:
        """vc1 ∥ vc2 (동시적 이벤트)인지 확인한다."""
        return (not VectorClock.happens_before(vc1, vc2) and
                not VectorClock.happens_before(vc2, vc1) and
                vc1 != vc2)

    @staticmethod
    def merge(vc1: Dict[str, int], vc2: Dict[str, int]) -> Dict[str, int]:
        """두 벡터 클럭을 병합한다 (컴포넌트별 max)."""
        all_keys = set(vc1.keys()) | set(vc2.keys())
        return {k: max(vc1.get(k, 0), vc2.get(k, 0)) for k in all_keys}

    @staticmethod
    def compare(vc1: Dict[str, int], vc2: Dict[str, int]) -> str:
        """두 벡터 클럭을 비교한다. 반환: 'before', 'after', 'concurrent', 'equal'."""
        if vc1 == vc2:
            return "equal"
        if VectorClock.happens_before(vc1, vc2):
            return "before"
        if VectorClock.happens_before(vc2, vc1):
            return "after"
        return "concurrent"


def demonstrate_vector_clocks():
    """인과성 추적을 위한 벡터 클럭을 시연한다."""
    print("=== Vector Clocks ===\n")

    procs = ["P1", "P2", "P3"]
    vc1 = VectorClock("P1", procs)
    vc2 = VectorClock("P2", procs)
    vc3 = VectorClock("P3", procs)

    # P1이 x를 쓰고, P2에 전송
    ts_a = vc1.local_event()                    # {P1:1, P2:0, P3:0}
    ts_send1 = vc1.send()                        # {P1:2, P2:0, P3:0}
    ts_b = vc2.receive(ts_send1)                  # {P1:2, P2:1, P3:0}

    # P3가 독립적으로 y를 씀 (P1의 이벤트와 동시적)
    ts_c = vc3.local_event()                    # {P1:0, P2:0, P3:1}

    # P2가 P3에 전송
    ts_send2 = vc2.send()                        # {P1:2, P2:2, P3:0}
    ts_d = vc3.receive(ts_send2)                  # {P1:2, P2:2, P3:2}

    print("Events and their vector timestamps:")
    print(f"  a (P1 write x): {ts_a}")
    print(f"  b (P2 receive): {ts_b}")
    print(f"  c (P3 write y): {ts_c}")
    print(f"  d (P3 receive): {ts_d}")

    print(f"\nCausality analysis:")
    pairs = [
        ("a", ts_a, "b", ts_b),
        ("a", ts_a, "c", ts_c),
        ("b", ts_b, "c", ts_c),
        ("c", ts_c, "d", ts_d),
        ("a", ts_a, "d", ts_d),
    ]

    for name1, vc_1, name2, vc_2 in pairs:
        relation = VectorClock.compare(vc_1, vc_2)
        symbol = {"before": "→", "after": "←", "concurrent": "∥", "equal": "="}[relation]
        print(f"  {name1} {symbol} {name2} ({relation})")


demonstrate_vector_clocks()
```

---

## 4. 벡터 클럭으로 이벤트 비교

### 4.1 반순서 시각화 (Partial Order Visualization)

```python
def visualize_partial_order():
    """벡터 클럭이 정의하는 반순서(partial order)를 시각화한다."""
    print("=== Partial Order Visualization ===\n")

    events = {
        "a": {"P1": 1, "P2": 0, "P3": 0},
        "b": {"P1": 2, "P2": 0, "P3": 0},
        "c": {"P1": 0, "P2": 1, "P3": 0},
        "d": {"P1": 2, "P2": 2, "P3": 0},
        "e": {"P1": 0, "P2": 0, "P3": 1},
        "f": {"P1": 2, "P2": 2, "P3": 2},
    }

    print("Events:")
    for name, vc in events.items():
        print(f"  {name}: {vc}")

    print(f"\nPartial order (→ means happens-before, ∥ means concurrent):")
    names = sorted(events.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            relation = VectorClock.compare(events[n1], events[n2])
            if relation != "equal":
                symbol = {"before": "→", "after": "←", "concurrent": "∥"}[relation]
                print(f"  {n1} {symbol} {n2}")

    # Hasse 다이어그램 구축 (직접 인과관계만, 전이성 없음)
    print(f"\nHasse diagram (direct causal links only):")
    for n1 in names:
        for n2 in names:
            if n1 != n2 and VectorClock.happens_before(events[n1], events[n2]):
                # 직접인지 확인 (중간 이벤트 없음)
                is_direct = True
                for n3 in names:
                    if n3 != n1 and n3 != n2:
                        if (VectorClock.happens_before(events[n1], events[n3]) and
                            VectorClock.happens_before(events[n3], events[n2])):
                            is_direct = False
                            break
                if is_direct:
                    print(f"  {n1} → {n2}")


visualize_partial_order()
```

---

## 5. 버전 벡터

### 5.1 복제 데이터를 위한 버전 벡터

```python
class VersionVector:
    """
    복제 시스템에서 데이터 버전을 추적하기 위한 버전 벡터(version vector).

    벡터 클럭(이벤트를 추적)과 달리, 버전 벡터는
    복제본 간 특정 데이터 항목의 버전을 추적한다.
    각 복제본은 쓰기 시 자신의 카운터를 증가시킨다.
    """

    def __init__(self):
        self.vector: Dict[str, int] = {}

    def increment(self, replica_id: str) -> 'VersionVector':
        """복제본의 버전을 증가시킨다 (쓰기 시)."""
        new = VersionVector()
        new.vector = dict(self.vector)
        new.vector[replica_id] = new.vector.get(replica_id, 0) + 1
        return new

    def merge(self, other: 'VersionVector') -> 'VersionVector':
        """두 버전 벡터를 병합한다 (컴포넌트별 max)."""
        new = VersionVector()
        all_keys = set(self.vector.keys()) | set(other.vector.keys())
        new.vector = {
            k: max(self.vector.get(k, 0), other.vector.get(k, 0))
            for k in all_keys
        }
        return new

    def dominates(self, other: 'VersionVector') -> bool:
        """이 버전 벡터가 다른 것을 지배(더 최신)하는지 확인한다."""
        for key in set(self.vector.keys()) | set(other.vector.keys()):
            if self.vector.get(key, 0) < other.vector.get(key, 0):
                return False
        return self.vector != other.vector

    def concurrent_with(self, other: 'VersionVector') -> bool:
        """두 버전 벡터가 동시적(충돌)인지 확인한다."""
        return (not self.dominates(other) and
                not other.dominates(self) and
                self.vector != other.vector)

    def __repr__(self):
        return f"VV({self.vector})"


class ReplicatedKVStore:
    """
    버전 벡터 충돌 감지가 포함된 복제 키-값 스토어.
    """

    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.data: Dict[str, list[Tuple[str, VersionVector]]] = {}

    def put(self, key: str, value: str,
            context: Optional[VersionVector] = None) -> VersionVector:
        """
        선택적 인과적 컨텍스트와 함께 값을 쓴다.

        컨텍스트가 제공되면 클라이언트가 읽은 버전을 나타낸다.
        쓰기는 해당 버전을 대체한다. 컨텍스트가 없으면
        시블링(동시 버전)을 생성할 수 있는 블라인드 쓰기(blind write)이다.
        """
        current_versions = self.data.get(key, [])

        if context is not None:
            # 컨텍스트에 의해 지배되는 버전 제거
            remaining = [
                (v, vv) for v, vv in current_versions
                if not context.dominates(vv) and context.vector != vv.vector
            ]
            new_vv = context.increment(self.replica_id)
        else:
            remaining = list(current_versions)
            # 모든 기존 버전을 병합하고 증가
            merged = VersionVector()
            for _, vv in current_versions:
                merged = merged.merge(vv)
            new_vv = merged.increment(self.replica_id)

        remaining.append((value, new_vv))
        self.data[key] = remaining
        return new_vv

    def get(self, key: str) -> list[Tuple[str, VersionVector]]:
        """키를 읽는다. 여러 동시 버전(시블링)을 반환할 수 있다."""
        return self.data.get(key, [])

    def sync_from(self, other: 'ReplicatedKVStore', key: str):
        """다른 복제본에서 키를 동기화한다."""
        remote_versions = other.get(key)
        local_versions = self.get(key)

        # 병합: 지배되지 않는 모든 버전 유지
        all_versions = local_versions + remote_versions
        merged = []
        for val, vv in all_versions:
            dominated = False
            for other_val, other_vv in all_versions:
                if other_vv.dominates(vv):
                    dominated = True
                    break
            if not dominated:
                # 중복 방지
                if not any(vv.vector == existing_vv.vector for _, existing_vv in merged):
                    merged.append((val, vv))

        self.data[key] = merged


def demonstrate_version_vectors():
    """충돌 감지를 위한 버전 벡터를 시연한다."""
    print("=== Version Vectors ===\n")

    r1 = ReplicatedKVStore("R1")
    r2 = ReplicatedKVStore("R2")

    # R1이 "x" = "alice"를 씀
    vv1 = r1.put("x", "alice")
    print(f"R1 writes x='alice': {vv1}")

    # R1 → R2 동기화
    r2.sync_from(r1, "x")
    print(f"R2 after sync: {r2.get('x')}")

    # R1 업데이트 (컨텍스트 포함)
    vv2 = r1.put("x", "alice_v2", context=vv1)
    print(f"\nR1 writes x='alice_v2': {vv2}")

    # R2가 동시에 업데이트 (이전 컨텍스트로!)
    vv3 = r2.put("x", "bob", context=vv1)
    print(f"R2 writes x='bob': {vv3}")

    # 동기화 — 충돌을 감지해야 함
    r1.sync_from(r2, "x")
    versions = r1.get("x")
    print(f"\nR1 after sync from R2:")
    for val, vv in versions:
        print(f"  value='{val}', version={vv}")

    if len(versions) > 1:
        print(f"\n  CONFLICT DETECTED! {len(versions)} concurrent versions")
        print(f"  Application must resolve: 'alice_v2' vs 'bob'")


demonstrate_version_vectors()
```

---

## 6. 충돌 감지와 해결

### 6.1 해결 전략

```python
class ConflictResolver:
    """
    동시 쓰기에 대한 충돌 해결 전략.
    """

    @staticmethod
    def last_writer_wins(versions: list[Tuple[str, VersionVector, float]]) -> str:
        """
        Last-Writer-Wins (LWW): 벽시계(wall-clock) 타임스탬프로 승자를 선택한다.

        단순하지만 데이터를 잃을 수 있다. Cassandra에서 사용한다.
        """
        return max(versions, key=lambda v: v[2])[0]

    @staticmethod
    def merge_values(versions: list[Tuple[str, VersionVector]]) -> str:
        """
        애플리케이션별 병합. 예: JSON 객체 병합.
        """
        merged = {}
        for val, _ in versions:
            try:
                obj = json.loads(val) if isinstance(val, str) else val
                if isinstance(obj, dict):
                    merged.update(obj)
            except (json.JSONDecodeError, TypeError):
                pass
        return json.dumps(merged) if merged else versions[0][0]

    @staticmethod
    def union_set(versions: list[Tuple[set, VersionVector]]) -> set:
        """
        동시 추가를 위한 집합 합집합.
        OR-Set CRDT에서 사용한다.
        """
        result = set()
        for val, _ in versions:
            result |= val
        return result

    @staticmethod
    def client_resolve(versions: list[Tuple[str, VersionVector]]) -> str:
        """
        수동 해결을 위해 모든 버전을 클라이언트에 반환한다.
        Riak에서 사용한다 (시블링 반환).
        """
        # 실제로는 모든 버전을 클라이언트에 반환
        # 클라이언트가 선택하거나 병합
        return versions  # 클라이언트가 해결해야 함


def demonstrate_conflict_resolution():
    """다양한 충돌 해결 전략을 시연한다."""
    print("=== Conflict Resolution Strategies ===\n")

    import json

    # 시나리오: 두 개의 동시 쓰기
    versions = [
        ('{"name": "Alice", "age": 30}', VersionVector(), time.time()),
        ('{"name": "Alice", "email": "a@b.com"}', VersionVector(), time.time() + 0.001),
    ]

    print("Concurrent versions:")
    for val, _, ts in versions:
        print(f"  {val}")

    # LWW
    winner = ConflictResolver.last_writer_wins(versions)
    print(f"\nLWW winner: {winner}")
    print(f"  Problem: Lost 'age: 30'")

    # 병합
    merge_input = [(v, vv) for v, vv, _ in versions]
    merged = ConflictResolver.merge_values(merge_input)
    print(f"\nMerged: {merged}")
    print(f"  All fields preserved")

    # 태그를 위한 집합 합집합
    set_versions = [
        ({"python", "rust"}, VersionVector()),
        ({"python", "go"}, VersionVector()),
    ]
    union = ConflictResolver.union_set(set_versions)
    print(f"\nSet union: {union}")


demonstrate_conflict_resolution()
```

---

## 7. Dotted Version Vector

### 7.1 일반 버전 벡터가 충분하지 않은 이유

```python
class DottedVersionVector:
    """
    KV 스토어에서 정확한 인과성을 위한 dotted version vector.

    일반 버전 벡터는 다음을 구분할 수 없다:
    - "이 클라이언트가 버전 X를 봤음" (인과적 컨텍스트)
    - "이것이 버전 X" (dot)

    Dotted VV = (dot, version_vector) 여기서:
    - dot = (replica, counter)는 이 특정 쓰기를 식별
    - version_vector = 쓰기의 인과적 컨텍스트
    """

    def __init__(self, dot: Optional[Tuple[str, int]] = None,
                 vv: Optional[Dict[str, int]] = None):
        self.dot = dot  # (replica_id, counter) — 이 이벤트를 식별
        self.vv = vv or {}  # 인과적 컨텍스트

    def __repr__(self):
        return f"DVV(dot={self.dot}, vv={self.vv})"

    def dominates(self, other: 'DottedVersionVector') -> bool:
        """이 DVV가 다른 것을 인과적으로 지배하는지 확인한다."""
        if other.dot:
            replica, counter = other.dot
            if self.vv.get(replica, 0) >= counter:
                return True
        # VV 지배도 확인
        for key, val in other.vv.items():
            if self.vv.get(key, 0) < val:
                return False
        return self.vv != other.vv or (self.dot is not None and other.dot is None)

    def merge(self, other: 'DottedVersionVector') -> 'DottedVersionVector':
        """두 DVV를 병합한다."""
        merged_vv = {}
        all_keys = set(self.vv.keys()) | set(other.vv.keys())
        for key in all_keys:
            merged_vv[key] = max(self.vv.get(key, 0), other.vv.get(key, 0))

        # dot을 VV에 흡수
        if self.dot:
            r, c = self.dot
            merged_vv[r] = max(merged_vv.get(r, 0), c)
        if other.dot:
            r, c = other.dot
            merged_vv[r] = max(merged_vv.get(r, 0), c)

        return DottedVersionVector(dot=None, vv=merged_vv)


def demonstrate_dotted_vv():
    """dotted version vector를 시연한다."""
    print("=== Dotted Version Vectors ===\n")

    # 쓰기 1: 클라이언트가 복제본 R1에 "x=1"을 씀
    dvv1 = DottedVersionVector(dot=("R1", 1), vv={})
    print(f"Write 1 (x=1 at R1): {dvv1}")

    # 쓰기 2: 클라이언트가 버전 1을 읽고 복제본 R2에 "x=2"를 씀
    dvv2 = DottedVersionVector(dot=("R2", 1), vv={"R1": 1})
    print(f"Write 2 (x=2 at R2, after reading v1): {dvv2}")

    # dvv2가 dvv1을 지배하는가? 예, dvv2.vv[R1] >= dvv1.dot[1]이므로
    print(f"\nWrite 2 dominates Write 1: {dvv2.dominates(dvv1)}")

    # 동시 쓰기: 다른 클라이언트가 v1을 읽지 않고 R3에 "x=3"을 씀
    dvv3 = DottedVersionVector(dot=("R3", 1), vv={})
    print(f"Write 3 (x=3 at R3, blind): {dvv3}")
    print(f"Write 2 dominates Write 3: {dvv2.dominates(dvv3)}")
    print(f"Write 3 dominates Write 2: {dvv3.dominates(dvv2)}")
    print(f"Concurrent: both are valid (conflict)")


demonstrate_dotted_vv()
```

---

## 8. 하이브리드 논리 클럭

### 8.1 HLC: 물리적 시간과 논리적 시간의 결합

```python
class HybridLogicalClock:
    """
    하이브리드 논리 클럭(Hybrid Logical Clock, HLC) — Kulkarni et al., 2014.

    물리적 시간과 논리적 카운터를 결합하여 다음을 제공한다:
    - 실시간에 가까운 타임스탬프 (클럭 스큐 내)
    - 인과적 순서 보장 (Lamport 클럭처럼)
    - 제한된 크기 (N에 비례하여 증가하는 벡터 클럭과 달리)

    HLC 타임스탬프 = (physical_time, logical_counter, process_id)
    """

    def __init__(self, process_id: str, max_clock_skew_ms: float = 500):
        self.process_id = process_id
        self.max_skew = max_clock_skew_ms / 1000.0
        self.l: float = 0.0  # 물리적 컴포넌트
        self.c: int = 0       # 논리적 컴포넌트

    def _physical_time(self) -> float:
        return time.time()

    def local_event(self) -> Tuple[float, int]:
        """로컬 또는 전송 이벤트를 기록한다."""
        pt = self._physical_time()
        old_l = self.l

        self.l = max(old_l, pt)
        if self.l == old_l:
            self.c += 1
        else:
            self.c = 0

        return (self.l, self.c)

    def receive(self, remote_l: float, remote_c: int) -> Tuple[float, int]:
        """수신 이벤트를 처리한다."""
        pt = self._physical_time()
        old_l = self.l

        self.l = max(old_l, remote_l, pt)

        if self.l == old_l == remote_l:
            self.c = max(self.c, remote_c) + 1
        elif self.l == old_l:
            self.c += 1
        elif self.l == remote_l:
            self.c = remote_c + 1
        else:
            self.c = 0

        return (self.l, self.c)

    def timestamp(self) -> Tuple[float, int, str]:
        """현재 HLC 타임스탬프를 가져온다."""
        return (self.l, self.c, self.process_id)

    @staticmethod
    def compare(ts1: Tuple[float, int, str],
                ts2: Tuple[float, int, str]) -> int:
        """두 HLC 타임스탬프를 비교한다. -1, 0, 또는 1을 반환한다."""
        if ts1[0] != ts2[0]:
            return -1 if ts1[0] < ts2[0] else 1
        if ts1[1] != ts2[1]:
            return -1 if ts1[1] < ts2[1] else 1
        if ts1[2] != ts2[2]:
            return -1 if ts1[2] < ts2[2] else 1
        return 0


def demonstrate_hlc():
    """하이브리드 논리 클럭을 시연한다."""
    print("=== Hybrid Logical Clocks ===\n")

    hlc1 = HybridLogicalClock("P1")
    hlc2 = HybridLogicalClock("P2")

    # 로컬 이벤트
    ts1 = hlc1.local_event()
    ts2 = hlc1.local_event()  # 같은 물리적 시간 → c 증가
    print(f"P1 event 1: l={ts1[0]:.6f}, c={ts1[1]}")
    print(f"P1 event 2: l={ts2[0]:.6f}, c={ts2[1]}")

    # P2에 전송
    ts_send = hlc1.local_event()
    ts_recv = hlc2.receive(ts_send[0], ts_send[1])
    print(f"\nP1 send:    l={ts_send[0]:.6f}, c={ts_send[1]}")
    print(f"P2 receive: l={ts_recv[0]:.6f}, c={ts_recv[1]}")

    print(f"\nAdvantages over vector clocks:")
    print(f"  - Fixed size (not proportional to number of processes)")
    print(f"  - Close to physical time (useful for TTL, ordering)")
    print(f"  - Still provides causal ordering")
    print(f"\nLimitation:")
    print(f"  - Cannot detect concurrency (total order, like Lamport)")
    print(f"  - Depends on bounded clock skew ({hlc1.max_skew*1000:.0f}ms)")


demonstrate_hlc()
```

---

## 9. 실제 응용

### 9.1 비교

```python
def compare_clock_mechanisms():
    """실제 시스템에서 사용되는 논리 클럭 메커니즘을 비교한다."""
    print("=== Logical Clocks in Real Systems ===\n")

    systems = [
        {"system": "Amazon DynamoDB", "clock": "Vector clocks → LWW",
         "notes": "Originally VV, switched to LWW for simplicity"},
        {"system": "Riak", "clock": "Dotted version vectors",
         "notes": "Accurate sibling detection, client-side resolution"},
        {"system": "CockroachDB", "clock": "Hybrid Logical Clock",
         "notes": "Causal ordering + real-time bounds for serializable isolation"},
        {"system": "Spanner", "clock": "TrueTime (physical)",
         "notes": "GPS + atomic clocks, bounded uncertainty interval"},
        {"system": "Cassandra", "clock": "Last-Writer-Wins (wall clock)",
         "notes": "Simple but can lose writes; depends on NTP"},
        {"system": "Git", "clock": "DAG (not clocks)",
         "notes": "Content-addressable; merge commits resolve concurrent edits"},
    ]

    for s in systems:
        print(f"  {s['system']:20s} | {s['clock']:30s} | {s['notes']}")


compare_clock_mechanisms()
```

---

## 10. 요약 및 핵심 정리

### 클럭 메커니즘 선택 가이드

> **논리 클럭 선택 (CHOOSING A LOGICAL CLOCK)**
>
> 동시성 감지가 필요한가? → 벡터 클럭(Vector Clock) 또는 Dotted VV
> 전체 순서(total order) + 실시간이 필요한가? → 하이브리드 논리 클럭(HLC)
> 단순함 + 전체 순서가 필요한가? → Lamport 클럭
> 정확한 물리적 시간이 필요한가? → TrueTime (하드웨어 필요)
> 데이터 손실을 허용할 수 있는가? → Last-Writer-Wins

### 핵심 원칙

1. **Lamport 클럭은 전체 순서를 제공하지만 동시성을 감지할 수 없다**: L(a) < L(b)이면 a가 b를 유발했을 수도 있고 아닐 수도 있다.
2. **벡터 클럭은 인과성을 정확하게 캡처한다**: a → b iff VC(a) < VC(b); a ∥ b iff 둘 다 지배하지 않는다.
3. **벡터 클럭은 N에 비례하여 증가한다**: 매우 큰 시스템에서는 비실용적이다; 버전 벡터나 HLC가 대안이다.
4. **충돌 감지는 충돌 해결과 별개이다**: 클럭이 충돌을 감지하고; 애플리케이션이 해결 방법을 결정한다.
5. **HLC가 실용적 최적점이다**: 제한된 크기, 실시간에 가까움, 인과적 순서 제공.

---

## 11. 연습 문제

### 문제 1: 벡터 클럭 계산

세 프로세스가 메시지를 교환한다. 모든 이벤트에 대해 벡터 타임스탬프를 계산한다:
- P1: P2에 전송, 로컬, P3에 전송
- P2: P1에서 수신, P3에 전송, 로컬
- P3: 로컬, P1에서 수신, P2에서 수신

### 문제 2: 충돌 감지

키 "x"에 대해 다음 버전 벡터가 주어졌다: VV_A = {R1:3, R2:1}, VV_B = {R1:2, R2:2}. 동시적인가? 그렇다면 장바구니(항목의 집합)에 대한 병합 함수를 설계한다.

### 문제 3: HLC 바운드

HLC 타임스탬프가 항상 실시간의 max_clock_skew 이내임을 증명한다. 노드의 클럭이 max_clock_skew보다 크게 뒤로 점프하면 무슨 일이 일어나는가?

### 문제 4: 구현 도전

다음을 포함하는 복제 키-값 스토어를 구현한다:
- 인과성 추적을 위한 dotted version vector
- 구성 가능한 충돌 해결 (LWW, 클라이언트 해결, CRDT 병합)
- 복제본 동기화를 위한 반엔트로피(anti-entropy) 프로토콜
- GET 시 읽기 복구(read-repair)

### 문제 5: 확장성 분석

벡터 클럭은 프로세스 수가 N일 때 O(N) 크기를 가진다. 영구적으로 제거된 프로세스의 벡터 클럭 항목을 정리하는 방법을 설계한다. 어떤 안전성 보장을 유지할 수 있는가?

---

## 12. 참고 문헌

1. Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." *Communications of the ACM*, 21(7).
2. Fidge, C. (1988). "Timestamps in Message-Passing Systems That Preserve the Partial Ordering." *Australian Computer Science Communications*.
3. Mattern, F. (1989). "Virtual Time and Global States of Distributed Systems." *Parallel and Distributed Algorithms*.
4. Preguica, N. et al. (2012). "Dotted Version Vectors: Logical Clocks for Optimistic Replication." arXiv:1011.5808.
5. Kulkarni, S. et al. (2014). "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases." *OPODIS*.
6. Corbett, J. et al. (2013). "Spanner: Google's Globally-Distributed Database." *ACM TOCS*.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 8. O'Reilly Media.

---

[다음: 레슨 26 — 분산 테스트](./26_Distributed_Testing.md)
