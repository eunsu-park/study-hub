# 레슨 20: 분산 해시 테이블

[개요](./00_Overview.md) | [이전: Raft 구현 Part 2](./19_Raft_Implementation_Part2.md) | [다음: Gossip 프로토콜](./21_Gossip_Protocols.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 가상 노드(virtual node)와 제한된 부하 밸런싱(bounded load balancing)을 사용한 일관된 해싱(consistent hashing) 구현
2. 핑거 테이블(finger table)과 안정화(stabilization)를 사용한 Chord 분산 해시 테이블 구축
3. XOR 기반 거리 라우팅을 사용한 Kademlia DHT 프로토콜 구현
4. 장애 허용(fault tolerance)을 위한 DHT 기반 복제 전략 설계
5. 룩업 지연시간(lookup latency), churn 처리, 부하 균형(load balance) 측면에서 다양한 DHT 설계 간의 트레이드오프 분석

---

## 목차

1. [DHT 소개](#1-dht-소개)
2. [일관된 해싱 기초](#2-일관된-해싱-기초)
3. [가상 노드](#3-가상-노드)
4. [Chord 프로토콜](#4-chord-프로토콜)
5. [Kademlia 프로토콜](#5-kademlia-프로토콜)
6. [DHT 기반 복제](#6-dht-기반-복제)
7. [Churn 처리](#7-churn-처리)
8. [부하 밸런싱](#8-부하-밸런싱)
9. [실제 DHT 시스템](#9-실제-dht-시스템)
10. [요약과 핵심 정리](#10-요약과-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. DHT 소개

### 1.1 DHT란 무엇인가?

분산 해시 테이블(Distributed Hash Table, DHT)은 해시 테이블과 유사한 룩업 서비스를 제공하는 분산 시스템이다. 네트워크의 각 노드는 키 공간(key space)의 일부를 담당하며, 어떤 노드든 주어진 키를 담당하는 노드로 효율적으로 라우팅할 수 있다.

```
전통적 해시 테이블:              분산 해시 테이블:
┌─────────────────────┐         ┌───┐ ┌───┐ ┌───┐ ┌───┐
│ key → bucket → value│         │ N1│ │ N2│ │ N3│ │ N4│
└─────────────────────┘         └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  단일 머신                      │     │     │     │
  O(1) 룩업                     │  ┌──┴─────┴──┐  │
  단일 장애점                    │  │ 키 공간    │  │
                                  │  │ 0 ──── 2^m │  │
                                  │  └────────────┘  │
                                  └──── O(log N) ────┘
```

### 1.2 핵심 속성

| 속성 | 설명 |
|------|------|
| **분산화(Decentralization)** | 중앙 코디네이터 없음; 모든 노드가 동등 |
| **확장성(Scalability)** | O(log N) 라우팅, 노드당 O(log N) 상태 |
| **장애 허용(Fault tolerance)** | 노드가 전역 재구성 없이 참여/이탈 가능 |
| **부하 균형(Load balance)** | 키가 노드 간에 균등하게 분배 |

---

## 2. 일관된 해싱 기초

### 2.1 링(Ring)

일관된 해싱은 키와 노드를 모두 원형 식별자 공간 [0, 2^m)에 매핑한다:

```python
import hashlib
import bisect
import random
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field


class ConsistentHashRing:
    """
    설정 가능한 해시 함수를 가진 일관된 해싱 링.

    키는 키의 해시 위치에서 시계 방향으로 만나는
    첫 번째 노드에 할당된다.
    """

    def __init__(self, hash_bits: int = 160):
        self.hash_bits = hash_bits
        self.ring_size = 2 ** hash_bits
        self.nodes: dict[int, str] = {}  # 위치 → node_id
        self.sorted_positions: list[int] = []
        self.node_positions: dict[str, list[int]] = defaultdict(list)

    def _hash(self, key: str) -> int:
        """키를 링 위의 위치로 해싱한다."""
        h = hashlib.sha1(key.encode()).hexdigest()
        return int(h, 16) % self.ring_size

    def add_node(self, node_id: str) -> int:
        """물리 노드를 링에 추가한다. 위치를 반환한다."""
        pos = self._hash(node_id)
        self.nodes[pos] = node_id
        bisect.insort(self.sorted_positions, pos)
        self.node_positions[node_id].append(pos)
        return pos

    def remove_node(self, node_id: str):
        """링에서 노드를 제거한다."""
        for pos in self.node_positions.get(node_id, []):
            if pos in self.nodes:
                del self.nodes[pos]
                self.sorted_positions.remove(pos)
        del self.node_positions[node_id]

    def get_node(self, key: str) -> Optional[str]:
        """키를 담당하는 노드를 찾는다."""
        if not self.sorted_positions:
            return None

        pos = self._hash(key)
        # pos에서 시계 방향으로 첫 번째 노드 찾기
        idx = bisect.bisect_right(self.sorted_positions, pos)
        if idx >= len(self.sorted_positions):
            idx = 0  # 순환
        return self.nodes[self.sorted_positions[idx]]

    def get_node_and_replicas(self, key: str, num_replicas: int = 3) -> list[str]:
        """키의 주 노드와 복제 노드를 찾는다."""
        if not self.sorted_positions:
            return []

        pos = self._hash(key)
        idx = bisect.bisect_right(self.sorted_positions, pos)
        result = []
        seen = set()

        for i in range(len(self.sorted_positions)):
            actual_idx = (idx + i) % len(self.sorted_positions)
            node_id = self.nodes[self.sorted_positions[actual_idx]]
            if node_id not in seen:
                result.append(node_id)
                seen.add(node_id)
                if len(result) >= num_replicas:
                    break

        return result

    def key_distribution(self, keys: list[str]) -> dict[str, int]:
        """키가 노드 간에 어떻게 분배되는지 분석한다."""
        distribution: dict[str, int] = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            if node:
                distribution[node] += 1
        return dict(distribution)


def demonstrate_consistent_hashing():
    """일관된 해싱 기초를 시연한다."""
    print("=== 일관된 해싱 ===\n")

    ring = ConsistentHashRing(hash_bits=16)  # 시연을 위해 작은 링

    # 노드 추가
    nodes = ["server-A", "server-B", "server-C"]
    for node in nodes:
        pos = ring.add_node(node)
        print(f"  {node}을 위치 {pos}에 추가")

    # 키 분배
    keys = [f"key-{i}" for i in range(1000)]
    dist = ring.key_distribution(keys)
    print(f"\n키 분배 ({len(keys)}개 키, {len(nodes)}개 노드):")
    for node, count in sorted(dist.items()):
        pct = count / len(keys) * 100
        bar = "█" * int(pct / 2)
        print(f"  {node}: {count:4d} ({pct:5.1f}%) {bar}")

    # 노드 추가/제거의 영향
    print(f"\nserver-D 추가 중...")
    ring.add_node("server-D")
    new_dist = ring.key_distribution(keys)

    moved = 0
    for key in keys:
        old_node = None
        for node, count in dist.items():
            pass  # 단순화 — 실제로는 키별로 추적
        new_node = ring.get_node(key)
        # 분배를 비교하여 이동 수 계산
    print(f"새 분배:")
    for node, count in sorted(new_dist.items()):
        pct = count / len(keys) * 100
        print(f"  {node}: {count:4d} ({pct:5.1f}%)")


demonstrate_consistent_hashing()
```

### 2.2 불균형 문제

N개의 물리 노드만으로는 키 분배가 매우 치우칠 수 있다. 3개 노드에서 이상적인 분배는 33.3%이지만, 실제로는 해시 충돌로 인해 10%에서 60%까지 범위가 될 수 있다.

---

## 3. 가상 노드

### 3.1 해결책: 노드당 다중 토큰

각 물리 노드를 링의 여러 위치(가상 노드)에 매핑한다:

```python
class VirtualNodeRing:
    """
    향상된 균형을 위한 가상 노드를 가진 일관된 해싱.

    각 물리 노드를 링의 `vnodes_per_node`개 위치에 매핑한다.
    이것은 부하 분배를 극적으로 개선한다.
    """

    def __init__(self, vnodes_per_node: int = 150, hash_bits: int = 160):
        self.vnodes_per_node = vnodes_per_node
        self.hash_bits = hash_bits
        self.ring_size = 2 ** hash_bits
        self.ring: dict[int, str] = {}  # 위치 → 물리 노드
        self.sorted_positions: list[int] = []
        self.physical_nodes: dict[str, list[int]] = defaultdict(list)

    def _hash(self, key: str) -> int:
        h = hashlib.sha1(key.encode()).hexdigest()
        return int(h, 16) % self.ring_size

    def add_node(self, node_id: str, weight: float = 1.0):
        """
        가상 노드와 함께 물리 노드를 추가한다.

        가중치(weight)는 이종 하드웨어를 허용한다: weight=2인
        노드는 두 배의 가상 노드를 얻어 두 배의 부하를 받는다.
        """
        num_vnodes = int(self.vnodes_per_node * weight)
        for i in range(num_vnodes):
            vnode_key = f"{node_id}#vnode{i}"
            pos = self._hash(vnode_key)
            self.ring[pos] = node_id
            bisect.insort(self.sorted_positions, pos)
            self.physical_nodes[node_id].append(pos)

    def remove_node(self, node_id: str) -> int:
        """물리 노드와 모든 가상 노드를 제거한다."""
        positions = self.physical_nodes.pop(node_id, [])
        for pos in positions:
            if pos in self.ring:
                del self.ring[pos]
                self.sorted_positions.remove(pos)
        return len(positions)

    def get_node(self, key: str) -> Optional[str]:
        """키를 담당하는 물리 노드를 찾는다."""
        if not self.sorted_positions:
            return None
        pos = self._hash(key)
        idx = bisect.bisect_right(self.sorted_positions, pos)
        if idx >= len(self.sorted_positions):
            idx = 0
        return self.ring[self.sorted_positions[idx]]

    def get_node_and_replicas(self, key: str, num_replicas: int = 3) -> list[str]:
        """키의 주 노드와 복제 노드를 찾는다."""
        if not self.sorted_positions:
            return []
        pos = self._hash(key)
        idx = bisect.bisect_right(self.sorted_positions, pos)
        result = []
        seen = set()
        for i in range(len(self.sorted_positions)):
            actual_idx = (idx + i) % len(self.sorted_positions)
            node_id = self.ring[self.sorted_positions[actual_idx]]
            if node_id not in seen:
                result.append(node_id)
                seen.add(node_id)
                if len(result) >= num_replicas:
                    break
        return result

    def analyze_balance(self, num_keys: int = 10000) -> dict:
        """물리 노드 간 부하 균형을 분석한다."""
        counts: dict[str, int] = defaultdict(int)
        for i in range(num_keys):
            key = f"test-key-{i}"
            node = self.get_node(key)
            if node:
                counts[node] += 1

        values = list(counts.values())
        if not values:
            return {}

        mean = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)
        std_dev = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

        return {
            "node_counts": dict(counts),
            "mean": mean,
            "max": max_val,
            "min": min_val,
            "std_dev": round(std_dev, 1),
            "imbalance_ratio": round(max_val / max(min_val, 1), 2),
        }


def compare_vnode_counts():
    """다양한 가상 노드 수에 따른 부하 균형을 비교한다."""
    print("=== 가상 노드: 균형 분석 ===\n")

    nodes = [f"node-{i}" for i in range(5)]
    num_keys = 10000

    for vnodes in [1, 10, 50, 150, 500]:
        ring = VirtualNodeRing(vnodes_per_node=vnodes)
        for node in nodes:
            ring.add_node(node)

        stats = ring.analyze_balance(num_keys)
        print(f"물리 노드당 가상 노드: {vnodes}")
        print(f"  이상적: 노드당 {num_keys // len(nodes)} 키")
        print(f"  실제: min={stats['min']}, max={stats['max']}, "
              f"std_dev={stats['std_dev']}")
        print(f"  불균형 비율: {stats['imbalance_ratio']}배")
        print()


compare_vnode_counts()
```

### 3.2 가중 가상 노드(Weighted Virtual Nodes)

노드 용량이 다른 이종 클러스터를 위한 것이다:

```python
def demonstrate_weighted_vnodes():
    """이종 하드웨어를 위한 가중 가상 노드를 시연한다."""
    print("=== 가중 가상 노드 ===\n")

    ring = VirtualNodeRing(vnodes_per_node=100)

    # 다른 하드웨어에 다른 가중치
    ring.add_node("small-1", weight=1.0)   # 100 가상 노드
    ring.add_node("small-2", weight=1.0)   # 100 가상 노드
    ring.add_node("large-1", weight=3.0)   # 300 가상 노드
    ring.add_node("large-2", weight=2.0)   # 200 가상 노드

    stats = ring.analyze_balance(10000)

    print("노드 용량과 실제 부하:")
    for node, count in sorted(stats["node_counts"].items()):
        weight = {"small-1": 1, "small-2": 1, "large-1": 3, "large-2": 2}[node]
        expected_pct = weight / 7 * 100  # 총 가중치 = 7
        actual_pct = count / 10000 * 100
        print(f"  {node} (weight={weight}): "
              f"예상={expected_pct:.1f}%, 실제={actual_pct:.1f}%, "
              f"키={count}")


demonstrate_weighted_vnodes()
```

---

## 4. Chord 프로토콜

### 4.1 Chord 개요

Chord는 핑거 테이블을 사용하여 노드당 O(log N) 상태로 O(log N) 룩업을 제공한다:

```python
class ChordNode:
    """
    Chord 분산 해시 테이블 프로토콜 구현.

    각 노드는 O(log N) 라우팅을 위한 핑거 테이블,
    장애 허용을 위한 후속자 리스트, 링 유지를 위한
    선행자 포인터를 유지한다.
    """

    M = 8  # 키 공간: 2^M = 256 식별자 (시연을 위해 작음)

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.finger_table: list[Optional[int]] = [None] * self.M
        self.predecessor: Optional[int] = None
        self.successor_list: list[int] = []  # 장애 허용을 위해
        self.data: dict[int, str] = {}  # key → value 저장
        self.lookup_hops: int = 0

    @staticmethod
    def in_range(x: int, start: int, end: int, inclusive_end: bool = False) -> bool:
        """원형 링에서 x가 범위 (start, end]에 있는지 확인한다."""
        ring_size = 2 ** ChordNode.M
        x = x % ring_size
        start = start % ring_size
        end = end % ring_size

        if start < end:
            if inclusive_end:
                return start < x <= end
            else:
                return start < x < end
        else:  # 순환
            if inclusive_end:
                return x > start or x <= end
            else:
                return x > start or x < end

    def successor(self) -> Optional[int]:
        """직접 후속자를 반환한다."""
        return self.finger_table[0]

    def closest_preceding_finger(self, key_id: int) -> int:
        """
        키에 가장 가까운 선행 핑거를 찾는다.

        이것이 Chord의 O(log N) 룩업의 핵심이다: 각 홉은
        대상까지 남은 거리의 최소 절반을 커버한다.
        """
        for i in range(self.M - 1, -1, -1):
            finger = self.finger_table[i]
            if finger is not None and self.in_range(finger, self.node_id, key_id):
                return finger
        return self.node_id

    def find_successor(self, key_id: int, network: dict) -> Tuple[int, int]:
        """
        키를 담당하는 노드를 찾는다.

        (담당_노드, 홉_수)를 반환한다.
        """
        hops = 0
        current = self.node_id

        for _ in range(self.M + 5):  # 안전을 위한 최대 반복
            node = network.get(current)
            if node is None:
                return current, hops

            successor = node.successor()
            if successor is None:
                return current, hops

            if self.in_range(key_id, current, successor, inclusive_end=True):
                return successor, hops

            next_node = node.closest_preceding_finger(key_id)
            if next_node == current:
                return successor, hops
            current = next_node
            hops += 1

        return current, hops


class ChordNetwork:
    """
    테스트를 위한 시뮬레이션된 Chord 네트워크.

    노드 생성, 핑거 테이블 초기화, 키 룩업을 처리한다.
    """

    def __init__(self, m: int = 8):
        ChordNode.M = m
        self.ring_size = 2 ** m
        self.nodes: dict[int, ChordNode] = {}

    def add_node(self, node_id: int) -> ChordNode:
        """Chord 네트워크에 노드를 추가한다."""
        node = ChordNode(node_id)
        self.nodes[node_id] = node
        return node

    def build_finger_tables(self):
        """
        모든 노드의 핑거 테이블을 구축한다.

        finger[i] = (node_id + 2^i) mod 2^M의 후속자

        실제 Chord 네트워크에서는 stabilize() 프로토콜을 통해
        점진적으로 수행된다. 여기서는 간단함을 위해 한 번에
        모두 구축한다.
        """
        sorted_ids = sorted(self.nodes.keys())
        if not sorted_ids:
            return

        for node_id, node in self.nodes.items():
            # 선행자 설정
            idx = sorted_ids.index(node_id)
            node.predecessor = sorted_ids[(idx - 1) % len(sorted_ids)]

            # 핑거 테이블 구축
            for i in range(ChordNode.M):
                target = (node_id + 2 ** i) % self.ring_size

                # target의 후속자 찾기
                found = False
                for sid in sorted_ids:
                    if sid >= target:
                        node.finger_table[i] = sid
                        found = True
                        break
                if not found:
                    node.finger_table[i] = sorted_ids[0]  # 순환

            # 후속자 리스트 구축 (다음 3개 후속자)
            node.successor_list = []
            for j in range(1, min(4, len(sorted_ids))):
                succ = sorted_ids[(idx + j) % len(sorted_ids)]
                node.successor_list.append(succ)

    def lookup(self, origin: int, key: int) -> Tuple[int, int]:
        """원점 노드에서 시작하여 키 룩업을 수행한다."""
        if origin not in self.nodes:
            raise ValueError(f"노드 {origin}이 네트워크에 없음")
        return self.nodes[origin].find_successor(key, self.nodes)

    def store(self, key: int, value: str, origin: int):
        """DHT 라우팅을 통해 키-값 쌍을 저장한다."""
        responsible, hops = self.lookup(origin, key)
        if responsible in self.nodes:
            self.nodes[responsible].data[key] = value

    def analyze_lookups(self, num_lookups: int = 1000) -> dict:
        """룩업 홉 수를 분석한다."""
        if not self.nodes:
            return {}

        node_ids = list(self.nodes.keys())
        hop_counts = []

        for _ in range(num_lookups):
            origin = random.choice(node_ids)
            key = random.randint(0, self.ring_size - 1)
            _, hops = self.lookup(origin, key)
            hop_counts.append(hops)

        return {
            "num_lookups": num_lookups,
            "num_nodes": len(self.nodes),
            "avg_hops": round(sum(hop_counts) / len(hop_counts), 2),
            "max_hops": max(hop_counts),
            "min_hops": min(hop_counts),
            "theoretical_max": ChordNode.M,  # O(log N)
        }


def demonstrate_chord():
    """Chord DHT 프로토콜을 시연한다."""
    print("=== Chord 프로토콜 ===\n")

    network = ChordNetwork(m=8)

    # 다양한 위치에 노드 추가
    node_ids = sorted(random.sample(range(256), 16))
    for nid in node_ids:
        network.add_node(nid)

    network.build_finger_tables()

    # 노드의 핑거 테이블 표시
    sample_node = network.nodes[node_ids[0]]
    print(f"노드 {sample_node.node_id} 핑거 테이블:")
    for i, finger in enumerate(sample_node.finger_table):
        target = (sample_node.node_id + 2 ** i) % 256
        print(f"  finger[{i}]: start={target:3d}, successor={finger}")

    # 룩업 분석
    stats = network.analyze_lookups(1000)
    print(f"\n룩업 분석 ({stats['num_nodes']}개 노드, "
          f"{stats['num_lookups']}번 룩업):")
    print(f"  평균 홉: {stats['avg_hops']}")
    print(f"  최대 홉: {stats['max_hops']}")
    print(f"  이론적 O(log N) = {stats['theoretical_max']}")
    print(f"  log2({stats['num_nodes']}) = {stats['num_nodes']:.0f} → "
          f"{len(bin(stats['num_nodes'])) - 2} 비트")


demonstrate_chord()
```

### 4.2 Chord 안정화(Stabilization)

```python
def demonstrate_chord_stabilization():
    """Chord가 안정화를 통해 노드 참여를 처리하는 방법을 시연한다."""
    print("=== Chord 안정화 ===\n")

    network = ChordNetwork(m=8)

    # 초기 네트워크: 4개 노드
    initial = [0, 64, 128, 192]
    for nid in initial:
        network.add_node(nid)
    network.build_finger_tables()

    stats_before = network.analyze_lookups(500)
    print(f"참여 전: {len(initial)}개 노드, 평균 홉={stats_before['avg_hops']}")

    # 새 노드 참여
    new_nodes = [32, 96, 160, 224]
    for nid in new_nodes:
        network.add_node(nid)
    network.build_finger_tables()

    stats_after = network.analyze_lookups(500)
    print(f"참여 후: {len(initial) + len(new_nodes)}개 노드, "
          f"평균 홉={stats_after['avg_hops']}")

    # 노드 이탈
    network.nodes.pop(64)
    network.build_finger_tables()

    stats_depart = network.analyze_lookups(500)
    print(f"이탈 후: {len(network.nodes)}개 노드, "
          f"평균 홉={stats_depart['avg_hops']}")


demonstrate_chord_stabilization()
```

---

## 5. Kademlia 프로토콜

### 5.1 XOR 거리 메트릭

Kademlia의 핵심 혁신은 XOR을 거리 메트릭으로 사용하는 것이다. XOR은 유효한 메트릭(대칭적이고 삼각 부등식을 만족)이며 효율적인 라우팅을 가능하게 한다:

```python
class KademliaNode:
    """
    Kademlia DHT 프로토콜 구현.

    핵심 특징:
    - XOR 기반 거리 메트릭
    - 라우팅을 위한 k-버킷 (거리 범위당 k개 연락처)
    - 병렬 반복 룩업
    - 지연된 라우팅 테이블 갱신
    """

    K = 20   # 복제 매개변수 (버킷 크기)
    ALPHA = 3  # 병렬성 매개변수
    B = 160    # 키 공간 비트

    def __init__(self, node_id: int):
        self.node_id = node_id
        # k-버킷: 비트당 하나 (bucket[i]는 거리 2^i ~ 2^(i+1)의 노드를 보유)
        self.buckets: list[list[int]] = [[] for _ in range(self.B)]
        self.data: dict[int, str] = {}
        self.lookup_messages: int = 0

    @staticmethod
    def distance(a: int, b: int) -> int:
        """두 노드 ID 사이의 XOR 거리."""
        return a ^ b

    @staticmethod
    def bucket_index(distance: int) -> int:
        """거리가 어떤 k-버킷에 해당하는지 결정한다."""
        if distance == 0:
            return 0
        return distance.bit_length() - 1

    def update_routing_table(self, other_id: int):
        """
        새로 발견된 노드로 라우팅 테이블을 업데이트한다.

        적절한 k-버킷이 가득 차지 않으면 노드를 추가한다.
        가득 차면 가장 오래 본 노드가 아직 살아있는지 확인하고,
        그렇지 않으면 교체한다.
        """
        if other_id == self.node_id:
            return

        dist = self.distance(self.node_id, other_id)
        bucket_idx = self.bucket_index(dist)

        if bucket_idx >= len(self.buckets):
            return

        bucket = self.buckets[bucket_idx]

        if other_id in bucket:
            # 끝으로 이동 (가장 최근에 본 것)
            bucket.remove(other_id)
            bucket.append(other_id)
        elif len(bucket) < self.K:
            bucket.append(other_id)
        # 그렇지 않으면: 버킷이 가득 참, 가장 오래 본 것에 ping

    def find_closest(self, target_id: int, count: int = None) -> list[int]:
        """
        라우팅 테이블에서 대상에 가장 가까운 노드를 찾는다.
        """
        if count is None:
            count = self.K

        all_nodes = []
        for bucket in self.buckets:
            all_nodes.extend(bucket)

        all_nodes.sort(key=lambda n: self.distance(n, target_id))
        return all_nodes[:count]


class KademliaNetwork:
    """시뮬레이션된 Kademlia 네트워크."""

    def __init__(self, key_bits: int = 16):
        KademliaNode.B = key_bits
        self.key_bits = key_bits
        self.key_space = 2 ** key_bits
        self.nodes: dict[int, KademliaNode] = {}

    def add_node(self, node_id: int) -> KademliaNode:
        """노드를 추가하고 라우팅 테이블을 업데이트한다."""
        node = KademliaNode(node_id)
        self.nodes[node_id] = node

        # 부트스트랩: 라우팅 테이블을 양방향으로 업데이트
        for existing_id, existing_node in self.nodes.items():
            if existing_id != node_id:
                node.update_routing_table(existing_id)
                existing_node.update_routing_table(node_id)

        return node

    def iterative_find_node(self, origin: int, target: int) -> Tuple[int, int]:
        """
        반복적 FIND_NODE 룩업을 수행한다.

        (가장_가까운_노드, 전송_메시지_수)를 반환한다.
        """
        if origin not in self.nodes:
            return origin, 0

        messages = 0
        queried: set[int] = set()
        closest = self.nodes[origin].find_closest(target, KademliaNode.K)

        for _ in range(20):  # 최대 반복
            # 대상에 가장 가까운 미질의 ALPHA개 노드 선택
            to_query = [
                n for n in closest if n not in queried
            ][:KademliaNode.ALPHA]

            if not to_query:
                break

            new_contacts = []
            for node_id in to_query:
                queried.add(node_id)
                messages += 1

                if node_id in self.nodes:
                    found = self.nodes[node_id].find_closest(target, KademliaNode.K)
                    new_contacts.extend(found)

            # 병합하고 K개의 가장 가까운 것 유지
            all_contacts = list(set(closest + new_contacts))
            all_contacts.sort(key=lambda n: KademliaNode.distance(n, target))
            new_closest = all_contacts[:KademliaNode.K]

            if new_closest == closest:
                break  # 개선 없음
            closest = new_closest

        result = closest[0] if closest else origin
        return result, messages

    def analyze_lookups(self, num_lookups: int = 500) -> dict:
        """Kademlia 룩업 성능을 분석한다."""
        node_ids = list(self.nodes.keys())
        if not node_ids:
            return {}

        hop_counts = []
        message_counts = []

        for _ in range(num_lookups):
            origin = random.choice(node_ids)
            target = random.randint(0, self.key_space - 1)
            _, messages = self.iterative_find_node(origin, target)
            message_counts.append(messages)

        return {
            "num_lookups": num_lookups,
            "num_nodes": len(self.nodes),
            "avg_messages": round(sum(message_counts) / len(message_counts), 2),
            "max_messages": max(message_counts),
            "theoretical": f"O(log({len(self.nodes)}))",
        }


def demonstrate_kademlia():
    """Kademlia DHT 프로토콜을 시연한다."""
    print("=== Kademlia 프로토콜 ===\n")

    network = KademliaNetwork(key_bits=16)

    # 노드 추가
    num_nodes = 100
    for _ in range(num_nodes):
        node_id = random.randint(0, 2 ** 16 - 1)
        while node_id in network.nodes:
            node_id = random.randint(0, 2 ** 16 - 1)
        network.add_node(node_id)

    # XOR 거리 속성 표시
    ids = list(network.nodes.keys())[:3]
    print("XOR 거리 속성:")
    a, b, c = ids[0], ids[1], ids[2]
    print(f"  d({a}, {b}) = {KademliaNode.distance(a, b)}")
    print(f"  d({b}, {a}) = {KademliaNode.distance(b, a)} (대칭)")
    print(f"  d({a}, {a}) = {KademliaNode.distance(a, a)} (항등원)")
    d_ab = KademliaNode.distance(a, b)
    d_bc = KademliaNode.distance(b, c)
    d_ac = KademliaNode.distance(a, c)
    print(f"  d(a,b) + d(b,c) = {d_ab + d_bc} >= d(a,c) = {d_ac} "
          f"(삼각부등식: {'✓' if d_ab + d_bc >= d_ac else '✗'})")

    # 룩업 성능
    stats = network.analyze_lookups(500)
    print(f"\n룩업 성능 ({stats['num_nodes']}개 노드):")
    print(f"  룩업당 평균 메시지: {stats['avg_messages']}")
    print(f"  최대 메시지: {stats['max_messages']}")
    print(f"  이론적: {stats['theoretical']}")


demonstrate_kademlia()
```

---

## 6. DHT 기반 복제

### 6.1 후속자 기반 복제(Successor-Based Replication)

```python
class ReplicatedDHT:
    """
    장애 허용을 위한 복제를 가진 DHT.

    각 키는 링의 N개 후속자 노드에 저장된다.
    읽기와 쓰기는 정족수(quorum) 프로토콜을 사용한다.
    """

    def __init__(self, replication_factor: int = 3):
        self.N = replication_factor  # 총 복제본
        self.W = 2  # 쓰기 정족수
        self.R = 2  # 읽기 정족수
        self.ring = VirtualNodeRing(vnodes_per_node=50)
        self.node_data: dict[str, dict] = defaultdict(dict)  # 노드 → {key: (value, version)}
        self.version_counter: int = 0

    def put(self, key: str, value: str) -> dict:
        """정족수를 사용한 쓰기."""
        replicas = self.ring.get_node_and_replicas(key, self.N)
        if len(replicas) < self.W:
            return {"ok": False, "error": "복제본이 충분하지 않음"}

        self.version_counter += 1
        version = self.version_counter

        acks = 0
        for node in replicas:
            self.node_data[node][key] = {"value": value, "version": version}
            acks += 1

        return {
            "ok": acks >= self.W,
            "replicas": replicas,
            "acks": acks,
            "version": version,
        }

    def get(self, key: str) -> dict:
        """정족수와 읽기 복구(read-repair)를 사용한 읽기."""
        replicas = self.ring.get_node_and_replicas(key, self.N)

        responses = []
        for node in replicas:
            if key in self.node_data[node]:
                responses.append({
                    "node": node,
                    **self.node_data[node][key],
                })

        if len(responses) < self.R:
            return {"ok": False, "error": "응답이 충분하지 않음"}

        # 가장 높은 버전 반환
        best = max(responses, key=lambda r: r["version"])

        # 읽기 복구: 오래된 복제본 업데이트
        for resp in responses:
            if resp["version"] < best["version"]:
                node = resp["node"]
                self.node_data[node][key] = {
                    "value": best["value"],
                    "version": best["version"],
                }

        return {"ok": True, "value": best["value"], "version": best["version"]}


def demonstrate_replicated_dht():
    """정족수 읽기/쓰기를 가진 복제된 DHT를 시연한다."""
    print("=== 복제된 DHT ===\n")

    dht = ReplicatedDHT(replication_factor=3)

    # 노드 추가
    for i in range(10):
        dht.ring.add_node(f"node-{i}")

    # 쓰기와 읽기
    result = dht.put("user:alice", "{'name': 'Alice', 'age': 30}")
    print(f"PUT user:alice → {result}")

    result = dht.get("user:alice")
    print(f"GET user:alice → {result}")

    # 정족수 분석
    print(f"\n정족수 구성:")
    print(f"  N={dht.N}, W={dht.W}, R={dht.R}")
    print(f"  W + R = {dht.W + dht.R} > N = {dht.N}: "
          f"{'강한 일관성' if dht.W + dht.R > dht.N else '최종 일관성'}")


demonstrate_replicated_dht()
```

---

## 7. Churn 처리

### 7.1 노드 참여/이탈 영향

```python
def analyze_churn_impact():
    """노드 churn이 DHT 성능에 미치는 영향을 분석한다."""
    print("=== Churn 영향 분석 ===\n")

    ring = VirtualNodeRing(vnodes_per_node=100)
    num_keys = 10000

    # 초기 키 할당 저장
    initial_nodes = [f"node-{i}" for i in range(10)]
    for node in initial_nodes:
        ring.add_node(node)

    initial_assignment = {}
    for i in range(num_keys):
        key = f"key-{i}"
        initial_assignment[key] = ring.get_node(key)

    # Churn 시뮬레이션: 2개 노드 제거, 3개 새 노드 추가
    ring.remove_node("node-3")
    ring.remove_node("node-7")
    ring.add_node("node-10")
    ring.add_node("node-11")
    ring.add_node("node-12")

    # 키 이동 수 계산
    moved = 0
    for i in range(num_keys):
        key = f"key-{i}"
        new_node = ring.get_node(key)
        if new_node != initial_assignment[key]:
            moved += 1

    pct_moved = moved / num_keys * 100
    print(f"Churn 이벤트: 2개 노드 제거, 3개 노드 추가")
    print(f"  이동된 키: {moved}/{num_keys} ({pct_moved:.1f}%)")
    print(f"  이상적 (최소 중단): ~{num_keys * 2 / 10:.0f} 키 "
          f"({2 / 10 * 100:.0f}%)")
    print(f"  오버헤드: {pct_moved - (2 / 10 * 100):.1f}% 추가 이동")


analyze_churn_impact()
```

---

## 8. 부하 밸런싱

### 8.1 제한된 부하 일관된 해싱(Bounded Load Consistent Hashing)

Google의 "제한된 부하(bounded load)" 확장은 어떤 노드도 (1 + epsilon) * 평균_부하 이상을 받지 않도록 보장한다:

```python
class BoundedLoadHashRing:
    """
    제한된 부하를 가진 일관된 해싱 (Google, 2017).

    어떤 노드도 평균 부하의 (1 + epsilon)배를 초과하여
    받지 않도록 보장한다. 노드가 과부하되면 키가 링의
    다음 과부하되지 않은 노드에 할당된다.
    """

    def __init__(self, epsilon: float = 0.25, vnodes: int = 100):
        self.epsilon = epsilon
        self.ring = VirtualNodeRing(vnodes_per_node=vnodes)
        self.node_load: dict[str, int] = defaultdict(int)
        self.total_keys: int = 0

    def add_node(self, node_id: str):
        self.ring.add_node(node_id)
        self.node_load[node_id] = 0

    def _max_load(self) -> int:
        """노드당 허용되는 최대 부하를 계산한다."""
        num_nodes = len(self.node_load)
        if num_nodes == 0:
            return 0
        avg_load = max(1, self.total_keys / num_nodes)
        return int(avg_load * (1 + self.epsilon)) + 1

    def assign(self, key: str) -> str:
        """부하 제한을 준수하여 노드에 키를 할당한다."""
        max_load = self._max_load()

        # 해시 위치에서 시계 방향으로 노드 시도
        candidates = self.ring.get_node_and_replicas(key, len(self.node_load))

        for node in candidates:
            if self.node_load[node] < max_load:
                self.node_load[node] += 1
                self.total_keys += 1
                return node

        # 폴백: 모든 노드가 과부하 (올바른 epsilon에서는 발생하지 않아야 함)
        first = candidates[0] if candidates else list(self.node_load.keys())[0]
        self.node_load[first] += 1
        self.total_keys += 1
        return first

    def stats(self) -> dict:
        loads = list(self.node_load.values())
        if not loads:
            return {}
        return {
            "max_load": max(loads),
            "min_load": min(loads),
            "avg_load": sum(loads) / len(loads),
            "max_allowed": self._max_load(),
            "imbalance": max(loads) / max(1, min(loads)),
        }


def demonstrate_bounded_load():
    """표준 vs 제한된 부하 일관된 해싱을 비교한다."""
    print("=== 제한된 부하 일관된 해싱 ===\n")

    # 표준 일관된 해싱
    standard = VirtualNodeRing(vnodes_per_node=100)
    standard_counts: dict[str, int] = defaultdict(int)

    # 제한된 부하
    bounded = BoundedLoadHashRing(epsilon=0.25, vnodes=100)

    nodes = [f"node-{i}" for i in range(8)]
    for node in nodes:
        standard.add_node(node)
        bounded.add_node(node)

    num_keys = 10000
    for i in range(num_keys):
        key = f"key-{i}"
        standard_node = standard.get_node(key)
        standard_counts[standard_node] += 1
        bounded.assign(key)

    # 비교
    std_values = list(standard_counts.values())
    print(f"표준 일관된 해싱:")
    print(f"  최대 부하: {max(std_values)}")
    print(f"  최소 부하: {min(std_values)}")
    print(f"  불균형: {max(std_values)/min(std_values):.2f}배")

    b_stats = bounded.stats()
    print(f"\n제한된 부하 (epsilon={bounded.epsilon}):")
    print(f"  최대 부하: {b_stats['max_load']}")
    print(f"  최소 부하: {b_stats['min_load']}")
    print(f"  최대 허용: {b_stats['max_allowed']}")
    print(f"  불균형: {b_stats['imbalance']:.2f}배")


demonstrate_bounded_load()
```

---

## 9. 실제 DHT 시스템

### 9.1 비교

| 시스템 | 프로토콜 | 거리 | 룩업 | 사용처 |
|--------|----------|------|------|--------|
| **Chord** | 링 + 핑거 테이블 | 시계방향 | O(log N) | 연구 |
| **Kademlia** | k-버킷, XOR | XOR | O(log N) | BitTorrent, Ethereum |
| **Pastry** | 접두사 라우팅 | 공유 접두사 | O(log N) | Microsoft (Halo) |
| **CAN** | d차원 공간 | 데카르트 | O(d·N^(1/d)) | 연구 |
| **Dynamo** | 일관된 해싱 | 링 위치 | O(1)* | Amazon (DynamoDB) |

*Dynamo는 전체 멤버십 지식(모든 노드가 다른 모든 노드를 앎)을 가진 일관된 해싱을 사용하므로 룩업은 O(1) 홉이지만 O(N) 상태가 필요하다.

### 9.2 Amazon Dynamo vs 학술 DHT

```python
def compare_dht_approaches():
    """학술 DHT와 Dynamo 같은 프로덕션 시스템을 비교한다."""
    print("=== 학술 DHT vs 프로덕션 시스템 ===\n")

    comparisons = {
        "멤버십 지식": {
            "Chord/Kademlia": "부분적 (O(log N) 상태)",
            "Dynamo": "전체 (O(N) 상태)",
        },
        "룩업 홉": {
            "Chord/Kademlia": "O(log N) 네트워크 홉",
            "Dynamo": "O(1) — 직접 라우팅",
        },
        "일관성": {
            "Chord/Kademlia": "최종적 (기본)",
            "Dynamo": "조정 가능 (W + R > N이면 강한 일관성)",
        },
        "장애 처리": {
            "Chord/Kademlia": "후속자 리스트, 안정화",
            "Dynamo": "느슨한 정족수, 힌트 핸드오프",
        },
        "규모": {
            "Chord/Kademlia": "수백만 노드 (P2P)",
            "Dynamo": "수백 노드 (데이터센터)",
        },
    }

    for aspect, values in comparisons.items():
        print(f"{aspect}:")
        for system, desc in values.items():
            print(f"  {system:20s}: {desc}")
        print()


compare_dht_approaches()
```

---

## 10. 요약과 핵심 정리

### DHT 설계 공간

> **DHT 설계 차원**
>
> 토폴로지:  링 (Chord) │ 트리 (Kademlia) │ 하이퍼큐브 (CAN)
> 거리:      시계방향    │ XOR             │ 데카르트
> 라우팅:    핑거 테이블 │ k-버킷          │ 이웃 테이블
> 노드당 상태: O(log N)  │ O(log N)        │ O(d)
> 룩업:      O(log N)    │ O(log N)        │ O(d·N^(1/d))
> 복제:      후속자       │ 가장 가까운 노드│ 이웃

### 핵심 원칙

1. **일관된 해싱은 중단을 최소화한다**: 노드가 참여/이탈할 때 K/N 키만 이동한다.
2. **가상 노드가 균형을 개선한다**: 물리 노드당 100개 이상의 가상 노드로 10% 미만의 불균형을 달성한다.
3. **핑거 테이블이 O(log N) 라우팅을 가능하게 한다**: 각 홉이 남은 거리의 절반을 커버한다.
4. **XOR은 우아한 거리 메트릭이다**: 대칭적이고, 삼각 부등식을 만족하며, 효율적인 k-버킷 조직을 가능하게 한다.
5. **프로덕션 시스템은 일반성을 성능과 교환한다**: Dynamo는 데이터센터 내에서 O(1) 룩업을 위해 O(N) 상태를 사용한다.

---

## 11. 연습 문제

### 문제 1: 일관된 해싱 분석

10개 물리 노드와 노드당 200개 가상 노드에서 100,000개 키의 키 분배 표준 편차를 계산하라. 물리 노드당 1개 가상 노드와 어떻게 비교되는가?

### 문제 2: Chord 핑거 테이블

m=6 (64개 위치)이고 노드가 위치 {1, 8, 14, 21, 32, 38, 42, 51}에 있는 Chord 링에서, 노드 14의 완전한 핑거 테이블을 구성하라.

### 문제 3: Kademlia 라우팅

B=8, ALPHA=3, K=4인 Kademlia 네트워크에서:
- 단일 룩업에 필요한 최대 메시지 수는 얼마인가?
- 단일 노드가 가질 수 있는 k-버킷은 몇 개인가?
- 버킷이 가득 차고 새 연락처가 발견되면, 퇴출 전략을 설명하라.

### 문제 4: 구현 과제

다음을 수행하는 `ChordNode.stabilize()` 메서드를 구현하라:
1. 후속자에게 선행자를 질문
2. 선행자가 노드와 후속자 사이에 있으면 새 후속자로 채택
3. 후속자에게 자신의 존재를 알림

### 문제 5: 제한된 부하 분석

epsilon = 0.25와 일관된 해싱으로 어떤 노드도 평균 부하의 1.25배를 초과하여 받지 않음을 증명하라. 부하 제한으로 인해 키가 리다이렉트될 때 룩업 효율에 어떤 일이 발생하는가?

---

## 12. 참고 문헌

1. Stoica, I. et al. (2001). "Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications." *ACM SIGCOMM*.
2. Maymounkov, P. & Mazieres, D. (2002). "Kademlia: A Peer-to-peer Information System Based on the XOR Metric." *IPTPS*.
3. Rowstron, A. & Druschel, P. (2001). "Pastry: Scalable, Decentralized Object Location, and Routing for Large-Scale Peer-to-Peer Systems." *Middleware*.
4. DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-value Store." *SOSP*.
5. Karger, D. et al. (1997). "Consistent Hashing and Random Trees." *STOC*.
6. Mirrokni, V. et al. (2018). "Consistent Hashing with Bounded Loads." arXiv:1608.01350.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 6. O'Reilly Media.

---

[다음: 레슨 21 — Gossip 프로토콜](./21_Gossip_Protocols.md)
