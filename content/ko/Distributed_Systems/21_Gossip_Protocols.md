# 레슨 21: Gossip 프로토콜

[개요](./00_Overview.md) | [이전: 분산 해시 테이블](./20_Distributed_Hash_Tables.md) | [다음: 서비스 디스커버리](./22_Service_Discovery.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 신뢰할 수 있는 정보 전파를 위한 전염병(epidemic/gossip) 프로토콜 구현
2. 의심(suspicion)과 프로토콜 주기 튜닝을 사용한 SWIM 장애 감지 프로토콜 구축
3. push, pull, push-pull gossip 변형 설계 및 수렴(convergence) 속성 분석
4. 적응형 타임아웃 관리를 위한 phi-accrual 장애 감지기(failure detector) 구현
5. 수렴 시간(convergence time), 메시지 오버헤드(message overhead), 거짓 양성률(false positive rate) 측면에서 gossip 프로토콜 성능 분석

---

## 목차

1. [Gossip 프로토콜 소개](#1-gossip-프로토콜-소개)
2. [전염병 전파 모델](#2-전염병-전파-모델)
3. [Push Gossip](#3-push-gossip)
4. [Pull과 Push-Pull Gossip](#4-pull과-push-pull-gossip)
5. [SWIM 장애 감지](#5-swim-장애-감지)
6. [Phi-Accrual 장애 감지기](#6-phi-accrual-장애-감지기)
7. [Gossip 기반 멤버십](#7-gossip-기반-멤버십)
8. [수렴 분석](#8-수렴-분석)
9. [실제 Gossip 시스템](#9-실제-gossip-시스템)
10. [요약과 핵심 정리](#10-요약과-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. Gossip 프로토콜 소개

### 1.1 전염병 비유(Epidemic Metaphor)

Gossip 프로토콜은 질병이 인구를 통해 퍼지는 것과 같은 방식으로 네트워크를 통해 정보를 전파한다. 각 노드는 주기적으로 무작위 피어에 연락하여 정보를 교환한다. 무작위성에도 불구하고, 정보는 O(log N) 라운드 내에 높은 확률로 모든 노드에 도달한다.

```
라운드 0: [I] [ ] [ ] [ ] [ ] [ ] [ ] [ ]     1 감염
라운드 1: [I] [ ] [ ] [I] [ ] [ ] [ ] [ ]     2 감염
라운드 2: [I] [I] [ ] [I] [ ] [I] [ ] [ ]     4 감염
라운드 3: [I] [I] [I] [I] [I] [I] [I] [ ]     7 감염
라운드 4: [I] [I] [I] [I] [I] [I] [I] [I]     8 감염 (전체)
```

### 1.2 왜 Gossip인가?

| 속성 | Gossip | 합의 (Raft/Paxos) |
|------|--------|-------------------|
| 일관성 | 최종적(eventual) | 강한(strong) |
| 확장성 | O(N log N) 메시지 | 결정당 O(N) |
| 장애 허용 | 확률적, 매우 견고 | 과반수 필요 |
| 지연시간 | O(log N) 라운드 | O(1) 라운드 |
| 복잡도 | 단순 | 복잡 |
| 사용 사례 | 멤버십, 메트릭, 설정 | 쓰기, 선출 |

---

## 2. 전염병 전파 모델

### 2.1 세 가지 모델

```python
import random
import time
import math
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class InfectionState(Enum):
    SUSCEPTIBLE = "S"    # 업데이트를 받지 않음
    INFECTED = "I"       # 업데이트를 가지고 있으며 적극적으로 전파 중
    REMOVED = "R"        # 업데이트를 가지고 있지만 전파를 중단함


class EpidemicModel(Enum):
    SI = "SI"       # 취약 → 감염 (전파를 멈추지 않음)
    SIR = "SIR"     # 취약 → 감염 → 제거 (k 라운드 후 중단)
    SIS = "SIS"     # 취약 → 감염 → 취약 (다시 감염 가능)


@dataclass
class GossipNode:
    """Gossip 네트워크의 노드."""
    node_id: str
    state: InfectionState = InfectionState.SUSCEPTIBLE
    data: dict = field(default_factory=dict)
    infection_round: int = -1
    spread_count: int = 0
    max_spreads: int = 3  # SIR 모델: 이만큼의 라운드 후 중단


class EpidemicSimulator:
    """
    전염병 정보 전파를 시뮬레이션한다.

    다양한 전염병 모델(SI, SIR, SIS)에서
    업데이트가 gossip 네트워크를 통해 전파되는 방식을 모델링한다.
    """

    def __init__(self, num_nodes: int, model: EpidemicModel = EpidemicModel.SIR,
                 fanout: int = 2):
        self.num_nodes = num_nodes
        self.model = model
        self.fanout = fanout  # 라운드당 접촉하는 피어 수
        self.nodes: Dict[str, GossipNode] = {}
        self.round_number: int = 0
        self.history: list[dict] = []

        # 노드 생성
        for i in range(num_nodes):
            self.nodes[f"n{i}"] = GossipNode(node_id=f"n{i}")

    def infect(self, node_id: str, data: dict):
        """단일 노드를 감염시켜 전염병을 시작한다."""
        node = self.nodes[node_id]
        node.state = InfectionState.INFECTED
        node.data = data
        node.infection_round = 0

    def round(self):
        """하나의 gossip 라운드를 실행한다."""
        self.round_number += 1
        all_ids = list(self.nodes.keys())

        # 감염된 각 노드가 `fanout`개의 무작위 피어에 접촉
        for node in list(self.nodes.values()):
            if node.state != InfectionState.INFECTED:
                continue

            # 무작위 피어 선택 (자신 제외)
            peers = random.sample(
                [nid for nid in all_ids if nid != node.node_id],
                min(self.fanout, len(all_ids) - 1),
            )

            for peer_id in peers:
                peer = self.nodes[peer_id]
                if peer.state == InfectionState.SUSCEPTIBLE:
                    peer.state = InfectionState.INFECTED
                    peer.data = dict(node.data)
                    peer.infection_round = self.round_number

            node.spread_count += 1

            # SIR: max_spreads 후 제거됨
            if self.model == EpidemicModel.SIR:
                if node.spread_count >= node.max_spreads:
                    node.state = InfectionState.REMOVED

        # 상태 기록
        counts = self._count_states()
        self.history.append(counts)

    def _count_states(self) -> dict:
        """각 상태의 노드 수를 센다."""
        counts = {"S": 0, "I": 0, "R": 0}
        for node in self.nodes.values():
            counts[node.state.value] += 1
        return counts

    def run_until_complete(self, max_rounds: int = 100) -> int:
        """모든 노드가 업데이트를 받을 때까지 실행한다."""
        for r in range(max_rounds):
            self.round()
            counts = self._count_states()
            if counts["S"] == 0:
                return self.round_number
        return max_rounds

    def convergence_report(self) -> dict:
        """수렴 보고서를 생성한다."""
        total_infected = sum(
            1 for n in self.nodes.values()
            if n.state != InfectionState.SUSCEPTIBLE
        )
        return {
            "rounds": self.round_number,
            "infected": total_infected,
            "total": self.num_nodes,
            "coverage": total_infected / self.num_nodes * 100,
            "model": self.model.value,
            "fanout": self.fanout,
        }


def compare_epidemic_models():
    """SI, SIR, SIS 전염병 모델을 비교한다."""
    print("=== 전염병 모델 비교 ===\n")

    num_nodes = 100
    num_trials = 50

    for model in [EpidemicModel.SI, EpidemicModel.SIR]:
        rounds_list = []
        for _ in range(num_trials):
            sim = EpidemicSimulator(num_nodes, model=model, fanout=3)
            sim.infect("n0", {"key": "value"})
            rounds = sim.run_until_complete()
            rounds_list.append(rounds)

        avg_rounds = sum(rounds_list) / len(rounds_list)
        theoretical = math.log(num_nodes) / math.log(3 + 1)  # log_{f+1}(N)
        print(f"{model.value} 모델 (N={num_nodes}, fanout=3):")
        print(f"  수렴까지 평균 라운드: {avg_rounds:.1f}")
        print(f"  이론적 O(log N): ~{theoretical:.1f}")
        print(f"  최소/최대: {min(rounds_list)}/{max(rounds_list)}")
        print()


compare_epidemic_models()
```

---

## 3. Push Gossip

### 3.1 Push 기반 전파

Push gossip에서 감염된 노드는 무작위 피어에 적극적으로 업데이트를 푸시한다:

```python
class PushGossipProtocol:
    """
    상태 전파를 위한 push 기반 gossip 프로토콜.

    각 노드는 로컬 상태(예: 멤버십 목록, 메트릭)를 유지한다.
    주기적으로 각 노드가 무작위 피어를 선택하고 자신의 상태를 전송한다.
    수신자는 받은 상태를 자신의 상태와 병합한다.
    """

    def __init__(self, node_id: str, all_nodes: list[str], fanout: int = 1):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.fanout = fanout
        self.state: Dict[str, dict] = {}  # key → {value, version, timestamp}
        self.messages_sent: int = 0
        self.messages_received: int = 0
        self.merges: int = 0

    def update_local(self, key: str, value: any):
        """로컬 상태 항목을 업데이트한다."""
        current = self.state.get(key, {})
        version = current.get("version", 0) + 1
        self.state[key] = {
            "value": value,
            "version": version,
            "timestamp": time.time(),
            "origin": self.node_id,
        }

    def prepare_push(self) -> list[dict]:
        """
        무작위 피어에 대한 push 메시지를 준비한다.

        전송할 메시지 목록을 반환한다.
        """
        peers = [n for n in self.all_nodes if n != self.node_id]
        targets = random.sample(peers, min(self.fanout, len(peers)))

        messages = []
        for target in targets:
            messages.append({
                "type": "gossip_push",
                "from": self.node_id,
                "to": target,
                "state": dict(self.state),
            })
            self.messages_sent += 1

        return messages

    def receive_push(self, msg: dict):
        """
        Push gossip 메시지를 수신하고 병합한다.

        각 키에 대해 가장 높은 버전의 항목을 유지한다.
        """
        remote_state = msg.get("state", {})
        self.messages_received += 1

        for key, remote_entry in remote_state.items():
            local_entry = self.state.get(key)

            if local_entry is None or remote_entry["version"] > local_entry["version"]:
                self.state[key] = dict(remote_entry)
                self.merges += 1

    def stats(self) -> dict:
        return {
            "node": self.node_id,
            "keys": len(self.state),
            "sent": self.messages_sent,
            "received": self.messages_received,
            "merges": self.merges,
        }


def simulate_push_gossip():
    """상태 전파를 위한 push gossip을 시뮬레이션한다."""
    print("=== Push Gossip 프로토콜 ===\n")

    num_nodes = 20
    node_ids = [f"n{i}" for i in range(num_nodes)]
    nodes = {nid: PushGossipProtocol(nid, node_ids, fanout=2) for nid in node_ids}

    # 노드 0이 업데이트를 가지고 있음
    nodes["n0"].update_local("config.version", "2.0.0")

    # Gossip 라운드 실행
    for round_num in range(15):
        # 각 노드가 push 메시지 준비
        all_messages = []
        for node in nodes.values():
            all_messages.extend(node.prepare_push())

        # 메시지 전달
        for msg in all_messages:
            target = msg["to"]
            if target in nodes:
                nodes[target].receive_push(msg)

        # 업데이트를 가진 노드 수 확인
        informed = sum(
            1 for n in nodes.values()
            if "config.version" in n.state
        )

        if round_num < 5 or informed == num_nodes:
            print(f"  라운드 {round_num + 1}: {informed}/{num_nodes} 노드 통보됨")

        if informed == num_nodes:
            total_msgs = sum(n.messages_sent for n in nodes.values())
            print(f"\n  {round_num + 1} 라운드 만에 수렴")
            print(f"  총 메시지: {total_msgs}")
            print(f"  노드당 메시지: {total_msgs / num_nodes:.1f}")
            break


simulate_push_gossip()
```

---

## 4. Pull과 Push-Pull Gossip

### 4.1 Pull Gossip

Pull gossip은 대부분의 노드가 이미 업데이트를 가지고 있을 때(전파의 "꼬리" 단계) 더 효율적이다:

```python
class PushPullGossipProtocol:
    """
    결합된 push-pull gossip 프로토콜.

    Push 단계: 소수의 노드가 업데이트를 가지고 있을 때의 초기 전파.
    Pull 단계: 대부분의 노드가 업데이트를 가지고 있을 때의 후기 전파.

    Push-pull은 양쪽을 결합한다: 각 교환이 상태 전송과
    요청 모두를 포함하여 어느 한쪽만보다 빠른 수렴을 달성한다.
    """

    def __init__(self, node_id: str, all_nodes: list[str]):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.state: Dict[str, dict] = {}
        self.digest: Dict[str, int] = {}  # key → 본 max_version
        self.messages_sent: int = 0

    def update_local(self, key: str, value: any):
        """로컬 상태를 업데이트한다."""
        version = self.digest.get(key, 0) + 1
        self.state[key] = {"value": value, "version": version}
        self.digest[key] = version

    def prepare_digest(self) -> dict:
        """효율적인 동기화를 위해 상태 버전의 다이제스트를 준비한다."""
        return dict(self.digest)

    def exchange(self, peer_digest: dict, peer_state: dict) -> Tuple[dict, dict]:
        """
        피어와 push-pull 교환을 수행한다.

        1. 다이제스트를 비교하여 차이점 찾기
        2. 피어에 없는 항목 전송 (push)
        3. 우리에게 없는 항목 요청 (pull 응답)

        (피어에_전송할_항목, 우리가_필요한_항목)을 반환한다.
        """
        to_send = {}
        to_request = {}

        # 우리가 가지고 있지만 피어에 없는 키 (또는 우리가 더 새로운 버전)
        for key, version in self.digest.items():
            peer_version = peer_digest.get(key, 0)
            if version > peer_version:
                to_send[key] = self.state[key]

        # 피어가 가지고 있지만 우리에게 없는 키 (또는 피어가 더 새로운)
        for key, peer_version in peer_digest.items():
            our_version = self.digest.get(key, 0)
            if peer_version > our_version:
                if key in peer_state:
                    # 피어에서 업데이트 적용
                    self.state[key] = dict(peer_state[key])
                    self.digest[key] = peer_version

        self.messages_sent += 1
        return to_send, to_request

    def apply_updates(self, updates: dict):
        """피어에서 받은 업데이트를 적용한다."""
        for key, entry in updates.items():
            version = entry.get("version", 0)
            if version > self.digest.get(key, 0):
                self.state[key] = dict(entry)
                self.digest[key] = version


def simulate_push_pull():
    """push 전용, pull 전용, push-pull gossip을 비교한다."""
    print("=== Push vs Pull vs Push-Pull ===\n")

    num_nodes = 50
    num_trials = 30
    node_ids = [f"n{i}" for i in range(num_nodes)]

    for mode in ["push", "pull", "push-pull"]:
        rounds_to_converge = []

        for trial in range(num_trials):
            nodes = {
                nid: PushPullGossipProtocol(nid, node_ids)
                for nid in node_ids
            }

            # 노드 0이 업데이트를 가지고 있음
            nodes["n0"].update_local("data", "update_v1")

            for round_num in range(50):
                # 각 노드가 무작위 피어를 선택
                for node in nodes.values():
                    peer_id = random.choice(
                        [n for n in node_ids if n != node.node_id]
                    )
                    peer = nodes[peer_id]

                    if mode == "push":
                        # 피어에 우리 상태 전송
                        peer.apply_updates(node.state)
                    elif mode == "pull":
                        # 피어에서 상태 요청
                        node.apply_updates(peer.state)
                    else:
                        # Push-pull: 양방향 교환
                        to_send, _ = node.exchange(
                            peer.prepare_digest(), peer.state
                        )
                        peer.apply_updates(to_send)

                informed = sum(1 for n in nodes.values() if "data" in n.state)
                if informed == num_nodes:
                    rounds_to_converge.append(round_num + 1)
                    break

        if rounds_to_converge:
            avg = sum(rounds_to_converge) / len(rounds_to_converge)
            print(f"{mode:12s}: 평균={avg:.1f} 라운드, "
                  f"최소={min(rounds_to_converge)}, "
                  f"최대={max(rounds_to_converge)}")
        else:
            print(f"{mode:12s}: 모든 시도에서 수렴하지 않음")


simulate_push_pull()
```

---

## 5. SWIM 장애 감지

### 5.1 SWIM 프로토콜 개요

SWIM(Scalable Weakly-consistent Infection-style process group Membership)은 프로토콜 주기당 멤버당 O(1) 메시지 부하를 달성하는 gossip 기반 장애 감지 프로토콜이다:

```python
class SWIMNodeState(Enum):
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"


@dataclass
class SWIMMember:
    """SWIM 그룹의 멤버."""
    node_id: str
    state: SWIMNodeState = SWIMNodeState.ALIVE
    incarnation: int = 0
    last_updated: float = field(default_factory=time.time)


class SWIMProtocol:
    """
    SWIM 장애 감지 프로토콜 구현.

    프로토콜 주기:
    1. 무작위 멤버 M 선택
    2. M에게 ping 전송
    3. M이 응답하면 → M은 살아있음
    4. M이 타임아웃 내에 응답하지 않으면:
       a. k개의 무작위 멤버 선택
       b. 그들에게 M에게 ping을 요청 (간접 ping)
       c. 간접 ping이 하나라도 성공하면 → M은 살아있음
       d. 모두 실패하면 → M을 suspect로 표시

    의심 메커니즘은 의심 대상에게 죽음 선언 전에
    유예 기간을 제공한다.
    """

    def __init__(self, node_id: str, members: list[str],
                 k_indirect: int = 3, suspect_timeout: float = 5.0):
        self.node_id = node_id
        self.k_indirect = k_indirect
        self.suspect_timeout: float = suspect_timeout
        self.protocol_period: float = 1.0  # 초

        self.members: Dict[str, SWIMMember] = {}
        for mid in members:
            self.members[mid] = SWIMMember(node_id=mid)

        # 통계
        self.pings_sent: int = 0
        self.indirect_pings_sent: int = 0
        self.false_positives: int = 0
        self.true_positives: int = 0
        self.suspects: Dict[str, float] = {}  # node_id → suspect_since

    def protocol_round(self, alive_nodes: set[str]) -> list[dict]:
        """
        하나의 SWIM 프로토콜 라운드를 실행한다.

        Args:
            alive_nodes: 실제로 살아있는 노드 집합 (시뮬레이션을 위한 실측)

        Returns:
            이벤트 목록 (멤버십 변경)
        """
        events = []

        # 프로브할 무작위 멤버 선택
        probe_candidates = [
            mid for mid in self.members
            if mid != self.node_id and self.members[mid].state != SWIMNodeState.DEAD
        ]

        if not probe_candidates:
            return events

        target_id = random.choice(probe_candidates)
        self.pings_sent += 1

        # 직접 ping
        if target_id in alive_nodes:
            # Ping 성공
            if target_id in self.suspects:
                del self.suspects[target_id]
                self.members[target_id].state = SWIMNodeState.ALIVE
                events.append({"type": "alive", "node": target_id})
        else:
            # 직접 ping 실패 — 간접 프로브 시도
            indirect_targets = random.sample(
                [m for m in probe_candidates if m != target_id],
                min(self.k_indirect, len(probe_candidates) - 1),
            )

            indirect_success = False
            for proxy in indirect_targets:
                self.indirect_pings_sent += 1
                if proxy in alive_nodes and target_id in alive_nodes:
                    indirect_success = True
                    break

            if indirect_success:
                if target_id in self.suspects:
                    del self.suspects[target_id]
                    self.members[target_id].state = SWIMNodeState.ALIVE
            else:
                # Suspect로 표시
                if target_id not in self.suspects:
                    self.suspects[target_id] = time.time()
                    self.members[target_id].state = SWIMNodeState.SUSPECT
                    events.append({"type": "suspect", "node": target_id})

        # Suspect 타임아웃 확인
        now = time.time()
        for suspect_id, suspect_since in list(self.suspects.items()):
            if now - suspect_since > self.suspect_timeout:
                self.members[suspect_id].state = SWIMNodeState.DEAD
                del self.suspects[suspect_id]
                events.append({"type": "dead", "node": suspect_id})

                # 정확도 추적
                if suspect_id not in alive_nodes:
                    self.true_positives += 1
                else:
                    self.false_positives += 1

        return events

    def get_alive_members(self) -> list[str]:
        """살아있다고 판단되는 멤버 목록을 반환한다."""
        return [
            mid for mid, member in self.members.items()
            if member.state == SWIMNodeState.ALIVE
        ]

    def accuracy_report(self) -> dict:
        """감지 정확도를 보고한다."""
        return {
            "pings": self.pings_sent,
            "indirect_pings": self.indirect_pings_sent,
            "suspects": len(self.suspects),
            "dead": sum(1 for m in self.members.values() if m.state == SWIMNodeState.DEAD),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
        }


def simulate_swim():
    """SWIM 장애 감지 프로토콜을 시뮬레이션한다."""
    print("=== SWIM 장애 감지 ===\n")

    num_nodes = 20
    node_ids = [f"n{i}" for i in range(num_nodes)]
    swim = SWIMProtocol("n0", node_ids, k_indirect=3, suspect_timeout=0.5)

    # 모든 노드가 살아있는 것으로 시작
    alive = set(node_ids)

    # 50 프로토콜 라운드 시뮬레이션
    for round_num in range(50):
        # 라운드 10에서 노드 5와 12 죽이기
        if round_num == 10:
            alive.discard("n5")
            alive.discard("n12")
            print(f"  라운드 {round_num}: n5와 n12 죽임")

        # 라운드 30에서 n5 부활
        if round_num == 30:
            alive.add("n5")
            print(f"  라운드 {round_num}: n5 부활")

        events = swim.protocol_round(alive)
        for event in events:
            print(f"  라운드 {round_num}: {event['type'].upper()} → {event['node']}")

        time.sleep(0.01)  # 시간 경과 시뮬레이션

    report = swim.accuracy_report()
    print(f"\nSWIM 정확도 보고:")
    for k, v in report.items():
        print(f"  {k}: {v}")


simulate_swim()
```

---

## 6. Phi-Accrual 장애 감지기

### 6.1 적응형 타임아웃(Adaptive Timeout)

고정된 타임아웃 대신, phi-accrual 감지기는 하트비트 도착 시간의 통계적 분포에 기반하여 연속적인 의심 수준(phi)을 출력한다:

```python
class PhiAccrualDetector:
    """
    Phi-accrual 장애 감지기 (Hayashibara et al., 2004).

    이진적인 alive/dead 출력 대신, 이 감지기는 연속적인
    의심 수준 φ (phi)를 출력한다. φ가 높을수록
    노드가 장애 났을 가능성이 높다.

    φ는 과거 도착 시간 분포를 기반으로 하트비트가
    지금까지 도착했을 확률에서 계산된다.

    φ = -log10(P(t_now - t_last > 관측된_간격))
    """

    def __init__(self, threshold: float = 8.0, window_size: int = 100,
                 min_std_dev_ms: float = 500.0):
        self.threshold = threshold  # φ가 이 이상이면 → 의심
        self.window_size = window_size
        self.min_std_dev_ms = min_std_dev_ms

        # 하트비트 도착 간격 (ms)
        self.intervals: list[float] = []
        self.last_heartbeat: Optional[float] = None
        self.heartbeat_count: int = 0

    def heartbeat(self):
        """하트비트 도착을 기록한다."""
        now = time.time() * 1000  # ms

        if self.last_heartbeat is not None:
            interval = now - self.last_heartbeat
            self.intervals.append(interval)
            if len(self.intervals) > self.window_size:
                self.intervals.pop(0)

        self.last_heartbeat = now
        self.heartbeat_count += 1

    def phi(self) -> float:
        """
        현재 phi 값을 계산한다.

        phi = -log10(1 - CDF(t_now - t_last))

        여기서 CDF는 하트비트 간격 분포(정규 분포 가정)의
        누적 분포 함수이다.
        """
        if self.last_heartbeat is None or len(self.intervals) < 2:
            return 0.0

        now = time.time() * 1000
        elapsed = now - self.last_heartbeat

        # 간격의 평균과 표준 편차 계산
        mean = sum(self.intervals) / len(self.intervals)
        variance = sum((x - mean) ** 2 for x in self.intervals) / len(self.intervals)
        std_dev = max(math.sqrt(variance), self.min_std_dev_ms)

        # 정규 분포 근사를 사용하여 P(X > elapsed) 계산
        # P(X > t) ≈ 1 - Φ((t - μ) / σ)
        z = (elapsed - mean) / std_dev

        # 로지스틱 함수를 사용하여 Φ(z) 근사
        cdf = 1.0 / (1.0 + math.exp(-1.7 * z))

        # φ = -log10(1 - CDF) = -log10(P(늦음))
        p_late = 1.0 - cdf
        if p_late <= 0:
            return float('inf')
        if p_late >= 1:
            return 0.0

        return -math.log10(p_late)

    def is_suspected(self) -> bool:
        """노드가 장애 의심되는지 확인한다."""
        return self.phi() >= self.threshold

    def status(self) -> dict:
        """감지기 상태를 가져온다."""
        intervals = self.intervals
        return {
            "phi": round(self.phi(), 2),
            "threshold": self.threshold,
            "suspected": self.is_suspected(),
            "heartbeats": self.heartbeat_count,
            "mean_interval_ms": round(sum(intervals) / len(intervals), 1) if intervals else 0,
            "std_dev_ms": round(
                (sum((x - sum(intervals)/len(intervals))**2
                     for x in intervals) / len(intervals)) ** 0.5, 1
            ) if len(intervals) > 1 else 0,
        }


def demonstrate_phi_detector():
    """Phi-accrual 장애 감지기를 시연한다."""
    print("=== Phi-Accrual 장애 감지기 ===\n")

    detector = PhiAccrualDetector(threshold=8.0)

    # 단계 1: 정규 하트비트 (약 100ms마다, 지터 포함)
    print("단계 1: 정규 하트비트")
    for _ in range(20):
        detector.heartbeat()
        time.sleep(random.uniform(0.08, 0.12))  # 80-120ms

    print(f"  상태: {detector.status()}")

    # 단계 2: 지연된 하트비트 (GC 일시 정지 시뮬레이션)
    print("\n단계 2: 500ms 지연 (GC 일시 정지)")
    time.sleep(0.5)
    phi_during_pause = detector.phi()
    print(f"  지연 중 Phi: {phi_during_pause:.2f}")
    print(f"  의심됨: {detector.is_suspected()}")

    # 하트비트 도착
    detector.heartbeat()
    print(f"  하트비트 후: phi={detector.phi():.2f}, 의심={detector.is_suspected()}")

    # 단계 3: 노드 장애 (더 이상 하트비트 없음)
    print("\n단계 3: 노드 장애 (2초 침묵)")
    time.sleep(0.3)
    for i in range(5):
        time.sleep(0.1)
        print(f"  t+{(i+1)*100 + 300}ms: phi={detector.phi():.2f}, "
              f"의심={detector.is_suspected()}")


demonstrate_phi_detector()
```

---

## 7. Gossip 기반 멤버십

### 7.1 피기백 업데이트가 있는 멤버십(Membership with Piggyback Updates)

SWIM은 추가 메시지를 피하기 위해 ping/ack 메시지에 멤버십 업데이트를 피기백한다:

```python
class GossipMembershipProtocol:
    """
    피기백된 업데이트가 있는 gossip 기반 멤버십 프로토콜.

    멤버십 변경(참여, 이탈, 장애)은 일반 프로토콜 메시지에
    피기백하여 전파된다. 각 업데이트는 순서를 위한
    incarnation 번호를 가진다.
    """

    def __init__(self, node_id: str, seed_nodes: list[str]):
        self.node_id = node_id
        self.members: Dict[str, dict] = {
            node_id: {
                "state": "alive",
                "incarnation": 0,
                "address": f"{node_id}:7946",
            }
        }
        self.update_queue: list[dict] = []  # 피기백할 업데이트
        self.max_piggyback: int = 10  # 메시지당 최대 업데이트
        self.update_retransmit: int = 3  # 각 업데이트를 N번 재전송

    def join(self, seed: str):
        """시드 노드를 통해 클러스터에 참여한다."""
        self.members[seed] = {
            "state": "alive",
            "incarnation": 0,
            "address": f"{seed}:7946",
        }
        self._queue_update({
            "type": "join",
            "node": self.node_id,
            "incarnation": 0,
        })

    def leave(self):
        """클러스터에서 정상적으로 이탈한다."""
        self.members[self.node_id]["state"] = "left"
        self._queue_update({
            "type": "leave",
            "node": self.node_id,
            "incarnation": self.members[self.node_id]["incarnation"],
        })

    def mark_suspect(self, node_id: str):
        """노드를 의심으로 표시한다."""
        if node_id in self.members and self.members[node_id]["state"] == "alive":
            self.members[node_id]["state"] = "suspect"
            self._queue_update({
                "type": "suspect",
                "node": node_id,
                "incarnation": self.members[node_id]["incarnation"],
            })

    def refute_suspect(self):
        """
        incarnation을 증가시켜 자신의 의심을 반박한다.

        이 노드가 자신이 의심받고 있음을 알게 되면,
        incarnation 번호를 증가시키고 alive 메시지를 브로드캐스트한다.
        이것은 suspect 메시지를 대체한다.
        """
        self.members[self.node_id]["incarnation"] += 1
        self.members[self.node_id]["state"] = "alive"
        self._queue_update({
            "type": "alive",
            "node": self.node_id,
            "incarnation": self.members[self.node_id]["incarnation"],
        })

    def receive_update(self, update: dict) -> bool:
        """
        수신된 멤버십 업데이트를 처리한다.

        업데이트가 적용되면(새 정보) True를 반환한다.
        """
        node_id = update["node"]
        update_type = update["type"]
        incarnation = update["incarnation"]

        current = self.members.get(node_id)

        if current is None:
            # 새 멤버
            self.members[node_id] = {
                "state": "alive" if update_type in ("join", "alive") else update_type,
                "incarnation": incarnation,
                "address": f"{node_id}:7946",
            }
            return True

        # Incarnation 기반 순서
        if incarnation < current["incarnation"]:
            return False  # 오래된 업데이트

        if incarnation == current["incarnation"]:
            # 같은 incarnation: alive < suspect < dead
            priority = {"alive": 0, "suspect": 1, "dead": 2, "left": 3}
            if priority.get(update_type, 0) <= priority.get(current["state"], 0):
                return False  # 더 새롭지 않음

        # 업데이트 적용
        self.members[node_id]["state"] = update_type if update_type != "join" else "alive"
        self.members[node_id]["incarnation"] = incarnation
        self._queue_update(update)
        return True

    def _queue_update(self, update: dict):
        """피기백을 위해 업데이트를 큐에 넣는다."""
        self.update_queue.append({
            **update,
            "retransmit_count": self.update_retransmit,
        })

    def get_piggyback_updates(self) -> list[dict]:
        """다음 메시지에 피기백할 업데이트를 가져온다."""
        updates = []
        remaining = []

        for entry in self.update_queue[:self.max_piggyback]:
            updates.append({k: v for k, v in entry.items() if k != "retransmit_count"})
            entry["retransmit_count"] -= 1
            if entry["retransmit_count"] > 0:
                remaining.append(entry)

        self.update_queue = remaining + self.update_queue[self.max_piggyback:]
        return updates

    def get_alive_members(self) -> list[str]:
        """살아있는 멤버를 반환한다."""
        return [
            mid for mid, info in self.members.items()
            if info["state"] == "alive"
        ]


def demonstrate_gossip_membership():
    """Gossip 기반 멤버십 관리를 시연한다."""
    print("=== Gossip 멤버십 프로토콜 ===\n")

    # 5개 노드 클러스터 생성
    nodes = {}
    node_ids = [f"n{i}" for i in range(5)]

    for nid in node_ids:
        nodes[nid] = GossipMembershipProtocol(nid, [])

    # 부트스트랩: 모든 노드가 서로를 앎
    for nid, node in nodes.items():
        for other in node_ids:
            if other != nid:
                node.join(other)

    print("초기 클러스터: ", [n.get_alive_members() for n in nodes.values()][0])

    # 시뮬레이션: n0이 n3를 의심
    nodes["n0"].mark_suspect("n3")
    update = nodes["n0"].get_piggyback_updates()
    print(f"\nn0이 n3를 의심: {update}")

    # 의심을 n1에 gossip
    for u in update:
        nodes["n1"].receive_update(u)
    print(f"n1의 n3 뷰: {nodes['n1'].members.get('n3', {}).get('state')}")

    # n3가 의심을 반박
    nodes["n3"].refute_suspect()
    refute = nodes["n3"].get_piggyback_updates()
    print(f"\nn3가 incarnation 증가로 반박: {refute}")

    # 반박을 gossip
    for u in refute:
        nodes["n0"].receive_update(u)
        nodes["n1"].receive_update(u)

    print(f"반박 후 n0의 n3 뷰: {nodes['n0'].members['n3']['state']}")
    print(f"반박 후 n1의 n3 뷰: {nodes['n1'].members['n3']['state']}")


demonstrate_gossip_membership()
```

---

## 8. 수렴 분석

### 8.1 수학적 분석

```python
def analyze_convergence():
    """Gossip 수렴 속성을 수학적으로 분석한다."""
    print("=== 수렴 분석 ===\n")

    # 이론적: fanout f와 N개 노드에서
    # r 라운드 후, 예상 미통보 노드: N * (1 - 1/N)^(f*r*...)
    # 단순화: O(log N) 라운드에 수렴

    for n in [10, 100, 1000, 10000]:
        for fanout in [1, 2, 3]:
            # 시뮬레이션
            trials = 100
            rounds_list = []
            for _ in range(trials):
                sim = EpidemicSimulator(n, EpidemicModel.SIR, fanout=fanout)
                sim.infect("n0", {"data": True})
                rounds = sim.run_until_complete(max_rounds=50)
                rounds_list.append(rounds)

            avg = sum(rounds_list) / len(rounds_list)
            theoretical = math.ceil(math.log(n) / math.log(fanout + 1))
            print(f"  N={n:>5}, f={fanout}: 평균_라운드={avg:.1f}, "
                  f"이론적≈{theoretical}")

    # 메시지 복잡도
    print(f"\n라운드당 메시지 복잡도:")
    print(f"  Push gossip: N × fanout 메시지")
    print(f"  Pull gossip: N × fanout 메시지")
    print(f"  SWIM: N × 1 직접 + N × k 간접 (최악의 경우)")

    for n in [100, 1000, 10000]:
        for fanout in [2, 3]:
            total_messages = n * fanout * math.ceil(math.log(n) / math.log(fanout + 1))
            print(f"  N={n}, f={fanout}: 수렴까지 ~{total_messages}개 총 메시지")


analyze_convergence()
```

---

## 9. 실제 Gossip 시스템

### 9.1 시스템 비교

```python
def compare_gossip_systems():
    """Gossip 프로토콜을 사용하는 실제 시스템을 비교한다."""
    print("=== 실제 Gossip 시스템 ===\n")

    systems = [
        {
            "name": "HashiCorp Serf/Consul",
            "protocol": "SWIM + Lifeguard 확장",
            "use_case": "서비스 디스커버리, 멤버십",
            "gossip_interval": "200ms",
            "failure_detection": "~2-5초",
        },
        {
            "name": "Apache Cassandra",
            "protocol": "Gossip (push-pull, phi-accrual)",
            "use_case": "멤버십, 스키마 전파",
            "gossip_interval": "1초",
            "failure_detection": "~10초",
        },
        {
            "name": "Amazon S3",
            "protocol": "커스텀 gossip",
            "use_case": "수천 노드 간 멤버십",
            "gossip_interval": "~1초",
            "failure_detection": "수초",
        },
        {
            "name": "Redis Cluster",
            "protocol": "Gossip (커스텀)",
            "use_case": "클러스터 상태, 슬롯 매핑",
            "gossip_interval": "1초",
            "failure_detection": "~15초 (설정 가능)",
        },
        {
            "name": "CockroachDB",
            "protocol": "Raft 위의 Gossip 오버레이",
            "use_case": "노드 생존성, 범위 메타데이터",
            "gossip_interval": "~1초",
            "failure_detection": "~9초",
        },
    ]

    for sys in systems:
        print(f"{sys['name']}:")
        for key in ["protocol", "use_case", "gossip_interval", "failure_detection"]:
            print(f"  {key}: {sys[key]}")
        print()


compare_gossip_systems()
```

---

## 10. 요약과 핵심 정리

### Gossip 프로토콜 설계 공간

> **GOSSIP 설계 차원**
>
> 전파:      Push │ Pull │ Push-Pull
> 모델:      SI   │ SIR  │ SIS
> 감지:      고정 타임아웃 │ Phi-accrual │ SWIM
> 멤버십:    중앙화된 시드 │ Gossip 기반 │ 하이브리드
> 수렴:      O(log N) 라운드 (높은 확률)

### 핵심 원칙

1. **O(log N) 수렴**: Gossip은 지수적으로 퍼진다 — 매 라운드 통보된 노드가 두 배가 된다.
2. **무작위성이 견고성을 제공한다**: 단일 장애점이 없으며; 임의의 노드 장애를 허용한다.
3. **Push-pull이 최적이다**: 빠른 초기 전파(push)와 효율적인 꼬리(pull)를 결합한다.
4. **SWIM은 하트비트보다 확장성이 좋다**: 주기당 멤버당 O(1) 메시지 부하.
5. **Phi-accrual은 네트워크에 적응한다**: 튜닝할 고정 타임아웃이 없으며; 관측된 지연시간에 맞게 조정한다.

---

## 11. 연습 문제

### 문제 1: 수렴 증명

fanout f를 가진 push gossip이 최소 1 - 1/N의 확률로 O(log_f N) 라운드에 수렴함을 증명하라.

### 문제 2: SWIM 분석

프로토콜 주기 1초, k=3인 1000개 노드의 SWIM 클러스터에서:
- 단일 노드 장애를 감지하는 예상 시간은 얼마인가?
- 네트워크 지연시간이 정상의 5배로 급등하면 거짓 양성률은 얼마인가?
- 클러스터 전체에서 초당 얼마나 많은 프로토콜 메시지가 전송되는가?

### 문제 3: Phi 임계값 튜닝

평균=100ms, 표준편차=10ms인 하트비트 간격에서, 120ms, 150ms, 200ms, 500ms 지연에서의 phi 값을 계산하라. 0.1% 미만의 거짓 양성을 제공하는 임계값은 무엇인가?

### 문제 4: 구현 과제

SWIM 장애 감지와 push-pull 상태 전파를 결합한 완전한 gossip 프로토콜을 구현하라. 프로토콜은:
- 3 프로토콜 주기 내에 장애 감지
- O(log N) 주기 내에 멤버십 업데이트 전파
- 추가 메시지를 피하기 위해 피기백 사용

### 문제 5: 비교 분석

10, 100, 1000, 10000 노드 규모에서 gossip 기반 장애 감지와 하트비트 기반 감지를 비교하는 실험을 설계하라. 측정: 감지 시간, 거짓 양성률, 네트워크 대역폭.

---

## 12. 참고 문헌

1. Demers, A. et al. (1987). "Epidemic Algorithms for Replicated Database Maintenance." *PODC*.
2. Das, A., Gupta, I., & Motivala, A. (2002). "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol." *DSN*.
3. Hayashibara, N. et al. (2004). "The φ Accrual Failure Detector." *SRDS*.
4. Leitao, J., Pereira, J., & Rodrigues, L. (2007). "HyParView: A Membership Protocol for Reliable Gossip-Based Broadcast." *DSN*.
5. HashiCorp (2017). "Lifeguard: SWIM-ing with Situational Awareness."
6. Lakshman, A. & Malik, P. (2010). "Cassandra — A Decentralized Structured Storage System." *Operating Systems Review*.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 5. O'Reilly Media.

---

[다음: 레슨 22 — 서비스 디스커버리](./22_Service_Discovery.md)
