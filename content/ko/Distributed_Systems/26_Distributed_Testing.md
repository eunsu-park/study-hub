# 레슨 26: 분산 테스트 (Distributed Testing)

[개요](./00_Overview.md) | [이전: 벡터 클럭](./25_Vector_Clocks.md) | [다음: 분산 관측 가능성](./27_Distributed_Observability.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 분산 데이터베이스와 합의(consensus) 시스템을 위한 Jepsen 스타일 일관성 테스트 설계
2. 네트워크 파티션(network partition), 크래시(crash), 클럭 스큐(clock skew)를 위한 장애 주입(fault injection) 프레임워크 구현
3. 재현 가능한 분산 시스템 검증을 위한 결정론적 시뮬레이션 테스트(deterministic simulation testing) 구축
4. 프로덕션 수준 분산 시스템에 카오스 엔지니어링(chaos engineering) 원칙 적용
5. 분산 시스템 특화 테스트 커버리지 전략 분석 (선형화 가능성(linearizability) 검사, 트레이스 분석)

---

## 목차

1. [왜 분산 테스트가 어려운가](#1-왜-분산-테스트가-어려운가)
2. [장애 주입 프레임워크](#2-장애-주입-프레임워크)
3. [Jepsen 스타일 테스트](#3-jepsen-스타일-테스트)
4. [선형화 가능성 검사](#4-선형화-가능성-검사)
5. [결정론적 시뮬레이션 테스트](#5-결정론적-시뮬레이션-테스트)
6. [카오스 엔지니어링](#6-카오스-엔지니어링)
7. [속성 기반 테스트](#7-속성-기반-테스트)
8. [트레이스 분석](#8-트레이스-분석)
9. [실제 테스트 프레임워크](#9-실제-테스트-프레임워크)
10. [요약 및 핵심 정리](#10-요약-및-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 왜 분산 테스트가 어려운가

### 1.1 도전 과제

```python
import random
import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


def explain_testing_challenges():
    """분산 시스템이 테스트하기에 왜 특별히 어려운지 설명한다."""
    print("=== Why Distributed Testing Is Hard ===\n")

    challenges = {
        "Non-determinism": (
            "Same inputs can produce different outputs due to message ordering, "
            "timing, and thread scheduling."
        ),
        "Partial failures": (
            "Any component can fail independently: nodes, network links, disks. "
            "A 5-node system has 2^5 - 1 = 31 failure combinations."
        ),
        "Heisenbugs": (
            "Adding logging or debugging changes timing, which changes behavior. "
            "The bug disappears when you look for it."
        ),
        "State space explosion": (
            "With N nodes, M messages, and F possible failures, the state space "
            "is O(M! × 2^F × N!) — astronomical even for small systems."
        ),
        "Emergent failures": (
            "Individual components work correctly in isolation, but the system "
            "fails when they interact under specific conditions."
        ),
    }

    for name, desc in challenges.items():
        print(f"  {name}:")
        print(f"    {desc}\n")


explain_testing_challenges()
```

---

## 2. 장애 주입 프레임워크

### 2.1 네트워크 장애 주입

```python
class FaultType(Enum):
    PARTITION = "partition"        # 그룹 간 네트워크 파티션
    DELAY = "delay"                # 메시지 지연
    DROP = "drop"                  # 메시지 드롭
    DUPLICATE = "duplicate"        # 메시지 복제
    REORDER = "reorder"            # 메시지 재정렬
    CORRUPT = "corrupt"            # 메시지 손상
    CRASH = "crash"                # 노드 크래시
    SLOW = "slow"                  # 느린 노드 (GC 일시 중지)
    CLOCK_SKEW = "clock_skew"     # 클럭 드리프트


@dataclass
class Fault:
    """시스템에 주입할 장애."""
    fault_type: FaultType
    target_nodes: list[str] = field(default_factory=list)
    duration_seconds: float = 5.0
    parameters: dict = field(default_factory=dict)
    start_time: float = 0.0


class FaultInjector:
    """
    분산 시스템 테스트를 위한 장애 주입(fault injection) 프레임워크.

    시뮬레이션된 분산 시스템에 장애(파티션, 지연, 크래시)를
    주입하고 결과를 기록한다.
    """

    def __init__(self):
        self.active_faults: list[Fault] = []
        self.fault_history: list[dict] = []
        self.message_interceptors: list[Callable] = []

    def inject(self, fault: Fault):
        """시스템에 장애를 주입한다."""
        fault.start_time = time.time()
        self.active_faults.append(fault)
        self.fault_history.append({
            "type": fault.fault_type.value,
            "targets": fault.target_nodes,
            "duration": fault.duration_seconds,
            "start": fault.start_time,
        })

    def clear_expired(self):
        """만료된 장애를 제거한다."""
        now = time.time()
        self.active_faults = [
            f for f in self.active_faults
            if now - f.start_time < f.duration_seconds
        ]

    def should_drop_message(self, src: str, dst: str) -> bool:
        """활성 장애로 인해 메시지를 드롭해야 하는지 확인한다."""
        self.clear_expired()
        for fault in self.active_faults:
            if fault.fault_type == FaultType.PARTITION:
                groups = fault.parameters.get("groups", [])
                for group in groups:
                    if src in group and dst not in group:
                        return True

            elif fault.fault_type == FaultType.DROP:
                rate = fault.parameters.get("rate", 0.5)
                if (src in fault.target_nodes or dst in fault.target_nodes):
                    if random.random() < rate:
                        return True

        return False

    def message_delay(self, src: str, dst: str) -> float:
        """메시지에 대한 추가 지연을 계산한다."""
        delay = 0.0
        for fault in self.active_faults:
            if fault.fault_type == FaultType.DELAY:
                if src in fault.target_nodes or dst in fault.target_nodes:
                    delay += fault.parameters.get("delay_ms", 100) / 1000.0
        return delay

    def is_node_crashed(self, node_id: str) -> bool:
        """노드가 현재 크래시 상태인지 확인한다."""
        for fault in self.active_faults:
            if fault.fault_type == FaultType.CRASH and node_id in fault.target_nodes:
                return True
        return False

    def generate_random_scenario(self, nodes: list[str],
                                  duration: float = 30.0) -> list[Fault]:
        """무작위 장애 주입 시나리오를 생성한다."""
        faults = []
        num_faults = random.randint(1, 5)

        for _ in range(num_faults):
            fault_type = random.choice([
                FaultType.PARTITION, FaultType.CRASH, FaultType.DELAY, FaultType.DROP
            ])
            num_targets = random.randint(1, len(nodes) // 2)
            targets = random.sample(nodes, num_targets)
            fault_duration = random.uniform(1.0, duration / 2)

            params = {}
            if fault_type == FaultType.PARTITION:
                # 두 그룹으로 분할
                mid = len(nodes) // 2
                params["groups"] = [nodes[:mid], nodes[mid:]]
            elif fault_type == FaultType.DELAY:
                params["delay_ms"] = random.randint(50, 500)
            elif fault_type == FaultType.DROP:
                params["rate"] = random.uniform(0.1, 0.5)

            faults.append(Fault(
                fault_type=fault_type,
                target_nodes=targets,
                duration_seconds=fault_duration,
                parameters=params,
            ))

        return faults


def demonstrate_fault_injection():
    """테스트를 위한 장애 주입을 시연한다."""
    print("=== Fault Injection Framework ===\n")

    injector = FaultInjector()
    nodes = ["n1", "n2", "n3", "n4", "n5"]

    # 네트워크 파티션 주입
    injector.inject(Fault(
        fault_type=FaultType.PARTITION,
        target_nodes=nodes,
        duration_seconds=5.0,
        parameters={"groups": [["n1", "n2"], ["n3", "n4", "n5"]]},
    ))

    print("Network partition: {n1,n2} | {n3,n4,n5}")
    test_pairs = [("n1", "n2"), ("n1", "n3"), ("n3", "n4"), ("n2", "n5")]
    for src, dst in test_pairs:
        dropped = injector.should_drop_message(src, dst)
        print(f"  {src} → {dst}: {'DROPPED' if dropped else 'delivered'}")

    # 무작위 시나리오 생성
    print(f"\nRandom fault scenario:")
    scenario = injector.generate_random_scenario(nodes, duration=10.0)
    for fault in scenario:
        print(f"  {fault.fault_type.value}: targets={fault.target_nodes}, "
              f"duration={fault.duration_seconds:.1f}s")


demonstrate_fault_injection()
```

---

## 3. Jepsen 스타일 테스트

### 3.1 테스트 구조

```python
class JepsenTest:
    """
    Jepsen 스타일 분산 시스템 테스트.

    구조:
    1. Setup: 클러스터 시작, 설정
    2. Nemesis: 장애 주입 (파티션, 크래시)
    3. Workload: 클라이언트 작업 실행 (읽기, 쓰기)
    4. Check: 일관성 속성 검증 (선형화 가능성)
    """

    def __init__(self, name: str, nodes: list[str]):
        self.name = name
        self.nodes = nodes
        self.history: list[dict] = []
        self.errors: list[str] = []
        self.injector = FaultInjector()

    def setup(self):
        """1단계: 클러스터를 설정한다."""
        print(f"  Setup: Initializing {len(self.nodes)} nodes")

    def nemesis(self, faults: list[Fault]):
        """2단계: 장애를 주입한다."""
        for fault in faults:
            self.injector.inject(fault)
            print(f"  Nemesis: {fault.fault_type.value} on {fault.target_nodes}")

    def workload(self, operations: list[dict]):
        """
        3단계: 클라이언트 작업을 실행하고 이력을 기록한다.

        각 작업은 다음을 기록한다:
        - type: "invoke" (시작) 또는 "ok"/"fail"/"info" (결과)
        - f: 함수 (read, write, cas)
        - value: 관련된 값
        - process: 어떤 클라이언트 프로세스
        """
        for op in operations:
            invoke = {
                "type": "invoke",
                "f": op["f"],
                "value": op.get("value"),
                "process": op.get("process", 0),
                "time": time.time(),
            }
            self.history.append(invoke)

            # 실행 시뮬레이션
            success = not self.injector.is_node_crashed(
                random.choice(self.nodes)
            )

            result = {
                "type": "ok" if success else "fail",
                "f": op["f"],
                "value": op.get("value"),
                "process": op.get("process", 0),
                "time": time.time(),
            }
            self.history.append(result)

    def check(self, checker: 'ConsistencyChecker') -> dict:
        """4단계: 일관성 속성을 검사한다."""
        return checker.check(self.history)

    def run(self, faults: list[Fault], operations: list[dict],
            checker: 'ConsistencyChecker') -> dict:
        """전체 테스트를 실행한다."""
        print(f"\n=== Jepsen Test: {self.name} ===")
        self.setup()
        self.nemesis(faults)
        self.workload(operations)
        result = self.check(checker)
        print(f"  Result: {'PASS' if result['valid'] else 'FAIL'}")
        return result


class ConsistencyChecker:
    """일관성 검사기의 기본 클래스."""

    def check(self, history: list[dict]) -> dict:
        """이력에서 일관성 위반을 검사한다."""
        raise NotImplementedError


class RegisterChecker(ConsistencyChecker):
    """
    단일 레지스터의 선형화 가능성(linearizability)을 검사한다.

    이력이 선형화 가능하려면 실시간 순서와 일관된
    작업의 전체 순서가 존재하여 모든 읽기가
    가장 최근 선행 쓰기의 값을 반환해야 한다.
    """

    def check(self, history: list[dict]) -> dict:
        # 호출(invocation)과 응답(response)을 쌍으로 연결
        ops = []
        pending = {}

        for entry in history:
            if entry["type"] == "invoke":
                pending[entry["process"]] = entry
            elif entry["type"] in ("ok", "fail"):
                invoke = pending.pop(entry["process"], None)
                if invoke and entry["type"] == "ok":
                    ops.append({
                        "f": entry["f"],
                        "value": entry["value"],
                        "invoke_time": invoke["time"],
                        "complete_time": entry["time"],
                        "process": entry["process"],
                    })

        # 레지스터에 대한 단순 선형화 가능성 검사
        # (전체 검사는 NP-완전; 이것은 단순화된 버전)
        writes = [op for op in ops if op["f"] == "write"]
        reads = [op for op in ops if op["f"] == "read"]

        violations = []
        for read in reads:
            # 이 읽기가 시작되기 전에 완료된 가장 최근 쓰기 찾기
            preceding_writes = [
                w for w in writes
                if w["complete_time"] <= read["invoke_time"]
            ]
            if preceding_writes:
                last_write = max(preceding_writes, key=lambda w: w["complete_time"])
                if read["value"] != last_write["value"]:
                    violations.append({
                        "read": read,
                        "expected": last_write["value"],
                        "got": read["value"],
                    })

        return {
            "valid": len(violations) == 0,
            "operations": len(ops),
            "violations": violations,
        }


def demonstrate_jepsen_test():
    """Jepsen 스타일 테스트를 시연한다."""
    print("=== Jepsen-Style Testing ===\n")

    # 테스트 1: 장애 없음 — 통과해야 함
    test = JepsenTest("register-no-faults", ["n1", "n2", "n3"])
    operations = [
        {"f": "write", "value": 1, "process": 0},
        {"f": "read", "value": 1, "process": 1},
        {"f": "write", "value": 2, "process": 0},
        {"f": "read", "value": 2, "process": 1},
    ]
    result = test.run([], operations, RegisterChecker())

    # 테스트 2: 파티션 포함 — 실패할 수 있음
    test2 = JepsenTest("register-with-partition", ["n1", "n2", "n3"])
    faults = [Fault(
        fault_type=FaultType.PARTITION,
        target_nodes=["n1", "n2", "n3"],
        duration_seconds=5.0,
        parameters={"groups": [["n1"], ["n2", "n3"]]},
    )]
    result2 = test2.run(faults, operations, RegisterChecker())

    print(f"\n  Test results:")
    print(f"    No faults: {'PASS' if result['valid'] else 'FAIL'}")
    print(f"    With partition: {'PASS' if result2['valid'] else 'FAIL'}")


demonstrate_jepsen_test()
```

---

## 4. 선형화 가능성 검사

### 4.1 WGL 알고리즘 (단순화)

```python
class LinearizabilityChecker:
    """
    무차별 대입 열거를 사용한 선형화 가능성(linearizability) 검사기.

    작은 이력의 경우, 가능한 모든 선형화를 열거하고
    유효한 것이 있는지 확인한다. 일반적으로 NP-완전이지만
    작은 테스트 케이스에서는 실행 가능하다.

    프로덕션에서는 Wing & Gong의 알고리즘이나 Knossos를 사용한다.
    """

    def __init__(self):
        self.checked: int = 0

    def check(self, operations: list[dict], model: dict) -> bool:
        """
        작업 이력이 선형화 가능한지 확인한다.

        Args:
            operations: {f, args, ret, start, end}의 목록
            model: 순차 사양의 초기 상태

        Returns:
            선형화 가능하면 True
        """
        self.checked = 0
        return self._search(operations, dict(model), set())

    def _search(self, remaining: list[dict], state: dict,
                linearized: set) -> bool:
        """유효한 선형화를 위한 재귀 탐색."""
        if not remaining:
            return True

        self.checked += 1

        # 다음으로 선형화할 수 있는 각 작업 시도
        for i, op in enumerate(remaining):
            # 작업의 간격이 "현재"와 겹치면 선형화 가능
            # 단순화: 남은 각 작업을 시도
            new_state = dict(state)
            valid = self._apply_op(new_state, op)

            if valid:
                rest = remaining[:i] + remaining[i+1:]
                if self._search(rest, new_state, linearized | {i}):
                    return True

        return False

    def _apply_op(self, state: dict, op: dict) -> bool:
        """모델 상태에 작업을 적용하고 반환값이 일치하는지 확인한다."""
        f = op["f"]
        if f == "write":
            state["register"] = op["args"]
            return True  # 모델에서 쓰기는 항상 성공
        elif f == "read":
            expected = state.get("register")
            return op["ret"] == expected
        elif f == "cas":
            old, new = op["args"]
            if state.get("register") == old:
                state["register"] = new
                return op["ret"] == True
            else:
                return op["ret"] == False
        return False


def demonstrate_linearizability_check():
    """선형화 가능성 검사를 시연한다."""
    print("=== Linearizability Checking ===\n")

    checker = LinearizabilityChecker()

    # 선형화 가능한 이력
    history1 = [
        {"f": "write", "args": 1, "ret": None, "start": 0, "end": 1},
        {"f": "read", "args": None, "ret": 1, "start": 2, "end": 3},
        {"f": "write", "args": 2, "ret": None, "start": 4, "end": 5},
        {"f": "read", "args": None, "ret": 2, "start": 6, "end": 7},
    ]
    result1 = checker.check(history1, {"register": None})
    print(f"History 1 (sequential): linearizable={result1} "
          f"(checked {checker.checked} orderings)")

    # 선형화 불가능한 이력
    history2 = [
        {"f": "write", "args": 1, "ret": None, "start": 0, "end": 2},
        {"f": "write", "args": 2, "ret": None, "start": 1, "end": 3},
        {"f": "read", "args": None, "ret": 1, "start": 4, "end": 5},
        # write(2) 완료 후 읽기가 1을 반환 → 선형화 불가능
    ]
    result2 = checker.check(history2, {"register": None})
    print(f"History 2 (stale read): linearizable={result2} "
          f"(checked {checker.checked} orderings)")


demonstrate_linearizability_check()
```

---

## 5. 결정론적 시뮬레이션 테스트

### 5.1 결정론적 시뮬레이션

```python
class DeterministicSimulator:
    """
    결정론적 시뮬레이션 테스트(deterministic simulation testing) 프레임워크.

    모든 비결정성 소스(시간, 네트워크, 랜덤)를
    시뮬레이터가 제어한다. 이로 인해 테스트가:
    - 재현 가능: 같은 시드 → 같은 실행
    - 빠름: 실제 I/O 없음, sleep 없음
    - 철저함: 많은 스케줄 탐색 가능

    FoundationDB, TigerBeetle 등에서 사용한다.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.virtual_time: float = 0.0
        self.event_queue: list[Tuple[float, str, dict]] = []
        self.nodes: Dict[str, Any] = {}
        self.message_log: list[dict] = []
        self.delivered: int = 0
        self.dropped: int = 0

    def register_node(self, node_id: str, handler: Callable):
        """노드와 그 메시지 핸들러를 등록한다."""
        self.nodes[node_id] = handler

    def send(self, src: str, dst: str, msg: dict, delay: Optional[float] = None):
        """메시지 전달을 스케줄링한다."""
        if delay is None:
            delay = self.rng.uniform(0.001, 0.050)  # 1-50ms

        deliver_time = self.virtual_time + delay
        self.event_queue.append((deliver_time, dst, {
            "from": src,
            "to": dst,
            **msg,
        }))
        # 시간순으로 큐 정렬 유지
        self.event_queue.sort(key=lambda x: x[0])

    def schedule_timer(self, node_id: str, delay: float, msg: dict):
        """노드에 대한 타이머 이벤트를 스케줄링한다."""
        deliver_time = self.virtual_time + delay
        self.event_queue.append((deliver_time, node_id, {
            "type": "timer",
            "node": node_id,
            **msg,
        }))
        self.event_queue.sort(key=lambda x: x[0])

    def step(self) -> bool:
        """다음 이벤트를 처리한다. 이벤트가 없으면 False를 반환한다."""
        if not self.event_queue:
            return False

        deliver_time, node_id, msg = self.event_queue.pop(0)
        self.virtual_time = deliver_time

        # 선택적으로 메시지 드롭 (재현성을 위해 RNG로 제어)
        if msg.get("type") != "timer" and self.rng.random() < 0.0:  # 0% 드롭률
            self.dropped += 1
            return True

        handler = self.nodes.get(node_id)
        if handler:
            handler(msg)
            self.delivered += 1

        self.message_log.append({
            "time": self.virtual_time,
            "node": node_id,
            "msg": msg,
        })

        return True

    def run(self, max_steps: int = 10000) -> int:
        """완료 또는 최대 스텝까지 시뮬레이션을 실행한다."""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return steps

    def stats(self) -> dict:
        return {
            "seed": self.seed,
            "virtual_time": round(self.virtual_time, 6),
            "delivered": self.delivered,
            "dropped": self.dropped,
            "remaining_events": len(self.event_queue),
        }


def demonstrate_deterministic_sim():
    """결정론적 시뮬레이션 테스트를 시연한다."""
    print("=== Deterministic Simulation Testing ===\n")

    # 단순 리더 선출 시뮬레이션
    elected_leader = {"leader": None}

    def make_handler(node_id, sim):
        def handler(msg):
            if msg.get("type") == "timer":
                # 선거 타임아웃 — 선거 시작
                for peer in ["n1", "n2", "n3"]:
                    if peer != node_id:
                        sim.send(node_id, peer, {
                            "type": "vote_request",
                            "candidate": node_id,
                        })
            elif msg.get("type") == "vote_request":
                sim.send(node_id, msg["from"], {
                    "type": "vote_response",
                    "voter": node_id,
                    "granted": True,
                })
            elif msg.get("type") == "vote_response":
                if msg.get("granted") and elected_leader["leader"] is None:
                    elected_leader["leader"] = node_id
        return handler

    # 두 가지 다른 시드로 실행
    for seed in [42, 123]:
        sim = DeterministicSimulator(seed=seed)
        elected_leader["leader"] = None

        for nid in ["n1", "n2", "n3"]:
            sim.register_node(nid, make_handler(nid, sim))
            # 무작위 선거 타임아웃
            timeout = sim.rng.uniform(0.150, 0.300)
            sim.schedule_timer(nid, timeout, {"election": True})

        steps = sim.run(max_steps=100)
        print(f"  Seed {seed}: leader={elected_leader['leader']}, "
              f"steps={steps}, {sim.stats()}")

    print(f"\n  Key insight: Same seed → same leader every time")
    print(f"  Different seeds explore different schedules")


demonstrate_deterministic_sim()
```

---

## 6. 카오스 엔지니어링

### 6.1 카오스 실험 설계

```python
@dataclass
class ChaosExperiment:
    """카오스 엔지니어링(chaos engineering) 실험 정의."""
    name: str
    hypothesis: str
    method: str
    abort_conditions: list[str]
    metrics: list[str]
    blast_radius: str = "single-service"


class ChaosRunner:
    """
    카오스 엔지니어링 실험 실행기.

    카오스 엔지니어링 원칙을 따른다:
    1. 정상 상태 정의 (정상 동작)
    2. 장애 중 정상 상태가 계속된다고 가설 수립
    3. 실제 장애 주입
    4. 차이 관찰
    """

    def __init__(self):
        self.experiments: list[ChaosExperiment] = []
        self.results: list[dict] = []

    def define(self, experiment: ChaosExperiment):
        self.experiments.append(experiment)

    def run(self, experiment: ChaosExperiment,
            steady_state_check: Callable,
            inject_fault: Callable,
            observe: Callable) -> dict:
        """단일 카오스 실험을 실행한다."""
        # 1. 정상 상태 확인
        baseline = steady_state_check()
        if not baseline["healthy"]:
            return {"status": "aborted", "reason": "System not in steady state"}

        # 2. 장애 주입
        inject_fault()

        # 3. 관찰
        observation = observe()

        # 4. 비교
        deviation = abs(observation.get("metric", 0) - baseline.get("metric", 0))
        passed = deviation < observation.get("tolerance", float("inf"))

        result = {
            "experiment": experiment.name,
            "hypothesis": experiment.hypothesis,
            "baseline": baseline,
            "observation": observation,
            "deviation": deviation,
            "passed": passed,
        }
        self.results.append(result)
        return result


def demonstrate_chaos_engineering():
    """카오스 엔지니어링 실험 설계를 시연한다."""
    print("=== Chaos Engineering ===\n")

    experiments = [
        ChaosExperiment(
            name="leader-crash",
            hypothesis="System elects new leader within 5s and maintains availability",
            method="Kill the Raft leader process",
            abort_conditions=["Error rate > 50%", "Latency p99 > 10s"],
            metrics=["election_time", "error_rate", "latency_p99"],
        ),
        ChaosExperiment(
            name="network-partition",
            hypothesis="Minority partition stops accepting writes; majority continues",
            method="iptables partition isolating 2 of 5 nodes",
            abort_conditions=["Data loss detected", "Split brain detected"],
            metrics=["write_availability", "read_consistency", "partition_duration"],
        ),
        ChaosExperiment(
            name="clock-skew",
            hypothesis="System maintains consistency with 500ms clock skew",
            method="Inject 500ms clock offset on one node via ntpd",
            abort_conditions=["Consistency violation", "Transaction rollback > 5%"],
            metrics=["consistency_violations", "transaction_success_rate"],
        ),
    ]

    for exp in experiments:
        print(f"  Experiment: {exp.name}")
        print(f"    Hypothesis: {exp.hypothesis}")
        print(f"    Method: {exp.method}")
        print(f"    Abort if: {', '.join(exp.abort_conditions)}")
        print(f"    Metrics: {', '.join(exp.metrics)}")
        print()

    # 하나의 실험 실행 시뮬레이션
    runner = ChaosRunner()
    result = runner.run(
        experiments[0],
        steady_state_check=lambda: {"healthy": True, "metric": 0.01},
        inject_fault=lambda: None,
        observe=lambda: {"metric": 0.03, "tolerance": 0.05},
    )
    print(f"  Result: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"  Deviation: {result['deviation']}")


demonstrate_chaos_engineering()
```

---

## 7. 속성 기반 테스트

### 7.1 무작위 작업 생성

```python
class DistributedPropertyTest:
    """
    분산 시스템을 위한 속성 기반 테스트(property-based testing).

    작업과 장애의 무작위 시퀀스를 생성한 후,
    각 시퀀스 후 불변식(invariant)을 검증한다.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.invariant_checks: list[Callable] = []

    def add_invariant(self, name: str, check: Callable):
        """모든 작업 시퀀스 후에 유지되어야 하는 불변식을 추가한다."""
        self.invariant_checks.append((name, check))

    def generate_operations(self, num_ops: int,
                            op_types: list[str]) -> list[dict]:
        """무작위 작업 시퀀스를 생성한다."""
        ops = []
        for _ in range(num_ops):
            op_type = self.rng.choice(op_types)
            key = f"key-{self.rng.randint(0, 9)}"
            value = f"val-{self.rng.randint(0, 99)}"

            if op_type == "write":
                ops.append({"f": "write", "key": key, "value": value})
            elif op_type == "read":
                ops.append({"f": "read", "key": key})
            elif op_type == "delete":
                ops.append({"f": "delete", "key": key})
            elif op_type == "cas":
                old = f"val-{self.rng.randint(0, 99)}"
                ops.append({"f": "cas", "key": key, "old": old, "new": value})
        return ops

    def run(self, system_under_test, num_trials: int = 100,
            ops_per_trial: int = 50) -> dict:
        """속성 기반 테스트를 실행한다."""
        failures = []

        for trial in range(num_trials):
            ops = self.generate_operations(
                ops_per_trial, ["write", "read", "delete", "cas"]
            )

            # 작업 실행
            for op in ops:
                system_under_test.execute(op)

            # 불변식 검사
            for name, check in self.invariant_checks:
                try:
                    result = check(system_under_test)
                    if not result:
                        failures.append({
                            "trial": trial,
                            "invariant": name,
                            "ops": ops,
                        })
                except Exception as e:
                    failures.append({
                        "trial": trial,
                        "invariant": name,
                        "error": str(e),
                    })

        return {
            "trials": num_trials,
            "failures": len(failures),
            "first_failure": failures[0] if failures else None,
        }


def demonstrate_property_testing():
    """분산 시스템을 위한 속성 기반 테스트를 시연한다."""
    print("=== Property-Based Testing ===\n")

    print("Properties to verify:")
    properties = [
        ("Monotonic reads", "Once a value is read, subsequent reads return same or newer"),
        ("Read-your-writes", "A write followed by a read returns the written value"),
        ("No phantom reads", "A read that returns 'not found' cannot later return 'found' without a write"),
        ("Consistent prefix", "If write A happened before write B, no observer sees B without A"),
        ("Convergence", "All replicas eventually have the same state after quiescence"),
    ]

    for name, desc in properties:
        print(f"  {name}: {desc}")


demonstrate_property_testing()
```

---

## 8. 트레이스 분석

### 8.1 분산 트레이스 검증

```python
class TraceAnalyzer:
    """
    분산 시스템의 실행 트레이스를 분석한다.

    다음을 검사한다:
    - 인과적 일관성(causal consistency) 위반
    - 메시지 순서 위반
    - 상태 머신 불변식 위반
    - 성능 이상
    """

    def __init__(self):
        self.events: list[dict] = []

    def add_event(self, event: dict):
        self.events.append(event)

    def check_causal_consistency(self) -> list[dict]:
        """트레이스에서 인과적 일관성 위반을 찾는다."""
        violations = []
        writes_by_key: Dict[str, list] = defaultdict(list)
        reads_by_key: Dict[str, list] = defaultdict(list)

        for event in self.events:
            if event.get("op") == "write":
                writes_by_key[event["key"]].append(event)
            elif event.get("op") == "read":
                reads_by_key[event["key"]].append(event)

        for key, reads in reads_by_key.items():
            writes = writes_by_key.get(key, [])
            for read in reads:
                # 읽기 값이 유효한 쓰기에 해당하는지 확인
                valid_values = {w["value"] for w in writes if w["time"] <= read["time"]}
                if read["value"] not in valid_values and read["value"] is not None:
                    violations.append({
                        "type": "stale_read",
                        "key": key,
                        "read_value": read["value"],
                        "valid_values": valid_values,
                    })

        return violations

    def check_state_machine_invariants(self,
                                        invariant: Callable) -> list[dict]:
        """트레이스의 각 시점에서 상태 머신 불변식을 검사한다."""
        violations = []
        state: Dict[str, Any] = {}

        for event in self.events:
            if event.get("op") == "write":
                state[event["key"]] = event["value"]
            elif event.get("op") == "delete":
                state.pop(event.get("key"), None)

            if not invariant(state):
                violations.append({
                    "type": "invariant_violation",
                    "event": event,
                    "state": dict(state),
                })

        return violations


def demonstrate_trace_analysis():
    """분산 시스템 검증을 위한 트레이스 분석을 시연한다."""
    print("=== Trace Analysis ===\n")

    analyzer = TraceAnalyzer()

    # 트레이스 이벤트 추가
    trace = [
        {"op": "write", "key": "x", "value": 1, "node": "n1", "time": 1.0},
        {"op": "write", "key": "x", "value": 2, "node": "n2", "time": 2.0},
        {"op": "read", "key": "x", "value": 1, "node": "n3", "time": 3.0},  # 오래된 값!
        {"op": "read", "key": "x", "value": 2, "node": "n1", "time": 4.0},
    ]

    for event in trace:
        analyzer.add_event(event)

    violations = analyzer.check_causal_consistency()
    print(f"Causal consistency violations: {len(violations)}")
    for v in violations:
        print(f"  {v}")

    # 불변식 검사: 잔액은 절대 음수가 되어서는 안 됨
    bank_trace = [
        {"op": "write", "key": "balance", "value": 100, "time": 1.0},
        {"op": "write", "key": "balance", "value": 50, "time": 2.0},
        {"op": "write", "key": "balance", "value": -10, "time": 3.0},
    ]

    analyzer2 = TraceAnalyzer()
    for event in bank_trace:
        analyzer2.add_event(event)

    inv_violations = analyzer2.check_state_machine_invariants(
        lambda state: state.get("balance", 0) >= 0
    )
    print(f"\nBalance invariant violations: {len(inv_violations)}")
    for v in inv_violations:
        print(f"  balance={v['state'].get('balance')} at event {v['event']}")


demonstrate_trace_analysis()
```

---

## 9. 실제 테스트 프레임워크

### 9.1 프레임워크 비교

```python
def compare_testing_frameworks():
    """분산 시스템 테스트 프레임워크를 비교한다."""
    print("=== Testing Framework Comparison ===\n")

    frameworks = [
        {"name": "Jepsen", "language": "Clojure",
         "approach": "Black-box, fault injection, linearizability checking",
         "used_by": "CockroachDB, etcd, MongoDB, Redis, Kafka"},
        {"name": "FoundationDB Simulation", "language": "C++",
         "approach": "Deterministic simulation, single-threaded, virtual time",
         "used_by": "FoundationDB (100M+ random test hours)"},
        {"name": "TLA+/TLC", "language": "TLA+",
         "approach": "Model checking, exhaustive state space exploration",
         "used_by": "AWS (S3, DynamoDB, EBS), Azure (Cosmos DB)"},
        {"name": "Chaos Monkey", "language": "Go",
         "approach": "Random instance termination in production",
         "used_by": "Netflix"},
        {"name": "Litmus", "language": "Go",
         "approach": "Kubernetes-native chaos engineering",
         "used_by": "CNCF ecosystem"},
    ]

    for fw in frameworks:
        print(f"  {fw['name']} ({fw['language']}):")
        print(f"    Approach: {fw['approach']}")
        print(f"    Used by: {fw['used_by']}")
        print()


compare_testing_frameworks()
```

---

## 10. 요약 및 핵심 정리

### 테스트 전략 매트릭스

> **분산 테스트 전략 (DISTRIBUTED TESTING STRATEGY)**
>
> 단위 테스트 → 개별 컴포넌트의 정확성
> 통합 테스트 → 컴포넌트 상호작용
> 결정론적 시뮬레이션 → 철저한 스케줄 탐색
> 속성 기반 테스트 → 무작위 작업 하의 불변식 검증
> Jepsen 테스트 → 실제 장애 하의 일관성
> 카오스 엔지니어링 → 프로덕션에서의 복원력

### 핵심 원칙

1. **결정론적 시뮬레이션이 골드 스탠다드(gold standard)이다**: 재현 가능하고, 빠르며, 철저하다. FoundationDB에서 사용한다.
2. **Jepsen이 실제 버그를 찾는다**: 모든 주요 데이터베이스에서 Jepsen에 의해 버그가 발견되었다.
3. **속성 기반 > 예제 기반**: 무작위 작업을 생성하고; 불변식이 유지되는지 확인한다.
4. **프로덕션의 카오스가 테스트가 놓치는 것을 발견한다**: 생각하지 못한 장애 모드를 발견한다.
5. **선형화 가능성 검사는 NP-완전이다**: 큰 이력에는 근사치를 사용한다.

---

## 11. 연습 문제

### 문제 1: 장애 시나리오 설계

5개 노드 Raft 클러스터에 대한 5가지 장애 주입 시나리오를 설계한다. 각각은 특정 안전성(safety) 또는 활동성(liveness) 속성을 대상으로 해야 한다. 예상 동작을 설명한다.

### 문제 2: 선형화 가능성 검사

다음 동시 이력이 선형화 가능한지 판단한다:
- 클라이언트 A: write(1) at t=0, ok at t=2
- 클라이언트 B: write(2) at t=1, ok at t=3
- 클라이언트 C: read() at t=4, returns 1

### 문제 3: 시뮬레이션 설계

가십 프로토콜(gossip protocol)을 위한 결정론적 시뮬레이터를 설계한다. 시뮬레이터는 메시지 순서, 타이밍, 장애를 제어해야 한다. O(log N) 라운드 내 수렴을 검증한다.

### 문제 4: 구현 도전

Jepsen과 유사한 테스트 프레임워크를 구축한다: 3개 노드 KV 스토어를 시작하고, 동시 쓰기/읽기 워크로드를 실행하고, 네트워크 파티션을 주입하고, 결과 이력의 선형화 가능성을 검사한다.

### 문제 5: 카오스 실험

3개 서비스(A→B→C)가 있는 마이크로서비스 시스템에 대한 카오스 실험을 설계한다. 다음을 정의한다: 정상 상태 가설, 장애 주입 방법, 롤백 기준, 예상 폭발 반경(blast radius).

---

## 12. 참고 문헌

1. Kingsbury, K. (2013-2024). "Jepsen: Distributed Systems Safety Research." https://jepsen.io
2. FoundationDB (2021). "Testing Distributed Systems w/ Deterministic Simulation." (FoundationDB paper, SIGMOD 2021)
3. Alvaro, P. et al. (2015). "Lineage-driven Fault Injection." *SIGMOD*.
4. Netflix (2012). "Chaos Monkey." (Principles of Chaos Engineering)
5. Holzmann, G. (2003). *The SPIN Model Checker*. Addison-Wesley.
6. Wing, J. & Gong, C. (1993). "Testing and Verifying Concurrent Objects." *JPSM*.
7. Lamport, L. (2002). "Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers."

---

[다음: 레슨 27 — 분산 관측 가능성](./27_Distributed_Observability.md)
