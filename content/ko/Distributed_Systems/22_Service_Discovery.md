# 레슨 22: 서비스 디스커버리

[개요](./00_Overview.md) | [이전: Gossip 프로토콜](./21_Gossip_Protocols.md) | [다음: 분산 속도 제한](./23_Distributed_Rate_Limiting.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. DNS 기반, 레지스트리 기반, gossip 기반 접근법을 사용한 서비스 디스커버리(service discovery) 시스템 설계
2. 생존 프로브(liveness probe), 준비 프로브(readiness probe), 서킷 브레이커(circuit breaker)를 사용한 헬스 체크(health checking) 구현
3. 분산 서비스를 위한 클라이언트 측(client-side) 및 서버 측(server-side) 로드 밸런싱(load balancing) 전략 구축
4. TTL, 하트비트(heartbeat), 등록 해제(deregistration)를 사용한 서비스 등록(service registration) 구현
5. 서비스 디스커버리 및 설정 관리를 위한 Consul, etcd, ZooKeeper 비교

---

## 목차

1. [서비스 디스커버리 기초](#1-서비스-디스커버리-기초)
2. [DNS 기반 디스커버리](#2-dns-기반-디스커버리)
3. [레지스트리 기반 디스커버리](#3-레지스트리-기반-디스커버리)
4. [헬스 체크](#4-헬스-체크)
5. [로드 밸런싱 전략](#5-로드-밸런싱-전략)
6. [클라이언트 측 vs 서버 측 디스커버리](#6-클라이언트-측-vs-서버-측-디스커버리)
7. [Consul 심층 분석](#7-consul-심층-분석)
8. [서비스 메시 통합](#8-서비스-메시-통합)
9. [실제 패턴](#9-실제-패턴)
10. [요약과 핵심 정리](#10-요약과-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 서비스 디스커버리 기초

### 1.1 문제

동적으로 스케일링되는 서비스를 가진 분산 시스템에서, 서비스 A는 서비스 B 인스턴스의 현재 네트워크 주소를 어떻게 찾는가?

```
정적 설정:                        동적 디스커버리:
  서비스 A → 10.0.1.5:8080         서비스 A → 레지스트리 → 10.0.1.5:8080
                                                       → 10.0.2.3:8080
  문제: B가 이동하면?                                   → 10.0.3.7:8080
  B가 3개 인스턴스로 스케일하면?      레지스트리가 라이브 인스턴스를 추적
```

### 1.2 디스커버리 패턴

```python
import time
import random
import hashlib
import threading
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class DiscoveryPattern(Enum):
    DNS_BASED = "dns"          # DNS SRV/A 레코드
    REGISTRY = "registry"      # 전용 서비스 레지스트리
    GOSSIP = "gossip"          # 피어-투-피어 gossip
    PLATFORM = "platform"      # 플랫폼 네이티브 (K8s, ECS)


@dataclass
class ServiceInstance:
    """서비스의 단일 인스턴스."""
    service_name: str
    instance_id: str
    host: str
    port: int
    metadata: dict = field(default_factory=dict)
    health: str = "healthy"  # healthy, unhealthy, draining
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    ttl: float = 30.0  # 자동 등록 해제 전 초
    weight: int = 100   # 로드 밸런싱 가중치
    zone: str = ""       # 가용 영역(availability zone)
```

---

## 2. DNS 기반 디스커버리

### 2.1 DNS SRV 레코드

```python
class DNSServiceDiscovery:
    """
    DNS SRV 레코드를 사용한 DNS 기반 서비스 디스커버리.

    DNS SRV 레코드가 제공하는 것:
    - 서비스 이름 → (호스트, 포트, 우선순위, 가중치)
    - 성능을 위한 TTL 기반 캐싱
    - 범용 클라이언트 지원

    한계:
    - TTL 캐싱이 변경 전파를 지연
    - 헬스 체크 없음 (외부 메커니즘 필요)
    - 제한된 메타데이터 지원
    """

    def __init__(self):
        self.records: Dict[str, list[dict]] = defaultdict(list)  # 서비스 → SRV 레코드
        self.cache: Dict[str, dict] = {}  # 캐시된 룩업
        self.cache_ttl: float = 5.0  # 기본 TTL 초

    def register(self, service_name: str, host: str, port: int,
                 priority: int = 10, weight: int = 100, ttl: float = 30.0):
        """DNS SRV 레코드를 통해 서비스 인스턴스를 등록한다."""
        record = {
            "host": host,
            "port": port,
            "priority": priority,
            "weight": weight,
            "ttl": ttl,
        }
        self.records[service_name].append(record)

    def resolve(self, service_name: str) -> list[dict]:
        """
        서비스 이름을 인스턴스로 해석한다.

        캐싱을 포함한 DNS 해석을 모방한다.
        """
        # 캐시 확인
        cached = self.cache.get(service_name)
        if cached and time.time() - cached["timestamp"] < self.cache_ttl:
            return cached["records"]

        # "DNS 룩업"
        records = self.records.get(service_name, [])

        # 우선순위순 (낮을수록 높은 우선순위), 그 다음 가중치순 정렬
        records = sorted(records, key=lambda r: (r["priority"], -r["weight"]))

        # 결과 캐시
        self.cache[service_name] = {
            "records": records,
            "timestamp": time.time(),
        }

        return records

    def weighted_select(self, service_name: str) -> Optional[dict]:
        """가중 무작위 선택을 사용하여 인스턴스를 선택한다."""
        records = self.resolve(service_name)
        if not records:
            return None

        # 우선순위별 그룹화 — 가장 높은 우선순위 그룹만 사용
        best_priority = records[0]["priority"]
        candidates = [r for r in records if r["priority"] == best_priority]

        # 우선순위 그룹 내에서 가중 무작위 선택
        total_weight = sum(r["weight"] for r in candidates)
        if total_weight == 0:
            return random.choice(candidates)

        r = random.uniform(0, total_weight)
        cumulative = 0
        for record in candidates:
            cumulative += record["weight"]
            if r <= cumulative:
                return record

        return candidates[-1]


def demonstrate_dns_discovery():
    """DNS 기반 서비스 디스커버리를 시연한다."""
    print("=== DNS 기반 서비스 디스커버리 ===\n")

    dns = DNSServiceDiscovery()

    # 인스턴스 등록
    dns.register("api.example.com", "10.0.1.5", 8080, priority=10, weight=70)
    dns.register("api.example.com", "10.0.2.3", 8080, priority=10, weight=30)
    dns.register("api.example.com", "10.0.3.7", 8080, priority=20, weight=100)  # 백업

    # 해석
    records = dns.resolve("api.example.com")
    print("api.example.com의 SRV 레코드:")
    for r in records:
        print(f"  {r['host']}:{r['port']} priority={r['priority']} weight={r['weight']}")

    # 가중 선택 시뮬레이션
    selections = defaultdict(int)
    for _ in range(1000):
        selected = dns.weighted_select("api.example.com")
        if selected:
            selections[f"{selected['host']}:{selected['port']}"] += 1

    print(f"\n가중 선택 (1000 요청):")
    for addr, count in sorted(selections.items()):
        print(f"  {addr}: {count} ({count/10:.1f}%)")


demonstrate_dns_discovery()
```

---

## 3. 레지스트리 기반 디스커버리

### 3.1 서비스 레지스트리(Service Registry)

```python
class ServiceRegistry:
    """
    헬스 체크가 포함된 중앙화된 서비스 레지스트리.

    서비스가 자신을 등록하고 주기적으로 하트비트를 보낸다.
    레지스트리는 하트비트를 놓친 인스턴스의 등록을 해제한다.
    클라이언트는 레지스트리에 질의하여 건강한 인스턴스를 발견한다.
    """

    def __init__(self):
        self.services: Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        self.watchers: Dict[str, list] = defaultdict(list)  # 서비스 → 콜백
        self.lock = threading.Lock()

    def register(self, instance: ServiceInstance) -> bool:
        """서비스 인스턴스를 등록한다."""
        with self.lock:
            self.services[instance.service_name][instance.instance_id] = instance
            self._notify_watchers(instance.service_name, "register", instance)
            return True

    def deregister(self, service_name: str, instance_id: str) -> bool:
        """서비스 인스턴스를 정상적으로 등록 해제한다."""
        with self.lock:
            instances = self.services.get(service_name, {})
            if instance_id in instances:
                instance = instances.pop(instance_id)
                self._notify_watchers(service_name, "deregister", instance)
                return True
            return False

    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """서비스 인스턴스의 하트비트를 처리한다."""
        with self.lock:
            instances = self.services.get(service_name, {})
            if instance_id in instances:
                instances[instance_id].last_heartbeat = time.time()
                return True
            return False

    def discover(self, service_name: str, healthy_only: bool = True) -> list[ServiceInstance]:
        """서비스의 인스턴스를 발견한다."""
        with self.lock:
            instances = list(self.services.get(service_name, {}).values())
            if healthy_only:
                instances = [i for i in instances if i.health == "healthy"]
            return instances

    def check_health(self):
        """
        만료된 인스턴스(놓친 하트비트)를 확인한다.

        레지스트리에서 주기적으로 호출된다.
        """
        now = time.time()
        with self.lock:
            for service_name in list(self.services.keys()):
                for iid in list(self.services[service_name].keys()):
                    instance = self.services[service_name][iid]
                    if now - instance.last_heartbeat > instance.ttl:
                        instance.health = "unhealthy"
                        if now - instance.last_heartbeat > instance.ttl * 3:
                            del self.services[service_name][iid]
                            self._notify_watchers(service_name, "expired", instance)

    def watch(self, service_name: str, callback):
        """서비스 변경에 대한 워처를 등록한다."""
        self.watchers[service_name].append(callback)

    def _notify_watchers(self, service_name: str, event: str, instance: ServiceInstance):
        """변경의 워처에 알린다."""
        for callback in self.watchers.get(service_name, []):
            try:
                callback(event, instance)
            except Exception:
                pass

    def stats(self) -> dict:
        """레지스트리 통계를 반환한다."""
        total = sum(len(instances) for instances in self.services.values())
        healthy = sum(
            sum(1 for i in instances.values() if i.health == "healthy")
            for instances in self.services.values()
        )
        return {
            "total_services": len(self.services),
            "total_instances": total,
            "healthy_instances": healthy,
        }


def demonstrate_service_registry():
    """등록과 디스커버리를 포함한 서비스 레지스트리를 시연한다."""
    print("=== 서비스 레지스트리 ===\n")

    registry = ServiceRegistry()

    # 인스턴스 등록
    for i in range(3):
        instance = ServiceInstance(
            service_name="user-service",
            instance_id=f"user-{i}",
            host=f"10.0.1.{i+1}",
            port=8080,
            metadata={"version": "2.1.0", "region": "us-east-1"},
            zone=f"us-east-1{'abc'[i]}",
        )
        registry.register(instance)

    for i in range(2):
        instance = ServiceInstance(
            service_name="order-service",
            instance_id=f"order-{i}",
            host=f"10.0.2.{i+1}",
            port=9090,
        )
        registry.register(instance)

    # 디스커버리
    print("user-service 발견:")
    for inst in registry.discover("user-service"):
        print(f"  {inst.instance_id}: {inst.host}:{inst.port} [{inst.zone}]")

    print(f"\n레지스트리 통계: {registry.stats()}")

    # 하트비트 실패 시뮬레이션
    print("\nuser-1의 하트비트 타임아웃 시뮬레이션...")
    registry.services["user-service"]["user-1"].last_heartbeat = time.time() - 100
    registry.check_health()

    print("헬스 체크 후:")
    for inst in registry.discover("user-service", healthy_only=True):
        print(f"  {inst.instance_id}: {inst.health}")


demonstrate_service_registry()
```

---

## 4. 헬스 체크

### 4.1 헬스 체크 유형

```python
class HealthCheckType(Enum):
    HTTP = "http"           # 헬스 엔드포인트에 HTTP GET
    TCP = "tcp"             # TCP 연결 검사
    GRPC = "grpc"           # gRPC 헬스 체크 프로토콜
    SCRIPT = "script"       # 스크립트/명령 실행
    TTL = "ttl"             # 수동 TTL 기반 (서비스가 보고)


@dataclass
class HealthCheck:
    """헬스 체크를 위한 설정."""
    check_type: HealthCheckType
    interval: float = 10.0       # 검사 간 초
    timeout: float = 5.0         # 검사 타임아웃 초
    deregister_after: float = 60.0  # 이 시간 동안 critical이면 등록 해제
    healthy_threshold: int = 3    # healthy로 표시하기 위한 연속 성공 횟수
    unhealthy_threshold: int = 2  # unhealthy로 표시하기 위한 연속 실패 횟수


class HealthChecker:
    """
    서비스 인스턴스를 위한 헬스 체크 시스템.

    생존 프로브(liveness probe, 프로세스가 살아있는가?),
    준비 프로브(readiness probe, 트래픽을 처리할 준비가 되었는가?),
    시작 프로브(startup probe, 초기화가 완료되었는가?)를 지원한다.
    """

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, list[bool]] = defaultdict(list)
        self.status: Dict[str, str] = {}  # instance_id → healthy/unhealthy/unknown
        self.consecutive_successes: Dict[str, int] = defaultdict(int)
        self.consecutive_failures: Dict[str, int] = defaultdict(int)

    def register_check(self, instance_id: str, check: HealthCheck):
        """인스턴스에 대한 헬스 체크를 등록한다."""
        self.checks[instance_id] = check
        self.status[instance_id] = "unknown"

    def record_result(self, instance_id: str, success: bool):
        """헬스 체크 결과를 기록한다."""
        self.results[instance_id].append(success)
        check = self.checks.get(instance_id)
        if not check:
            return

        if success:
            self.consecutive_successes[instance_id] += 1
            self.consecutive_failures[instance_id] = 0

            if self.consecutive_successes[instance_id] >= check.healthy_threshold:
                self.status[instance_id] = "healthy"
        else:
            self.consecutive_failures[instance_id] += 1
            self.consecutive_successes[instance_id] = 0

            if self.consecutive_failures[instance_id] >= check.unhealthy_threshold:
                self.status[instance_id] = "unhealthy"

    def get_healthy(self) -> list[str]:
        """건강한 인스턴스 목록을 반환한다."""
        return [iid for iid, s in self.status.items() if s == "healthy"]

    def simulate_checks(self, instance_id: str, results: list[bool]):
        """헬스 체크 결과 시퀀스를 시뮬레이션한다."""
        for result in results:
            self.record_result(instance_id, result)


class CircuitBreaker:
    """
    서비스 호출을 위한 서킷 브레이커.

    상태:
    - CLOSED: 정상 동작, 요청이 통과
    - OPEN: 서비스가 실패 중, 요청이 거부됨
    - HALF_OPEN: 서비스가 복구되었는지 테스트 중
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0,
                 half_open_max: int = 3):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max = half_open_max

        self.state: str = "closed"
        self.failure_count: int = 0
        self.success_count: int = 0
        self.last_failure_time: float = 0
        self.half_open_attempts: int = 0

    def can_execute(self) -> bool:
        """요청을 허용해야 하는지 확인한다."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # 리셋 타임아웃이 경과했는지 확인
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = "half_open"
                self.half_open_attempts = 0
                return True
            return False
        elif self.state == "half_open":
            return self.half_open_attempts < self.half_open_max
        return False

    def record_success(self):
        """성공적인 요청을 기록한다."""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.half_open_max:
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """실패한 요청을 기록한다."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "half_open":
            self.state = "open"
        elif self.state == "closed" and self.failure_count >= self.failure_threshold:
            self.state = "open"

    def get_state(self) -> dict:
        return {
            "state": self.state,
            "failures": self.failure_count,
            "successes": self.success_count,
        }


def demonstrate_health_checking():
    """서킷 브레이커를 포함한 헬스 체크를 시연한다."""
    print("=== 헬스 체크 ===\n")

    checker = HealthChecker()
    checker.register_check("api-1", HealthCheck(
        check_type=HealthCheckType.HTTP,
        healthy_threshold=3,
        unhealthy_threshold=2,
    ))

    # 시뮬레이션: 건강, 그 다음 실패, 그 다음 복구
    sequence = [True, True, True, True, False, False, True, True, True, True]
    print("api-1의 헬스 체크 시퀀스:")
    for i, result in enumerate(sequence):
        checker.record_result("api-1", result)
        print(f"  검사 {i+1}: {'통과' if result else '실패'} → "
              f"상태={checker.status['api-1']}")

    # 서킷 브레이커 시연
    print("\n=== 서킷 브레이커 ===\n")
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=0.5)

    operations = [
        ("success", True), ("success", True), ("failure", False),
        ("failure", False), ("failure", False),  # open으로 전환
        ("blocked", None), ("blocked", None),    # 요청 차단됨
    ]

    for desc, success in operations:
        allowed = cb.can_execute()
        if allowed and success is not None:
            if success:
                cb.record_success()
            else:
                cb.record_failure()
        print(f"  {desc:10s}: allowed={allowed}, state={cb.get_state()}")

    # 리셋 대기
    time.sleep(0.6)
    print(f"\n  타임아웃 후: allowed={cb.can_execute()}, state={cb.get_state()}")


demonstrate_health_checking()
```

---

## 5. 로드 밸런싱 전략

### 5.1 알고리즘 구현

```python
class LoadBalancer:
    """
    다중 전략을 가진 로드 밸런서.

    지원: round-robin, 가중 round-robin, 최소 연결(least-connections),
    무작위, 일관된 해싱, 영역 인식 라우팅.
    """

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances: list[ServiceInstance] = []
        self.rr_index: int = 0
        self.connections: Dict[str, int] = defaultdict(int)
        self.wrr_state: Dict[str, int] = {}

    def update_instances(self, instances: list[ServiceInstance]):
        """사용 가능한 인스턴스 목록을 업데이트한다."""
        self.instances = [i for i in instances if i.health == "healthy"]

    def select(self, key: str = "") -> Optional[ServiceInstance]:
        """설정된 전략을 사용하여 인스턴스를 선택한다."""
        if not self.instances:
            return None

        if self.strategy == "round_robin":
            return self._round_robin()
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin()
        elif self.strategy == "least_connections":
            return self._least_connections()
        elif self.strategy == "random":
            return self._random_select()
        elif self.strategy == "consistent_hash":
            return self._consistent_hash(key)
        elif self.strategy == "power_of_two":
            return self._power_of_two()
        else:
            return self._round_robin()

    def _round_robin(self) -> ServiceInstance:
        """단순 라운드 로빈 선택."""
        instance = self.instances[self.rr_index % len(self.instances)]
        self.rr_index += 1
        return instance

    def _weighted_round_robin(self) -> ServiceInstance:
        """부드러운 가중 알고리즘을 사용한 가중 라운드 로빈."""
        if not self.wrr_state:
            self.wrr_state = {i.instance_id: 0 for i in self.instances}

        total_weight = sum(i.weight for i in self.instances)

        # 각 인스턴스의 가중치만큼 증가
        for inst in self.instances:
            self.wrr_state[inst.instance_id] = (
                self.wrr_state.get(inst.instance_id, 0) + inst.weight
            )

        # 현재 가중치가 가장 높은 것 선택
        best = max(self.instances, key=lambda i: self.wrr_state.get(i.instance_id, 0))
        self.wrr_state[best.instance_id] -= total_weight
        return best

    def _least_connections(self) -> ServiceInstance:
        """활성 연결이 가장 적은 인스턴스를 선택한다."""
        return min(self.instances,
                   key=lambda i: self.connections.get(i.instance_id, 0))

    def _random_select(self) -> ServiceInstance:
        """무작위 선택."""
        return random.choice(self.instances)

    def _consistent_hash(self, key: str) -> ServiceInstance:
        """세션 어피니티를 위한 일관된 해시 기반 선택."""
        if not key:
            return self._random_select()
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        idx = h % len(self.instances)
        return self.instances[idx]

    def _power_of_two(self) -> ServiceInstance:
        """
        Power-of-two-choices: 2개의 무작위 인스턴스를 선택하고,
        연결이 더 적은 쪽을 선택한다.

        이 간단한 전략은 최소한의 오버헤드(2개의 무작위 선택만)로
        거의 최적의 부하 분배를 달성한다.
        """
        if len(self.instances) < 2:
            return self.instances[0]

        a, b = random.sample(self.instances, 2)
        conn_a = self.connections.get(a.instance_id, 0)
        conn_b = self.connections.get(b.instance_id, 0)
        return a if conn_a <= conn_b else b

    def connect(self, instance_id: str):
        """새 연결을 기록한다."""
        self.connections[instance_id] += 1

    def disconnect(self, instance_id: str):
        """연결 해제를 기록한다."""
        self.connections[instance_id] = max(0, self.connections.get(instance_id, 0) - 1)


def compare_load_balancing():
    """로드 밸런싱 전략을 비교한다."""
    print("=== 로드 밸런싱 전략 ===\n")

    instances = []
    for i in range(4):
        instances.append(ServiceInstance(
            service_name="api",
            instance_id=f"api-{i}",
            host=f"10.0.1.{i+1}",
            port=8080,
            weight=[100, 200, 50, 150][i],
        ))

    num_requests = 10000

    for strategy in ["round_robin", "weighted_round_robin", "random",
                     "least_connections", "power_of_two"]:
        lb = LoadBalancer(strategy=strategy)
        lb.update_instances(instances)

        counts = defaultdict(int)
        for _ in range(num_requests):
            selected = lb.select()
            if selected:
                counts[selected.instance_id] += 1
                lb.connect(selected.instance_id)
                # 다양한 요청 지속 시간 시뮬레이션
                if random.random() < 0.3:
                    lb.disconnect(selected.instance_id)

        print(f"{strategy}:")
        for iid in sorted(counts.keys()):
            pct = counts[iid] / num_requests * 100
            print(f"  {iid}: {counts[iid]:5d} ({pct:5.1f}%)")
        print()


compare_load_balancing()
```

---

## 6. 클라이언트 측 vs 서버 측 디스커버리

### 6.1 비교

```python
def compare_discovery_patterns():
    """클라이언트 측과 서버 측 서비스 디스커버리를 비교한다."""
    print("=== 디스커버리 패턴 비교 ===\n")

    patterns = {
        "클라이언트 측 디스커버리": {
            "description": "클라이언트가 레지스트리에 직접 질의하고, 로드 밸런싱 수행",
            "examples": "Netflix Eureka + Ribbon, gRPC 클라이언트 LB",
            "pros": ["추가 홉 없음", "클라이언트가 스마트 선택 가능", "LB 병목 없음"],
            "cons": ["클라이언트 복잡도", "언어별 구현", "밀접 결합"],
            "diagram": """
  클라이언트 → 레지스트리 → [인스턴스 목록]
    ↓
  클라이언트 → 인스턴스 (직접)
""",
        },
        "서버 측 디스커버리": {
            "description": "로드 밸런서/프록시가 디스커버리와 라우팅 처리",
            "examples": "AWS ALB, Nginx, Envoy, Kubernetes Service",
            "pros": ["단순한 클라이언트", "중앙화된 정책", "언어 독립적"],
            "cons": ["추가 네트워크 홉", "LB가 병목될 수 있음", "더 많은 인프라"],
            "diagram": """
  클라이언트 → 로드 밸런서 → 인스턴스
                 ↕
              레지스트리
""",
        },
        "서비스 메시 (사이드카)": {
            "description": "사이드카 프록시가 투명하게 디스커버리 처리",
            "examples": "Istio/Envoy, Linkerd, Consul Connect",
            "pros": ["앱에 투명", "풍부한 기능 (mTLS, 재시도)", "통일됨"],
            "cons": ["리소스 오버헤드", "운영 복잡도", "지연시간"],
            "diagram": """
  앱 → 사이드카 프록시 → 사이드카 프록시 → 앱
          ↕                 ↕
       컨트롤 플레인 (레지스트리, 설정)
""",
        },
    }

    for name, info in patterns.items():
        print(f"── {name} ──")
        print(f"  {info['description']}")
        print(f"  예시: {info['examples']}")
        print(f"  장점: {', '.join(info['pros'])}")
        print(f"  단점: {', '.join(info['cons'])}")
        print(f"  {info['diagram']}")


compare_discovery_patterns()
```

---

## 7. Consul 심층 분석

### 7.1 Consul 아키텍처 시뮬레이션

```python
class ConsulAgent:
    """
    시뮬레이션된 Consul 에이전트.

    Consul이 사용하는 것:
    - 카탈로그를 위한 Raft 합의 (서버 노드)
    - 멤버십과 장애 감지를 위한 Gossip (Serf)
    - 서비스 디스커버리를 위한 DNS와 HTTP API
    - 자동 서비스 관리를 위한 헬스 체크
    """

    def __init__(self, node_id: str, datacenter: str = "dc1", is_server: bool = False):
        self.node_id = node_id
        self.datacenter = datacenter
        self.is_server = is_server

        # 서비스 카탈로그 (서버에서 Raft로 복제)
        self.catalog: Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        # 로컬 서비스 (이 에이전트에 등록된)
        self.local_services: Dict[str, ServiceInstance] = {}
        # 헬스 체크 결과
        self.health_checks: Dict[str, bool] = {}
        # KV 스토어
        self.kv_store: Dict[str, dict] = {}

    def register_service(self, instance: ServiceInstance, check: Optional[HealthCheck] = None):
        """선택적 헬스 체크와 함께 로컬 서비스를 등록한다."""
        self.local_services[instance.instance_id] = instance
        self.catalog[instance.service_name][instance.instance_id] = instance
        if check:
            self.health_checks[instance.instance_id] = True  # 초기에 건강

    def discover(self, service_name: str, dc: str = "", tag: str = "",
                 healthy_only: bool = True) -> list[ServiceInstance]:
        """
        서비스 인스턴스를 발견한다.

        지원:
        - 교차 데이터센터 질의
        - 태그 기반 필터링
        - 헬스 기반 필터링
        """
        instances = list(self.catalog.get(service_name, {}).values())

        if dc and dc != self.datacenter:
            return []  # 실제 Consul에서는 원격 DC에 질의

        if tag:
            instances = [
                i for i in instances
                if tag in i.metadata.get("tags", [])
            ]

        if healthy_only:
            instances = [
                i for i in instances
                if self.health_checks.get(i.instance_id, True)
            ]

        return instances

    def dns_query(self, name: str) -> list[dict]:
        """
        Consul DNS 인터페이스에 대한 DNS 질의를 시뮬레이션한다.

        형식: <service>.service[.datacenter].consul
        """
        parts = name.split(".")
        if len(parts) >= 3 and parts[-1] == "consul" and parts[-2] == "service":
            service_name = parts[0]
            instances = self.discover(service_name)
            return [{"host": i.host, "port": i.port} for i in instances]
        return []

    def kv_put(self, key: str, value: str, flags: int = 0) -> bool:
        """KV 스토어에 값을 넣는다."""
        self.kv_store[key] = {
            "value": value,
            "flags": flags,
            "modify_index": time.time(),
        }
        return True

    def kv_get(self, key: str) -> Optional[dict]:
        """KV 스토어에서 값을 가져온다."""
        return self.kv_store.get(key)

    def kv_list(self, prefix: str) -> list[str]:
        """주어진 접두사를 가진 키를 나열한다."""
        return [k for k in self.kv_store if k.startswith(prefix)]


def demonstrate_consul():
    """Consul 서비스 디스커버리 기능을 시연한다."""
    print("=== Consul 서비스 디스커버리 ===\n")

    agent = ConsulAgent("agent-1", datacenter="us-east-1", is_server=True)

    # 서비스 등록
    services = [
        ServiceInstance("web", "web-1", "10.0.1.1", 8080,
                       metadata={"tags": ["v2", "primary"]}, zone="us-east-1a"),
        ServiceInstance("web", "web-2", "10.0.1.2", 8080,
                       metadata={"tags": ["v2"]}, zone="us-east-1b"),
        ServiceInstance("api", "api-1", "10.0.2.1", 9090,
                       metadata={"tags": ["v1"]}, zone="us-east-1a"),
        ServiceInstance("api", "api-2", "10.0.2.2", 9090,
                       metadata={"tags": ["v2", "canary"]}, zone="us-east-1b"),
    ]

    for svc in services:
        agent.register_service(svc, HealthCheck(check_type=HealthCheckType.HTTP))

    # DNS 스타일 디스커버리
    print("DNS 질의: web.service.consul")
    results = agent.dns_query("web.service.consul")
    for r in results:
        print(f"  {r['host']}:{r['port']}")

    # API 스타일 디스커버리
    print("\nAPI 질의: api 서비스 (건강한 것만)")
    instances = agent.discover("api")
    for i in instances:
        print(f"  {i.instance_id}: {i.host}:{i.port} tags={i.metadata.get('tags')}")

    # 하나를 비건강으로 표시
    agent.health_checks["api-1"] = False
    print("\napi-1 헬스 체크 실패 후:")
    instances = agent.discover("api", healthy_only=True)
    for i in instances:
        print(f"  {i.instance_id}: {i.host}:{i.port}")

    # 설정을 위한 KV 스토어
    agent.kv_put("config/api/rate_limit", "1000")
    agent.kv_put("config/api/timeout_ms", "5000")
    agent.kv_put("config/web/cache_ttl", "300")

    print(f"\nKV 스토어 (config/ 접두사):")
    for key in agent.kv_list("config/"):
        val = agent.kv_get(key)
        print(f"  {key} = {val['value']}")


demonstrate_consul()
```

---

## 8. 서비스 메시 통합

### 8.1 사이드카 기반 디스커버리

```python
class SidecarProxy:
    """
    시뮬레이션된 서비스 메시 사이드카 프록시.

    사이드카는 모든 인바운드 및 아웃바운드 트래픽을 가로채며,
    서비스 디스커버리, 로드 밸런싱, mTLS, 재시도,
    관찰 가능성(observability)을 애플리케이션에 투명하게 처리한다.
    """

    def __init__(self, service_name: str, instance_id: str, registry: ServiceRegistry):
        self.service_name = service_name
        self.instance_id = instance_id
        self.registry = registry
        self.outbound_lb: Dict[str, LoadBalancer] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.request_count: int = 0
        self.retry_count: int = 0

    def resolve(self, target_service: str) -> Optional[ServiceInstance]:
        """대상 서비스를 인스턴스로 해석한다."""
        if target_service not in self.outbound_lb:
            self.outbound_lb[target_service] = LoadBalancer("power_of_two")

        instances = self.registry.discover(target_service)
        self.outbound_lb[target_service].update_instances(instances)
        return self.outbound_lb[target_service].select()

    def call(self, target_service: str, request: dict,
             max_retries: int = 3) -> dict:
        """
        자동 재시도와 서킷 브레이킹이 포함된 서비스 호출.
        """
        self.request_count += 1

        # 서킷 브레이커 확인
        if target_service not in self.circuit_breakers:
            self.circuit_breakers[target_service] = CircuitBreaker()

        cb = self.circuit_breakers[target_service]
        if not cb.can_execute():
            return {"error": "circuit_breaker_open", "service": target_service}

        # 재시도와 함께 해석 및 호출
        for attempt in range(max_retries + 1):
            instance = self.resolve(target_service)
            if not instance:
                return {"error": "no_instances", "service": target_service}

            # 호출 시뮬레이션 (80% 성공률)
            success = random.random() < 0.8
            if success:
                cb.record_success()
                return {
                    "ok": True,
                    "instance": instance.instance_id,
                    "attempts": attempt + 1,
                }
            else:
                cb.record_failure()
                if attempt < max_retries:
                    self.retry_count += 1

        return {"error": "all_retries_failed", "attempts": max_retries + 1}


def demonstrate_service_mesh():
    """서비스 메시 사이드카 디스커버리를 시연한다."""
    print("=== 서비스 메시 사이드카 ===\n")

    registry = ServiceRegistry()

    # 백엔드 서비스 등록
    for i in range(3):
        registry.register(ServiceInstance(
            service_name="payment-service",
            instance_id=f"payment-{i}",
            host=f"10.0.3.{i+1}",
            port=8080,
        ))

    # order-service를 위한 사이드카 생성
    sidecar = SidecarProxy("order-service", "order-0", registry)

    # payment-service에 20번 호출
    results = {"ok": 0, "retry": 0, "fail": 0}
    for i in range(20):
        result = sidecar.call("payment-service", {"order_id": i})
        if result.get("ok"):
            results["ok"] += 1
        else:
            results["fail"] += 1

    print(f"결과: {results}")
    print(f"총 요청: {sidecar.request_count}")
    print(f"총 재시도: {sidecar.retry_count}")
    print(f"서킷 브레이커 상태: {sidecar.circuit_breakers['payment-service'].get_state()}")


demonstrate_service_mesh()
```

---

## 9. 실제 패턴

### 9.1 시스템 비교

```python
def compare_discovery_systems():
    """실제 서비스 디스커버리 시스템을 비교한다."""
    print("=== 서비스 디스커버리 시스템 ===\n")

    systems = [
        {"name": "Consul", "consensus": "Raft", "health": "에이전트 기반",
         "dns": "예", "kv": "예", "mesh": "Connect"},
        {"name": "etcd", "consensus": "Raft", "health": "TTL 리스",
         "dns": "아니오 (CoreDNS)", "kv": "예", "mesh": "아니오"},
        {"name": "ZooKeeper", "consensus": "ZAB", "health": "임시 노드",
         "dns": "아니오", "kv": "예 (znode)", "mesh": "아니오"},
        {"name": "Kubernetes", "consensus": "etcd (Raft)", "health": "프로브",
         "dns": "CoreDNS", "kv": "ConfigMap", "mesh": "Istio/Linkerd"},
        {"name": "Eureka", "consensus": "AP (피어 복제)", "health": "하트비트",
         "dns": "아니오", "kv": "아니오", "mesh": "아니오"},
    ]

    header = f"{'시스템':<12} {'합의':<12} {'헬스':<18} {'DNS':<14} {'KV':<14} {'메시'}"
    print(header)
    print("-" * len(header))
    for s in systems:
        print(f"{s['name']:<12} {s['consensus']:<12} {s['health']:<18} "
              f"{s['dns']:<14} {s['kv']:<14} {s['mesh']}")


compare_discovery_systems()
```

---

## 10. 요약과 핵심 정리

### 서비스 디스커버리 체크리스트

> **서비스 디스커버리 요구사항**
>
> ☐ TTL과 하트비트를 사용한 등록
> ☐ 헬스 체크 (생존 + 준비)
> ☐ 로드 밸런싱 (최소 round-robin + power-of-two)
> ☐ 연쇄 장애 방지를 위한 서킷 브레이커
> ☐ 단순 클라이언트를 위한 DNS 인터페이스
> ☐ 반응형 업데이트를 위한 워치/알림
> ☐ 다중 데이터센터 지원
> ☐ 종료 시 정상적인 등록 해제

### 핵심 원칙

1. **헬스 체크는 타협할 수 없다**: 없으면 클라이언트가 죽은 인스턴스로 라우팅한다.
2. **클라이언트 측 LB가 빠르지만 어렵다**: 서버 측이 더 단순하고, 사이드카가 두 가지의 장점을 결합한다.
3. **Power-of-two-choices는 과소평가된다**: 최소한의 오버헤드로 거의 최적의 균형을 달성한다.
4. **서킷 브레이커가 연쇄 장애를 방지한다**: 실패한 서비스가 호출자를 함께 다운시키면 안 된다.
5. **DNS는 범용적이지만 업데이트가 느리다**: 부트스트래핑에 사용하고; 동적 업데이트에는 레지스트리를 선호한다.

---

## 11. 연습 문제

### 문제 1: TTL 튜닝

서비스 레지스트리가 TTL=30초, 하트비트 간격=10초를 사용한다. 네트워크 지연시간이 1분 동안 15초로 급등하면 어떻게 되는가? 거짓 등록 해제를 피하면서 60초 이내에 실제 장애를 감지하는 TTL 전략을 설계하라.

### 문제 2: 로드 밸런서 비교

2개 인스턴스가 3배 느린 10개 인스턴스에 대한 100,000 요청을 시뮬레이션하라. round-robin, least-connections, power-of-two를 비교하라. 어느 것이 가장 좋은 p99 지연시간을 달성하는가?

### 문제 3: 다중 DC 디스커버리

3개 데이터센터에서 작동하는 서비스 디스커버리 시스템을 설계하라:
- 로컬 우선 라우팅 (같은 DC 선호)
- 로컬 인스턴스가 비건강하면 다른 DC로 페일오버
- DC 간 설정 복제

### 문제 4: 구현 과제

다음을 갖춘 완전한 서비스 디스커버리 시스템을 구축하라:
- HTTP 기반 등록 및 디스커버리 API
- TTL 기반 헬스 체크
- 가중 라운드 로빈 로드 밸런싱
- 변경 알림을 위한 워치 API

### 문제 5: 장애 시나리오

각 시나리오에서 어떤 일이 발생하는지 분석하고 완화 방안을 제안하라:
1. 높은 트래픽 중 레지스트리 리더 선출
2. 두 DC 사이의 스플릿 브레인
3. 서비스의 모든 인스턴스가 동시에 충돌
4. 레지스트리에 접근 불가하지만 서비스는 건강

---

## 12. 참고 문헌

1. HashiCorp (2024). Consul 문서: https://developer.hashicorp.com/consul
2. etcd 문서: https://etcd.io/docs/
3. Hunt, P. et al. (2010). "ZooKeeper: Wait-free Coordination for Internet-scale Systems." *USENIX ATC*.
4. Netflix (2012). "Eureka! Why You Shouldn't Use ZooKeeper for Service Discovery."
5. Burns, B. (2018). "Designing Distributed Systems." O'Reilly Media.
6. Mielikainen, T. (2019). "The Power of Two Random Choices." (서베이 논문)
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 8. O'Reilly Media.

---

[다음: 레슨 23 — 분산 속도 제한](./23_Distributed_Rate_Limiting.md)
