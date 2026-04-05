# 레슨 23: 분산 속도 제한 (Distributed Rate Limiting)

[개요](./00_Overview.md) | [이전: 서비스 디스커버리](./22_Service_Discovery.md) | [다음: 이벤트 소싱과 CQRS](./24_Event_Sourcing_CQRS.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 단일 노드 및 분산 환경을 위한 토큰 버킷(token bucket)과 슬라이딩 윈도우(sliding window) 속도 제한기 구현
2. Redis 기반 원자적 연산(atomic operation)을 사용한 분산 카운터 구축
3. 구성 가능한 정책(사용자별, API별, 전역)의 속도 제한(rate limiting) 설계
4. 분산 속도 제한에서 클럭 스큐(clock skew)와 네트워크 파티션(network partition) 엣지 케이스 처리
5. 정확도, 지연시간, 일관성 간의 트레이드오프 분석

---

## 목차

1. [속도 제한 기초](#1-속도-제한-기초)
2. [토큰 버킷 알고리즘](#2-토큰-버킷-알고리즘)
3. [슬라이딩 윈도우 알고리즘](#3-슬라이딩-윈도우-알고리즘)
4. [분산 속도 제한 과제](#4-분산-속도-제한-과제)
5. [Redis 기반 구현](#5-redis-기반-구현)
6. [분산 카운터](#6-분산-카운터)
7. [정책 구성](#7-정책-구성)
8. [엣지 케이스와 장애 모드](#8-엣지-케이스와-장애-모드)
9. [프로덕션 패턴](#9-프로덕션-패턴)
10. [요약 및 핵심 정리](#10-요약-및-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 속도 제한 기초

### 1.1 왜 속도 제한을 하는가?

속도 제한은 서비스를 과부하로부터 보호하고, 남용을 방지하며, 공정한 사용 정책을 시행한다. 분산 시스템에서 속도 제한은 특히 어려운데, 요청이 서로 다른 노드에 도착하고 카운터를 위한 공유 메모리가 없기 때문이다.

```
속도 제한 없이:                     속도 제한 사용:
  클라이언트 → [1000 req/s] → 서버   클라이언트 → [1000 req/s] → 속도 제한기 → [100 req/s] → 서버
  서버 과부하, 충돌                  서버가 용량 내에서 처리
```

### 1.2 알고리즘 개요

| 알고리즘 | 정확도 | 메모리 | 버스트 처리 | 복잡도 |
|---------|--------|--------|-----------|--------|
| 토큰 버킷 (Token Bucket) | 높음 | O(1) | 제어된 버스트 허용 | 낮음 |
| 리키 버킷 (Leaky Bucket) | 높음 | O(1) | 고정 속도로 평탄화 | 낮음 |
| 고정 윈도우 (Fixed Window) | 낮음 (경계 버스트) | O(1) | 경계에서 2x 버스트 | 최저 |
| 슬라이딩 윈도우 로그 (Sliding Window Log) | 정확 | O(N) | 버스트 없음 | 높음 |
| 슬라이딩 윈도우 카운터 (Sliding Window Counter) | 근사 | O(1) | 최소 버스트 | 낮음 |

---

## 2. 토큰 버킷 알고리즘

### 2.1 구현

```python
import time
import random
import threading
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


@dataclass
class TokenBucket:
    """
    토큰 버킷(token bucket) 속도 제한기.

    토큰이 고정 속도로 추가된다. 각 요청은 하나의 토큰을 소비한다.
    사용 가능한 토큰이 없으면 요청이 거부된다.

    버킷은 최대 용량을 가지며, 용량 크기까지
    제어된 버스트를 허용한다.
    """
    rate: float           # 초당 추가되는 토큰 수
    capacity: float       # 최대 토큰 수 (버스트 크기)
    tokens: float = 0.0   # 현재 토큰 수
    last_refill: float = field(default_factory=time.time)
    total_allowed: int = 0
    total_rejected: int = 0

    def __post_init__(self):
        self.tokens = self.capacity  # 가득 찬 상태로 시작

    def _refill(self):
        """경과 시간에 따라 토큰을 보충한다."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def allow(self, tokens: float = 1.0) -> bool:
        """
        요청이 허용되는지 확인한다.

        허용되면 True를 반환하고 토큰을 소비한다.
        토큰이 부족하면 False를 반환한다.
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            self.total_allowed += 1
            return True
        else:
            self.total_rejected += 1
            return False

    def wait_time(self, tokens: float = 1.0) -> float:
        """충분한 토큰을 위해 대기해야 하는 시간을 계산한다."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        deficit = tokens - self.tokens
        return deficit / self.rate

    def stats(self) -> dict:
        self._refill()
        return {
            "tokens": round(self.tokens, 2),
            "capacity": self.capacity,
            "rate": self.rate,
            "allowed": self.total_allowed,
            "rejected": self.total_rejected,
            "utilization": round(
                self.total_allowed / max(1, self.total_allowed + self.total_rejected) * 100, 1
            ),
        }


def demonstrate_token_bucket():
    """토큰 버킷 알고리즘을 시연한다."""
    print("=== Token Bucket Rate Limiter ===\n")

    bucket = TokenBucket(rate=10.0, capacity=20.0)  # 10 req/s, 20 버스트

    # 1단계: 25개 요청 버스트
    print("Phase 1: Burst of 25 requests")
    allowed = sum(1 for _ in range(25) if bucket.allow())
    print(f"  Allowed: {allowed}/25 (bucket capacity = 20)")

    # 2단계: 대기 후 재시도
    time.sleep(0.5)  # 0.5초에 5개 토큰 보충
    print(f"\nPhase 2: After 500ms wait")
    allowed = sum(1 for _ in range(10) if bucket.allow())
    print(f"  Allowed: {allowed}/10 (5 tokens refilled)")

    # 3단계: 정상 상태
    print(f"\nPhase 3: Steady state (10 req/s for 2 seconds)")
    for second in range(2):
        time.sleep(0.1)
        allowed = sum(1 for _ in range(15) if bucket.allow())
        print(f"  Second {second + 1}: Allowed {allowed}/15 requests")

    print(f"\nStats: {bucket.stats()}")


demonstrate_token_bucket()
```

---

## 3. 슬라이딩 윈도우 알고리즘

### 3.1 고정 윈도우 카운터 (Fixed Window Counter)

```python
class FixedWindowCounter:
    """
    고정 윈도우(fixed window) 속도 제한기.

    시간을 고정 윈도우(예: 1초 간격)로 나눈다.
    윈도우당 요청 수를 세고 제한에 도달하면 거부한다.

    문제: 두 윈도우 경계에서의 요청 버스트가
    속도 제한의 최대 2배를 허용할 수 있다.
    """

    def __init__(self, limit: int, window_size: float = 1.0):
        self.limit = limit
        self.window_size = window_size
        self.current_window: int = 0
        self.count: int = 0

    def _current_window_id(self) -> int:
        return int(time.time() / self.window_size)

    def allow(self) -> bool:
        window = self._current_window_id()
        if window != self.current_window:
            self.current_window = window
            self.count = 0

        if self.count < self.limit:
            self.count += 1
            return True
        return False


class SlidingWindowLog:
    """
    슬라이딩 윈도우 로그(sliding window log) 속도 제한기.

    윈도우 내 모든 요청 타임스탬프의 로그를 유지한다.
    정확한 속도 제한을 제공하지만 클라이언트당 O(N) 메모리를 사용한다.
    """

    def __init__(self, limit: int, window_size: float = 1.0):
        self.limit = limit
        self.window_size = window_size
        self.timestamps: list[float] = []

    def allow(self) -> bool:
        now = time.time()
        cutoff = now - self.window_size

        # 만료된 항목 제거
        self.timestamps = [t for t in self.timestamps if t > cutoff]

        if len(self.timestamps) < self.limit:
            self.timestamps.append(now)
            return True
        return False


class SlidingWindowCounter:
    """
    슬라이딩 윈도우 카운터(sliding window counter) 속도 제한기.

    인접한 두 고정 윈도우를 결합하는 근사법이다.
    현재 윈도우에 얼마나 진행했는지에 따라
    가중 카운팅을 사용한다.

    메모리: O(1) — 두 개의 카운터만 저장한다.
    정확도: 근사적이지만 경계에서 고정 윈도우보다 훨씬 좋다.
    """

    def __init__(self, limit: int, window_size: float = 1.0):
        self.limit = limit
        self.window_size = window_size
        self.prev_count: int = 0
        self.curr_count: int = 0
        self.prev_window: int = 0
        self.curr_window: int = 0

    def _current_window_id(self) -> int:
        return int(time.time() / self.window_size)

    def _window_progress(self) -> float:
        """현재 윈도우에 얼마나 진행했는지 (0.0에서 1.0)."""
        return (time.time() % self.window_size) / self.window_size

    def allow(self) -> bool:
        window = self._current_window_id()

        if window != self.curr_window:
            if window == self.curr_window + 1:
                self.prev_count = self.curr_count
                self.prev_window = self.curr_window
            else:
                self.prev_count = 0
            self.curr_count = 0
            self.curr_window = window

        # 가중 추정: prev_count * (1 - progress) + curr_count
        progress = self._window_progress()
        estimated = self.prev_count * (1.0 - progress) + self.curr_count

        if estimated < self.limit:
            self.curr_count += 1
            return True
        return False


def compare_window_algorithms():
    """고정 윈도우, 슬라이딩 로그, 슬라이딩 카운터를 비교한다."""
    print("=== Window Algorithm Comparison ===\n")

    limit = 10  # 초당 10개 요청

    for name, limiter in [
        ("Fixed Window", FixedWindowCounter(limit)),
        ("Sliding Log", SlidingWindowLog(limit)),
        ("Sliding Counter", SlidingWindowCounter(limit)),
    ]:
        allowed = 0
        rejected = 0

        # 한 윈도우 끝에 5개 요청, 다음 윈도우 시작에 5개 요청 전송
        for _ in range(15):
            if limiter.allow():
                allowed += 1
            else:
                rejected += 1

        print(f"  {name:20s}: allowed={allowed}, rejected={rejected}")


compare_window_algorithms()
```

---

## 4. 분산 속도 제한 과제

### 4.1 다중 노드 문제

```python
def illustrate_distributed_challenge():
    """분산 속도 제한이 어려운 이유를 설명한다."""
    print("=== Distributed Rate Limiting Challenges ===\n")

    print("Scenario: 100 req/s limit, 5 API servers\n")

    approaches = {
        "Local only (no coordination)": {
            "strategy": "Each server limits to 100/5 = 20 req/s",
            "problem": "Uneven traffic → some servers waste capacity",
            "effective_limit": "20-100 req/s depending on distribution",
        },
        "Central counter (Redis)": {
            "strategy": "All servers check/increment a shared counter",
            "problem": "Redis latency added to every request",
            "effective_limit": "~100 req/s (accurate)",
        },
        "Leaky bucket with sync": {
            "strategy": "Local buckets with periodic sync",
            "problem": "Brief over-limit between syncs",
            "effective_limit": "100-120 req/s (slightly over)",
        },
        "Token bucket with prefetch": {
            "strategy": "Prefetch tokens from central store",
            "problem": "Wasted tokens if traffic shifts",
            "effective_limit": "90-110 req/s (close)",
        },
    }

    for name, info in approaches.items():
        print(f"  {name}:")
        print(f"    Strategy: {info['strategy']}")
        print(f"    Problem:  {info['problem']}")
        print(f"    Effective limit: {info['effective_limit']}")
        print()


illustrate_distributed_challenge()
```

### 4.2 분산 토큰 버킷

```python
class DistributedTokenBucket:
    """
    중앙 조율이 포함된 분산 토큰 버킷(distributed token bucket).

    각 노드가 로컬 버킷을 유지하고 주기적으로
    중앙 조율기(예: Redis)와 동기화하여 토큰을 보충한다.

    이를 통해 요청별 네트워크 호출을 피하면서
    근사적 전역 속도 제한을 유지한다.
    """

    def __init__(self, node_id: str, global_rate: float, global_capacity: float,
                 num_nodes: int, sync_interval: float = 0.5):
        self.node_id = node_id
        self.global_rate = global_rate
        self.global_capacity = global_capacity
        self.num_nodes = num_nodes
        self.sync_interval = sync_interval

        # 로컬 버킷: 비례 분배
        self.local_rate = global_rate / num_nodes
        self.local_capacity = global_capacity / num_nodes
        self.local_bucket = TokenBucket(
            rate=self.local_rate,
            capacity=self.local_capacity,
        )

        # 동기화 상태
        self.last_sync = time.time()
        self.tokens_borrowed: float = 0
        self.tokens_returned: float = 0

    def allow(self) -> bool:
        """로컬 버킷을 사용하여 요청이 허용되는지 확인한다."""
        return self.local_bucket.allow()

    def sync_with_coordinator(self, coordinator: 'RateLimitCoordinator'):
        """
        중앙 조율기와 로컬 버킷을 동기화한다.

        - 사용하지 않은 토큰 보고 (풀에 반환)
        - 필요시 추가 토큰 요청
        """
        now = time.time()
        if now - self.last_sync < self.sync_interval:
            return

        self.last_sync = now

        # 미사용 용량 계산
        unused = self.local_bucket.tokens
        used_pct = 1.0 - (unused / max(0.01, self.local_capacity))

        # 과소 활용 시 토큰 반환
        if used_pct < 0.5:
            return_amount = unused * 0.3
            self.local_bucket.tokens -= return_amount
            coordinator.return_tokens(self.node_id, return_amount)
            self.tokens_returned += return_amount

        # 과부하 시 추가 요청
        elif used_pct > 0.9:
            request_amount = self.local_capacity * 0.5
            granted = coordinator.request_tokens(self.node_id, request_amount)
            self.local_bucket.tokens += granted
            self.tokens_borrowed += granted

    def stats(self) -> dict:
        return {
            "node": self.node_id,
            "local_tokens": round(self.local_bucket.tokens, 2),
            "local_capacity": self.local_capacity,
            "borrowed": round(self.tokens_borrowed, 2),
            "returned": round(self.tokens_returned, 2),
            **self.local_bucket.stats(),
        }


class RateLimitCoordinator:
    """
    중앙 속도 제한 조율기(프로덕션에서는 Redis로 지원됨).

    노드가 빌리고 반환할 수 있는 전역 토큰 풀을 관리한다.
    """

    def __init__(self, global_rate: float, global_capacity: float):
        self.global_rate = global_rate
        self.global_capacity = global_capacity
        self.pool: float = global_capacity * 0.2  # 풀에 20% 예비
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        # 전역 속도의 일부를 풀에 추가
        self.pool = min(
            self.global_capacity * 0.5,  # 최대 50% 예비
            self.pool + elapsed * self.global_rate * 0.2,
        )
        self.last_refill = now

    def request_tokens(self, node_id: str, amount: float) -> float:
        """중앙 풀에서 토큰을 부여한다."""
        with self.lock:
            self._refill()
            granted = min(amount, self.pool)
            self.pool -= granted
            return granted

    def return_tokens(self, node_id: str, amount: float):
        """반환된 토큰을 풀에 수용한다."""
        with self.lock:
            self.pool = min(self.global_capacity * 0.5, self.pool + amount)


def demonstrate_distributed_rate_limiting():
    """조율이 포함된 분산 속도 제한을 시연한다."""
    print("=== Distributed Rate Limiting ===\n")

    global_rate = 100.0  # 전역 100 req/s
    global_capacity = 200.0
    num_nodes = 5

    coordinator = RateLimitCoordinator(global_rate, global_capacity)
    nodes = [
        DistributedTokenBucket(f"node-{i}", global_rate, global_capacity, num_nodes)
        for i in range(num_nodes)
    ]

    # 트래픽 시뮬레이션 (불균등 분배)
    traffic_weights = [0.4, 0.25, 0.15, 0.1, 0.1]  # 노드 0이 트래픽의 40% 수신
    total_allowed = 0
    total_rejected = 0

    for _ in range(10):  # 각 0.1초씩 10라운드
        for i, node in enumerate(nodes):
            # 조율기와 동기화
            node.sync_with_coordinator(coordinator)

            # 트래픽 가중치에 비례하여 요청 처리
            num_requests = int(15 * traffic_weights[i])  # 100/s에서 ~15 req/0.1s
            for _ in range(num_requests):
                if node.allow():
                    total_allowed += 1
                else:
                    total_rejected += 1

        time.sleep(0.05)

    print(f"Global limit: {global_rate} req/s")
    print(f"Total allowed: {total_allowed}")
    print(f"Total rejected: {total_rejected}")
    print(f"\nPer-node stats:")
    for node in nodes:
        s = node.stats()
        print(f"  {s['node']}: allowed={s['allowed']}, rejected={s['rejected']}, "
              f"tokens={s['local_tokens']}")


demonstrate_distributed_rate_limiting()
```

---

## 5. Redis 기반 구현

### 5.1 Lua 스크립트를 사용한 원자적 속도 제한

```python
class RedisRateLimiter:
    """
    시뮬레이션된 Redis 기반 분산 속도 제한기.

    프로덕션에서는 핵심 로직이 원자성을 위해 Redis Lua 스크립트로
    실행된다. 이 시뮬레이션은 알고리즘을 시연한다.
    """

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def _execute_lua(self, script_name: str, keys: list, args: list) -> Any:
        """원자적 Redis Lua 스크립트 실행을 시뮬레이션한다."""
        with self.lock:
            if script_name == "token_bucket":
                return self._lua_token_bucket(keys, args)
            elif script_name == "sliding_window":
                return self._lua_sliding_window(keys, args)

    def _lua_token_bucket(self, keys: list, args: list) -> Tuple[bool, float]:
        """
        Redis Lua 스크립트로 구현된 토큰 버킷.

        KEYS[1] = 속도 제한 키
        ARGV[1] = 속도 (토큰/초)
        ARGV[2] = 용량
        ARGV[3] = 현재 시간 (타임스탬프)
        ARGV[4] = 요청 토큰 수

        Redis에서 원자적으로 실행되어 여러 API 서버 간의
        경쟁 조건(race condition)을 방지한다.
        """
        key = keys[0]
        rate = float(args[0])
        capacity = float(args[1])
        now = float(args[2])
        requested = float(args[3])

        # 현재 상태 가져오기
        state = self.store.get(key, {"tokens": capacity, "last_refill": now})

        # 보충
        elapsed = now - state["last_refill"]
        tokens = min(capacity, state["tokens"] + elapsed * rate)

        # 확인
        allowed = tokens >= requested
        if allowed:
            tokens -= requested

        # 저장
        self.store[key] = {"tokens": tokens, "last_refill": now}
        return allowed, tokens

    def _lua_sliding_window(self, keys: list, args: list) -> Tuple[bool, int]:
        """
        Redis Lua 스크립트로 구현된 슬라이딩 윈도우 카운터.

        두 개의 정렬된 집합(현재 및 이전 윈도우)을 사용하여
        가중 카운팅을 수행한다.
        """
        key = keys[0]
        limit = int(args[0])
        window_size = float(args[1])
        now = float(args[2])

        window_id = int(now / window_size)
        progress = (now % window_size) / window_size

        curr_key = f"{key}:{window_id}"
        prev_key = f"{key}:{window_id - 1}"

        curr_count = self.store.get(curr_key, 0)
        prev_count = self.store.get(prev_key, 0)

        estimated = prev_count * (1.0 - progress) + curr_count

        if estimated < limit:
            self.store[curr_key] = curr_count + 1
            return True, int(estimated + 1)
        return False, int(estimated)

    def check_rate_limit(self, client_id: str, rate: float = 10.0,
                         capacity: float = 20.0) -> dict:
        """토큰 버킷을 사용하여 클라이언트의 속도 제한을 확인한다."""
        now = time.time()
        key = f"ratelimit:token:{client_id}"
        allowed, tokens = self._execute_lua(
            "token_bucket", [key], [rate, capacity, now, 1.0]
        )
        return {
            "allowed": allowed,
            "remaining_tokens": round(tokens, 2),
            "limit": rate,
        }

    def check_sliding_window(self, client_id: str, limit: int = 100,
                              window: float = 60.0) -> dict:
        """슬라이딩 윈도우 카운터를 사용하여 속도 제한을 확인한다."""
        now = time.time()
        key = f"ratelimit:window:{client_id}"
        allowed, count = self._execute_lua(
            "sliding_window", [key], [limit, window, now]
        )
        return {
            "allowed": allowed,
            "current_count": count,
            "limit": limit,
            "window_seconds": window,
        }


def demonstrate_redis_rate_limiter():
    """Redis 기반 속도 제한을 시연한다."""
    print("=== Redis-Based Rate Limiting ===\n")

    limiter = RedisRateLimiter()

    # 토큰 버킷: 5 req/s, 10 버스트
    print("Token Bucket (5 req/s, burst=10):")
    for i in range(15):
        result = limiter.check_rate_limit("user-123", rate=5.0, capacity=10.0)
        status = "ALLOW" if result["allowed"] else "DENY "
        print(f"  Request {i+1:2d}: {status} (remaining={result['remaining_tokens']})")

    # 슬라이딩 윈도우: 1초당 10개 요청
    print(f"\nSliding Window (10 req/1s):")
    for i in range(15):
        result = limiter.check_sliding_window("user-456", limit=10, window=1.0)
        status = "ALLOW" if result["allowed"] else "DENY "
        print(f"  Request {i+1:2d}: {status} (count={result['current_count']}/{result['limit']})")


demonstrate_redis_rate_limiter()
```

---

## 6. 분산 카운터

### 6.1 근사적 제한을 위한 CRDT 기반 카운터

```python
class CRDTCounter:
    """
    근사적 속도 제한을 위한 CRDT 기반 분산 카운터.

    각 노드가 로컬 카운터를 유지한다. 카운터는
    max()를 사용하여 병합된다 (PN-Counter 접근법).
    이는 조율 없이 최종 일관성(eventual consistency)을 제공한다.
    """

    def __init__(self, node_id: str, num_nodes: int):
        self.node_id = node_id
        self.num_nodes = num_nodes
        # 노드별 양의 카운터
        self.increments: Dict[str, int] = defaultdict(int)
        self.increments[node_id] = 0

    def increment(self):
        """로컬 카운터를 증가시킨다."""
        self.increments[self.node_id] += 1

    def value(self) -> int:
        """현재 카운터 값을 가져온다."""
        return sum(self.increments.values())

    def merge(self, other: 'CRDTCounter'):
        """다른 카운터와 병합한다 (각 노드 카운트의 max 취함)."""
        for node_id, count in other.increments.items():
            self.increments[node_id] = max(self.increments[node_id], count)


def demonstrate_crdt_counter():
    """속도 제한을 위한 CRDT 기반 분산 카운터를 시연한다."""
    print("=== CRDT Counter for Rate Limiting ===\n")

    # 3개 노드, 각각 독립적으로 카운팅
    counters = {
        f"node-{i}": CRDTCounter(f"node-{i}", 3)
        for i in range(3)
    }

    # 각 노드가 일부 요청을 수신
    for _ in range(10):
        counters["node-0"].increment()
    for _ in range(7):
        counters["node-1"].increment()
    for _ in range(3):
        counters["node-2"].increment()

    print("Before merge:")
    for nid, c in counters.items():
        print(f"  {nid}: local={c.increments[nid]}, total={c.value()}")

    # 병합 (가십 라운드)
    for nid1 in counters:
        for nid2 in counters:
            if nid1 != nid2:
                counters[nid1].merge(counters[nid2])

    print("\nAfter merge:")
    for nid, c in counters.items():
        print(f"  {nid}: total={c.value()}")

    limit = 25
    print(f"\nGlobal limit: {limit}")
    print(f"Global count: {counters['node-0'].value()}")
    print(f"Over limit: {counters['node-0'].value() > limit}")


demonstrate_crdt_counter()
```

---

## 7. 정책 구성

### 7.1 다층 속도 제한 (Multi-Tier Rate Limiting)

```python
@dataclass
class RateLimitPolicy:
    """속도 제한 정책 구성."""
    name: str
    limit: int
    window_seconds: float
    scope: str  # "global", "per_user", "per_ip", "per_api_key"
    algorithm: str  # "token_bucket", "sliding_window", "fixed_window"
    burst_multiplier: float = 1.5  # limit * multiplier까지 버스트 허용
    retry_after_seconds: float = 1.0


class MultiTierRateLimiter:
    """
    구성 가능한 정책을 사용하는 다층 속도 제한기.

    여러 속도 제한을 동시에 적용하며(예: 초당과 분당),
    가장 제한적인 것이 승리한다.
    """

    def __init__(self):
        self.policies: Dict[str, list[RateLimitPolicy]] = defaultdict(list)
        self.buckets: Dict[str, TokenBucket] = {}

    def add_policy(self, scope_value: str, policy: RateLimitPolicy):
        """범위 값에 대한 속도 제한 정책을 추가한다."""
        self.policies[scope_value].append(policy)
        key = f"{scope_value}:{policy.name}"
        self.buckets[key] = TokenBucket(
            rate=policy.limit / policy.window_seconds,
            capacity=policy.limit * policy.burst_multiplier,
        )

    def check(self, scope_value: str) -> dict:
        """
        모든 적용 가능한 속도 제한을 확인한다.

        가장 제한적인 결과를 반환한다.
        """
        policies = self.policies.get(scope_value, [])
        if not policies:
            return {"allowed": True, "policy": None}

        for policy in policies:
            key = f"{scope_value}:{policy.name}"
            bucket = self.buckets.get(key)
            if bucket and not bucket.allow():
                return {
                    "allowed": False,
                    "policy": policy.name,
                    "retry_after": policy.retry_after_seconds,
                    "limit": policy.limit,
                    "window": policy.window_seconds,
                }

        return {"allowed": True, "policy": None}


def demonstrate_multi_tier():
    """다층 속도 제한을 시연한다."""
    print("=== Multi-Tier Rate Limiting ===\n")

    limiter = MultiTierRateLimiter()

    # 사용자 "alice"에게 세 가지 계층 적용
    limiter.add_policy("user:alice", RateLimitPolicy(
        name="per_second", limit=10, window_seconds=1.0, scope="per_user",
        algorithm="token_bucket",
    ))
    limiter.add_policy("user:alice", RateLimitPolicy(
        name="per_minute", limit=100, window_seconds=60.0, scope="per_user",
        algorithm="token_bucket",
    ))
    limiter.add_policy("user:alice", RateLimitPolicy(
        name="per_hour", limit=1000, window_seconds=3600.0, scope="per_user",
        algorithm="token_bucket",
    ))

    print("Policies for user:alice:")
    for p in limiter.policies["user:alice"]:
        print(f"  {p.name}: {p.limit} per {p.window_seconds}s")

    # 20개 요청 버스트
    allowed = 0
    first_reject_policy = None
    for i in range(20):
        result = limiter.check("user:alice")
        if result["allowed"]:
            allowed += 1
        elif first_reject_policy is None:
            first_reject_policy = result["policy"]

    print(f"\nBurst of 20 requests: {allowed} allowed")
    print(f"First rejection by: {first_reject_policy}")


demonstrate_multi_tier()
```

---

## 8. 엣지 케이스와 장애 모드

### 8.1 클럭 스큐 (Clock Skew)

```python
def analyze_clock_skew_impact():
    """분산 속도 제한에 대한 클럭 스큐(clock skew)의 영향을 분석한다."""
    print("=== Clock Skew Impact ===\n")

    # 시나리오: 클럭 스큐가 있는 3개 노드
    node_offsets = {
        "node-0": 0.0,       # 기준 클럭
        "node-1": 0.5,       # 500ms 앞서 있음
        "node-2": -0.3,      # 300ms 뒤처져 있음
    }

    window_size = 1.0
    limit = 10

    print("Node clock offsets:")
    for node, offset in node_offsets.items():
        print(f"  {node}: {offset:+.1f}s")

    # 같은 실제 시점에 각 노드가 다른 윈도우에 있다고 판단
    real_time = 100.5
    print(f"\nReal time: {real_time}")
    for node, offset in node_offsets.items():
        perceived = real_time + offset
        window_id = int(perceived / window_size)
        print(f"  {node}: perceived={perceived}, window={window_id}")

    print(f"\nImpact:")
    print(f"  node-1 and node-2 disagree on window by {0.5 + 0.3:.1f}s")
    print(f"  Requests near window boundaries may be counted in wrong window")
    print(f"  Max effective rate: {limit * 2} (2x burst at boundary with skew)")
    print(f"\nMitigation:")
    print(f"  1. Use NTP with tight synchronization (<10ms)")
    print(f"  2. Use sliding window (reduces boundary effect)")
    print(f"  3. Build clock skew tolerance into limits (set to 0.9 * desired)")


analyze_clock_skew_impact()
```

### 8.2 파티션 내구성 (Partition Tolerance)

```python
def analyze_partition_impact():
    """네트워크 파티션(network partition) 중 속도 제한을 분석한다."""
    print("=== Network Partition Impact ===\n")

    scenarios = [
        {
            "name": "Redis unreachable",
            "strategy": "Local fallback with conservative limit",
            "risk": "Over-limiting (lost capacity) or under-limiting (no coordination)",
            "recommendation": "Use local bucket at 1/N rate as fallback",
        },
        {
            "name": "Partial partition (some nodes can reach Redis)",
            "strategy": "Nodes that can reach Redis rate-limit normally",
            "risk": "Unfair: reachable nodes are rate-limited, others aren't",
            "recommendation": "Track last successful sync; degrade gracefully",
        },
        {
            "name": "Full network partition",
            "strategy": "Each partition operates independently",
            "risk": "Each partition allows full rate → 2x total during partition",
            "recommendation": "Accept over-limit; set alerts for anomalies",
        },
    ]

    for s in scenarios:
        print(f"  {s['name']}:")
        print(f"    Strategy: {s['strategy']}")
        print(f"    Risk: {s['risk']}")
        print(f"    Recommendation: {s['recommendation']}")
        print()


analyze_partition_impact()
```

---

## 9. 프로덕션 패턴

### 9.1 속도 제한 아키텍처

```python
def production_patterns():
    """프로덕션 속도 제한 패턴을 설명한다."""
    print("=== Production Rate Limiting Patterns ===\n")

    patterns = [
        {
            "name": "API Gateway Rate Limiting",
            "where": "Edge (API gateway / load balancer)",
            "why": "Protect backend from external abuse",
            "how": "Redis + sliding window per API key",
            "examples": "Kong, AWS API Gateway, Nginx",
        },
        {
            "name": "Application-Level Rate Limiting",
            "where": "Within the application code",
            "why": "Business-logic-aware limits",
            "how": "Token bucket per user/tenant",
            "examples": "Stripe API, GitHub API",
        },
        {
            "name": "Service Mesh Rate Limiting",
            "where": "Sidecar proxy (Envoy)",
            "why": "Protect internal services from each other",
            "how": "Local token bucket + global rate limit service",
            "examples": "Istio, Lyft Ratelimit",
        },
        {
            "name": "Database Rate Limiting",
            "where": "Database proxy / connection pool",
            "why": "Prevent query overload",
            "how": "Admission control + queue",
            "examples": "PgBouncer, ProxySQL",
        },
    ]

    for p in patterns:
        print(f"  {p['name']}:")
        print(f"    Where: {p['where']}")
        print(f"    Why: {p['why']}")
        print(f"    How: {p['how']}")
        print(f"    Examples: {p['examples']}")
        print()


production_patterns()
```

---

## 10. 요약 및 핵심 정리

### 속도 제한 알고리즘 선택

> **속도 제한 결정 트리 (RATE LIMITING DECISION TREE)**
>
> 정확한 제한이 필요한가?
>   예 → 슬라이딩 윈도우 로그 (O(N) 메모리)
>   아니오 → 버스트 허용이 필요한가?
>     예 → 토큰 버킷 (구성 가능한 버스트)
>     아니오 → 슬라이딩 윈도우 카운터 (근사적, O(1) 메모리)
>
> 분산 환경?
>   단일 노드 → 로컬 알고리즘
>   다중 노드 → Redis Lua 스크립트 또는 동기화가 포함된 분산 토큰 버킷

### 핵심 원칙

1. **토큰 버킷이 기본 선택이다**: 단순하고, 버스트를 지원하며, O(1) 메모리이다.
2. **Redis Lua 스크립트가 원자성을 제공한다**: 여러 API 서버에서의 정확성에 필수적이다.
3. **분산 환경에서 근사적 제한을 수용한다**: 정확한 제한은 모든 요청마다 동기화가 필요하다.
4. **다층 제한이 다른 남용 패턴을 포착한다**: 초당은 버스트를, 시간당은 지속적 남용을 잡는다.
5. **항상 폴백(fallback)을 보유한다**: Redis 다운 시 모든 요청을 거부하지 말고 로컬 속도 제한으로 저하한다.

---

## 11. 연습 문제

### 문제 1: 알고리즘 비교

다섯 가지 속도 제한 알고리즘을 모두 구현하고 초당 10,000개 요청 워크로드에서 벤치마크한다. 메모리 사용량, 정확도(목표 대비 편차 %), CPU 오버헤드를 비교한다.

### 문제 2: 분산 도전

10개 노드 클러스터에서 1000 req/s 전역 제한을 가진 속도 제한기를 설계한다. 속도 제한기는 Redis 장애 시 로컬 제한으로 폴백해야 한다. 30초 Redis 장애 중 최대 초과 제한을 계산한다.

### 문제 3: 공정성

다중 테넌트(multi-tenant) API가 다른 속도 제한(10-10000 req/s)을 가진 100개 테넌트에 서비스한다. 높은 속도의 테넌트가 낮은 속도의 테넌트를 기아(starvation)시키지 않는 공정한 큐잉 시스템을 설계한다.

### 문제 4: 구현 도전

다음을 포함하는 완전한 분산 속도 제한기를 구축한다:
- Redis 백엔드가 있는 토큰 버킷 알고리즘
- 분당 제한을 위한 슬라이딩 윈도우
- Redis 장애 시 로컬 제한으로 자동 폴백
- HTTP 헤더: X-RateLimit-Limit, X-RateLimit-Remaining, Retry-After

### 문제 5: 비용 분석

클라우드 속도 제한기가 중앙 카운팅에 Redis를 사용한다. 각 속도 제한 확인은 1회 Redis 호출(0.5ms 지연시간)이 소요된다. 100,000 req/s에서 다음을 계산한다: 총 Redis 연산, 필요한 Redis 클러스터 크기, 추가된 p50/p99 지연시간. Redis 호출을 80% 줄이는 최적화를 제안한다.

---

## 12. 참고 문헌

1. Stripe Engineering (2017). "Rate Limiters and Load Shedders." Stripe Blog.
2. Cloudflare (2017). "How We Built Rate Limiting Capable of Scaling to Millions of Domains."
3. Redis documentation: "Rate Limiting with Redis."
4. Lyft Engineering (2017). "Ratelimit: A Generic Rate Limit Service."
5. Veeraraghavan, K. et al. (2016). "Maelstrom: Mitigating Datacenter-level Disasters." *OSDI*.
6. Google (2022). "Rate Limiting Strategies and Techniques." Cloud Architecture Center.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 4. O'Reilly Media.

---

[다음: 레슨 24 — 이벤트 소싱과 CQRS](./24_Event_Sourcing_CQRS.md)
