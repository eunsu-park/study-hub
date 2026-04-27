# 20. Redis 캐싱 패턴

**이전**: [Go 웹 기초](./19_Go_Web_Basics.md) | **다음**: [작업 큐](./21_Job_Queues.md)

**난이도**: ⭐⭐⭐

## 학습 목표

- Redis 데이터 타입과 백엔드 시스템에서의 사용 사례를 이해한다
- 읽기 중심 워크로드를 위한 캐시 어사이드(cache-aside, lazy loading) 패턴을 구현한다
- 쓰기 일관성을 위한 라이트스루(write-through)와 라이트비하인드(write-behind) 패턴을 적용한다
- TTL과 이벤트 기반 접근법을 사용하여 효과적인 캐시 무효화 전략을 설계한다
- 무상태 애플리케이션 서버를 위한 세션 저장소로 Redis를 사용한다
- Redis 원자적 연산으로 속도 제한기(rate limiter)를 구축한다
- 실시간 통신을 위한 Pub/Sub 메시징을 구현한다
- 이벤트 소싱과 메시지 처리를 위한 Redis Streams를 사용한다
- FastAPI, Express, Django 애플리케이션에 Redis를 통합한다

## 목차

1. [Redis 기초](#1-redis-기초)
2. [캐시 어사이드 패턴](#2-캐시-어사이드-패턴)
3. [라이트스루와 라이트비하인드](#3-라이트스루와-라이트비하인드)
4. [캐시 무효화 전략](#4-캐시-무효화-전략)
5. [세션 저장소로서의 Redis](#5-세션-저장소로서의-redis)
6. [Redis를 이용한 속도 제한](#6-redis를-이용한-속도-제한)
7. [Pub/Sub 메시징](#7-pubsub-메시징)
8. [Redis Streams](#8-redis-streams)
9. [프레임워크 통합](#9-프레임워크-통합)
10. [연습 문제](#10-연습-문제)

---

## 1. Redis 기초

Redis(Remote Dictionary Server)는 데이터베이스, 캐시, 메시지 브로커, 스트리밍 엔진으로 사용되는 인메모리 데이터 구조 저장소이다. 밀리초 이하의 응답 시간을 지원하며 초당 수백만 건의 요청을 처리할 수 있다.

### 핵심 데이터 타입

| 타입 | 설명 | 주요 사용 사례 |
|---|---|---|
| String | 바이너리 안전 문자열 (최대 512 MB) | 캐시 값, 카운터, 세션 토큰 |
| Hash | 필드-값 쌍 | 사용자 프로필, 객체 저장 |
| List | 순서 있는 컬렉션 (연결 리스트) | 메시지 큐, 활동 피드 |
| Set | 순서 없는 고유 요소 | 태그, 고유 방문자, 집합 연산 |
| Sorted Set | 점수가 있는 고유 요소 | 리더보드, 속도 제한, 시계열 |
| Stream | 추가 전용 로그 | 이벤트 소싱, 메시지 큐 |

### 데이터 타입 빠른 참조(Data Type Quick Reference)

| 데이터 타입(Data Type) | 사용 사례(Use Case) | 예시 명령어(Example Command) | 시간 복잡도(Time Complexity) |
|----------------------|-------------------|---------------------------|--------------------------|
| String | 캐시, 카운터 | GET/SET/INCR | O(1) |
| Hash | 객체 저장 | HGET/HSET/HGETALL | O(1) per field |
| List | 큐, 피드 | LPUSH/RPOP/LRANGE | O(1) push/pop |
| Set | 태그, 고유 항목 | SADD/SMEMBERS/SINTER | O(1) add |
| Sorted Set | 리더보드, 스케줄링 | ZADD/ZRANGE/ZRANK | O(log N) |
| Stream | 이벤트 로그, 메시징 | XADD/XREAD/XGROUP | O(1) add |

### 필수 명령어

```bash
# Strings
SET user:1:name "Alice"              # 키 설정
GET user:1:name                       # 키 조회
SETEX session:abc 3600 "user_data"   # TTL과 함께 설정 (초 단위)
INCR page:views                       # 원자적 증가
MSET k1 "v1" k2 "v2"                # 여러 키 설정
MGET k1 k2                           # 여러 키 조회

# Hashes
HSET user:1 name "Alice" email "alice@example.com" age "30"
HGET user:1 name
HGETALL user:1
HINCRBY user:1 age 1

# Lists
LPUSH queue:tasks "task1" "task2"     # 헤드에 삽입
RPOP queue:tasks                      # 테일에서 꺼내기 (FIFO 큐)
LRANGE queue:tasks 0 -1              # 모든 요소 조회
LLEN queue:tasks                      # 길이

# Sets
SADD tags:post:1 "python" "redis" "backend"
SMEMBERS tags:post:1
SISMEMBER tags:post:1 "python"       # 멤버십 확인
SINTER tags:post:1 tags:post:2       # 교집합

# Sorted Sets
ZADD leaderboard 100 "alice" 85 "bob" 92 "charlie"
ZREVRANGE leaderboard 0 9 WITHSCORES  # 상위 10개
ZRANK leaderboard "alice"             # 순위 (0부터)
ZINCRBY leaderboard 5 "bob"           # 점수 증가

# 키 관리
DEL key1 key2                         # 키 삭제
EXISTS key1                           # 존재 확인
EXPIRE key1 300                       # TTL 설정
TTL key1                              # 남은 TTL 확인
KEYS "user:*"                         # 키 찾기 (프로덕션에서 사용 금지)
SCAN 0 MATCH "user:*" COUNT 100      # 안전하게 키 반복
```

### Python Redis 클라이언트 설정

```python
import redis
import json
from typing import Optional, Any

# 연결
r = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,       # 바이트 대신 문자열 반환
    socket_connect_timeout=5,
    retry_on_timeout=True,
)

# 커넥션 풀 (프로덕션 권장)
pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    max_connections=20,
    decode_responses=True,
)
r = redis.Redis(connection_pool=pool)

# 상태 확인
r.ping()  # True 반환
```

### Node.js Redis 클라이언트 설정

```javascript
import { createClient } from 'redis';

const client = createClient({
    url: 'redis://localhost:6379',
    socket: {
        connectTimeout: 5000,
        reconnectStrategy: (retries) => Math.min(retries * 100, 5000),
    },
});

client.on('error', (err) => console.error('Redis error:', err));
client.on('connect', () => console.log('Redis connected'));

await client.connect();

// 기본 연산
await client.set('key', 'value');
const value = await client.get('key');
await client.setEx('session:abc', 3600, JSON.stringify({ userId: 1 }));
```

---

## 2. 캐시 어사이드 패턴

캐시 어사이드(cache-aside, 지연 로딩이라고도 함)는 가장 일반적인 캐싱 패턴이다. 애플리케이션이 캐시를 명시적으로 관리한다: 먼저 캐시를 확인하고, 미스(miss) 시 데이터베이스에서 가져와 캐시에 저장한다.

### 이론: 캐시 스탬피드 / Dogpile

캐시가 키 X를 TTL 300s로 보관합니다. 300초에 X가 만료됩니다. 301초에 1000개의 동시 요청이 모두 miss합니다. 모두 1000개가 X를 재계산하기 위해 동시에 데이터베이스를 쿼리합니다. 데이터베이스가 녹습니다.

이것이 **캐시 스탬피드**(또는 "dogpile")입니다. 가장 흔한 프로덕션 캐시 실패 모드입니다. 단순한 것부터 정교한 것까지 세 가지 방어.

#### C.1 Locking (single-flight)

첫 요청이 miss하면 `lock:X`로 키된 Redis 락을 잡습니다. miss하는 다른 요청은 그 락에서 블로킹됩니다. 첫 요청이 X를 계산하고, 캐시를 채우고, 락을 해제합니다. 다른 요청이 재시도하고, 캐시에서 X를 찾고, 반환합니다.

```python
def get(key):
    val = cache.get(key)
    if val is not None: return val
    with redis.lock(f"lock:{key}", timeout=5):
        # 락 획득 후 더블 체크
        val = cache.get(key)
        if val is not None: return val
        val = db.get(key)
        cache.set(key, val, ttl=300)
        return val
```

락은 N개의 동시 계산을 1개로 만듭니다. 위험: X가 *매우* hot이면 락 contention — 많은 요청이 락에서 직렬화됩니다. 보유자가 죽으면 데드락을 피하기 위해 락 타임아웃을 사용하세요.

#### C.2 확률적 조기 갱신 (XFetch)

TTL이 발화하기를 기다리는 대신, 만료 약간 전에 *확률적으로* 캐시를 갱신합니다. 만료에 가까울수록 확률이 높아집니다.

```python
def get(key):
    val, ttl_remaining = cache.get_with_ttl(key)
    # 만료에 가까울 때 무작위로 재계산
    if val is None or random() < beta * ttl_remaining_factor:
        val = db.get(key)
        cache.set(key, val, ttl=300)
    return val
```

XFetch 알고리즘이 이를 형식화합니다. 이득: 캐시가 좀처럼 완전히 만료되지 않으므로 스탬피드가 시작되지 않습니다. 락이 필요 없습니다.

#### C.3 Request coalescing

캐시 라이브러리가 같은 키에 대한 보류 중 lookup을 감지하고 하나의 진행 중 백엔드 호출에 합칩니다. 첫 miss가 데이터베이스 쿼리를 시작하고, 같은 키에 대한 동시 miss가 자체 쿼리를 발행하지 않고 그 결과를 기다립니다.

이것이 Go의 `golang.org/x/sync/singleflight`가 하는 것입니다. Caffeine에 비슷한 `LoadingCache`가 있습니다. 규율은 애플리케이션이 아니라 *캐시 라이브러리*에 있습니다 — 일단 그것이 있으면 모든 캐시 lookup이 스탬피드 안전합니다.

#### C.4 방어책 비교

| 방어 | 복잡성 | 효과 | 사용 시기 |
|---------|------------|---------------|-------------|
| Locking | 낮음 | 높음 | 비싼 재계산이 있는 hot 키 |
| 확률적 조기 갱신 | 중간 | 매우 높음 | 동기 캐시 라이브러리가 뒷받침하는 캐시 |
| Request coalescing | 라이브러리 수준 | 높음 | 현대 캐시 라이브러리에 내장 |

프로덕션 캐시는 보통 결합합니다: 확률적 갱신이 채워진 상태를 유지하고, coalescing이 잔여 동시 miss를 처리합니다.

### 흐름

```
1. 애플리케이션이 요청을 수신
2. Redis 캐시에서 키 확인
3. 캐시 HIT  → 캐시된 데이터 반환
4. 캐시 MISS → 데이터베이스 쿼리 → Redis에 결과 저장 → 데이터 반환
```

### Python 구현

```python
import redis
import json
import hashlib
from typing import Optional
from datetime import timedelta

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

class CacheAside:
    def __init__(self, redis_client: redis.Redis, default_ttl: int = 300):
        self.redis = redis_client
        self.default_ttl = default_ttl

    def _make_key(self, prefix: str, identifier: str) -> str:
        return f"{prefix}:{identifier}"

    def get_user(self, user_id: int) -> Optional[dict]:
        cache_key = self._make_key("user", str(user_id))

        # 1단계: 캐시 확인
        cached = self.redis.get(cache_key)
        if cached:
            print(f"Cache HIT: {cache_key}")
            return json.loads(cached)

        # 2단계: 캐시 미스 — 데이터베이스에서 조회
        print(f"Cache MISS: {cache_key}")
        user = self._fetch_user_from_db(user_id)
        if user is None:
            return None

        # 3단계: 캐시 채우기
        self.redis.setex(cache_key, self.default_ttl, json.dumps(user))
        return user

    def invalidate_user(self, user_id: int) -> None:
        cache_key = self._make_key("user", str(user_id))
        self.redis.delete(cache_key)

    def _fetch_user_from_db(self, user_id: int) -> Optional[dict]:
        # 시뮬레이션된 데이터베이스 쿼리
        return {"id": user_id, "name": "Alice", "email": "alice@example.com"}

# 사용법
cache = CacheAside(r, default_ttl=600)
user = cache.get_user(42)       # Cache MISS → DB에서 조회
user = cache.get_user(42)       # Cache HIT → Redis에서 반환
cache.invalidate_user(42)
user = cache.get_user(42)       # Cache MISS → 다시 조회
```

### 배치 캐시 어사이드(Batch Cache-Aside)

단일 왕복으로 여러 항목을 가져오기:

```python
def get_users_batch(self, user_ids: list[int]) -> list[dict]:
    cache_keys = [self._make_key("user", str(uid)) for uid in user_ids]

    # 1단계: 캐시에서 다중 조회
    cached_values = self.redis.mget(cache_keys)

    results = {}
    missing_ids = []

    for uid, cached in zip(user_ids, cached_values):
        if cached:
            results[uid] = json.loads(cached)
        else:
            missing_ids.append(uid)

    # 2단계: 누락된 항목을 데이터베이스에서 조회
    if missing_ids:
        db_users = self._fetch_users_batch_from_db(missing_ids)
        pipe = self.redis.pipeline()
        for user in db_users:
            results[user["id"]] = user
            key = self._make_key("user", str(user["id"]))
            pipe.setex(key, self.default_ttl, json.dumps(user))
        pipe.execute()

    return [results[uid] for uid in user_ids if uid in results]
```

### 캐시 스탬피드 방지(Cache Stampede Prevention)

인기 있는 캐시 키가 만료되면 많은 동시 요청이 데이터베이스에 동시에 도달할 수 있다. 이를 방지하기 위해 잠금(lock)을 사용한다:

```python
import time

def get_user_with_lock(self, user_id: int) -> Optional[dict]:
    cache_key = self._make_key("user", str(user_id))
    lock_key = f"lock:{cache_key}"

    cached = self.redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # 잠금 획득 시도 (NX = 존재하지 않을 때만 설정, EX = 만료)
    acquired = self.redis.set(lock_key, "1", nx=True, ex=10)

    if acquired:
        try:
            user = self._fetch_user_from_db(user_id)
            if user:
                self.redis.setex(cache_key, self.default_ttl, json.dumps(user))
            return user
        finally:
            self.redis.delete(lock_key)
    else:
        # 다른 프로세스가 조회 중; 대기 후 재시도
        for _ in range(50):  # 50 * 0.1s = 최대 5초
            time.sleep(0.1)
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        # 데이터베이스로 폴백
        return self._fetch_user_from_db(user_id)
```

---

## 3. 라이트스루와 라이트비하인드

### 이론: 세 가지 캐싱 전략

캐시 전략은 본질적으로 *누가 언제 캐시에 쓰는가*에 관한 것입니다. 세 패턴이 지배합니다.

#### A.1 Cache-aside (lazy loading)

애플리케이션이 캐시에서 읽고, miss 시 데이터베이스에서 읽고 캐시를 채웁니다.

```
읽기:
  if cache.hit(key):  return cache.get(key)
  data = db.get(key)
  cache.set(key, data, ttl=300)
  return data

쓰기:
  db.update(key, value)
  cache.delete(key)   # 또는 새 값으로 set
```

성질:

- 캐시는 *side cache*입니다 — 애플리케이션이 책임자입니다.
- 쓰기나 축출 후 첫 읽기는 느립니다(캐시 miss).
- 강한 일관성은 모든 쓰기에서 캐시를 무효화해야 합니다 — 한 프로세스에서는 쉽고, 서비스 간에는 어렵습니다.

이것이 기본 패턴입니다. cache-aside가 맞지 않을 때만 다른 것에 손을 대세요.

#### A.2 Write-through

쓰기는 캐시를 거쳐 데이터베이스로 갑니다. 읽기는 항상 캐시에서 옵니다.

```
쓰기:
  cache.set(key, value)
  db.update(key, value)  # 동기

읽기:
  return cache.get(key)  # 항상 채워져 있음
```

성질:

- 캐시는 항상 데이터베이스와 일관됩니다(둘 다 쓰기에 성공한다고 가정).
- 쓰기 지연시간은 `cache_write + db_write`입니다(write-behind보다 느림).
- 캐시 크기가 모든 읽기를 보관해야 합니다 — 큰 데이터셋에는 비쌉니다.
- 캐시가 차가우면(재시작 후) 채워질 때까지 모든 읽기가 miss합니다. 종종 cache-warming 전략과 짝지어 사용.

읽기가 빨라야 하고 일관성이 쓰기 지연시간보다 더 중요할 때 사용하세요.

#### A.3 Write-behind (write-back)

쓰기는 캐시로 가고, 데이터베이스는 비동기적으로(종종 배치로) 업데이트됩니다.

```
쓰기:
  cache.set(key, value)
  queue.push(("write", key, value))  # async 워커가 이를 읽음

워커 (별도):
  for 큐의 쓰기 배치:
    db.bulk_update(batch)
```

성질:

- 가장 낮은 쓰기 지연시간(요청을 블로킹하는 것은 캐시 쓰기뿐).
- 가장 높은 처리량(쓰기가 배치됨).
- 데이터 손실 위험: 큐가 비기 전에 캐시가 충돌하면, 큐된 쓰기가 사라집니다.
- 결과적 일관성: 지연 창 동안 데이터베이스 읽기는 stale 데이터를 봅니다.

데이터가 교체 가능한(analytics 카운터, 텔레메트리) 또는 캐시가 지속성 있는(AOF persistence를 가진 Redis) 고처리량 쓰기 워크로드에 사용하세요.

#### A.4 전략 고르기

| 전략 | 일관성 | 읽기 지연시간 | 쓰기 지연시간 | 복잡성 |
|----------|-------------|--------------|---------------|------------|
| Cache-aside | 결과적 | miss 시 DB | 빠름 | 낮음 |
| Write-through | 강함 | 캐시만 | 느림 | 중간 |
| Write-behind | 결과적 | 캐시만 | 가장 빠름 | 높음 |

대부분의 앱이 cache-aside를 씁니다. 특화된 워크로드가 write-through(실시간 대시보드)나 write-behind(카운터, analytics)를 씁니다.

### 라이트스루 패턴(Write-Through Pattern)

모든 쓰기가 캐시와 데이터베이스에 동기적으로 전달된다. 캐시 일관성을 보장하지만 쓰기 지연 시간이 증가한다.

```python
class WriteThrough:
    def __init__(self, redis_client, db_session, ttl=600):
        self.redis = redis_client
        self.db = db_session
        self.ttl = ttl

    def update_user(self, user_id: int, data: dict) -> dict:
        # 1단계: 데이터베이스에 쓰기
        user = self._update_db(user_id, data)

        # 2단계: 캐시에 쓰기 (동일 트랜잭션 컨텍스트)
        cache_key = f"user:{user_id}"
        self.redis.setex(cache_key, self.ttl, json.dumps(user))

        return user

    def create_user(self, data: dict) -> dict:
        # 1단계: 데이터베이스에 삽입
        user = self._insert_db(data)

        # 2단계: 캐시 채우기
        cache_key = f"user:{user['id']}"
        self.redis.setex(cache_key, self.ttl, json.dumps(user))

        return user

    def _update_db(self, user_id, data):
        # 데이터베이스 UPDATE 쿼리
        return {"id": user_id, **data}

    def _insert_db(self, data):
        # 데이터베이스 INSERT 쿼리
        return {"id": 1, **data}
```

### 라이트비하인드(Write-Back) 패턴

쓰기가 즉시 캐시에 반영되고, 백그라운드 프로세스가 비동기적으로 변경사항을 데이터베이스에 플러시한다. 쓰기 지연 시간을 줄이지만 데이터 손실 위험이 있다.

```python
import threading
import queue

class WriteBehind:
    def __init__(self, redis_client, db_session, flush_interval=5):
        self.redis = redis_client
        self.db = db_session
        self.write_queue = queue.Queue()
        self.flush_interval = flush_interval
        self._start_flusher()

    def update_user(self, user_id: int, data: dict) -> dict:
        user = {"id": user_id, **data}
        cache_key = f"user:{user_id}"

        # 1단계: 캐시에 즉시 쓰기
        self.redis.setex(cache_key, 3600, json.dumps(user))

        # 2단계: 비동기 데이터베이스 플러시를 위해 큐에 추가
        self.write_queue.put(("update", "user", user_id, data))

        return user

    def _start_flusher(self):
        def flush_worker():
            while True:
                batch = []
                try:
                    while len(batch) < 100:
                        item = self.write_queue.get(timeout=self.flush_interval)
                        batch.append(item)
                except queue.Empty:
                    pass

                if batch:
                    self._flush_to_db(batch)

        thread = threading.Thread(target=flush_worker, daemon=True)
        thread.start()

    def _flush_to_db(self, batch):
        for operation, table, entity_id, data in batch:
            try:
                if operation == "update":
                    # UPDATE 쿼리 실행
                    print(f"Flushed {operation} {table}:{entity_id}")
            except Exception as e:
                print(f"Flush error: {e}")
                # 재큐잉 또는 데드레터 큐에 쓰기
```

### 비교

| 측면 | 캐시 어사이드 | 라이트스루 | 라이트비하인드 |
|---|---|---|---|
| 읽기 지연 | 미스 패널티 | 항상 빠름 | 항상 빠름 |
| 쓰기 지연 | N/A (캐시 우회) | 높음 (이중 쓰기) | 가장 낮음 (캐시만) |
| 일관성 | 최종적(eventual) | 강한(strong) | 최종적(eventual) |
| 복잡도 | 낮음 | 중간 | 높음 |
| 데이터 손실 위험 | 없음 | 없음 | 가능 |

---

## 4. 캐시 무효화 전략

캐시 무효화(cache invalidation)는 컴퓨터 과학에서 가장 어려운 문제 중 하나이다. 실용적인 전략들을 살펴보자.

### 이론: 축출 정책

캐시는 유한한 RAM을 보관합니다. 새 키를 set하는데 캐시가 가득 차 있으면, 무언가가 축출되어야 합니다. 축출 정책이 무엇을 결정합니다.

#### B.1 고전들

- **LRU (Least Recently Used).** 가장 오래 접근되지 않은 키를 축출. 이중 연결 리스트 + 해시맵으로 구현되며, 접근이 노드를 앞으로 이동시킵니다. 단순, 낮은 오버헤드, 최근 사용된 항목이 다시 사용될 가능성이 높은 전형적 접근 패턴에 좋음.
- **LFU (Least Frequently Used).** 가장 낮은 접근 빈도의 키를 축출. 거의 접근되지 않는 키의 long-tail이 있는 워크로드에서 LRU보다 좋지만, 키당 카운터가 필요. "scan pollution"에 취약 — 차가운 데이터의 일회성 스캔이 빈도를 부풀려 hot 데이터를 축출시킵니다.
- **FIFO (First In, First Out).** 삽입 시간으로 가장 오래된 키를 축출. 가장 단순; 거의 최선이 아님.

#### B.2 현대 하이브리드

- **ARC (Adaptive Replacement Cache).** 두 개의 LRU 리스트(최근에 한 번 사용, 자주 사용)를 유지하고 동적으로 균형을 맞춥니다. Scan pollution에 저항. 일부 사용에 특허; 메인라인 Redis에 없음.
- **TinyLFU / W-TinyLFU.** 빈도 스케치(Count-Min)가 새 키의 예측 빈도가 LFU 후보를 초과할 때만 그것을 받아들입니다. Caffeine(Java 캐시 라이브러리)에서 사용, 일반 캐시에 대해 state-of-the-art.

#### B.3 Redis의 축출 정책

Redis는 `maxmemory-policy`를 통해 메뉴를 제공합니다.

```
allkeys-lru           # 모든 키에 걸친 LRU
allkeys-lfu           # 모든 키에 걸친 LFU (Redis 4+)
volatile-lru          # TTL이 설정된 키에만 LRU
volatile-lfu          # TTL이 설정된 키에만 LFU
allkeys-random        # 무작위 축출 (거의 유용하지 않음)
volatile-ttl          # TTL에 가장 가까운 키 축출
noeviction            # 가득 찼을 때 쓰기 거부 (오류 반환)
```

선택은 워크로드에 달려 있습니다.

- **순수 캐시** (모든 것이 best-effort): `allkeys-lru` 또는 `allkeys-lfu`.
- **같은 Redis의 캐시 + 영구 데이터 혼합**: `volatile-lru`(TTL이 있는 키만 축출).
- **축출 받아들일 수 없음** (예: 세션 저장소): `noeviction`과 RAM 과대 공급.

### TTL 기반 만료

가장 간단한 접근법: 모든 캐시 항목에 TTL(Time-to-Live)을 설정한다.

```python
# 자주 변경되는 데이터에 짧은 TTL
r.setex("stock:AAPL:price", 30, "150.25")       # 30초

# 사용자 프로필에 중간 TTL
r.setex("user:42:profile", 600, json.dumps(profile))  # 10분

# 드물게 변경되는 데이터에 긴 TTL
r.setex("config:feature_flags", 3600, json.dumps(flags))  # 1시간

# 접근 패턴에 따른 적응형 TTL
def get_adaptive_ttl(key: str, base_ttl: int = 300) -> int:
    access_count = r.incr(f"access_count:{key}")
    r.expire(f"access_count:{key}", 3600)

    if access_count > 100:
        return base_ttl * 3   # 핫 키: TTL 연장
    elif access_count > 10:
        return base_ttl       # 보통
    else:
        return base_ttl // 2  # 콜드 키: 짧은 TTL
```

### 이벤트 기반 무효화(Event-Driven Invalidation)

기반 데이터가 변경될 때 캐시 항목을 무효화한다:

```python
class EventDrivenCache:
    def __init__(self, redis_client):
        self.redis = redis_client

    def on_user_updated(self, user_id: int):
        """사용자 테이블에서 UPDATE 후 호출된다."""
        # 사용자 캐시 삭제
        self.redis.delete(f"user:{user_id}")

        # 관련 캐시 삭제
        self.redis.delete(f"user:{user_id}:posts")
        self.redis.delete(f"user:{user_id}:stats")

        # 이 사용자를 포함할 수 있는 목록 캐시 무효화
        self._invalidate_pattern(f"users:list:*")

    def on_post_created(self, post: dict):
        """새 게시물 생성 후 호출된다."""
        author_id = post["author_id"]

        # 작성자의 게시물 목록 무효화
        self.redis.delete(f"user:{author_id}:posts")

        # 페이지네이션된 게시물 목록 무효화
        self._invalidate_pattern("posts:page:*")

        # 삭제 대신 버전 카운터 증가
        self.redis.incr("posts:version")

    def _invalidate_pattern(self, pattern: str):
        """SCAN을 사용하여 패턴에 맞는 모든 키를 삭제한다."""
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break
```

### 버전 기반 무효화(Version-Based Invalidation)

캐시 항목을 삭제하는 대신, 캐시 키에 버전 번호를 사용한다:

```python
def get_posts_versioned(self, page: int) -> list:
    version = self.redis.get("posts:version") or "0"
    cache_key = f"posts:v{version}:page:{page}"

    cached = self.redis.get(cache_key)
    if cached:
        return json.loads(cached)

    posts = self._fetch_posts_from_db(page)
    self.redis.setex(cache_key, 300, json.dumps(posts))
    return posts

def invalidate_posts(self):
    # 단순히 버전을 증가시킨다; 이전 키는 TTL을 통해 만료
    self.redis.incr("posts:version")
```

---

## 5. 세션 저장소로서의 Redis

세션 저장소에 Redis를 사용하면 무상태(stateless) 애플리케이션 서버가 가능해져 수평 확장이 간단해진다.

### Express.js 세션 저장소

```javascript
import express from 'express';
import session from 'express-session';
import RedisStore from 'connect-redis';
import { createClient } from 'redis';

const redisClient = createClient({ url: 'redis://localhost:6379' });
await redisClient.connect();

const app = express();

app.use(session({
    store: new RedisStore({ client: redisClient }),
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: process.env.NODE_ENV === 'production',
        httpOnly: true,
        maxAge: 24 * 60 * 60 * 1000,  // 24시간
        sameSite: 'strict',
    },
}));

app.post('/login', (req, res) => {
    // 인증 후
    req.session.userId = user.id;
    req.session.role = user.role;
    res.json({ message: 'Logged in' });
});

app.get('/profile', (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({ error: 'Not authenticated' });
    }
    res.json({ userId: req.session.userId });
});

app.post('/logout', (req, res) => {
    req.session.destroy((err) => {
        res.json({ message: 'Logged out' });
    });
});
```

### Python Flask 세션 저장소

```python
from flask import Flask, session
from flask_session import Session
import redis

app = Flask(__name__)
app.config.update(
    SESSION_TYPE="redis",
    SESSION_REDIS=redis.Redis(host="localhost", port=6379, db=1),
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=86400,  # 24시간
    SESSION_KEY_PREFIX="session:",
    SESSION_USE_SIGNER=True,
    SECRET_KEY="your-secret-key",
)
Session(app)

@app.post("/login")
def login():
    # 인증 후
    session["user_id"] = user.id
    session["role"] = user.role
    return {"message": "Logged in"}

@app.get("/profile")
def profile():
    user_id = session.get("user_id")
    if not user_id:
        return {"error": "Not authenticated"}, 401
    return {"user_id": user_id}
```

---

## 6. Redis를 이용한 속도 제한

Redis의 원자적 연산은 분산 속도 제한기(rate limiter) 구현에 이상적이다.

### 고정 윈도우 속도 제한기(Fixed Window Rate Limiter)

```python
def fixed_window_rate_limit(
    redis_client: redis.Redis,
    key: str,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int]:
    """
    (allowed: bool, remaining: int)을 반환한다.
    """
    window_key = f"ratelimit:{key}:{int(time.time()) // window_seconds}"

    pipe = redis_client.pipeline()
    pipe.incr(window_key)
    pipe.expire(window_key, window_seconds)
    count, _ = pipe.execute()

    allowed = count <= limit
    remaining = max(0, limit - count)
    return allowed, remaining

# 사용법
allowed, remaining = fixed_window_rate_limit(r, "api:user:42", limit=100, window_seconds=60)
if not allowed:
    print("Rate limit exceeded")
```

### 슬라이딩 윈도우 속도 제한기(Sliding Window Rate Limiter)

고정 윈도우보다 정확하며, 윈도우 경계에서의 버스트(burst)를 방지한다:

```python
def sliding_window_rate_limit(
    redis_client: redis.Redis,
    key: str,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int]:
    now = time.time()
    window_start = now - window_seconds
    member = f"{now}"

    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)  # 이전 항목 제거
    pipe.zadd(key, {member: now})                 # 현재 요청 추가
    pipe.zcard(key)                               # 윈도우 내 항목 수
    pipe.expire(key, window_seconds)              # 정리를 위한 TTL 설정
    _, _, count, _ = pipe.execute()

    allowed = count <= limit
    remaining = max(0, limit - count)
    return allowed, remaining
```

### 토큰 버킷 속도 제한기 (Lua 스크립트)

최고 성능을 위해 원자적 토큰 버킷 로직에 Lua 스크립트를 사용한다:

```python
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])  -- tokens per second
local now = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1]) or max_tokens
local last_refill = tonumber(data[2]) or now

-- 토큰 충전
local elapsed = now - last_refill
local new_tokens = math.min(max_tokens, tokens + elapsed * refill_rate)

if new_tokens >= 1 then
    new_tokens = new_tokens - 1
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    redis.call('EXPIRE', key, math.ceil(max_tokens / refill_rate) * 2)
    return {1, math.floor(new_tokens)}  -- 허용, 남은 수
else
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    return {0, 0}  -- 거부, 남은 수
end
"""

class TokenBucketLimiter:
    def __init__(self, redis_client, max_tokens=10, refill_rate=1.0):
        self.redis = redis_client
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.script = self.redis.register_script(TOKEN_BUCKET_SCRIPT)

    def allow(self, key: str) -> tuple[bool, int]:
        result = self.script(
            keys=[f"bucket:{key}"],
            args=[self.max_tokens, self.refill_rate, time.time()],
        )
        return bool(result[0]), int(result[1])
```

---

## 7. Pub/Sub 메시징

Redis Pub/Sub는 실시간 기능을 위한 발행-후-잊기(fire-and-forget) 메시징을 제공한다.

### 발행자(Publisher)

```python
import redis
import json
import time

publisher = redis.Redis(host="localhost", port=6379, decode_responses=True)

def publish_event(channel: str, event_type: str, data: dict):
    message = json.dumps({
        "type": event_type,
        "data": data,
        "timestamp": time.time(),
    })
    subscriber_count = publisher.publish(channel, message)
    print(f"Published to {channel}, {subscriber_count} subscribers received")

# 사용법
publish_event("notifications", "new_order", {"order_id": 123, "total": 59.99})
publish_event("chat:room:42", "message", {"user": "Alice", "text": "Hello!"})
```

### 구독자(Subscriber)

```python
import redis
import json

subscriber = redis.Redis(host="localhost", port=6379, decode_responses=True)
pubsub = subscriber.pubsub()

def handle_notification(message):
    if message["type"] == "message":
        data = json.loads(message["data"])
        print(f"Received: {data['type']} -> {data['data']}")

pubsub.subscribe(**{"notifications": handle_notification})

# 블로킹 리스너
thread = pubsub.run_in_thread(sleep_time=0.01)

# 중지하려면:
# thread.stop()
# pubsub.unsubscribe()
```

### Node.js Pub/Sub

```javascript
import { createClient } from 'redis';

// 발행자
const publisher = createClient({ url: 'redis://localhost:6379' });
await publisher.connect();

await publisher.publish('events', JSON.stringify({
    type: 'user_signup',
    data: { userId: 1, email: 'alice@example.com' },
}));

// 구독자 (별도 연결을 사용해야 함)
const subscriber = createClient({ url: 'redis://localhost:6379' });
await subscriber.connect();

await subscriber.subscribe('events', (message) => {
    const event = JSON.parse(message);
    console.log(`Event: ${event.type}`, event.data);
});
```

### Pub/Sub의 한계

- **영속성 없음**: 구독자가 없으면 메시지가 손실됨
- **확인 없음**: 구독자가 메시지를 처리했다는 보장 없음
- **재생 불가**: 과거 메시지를 다시 읽을 수 없음
- 안정적인 메시징이 필요하면 **Redis Streams**(다음 섹션)나 전용 메시지 브로커를 사용하라

---

## 8. Redis Streams

Redis Streams는 소비자 그룹(consumer group), 확인(acknowledgment), 메시지 재생(replay)을 지원하는 영속적이고 추가 전용(append-only)인 로그를 제공한다. Pub/Sub의 신뢰할 수 있는 대안이다.

### 메시지 생산(Producing Messages)

```python
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# 스트림에 항목 추가
message_id = r.xadd("orders", {
    "order_id": "1001",
    "customer_id": "42",
    "total": "59.99",
    "status": "pending",
})
print(f"Added message: {message_id}")  # 예: "1678901234567-0"

# 최대 길이로 추가 (제한된 스트림)
r.xadd("logs", {"level": "info", "msg": "Server started"}, maxlen=10000)
```

### 소비자 그룹(Consumer Groups)

소비자 그룹을 사용하면 여러 소비자가 단일 스트림의 작업을 분배할 수 있다:

```python
# 소비자 그룹 생성 (처음부터 시작)
try:
    r.xgroup_create("orders", "order_processors", id="0", mkstream=True)
except redis.exceptions.ResponseError:
    pass  # 그룹이 이미 존재

# 소비자 1: 메시지 읽기 및 처리
def process_orders(consumer_name: str):
    while True:
        # 이 소비자를 위한 새 메시지 읽기
        messages = r.xreadgroup(
            groupname="order_processors",
            consumername=consumer_name,
            streams={"orders": ">"},   # ">"는 미전달 메시지만을 의미
            count=10,
            block=5000,               # 메시지가 없으면 5초간 블로킹
        )

        if not messages:
            continue

        for stream, entries in messages:
            for message_id, data in entries:
                try:
                    print(f"[{consumer_name}] Processing order {data['order_id']}")
                    # ... 주문 처리 ...

                    # 성공적 처리 확인
                    r.xack("orders", "order_processors", message_id)
                except Exception as e:
                    print(f"Error processing {message_id}: {e}")
                    # 메시지는 다른 소비자에게 재전달됨

# 별도의 스레드/프로세스에서 소비자 실행
import threading
for i in range(3):
    t = threading.Thread(target=process_orders, args=(f"worker-{i}",))
    t.daemon = True
    t.start()
```

### 보류 중인 메시지 클레임(Claiming Pending Messages)

충돌한 소비자가 확인하지 않은 메시지를 처리한다:

```python
def claim_stale_messages(group: str, consumer: str, min_idle_ms: int = 60000):
    """min_idle_ms 이상 유휴 상태인 메시지를 클레임한다."""
    # 보류 중인 메시지 조회
    pending = r.xpending_range("orders", group, "-", "+", count=100)

    stale_ids = [
        entry["message_id"]
        for entry in pending
        if entry["time_since_delivered"] > min_idle_ms
    ]

    if stale_ids:
        claimed = r.xclaim(
            "orders", group, consumer,
            min_idle_time=min_idle_ms,
            message_ids=stale_ids,
        )
        print(f"Claimed {len(claimed)} stale messages")
        return claimed
    return []
```

### 스트림 정보

```python
# 스트림 메타데이터
info = r.xinfo_stream("orders")
print(f"Length: {info['length']}")
print(f"First entry: {info['first-entry']}")
print(f"Last entry: {info['last-entry']}")

# 소비자 그룹 정보
groups = r.xinfo_groups("orders")
for g in groups:
    print(f"Group: {g['name']}, Pending: {g['pending']}, Consumers: {g['consumers']}")
```

---

## 9. 프레임워크 통합

### FastAPI + Redis

```python
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
import json

redis_client: aioredis.Redis = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
    yield
    await redis_client.close()

app = FastAPI(lifespan=lifespan)

async def get_redis() -> aioredis.Redis:
    return redis_client

@app.get("/products/{product_id}")
async def get_product(product_id: int, cache: aioredis.Redis = Depends(get_redis)):
    # 캐시 확인
    cached = await cache.get(f"product:{product_id}")
    if cached:
        return json.loads(cached)

    # 데이터베이스에서 조회
    product = await fetch_product_from_db(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # 5분간 캐시
    await cache.setex(f"product:{product_id}", 300, json.dumps(product))
    return product

@app.put("/products/{product_id}")
async def update_product(
    product_id: int,
    data: dict,
    cache: aioredis.Redis = Depends(get_redis),
):
    product = await update_product_in_db(product_id, data)

    # 캐시 무효화
    await cache.delete(f"product:{product_id}")

    return product
```

### Express.js + Redis 미들웨어

```javascript
import express from 'express';
import { createClient } from 'redis';

const app = express();
const redis = createClient({ url: 'redis://localhost:6379' });
await redis.connect();

// 캐시 미들웨어 팩토리
function cacheMiddleware(ttl = 300) {
    return async (req, res, next) => {
        if (req.method !== 'GET') return next();

        const key = `cache:${req.originalUrl}`;
        const cached = await redis.get(key);

        if (cached) {
            return res.json(JSON.parse(cached));
        }

        // res.json을 오버라이드하여 응답을 캐시
        const originalJson = res.json.bind(res);
        res.json = async (data) => {
            await redis.setEx(key, ttl, JSON.stringify(data));
            return originalJson(data);
        };

        next();
    };
}

// 사용법
app.get('/api/products', cacheMiddleware(60), async (req, res) => {
    const products = await db.query('SELECT * FROM products');
    res.json(products);
});

app.put('/api/products/:id', async (req, res) => {
    const product = await db.updateProduct(req.params.id, req.body);

    // 관련 캐시 무효화
    await redis.del(`cache:/api/products`);
    await redis.del(`cache:/api/products/${req.params.id}`);

    res.json(product);
});
```

### Django + Redis 캐시 백엔드

```python
# settings.py
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": "redis://localhost:6379/0",
        "OPTIONS": {
            "db": 0,
            "parser_class": "redis.connection.DefaultParser",
            "pool_class": "redis.BlockingConnectionPool",
        },
    }
}

# views.py
from django.core.cache import cache
from django.views.decorators.cache import cache_page

# 저수준 캐시 API
def get_product(request, product_id):
    cache_key = f"product:{product_id}"
    product = cache.get(cache_key)

    if product is None:
        product = Product.objects.get(id=product_id)
        cache.set(cache_key, product, timeout=300)

    return JsonResponse(product.to_dict())

# 뷰 수준 캐싱 데코레이터
@cache_page(60 * 5)  # 전체 뷰를 5분간 캐시
def product_list(request):
    products = Product.objects.all()
    return JsonResponse([p.to_dict() for p in products], safe=False)
```

---

## 10. 연습 문제

### 연습 1: 캐시 어사이드 계층 구축

다음 기능을 가진 Python 클래스 `ArticleCache`를 생성하라:
- 기사(id, title, content, author, published_at)에 대한 캐시 어사이드 구현
- `MGET`을 이용한 배치 조회 지원
- 분산 잠금을 이용한 캐시 스탬피드 방지
- `HINCRBY`를 사용하여 Redis에 히트/미스 통계 추적

```python
# 시작 코드
class ArticleCache:
    def __init__(self, redis_client, db, ttl=300):
        self.redis = redis_client
        self.db = db
        self.ttl = ttl

    def get(self, article_id: int) -> dict:
        # TODO: 잠금을 포함한 캐시 어사이드 구현
        pass

    def get_batch(self, article_ids: list[int]) -> list[dict]:
        # TODO: MGET을 이용한 배치 캐시 어사이드 구현
        pass

    def invalidate(self, article_id: int):
        # TODO: 기사 및 관련 캐시 무효화
        pass

    def stats(self) -> dict:
        # TODO: Redis에서 히트/미스 카운트 반환
        pass
```

### 연습 2: 슬라이딩 윈도우 속도 제한기

Redis 정렬 집합(sorted set)을 사용하여 FastAPI용 속도 제한 미들웨어를 구축하라:
- API 키당 분당 100 요청
- `Retry-After` 헤더와 함께 `429 Too Many Requests` 반환
- 속도 제한 헤더 포함: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### 연습 3: Pub/Sub를 이용한 실시간 알림

두 개의 컴포넌트로 알림 시스템을 구축하라:
- **발행자**: 채널에 메시지를 발행하는 FastAPI 엔드포인트 `POST /notify`
- **구독자**: 알림을 수신하고 로깅하는 백그라운드 프로세스
- 채널 패턴 지원 (예: `user:*:notifications`)
- Redis Lists를 사용한 메시지 이력 기능 추가 (최근 50개 메시지)

### 연습 4: Streams를 이용한 주문 처리 파이프라인

Redis Streams를 사용하여 주문 처리 시스템을 구현하라:
- 생산자(Producer): `orders` 스트림에 주문 추가
- 3개의 워커를 가진 소비자 그룹: 각 워커가 주문을 처리하고 확인
- 데드레터 처리: 60초 이상 유휴 상태인 메시지 클레임
- 대시보드: 스트림 길이, 보류 카운트, 소비자 지연(lag)을 반환하는 엔드포인트

```python
# 시작 코드
class OrderPipeline:
    STREAM = "orders"
    GROUP = "order_processors"

    def __init__(self, redis_client):
        self.redis = redis_client
        self._ensure_group()

    def submit_order(self, order: dict) -> str:
        # TODO: 스트림에 주문 추가
        pass

    def process(self, consumer_name: str):
        # TODO: 주문 읽기, 처리, 확인
        pass

    def claim_stale(self, consumer_name: str, min_idle_ms=60000):
        # TODO: 미확인 메시지 클레임
        pass

    def dashboard(self) -> dict:
        # TODO: 스트림 및 소비자 그룹 통계 반환
        pass
```

---

## 참고 자료

- [Redis Documentation](https://redis.io/docs/)
- [Redis University](https://university.redis.io/)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)
- [Caching Strategies and How to Choose the Right One](https://codeahoy.com/2017/08/11/caching-strategies-and-how-to-choose-the-right-one/)
- [redis-py Documentation](https://redis-py.readthedocs.io/)
- [ioredis (Node.js)](https://github.com/redis/ioredis)

---

**이전**: [Go 웹 기초](./19_Go_Web_Basics.md) | **다음**: [작업 큐](./21_Job_Queues.md)
