# 21. 작업 큐

**이전**: [Redis 캐싱 패턴](./20_Redis_Caching_Patterns.md)

**난이도**: ⭐⭐⭐

## 학습 목표

- 안정적인 비동기 처리를 위해 작업 큐(job queue)가 필수적인 이유를 이해한다
- Redis와 RabbitMQ를 백엔드로 사용하는 Celery(Python) 태스크 큐를 구현한다
- Redis를 사용하는 Bull/BullMQ(Node.js)로 작업 프로세서를 구축한다
- 작업의 전체 수명 주기를 관리한다: 큐 등록, 처리, 재시도, 데드레터(dead-letter)
- 복잡한 워크플로우를 위한 우선순위 큐와 지연/예약 작업을 사용한다
- 동시성 제어와 리소스 관리를 통해 워커를 확장한다
- Flower(Python)와 Bull Board(Node.js)로 작업 큐를 모니터링한다
- 일반적인 패턴을 적용한다: 이메일 발송, 이미지 처리, 리포트 생성
- 견고한 오류 처리와 재시도 전략을 설계한다

## 목차

1. [작업 큐가 필요한 이유](#1-작업-큐가-필요한-이유)
2. [Redis/RabbitMQ를 이용한 Celery](#2-redisrabbitmq를-이용한-celery)
3. [Redis를 이용한 Bull/BullMQ](#3-redis를-이용한-bullbullmq)
4. [작업 수명 주기](#4-작업-수명-주기)
5. [우선순위 큐와 지연 작업](#5-우선순위-큐와-지연-작업)
6. [워커 확장과 동시성](#6-워커-확장과-동시성)
7. [모니터링과 관찰 가능성](#7-모니터링과-관찰-가능성)
8. [일반적인 패턴](#8-일반적인-패턴)
9. [오류 처리와 재시도 전략](#9-오류-처리와-재시도-전략)
10. [신뢰성 패턴](#10-신뢰성-패턴)
11. [연습 문제](#11-연습-문제)

---

## 1. 작업 큐가 필요한 이유

작업 큐(job queue)는 시간이 오래 걸리거나 신뢰할 수 없는 작업을 요청-응답 주기에서 분리한다. 사용자를 기다리게 하는 대신, 작업을 백그라운드 큐로 푸시하여 비동기적으로 처리한다.

### 이론: 지속성 큐 위의 Producer-Consumer

패턴: producer가 작업을 큐에 넣고, consumer가 작업을 빼서 처리하고 acknowledge합니다. 그 사이의 큐는 *지속성*이 있습니다 — 양쪽의 충돌에서 살아남습니다.

#### A.1 큐가 실제로 사 주는 것

큐 없이는:

```
웹 요청 → 핸들러 → send_email() (10s) → 응답
```

사용자가 10초를 기다립니다. `send_email()`이 throw하면 요청이 실패합니다. 이메일 서버가 다운되면 이메일이 필요한 모든 요청이 실패합니다.

큐와 함께:

```
웹 요청 → 핸들러 → queue.enqueue("send_email", ...) → 응답 (즉시)
                                ↓
                            워커가 작업을 가져와 send_email 실행, ACK 또는 재시도
```

세 가지가 바뀝니다.

1. **지연시간.** 사용자가 응답을 즉시 봅니다. 이메일은 "진행 중"이지만 요청은 끝났습니다.
2. **신뢰성.** 이메일 서버가 다운되어도 작업은 큐에 남아 재시도됩니다. 웹 요청은 여전히 성공했습니다.
3. **확장.** 이메일 처리 용량이 웹 요청 용량과 독립적으로 확장됩니다. 워커를 추가하면 큐가 더 빨리 빕니다.

#### A.2 지속성 요구사항

이 모든 것이 작동하려면 큐가 워커 충돌보다 오래 살아야 합니다. 워커가 작업을 가져와 처리를 시작하고 완료 전에 충돌하면, 작업은 *잃어버려서는 안 됩니다*. 두 메커니즘이 이를 달성합니다.

- **Acknowledgment (ACK) 의미.** 워커가 작업을 가져오면 큐가 그것을 "in flight"로 표시합니다 — 그러나 삭제하지 않습니다. 워커가 성공적으로 처리하고 ACK한 후에야 큐가 실제로 삭제합니다. 워커가 충돌하면(또는 타임아웃 안에 ACK하지 못하면) 큐가 작업을 다른 워커에 재전달합니다.
- **지속성 저장.** 작업이 큐 서버 재시작에서 살아남습니다. RabbitMQ는 디스크에 씁니다. AOF persistence를 가진 Redis도 그렇습니다. Kafka는 기본으로 모든 것을 디스크에 씁니다.

조합 — ACK + 지속성 — 이 "잃어버리는 작업이 없다"를 줍니다. 그러나 또한 "작업이 두 번 전달될 수 있다"도 줍니다(다음 절).

#### A.3 두 단계 가시성

한 작업의 표준 큐 생명주기:

```
1. enqueued       ← producer가 푸시
2. running         ← 워커가 가져옴, 아직 ACK 안 함
3. completed       ← 워커가 ACK; 큐가 삭제
   또는
3'. failed/retry   ← 워커가 NACK 또는 타임아웃; 큐가 재전달
4. dead letter     ← 재시도 소진; 검사를 위해 별도 큐로 이동
```

각 전이는 지속성 있습니다. 운영자가 "running" 작업(stuck 워커 찾기), "failed" 작업(재시도 카운터), "dead letter" 큐(재시도를 소진한 작업)를 검사할 수 있습니다. 셋 다 관측성에 결정적입니다 — 레슨 17 §7 참조.

### 작업 큐가 해결하는 문제

**사용자 경험**: 이메일 발송, PDF 생성, 이미지 크기 조정을 수행하는 요청이 HTTP 응답을 차단해서는 안 된다.

**신뢰성(Reliability)**: 외부 API가 다운되면, 사용자의 요청을 실패시키는 대신 나중에 작업을 재시도할 수 있다.

**속도 제한(Rate limiting)**: 외부 API에는 종종 속도 제한이 있다. 큐를 통해 처리량을 제어할 수 있다.

**리소스 관리**: CPU 집약적 작업(이미지 처리, ML 추론)을 웹 서버에 영향을 주지 않고 전용 워커에서 처리할 수 있다.

**순서 보장**: 일부 워크플로우는 작업을 특정 순서로 처리해야 한다.

### 아키텍처 개요

```
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐
│  Web App │────▶│  Message   │────▶│  Worker  │────▶│  Result  │
│ (Producer)│     │  Broker    │     │(Consumer)│     │  Store   │
└──────────┘     │(Redis/RMQ) │     └──────────┘     │ (Redis)  │
                 └───────────┘                       └──────────┘
```

### 메시지 브로커 비교

| 기능 | Redis | RabbitMQ | Amazon SQS |
|---|---|---|---|
| 프로토콜 | Custom | AMQP | HTTP/SQS |
| 영속성 | 선택적 (AOF/RDB) | 기본적으로 디스크 | 관리형 |
| 최대 메시지 크기 | 512 MB | 128 MB (설정 가능) | 256 KB |
| 처리량 | 매우 높음 | 높음 | 보통 |
| 메시지 순서 | 큐별 FIFO | 큐별 FIFO | 최선 노력 (FIFO 가능) |
| 데드레터 큐 | 수동 | 내장 | 내장 |
| 복잡도 | 낮음 | 중간 | 낮음 (관리형) |

### 태스크 큐 라이브러리 비교(Task Queue Library Comparison)

| 기능 | Celery (Python) | BullMQ (Node.js) |
|------|----------------|-------------------|
| 브로커(Broker) | Redis, RabbitMQ | Redis 전용 |
| 결과 백엔드(Result backend) | Redis, DB, S3 | Redis |
| 스케줄링(Scheduling) | celery-beat | 내장 반복 가능 |
| 우선순위 큐(Priority queues) | 지원 (제한적) | 지원 (네이티브) |
| 속도 제한(Rate limiting) | 태스크별 데코레이터 | 큐별 설정 |
| 데드레터 큐(Dead letter queue) | 수동 (on_failure) | 내장 |
| 모니터링(Monitoring) | Flower 대시보드 | Bull Board |

---

## 2. Redis/RabbitMQ를 이용한 Celery

[Celery](https://docs.celeryq.dev/)는 Python에서 태스크 큐의 사실상 표준이다. Redis와 RabbitMQ를 메시지 브로커로 지원한다.

### 이론: 비교된 분산 큐

세 가지 큐 카테고리가 백엔드 생태계를 지배합니다. 각각이 다른 설계 중심을 갖습니다.

#### C.1 Redis 기반 큐 (Celery, BullMQ, Sidekiq)

Redis 리스트나 stream을 큐로 사용. 성질:

- **장점:** 운영이 단순, 낮은 지연시간, 이미 캐싱을 위해 배포된 경우가 많음.
- **단점:** Redis 지속성이 제한적(RDB 스냅샷, AOF append-only file). 충돌한 Redis가 최근에 ACK된 작업을 잃을 수 있음. 단일 노드 처리량 한계 ~100K 작업/s.
- **최적:** Redis가 이미 스택에 있는 중소 앱.

#### C.2 RabbitMQ (AMQP 기반)

목적 빌트 메시지 브로커. 성질:

- **장점:** 강한 전달 보장, 정교한 라우팅(exchange, binding), 메시지별 ACK, 내장 dead-letter exchange. 성숙한 운영 도구.
- **단점:** 운영해야 할 또 다른 인프라 조각. 클러스터 설정이 사소하지 않음.
- **최적:** 엔터프라이즈 메시징 기능, 복잡한 라우팅, 엄격한 전달 보장이 필요한 앱.

Exchange/binding 모델은 fan-out(하나의 메시지를 여러 큐에), topic 기반 라우팅(regex 매칭), 또는 단순 direct 큐 — 모두 같은 브로커에서 — 를 구현하게 해줍니다.

#### C.3 Kafka (로그 기반)

전통적 큐가 *아닙니다* — *분산 commit log*입니다. Producer가 *topic*에 메시지를 append합니다. Consumer가 원하는 offset에서 읽습니다. 성질:

- **장점:** 거대한 처리량(클러스터에서 초당 수백만 메시지), 지속성 있는 보유(메시지를 며칠/몇 주간 보관), reprocessing(어떤 offset으로든 되감기), partition 안에서 엄격한 순서.
- **단점:** Redis나 RabbitMQ보다 더 높은 지연시간. 운영적으로 복잡(ZooKeeper나 KRaft, partitioning, consumer group). 낮은 양의 워크로드에는 과합니다.
- **최적:** 이벤트 스트리밍, audit log, analytics 파이프라인, 보유와 replay가 중요한 어떤 것이든.

#### C.4 Partitioning vs exchange

Kafka와 RabbitMQ 사이의 가장 깊은 아키텍처 차이는 *메시지가 consumer에 어떻게 분배되는가*입니다.

- **RabbitMQ exchange + queue.** 메시지가 라우팅 규칙에 따라 하나 이상의 큐로 갑니다. 각 큐가 자체 consumer를 가집니다. 순서는 큐별입니다.
- **Kafka partition.** Topic이 partition으로 나뉩니다. 각 partition은 consumer group의 정확히 한 consumer가 소비합니다. 순서는 partition별입니다. 확장 = partition 추가 + consumer 추가.

Kafka의 partitioning은 within-key 순서가 co-located 처리를 필요로 한다는 비용으로 자연스러운 수평 확장을 줍니다. RabbitMQ의 exchange 모델은 유연한 라우팅을 주지만 내장 partition 개념이 없습니다 — 여러 큐로 직접 만들어야 합니다.

#### C.5 큐 고르기

| 워크로드 | 큐 |
|----------|-------|
| 기존 Redis 사용 앱의 백그라운드 작업 | Redis 위 Celery / BullMQ |
| 풍부한 라우팅이 있는 팀 간 메시징 | RabbitMQ |
| 고볼륨 이벤트 스트림, 보유 중요 | Kafka |
| AWS/GCP 네이티브 | SQS / Pub/Sub (위의 관리형 등가물) |

### 설정

```bash
pip install celery[redis]
# 또는 RabbitMQ의 경우:
pip install celery[rabbitmq]
```

### 기본 설정

```python
# celery_app.py
from celery import Celery

app = Celery(
    "myapp",
    broker="redis://localhost:6379/0",       # 메시지 브로커
    backend="redis://localhost:6379/1",      # 결과 백엔드
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,            # 하드 제한: 5분
    task_soft_time_limit=240,       # 소프트 제한: 4분 (SoftTimeLimitExceeded 발생)
    worker_prefetch_multiplier=1,   # 한 번에 하나의 태스크만 가져옴
    worker_max_tasks_per_child=100, # 100개 태스크 후 워커 재시작 (메모리 누수 방지)
)
```

### 태스크 정의

```python
# tasks.py
from celery_app import app
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_email(self, to: str, subject: str, body: str):
    """비동기적으로 이메일을 발송한다."""
    try:
        logger.info(f"Sending email to {to}: {subject}")
        # ... SMTP 로직 ...
        return {"status": "sent", "to": to}
    except ConnectionError as exc:
        logger.warning(f"Email failed, retrying: {exc}")
        raise self.retry(exc=exc)

@app.task(bind=True, max_retries=5)
def process_image(self, image_path: str, operations: list[str]):
    """이미지 크기 조정, 자르기 또는 변환."""
    try:
        logger.info(f"Processing image: {image_path}")
        # ... PIL/Pillow 로직 ...
        return {"status": "processed", "path": image_path}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)

@app.task
def generate_report(report_type: str, params: dict):
    """리포트를 생성한다 (PDF, CSV 등)."""
    logger.info(f"Generating {report_type} report with params: {params}")
    # ... 리포트 생성 로직 ...
    return {"status": "completed", "download_url": "/reports/123.pdf"}
```

### 태스크 호출

```python
# 비동기 호출 (즉시 AsyncResult 반환)
result = send_email.delay("user@example.com", "Welcome!", "Hello there")
print(f"Task ID: {result.id}")
print(f"Status: {result.status}")   # PENDING, STARTED, SUCCESS, FAILURE

# 결과 대기 (블로킹)
output = result.get(timeout=30)
print(f"Result: {output}")

# 더 많은 제어를 위한 apply_async
result = send_email.apply_async(
    args=["user@example.com", "Welcome!", "Hello"],
    countdown=60,                    # 실행을 60초 지연
    expires=3600,                    # 1시간 내에 시작되지 않으면 만료
    queue="high_priority",           # 특정 큐로 라우팅
    priority=9,                      # 우선순위 (0-9, 높을수록 우선)
)

# 블로킹 없이 결과 확인
if result.ready():
    if result.successful():
        print(f"Done: {result.result}")
    else:
        print(f"Failed: {result.traceback}")
```

### 태스크 체인, 그룹, 코드(Chord)

```python
from celery import chain, group, chord

# Chain: 순차 실행 (하나의 출력이 다음 입력으로)
workflow = chain(
    fetch_data.s("https://api.example.com/data"),
    transform_data.s(),
    save_to_database.s(),
)
result = workflow.apply_async()

# Group: 병렬 실행
batch = group(
    process_image.s(f"image_{i}.jpg", ["resize", "compress"])
    for i in range(100)
)
result = batch.apply_async()

# Chord: 그룹 + 모든 태스크 완료 후 콜백
workflow = chord(
    group(
        process_image.s(f"image_{i}.jpg", ["resize"])
        for i in range(10)
    ),
    create_gallery.s()  # 모든 이미지 처리 완료 시 호출
)
result = workflow.apply_async()
```

### 워커 실행

```bash
# 워커 시작
celery -A celery_app worker --loglevel=info --concurrency=4

# 특정 큐로 시작
celery -A celery_app worker -Q high_priority,default --concurrency=8

# Celery Beat 시작 (주기적 태스크 스케줄러)
celery -A celery_app beat --loglevel=info
```

---

## 3. Redis를 이용한 Bull/BullMQ

[BullMQ](https://docs.bullmq.io/)는 Redis를 기반으로 하는 Node.js 표준 작업 큐이다. 이전 Bull 라이브러리를 대체한다.

### 설정

```bash
npm install bullmq ioredis
```

### 기본 큐와 워커

```javascript
// queue.js
import { Queue } from 'bullmq';
import IORedis from 'ioredis';

const connection = new IORedis({
    host: 'localhost',
    port: 6379,
    maxRetriesPerRequest: null,  // BullMQ에서 필수
});

export const emailQueue = new Queue('email', { connection });
export const imageQueue = new Queue('image-processing', { connection });

// 작업 추가
await emailQueue.add('send-welcome', {
    to: 'user@example.com',
    subject: 'Welcome!',
    body: 'Hello and welcome to our platform.',
});

// 옵션과 함께 추가
await emailQueue.add('send-notification', {
    to: 'user@example.com',
    subject: 'Order shipped',
}, {
    attempts: 3,
    backoff: { type: 'exponential', delay: 1000 },
    removeOnComplete: { age: 3600 },  // 완료된 작업 1시간 후 제거
    removeOnFail: { age: 86400 },     // 실패한 작업 24시간 후 제거
});
```

### 워커(Worker)

```javascript
// worker.js
import { Worker } from 'bullmq';
import IORedis from 'ioredis';

const connection = new IORedis({
    host: 'localhost',
    port: 6379,
    maxRetriesPerRequest: null,
});

const emailWorker = new Worker('email', async (job) => {
    console.log(`Processing job ${job.id}: ${job.name}`);
    const { to, subject, body } = job.data;

    // 진행률 업데이트
    await job.updateProgress(10);

    // 이메일 발송 시뮬레이션
    await sendEmail(to, subject, body);

    await job.updateProgress(100);

    return { status: 'sent', to };
}, {
    connection,
    concurrency: 5,            // 5개 작업 동시 처리
    limiter: {
        max: 10,               // 최대 10개 작업
        duration: 1000,        // 초당
    },
});

// 이벤트 리스너
emailWorker.on('completed', (job, result) => {
    console.log(`Job ${job.id} completed:`, result);
});

emailWorker.on('failed', (job, err) => {
    console.error(`Job ${job.id} failed:`, err.message);
});

emailWorker.on('progress', (job, progress) => {
    console.log(`Job ${job.id} progress: ${progress}%`);
});
```

### 플로우 프로듀서(Flow Producer) — 작업 의존성

BullMQ는 부모-자식 작업 관계를 지원한다:

```javascript
import { FlowProducer } from 'bullmq';

const flowProducer = new FlowProducer({ connection });

// 의존성이 있는 워크플로우 정의
const flow = await flowProducer.add({
    name: 'send-report',
    queueName: 'email',
    data: { to: 'manager@example.com' },
    children: [
        {
            name: 'generate-pdf',
            queueName: 'reports',
            data: { type: 'monthly-sales' },
        },
        {
            name: 'generate-charts',
            queueName: 'reports',
            data: { type: 'sales-charts' },
        },
    ],
});

// 부모 작업(send-report)은 모든 자식이 완료될 때까지 대기
```

---

## 4. 작업 수명 주기

신뢰할 수 있는 시스템을 구축하기 위해 작업 수명 주기를 이해하는 것이 중요하다.

### 이론: 전달 의미

큐가 작업을 재전달하면 consumer가 두 번 처리할 수 있습니다. 이는 충돌 복구의 결과입니다. 작업 도중에 사라진 워커가 실제로 작업을 완료했는지 *알* 방법이 없습니다. 세 가지 전달 계약.

#### B.1 At-most-once

Producer가 던지고 잊습니다. 큐가 재시도 보장을 하지 않습니다. 워커가 충돌하면 작업이 사라집니다.

```
producer → queue → consumer (충돌) → 아무것도 없음
```

작업을 잃어도 받아들일 수 있을 때 사용: 텔레메트리 샘플, "best-effort" 알림, 가치가 낮은 이벤트. 결제, 주문, 사용자가 보는 어떤 것에도 절대 사용하지 마세요.

#### B.2 At-least-once

기본값. 큐가 ACK될 때까지 재전달합니다. 작업은 *반드시* 성공합니다 — 그러나 한 번 이상 전달(그리고 처리)될 수 있습니다.

```
producer → queue → consumer (충돌) → queue 재전달 → consumer (성공, ACK)
```

이것이 거의 모든 워크로드에 올바른 기본값입니다. 또한 **consumer가 멱등이어야 한다**는 뜻입니다 — 같은 입력으로 두 번 실행해도 같은 결과를 만들어야 합니다. 그렇지 않으면 사용자가 이메일 두 개, 청구 두 번, 기록 두 개를 받습니다.

#### B.3 Exactly-once: 정말로는 아니지만 — 실용적으로는 그렇다

진정한 "exactly-once" 전달은 무한한 조정 없이는 분산 시스템에서 불가능합니다. 달성 가능한 것: **at-least-once 전달 + 멱등 consumer + idempotency key**, 사용자 관점에서 exactly-once처럼 동작합니다.

Idempotency key 패턴:

```python
def send_email_job(idempotency_key, to, subject, body):
    if processed_keys.exists(idempotency_key):
        return  # 이미 완료; 건너뛰기
    send_email(to, subject, body)
    processed_keys.add(idempotency_key)
```

Producer가 논리적 작업당 고유 키(UUID)를 붙입니다. Consumer가 그 키가 처리되었는지 확인합니다. 그렇다면 no-op. 큐가 재전달하면 두 번째 워커가 `processed_keys`에서 키를 보고 건너뜁니다. 순 결과: 큐가 몇 번 전달했든 이메일이 한 번 보내집니다.

Idempotency-key 패턴이 큐 기반 시스템에서 가장 중요한 단일 설계 규율입니다. 모든 consumer는 at-least-once 전달 아래에서 멱등이도록 설계되어야 합니다.

#### B.4 재시도 전략

작업이 실패하면 큐가 백오프와 함께 재시도합니다.

```
시도 1: 실패, 1s 후 재시도
시도 2: 실패, 2s 후 재시도
시도 3: 실패, 4s 후 재시도
...
시도 N: 실패 → dead-letter queue로 이동
```

지수 백오프는 어려움을 겪는 다운스트림 서비스에 대한 재시도 홍수를 막습니다. N번 시도(보통 5-10) 후 작업은 *dead-letter queue*(DLQ) — 사람의 검사를 위한 별도 큐 — 로 이동합니다. DLQ는 운영적으로 결정적입니다. 어떤 작업이 체계적으로 실패하는지를 잃지 않고 보여줍니다.

### 상태 전환

```
                    ┌─────────────┐
                    │   WAITING   │ (큐에 있음, 아직 픽업되지 않음)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   ACTIVE    │ (워커가 처리 중)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐  ┌─▼──────┐  ┌──▼──────────┐
       │  COMPLETED   │  │ FAILED │  │   DELAYED    │
       │              │  │        │  │ (나중에 재시도)│
       └──────────────┘  └───┬────┘  └──────┬───────┘
                             │              │
                        ┌────▼────┐         │
                        │  RETRY  │─────────┘
                        └────┬────┘
                             │ (최대 재시도 초과)
                        ┌────▼─────────┐
                        │ DEAD-LETTER  │
                        └──────────────┘
```

### Celery 작업 상태

```python
from celery_app import app
from celery.result import AsyncResult

def check_job_status(task_id: str) -> dict:
    result = AsyncResult(task_id, app=app)

    status = {
        "task_id": task_id,
        "status": result.status,  # PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
    }

    if result.ready():
        if result.successful():
            status["result"] = result.result
        else:
            status["error"] = str(result.result)
            status["traceback"] = result.traceback

    return status

# 태스크 취소(revoke)
def cancel_job(task_id: str):
    app.control.revoke(task_id, terminate=True, signal="SIGTERM")
```

### BullMQ 작업 상태

```javascript
import { Queue } from 'bullmq';

const queue = new Queue('email', { connection });

async function checkJobStatus(jobId) {
    const job = await queue.getJob(jobId);
    if (!job) return { error: 'Job not found' };

    const state = await job.getState();
    return {
        id: job.id,
        name: job.name,
        state,                          // waiting, active, completed, failed, delayed
        progress: job.progress,
        data: job.data,
        returnValue: job.returnvalue,
        failedReason: job.failedReason,
        attemptsMade: job.attemptsMade,
        timestamp: job.timestamp,
        processedOn: job.processedOn,
        finishedOn: job.finishedOn,
    };
}

// 큐 통계 조회
async function getQueueStats() {
    const [waiting, active, completed, failed, delayed] = await Promise.all([
        queue.getWaitingCount(),
        queue.getActiveCount(),
        queue.getCompletedCount(),
        queue.getFailedCount(),
        queue.getDelayedCount(),
    ]);
    return { waiting, active, completed, failed, delayed };
}
```

---

## 5. 우선순위 큐와 지연 작업

### Celery 우선순위 큐

```python
# 큐 라우팅 정의
app.conf.task_routes = {
    "tasks.send_email": {"queue": "high_priority"},
    "tasks.generate_report": {"queue": "low_priority"},
    "tasks.process_image": {"queue": "default"},
}

# 또는 동적으로 라우팅
@app.task(bind=True)
def flexible_task(self, data, priority="default"):
    pass

flexible_task.apply_async(
    args=[data],
    queue="high_priority",
    priority=9,  # 0 (가장 낮음) ~ 9 (가장 높음)
)
```

### Celery 주기적 태스크 (Beat)

```python
from celery.schedules import crontab

app.conf.beat_schedule = {
    "cleanup-expired-sessions": {
        "task": "tasks.cleanup_sessions",
        "schedule": crontab(minute=0, hour="*/6"),  # 6시간마다
    },
    "send-daily-digest": {
        "task": "tasks.send_digest",
        "schedule": crontab(minute=0, hour=8),       # 매일 UTC 오전 8시
        "args": ["daily"],
    },
    "check-health": {
        "task": "tasks.health_check",
        "schedule": 30.0,                             # 30초마다
    },
}
```

### BullMQ 우선순위와 지연 작업

```javascript
// 우선순위: 낮은 숫자 = 높은 우선순위
await emailQueue.add('critical-alert', { to: 'admin@example.com' }, {
    priority: 1,  // 우선순위 2, 3 등보다 먼저 처리
});

await emailQueue.add('newsletter', { to: 'user@example.com' }, {
    priority: 10, // 낮은 우선순위
});

// 지연 작업: 5분 후 처리
await emailQueue.add('follow-up', { to: 'user@example.com' }, {
    delay: 5 * 60 * 1000,  // 밀리초 단위로 5분
});

// 예약 작업: 특정 시간에 처리
const targetTime = new Date('2026-03-15T10:00:00Z');
await emailQueue.add('scheduled-email', { to: 'user@example.com' }, {
    delay: targetTime.getTime() - Date.now(),
});

// 반복 작업 (cron과 유사)
await emailQueue.add('daily-report', { type: 'sales' }, {
    repeat: {
        pattern: '0 8 * * *',  // 매일 오전 8시
        tz: 'America/New_York',
    },
});

await emailQueue.add('cleanup', {}, {
    repeat: {
        every: 30 * 60 * 1000,  // 30분마다
    },
});
```

---

## 6. 워커 확장과 동시성

### Celery 동시성 모델

```bash
# Prefork (기본값): 멀티프로세싱, CPU 바운드 태스크에 적합
celery -A celery_app worker --pool=prefork --concurrency=4

# Gevent: 그린 스레드, I/O 바운드 태스크에 적합
celery -A celery_app worker --pool=gevent --concurrency=100

# Eventlet: gevent와 유사
celery -A celery_app worker --pool=eventlet --concurrency=100

# Solo: 단일 스레드, 디버깅에 적합
celery -A celery_app worker --pool=solo
```

### 확장 전략

```python
# 다른 태스크 유형에 대해 다른 워커 구성
# 워커 1: 높은 우선순위, 낮은 동시성 (CPU 바운드)
# celery -A celery_app worker -Q image_processing --concurrency=2 --pool=prefork

# 워커 2: 낮은 우선순위, 높은 동시성 (I/O 바운드)
# celery -A celery_app worker -Q email,notifications --concurrency=50 --pool=gevent

# 워커 3: 기본 큐
# celery -A celery_app worker -Q default --concurrency=8 --pool=prefork
```

### BullMQ 동시성과 속도 제한

```javascript
const worker = new Worker('image-processing', processImage, {
    connection,
    concurrency: 4,                  // 4개 작업 병렬 처리
    limiter: {
        max: 20,                     // 최대 20개 작업
        duration: 60 * 1000,         // 분당
    },
    lockDuration: 300000,            // 작업당 5분 잠금
    stalledInterval: 30000,          // 30초마다 정체된 작업 확인
    maxStalledCount: 2,              // 실패 전 최대 정체 횟수
});

// 샌드박스 워커 (별도 프로세스에서 실행)
const worker = new Worker('cpu-intensive', './processor.js', {
    connection,
    concurrency: 2,
    useWorkerThreads: true,  // 자식 프로세스 대신 워커 스레드 사용
});
```

### Docker Compose 확장

```yaml
# docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  web:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis

  worker-high:
    build: .
    command: celery -A celery_app worker -Q high_priority --concurrency=4
    depends_on:
      - redis
    deploy:
      replicas: 2

  worker-default:
    build: .
    command: celery -A celery_app worker -Q default --concurrency=8
    depends_on:
      - redis
    deploy:
      replicas: 3

  worker-low:
    build: .
    command: celery -A celery_app worker -Q low_priority --concurrency=2
    depends_on:
      - redis
    deploy:
      replicas: 1

  beat:
    build: .
    command: celery -A celery_app beat --loglevel=info
    depends_on:
      - redis
```

---

## 7. 모니터링과 관찰 가능성

### Flower (Celery)

[Flower](https://flower.readthedocs.io/)는 Celery를 위한 실시간 웹 모니터이다.

```bash
pip install flower
celery -A celery_app flower --port=5555 --broker=redis://localhost:6379/0
```

Flower가 제공하는 기능:
- 실시간 워커 상태 (온라인, 하트비트, 처리/실패 횟수)
- 태스크 진행률과 이력
- 큐 길이와 소비자 수
- 태스크 속도 그래프
- 원격 워커 제어 (종료, 풀 크기 조정)

### 프로그래밍 방식 모니터링 (Celery)

```python
from celery_app import app

def get_worker_stats():
    """모든 활성 워커에서 통계를 가져온다."""
    inspector = app.control.inspect()

    return {
        "active_tasks": inspector.active(),
        "reserved_tasks": inspector.reserved(),
        "registered_tasks": inspector.registered(),
        "stats": inspector.stats(),
        "queues": inspector.active_queues(),
    }

def get_queue_lengths():
    """각 큐의 메시지 수를 가져온다."""
    with app.connection_or_acquire() as conn:
        return {
            "high_priority": conn.default_channel.queue_declare(
                "high_priority", passive=True
            ).message_count,
            "default": conn.default_channel.queue_declare(
                "default", passive=True
            ).message_count,
        }
```

### Bull Board (BullMQ)

[Bull Board](https://github.com/felixmosh/bull-board)는 BullMQ를 위한 UI 대시보드를 제공한다.

```javascript
import { createBullBoard } from '@bull-board/api';
import { BullMQAdapter } from '@bull-board/api/bullMQAdapter';
import { ExpressAdapter } from '@bull-board/express';
import express from 'express';

const serverAdapter = new ExpressAdapter();
serverAdapter.setBasePath('/admin/queues');

createBullBoard({
    queues: [
        new BullMQAdapter(emailQueue),
        new BullMQAdapter(imageQueue),
    ],
    serverAdapter,
});

const app = express();
app.use('/admin/queues', serverAdapter.getRouter());
app.listen(3000);
```

### 커스텀 메트릭

```javascript
// Prometheus/Grafana를 위한 메트릭 발행
import { Queue, QueueEvents } from 'bullmq';

const queueEvents = new QueueEvents('email', { connection });

queueEvents.on('completed', ({ jobId, returnvalue }) => {
    metrics.increment('jobs.completed', { queue: 'email' });
});

queueEvents.on('failed', ({ jobId, failedReason }) => {
    metrics.increment('jobs.failed', { queue: 'email' });
});

// 주기적 통계 수집
setInterval(async () => {
    const counts = await emailQueue.getJobCounts();
    metrics.gauge('queue.waiting', counts.waiting, { queue: 'email' });
    metrics.gauge('queue.active', counts.active, { queue: 'email' });
    metrics.gauge('queue.failed', counts.failed, { queue: 'email' });
}, 10000);
```

---

## 8. 일반적인 패턴

### 이메일 발송 파이프라인

```python
@app.task(bind=True, max_retries=3, default_retry_delay=30)
def send_transactional_email(self, template: str, recipient: str, context: dict):
    """템플릿 렌더링을 포함한 트랜잭션 이메일 발송."""
    try:
        # 템플릿 렌더링
        html_content = render_template(template, context)

        # SMTP/API를 통해 발송
        response = email_client.send(
            to=recipient,
            subject=context.get("subject", "Notification"),
            html=html_content,
        )

        return {"message_id": response.id, "status": "sent"}
    except RateLimitError:
        # 지수 백오프로 재시도
        raise self.retry(countdown=60 * (2 ** self.request.retries))
    except InvalidRecipientError:
        # 영구적 실패에는 재시도하지 않음
        return {"status": "skipped", "reason": "invalid_recipient"}

# 배치 이메일 발송
@app.task
def send_newsletter(campaign_id: int):
    """모든 구독자에게 뉴스레터를 발송한다."""
    subscribers = get_subscribers(campaign_id)

    # 개별 이메일 태스크의 그룹 생성
    batch = group(
        send_transactional_email.s("newsletter", sub.email, {
            "subject": "Weekly Newsletter",
            "name": sub.name,
            "campaign_id": campaign_id,
        })
        for sub in subscribers
    )

    result = batch.apply_async()
    return {"total": len(subscribers), "group_id": result.id}
```

### 이미지 처리 파이프라인

```python
from celery import chain

@app.task
def download_image(url: str) -> str:
    """이미지를 다운로드하고 로컬 경로를 반환한다."""
    response = requests.get(url)
    path = f"/tmp/images/{uuid4()}.jpg"
    with open(path, "wb") as f:
        f.write(response.content)
    return path

@app.task
def resize_image(path: str, width: int = 800, height: int = 600) -> str:
    """이미지를 지정된 크기로 조정한다."""
    from PIL import Image
    img = Image.open(path)
    img = img.resize((width, height), Image.LANCZOS)
    output_path = path.replace(".jpg", f"_{width}x{height}.jpg")
    img.save(output_path, quality=85)
    return output_path

@app.task
def upload_to_s3(path: str) -> str:
    """처리된 이미지를 S3에 업로드한다."""
    key = f"images/{os.path.basename(path)}"
    s3_client.upload_file(path, "my-bucket", key)
    os.remove(path)  # 로컬 파일 정리
    return f"https://my-bucket.s3.amazonaws.com/{key}"

# Chain: 다운로드 → 크기 조정 → 업로드
def process_user_avatar(image_url: str):
    workflow = chain(
        download_image.s(image_url),
        resize_image.s(200, 200),
        upload_to_s3.s(),
    )
    return workflow.apply_async()
```

### 리포트 생성

```javascript
// BullMQ 리포트 생성
import { Queue, Worker } from 'bullmq';

const reportQueue = new Queue('reports', { connection });

// 생산자: API 엔드포인트
app.post('/api/reports', async (req, res) => {
    const { type, dateRange, format } = req.body;

    const job = await reportQueue.add('generate', {
        type,
        dateRange,
        format: format || 'pdf',
        requestedBy: req.user.id,
    }, {
        attempts: 2,
        backoff: { type: 'fixed', delay: 30000 },
    });

    res.json({ jobId: job.id, status: 'queued' });
});

// 상태 엔드포인트
app.get('/api/reports/:jobId', async (req, res) => {
    const job = await reportQueue.getJob(req.params.jobId);
    if (!job) return res.status(404).json({ error: 'Not found' });

    const state = await job.getState();
    res.json({
        id: job.id,
        state,
        progress: job.progress,
        result: job.returnvalue,
    });
});

// 워커
const reportWorker = new Worker('reports', async (job) => {
    const { type, dateRange, format } = job.data;

    await job.updateProgress(10);
    const data = await fetchReportData(type, dateRange);

    await job.updateProgress(50);
    const file = await generateDocument(data, format);

    await job.updateProgress(90);
    const url = await uploadToStorage(file);

    await job.updateProgress(100);
    return { downloadUrl: url, generatedAt: new Date().toISOString() };
}, { connection, concurrency: 2 });
```

---

## 9. 오류 처리와 재시도 전략

### 재시도 전략

```python
# 고정 지연 재시도
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def fixed_retry_task(self, data):
    try:
        process(data)
    except TransientError as exc:
        raise self.retry(exc=exc)  # 60초 후 재시도

# 지수 백오프(Exponential backoff)
@app.task(bind=True, max_retries=5)
def exponential_retry_task(self, data):
    try:
        process(data)
    except TransientError as exc:
        countdown = 2 ** self.request.retries * 30  # 30초, 60초, 120초, 240초, 480초
        raise self.retry(exc=exc, countdown=countdown)

# 지터를 포함한 지수 백오프 (권장)
import random

@app.task(bind=True, max_retries=5)
def jittered_retry_task(self, data):
    try:
        process(data)
    except TransientError as exc:
        base_delay = 2 ** self.request.retries * 30
        jitter = random.uniform(0, base_delay * 0.1)
        raise self.retry(exc=exc, countdown=base_delay + jitter)
```

### BullMQ 백오프 전략

```javascript
// 지수 백오프
await queue.add('task', data, {
    attempts: 5,
    backoff: {
        type: 'exponential',
        delay: 1000,  // 1초, 2초, 4초, 8초, 16초
    },
});

// 고정 백오프
await queue.add('task', data, {
    attempts: 3,
    backoff: {
        type: 'fixed',
        delay: 5000,  // 재시도 간격 항상 5초
    },
});

// 커스텀 백오프 전략
await queue.add('task', data, {
    attempts: 5,
    backoff: {
        type: 'custom',
    },
});

// 워커에서 커스텀 백오프 구현
const worker = new Worker('queue', processor, {
    connection,
    settings: {
        backoffStrategy: (attemptsMade) => {
            // 커스텀: 10초, 30초, 90초, 270초, 810초
            return Math.pow(3, attemptsMade) * 10000;
        },
    },
});
```

### 데드레터 큐 패턴(Dead-Letter Queue Pattern)

```python
# Celery 데드레터 처리
@app.task(bind=True, max_retries=3)
def process_order(self, order_id: int):
    try:
        # ... 주문 처리 ...
        pass
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            # 최대 재시도 초과 — 데드레터 큐로 전송
            dead_letter_task.delay(
                original_task="process_order",
                args=[order_id],
                error=str(exc),
                retries=self.request.retries,
            )
            return {"status": "dead_lettered", "order_id": order_id}
        raise self.retry(exc=exc, countdown=60)

@app.task
def dead_letter_task(original_task: str, args: list, error: str, retries: int):
    """수동 검토를 위해 실패한 태스크를 저장한다."""
    logger.error(
        f"Dead letter: task={original_task}, args={args}, "
        f"error={error}, retries={retries}"
    )
    # 관리자 검토를 위해 데이터베이스에 저장
    FailedJob.objects.create(
        task_name=original_task,
        arguments=json.dumps(args),
        error_message=error,
        retry_count=retries,
    )
```

### 멱등성(Idempotency)

작업은 두 번 이상 실행될 수 있다(최소 한 번 전달, at-least-once delivery). 태스크를 멱등적으로 설계해야 한다:

```python
@app.task(bind=True)
def charge_payment(self, payment_id: str, amount: float):
    """멱등적 결제 처리."""
    # 이미 처리되었는지 확인 (멱등성 키)
    if redis_client.get(f"payment:processed:{payment_id}"):
        logger.info(f"Payment {payment_id} already processed, skipping")
        return {"status": "already_processed"}

    try:
        result = payment_gateway.charge(payment_id, amount)

        # TTL과 함께 처리 완료로 표시
        redis_client.setex(
            f"payment:processed:{payment_id}",
            86400 * 7,  # 7일
            json.dumps(result),
        )

        return {"status": "charged", "transaction_id": result.transaction_id}
    except PaymentError as exc:
        raise self.retry(exc=exc, countdown=30)
```

---

## 10. 신뢰성 패턴

### Celery 신뢰성 설정

두 가지 Celery 설정이 워커가 실행 중에 충돌할 때 태스크 손실을 방지한다:

```python
app.conf.update(
    # acks_late: 태스크가 반환된 후에 메시지를 확인(acknowledge)한다. 기본값(acks_early)은
    # 워커가 메시지를 픽업하는 순간 큐에서 제거한다. 워커가 완료 전에 죽으면 태스크가 사라진다.
    task_acks_late=True,

    # reject_on_worker_lost: 워커가 강제 종료(OOM, SIGKILL)되면
    # 태스크를 잃는 대신 재큐잉한다. acks_late=True가 필요하다.
    task_reject_on_worker_lost=True,

    # 늦은 확인(late-ack)이 많은 메시지를 보류하지 않도록 한 번에 하나의 태스크만 프리패치한다.
    worker_prefetch_multiplier=1,
)
```

이 두 설정을 함께 사용하면 **최소 한 번 전달(at-least-once delivery)**이 보장된다. 중복 실행을 안전하게 처리하려면 태스크를 멱등적으로 만들어야 한다(9절 참고).

### 분산 트랜잭션을 위한 Saga 패턴

작업이 여러 서비스에 걸쳐 있을 때(예: 결제 처리 → 재고 예약 → 이메일 발송), 단일 데이터베이스 트랜잭션은 서비스 경계를 넘을 수 없다. **Saga** 패턴은 단계를 조율하고 실패 시 보상(compensating) 작업을 실행한다.

**코레오그래피 기반 Saga(Choreography-based Saga)** (이벤트 기반):

```python
@app.task(bind=True, max_retries=3)
def saga_charge_payment(self, order_id: str):
    try:
        charge_payment(order_id)
        # 성공 시 다음 단계를 트리거
        saga_reserve_stock.delay(order_id)
    except PaymentError as exc:
        # 보상 불필요 — 아직 커밋된 것이 없음
        raise self.retry(exc=exc, countdown=30)

@app.task(bind=True, max_retries=3)
def saga_reserve_stock(self, order_id: str):
    try:
        reserve_stock(order_id)
        saga_send_confirmation.delay(order_id)
    except StockUnavailableError as exc:
        # 보상: 이미 성공한 결제를 환불
        saga_refund_payment.delay(order_id, reason="out_of_stock")
        raise

@app.task
def saga_refund_payment(order_id: str, reason: str):
    """saga_charge_payment의 보상 트랜잭션."""
    refund_payment(order_id)
    notify_customer(order_id, f"Order cancelled: {reason}")
```

각 단계는 성공 시 이벤트를 발행(또는 다음 태스크를 호출)하고, 실패 시 이미 완료된 단계를 되돌리기 위해 **보상 태스크**를 트리거한다. 보상 태스크도 재시도될 수 있으므로 멱등적으로 유지한다.

---

## 11. 연습 문제

### 연습 1: Celery를 이용한 이메일 큐

이메일 발송을 위한 Celery 애플리케이션을 구축하라:
- 세 가지 태스크 유형 정의: 환영 이메일, 비밀번호 재설정, 주문 확인
- 각 태스크 유형은 다른 템플릿을 사용
- 지터를 포함한 지수 백오프 재시도 구현
- 매일 오전 8시에 일일 요약을 보내는 주기적 태스크 추가
- Redis에 발송/실패 횟수 추적

```python
# 시작 코드
from celery import Celery

app = Celery("email_service", broker="redis://localhost:6379/0")

@app.task(bind=True, max_retries=3)
def send_welcome_email(self, user_id: int, email: str):
    # TODO: 재시도 로직과 함께 구현
    pass

@app.task
def send_daily_digest():
    # TODO: 요약 콘텐츠를 수집하고 모든 구독자에게 전송
    pass

# TODO: 주기적 태스크를 위한 beat_schedule 설정
# TODO: 데드레터 처리 추가
```

### 연습 2: BullMQ를 이용한 이미지 처리 파이프라인

Node.js 이미지 처리 서비스를 구축하라:
- 큐: 처리를 위한 이미지 URL 수신
- 워커: 다운로드, 3가지 크기(썸네일, 중간, 대형)로 조정, 업로드
- 플로우: 다운로드 → 크기 조정 → 업로드 체인에 FlowProducer 사용
- 진행률: 각 단계별 백분율 보고
- 대시보드: `/admin/queues`에 Bull Board 설정

```javascript
// 시작 코드
import { Queue, Worker, FlowProducer } from 'bullmq';

const imageQueue = new Queue('images', { connection });

// TODO: 'download', 'resize', 'upload' 작업 이름을 처리하는 워커 정의
// TODO: 체인 처리를 위한 FlowProducer 설정
// TODO: Bull Board 대시보드 추가
// TODO: 진행률 보고 구현
```

### 연습 3: 우선순위 태스크 스케줄러

태스크 스케줄링 시스템을 설계하라:
- 세 가지 우선순위 레벨: 긴급(P1), 보통(P2), 낮음(P3)
- 긴급 태스크는 즉시 처리
- 보통 태스크는 최대 10 동시성으로 처리
- 낮은 태스크는 더 높은 우선순위 태스크가 대기하지 않을 때만 처리
- 큐 깊이와 처리 속도를 보여주는 대시보드 엔드포인트 구현
- 모든 우선순위에 걸쳐 1000개 태스크를 제출하는 부하 테스트 작성

### 연습 4: 신뢰할 수 있는 주문 처리

보장된 전달을 가진 주문 처리 시스템을 구축하라:
- 주문 제출 엔드포인트가 큐에 작업을 추가
- 워커가 주문을 처리: 재고 확인, 결제 처리, 재고 업데이트
- 결제 단계를 멱등적으로 만들기 (멱등성 키 사용)
- 3번 재시도 후 실패하는 주문을 위한 데드레터 큐 구현
- 데드레터된 주문을 조회하고 재시도하는 관리자 엔드포인트 구축
- 모니터링 추가: 처리 시간, 성공률, 큐 깊이 추적

```python
# 시작 코드
@app.task(bind=True, max_retries=3)
def process_order(self, order_id: str):
    """
    단계:
    1. 재고 가용성 확인
    2. 결제 처리 (멱등적이어야 함)
    3. 재고 업데이트
    4. 확인 이메일 발송
    """
    # TODO: 각 단계 구현
    # TODO: 부분 실패 처리 (예: 결제는 되었으나 재고 업데이트 실패)
    # TODO: 최대 재시도 시 데드레터 큐로 전송
    pass
```

---

## 참고 자료

- [Celery Documentation](https://docs.celeryq.dev/)
- [BullMQ Documentation](https://docs.bullmq.io/)
- [RabbitMQ Tutorials](https://www.rabbitmq.com/tutorials)
- [Flower (Celery Monitor)](https://flower.readthedocs.io/)
- [Bull Board](https://github.com/felixmosh/bull-board)
- [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/)
- [Designing Data-Intensive Applications (Chapter 11: Stream Processing)](https://dataintensive.net/)

---

**이전**: [Redis 캐싱 패턴](./20_Redis_Caching_Patterns.md)
