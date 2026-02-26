# Kafka 스트리밍

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Apache Kafka의 아키텍처(브로커, 토픽, 파티션, 프로듀서, 컨슈머, 컨슈머 그룹)를 설명하고, 복제(Replication)가 내결함성(Fault Tolerance)을 보장하는 방식을 기술할 수 있다
2. Python으로 Kafka 프로듀서(Producer)를 구현하여 직렬화(Serialization) 및 파티셔닝 전략과 함께 메시지를 발행할 수 있다
3. 적절한 오프셋(Offset) 관리와 컨슈머 그룹 협력을 갖춘 Kafka 컨슈머(Consumer)를 구현할 수 있다
4. 처리량(Throughput), 지연 시간(Latency), 전달 보장(at-most-once, at-least-once, exactly-once)을 조정하기 위한 핵심 Kafka 파라미터를 설정할 수 있다
5. Kafka를 Spark Structured Streaming과 통합하여 엔드투엔드(end-to-end) 실시간 데이터 파이프라인을 구축할 수 있다
6. 토픽 설계, 파티션 수 결정, 컨슈머 그룹 설정을 포함하여 실제 사용 사례에 대한 Kafka 기반 스트리밍 아키텍처를 설계할 수 있다

---

## 개요

Apache Kafka는 분산 이벤트 스트리밍 플랫폼으로, 실시간 데이터 파이프라인과 스트리밍 애플리케이션 구축에 사용됩니다. 높은 처리량과 내결함성을 제공합니다.

---

## 1. Kafka 개요

### 1.1 Kafka 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                      Kafka Architecture                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Producers                         Consumers                    │
│   ┌─────────┐ ┌─────────┐          ┌─────────┐ ┌─────────┐      │
│   │Producer1│ │Producer2│          │Consumer1│ │Consumer2│      │
│   └────┬────┘ └────┬────┘          └────┬────┘ └────┬────┘      │
│        │           │                    │           │            │
│        └─────┬─────┘                    └─────┬─────┘            │
│              ↓                                ↑                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    Kafka Cluster                          │  │
│   │  ┌──────────────────────────────────────────────────────┐│  │
│   │  │                    Topic: orders                      ││  │
│   │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐       ││  │
│   │  │  │Partition 0 │ │Partition 1 │ │Partition 2 │       ││  │
│   │  │  │ [0,1,2,3]  │ │ [0,1,2]    │ │ [0,1,2,3,4]│       ││  │
│   │  │  └────────────┘ └────────────┘ └────────────┘       ││  │
│   │  └──────────────────────────────────────────────────────┘│  │
│   │                                                          │  │
│   │  Broker 1         Broker 2         Broker 3              │  │
│   │  ┌──────────┐    ┌──────────┐    ┌──────────┐           │  │
│   │  │ P0(L)    │    │ P1(L)    │    │ P2(L)    │           │  │
│   │  │ P1(R)    │    │ P2(R)    │    │ P0(R)    │           │  │
│   │  └──────────┘    └──────────┘    └──────────┘           │  │
│   │                   L=Leader, R=Replica                    │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              ↑                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    ZooKeeper / KRaft                      │  │
│   │             (클러스터 메타데이터 관리)                      │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 개념

| 개념 | 설명 |
|------|------|
| **Broker** | Kafka 서버, 메시지 저장/전달 |
| **Topic** | 메시지 카테고리 (논리적 채널) |
| **Partition** | Topic의 물리적 분할, 병렬 처리 |
| **Producer** | 메시지 발행자 |
| **Consumer** | 메시지 소비자 |
| **Consumer Group** | 협력하여 소비하는 Consumer 그룹 |
| **Offset** | 파티션 내 메시지 위치 |
| **Replication** | 파티션 복제로 내결함성 확보 |

---

## 2. 설치 및 설정

### 2.1 Docker Compose 설정

```yaml
# docker-compose.yaml — 로컬 개발용 단일 브로커 설정.
# 프로덕션 클러스터는 복제 및 내결함성을 위해 3개 이상의 브로커를 사용한다.
version: '3'

services:
  # ZooKeeper는 브로커 메타데이터와 리더 선출을 관리한다. Kafka 3.3+는
  # KRaft 모드(ZooKeeper 없음)를 지원한다 — 운영 복잡성 감소를 위해 신규 클러스터에 KRaft를 고려한다.
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000     # ZK 세션 타임아웃의 기본 단위 — 2초가 표준 기본값
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092  # 클라이언트가 연결에 사용하는 주소 — Docker 외부에서 도달 가능해야 한다
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1   # 단일 브로커 개발 환경에서는 1로 설정; 프로덕션에서는 내구성을 위해 3을 사용한다
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"     # 개발에 편리하다; 프로덕션에서는 토픽 거버넌스를 강제하기 위해 비활성화한다

  # 토픽, 컨슈머, 지연(lag)을 모니터링하기 위한 웹 UI — 프로덕션에는 필요 없다
  # (Confluent Control Center나 Grafana + JMX 메트릭 사용)
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092  # localhost가 아닌 내부 Docker 네트워크 이름
```

```bash
# 실행
docker-compose up -d

# 토픽 생성 (컨테이너 내부에서)
docker exec -it kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic my-topic \
    --partitions 3 \
    --replication-factor 1
```

### 2.2 Python 클라이언트 설치

```bash
# confluent-kafka (권장)
pip install confluent-kafka

# kafka-python (대안)
pip install kafka-python
```

---

## 3. Topic과 Partition

### 3.1 Topic 관리

```bash
# 토픽 생성
kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic orders \
    --partitions 6 \
    --replication-factor 3

# 토픽 목록
kafka-topics --list --bootstrap-server localhost:9092

# 토픽 상세 정보
kafka-topics --describe \
    --bootstrap-server localhost:9092 \
    --topic orders

# 토픽 삭제
kafka-topics --delete \
    --bootstrap-server localhost:9092 \
    --topic orders

# 파티션 수 증가 (축소 불가)
kafka-topics --alter \
    --bootstrap-server localhost:9092 \
    --topic orders \
    --partitions 12
```

### 3.2 Partition 전략

```python
"""
파티션 선택 전략:
1. Key가 있으면: hash(key) % partitions — 동일한 키가 항상 동일한 파티션으로 가는 것을 보장하여
   키당 순서 보존 (이벤트 소싱에 중요하다).
2. Key가 없으면: Round-robin — 균등하게 분산되지만 순서 보장 없음.

파티션 수 결정 요소:
- 예상 처리량 / 단일 파티션 처리량
- Consumer 수 (파티션 >= Consumer — 초과 컨슈머는 유휴 상태)
- 디스크 I/O 고려
"""

# 파티션 수 권장
"""
- 파티션당 100MB/s 처리 가정 (현대 SSD에 대한 보수적 추정)
- 1GB/s 처리 필요 → 최소 10개 파티션
- Consumer 확장성 고려 → 예상 Consumer 수의 2-3배 (성장 여유)

주의:
- 너무 많은 파티션 → 더 많은 열린 파일 핸들, 브로커 장애 시 느린 리더 선출,
  증가된 엔드투엔드 지연 (각 파티션이 오버헤드를 추가한다)
- 너무 적은 파티션 → 컨슈머 병렬성이 제한되어 파티션 수를 넘어 확장할 수 없다
- 파티션은 나중에 늘릴 수 있지만 줄일 수는 없다 — 성장을 위해 계획한다
"""
```

---

## 4. Producer

### 4.1 기본 Producer

```python
from confluent_kafka import Producer
import json

# Producer 설정
config = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'my-producer',
    'acks': 'all',  # 모든 인-싱크 레플리카가 확인할 때까지 대기 — 가장 강력한 내구성 보장.
                     # 쓰기 직후 리더가 충돌해도 데이터 손실을 방지한다.
                     # 트레이드오프: 더 높은 지연 (~1ms acks=1 대비 ~5-10ms).
}

producer = Producer(config)

# 비동기 배달 콜백 — produce()는 논블로킹이다; 이 콜백은 브로커가
# 메시지를 승인(또는 거부)할 때 발생한다. 조용한 실패를 감지하는 데 필수적이다.
def delivery_callback(err, msg):
    if err:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] @ {msg.offset()}')

# 메시지 키가 파티션 배치를 결정한다 — 동일한 키를 가진 모든 메시지는
# 동일한 파티션으로 가서 해당 키에 대한 순서를 보존한다 (예: order-123의 모든 이벤트가 순서대로 도착한다).
def send_message(topic: str, key: str, value: dict):
    producer.produce(
        topic=topic,
        key=key.encode('utf-8'),       # 키는 바이트여야 한다 — 일관된 파티셔닝을 위해 안정적인 인코딩 사용
        value=json.dumps(value).encode('utf-8'),
        callback=delivery_callback
    )
    # flush()는 모든 버퍼된 메시지가 전달될 때까지 블로킹한다 — 단일 메시지 스크립트에 사용한다.
    # 고처리량의 경우, 대신 주기적으로 poll()을 호출한다.
    producer.flush()

# 사용 예시
send_message(
    topic='orders',
    key='order-123',
    value={
        'order_id': 'order-123',
        'customer_id': 'cust-456',
        'amount': 99.99,
        'timestamp': '2024-01-15T10:30:00Z'
    }
)
```

### 4.2 고성능 Producer

```python
from confluent_kafka import Producer
import json
import time

class HighThroughputProducer:
    """고처리량 Producer"""

    def __init__(self, bootstrap_servers: str):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'high-throughput-producer',

            # 성능 설정 — 각 선택은 처리량, 지연, 내구성 간의 트레이드오프다
            'acks': '1',                    # 리더 전용 ack: 빠르지만 리더가 복제 전에 충돌하면 데이터 손실 위험.
                                            # 중요한 데이터(금융, 청구)에는 'all'을 사용한다.
            'linger.ms': 5,                 # Producer는 메시지를 전송하기 전에 최대 5ms 대기 — 처리량을 높인다
                                            # (더 큰 배치 = 적은 네트워크 왕복)하지만 5ms 추가 지연이 발생한다. 최소 지연에는 0으로 설정한다.
            'batch.size': 16384,            # 최대 배치 크기 (16KB) — 더 큰 배치는 압축과 처리량을 개선하지만
                                            # 파티션당 더 많은 메모리를 소비한다. linger.ms와 함께 조정한다.
            'buffer.memory': 33554432,      # 전송되지 않은 메시지를 위한 총 32MB 버퍼 — 버퍼가 가득 차면
                                            # (느린 브로커) produce()가 블로킹된다. 버스티한 워크로드에서 늘린다.
            'compression.type': 'snappy',   # Snappy는 최소 CPU 오버헤드로 ~2배 압축을 제공한다 — 네트워크
                                            # 대역폭과 브로커 디스크 사용량을 줄인다. 더 빠른 압축에는 'lz4',
                                            # 콜드 데이터에서 최상의 비율을 위해 'zstd'를 사용한다.

            # 재시도 설정
            'retries': 3,
            'retry.backoff.ms': 100,
        }
        self.producer = Producer(self.config)
        self.message_count = 0

    def send(self, topic: str, key: str, value: dict):
        """비동기 전송"""
        self.producer.produce(
            topic=topic,
            key=key.encode('utf-8') if key else None,  # None 키 = 라운드로빈 파티션 배정
            value=json.dumps(value).encode('utf-8'),
            callback=self._on_delivery
        )
        self.message_count += 1

        # poll(0)은 블로킹 없이 콜백 처리를 트리거한다 — 배달 보고서 큐를 비우기 위해
        # 주기적으로 호출해야 한다. 없으면 내부 큐가 가득 차서 produce()가 결국 블로킹되거나 BufferError를 발생시킨다.
        if self.message_count % 1000 == 0:
            self.producer.poll(0)

    def _on_delivery(self, err, msg):
        if err:
            print(f'Delivery failed: {err}')

    def flush(self):
        """모든 메시지 전송 완료 대기"""
        self.producer.flush()

    def close(self):
        self.flush()


# 대량 전송 예시
producer = HighThroughputProducer('localhost:9092')

start = time.time()
for i in range(100000):
    producer.send(
        topic='events',
        key=f'key-{i % 100}',
        value={'event_id': i, 'data': 'test'}
    )

producer.flush()
print(f'Sent 100,000 messages in {time.time() - start:.2f} seconds')
```

---

## 5. Consumer

### 5.1 기본 Consumer

```python
from confluent_kafka import Consumer
import json

# Consumer 설정
config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-consumer-group',   # 동일한 group.id를 가진 모든 컨슈머가 파티션 부하를 공유한다
    'auto.offset.reset': 'earliest',   # 첫 번째 참여 시(커밋된 오프셋 없음) 처음부터 시작한다.
                                        # 기존 데이터를 건너뛰고 새 메시지만 처리하려면 'latest'를 사용한다.
    'enable.auto.commit': True,        # 오프셋이 auto.commit.interval.ms마다 커밋된다 — 단순하지만
                                        # 충돌 후 처리 중복 위험이 있다 (메시지가 처리되었지만
                                        # 오프셋이 아직 커밋되지 않은 경우). 정확히 한 번을 위해 수동 커밋을 사용한다.
    'auto.commit.interval.ms': 5000,   # 5초 간격 — 낮을수록 중복 위험이 줄지만 브로커 부하가 증가한다
}

consumer = Consumer(config)

# subscribe()는 컨슈머 그룹 리밸런싱을 트리거한다 — 파티션이 그룹의 모든 컨슈머에
# 분배된다. 컨슈머 추가/제거가 리밸런싱을 트리거한다.
consumer.subscribe(['orders'])

# poll()은 메인 루프다 — 메시지를 가져오고, 하트비트를 전송하고, 리밸런싱 콜백을 트리거한다.
# 타임아웃은 새 메시지를 기다리는 시간을 결정한다 (None 반환은 오류가 아니다).
try:
    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            print(f'Consumer error: {msg.error()}')
            continue

        # 키/값을 디코딩한다 — Kafka는 원시 바이트를 저장한다; 직렬화 형식은
        # 프로듀서와 컨슈머 간의 계약이다 (여기서는 JSON, 하지만 프로덕션에서는
        # 스키마 강제를 위해 Avro/Protobuf가 선호된다).
        key = msg.key().decode('utf-8') if msg.key() else None
        value = json.loads(msg.value().decode('utf-8'))

        print(f'Received: topic={msg.topic()}, partition={msg.partition()}, '
              f'offset={msg.offset()}, key={key}, value={value}')

except KeyboardInterrupt:
    pass
finally:
    # close()는 최종 오프셋을 커밋하고 컨슈머 그룹에서 깔끔하게 탈퇴한다 —
    # 없으면 브로커가 session.timeout.ms 동안 기다린 후 리밸런싱한다
    consumer.close()
```

### 5.2 수동 커밋

```python
from confluent_kafka import Consumer
import json

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'manual-commit-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,  # 수동 커밋 = 최소 한 번 전달: 메시지는 성공적인 처리 후에만
                                   # 커밋된다. 컨슈머가 충돌하면 커밋되지 않은 메시지가 재전달된다 —
                                   # 처리 로직이 이를 안전하게 처리하려면 멱등성이 있어야 한다.
}

consumer = Consumer(config)
consumer.subscribe(['orders'])

def process_message(value: dict) -> bool:
    """메시지 처리 로직"""
    try:
        # 실제 비즈니스 로직 — 컨슈머가 커밋 전에 충돌할 경우 동일한 메시지가
        # 다시 전달될 수 있으므로 멱등성이 있어야 한다
        print(f"Processing: {value}")
        return True
    except Exception as e:
        print(f"Processing failed: {e}")
        return False

try:
    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue
        if msg.error():
            continue

        value = json.loads(msg.value().decode('utf-8'))

        # 처리 후 커밋 패턴: 처리가 성공할 때만 오프셋을 전진시킨다.
        # 실패 시 메시지는 커밋되지 않고 다음 poll이나 리밸런싱 후 재전달된다.
        if process_message(value):
            consumer.commit(msg)  # 이 특정 오프셋을 커밋한다 — consumer.commit()보다 정확하다
            # consumer.commit()은 이 시점까지의 모든 소비된 오프셋을 커밋했을 것이다
        else:
            # 커밋하지 않으면 재시작 시 이 메시지가 재처리된다.
            # 항상 실패하는 독성 메시지의 경우, 무한 루프를 피하기 위해 데드 레터 큐를 추가한다.
            print("Message processing failed, not committing")

except KeyboardInterrupt:
    pass
finally:
    consumer.close()
```

---

## 6. Consumer Group

### 6.1 Consumer Group 개념

```
┌────────────────────────────────────────────────────────────────┐
│                    Consumer Group 동작                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Topic: orders (6 partitions)                                 │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
│   │ P0   │ │ P1   │ │ P2   │ │ P3   │ │ P4   │ │ P5   │      │
│   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘      │
│      │        │        │        │        │        │           │
│   Consumer Group A (3 consumers)                               │
│      │        │        │        │        │        │           │
│      ↓        ↓        ↓        ↓        ↓        ↓           │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│   │ Consumer 1  │  │ Consumer 2  │  │ Consumer 3  │          │
│   │  P0, P1     │  │  P2, P3     │  │  P4, P5     │          │
│   └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                                │
│   각 파티션은 그룹 내 하나의 Consumer에만 할당                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 리밸런싱

```python
from confluent_kafka import Consumer

def on_assign(consumer, partitions):
    """파티션 할당 콜백"""
    print(f"Partitions assigned: {[p.partition for p in partitions]}")

def on_revoke(consumer, partitions):
    """파티션 해제 콜백"""
    print(f"Partitions revoked: {[p.partition for p in partitions]}")
    # 처리 중인 메시지 커밋
    consumer.commit()

config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest',
    'partition.assignment.strategy': 'cooperative-sticky',  # 점진적 리밸런싱
}

consumer = Consumer(config)
consumer.subscribe(
    ['orders'],
    on_assign=on_assign,
    on_revoke=on_revoke
)
```

### 6.3 Consumer Group 모니터링

```bash
# Consumer Group 목록
kafka-consumer-groups --list --bootstrap-server localhost:9092

# Consumer Group 상세
kafka-consumer-groups --describe \
    --bootstrap-server localhost:9092 \
    --group my-consumer-group

# 출력 예시:
# GROUP           TOPIC    PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# my-group        orders   0          1500            1550            50
# my-group        orders   1          1200            1200            0

# Lag 모니터링 (처리 지연)
kafka-consumer-groups --describe \
    --bootstrap-server localhost:9092 \
    --group my-consumer-group \
    --members
```

---

## 7. 실시간 데이터 처리 패턴

### 7.1 이벤트 기반 처리

```python
from confluent_kafka import Consumer, Producer
import json

class EventProcessor:
    """이벤트 기반 처리 파이프라인"""

    def __init__(self, bootstrap_servers: str, group_id: str):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        })
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
        })

    def process_and_forward(
        self,
        source_topic: str,
        target_topic: str,
        transform_func
    ):
        """메시지 처리 후 다른 토픽으로 전달"""
        self.consumer.subscribe([source_topic])

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue

                # 변환
                value = json.loads(msg.value().decode('utf-8'))
                transformed = transform_func(value)

                if transformed:
                    # 다음 토픽으로 전달
                    self.producer.produce(
                        topic=target_topic,
                        key=msg.key(),
                        value=json.dumps(transformed).encode('utf-8')
                    )
                    self.producer.poll(0)

                # 커밋
                self.consumer.commit(msg)

        except KeyboardInterrupt:
            pass
        finally:
            self.producer.flush()
            self.consumer.close()


# 사용 예시: 주문 → 배송 이벤트 변환
def order_to_shipment(order: dict) -> dict:
    """주문 이벤트를 배송 이벤트로 변환"""
    return {
        'shipment_id': f"ship-{order['order_id']}",
        'order_id': order['order_id'],
        'customer_id': order['customer_id'],
        'status': 'pending',
        'created_at': order['timestamp']
    }

processor = EventProcessor('localhost:9092', 'order-processor')
processor.process_and_forward('orders', 'shipments', order_to_shipment)
```

### 7.2 집계 처리 (Windowing)

```python
from confluent_kafka import Consumer
from collections import defaultdict
from datetime import datetime, timedelta
import json
import threading
import time

class WindowedAggregator:
    """시간 윈도우 기반 집계"""

    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        self.windows = defaultdict(lambda: defaultdict(int))
        self.lock = threading.Lock()

    def add(self, key: str, value: int, timestamp: datetime):
        """값 추가"""
        window_start = self._get_window_start(timestamp)
        with self.lock:
            self.windows[window_start][key] += value

    def _get_window_start(self, timestamp: datetime) -> datetime:
        """윈도우 시작 시간 계산"""
        seconds = int(timestamp.timestamp())
        window_start_seconds = (seconds // self.window_size) * self.window_size
        return datetime.fromtimestamp(window_start_seconds)

    def get_and_clear_completed_windows(self) -> dict:
        """완료된 윈도우 결과 반환"""
        current_window = self._get_window_start(datetime.now())
        completed = {}

        with self.lock:
            for window_start, data in list(self.windows.items()):
                if window_start < current_window:
                    completed[window_start] = dict(data)
                    del self.windows[window_start]

        return completed


# 사용 예시: 분당 카테고리별 판매 수 집계
aggregator = WindowedAggregator(window_size_seconds=60)

def process_sales():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'sales-aggregator',
        'auto.offset.reset': 'earliest',
    })
    consumer.subscribe(['sales'])

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg and not msg.error():
            value = json.loads(msg.value().decode('utf-8'))
            aggregator.add(
                key=value['category'],
                value=1,
                timestamp=datetime.fromisoformat(value['timestamp'])
            )

        # 완료된 윈도우 출력
        completed = aggregator.get_and_clear_completed_windows()
        for window, data in completed.items():
            print(f"Window {window}: {data}")
```

---

## 8. Kafka Streams와 대안

### 8.1 Faust (Python Kafka Streams)

```python
import faust

# Faust 앱 생성
app = faust.App(
    'myapp',
    broker='kafka://localhost:9092',
    value_serializer='json',
)

# 토픽 정의
orders_topic = app.topic('orders', value_type=dict)
processed_topic = app.topic('processed_orders', value_type=dict)

# 스트림 처리 에이전트
@app.agent(orders_topic)
async def process_orders(orders):
    async for order in orders:
        # 처리 로직
        processed = {
            **order,
            'processed': True,
            'processed_at': str(datetime.now())
        }
        # 다른 토픽으로 전송
        await processed_topic.send(value=processed)

# 테이블 (상태 저장)
order_counts = app.Table('order_counts', default=int)

@app.agent(orders_topic)
async def count_orders(orders):
    async for order in orders:
        customer_id = order['customer_id']
        order_counts[customer_id] += 1

# 실행: faust -A myapp worker
```

---

## 연습 문제

### 문제 1: Producer/Consumer
주문 이벤트를 생성하는 Producer와 소비하는 Consumer를 작성하세요.

### 문제 2: Consumer Group
3개의 Consumer로 구성된 Consumer Group을 만들고 파티션 할당을 확인하세요.

### 문제 3: 실시간 집계
실시간 판매 이벤트에서 분당 총 매출을 계산하는 스트리밍 애플리케이션을 작성하세요.

---

## 요약

| 개념 | 설명 |
|------|------|
| **Topic** | 메시지의 논리적 카테고리 |
| **Partition** | Topic의 물리적 분할, 병렬 처리 단위 |
| **Producer** | 메시지 발행자 |
| **Consumer** | 메시지 소비자 |
| **Consumer Group** | 협력적으로 소비하는 Consumer 집합 |
| **Offset** | 파티션 내 메시지 위치 |

---

## 참고 자료

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Kafka Python](https://docs.confluent.io/kafka-clients/python/current/overview.html)
- [Kafka: The Definitive Guide](https://www.oreilly.com/library/view/kafka-the-definitive/9781491936153/)
