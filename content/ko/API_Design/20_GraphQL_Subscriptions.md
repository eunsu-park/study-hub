# 20. GraphQL 서브스크립션(GraphQL Subscriptions)

**이전**: [GraphQL 리졸버](./19_GraphQL_Resolvers.md) | **다음**: [GraphQL 서버 구현](./21_GraphQL_Server_Implementation.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- GraphQL 서브스크립션이 클라이언트에 실시간 데이터를 전달하는 방식을 설명할 수 있다
- Python에서 비동기 제너레이터를 사용하여 서브스크립션 리졸버를 구현할 수 있다
- GraphQL 서브스크립션 연결을 위한 WebSocket 전송을 구성할 수 있다
- 인메모리 및 Redis 기반 브로커로 발행/구독 패턴을 적용할 수 있다
- 서브스크립션 생명주기 이벤트(연결, 킵얼라이브, 종료)를 처리할 수 있다
- 프로덕션에서 효율적이고 안전한 서브스크립션 스키마를 설계할 수 있다

---

## 목차

1. [서브스크립션이란?](#1-서브스크립션이란)
2. [서브스크립션 vs. 폴링 vs. SSE](#2-서브스크립션-vs-폴링-vs-sse)
3. [서브스크립션 스키마 설계](#3-서브스크립션-스키마-설계)
4. [WebSocket 전송](#4-websocket-전송)
5. [Strawberry에서 서브스크립션 구현](#5-strawberry에서-서브스크립션-구현)
6. [Pub/Sub 패턴](#6-pubsub-패턴)
7. [Redis 기반 Pub/Sub](#7-redis-기반-pubsub)
8. [서브스크립션 생명주기](#8-서브스크립션-생명주기)
9. [필터링과 권한 부여](#9-필터링과-권한-부여)
10. [프로덕션 고려사항](#10-프로덕션-고려사항)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. 서브스크립션이란?

서브스크립션은 실시간 서버 푸시 업데이트를 가능하게 하는 GraphQL 연산 타입입니다. 쿼리와 뮤테이션이 요청-응답 패턴을 따르는 반면, 서브스크립션은 클라이언트와 서버 간에 지속적인 연결을 유지합니다.

### 세 가지 연산 타입

| 연산 | 방향 | 연결 | 사용 사례 |
|------|------|------|----------|
| Query | 클라이언트 → 서버 → 클라이언트 | 단기 | 데이터 읽기 |
| Mutation | 클라이언트 → 서버 → 클라이언트 | 단기 | 데이터 쓰기 |
| Subscription | 클라이언트 → 서버 ← 서버 (지속적) | 장기 | 실시간 업데이트 |

### 동작 방식

```
1. 클라이언트가 WebSocket을 통해 서브스크립션 연산 전송
2. 서버가 확인하고 이벤트 수신 대기 시작
3. 이벤트 발생 시 서버가 클라이언트에 데이터 푸시
4. 클라이언트가 구독 취소하거나 연결이 끊길 때까지 연결 유지

Client                         Server
  |--- subscribe(postCreated) -->|
  |<-- ack ----------------------|
  |                              |  (새 게시글 생성)
  |<-- data: { postCreated }  ---|
  |                              |  (다른 게시글)
  |<-- data: { postCreated }  ---|
  |--- unsubscribe ------------->|
  |<-- complete -----------------|
```

### 일반적인 사용 사례

- **채팅 메시지**: 대화의 새 메시지
- **알림**: 사용자 활동에 대한 실시간 알림
- **라이브 대시보드**: 메트릭, 주가, 점수
- **협업 편집**: 문서 변경, 커서 위치
- **주문 추적**: 배달 상태 업데이트
- **IoT 데이터 스트림**: 센서 판독값, 디바이스 상태

---

## 2. 서브스크립션 vs. 폴링 vs. SSE

### 비교

| 접근법 | 지연시간 | 복잡도 | 양방향 | 효율성 |
|--------|---------|--------|--------|--------|
| **폴링** | 높음 (간격에 묶임) | 낮음 | 아니오 | 낮음 (빈 응답 많음) |
| **롱 폴링** | 중간 | 중간 | 아니오 | 중간 |
| **SSE** | 낮음 | 낮음 | 아니오 | 높음 |
| **WebSocket** | 낮음 | 높음 | 예 | 높음 |
| **GraphQL 서브스크립션** | 낮음 | 중간 | 예 (WS 통해) | 높음 |

### 사용 시점

```
폴링:
  - 업데이트 빈도 > 30초
  - 단순한 구현이 필요할 때
  - HTTP 캐싱이 유익할 때

SSE (Server-Sent Events):
  - 서버에서 클라이언트로만
  - 단순한 text/event-stream
  - 자동 재연결 내장
  - HTTP 프록시를 통해 작동

WebSocket / GraphQL 서브스크립션:
  - 실시간 양방향 필요
  - 복잡한 이벤트 필터링
  - 연결당 다중 이벤트 스트림
  - 초 미만의 지연시간 필요
```

### SSE 대안(SSE Alternative)

더 단순한 경우라면 서브스크립션 대신 SSE를 고려하세요:

```python
# FastAPI SSE endpoint (simpler than subscriptions)
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio

app = FastAPI()

@app.get("/events/posts")
async def post_events():
    async def event_generator():
        while True:
            post = await post_queue.get()
            yield {
                "event": "new_post",
                "data": json.dumps({"id": post.id, "title": post.title}),
            }
    return EventSourceResponse(event_generator())
```

---

## 3. 서브스크립션 스키마 설계

### 기본 스키마

```graphql
type Subscription {
  postCreated: Post!
  postUpdated(id: ID): Post!
  commentAdded(postId: ID!): Comment!
  notificationReceived: Notification!
}
```

### 필터링된 서브스크립션

```graphql
type Subscription {
  messageAdded(channelId: ID!): Message!
  orderStatusChanged(orderId: ID!): OrderStatusEvent!
  priceUpdated(symbol: String!): PriceEvent!
}

type OrderStatusEvent {
  order: Order!
  previousStatus: OrderStatus!
  newStatus: OrderStatus!
  timestamp: DateTime!
}
```

### 이벤트 페이로드 설계

컨텍스트가 포함된 풍부한 페이로드를 반환합니다:

```graphql
type Subscription {
  chatEvent(roomId: ID!): ChatEvent!
}

union ChatEvent = MessageSent | MessageEdited | MessageDeleted
                | UserJoined | UserLeft | UserTyping

type MessageSent {
  message: Message!
  room: ChatRoom!
}

type UserTyping {
  user: User!
  room: ChatRoom!
  isTyping: Boolean!
}
```

---

## 4. WebSocket 전송

### graphql-ws 프로토콜

`graphql-ws` 프로토콜(레거시 `subscriptions-transport-ws`의 후속)이 현대적 표준입니다.

### 프로토콜 메시지 흐름

```
Client                            Server
  |--- ConnectionInit { payload } -->|
  |<-- ConnectionAck ----------------|
  |                                  |
  |--- Subscribe { id, payload }  -->|
  |<-- Next { id, payload }       ---|  (이벤트 1)
  |<-- Next { id, payload }       ---|  (이벤트 2)
  |--- Complete { id }            -->|  (구독 취소)
  |                                  |
  |--- Ping                       -->|  (킵얼라이브)
  |<-- Pong                       ---|
```

### 메시지 타입(Message Types)

| 방향 | 메시지 | 목적 |
|------|--------|------|
| C → S | `ConnectionInit` | 연결 초기화 (인증 페이로드) |
| S → C | `ConnectionAck` | 연결 수락 |
| C → S | `Subscribe` | 서브스크립션 시작 |
| S → C | `Next` | 서브스크립션 데이터 |
| S → C | `Error` | 서브스크립션 오류 |
| S → C | `Complete` | 서브스크립션 종료 |
| C → S | `Complete` | 클라이언트 구독 취소 |
| 양방향 | `Ping` / `Pong` | 킵얼라이브 |

### 클라이언트 JavaScript

```javascript
import { createClient } from 'graphql-ws';

const client = createClient({
  url: 'ws://localhost:8000/graphql',
  connectionParams: {
    authToken: 'Bearer eyJhbGciOi...',
  },
  // Automatic reconnection
  retryAttempts: 5,
  retryWait: async (retries) => {
    await new Promise(resolve =>
      setTimeout(resolve, Math.min(1000 * 2 ** retries, 30000))
    );
  },
});

// Subscribe
const unsubscribe = client.subscribe(
  {
    query: `
      subscription OnPostCreated {
        postCreated {
          id
          title
          author { username }
        }
      }
    `,
  },
  {
    next: (data) => {
      console.log('New post:', data.data.postCreated);
    },
    error: (err) => {
      console.error('Subscription error:', err);
    },
    complete: () => {
      console.log('Subscription complete');
    },
  }
);

// Later: unsubscribe
unsubscribe();
```

---

## 5. Strawberry에서 서브스크립션 구현

### 비동기 제너레이터를 이용한 기본 서브스크립션

```python
import strawberry
import asyncio
from typing import AsyncGenerator


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def count(self, target: int = 10) -> AsyncGenerator[int, None]:
        """0부터 target까지 매초 하나의 값을 방출합니다."""
        for i in range(target):
            yield i
            await asyncio.sleep(1)

    @strawberry.subscription
    async def post_created(self) -> AsyncGenerator["Post", None]:
        """새 게시글이 생성될 때 이벤트를 방출합니다."""
        async for event in post_event_bus.subscribe("post.created"):
            yield event.payload
```

### FastAPI + WebSocket 설정

```python
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)

graphql_router = GraphQLRouter(
    schema,
    context_getter=get_context,
    subscription_protocols=[GRAPHQL_TRANSPORT_WS_PROTOCOL],
)

app = FastAPI()
app.include_router(graphql_router, prefix="/graphql")
```

### 컨텍스트를 활용한 서브스크립션(Subscription with Context)

```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def notifications(
        self, info: strawberry.types.Info
    ) -> AsyncGenerator["Notification", None]:
        """인증된 사용자의 알림을 방출합니다."""
        user = info.context.current_user
        if user is None:
            raise PermissionError("Authentication required")

        channel = f"user.{user.id}.notifications"
        async for event in info.context.pubsub.subscribe(channel):
            yield Notification(
                id=event["id"],
                message=event["message"],
                type=event["type"],
                created_at=event["created_at"],
            )
```

---

## 6. Pub/Sub 패턴

### 인메모리 Pub/Sub

단일 서버 배포에 적합합니다:

```python
import asyncio
from collections import defaultdict
from typing import Any, AsyncGenerator
from dataclasses import dataclass, field


@dataclass
class PubSub:
    """GraphQL 서브스크립션을 위한 간단한 인메모리 pub/sub."""
    _subscribers: dict[str, list[asyncio.Queue]] = field(
        default_factory=lambda: defaultdict(list)
    )

    async def publish(self, channel: str, payload: Any) -> None:
        """채널의 모든 구독자에게 메시지를 발행합니다."""
        for queue in self._subscribers[channel]:
            await queue.put(payload)

    async def subscribe(self, channel: str) -> AsyncGenerator[Any, None]:
        """채널을 구독하고 메시지를 yield합니다."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[channel].append(queue)
        try:
            while True:
                payload = await queue.get()
                yield payload
        finally:
            self._subscribers[channel].remove(queue)

    @property
    def subscriber_count(self) -> dict[str, int]:
        return {ch: len(subs) for ch, subs in self._subscribers.items()}


# Global instance
pubsub = PubSub()
```

### 뮤테이션에서 발행

```python
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_post(
        self, info: strawberry.types.Info, input: CreatePostInput
    ) -> CreatePostPayload:
        post = await post_repository.create(
            title=input.title,
            content=input.content,
            author_id=info.context.current_user.id,
        )
        # 서브스크립션을 위한 이벤트 발행
        await info.context.pubsub.publish("post.created", post)
        return CreatePostPayload(post=post)

    @strawberry.mutation
    async def send_message(
        self, info: strawberry.types.Info, input: SendMessageInput
    ) -> SendMessagePayload:
        message = await message_repository.create(
            channel_id=input.channel_id,
            content=input.content,
            author_id=info.context.current_user.id,
        )

        # Publish to channel-specific topic
        channel = f"channel.{input.channel_id}.messages"
        await info.context.pubsub.publish(channel, message)

        return SendMessagePayload(message=message)
```

### Pub/Sub를 이용한 서브스크립션 리졸버

```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def post_created(
        self, info: strawberry.types.Info
    ) -> AsyncGenerator["Post", None]:
        async for post in info.context.pubsub.subscribe("post.created"):
            yield post

    @strawberry.subscription
    async def message_added(
        self, info: strawberry.types.Info, channel_id: strawberry.ID
    ) -> AsyncGenerator["Message", None]:
        topic = f"channel.{channel_id}.messages"
        async for message in info.context.pubsub.subscribe(topic):
            yield message
```

---

## 7. Redis 기반 Pub/Sub

다중 서버 배포의 경우 Redis를 메시지 브로커로 사용합니다.

### 구현

```python
import aioredis
import json
from typing import Any, AsyncGenerator


class RedisPubSub:
    """분산 서브스크립션 지원을 위한 Redis 기반 pub/sub."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(self.redis_url)

    async def publish(self, channel: str, payload: Any) -> None:
        message = json.dumps(payload, default=str)
        await self._redis.publish(channel, message)

    async def subscribe(self, channel: str) -> AsyncGenerator[Any, None]:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield data
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
```

### 다중 서버 아키텍처

```
                    ┌─────────────────┐
                    │   Redis Pub/Sub  │
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  Server 1  │  │  Server 2  │  │  Server 3  │
     │ (WS conns) │  │ (WS conns) │  │ (WS conns) │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
       ▲  ▲  ▲          ▲  ▲            ▲  ▲  ▲
     Clients          Clients         Clients
```

---

## 8. 서브스크립션 생명주기

### 연결 단계

```python
async def on_ws_connect(context: dict) -> bool:
    """WebSocket 연결이 시작될 때 호출됩니다."""
    auth_token = context.get("connectionParams", {}).get("authToken")
    if not auth_token:
        raise ConnectionRefusedError("인증 토큰이 없습니다")
    user = await authenticate(auth_token)
    if user is None:
        raise ConnectionRefusedError("유효하지 않은 인증 토큰입니다")
    context["current_user"] = user
    return True
```

### 킵얼라이브(Keep-Alive)

```python
# Strawberry handles ping/pong automatically
# Configure timeout in the router:
graphql_router = GraphQLRouter(
    schema,
    subscription_protocols=[GRAPHQL_TRANSPORT_WS_PROTOCOL],
    keep_alive=True,
    keep_alive_interval=30,  # seconds
)
```

### 연결 해제 시 정리

```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def live_metrics(
        self, info: strawberry.types.Info
    ) -> AsyncGenerator["Metric", None]:
        user = info.context.current_user
        session_id = str(uuid.uuid4())
        await subscription_tracker.add(user.id, session_id)
        try:
            async for metric in metrics_stream.subscribe():
                yield metric
        finally:
            await subscription_tracker.remove(user.id, session_id)
```

---

## 9. 필터링과 권한 부여

### 서버 측 필터링

```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def order_status_changed(
        self, info: strawberry.types.Info, order_id: strawberry.ID
    ) -> AsyncGenerator["OrderStatusEvent", None]:
        user = require_auth(info)
        order = await order_repository.find_by_id(order_id)
        if order is None or order.customer_id != user.id:
            raise PermissionError("이 주문을 추적할 권한이 없습니다")
        topic = f"order.{order_id}.status"
        async for event in info.context.pubsub.subscribe(topic):
            yield OrderStatusEvent(**event)

    @strawberry.subscription
    async def activity_feed(
        self, info: strawberry.types.Info
    ) -> AsyncGenerator["Activity", None]:
        user = require_auth(info)

        async for activity in info.context.pubsub.subscribe("activity.global"):
            # 필터: 이 사용자에게 관련된 활동만 yield
            if activity["user_id"] in user.following_ids:
                yield Activity(**activity)
```

### 서브스크립션 속도 제한

```python
MAX_SUBSCRIPTIONS_PER_USER = 10

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def notifications(
        self, info: strawberry.types.Info
    ) -> AsyncGenerator["Notification", None]:
        user = require_auth(info)
        active_count = await subscription_tracker.count(user.id)
        if active_count >= MAX_SUBSCRIPTIONS_PER_USER:
            raise ValueError(
                f"최대 {MAX_SUBSCRIPTIONS_PER_USER}개의 동시 서브스크립션을 초과했습니다"
            )
        async for notification in info.context.pubsub.subscribe(
            f"user.{user.id}.notifications"
        ):
            yield Notification(**notification)
```

---

## 10. 프로덕션 고려사항

### 연결 제한

| 설정 | 권장값 | 목적 |
|------|--------|------|
| 서버당 최대 연결 | 10,000 | 리소스 고갈 방지 |
| 사용자당 최대 서브스크립션 | 10 | 공정 사용 |
| 연결 타임아웃 | 5초 | 느린 연결 거부 |
| 킵얼라이브 간격 | 30초 | 죽은 연결 감지 |
| 최대 유휴 시간 | 5분 | 비활성 연결 종료 |

### 확장 전략

```
소규모 (단일 서버):
  - 인메모리 pub/sub
  - < 1,000 동시 서브스크립션에 충분

중규모 (소수 서버):
  - Redis pub/sub
  - 10,000+ 동시 서브스크립션 지원

대규모 (다수 서버):
  - Kafka 또는 NATS 메시지 브로커
  - 전용 WebSocket 게이트웨이 서버
  - Redis에 연결 상태 저장
```

### 모니터링(Monitoring)

다음 서브스크립션 메트릭을 추적하세요:

- 서버당 활성 WebSocket 연결 수
- 토픽별 서브스크립션 수
- 메시지 처리량 (메시지/초)
- 평균 메시지 지연시간
- 연결 오류율
- 재연결율

### 오류 복구(Error Recovery)

```javascript
// Client-side reconnection with exponential backoff
const client = createClient({
  url: 'ws://api.example.com/graphql',
  retryAttempts: Infinity,
  retryWait: async (retries) => {
    // Exponential backoff: 1s, 2s, 4s, 8s, ..., max 30s
    const delay = Math.min(1000 * Math.pow(2, retries), 30000);
    // Add jitter: +-25%
    const jitter = delay * 0.25 * (Math.random() * 2 - 1);
    await new Promise(r => setTimeout(r, delay + jitter));
  },
});
```

---

## 11. 연습 문제

### 연습 1: 채팅방 서브스크립션

완전한 채팅 시스템을 구현하세요:
- `Subscription.messageAdded(roomId: ID!)` — 방의 새 메시지
- `Subscription.userTyping(roomId: ID!)` — 타이핑 인디케이터
- `Mutation.sendMessage(input: SendMessageInput!)` — 메시지 전송
- `Mutation.setTyping(roomId: ID!, isTyping: Boolean!)` — 타이핑 상태 업데이트

### 연습 2: 라이브 대시보드

시스템 메트릭을 스트리밍하는 서브스크립션을 구축하세요:
- CPU 사용량, 메모리 사용량, 요청 수
- 2초마다 업데이트 방출
- `metricType` 필터 인자 허용

### 연습 3: 주문 추적

주문 추적 서브스크립션을 구현하세요:
- 특정 주문의 상태 변경 구독
- 구독자가 주문 소유자인지 확인
- 이벤트 페이로드에 이전 및 새 상태 포함
- 주문이 최종 상태(DELIVERED, CANCELLED)에 도달하면 서브스크립션 정리

### 연습 4: Redis Pub/Sub 마이그레이션

인메모리 PubSub 구현을 다음과 같이 수정하세요:
- Redis 백엔드 추가
- 메시지가 JSON 직렬화 가능하도록 보장
- Redis 연결 실패를 우아하게 처리
- 재연결 로직 추가

---

## 12. 참고 자료

### 사양
- [graphql-ws Protocol](https://github.com/enisdenjo/graphql-ws/blob/master/PROTOCOL.md)
- [GraphQL Subscriptions RFC](https://github.com/graphql/graphql-spec/blob/main/spec/Section%206%20--%20Execution.md)

### 라이브러리
- [Strawberry Subscriptions](https://strawberry.rocks/docs/general/subscriptions)
- [graphql-ws (JavaScript Client)](https://github.com/enisdenjo/graphql-ws)
- [aioredis — Async Redis for Python](https://aioredis.readthedocs.io/)

---

**License**: CC BY-NC 4.0
