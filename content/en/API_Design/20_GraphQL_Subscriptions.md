# 20. GraphQL Subscriptions

**Previous**: [GraphQL Resolvers](./19_GraphQL_Resolvers.md) | **Next**: [GraphQL Server Implementation](./21_GraphQL_Server_Implementation.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Explain how GraphQL subscriptions provide real-time data delivery to clients
- Implement subscription resolvers using async generators in Python
- Configure WebSocket transport for GraphQL subscription connections
- Apply the publish/subscribe pattern with in-memory and Redis-backed brokers
- Handle subscription lifecycle events: connection, keep-alive, and termination
- Design subscription schemas that are efficient and secure in production

---

## Table of Contents

1. [What Are Subscriptions?](#1-what-are-subscriptions)
2. [Subscriptions vs. Polling vs. SSE](#2-subscriptions-vs-polling-vs-sse)
3. [Subscription Schema Design](#3-subscription-schema-design)
4. [WebSocket Transport](#4-websocket-transport)
5. [Implementing Subscriptions in Strawberry](#5-implementing-subscriptions-in-strawberry)
6. [Pub/Sub Pattern](#6-pubsub-pattern)
7. [Redis-Backed Pub/Sub](#7-redis-backed-pubsub)
8. [Subscription Lifecycle](#8-subscription-lifecycle)
9. [Filtering and Authorization](#9-filtering-and-authorization)
10. [Production Considerations](#10-production-considerations)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. What Are Subscriptions?

Subscriptions are a GraphQL operation type that enables real-time, server-pushed updates. While queries and mutations follow a request-response pattern, subscriptions maintain a persistent connection between client and server.

### The Three Operation Types

| Operation | Direction | Connection | Use Case |
|-----------|-----------|------------|----------|
| Query | Client → Server → Client | Short-lived | Read data |
| Mutation | Client → Server → Client | Short-lived | Write data |
| Subscription | Client → Server ← Server (continuous) | Long-lived | Real-time updates |

### How Subscriptions Work

```
1. Client sends subscription operation over WebSocket
2. Server acknowledges and starts listening for events
3. When an event occurs, server pushes data to client
4. Connection remains open until client unsubscribes or disconnects

Client                         Server
  |--- subscribe(postCreated) -->|
  |<-- ack ----------------------|
  |                              |  (new post created)
  |<-- data: { postCreated }  ---|
  |                              |  (another post)
  |<-- data: { postCreated }  ---|
  |--- unsubscribe ------------->|
  |<-- complete -----------------|
```

### Common Use Cases

- **Chat messages**: New messages in a conversation
- **Notifications**: Real-time alerts for user activity
- **Live dashboards**: Metrics, stock prices, scores
- **Collaborative editing**: Document changes, cursor positions
- **Order tracking**: Status updates for deliveries
- **IoT data streams**: Sensor readings, device status

---

## 2. Subscriptions vs. Polling vs. SSE

### Comparison

| Approach | Latency | Complexity | Bidirectional | Efficiency |
|----------|---------|------------|---------------|------------|
| **Polling** | High (interval-bound) | Low | No | Low (many empty responses) |
| **Long polling** | Medium | Medium | No | Medium |
| **SSE** | Low | Low | No | High |
| **WebSocket** | Low | High | Yes | High |
| **GraphQL Subscriptions** | Low | Medium | Yes (via WS) | High |

### When to Use Each

```
Polling:
  - Update frequency > 30 seconds
  - Simple implementation needed
  - HTTP caching beneficial

SSE (Server-Sent Events):
  - Server-to-client only
  - Simple text/event-stream
  - Auto-reconnection built in
  - Works through HTTP proxies

WebSocket / GraphQL Subscriptions:
  - Real-time bidirectional needs
  - Complex event filtering
  - Multiple event streams per connection
  - Sub-second latency required
```

### SSE Alternative

For simpler cases, consider SSE over subscriptions:

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

## 3. Subscription Schema Design

### Basic Schema

```graphql
type Subscription {
  postCreated: Post!
  postUpdated(id: ID): Post!
  commentAdded(postId: ID!): Comment!
  notificationReceived: Notification!
}
```

### Filtered Subscriptions

```graphql
type Subscription {
  # Subscribe to specific channel
  messageAdded(channelId: ID!): Message!

  # Subscribe with filter
  orderStatusChanged(orderId: ID!): OrderStatusEvent!

  # Subscribe to category
  priceUpdated(symbol: String!): PriceEvent!
}

type OrderStatusEvent {
  order: Order!
  previousStatus: OrderStatus!
  newStatus: OrderStatus!
  timestamp: DateTime!
}

type PriceEvent {
  symbol: String!
  price: Decimal!
  change: Decimal!
  percentChange: Float!
  timestamp: DateTime!
}
```

### Event Payload Design

Return rich payloads that include context:

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

type MessageEdited {
  message: Message!
  previousContent: String!
  editedAt: DateTime!
}

type MessageDeleted {
  messageId: ID!
  deletedBy: User!
  deletedAt: DateTime!
}

type UserTyping {
  user: User!
  room: ChatRoom!
  isTyping: Boolean!
}
```

---

## 4. WebSocket Transport

### graphql-ws Protocol

The `graphql-ws` protocol (successor to the legacy `subscriptions-transport-ws`) is the modern standard.

### Protocol Message Flow

```
Client                            Server
  |--- ConnectionInit { payload } -->|
  |<-- ConnectionAck ----------------|
  |                                  |
  |--- Subscribe { id, payload }  -->|
  |<-- Next { id, payload }       ---|  (event 1)
  |<-- Next { id, payload }       ---|  (event 2)
  |--- Complete { id }            -->|  (unsubscribe)
  |                                  |
  |--- Ping                       -->|  (keep-alive)
  |<-- Pong                       ---|
```

### Message Types

| Direction | Message | Purpose |
|-----------|---------|---------|
| C → S | `ConnectionInit` | Initialize connection (auth payload) |
| S → C | `ConnectionAck` | Connection accepted |
| C → S | `Subscribe` | Start subscription |
| S → C | `Next` | Subscription data |
| S → C | `Error` | Subscription error |
| S → C | `Complete` | Subscription ended |
| C → S | `Complete` | Client unsubscribes |
| Both | `Ping` / `Pong` | Keep-alive |

### Client-Side JavaScript

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

## 5. Implementing Subscriptions in Strawberry

### Basic Subscription with Async Generator

```python
import strawberry
import asyncio
from typing import AsyncGenerator


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def count(self, target: int = 10) -> AsyncGenerator[int, None]:
        """Count from 0 to target, emitting one value per second."""
        for i in range(target):
            yield i
            await asyncio.sleep(1)

    @strawberry.subscription
    async def post_created(self) -> AsyncGenerator["Post", None]:
        """Emit events when new posts are created."""
        async for event in post_event_bus.subscribe("post.created"):
            yield event.payload
```

### FastAPI + WebSocket Setup

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

### Subscription with Context

```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def notifications(
        self, info: strawberry.types.Info
    ) -> AsyncGenerator["Notification", None]:
        """Emit notifications for the authenticated user."""
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

## 6. Pub/Sub Pattern

### In-Memory Pub/Sub

Suitable for single-server deployments:

```python
import asyncio
from collections import defaultdict
from typing import Any, AsyncGenerator
from dataclasses import dataclass, field


@dataclass
class PubSub:
    """Simple in-memory pub/sub for GraphQL subscriptions."""
    _subscribers: dict[str, list[asyncio.Queue]] = field(
        default_factory=lambda: defaultdict(list)
    )

    async def publish(self, channel: str, payload: Any) -> None:
        """Publish a message to all subscribers of a channel."""
        for queue in self._subscribers[channel]:
            await queue.put(payload)

    async def subscribe(self, channel: str) -> AsyncGenerator[Any, None]:
        """Subscribe to a channel and yield messages."""
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

### Publishing from Mutations

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

        # Publish event for subscriptions
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

### Subscription Resolvers with Pub/Sub

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

## 7. Redis-Backed Pub/Sub

For multi-server deployments, use Redis as the message broker.

### Implementation

```python
import aioredis
import json
from typing import Any, AsyncGenerator


class RedisPubSub:
    """Redis-backed pub/sub for distributed subscription support."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(self.redis_url)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.close()

    async def publish(self, channel: str, payload: Any) -> None:
        """Publish a JSON-serialized message to a Redis channel."""
        if self._redis is None:
            raise RuntimeError("RedisPubSub not connected")
        message = json.dumps(payload, default=str)
        await self._redis.publish(channel, message)

    async def subscribe(self, channel: str) -> AsyncGenerator[Any, None]:
        """Subscribe to a Redis channel and yield parsed messages."""
        if self._redis is None:
            raise RuntimeError("RedisPubSub not connected")

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


# Application setup
redis_pubsub = RedisPubSub("redis://localhost:6379")


# FastAPI lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await redis_pubsub.connect()
    yield
    await redis_pubsub.disconnect()

app = FastAPI(lifespan=lifespan)
```

### Multi-Server Architecture

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
       │  │  │          │  │            │  │  │
     Clients          Clients         Clients
```

When Server 1 receives a mutation, it publishes to Redis. All servers (including Server 1) receive the message and forward it to their connected subscription clients.

---

## 8. Subscription Lifecycle

### Connection Phase

```python
async def on_ws_connect(context: dict) -> bool:
    """Called when a WebSocket connection is initiated.

    Return True to accept, raise to reject.
    """
    auth_token = context.get("connectionParams", {}).get("authToken")
    if not auth_token:
        raise ConnectionRefusedError("Missing authentication token")

    user = await authenticate(auth_token)
    if user is None:
        raise ConnectionRefusedError("Invalid authentication token")

    # Store user in connection context
    context["current_user"] = user
    return True
```

### Keep-Alive

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

### Cleanup on Disconnect

```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def live_metrics(
        self, info: strawberry.types.Info
    ) -> AsyncGenerator["Metric", None]:
        user = info.context.current_user
        session_id = str(uuid.uuid4())

        # Track active subscription
        await subscription_tracker.add(user.id, session_id)

        try:
            async for metric in metrics_stream.subscribe():
                yield metric
        finally:
            # Clean up on disconnect
            await subscription_tracker.remove(user.id, session_id)
```

---

## 9. Filtering and Authorization

### Server-Side Filtering

```python
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def order_status_changed(
        self,
        info: strawberry.types.Info,
        order_id: strawberry.ID,
    ) -> AsyncGenerator["OrderStatusEvent", None]:
        user = require_auth(info)

        # Verify user owns the order
        order = await order_repository.find_by_id(order_id)
        if order is None or order.customer_id != user.id:
            raise PermissionError("Not authorized to track this order")

        topic = f"order.{order_id}.status"
        async for event in info.context.pubsub.subscribe(topic):
            yield OrderStatusEvent(**event)

    @strawberry.subscription
    async def activity_feed(
        self, info: strawberry.types.Info
    ) -> AsyncGenerator["Activity", None]:
        user = require_auth(info)

        async for activity in info.context.pubsub.subscribe("activity.global"):
            # Filter: only yield activities relevant to this user
            if activity["user_id"] in user.following_ids:
                yield Activity(**activity)
```

### Rate Limiting Subscriptions

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
                f"Maximum {MAX_SUBSCRIPTIONS_PER_USER} concurrent subscriptions exceeded"
            )

        async for notification in info.context.pubsub.subscribe(
            f"user.{user.id}.notifications"
        ):
            yield Notification(**notification)
```

---

## 10. Production Considerations

### Connection Limits

| Setting | Recommended | Purpose |
|---------|-------------|---------|
| Max connections per server | 10,000 | Prevent resource exhaustion |
| Max subscriptions per user | 10 | Fair usage |
| Connection timeout | 5 seconds | Reject slow connections |
| Keep-alive interval | 30 seconds | Detect dead connections |
| Max idle time | 5 minutes | Close inactive connections |

### Scaling Strategy

```
Small (single server):
  - In-memory pub/sub
  - Sufficient for < 1,000 concurrent subscriptions

Medium (few servers):
  - Redis pub/sub
  - Supports 10,000+ concurrent subscriptions

Large (many servers):
  - Kafka or NATS for message broker
  - Dedicated WebSocket gateway servers
  - Connection state in Redis
```

### Monitoring

Track these subscription metrics:

- Active WebSocket connections per server
- Subscriptions per topic
- Message throughput (messages/sec)
- Average message latency
- Connection error rate
- Reconnection rate

### Error Recovery

```python
# Client-side reconnection with exponential backoff
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

## 11. Exercises

### Exercise 1: Chat Room Subscriptions

Implement a complete chat system with:
- `Subscription.messageAdded(roomId: ID!)` — new messages in a room
- `Subscription.userTyping(roomId: ID!)` — typing indicators
- `Mutation.sendMessage(input: SendMessageInput!)` — send a message
- `Mutation.setTyping(roomId: ID!, isTyping: Boolean!)` — update typing status

Use the in-memory PubSub class from Section 6.

### Exercise 2: Live Dashboard

Build a subscription that streams system metrics:
- CPU usage, memory usage, request count
- Emit updates every 2 seconds
- Accept a `metricType` filter argument

### Exercise 3: Order Tracking

Implement an order tracking subscription:
- Subscribe to status changes for a specific order
- Verify the subscriber is the order owner
- Include previous and new status in the event payload
- Clean up subscription when order reaches terminal state (DELIVERED, CANCELLED)

### Exercise 4: Redis Pub/Sub Migration

Take the in-memory PubSub implementation and:
- Add a Redis backend
- Ensure messages are JSON-serializable
- Handle Redis connection failures gracefully
- Add reconnection logic

---

## 12. References

### Specifications
- [graphql-ws Protocol](https://github.com/enisdenjo/graphql-ws/blob/master/PROTOCOL.md)
- [GraphQL Subscriptions RFC](https://github.com/graphql/graphql-spec/blob/main/spec/Section%206%20--%20Execution.md)

### Libraries
- [Strawberry Subscriptions](https://strawberry.rocks/docs/general/subscriptions)
- [graphql-ws (JavaScript Client)](https://github.com/enisdenjo/graphql-ws)
- [aioredis — Async Redis for Python](https://aioredis.readthedocs.io/)

### Articles
- "GraphQL Subscriptions at Scale" — The Guild Blog
- "Real-time Data with GraphQL Subscriptions" — Apollo Blog
- "WebSocket vs. SSE for Real-time APIs" — Ably Engineering

---

**License**: CC BY-NC 4.0
