#!/usr/bin/env python3
"""Example: GraphQL Subscriptions

Demonstrates real-time updates with GraphQL subscriptions:
- Subscription type definition
- Async generator pattern for streaming events
- Event filtering (subscribe to specific channels)
- Combining queries, mutations, and subscriptions
- WebSocket transport

Related lesson: 18_GraphQL_Subscriptions.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]"
    uvicorn 18_graphql_subscriptions:app --reload --port 8000

    # GraphQL Playground: http://localhost:8000/graphql
    # Subscriptions use WebSocket: ws://localhost:8000/graphql
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Optional
from uuid import uuid4

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("subscriptions")

# =============================================================================
# EVENT BUS — Simple pub/sub for in-process events
# =============================================================================
# In production, use Redis Pub/Sub, Kafka, or NATS for cross-process events.

class EventBus:
    """Async event bus that powers GraphQL subscriptions.

    Subscribers receive events via asyncio.Queue. Each subscription
    gets its own queue so slow consumers do not block others.
    """

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def publish(self, channel: str, event: dict):
        """Publish an event to all subscribers of a channel."""
        for queue in self._subscribers.get(channel, []):
            await queue.put(event)
        logger.info(f"Published to '{channel}': {event.get('type', 'unknown')}")

    async def subscribe(self, channel: str) -> AsyncGenerator[dict, None]:
        """Subscribe to a channel. Yields events as they arrive."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[channel].append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers[channel].remove(queue)


bus = EventBus()


# =============================================================================
# ENUMS & TYPES
# =============================================================================

@strawberry.enum
class MessageType(Enum):
    TEXT = "TEXT"
    SYSTEM = "SYSTEM"
    JOIN = "JOIN"
    LEAVE = "LEAVE"


@strawberry.type
class ChatMessage:
    id: str
    room: str
    sender: str
    content: str
    message_type: MessageType
    timestamp: datetime


@strawberry.type
class Notification:
    id: str
    title: str
    body: str
    timestamp: datetime


# =============================================================================
# DATA STORE
# =============================================================================

_messages: list[dict] = []


# =============================================================================
# QUERY
# =============================================================================

@strawberry.type
class Query:
    @strawberry.field(description="Get recent messages in a chat room.")
    def messages(self, room: str, limit: int = 20) -> list[ChatMessage]:
        room_msgs = [m for m in _messages if m["room"] == room]
        return [ChatMessage(**m) for m in room_msgs[-limit:]]


# =============================================================================
# MUTATION — Publishes events that subscriptions pick up
# =============================================================================

@strawberry.type
class Mutation:
    @strawberry.mutation(description="Send a message to a chat room.")
    async def send_message(self, room: str, sender: str, content: str) -> ChatMessage:
        msg = {
            "id": str(uuid4())[:8],
            "room": room,
            "sender": sender,
            "content": content,
            "message_type": MessageType.TEXT,
            "timestamp": datetime.now(timezone.utc),
        }
        _messages.append(msg)

        # Publish to room-specific channel
        await bus.publish(f"chat:{room}", {
            "type": "message",
            "data": msg,
        })
        return ChatMessage(**msg)

    @strawberry.mutation(description="Send a system notification to all subscribers.")
    async def send_notification(self, title: str, body: str) -> Notification:
        notif = {
            "id": str(uuid4())[:8],
            "title": title,
            "body": body,
            "timestamp": datetime.now(timezone.utc),
        }
        await bus.publish("notifications", {"type": "notification", "data": notif})
        return Notification(**notif)


# =============================================================================
# SUBSCRIPTION — Real-time event streams over WebSocket
# =============================================================================
# Subscriptions use async generators. The GraphQL server keeps the
# WebSocket open and pushes each yielded value to the client.

@strawberry.type
class Subscription:
    @strawberry.subscription(description="Subscribe to new messages in a chat room.")
    async def on_message(self, room: str) -> AsyncGenerator[ChatMessage, None]:
        """Clients receive messages as they are sent to the specified room.

        Example subscription:
            subscription {
              onMessage(room: "general") {
                sender
                content
                timestamp
              }
            }
        """
        async for event in bus.subscribe(f"chat:{room}"):
            if event["type"] == "message":
                yield ChatMessage(**event["data"])

    @strawberry.subscription(description="Subscribe to system notifications.")
    async def on_notification(self) -> AsyncGenerator[Notification, None]:
        """Receive all system-wide notifications in real time."""
        async for event in bus.subscribe("notifications"):
            if event["type"] == "notification":
                yield Notification(**event["data"])


# =============================================================================
# SCHEMA & APP
# =============================================================================

schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="GraphQL Subscriptions Demo")
app.include_router(graphql_app, prefix="/graphql")

# =============================================================================
# EXAMPLE OPERATIONS
# =============================================================================

EXAMPLES = """
# === Start a subscription (in GraphQL Playground, use WS tab) ===
subscription {
  onMessage(room: "general") {
    sender
    content
    timestamp
  }
}

# === Then send a message via mutation (in a separate tab) ===
mutation {
  sendMessage(room: "general", sender: "Alice", content: "Hello!") {
    id
    timestamp
  }
}

# === Subscribe to notifications ===
subscription {
  onNotification {
    title
    body
    timestamp
  }
}
"""

if __name__ == "__main__":
    import uvicorn
    print(EXAMPLES)
    uvicorn.run("18_graphql_subscriptions:app", host="127.0.0.1", port=8000, reload=True)
