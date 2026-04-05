#!/bin/bash
# Exercises for Lesson 20: GraphQL Subscriptions
# Topic: API_Design
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Chat Room Subscriptions ==="
    cat << 'SOLUTION'
import strawberry
import asyncio
from typing import AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict

# PubSub (from lesson)
@dataclass
class PubSub:
    _subscribers: dict[str, list[asyncio.Queue]] = field(
        default_factory=lambda: defaultdict(list)
    )
    async def publish(self, channel, payload):
        for queue in self._subscribers[channel]:
            await queue.put(payload)
    async def subscribe(self, channel):
        queue = asyncio.Queue()
        self._subscribers[channel].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers[channel].remove(queue)

pubsub = PubSub()

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def message_added(
        self, info, room_id: strawberry.ID
    ) -> AsyncGenerator["Message", None]:
        async for msg in pubsub.subscribe(f"room.{room_id}.messages"):
            yield msg

    @strawberry.subscription
    async def user_typing(
        self, info, room_id: strawberry.ID
    ) -> AsyncGenerator["TypingEvent", None]:
        async for event in pubsub.subscribe(f"room.{room_id}.typing"):
            yield event

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def send_message(self, info, input: "SendMessageInput") -> "Message":
        message = Message(
            id=str(uuid4()), body=input.body,
            author_id=info.context.current_user.id,
            room_id=input.room_id,
        )
        await pubsub.publish(f"room.{input.room_id}.messages", message)
        return message

    @strawberry.mutation
    async def set_typing(
        self, info, room_id: strawberry.ID, is_typing: bool
    ) -> bool:
        event = TypingEvent(
            user_id=info.context.current_user.id,
            room_id=room_id,
            is_typing=is_typing,
        )
        await pubsub.publish(f"room.{room_id}.typing", event)
        return True
SOLUTION
}

exercise_2() {
    echo "=== Exercise 2: Live Dashboard ==="
    cat << 'SOLUTION'
import psutil

@strawberry.enum
class MetricType:
    CPU = "cpu"
    MEMORY = "memory"
    REQUESTS = "requests"

@strawberry.type
class SystemMetric:
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def system_metrics(
        self, info, metric_type: MetricType | None = None
    ) -> AsyncGenerator[SystemMetric, None]:
        while True:
            metrics = {
                MetricType.CPU: SystemMetric(
                    metric_type=MetricType.CPU,
                    value=psutil.cpu_percent(),
                    unit="percent", timestamp=datetime.now(),
                ),
                MetricType.MEMORY: SystemMetric(
                    metric_type=MetricType.MEMORY,
                    value=psutil.virtual_memory().percent,
                    unit="percent", timestamp=datetime.now(),
                ),
            }
            if metric_type:
                yield metrics[metric_type]
            else:
                for m in metrics.values():
                    yield m
            await asyncio.sleep(2)
SOLUTION
}

exercise_3() {
    echo "=== Exercise 3: Order Tracking ==="
    cat << 'SOLUTION'
TERMINAL_STATUSES = {"DELIVERED", "CANCELLED"}

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def order_status_changed(
        self, info, order_id: strawberry.ID
    ) -> AsyncGenerator["OrderStatusEvent", None]:
        user = require_auth(info)
        order = await order_repo.find_by_id(order_id)
        if not order or order.customer_id != user.id:
            raise PermissionError("Not authorized")

        topic = f"order.{order_id}.status"
        async for event in info.context.pubsub.subscribe(topic):
            yield OrderStatusEvent(
                order_id=order_id,
                previous_status=event["previous"],
                new_status=event["new"],
                timestamp=event["timestamp"],
            )
            if event["new"] in TERMINAL_STATUSES:
                return  # End subscription
SOLUTION
}

exercise_4() {
    echo "=== Exercise 4: Redis Pub/Sub Migration ==="
    cat << 'SOLUTION'
import aioredis
import json

class RedisPubSub:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
        self._reconnect_delay = 1

    async def connect(self):
        try:
            self._redis = await aioredis.from_url(self.redis_url)
            self._reconnect_delay = 1
        except Exception as e:
            print(f"Redis connection failed: {e}")
            raise

    async def _ensure_connected(self):
        if self._redis is None:
            await self.connect()

    async def publish(self, channel, payload):
        await self._ensure_connected()
        message = json.dumps(payload, default=str)
        await self._redis.publish(channel, message)

    async def subscribe(self, channel):
        await self._ensure_connected()
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        except aioredis.ConnectionError:
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 30)
            self._redis = None
            async for msg in self.subscribe(channel):
                yield msg
        finally:
            await pubsub.unsubscribe(channel)
SOLUTION
}

main() { exercise_1; echo ""; exercise_2; echo ""; exercise_3; echo ""; exercise_4; }
main "$@"
