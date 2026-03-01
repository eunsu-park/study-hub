"""
Message Queue

Demonstrates:
- Point-to-point queue (competing consumers)
- Publish-subscribe (fan-out)
- Message acknowledgment and retry
- Dead letter queue
- Priority queue

Theory:
- Message queues decouple producers and consumers, enabling async
  communication, load leveling, and fault tolerance.
- Point-to-point: each message consumed by exactly one consumer.
  Good for task distribution.
- Pub/sub: each message delivered to all subscribers.
  Good for event notification.
- Ack/retry: consumers acknowledge processing; unacked messages
  are redelivered after timeout.
- Dead letter queue (DLQ): messages that fail repeatedly are
  moved to a separate queue for investigation.

Adapted from System Design Lesson 11.
"""

import time
import heapq
from collections import deque, defaultdict
from threading import Lock
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class MessageState(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKED = "acked"
    NACKED = "nacked"


@dataclass
class Message:
    id: int
    body: Any
    topic: str = "default"
    priority: int = 0  # lower = higher priority
    timestamp: float = field(default_factory=time.monotonic)
    delivery_count: int = 0
    state: MessageState = MessageState.PENDING

    def __lt__(self, other: "Message") -> bool:
        # Why: Breaking priority ties by timestamp ensures FIFO ordering within
        # the same priority level, preventing starvation of older messages.
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)


# ── Point-to-Point Queue ──────────────────────────────────────────────

# Why: Point-to-point (competing consumers) ensures each message is processed
# exactly once, making it ideal for task distribution (e.g., worker pools).
# The in_flight dict tracks unacknowledged messages for at-least-once delivery.
class SimpleQueue:
    """Point-to-point message queue with ack/retry."""

    def __init__(self, max_retries: int = 3):
        self.queue: deque[Message] = deque()
        # Why: Separating in-flight messages from the main queue prevents
        # re-delivery of messages already being processed, while still enabling
        # retry if the consumer fails to acknowledge within the timeout.
        self.in_flight: dict[int, Message] = {}
        self.dead_letter: list[Message] = []
        self.max_retries = max_retries
        self.next_id = 0
        self.lock = Lock()

    def send(self, body: Any, priority: int = 0) -> int:
        with self.lock:
            msg = Message(id=self.next_id, body=body, priority=priority)
            self.next_id += 1
            self.queue.append(msg)
            return msg.id

    def receive(self) -> Message | None:
        with self.lock:
            if not self.queue:
                return None
            msg = self.queue.popleft()
            msg.state = MessageState.DELIVERED
            msg.delivery_count += 1
            self.in_flight[msg.id] = msg
            return msg

    def ack(self, msg_id: int) -> bool:
        with self.lock:
            if msg_id in self.in_flight:
                msg = self.in_flight.pop(msg_id)
                msg.state = MessageState.ACKED
                return True
            return False

    def nack(self, msg_id: int) -> str:
        """Negative ack: retry or dead-letter."""
        with self.lock:
            if msg_id not in self.in_flight:
                return "NOT_FOUND"
            msg = self.in_flight.pop(msg_id)
            msg.state = MessageState.NACKED
            # Why: Capping retries and moving to a dead-letter queue prevents
            # "poison pill" messages from blocking the queue indefinitely.
            # DLQ messages can be inspected and fixed by operators later.
            if msg.delivery_count >= self.max_retries:
                self.dead_letter.append(msg)
                return "DEAD_LETTER"
            self.queue.append(msg)
            return "RETRY"

    @property
    def size(self) -> int:
        return len(self.queue)


# ── Pub/Sub ────────────────────────────────────────────────────────────

# Why: Pub/sub decouples publishers from subscribers entirely — the publisher
# doesn't need to know who is listening. This enables fan-out patterns where
# a single event (e.g., "order created") triggers analytics, billing, and
# notifications independently.
class PubSub:
    """Publish-subscribe message broker."""

    def __init__(self):
        self.topics: dict[str, list[str]] = defaultdict(list)
        self.subscriber_queues: dict[str, deque[Message]] = {}
        self.next_id = 0
        self.lock = Lock()

    def subscribe(self, subscriber: str, topic: str) -> None:
        with self.lock:
            if subscriber not in self.topics[topic]:
                self.topics[topic].append(subscriber)
            if subscriber not in self.subscriber_queues:
                self.subscriber_queues[subscriber] = deque()

    def unsubscribe(self, subscriber: str, topic: str) -> None:
        with self.lock:
            if subscriber in self.topics[topic]:
                self.topics[topic].remove(subscriber)

    def publish(self, topic: str, body: Any) -> int:
        """Publish to all subscribers of a topic."""
        with self.lock:
            delivered = 0
            for subscriber in self.topics.get(topic, []):
                msg = Message(
                    id=self.next_id, body=body, topic=topic
                )
                self.next_id += 1
                self.subscriber_queues[subscriber].append(msg)
                delivered += 1
            return delivered

    def poll(self, subscriber: str) -> Message | None:
        with self.lock:
            queue = self.subscriber_queues.get(subscriber)
            if queue:
                return queue.popleft()
            return None

    def pending(self, subscriber: str) -> int:
        queue = self.subscriber_queues.get(subscriber, deque())
        return len(queue)


# ── Priority Queue ─────────────────────────────────────────────────────

# Why: Priority queues ensure critical messages (e.g., alerts, payment failures)
# are processed before routine work, even under high load. The heap gives O(log n)
# enqueue/dequeue while maintaining priority ordering.
class PriorityMessageQueue:
    """Message queue with priority ordering."""

    def __init__(self):
        self.heap: list[Message] = []
        self.next_id = 0
        self.lock = Lock()

    def send(self, body: Any, priority: int = 0) -> int:
        with self.lock:
            msg = Message(id=self.next_id, body=body, priority=priority)
            self.next_id += 1
            heapq.heappush(self.heap, msg)
            return msg.id

    def receive(self) -> Message | None:
        with self.lock:
            if self.heap:
                return heapq.heappop(self.heap)
            return None

    @property
    def size(self) -> int:
        return len(self.heap)


# ── Demos ──────────────────────────────────────────────────────────────

def demo_point_to_point():
    print("=" * 60)
    print("POINT-TO-POINT QUEUE")
    print("=" * 60)

    q = SimpleQueue(max_retries=3)

    # Producer sends messages
    print(f"\n  Producer sends 5 messages:")
    for i in range(5):
        msg_id = q.send(f"task-{i}")
        print(f"    Sent message {msg_id}: 'task-{i}'")

    # Consumer processes messages
    print(f"\n  Consumer processes messages:")
    for _ in range(5):
        msg = q.receive()
        if msg:
            print(f"    Received msg {msg.id}: {msg.body} "
                  f"(delivery #{msg.delivery_count})")
            q.ack(msg.id)
    print(f"    Queue size: {q.size}")


def demo_ack_retry():
    print("\n" + "=" * 60)
    print("ACK/NACK AND DEAD LETTER QUEUE")
    print("=" * 60)

    q = SimpleQueue(max_retries=3)
    q.send("always-fails")
    q.send("succeeds-second-try")

    print(f"\n  Processing with failures:")

    # Message 0 fails 3 times → dead letter
    for attempt in range(3):
        msg = q.receive()
        if msg and msg.body == "always-fails":
            result = q.nack(msg.id)
            print(f"    msg {msg.id} '{msg.body}' attempt {msg.delivery_count}: "
                  f"NACK → {result}")

    # Message 1 fails once, succeeds second time
    msg = q.receive()
    if msg:
        print(f"    msg {msg.id} '{msg.body}' attempt {msg.delivery_count}: "
              f"NACK → {q.nack(msg.id)}")
        msg = q.receive()
        if msg:
            print(f"    msg {msg.id} '{msg.body}' attempt {msg.delivery_count}: "
                  f"ACK → success")
            q.ack(msg.id)

    print(f"\n  Dead letter queue: {len(q.dead_letter)} message(s)")
    for msg in q.dead_letter:
        print(f"    msg {msg.id}: {msg.body} "
              f"(delivered {msg.delivery_count} times)")


def demo_pubsub():
    print("\n" + "=" * 60)
    print("PUBLISH-SUBSCRIBE")
    print("=" * 60)

    ps = PubSub()

    # Subscribe
    ps.subscribe("analytics", "orders")
    ps.subscribe("analytics", "users")
    ps.subscribe("billing", "orders")
    ps.subscribe("notifications", "orders")
    ps.subscribe("notifications", "users")

    print(f"\n  Subscriptions:")
    print(f"    analytics → orders, users")
    print(f"    billing → orders")
    print(f"    notifications → orders, users")

    # Publish
    print(f"\n  Publishing events:")
    n = ps.publish("orders", {"type": "order_created", "id": 42})
    print(f"    'orders' topic: delivered to {n} subscribers")

    n = ps.publish("users", {"type": "user_signup", "id": 100})
    print(f"    'users' topic: delivered to {n} subscribers")

    # Each subscriber receives their messages
    print(f"\n  Subscriber queues:")
    for sub in ["analytics", "billing", "notifications"]:
        pending = ps.pending(sub)
        print(f"    {sub}: {pending} message(s)")
        while True:
            msg = ps.poll(sub)
            if not msg:
                break
            print(f"      → {msg.topic}: {msg.body}")


def demo_priority():
    print("\n" + "=" * 60)
    print("PRIORITY MESSAGE QUEUE")
    print("=" * 60)

    pq = PriorityMessageQueue()

    # Send with varying priorities
    messages = [
        ("low-priority-task", 10),
        ("high-priority-alert", 1),
        ("medium-task", 5),
        ("critical-alert", 0),
        ("normal-task", 5),
    ]

    print(f"\n  Sending messages (lower number = higher priority):")
    for body, priority in messages:
        pq.send(body, priority)
        print(f"    '{body}' (priority={priority})")

    print(f"\n  Receiving (priority order):")
    while pq.size > 0:
        msg = pq.receive()
        if msg:
            print(f"    [{msg.priority}] {msg.body}")


if __name__ == "__main__":
    demo_point_to_point()
    demo_ack_retry()
    demo_pubsub()
    demo_priority()
