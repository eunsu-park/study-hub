"""
Message System Comparison — Queue vs Pub/Sub vs Streaming

Demonstrates:
- Point-to-point message queue (competing consumers)
- Publish/subscribe (fan-out)
- Event streaming (log-based, replay)
- Comparison of delivery semantics and use cases

Theory:
- Message Queue: producers send to a queue, exactly one consumer processes
  each message. Good for task distribution and load leveling.
- Pub/Sub: publishers broadcast to topics, all subscribers receive every
  message. Good for event notification and fan-out.
- Event Streaming (e.g., Kafka): append-only log, consumers track offsets.
  Supports replay, ordering, and both queue and pub/sub patterns.
- Delivery guarantees: at-most-once, at-least-once, exactly-once (via
  idempotency or transactions).

Adapted from System Design Lesson 12.
"""

import time
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock


class DeliveryGuarantee(Enum):
    AT_MOST_ONCE = "at-most-once"
    AT_LEAST_ONCE = "at-least-once"
    EXACTLY_ONCE = "exactly-once"


@dataclass
class Message:
    id: int
    topic: str
    payload: str
    timestamp: float
    acknowledged: bool = False


# ── Point-to-Point Queue ─────────────────────────────────────────────

# Why: Point-to-point queues ensure each message is processed by exactly one
# consumer. This is the right pattern for task distribution (e.g., job queues)
# where you want work spread across workers, not duplicated.
class MessageQueue:
    """Point-to-point message queue with competing consumers."""

    def __init__(self, name: str):
        self.name = name
        self.queue: deque[Message] = deque()
        self.processed: list[tuple[str, Message]] = []
        self._msg_id = 0
        self.lock = Lock()

    def send(self, payload: str) -> Message:
        self._msg_id += 1
        msg = Message(self._msg_id, self.name, payload, time.monotonic())
        self.queue.append(msg)
        return msg

    def receive(self, consumer_name: str) -> Message | None:
        """One consumer gets one message (competing consumer pattern)."""
        with self.lock:
            if self.queue:
                msg = self.queue.popleft()
                msg.acknowledged = True
                self.processed.append((consumer_name, msg))
                return msg
            return None

    def depth(self) -> int:
        return len(self.queue)


# ── Pub/Sub System ────────────────────────────────────────────────────

# Why: Pub/Sub decouples producers from consumers — the publisher doesn't know
# (or care) how many subscribers exist. This is ideal for event-driven
# architectures where adding a new consumer shouldn't require changing producers.
class PubSubTopic:
    """Publish/subscribe topic with fan-out delivery."""

    def __init__(self, name: str):
        self.name = name
        self.subscribers: dict[str, list[Message]] = {}
        self._msg_id = 0

    def subscribe(self, subscriber_name: str) -> None:
        self.subscribers[subscriber_name] = []

    def unsubscribe(self, subscriber_name: str) -> None:
        self.subscribers.pop(subscriber_name, None)

    def publish(self, payload: str) -> Message:
        """Publish message to ALL subscribers."""
        self._msg_id += 1
        msg = Message(self._msg_id, self.name, payload, time.monotonic())
        for sub_name in self.subscribers:
            self.subscribers[sub_name].append(msg)
        return msg

    def poll(self, subscriber_name: str) -> Message | None:
        """Subscriber polls for next message."""
        inbox = self.subscribers.get(subscriber_name, [])
        if inbox:
            return inbox.pop(0)
        return None

    def pending(self, subscriber_name: str) -> int:
        return len(self.subscribers.get(subscriber_name, []))


# ── Event Streaming (Log-based) ──────────────────────────────────────

@dataclass
class StreamPartition:
    """A single partition in an event stream."""
    partition_id: int
    log: list[Message] = field(default_factory=list)

    def append(self, msg: Message) -> int:
        offset = len(self.log)
        self.log.append(msg)
        return offset

    def read(self, offset: int, max_messages: int = 10) -> list[Message]:
        return self.log[offset:offset + max_messages]


# Why: Event streaming (Kafka model) stores messages in an append-only log.
# Unlike queues, messages are NOT deleted after consumption — consumers track
# their own offsets. This enables replay (reprocess old events), multiple
# consumer groups (both queue and pub/sub semantics), and time-travel debugging.
class EventStream:
    """Log-based event streaming (Kafka-like)."""

    def __init__(self, topic: str, num_partitions: int = 3):
        self.topic = topic
        self.partitions = [
            StreamPartition(i) for i in range(num_partitions)
        ]
        self.consumer_offsets: dict[str, dict[int, int]] = {}
        self._msg_id = 0

    def produce(self, key: str, payload: str) -> tuple[int, int]:
        """Produce message, partitioned by key hash."""
        self._msg_id += 1
        partition_id = hash(key) % len(self.partitions)
        msg = Message(self._msg_id, self.topic, payload, time.monotonic())
        offset = self.partitions[partition_id].append(msg)
        return partition_id, offset

    def create_consumer_group(self, group: str) -> None:
        self.consumer_offsets[group] = {
            p.partition_id: 0 for p in self.partitions
        }

    def consume(self, group: str, partition_id: int,
                max_messages: int = 5) -> list[Message]:
        """Consume from a partition, advancing the group's offset."""
        if group not in self.consumer_offsets:
            return []
        offset = self.consumer_offsets[group].get(partition_id, 0)
        messages = self.partitions[partition_id].read(offset, max_messages)
        self.consumer_offsets[group][partition_id] = offset + len(messages)
        return messages

    def replay(self, group: str, partition_id: int, from_offset: int = 0) -> None:
        """Reset consumer offset for replay."""
        if group in self.consumer_offsets:
            self.consumer_offsets[group][partition_id] = from_offset

    def total_messages(self) -> int:
        return sum(len(p.log) for p in self.partitions)


# ── Demos ─────────────────────────────────────────────────────────────

def demo_message_queue():
    print("=" * 60)
    print("POINT-TO-POINT MESSAGE QUEUE")
    print("=" * 60)

    queue = MessageQueue("task-queue")

    # Produce messages
    print(f"\n  Producing 10 tasks:")
    for i in range(10):
        msg = queue.send(f"task-{i}")
        print(f"    Sent: {msg.payload} (id={msg.id})")

    # Competing consumers
    print(f"\n  3 competing consumers processing:")
    consumers = ["worker-A", "worker-B", "worker-C"]
    while queue.depth() > 0:
        consumer = random.choice(consumers)
        msg = queue.receive(consumer)
        if msg:
            print(f"    {consumer} processed: {msg.payload}")

    # Count distribution
    counts: dict[str, int] = {}
    for consumer, _ in queue.processed:
        counts[consumer] = counts.get(consumer, 0) + 1
    print(f"\n  Work distribution:")
    for c, n in sorted(counts.items()):
        print(f"    {c}: {n} tasks")
    print(f"\n  Each message processed by exactly ONE consumer.")


def demo_pubsub():
    print("\n" + "=" * 60)
    print("PUBLISH/SUBSCRIBE (FAN-OUT)")
    print("=" * 60)

    topic = PubSubTopic("user-events")

    # Subscribe
    subscribers = ["email-service", "analytics", "audit-log"]
    for sub in subscribers:
        topic.subscribe(sub)
    print(f"\n  Subscribers: {subscribers}")

    # Publish events
    events = ["user.signup", "user.login", "user.purchase"]
    print(f"\n  Publishing {len(events)} events:")
    for event in events:
        msg = topic.publish(event)
        print(f"    Published: {event} (id={msg.id})")

    # Each subscriber gets ALL messages
    print(f"\n  Each subscriber receives ALL events:")
    for sub in subscribers:
        received = []
        while True:
            msg = topic.poll(sub)
            if msg is None:
                break
            received.append(msg.payload)
        print(f"    {sub}: {received}")

    print(f"\n  Total deliveries: {len(events)} events x "
          f"{len(subscribers)} subscribers = {len(events) * len(subscribers)}")


def demo_event_streaming():
    print("\n" + "=" * 60)
    print("EVENT STREAMING (LOG-BASED)")
    print("=" * 60)

    stream = EventStream("orders", num_partitions=3)

    # Produce events
    print(f"\n  Producing 12 order events (3 partitions):")
    for i in range(12):
        key = f"user-{i % 4}"
        payload = f"order-{i}"
        part, offset = stream.produce(key, payload)
        print(f"    {payload} (key={key}) → partition {part}, offset {offset}")

    # Consumer groups
    stream.create_consumer_group("payment-service")
    stream.create_consumer_group("analytics")

    print(f"\n  Consumer group 'payment-service' reads:")
    for p in range(3):
        msgs = stream.consume("payment-service", p)
        if msgs:
            print(f"    Partition {p}: {[m.payload for m in msgs]}")

    print(f"\n  Consumer group 'analytics' reads (independent offset):")
    for p in range(3):
        msgs = stream.consume("analytics", p)
        if msgs:
            print(f"    Partition {p}: {[m.payload for m in msgs]}")

    # Replay
    print(f"\n  Replaying partition 0 for 'analytics':")
    stream.replay("analytics", 0, from_offset=0)
    msgs = stream.consume("analytics", 0)
    print(f"    Re-read: {[m.payload for m in msgs]}")
    print(f"\n  Key feature: messages persist, consumers can replay.")


def demo_comparison():
    print("\n" + "=" * 60)
    print("MESSAGE SYSTEM COMPARISON")
    print("=" * 60)

    print(f"\n  {'Feature':<25} {'Queue':>15} {'Pub/Sub':>15} {'Streaming':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    rows = [
        ("Delivery",         "1 consumer",    "All subs",      "Per group"),
        ("Message retention", "Until consumed","Until consumed", "Time/size"),
        ("Replay",           "No",            "No",             "Yes"),
        ("Ordering",         "FIFO",          "Per-topic",      "Per-partition"),
        ("Throughput",       "Medium",        "High (fan-out)", "Very high"),
        ("Back-pressure",    "Queue depth",   "Per-subscriber", "Consumer lag"),
        ("Use case",         "Task queue",    "Notifications",  "Event sourcing"),
        ("Example",          "RabbitMQ",      "Redis Pub/Sub",  "Apache Kafka"),
    ]
    for feat, q, ps, st in rows:
        print(f"  {feat:<25} {q:>15} {ps:>15} {st:>15}")

    print(f"\n  Delivery guarantees:")
    print(f"    at-most-once:  Fire and forget (fast, may lose messages)")
    print(f"    at-least-once: Retry until ack (may duplicate)")
    print(f"    exactly-once:  Idempotent consumers or transactions (complex)")


if __name__ == "__main__":
    demo_message_queue()
    demo_pubsub()
    demo_event_streaming()
    demo_comparison()
