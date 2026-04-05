"""
Exercises for Lesson 12: Message System Comparison
Topic: System_Design

Solutions to practice problems from the lesson.
Covers message system selection, Kafka topic design, and RabbitMQ routing.
"""

import random
import time
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# === Exercise 1: System Selection ===
# Problem: Choose the appropriate message system for requirements.

def exercise_1():
    """Message system selection."""
    scenarios = [
        {
            "requirement": "Real-time stock trading system (millisecond latency)",
            "system": "Kafka or custom TCP (e.g., Aeron)",
            "reason": "Kafka provides consistent low latency at scale. "
                      "For ultra-low latency (<1ms), consider Aeron or ZeroMQ. "
                      "Kafka's persistent log enables replay for audit.",
        },
        {
            "requirement": "E-commerce order event log (3-year retention)",
            "system": "Kafka",
            "reason": "Kafka's persistent log with configurable retention. "
                      "Can retain events for years. Supports replay for "
                      "analytics and rebuilding derived views.",
        },
        {
            "requirement": "Image resizing work queue (priority needed)",
            "system": "RabbitMQ",
            "reason": "RabbitMQ has built-in priority queue support. "
                      "AMQP routing features for different priority levels. "
                      "Work queue pattern with acknowledgment.",
        },
        {
            "requirement": "Serverless-based startup MVP",
            "system": "AWS SQS/SNS",
            "reason": "Zero operational overhead - fully managed. "
                      "Pay-per-use pricing perfect for startups. "
                      "Easy integration with Lambda and other AWS services.",
        },
        {
            "requirement": "IoT sensor data collection (1M events/second)",
            "system": "Kafka",
            "reason": "Handles millions of events/second with horizontal scaling. "
                      "Partitions across multiple brokers. "
                      "Durable storage for batch analytics.",
        },
    ]

    print("Message System Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        print(f"\n{i}. {s['requirement']}")
        print(f"   System: {s['system']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 2: Kafka Topic Design ===
# Problem: Design Kafka topics for online shopping order system.

class KafkaTopicSimulator:
    """Simplified Kafka topic simulator."""

    def __init__(self, name, num_partitions, retention_hours=168):
        self.name = name
        self.num_partitions = num_partitions
        self.retention_hours = retention_hours
        self.partitions = {i: [] for i in range(num_partitions)}
        self.consumer_groups = {}

    def produce(self, key, value):
        """Produce message to partition based on key hash."""
        partition_id = hash(key) % self.num_partitions
        offset = len(self.partitions[partition_id])
        self.partitions[partition_id].append({
            "key": key, "value": value, "offset": offset,
            "timestamp": time.time()
        })
        return partition_id, offset

    def add_consumer_group(self, group_name, num_consumers):
        """Add a consumer group with partition assignment."""
        assignment = {}
        for i in range(num_consumers):
            assigned_partitions = [
                p for p in range(self.num_partitions)
                if p % num_consumers == i
            ]
            assignment[f"{group_name}-{i}"] = assigned_partitions
        self.consumer_groups[group_name] = assignment


def exercise_2():
    """Kafka topic design for e-commerce order system."""
    print("Kafka Topic Design: E-Commerce Orders:")
    print("=" * 60)

    # Topic design
    topics = {
        "orders": {
            "partitions": 12,
            "partition_key": "user_id",
            "retention": "7 days",
            "reason": "12 partitions for parallelism. Partition by user_id "
                      "ensures all orders from one user go to same partition "
                      "(preserves ordering per user).",
        },
        "order-events": {
            "partitions": 12,
            "partition_key": "order_id",
            "retention": "30 days",
            "reason": "Events (created, paid, shipped, delivered) for an order "
                      "go to same partition. Longer retention for analytics.",
        },
        "inventory-updates": {
            "partitions": 6,
            "partition_key": "product_id",
            "retention": "3 days",
            "reason": "Product inventory changes co-located by product. "
                      "Fewer partitions since inventory is lower volume.",
        },
        "notifications": {
            "partitions": 8,
            "partition_key": "user_id",
            "retention": "1 day",
            "reason": "Email/SMS/push notification events. "
                      "Short retention since notifications are fire-and-forget.",
        },
    }

    for topic_name, config in topics.items():
        print(f"\n  Topic: {topic_name}")
        print(f"    Partitions: {config['partitions']}")
        print(f"    Partition Key: {config['partition_key']}")
        print(f"    Retention: {config['retention']}")
        print(f"    Reason: {config['reason']}")

    # Consumer group design
    print("\n  Consumer Groups:")
    groups = [
        ("order-processor", "orders", 4,
         "Processes new orders, validates, sends to payment"),
        ("inventory-updater", "order-events", 3,
         "Updates inventory on order creation/cancellation"),
        ("notification-sender", "order-events", 6,
         "Sends notifications on order status changes"),
        ("analytics-consumer", "order-events", 2,
         "Aggregates order metrics for dashboards"),
    ]

    for group, topic, consumers, desc in groups:
        print(f"    {group} (topic={topic}, consumers={consumers}): {desc}")

    # Simulate message distribution
    print("\n--- Simulation: Order Distribution ---")
    topic = KafkaTopicSimulator("orders", num_partitions=12)

    random.seed(42)
    for _ in range(1000):
        user_id = f"user_{random.randint(1, 200)}"
        order_id = f"ORD-{random.randint(10000, 99999)}"
        partition, offset = topic.produce(user_id, {"order_id": order_id})

    print("  Partition distribution (1000 orders, 200 users):")
    for pid in range(12):
        count = len(topic.partitions[pid])
        bar = "#" * (count // 5)
        print(f"    P{pid:2d}: {count:3d} messages  [{bar}]")


# === Exercise 3: RabbitMQ Routing Design ===
# Problem: Design Exchange/Queue for a log collection system.

@dataclass
class Message:
    routing_key: str
    body: str
    headers: Dict = field(default_factory=dict)


class Exchange:
    """Simulated RabbitMQ Exchange."""
    def __init__(self, name, exchange_type="topic"):
        self.name = name
        self.type = exchange_type
        self.bindings = []  # (pattern, queue)

    def bind(self, queue, routing_key_pattern):
        self.bindings.append((routing_key_pattern, queue))

    def publish(self, message):
        """Route message to matching queues based on exchange type."""
        for pattern, queue in self.bindings:
            if self._matches(message.routing_key, pattern):
                queue.enqueue(message)

    def _matches(self, routing_key, pattern):
        """Match routing key against pattern with * and # wildcards."""
        key_parts = routing_key.split(".")
        pattern_parts = pattern.split(".")

        ki, pi = 0, 0
        while ki < len(key_parts) and pi < len(pattern_parts):
            if pattern_parts[pi] == "#":
                return True  # # matches zero or more words
            elif pattern_parts[pi] == "*":
                ki += 1
                pi += 1
            elif pattern_parts[pi] == key_parts[ki]:
                ki += 1
                pi += 1
            else:
                return False

        return ki == len(key_parts) and pi == len(pattern_parts)


class Queue:
    """Simulated RabbitMQ Queue."""
    def __init__(self, name):
        self.name = name
        self.messages = []

    def enqueue(self, message):
        self.messages.append(message)

    def dequeue(self):
        return self.messages.pop(0) if self.messages else None

    def __len__(self):
        return len(self.messages)


def exercise_3():
    """RabbitMQ routing design for log collection."""
    print("RabbitMQ Routing: Log Collection System:")
    print("=" * 60)

    # Create exchange
    log_exchange = Exchange("logs", exchange_type="topic")

    # Create queues
    error_notification_queue = Queue("error-notifications")
    web_debug_queue = Queue("web-debugging")
    all_errors_queue = Queue("all-errors")
    all_logs_queue = Queue("all-logs")
    api_logs_queue = Queue("api-logs")

    # Bind queues with routing patterns
    # Routing key format: <service>.<level>
    # e.g., web.error, api.warn, db.info

    # All error logs to notification service
    log_exchange.bind(error_notification_queue, "*.error")

    # All logs from web service to debugging service
    log_exchange.bind(web_debug_queue, "web.#")

    # All error logs (duplicate binding for separate processing)
    log_exchange.bind(all_errors_queue, "*.error")

    # All logs from api service
    log_exchange.bind(api_logs_queue, "api.#")

    # All logs (catch-all)
    log_exchange.bind(all_logs_queue, "#")

    # Publish test messages
    test_messages = [
        Message("web.error", "NullPointerException in handler"),
        Message("web.warn", "Slow query detected: 500ms"),
        Message("web.info", "User login: user_123"),
        Message("api.error", "Timeout connecting to payment service"),
        Message("api.info", "Request processed: POST /orders"),
        Message("db.error", "Connection pool exhausted"),
        Message("db.warn", "Slow query: SELECT * FROM orders"),
        Message("web.debug", "Request params: {id: 123}"),
    ]

    print("\n  Publishing messages:")
    for msg in test_messages:
        log_exchange.publish(msg)
        print(f"    [{msg.routing_key}] {msg.body}")

    print(f"\n  Queue message counts:")
    for queue in [error_notification_queue, web_debug_queue,
                  all_errors_queue, api_logs_queue, all_logs_queue]:
        print(f"    {queue.name}: {len(queue)} messages")
        for msg in queue.messages:
            print(f"      [{msg.routing_key}] {msg.body[:50]}")

    print("\n  Routing Design Summary:")
    print("    Exchange: logs (topic type)")
    print("    Routing key format: <service>.<level>")
    print("    Bindings:")
    print("      *.error       -> error-notifications (alert on all errors)")
    print("      web.#         -> web-debugging (all web service logs)")
    print("      api.#         -> api-logs (all API service logs)")
    print("      #             -> all-logs (archive/analytics)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Message System Selection ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Kafka Topic Design ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: RabbitMQ Routing Design ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
