"""
Exercise Solutions: Lesson 10 - Kafka Streaming

Covers:
  - Problem 1: Producer/Consumer (order events)
  - Problem 2: Consumer Group (3 consumers, partition assignment)
  - Problem 3: Real-Time Aggregation (total sales per minute)

Note: Pure Python simulation of Kafka concepts using queues and threads.
"""

import json
import random
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from queue import Queue


# ---------------------------------------------------------------------------
# Simulated Kafka primitives
# ---------------------------------------------------------------------------

class SimulatedPartition:
    """A single partition in a Kafka topic.

    In Kafka, a partition is an ordered, immutable sequence of messages.
    Each message has an offset (its position in the partition).
    """
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.messages: list[dict] = []
        self.lock = threading.Lock()

    def append(self, message: dict) -> int:
        """Append a message and return its offset."""
        with self.lock:
            offset = len(self.messages)
            message["_offset"] = offset
            message["_partition"] = self.partition_id
            self.messages.append(message)
            return offset

    def read(self, offset: int, max_messages: int = 10) -> list[dict]:
        """Read messages starting from the given offset."""
        with self.lock:
            return self.messages[offset:offset + max_messages]


class SimulatedTopic:
    """A Kafka topic with multiple partitions.

    Messages are distributed across partitions using:
    - The message key (hash(key) % num_partitions) if key is provided
    - Round-robin if no key
    """
    def __init__(self, name: str, num_partitions: int = 3):
        self.name = name
        self.partitions = [SimulatedPartition(i) for i in range(num_partitions)]
        self._round_robin = 0

    def send(self, key: str | None, value: dict) -> tuple[int, int]:
        """Send a message to the topic. Returns (partition_id, offset).

        Key-based partitioning ensures all messages with the same key
        go to the same partition (preserving per-key ordering).
        """
        if key is not None:
            partition_id = hash(key) % len(self.partitions)
        else:
            partition_id = self._round_robin % len(self.partitions)
            self._round_robin += 1
        offset = self.partitions[partition_id].append(value)
        return partition_id, offset


class SimulatedConsumer:
    """Simulates a Kafka Consumer with offset tracking."""
    def __init__(self, consumer_id: str, group_id: str):
        self.consumer_id = consumer_id
        self.group_id = group_id
        self.assigned_partitions: list[int] = []
        self.offsets: dict[int, int] = {}  # partition_id -> current offset
        self.consumed: list[dict] = []

    def poll(self, topic: SimulatedTopic, max_messages: int = 5) -> list[dict]:
        """Poll messages from assigned partitions."""
        batch = []
        for pid in self.assigned_partitions:
            offset = self.offsets.get(pid, 0)
            messages = topic.partitions[pid].read(offset, max_messages)
            for msg in messages:
                batch.append(msg)
                self.consumed.append(msg)
            self.offsets[pid] = offset + len(messages)
        return batch


# ---------------------------------------------------------------------------
# Problem 1: Producer/Consumer
# Write a Producer that generates order events and a Consumer that
# consumes them.
# ---------------------------------------------------------------------------

def problem1_producer_consumer():
    """
    Kafka Producer equivalent (confluent-kafka-python):

        from confluent_kafka import Producer
        producer = Producer({'bootstrap.servers': 'localhost:9092'})
        for order in orders:
            producer.produce(
                'orders',
                key=order['customer_id'],
                value=json.dumps(order),
                callback=delivery_report,
            )
        producer.flush()

    Kafka Consumer equivalent:

        from confluent_kafka import Consumer
        consumer = Consumer({
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'order-processor',
            'auto.offset.reset': 'earliest',
        })
        consumer.subscribe(['orders'])
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            order = json.loads(msg.value())
            print(f"Received: {order}")
    """
    topic = SimulatedTopic("orders", num_partitions=3)

    # --- Producer ---
    print("\n  --- Producer ---")
    customers = [f"customer_{i}" for i in range(1, 6)]
    products = ["Laptop", "Phone", "Tablet", "Watch", "Headphones"]

    produced_count = 0
    for i in range(15):
        customer_id = random.choice(customers)
        order = {
            "order_id": f"ORD-{i+1:04d}",
            "customer_id": customer_id,
            "product": random.choice(products),
            "amount": round(random.uniform(50, 1000), 2),
            "timestamp": datetime.now().isoformat(),
        }
        # Key = customer_id ensures all orders from the same customer
        # go to the same partition (preserving per-customer ordering)
        part_id, offset = topic.send(key=customer_id, value=order)
        produced_count += 1
        if i < 5:  # Show first 5
            print(f"    Produced: {order['order_id']} -> partition={part_id}, offset={offset}")
    print(f"    ... produced {produced_count} orders total")

    # --- Consumer ---
    print("\n  --- Consumer ---")
    consumer = SimulatedConsumer("consumer-1", "order-processor")
    consumer.assigned_partitions = list(range(len(topic.partitions)))

    all_messages = consumer.poll(topic, max_messages=100)
    print(f"    Consumed {len(all_messages)} orders:")
    for msg in all_messages[:5]:
        print(f"      partition={msg['_partition']}, offset={msg['_offset']}, "
              f"order_id={msg['order_id']}, customer={msg['customer_id']}, "
              f"amount=${msg['amount']:.2f}")
    if len(all_messages) > 5:
        print(f"      ... ({len(all_messages) - 5} more)")

    # Show partition distribution
    part_counts = defaultdict(int)
    for msg in all_messages:
        part_counts[msg["_partition"]] += 1
    print(f"\n    Partition distribution: {dict(sorted(part_counts.items()))}")

    return all_messages


# ---------------------------------------------------------------------------
# Problem 2: Consumer Group
# Create a Consumer Group with 3 consumers and verify partition assignment.
# ---------------------------------------------------------------------------

def assign_partitions(consumers: list[SimulatedConsumer],
                      num_partitions: int) -> None:
    """Simulate Kafka's partition assignment (Range Assignor).

    Kafka's Range Assignor divides partitions among consumers:
    - n_partitions / n_consumers partitions per consumer (integer division)
    - The first n_partitions % n_consumers consumers get one extra partition

    Important: Each partition is assigned to exactly ONE consumer in a group.
    This is how Kafka achieves parallel consumption without duplicates.
    """
    partitions = list(range(num_partitions))
    n = len(consumers)
    for c in consumers:
        c.assigned_partitions = []
        c.offsets = {}

    for i, pid in enumerate(partitions):
        consumer_idx = i % n
        consumers[consumer_idx].assigned_partitions.append(pid)
        consumers[consumer_idx].offsets[pid] = 0


def problem2_consumer_group():
    """
    Kafka equivalent:

        # All three consumers share the same group.id
        for i in range(3):
            consumer = Consumer({
                'bootstrap.servers': 'localhost:9092',
                'group.id': 'order-processors',  # Same group
                'auto.offset.reset': 'earliest',
            })
            consumer.subscribe(['orders'])

    Kafka automatically assigns partitions to consumers within a group.
    When a consumer joins or leaves, rebalancing occurs.
    """
    NUM_PARTITIONS = 6
    topic = SimulatedTopic("orders", num_partitions=NUM_PARTITIONS)

    # Produce 30 messages
    for i in range(30):
        topic.send(
            key=f"customer_{random.randint(1, 10)}",
            value={"order_id": f"ORD-{i+1:04d}", "amount": round(random.uniform(50, 500), 2)},
        )

    # Create 3 consumers in the same group
    consumers = [
        SimulatedConsumer(f"consumer-{i+1}", "order-processors")
        for i in range(3)
    ]

    # Assign partitions (simulates Kafka's rebalancing)
    assign_partitions(consumers, NUM_PARTITIONS)

    print(f"\n  Topic '{topic.name}' has {NUM_PARTITIONS} partitions")
    print(f"  Consumer group 'order-processors' has {len(consumers)} consumers\n")

    print("  Partition Assignment:")
    for c in consumers:
        print(f"    {c.consumer_id}: partitions {c.assigned_partitions}")

    # Each consumer polls
    print("\n  Consumption Results:")
    total_consumed = 0
    for c in consumers:
        messages = c.poll(topic, max_messages=100)
        total_consumed += len(messages)
        print(f"    {c.consumer_id}: consumed {len(messages)} messages "
              f"from partitions {c.assigned_partitions}")

    print(f"\n  Total consumed: {total_consumed} (should = 30)")
    print(f"  No duplicates: each partition assigned to exactly one consumer")

    # Simulate rebalancing when a consumer leaves
    print("\n  --- Simulating Rebalance (consumer-3 leaves) ---")
    remaining = consumers[:2]
    assign_partitions(remaining, NUM_PARTITIONS)
    for c in remaining:
        print(f"    {c.consumer_id}: now assigned partitions {c.assigned_partitions}")
    print(f"    Note: partitions from consumer-3 were redistributed")

    return consumers


# ---------------------------------------------------------------------------
# Problem 3: Real-Time Aggregation
# Calculate total sales revenue per minute from real-time sales events.
# ---------------------------------------------------------------------------

def problem3_realtime_aggregation():
    """Simulate real-time per-minute sales aggregation.

    In a real Kafka Streams / Faust application:

        import faust
        app = faust.App('sales-agg', broker='kafka://localhost:9092')

        class Sale(faust.Record):
            sale_id: str
            amount: float
            timestamp: str

        sales_topic = app.topic('sales', value_type=Sale)
        minute_totals = app.Table('minute_totals', default=float)

        @app.agent(sales_topic)
        async def aggregate_sales(stream):
            async for sale in stream:
                minute_key = sale.timestamp[:16]  # '2024-01-01T12:05'
                minute_totals[minute_key] += sale.amount
                print(f'{minute_key}: ${minute_totals[minute_key]:.2f}')
    """
    topic = SimulatedTopic("sales", num_partitions=2)

    # Produce sales events spread across 5 minutes
    base_time = datetime(2024, 11, 15, 14, 0, 0)
    events = []
    for i in range(100):
        offset_seconds = random.randint(0, 299)  # 5 minutes = 300 seconds
        event_time = base_time + timedelta(seconds=offset_seconds)
        event = {
            "sale_id": f"SALE-{i+1:04d}",
            "amount": round(random.uniform(10, 200), 2),
            "product": random.choice(["A", "B", "C", "D"]),
            "timestamp": event_time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        topic.send(key=None, value=event)
        events.append(event)

    print(f"\n  Produced {len(events)} sales events across 5 minutes")

    # Consume and aggregate per minute
    consumer = SimulatedConsumer("aggregator-1", "sales-agg")
    consumer.assigned_partitions = list(range(len(topic.partitions)))
    messages = consumer.poll(topic, max_messages=200)

    # Per-minute aggregation (tumbling window = 1 minute)
    minute_agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total": 0.0, "amounts": []})
    for msg in messages:
        minute_key = msg["timestamp"][:16]  # Truncate to minute
        minute_agg[minute_key]["count"] += 1
        minute_agg[minute_key]["total"] += msg["amount"]
        minute_agg[minute_key]["amounts"].append(msg["amount"])

    print(f"\n  Per-Minute Sales Aggregation (Tumbling Window = 1 min):")
    print(f"  {'Minute':<20} {'Count':>6} {'Total Revenue':>15} {'Avg Sale':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*15} {'-'*10}")

    grand_total = 0.0
    for minute_key in sorted(minute_agg.keys()):
        stats = minute_agg[minute_key]
        avg_sale = stats["total"] / stats["count"] if stats["count"] > 0 else 0
        grand_total += stats["total"]
        print(f"  {minute_key:<20} {stats['count']:>6} ${stats['total']:>13,.2f} ${avg_sale:>8,.2f}")

    print(f"  {'-'*20} {'-'*6} {'-'*15}")
    print(f"  {'TOTAL':<20} {len(messages):>6} ${grand_total:>13,.2f}")

    return dict(minute_agg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Producer/Consumer (Order Events)")
    print("=" * 70)
    problem1_producer_consumer()

    print()
    print("=" * 70)
    print("Problem 2: Consumer Group (3 Consumers)")
    print("=" * 70)
    problem2_consumer_group()

    print()
    print("=" * 70)
    print("Problem 3: Real-Time Aggregation (Sales per Minute)")
    print("=" * 70)
    problem3_realtime_aggregation()
