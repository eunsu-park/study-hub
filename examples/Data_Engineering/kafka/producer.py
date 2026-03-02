"""
Kafka Producer Example

An example of a Producer that publishes messages to Kafka topics.

Required package: pip install confluent-kafka

Kafka must be running before execution:
  docker run -d --name kafka -p 9092:9092 \
    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
    confluentinc/cp-kafka:latest

Run: python producer.py
"""

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
import json
import time
import random
from datetime import datetime
from typing import Optional


class KafkaProducerExample:
    """Kafka Producer example class"""

    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'example-producer',
            'acks': 'all',  # Wait for all replicas to acknowledge
            'retries': 3,
            'retry.backoff.ms': 100,
            'linger.ms': 5,  # Batch wait time
            'batch.size': 16384,  # Batch size
        }
        self.producer = Producer(self.config)
        self.message_count = 0
        self.error_count = 0

    def delivery_callback(self, err, msg):
        """Message delivery result callback"""
        if err:
            print(f'[ERROR] Message delivery failed: {err}')
            self.error_count += 1
        else:
            self.message_count += 1
            if self.message_count % 100 == 0:
                print(f'[INFO] Delivered {self.message_count} messages')

    def create_topic(self, topic_name: str, num_partitions: int = 3, replication_factor: int = 1):
        """Create a topic"""
        admin_client = AdminClient({'bootstrap.servers': self.config['bootstrap.servers']})

        topic = NewTopic(
            topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )

        try:
            futures = admin_client.create_topics([topic])
            futures[topic_name].result()  # Wait for creation to complete
            print(f'[INFO] Topic "{topic_name}" created')
        except Exception as e:
            if 'already exists' in str(e):
                print(f'[INFO] Topic "{topic_name}" already exists')
            else:
                raise e

    def send_message(self, topic: str, key: Optional[str], value: dict):
        """Send a single message"""
        try:
            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8') if key else None,
                value=json.dumps(value).encode('utf-8'),
                callback=self.delivery_callback
            )
            # Periodically process events
            self.producer.poll(0)
        except BufferError:
            # Wait if buffer is full
            print('[WARN] Buffer full, waiting...')
            self.producer.flush()
            self.send_message(topic, key, value)

    def flush(self):
        """Wait for all messages to be delivered"""
        self.producer.flush()

    def close(self):
        """Close the producer"""
        self.flush()
        print(f'\n[SUMMARY] Total sent: {self.message_count}, Errors: {self.error_count}')


def generate_order_event() -> dict:
    """Generate an order event"""
    products = ['laptop', 'phone', 'tablet', 'headphones', 'keyboard', 'mouse']
    statuses = ['created', 'confirmed', 'shipped', 'delivered']

    return {
        'event_type': 'order',
        'order_id': f'ORD-{random.randint(10000, 99999)}',
        'customer_id': f'CUST-{random.randint(1, 1000)}',
        'product': random.choice(products),
        'quantity': random.randint(1, 5),
        'amount': round(random.uniform(10, 1000), 2),
        'status': random.choice(statuses),
        'timestamp': datetime.now().isoformat(),
    }


def generate_clickstream_event() -> dict:
    """Generate a clickstream event"""
    pages = ['/home', '/products', '/cart', '/checkout', '/profile', '/search']
    actions = ['view', 'click', 'scroll', 'hover']

    return {
        'event_type': 'clickstream',
        'event_id': f'EVT-{random.randint(100000, 999999)}',
        'user_id': f'USER-{random.randint(1, 500)}',
        'session_id': f'SESS-{random.randint(1000, 9999)}',
        'page': random.choice(pages),
        'action': random.choice(actions),
        'timestamp': datetime.now().isoformat(),
    }


def generate_inventory_event() -> dict:
    """Generate an inventory event"""
    products = ['SKU-001', 'SKU-002', 'SKU-003', 'SKU-004', 'SKU-005']
    warehouses = ['WH-EAST', 'WH-WEST', 'WH-CENTRAL']

    return {
        'event_type': 'inventory',
        'product_sku': random.choice(products),
        'warehouse': random.choice(warehouses),
        'quantity_change': random.randint(-50, 100),
        'current_stock': random.randint(0, 500),
        'timestamp': datetime.now().isoformat(),
    }


def demo_simple_producer():
    """Simple Producer demo"""
    print("=" * 60)
    print("Simple Producer Demo")
    print("=" * 60)

    producer = KafkaProducerExample()

    # Create topic
    producer.create_topic('demo-topic')

    # Send messages
    for i in range(10):
        message = {
            'id': i,
            'message': f'Hello Kafka #{i}',
            'timestamp': datetime.now().isoformat()
        }
        producer.send_message(
            topic='demo-topic',
            key=f'key-{i % 3}',  # Distribute across 3 partitions
            value=message
        )
        print(f'Sent: {message}')

    producer.close()


def demo_event_stream():
    """Event stream demo"""
    print("\n" + "=" * 60)
    print("Event Stream Demo")
    print("=" * 60)

    producer = KafkaProducerExample()

    # Create topics
    producer.create_topic('orders', num_partitions=3)
    producer.create_topic('clickstream', num_partitions=6)
    producer.create_topic('inventory', num_partitions=3)

    print("\nStreaming events (Press Ctrl+C to stop)...")

    try:
        event_count = 0
        while True:
            # Order events (low frequency)
            if random.random() < 0.3:
                event = generate_order_event()
                producer.send_message('orders', event['customer_id'], event)

            # Clickstream events (high frequency)
            for _ in range(random.randint(1, 5)):
                event = generate_clickstream_event()
                producer.send_message('clickstream', event['user_id'], event)

            # Inventory events (medium frequency)
            if random.random() < 0.5:
                event = generate_inventory_event()
                producer.send_message('inventory', event['product_sku'], event)

            event_count += 1
            if event_count % 50 == 0:
                print(f'Generated {event_count} event batches')

            time.sleep(0.1)  # 100ms interval

    except KeyboardInterrupt:
        print("\nStopping...")

    producer.close()


def demo_batch_producer():
    """Batch Producer demo"""
    print("\n" + "=" * 60)
    print("Batch Producer Demo")
    print("=" * 60)

    producer = KafkaProducerExample()
    producer.create_topic('batch-orders')

    batch_size = 1000
    print(f"\nSending {batch_size} messages in batch...")

    start_time = time.time()

    for i in range(batch_size):
        event = generate_order_event()
        producer.send_message('batch-orders', event['order_id'], event)

    producer.flush()
    elapsed = time.time() - start_time

    print(f"\nBatch completed:")
    print(f"  Messages: {batch_size}")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Throughput: {batch_size / elapsed:.0f} messages/second")

    producer.close()


def main():
    print("Kafka Producer Examples")
    print("=" * 60)
    print("Make sure Kafka is running on localhost:9092")
    print()

    # Select demo
    demos = {
        '1': ('Simple Producer', demo_simple_producer),
        '2': ('Event Stream', demo_event_stream),
        '3': ('Batch Producer', demo_batch_producer),
    }

    print("Available demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")

    choice = input("\nSelect demo (1-3) or 'all': ").strip()

    if choice == 'all':
        for name, func in demos.values():
            func()
    elif choice in demos:
        demos[choice][1]()
    else:
        print("Invalid choice, running simple demo...")
        demo_simple_producer()


if __name__ == "__main__":
    main()
