"""
CDC Consumer — Debezium Change Event Processing
================================================
Demonstrates:
- Consuming Debezium CDC events from Kafka
- Parsing change events (INSERT/UPDATE/DELETE)
- Building a materialized view from CDC stream
- Outbox pattern event consumer
- Multi-table CDC router

Requirements:
    pip install kafka-python
"""

import json
import logging
from collections import defaultdict
from datetime import datetime

from kafka import KafkaConsumer, TopicPartition

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 1. Basic CDC Consumer ──────────────────────────────────────────

class CDCConsumer:
    """Consume and process Debezium CDC events from Kafka."""

    def __init__(self, bootstrap_servers, topics, group_id="cdc-consumer"):
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            group_id=group_id,
            value_deserializer=self._deserialize,
            key_deserializer=self._deserialize,
            max_poll_records=500,
            session_timeout_ms=30000,
        )
        self.handlers = {
            "c": self.on_insert,
            "r": self.on_insert,   # Snapshot reads treated as inserts
            "u": self.on_update,
            "d": self.on_delete,
        }
        self.stats = defaultdict(int)

    @staticmethod
    def _deserialize(data):
        if data is None:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return data

    def process_event(self, message):
        """Process a single Debezium change event."""
        value = message.value
        if value is None:
            logger.debug("Tombstone event (key=%s)", message.key)
            return

        # Handle both envelope and unwrapped formats
        payload = value.get("payload", value)
        op = payload.get("op")

        if op is None:
            # Possibly an unwrapped event (ExtractNewRecordState SMT)
            op = payload.get("__op", "c")

        handler = self.handlers.get(op)
        if handler:
            handler(payload, message)
            self.stats[op] += 1
        else:
            logger.warning("Unknown operation: %s", op)

    def on_insert(self, payload, message):
        """Handle INSERT (op=c) or SNAPSHOT (op=r) events."""
        after = payload.get("after", payload)
        source = payload.get("source", {})
        table = source.get("table", message.topic)
        logger.info("[INSERT] %s: %s", table, after)

    def on_update(self, payload, message):
        """Handle UPDATE (op=u) events."""
        before = payload.get("before", {})
        after = payload.get("after", {})
        source = payload.get("source", {})
        table = source.get("table", message.topic)

        # Identify changed fields
        changed = {
            k: {"old": before.get(k), "new": after.get(k)}
            for k in after
            if after.get(k) != before.get(k)
        }
        logger.info("[UPDATE] %s: changed=%s", table, changed)

    def on_delete(self, payload, message):
        """Handle DELETE (op=d) events."""
        before = payload.get("before", {})
        source = payload.get("source", {})
        table = source.get("table", message.topic)
        logger.info("[DELETE] %s: %s", table, before)

    def run(self, max_events=None):
        """Main consumer loop."""
        count = 0
        try:
            for message in self.consumer:
                self.process_event(message)
                self.consumer.commit()
                count += 1

                if count % 100 == 0:
                    logger.info("Processed %d events. Stats: %s", count, dict(self.stats))

                if max_events and count >= max_events:
                    break
        except KeyboardInterrupt:
            logger.info("Shutting down after %d events", count)
        finally:
            self.consumer.close()
            logger.info("Final stats: %s", dict(self.stats))


# ── 2. Materialized View ──────────────────────────────────────────

class MaterializedView:
    """Maintain an in-memory materialized view from CDC events.

    Keeps a current snapshot of the source table by applying
    INSERT/UPDATE/DELETE operations as they arrive.
    """

    def __init__(self, key_field="id"):
        self.key_field = key_field
        self.data = {}        # key → current row
        self.versions = {}    # key → version counter
        self.stats = defaultdict(int)
        self.last_updated = None

    def apply_event(self, event):
        """Apply a CDC event to the materialized view."""
        if event is None:
            return

        payload = event.get("payload", event)
        op = payload.get("op", payload.get("__op", "c"))
        before = payload.get("before")
        after = payload.get("after", payload)

        if op in ("c", "r"):
            key = after[self.key_field]
            self.data[key] = after
            self.versions[key] = self.versions.get(key, 0) + 1
            self.stats["inserts"] += 1

        elif op == "u":
            key = after[self.key_field]
            self.data[key] = after
            self.versions[key] = self.versions.get(key, 0) + 1
            self.stats["updates"] += 1

        elif op == "d":
            key = before[self.key_field] if before else None
            if key and key in self.data:
                del self.data[key]
                self.versions.pop(key, None)
            self.stats["deletes"] += 1

        self.last_updated = datetime.now()

    def query(self, predicate=None, fields=None):
        """Query the materialized view.

        Args:
            predicate: Optional filter function (row → bool)
            fields: Optional list of field names to project
        """
        rows = self.data.values()
        if predicate:
            rows = [r for r in rows if predicate(r)]
        if fields:
            rows = [{k: r.get(k) for k in fields} for r in rows]
        return list(rows)

    def get(self, key):
        """Get a single record by key."""
        return self.data.get(key)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return (f"MaterializedView(rows={len(self)}, "
                f"stats={dict(self.stats)}, "
                f"last_updated={self.last_updated})")


def run_materialized_view(bootstrap_servers, topic, key_field="id"):
    """Build and query a materialized view from CDC events."""
    view = MaterializedView(key_field=key_field)

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        group_id="materialized-view",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
    )

    logger.info("Building materialized view from topic: %s", topic)
    for message in consumer:
        view.apply_event(message.value)

        if len(view) % 100 == 0:
            logger.info("View: %s", view)

            # Example query: high-value orders
            high_value = view.query(
                predicate=lambda r: r.get("amount", 0) > 1000,
                fields=["id", "amount", "status"],
            )
            if high_value:
                logger.info("High-value orders: %d", len(high_value))


# ── 3. Multi-Table CDC Router ────────────────────────────────────

class CDCRouter:
    """Route CDC events from multiple tables to specific handlers."""

    def __init__(self):
        self._handlers = {}

    def register(self, table_name, handler_fn):
        """Register a handler for a specific table."""
        self._handlers[table_name] = handler_fn

    def route(self, event, topic):
        """Route an event to the appropriate handler."""
        if event is None:
            return

        payload = event.get("payload", event)
        source = payload.get("source", {})
        table = source.get("table", topic.split(".")[-1])

        handler = self._handlers.get(table)
        if handler:
            handler(payload)
        else:
            logger.debug("No handler for table: %s", table)


def handle_orders(payload):
    """Handle CDC events from the orders table."""
    op = payload.get("op")
    after = payload.get("after", {})
    if op == "c" and after.get("amount", 0) > 500:
        logger.info("HIGH-VALUE ORDER: %s (amount=%.2f)",
                     after.get("order_id"), after.get("amount", 0))


def handle_customers(payload):
    """Handle CDC events from the customers table."""
    op = payload.get("op")
    if op == "u":
        before = payload.get("before", {})
        after = payload.get("after", {})
        if before.get("email") != after.get("email"):
            logger.info("EMAIL CHANGED: customer %s: %s → %s",
                         after.get("customer_id"),
                         before.get("email"),
                         after.get("email"))


def run_router(bootstrap_servers, topics):
    """Run the multi-table CDC router."""
    router = CDCRouter()
    router.register("orders", handle_orders)
    router.register("customers", handle_customers)

    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        group_id="cdc-router",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
    )

    for message in consumer:
        router.route(message.value, message.topic)


# ── 4. Connector Management via REST API ──────────────────────────

def manage_connectors():
    """Manage Debezium connectors via Kafka Connect REST API.

    These are example curl commands (translated to Python requests):
    """
    import requests

    CONNECT_URL = "http://localhost:8083"

    # List connectors
    resp = requests.get(f"{CONNECT_URL}/connectors")
    print("Connectors:", resp.json())

    # Deploy connector from config file
    with open("debezium_config.json") as f:
        config = json.load(f)

    resp = requests.post(
        f"{CONNECT_URL}/connectors",
        headers={"Content-Type": "application/json"},
        json=config,
    )
    print(f"Deploy: {resp.status_code} {resp.json()}")

    # Check connector status
    name = config["name"]
    resp = requests.get(f"{CONNECT_URL}/connectors/{name}/status")
    status = resp.json()
    print(f"Status: {status['connector']['state']}")
    for task in status.get("tasks", []):
        print(f"  Task {task['id']}: {task['state']}")

    # Restart connector
    requests.post(f"{CONNECT_URL}/connectors/{name}/restart")

    # Pause / Resume
    requests.put(f"{CONNECT_URL}/connectors/{name}/pause")
    requests.put(f"{CONNECT_URL}/connectors/{name}/resume")

    # Delete connector
    # requests.delete(f"{CONNECT_URL}/connectors/{name}")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    examples = {
        "consumer": lambda: CDCConsumer(
            bootstrap_servers=["localhost:9092"],
            topics=["dbserver1.public.orders"],
        ).run(),
        "view": lambda: run_materialized_view(
            bootstrap_servers=["localhost:9092"],
            topic="dbserver1.public.orders",
            key_field="order_id",
        ),
        "router": lambda: run_router(
            bootstrap_servers=["localhost:9092"],
            topics=["dbserver1.public.orders", "dbserver1.public.customers"],
        ),
        "manage": manage_connectors,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in examples:
        print("Usage: python cdc_consumer.py <example>")
        print(f"Available: {', '.join(examples.keys())}")
        print("\n  consumer - Basic CDC event consumer")
        print("  view     - Build materialized view from CDC")
        print("  router   - Multi-table CDC router")
        print("  manage   - Manage connectors via REST API")
    else:
        examples[sys.argv[1]]()
