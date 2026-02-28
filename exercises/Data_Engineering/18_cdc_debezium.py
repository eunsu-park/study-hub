"""
Exercise Solutions: Lesson 18 - CDC with Debezium

Covers:
  - Exercise 1: Deploy and Verify a Debezium Connector
  - Exercise 2: Build a Materialized Cache (Multi-Table Consumer)
  - Exercise 3: Apply Single Message Transforms (SMTs)
  - Exercise 4: Schema Evolution and Registry Integration
  - Exercise 5: Implement the Transactional Outbox Pattern

Note: Pure Python simulation of CDC/Debezium concepts.
"""

import json
import uuid
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from copy import deepcopy


# ---------------------------------------------------------------------------
# Simulated CDC primitives
# ---------------------------------------------------------------------------

@dataclass
class CDCEvent:
    """Simulates a Debezium CDC event envelope.

    Debezium event structure:
    {
        "before": {...} or null,   -- row state before the change
        "after":  {...} or null,   -- row state after the change
        "source": { "table": "...", "ts_ms": ..., ... },
        "op":     "c" | "u" | "d" | "r",  -- create, update, delete, read(snapshot)
        "ts_ms":  1234567890123
    }
    """
    before: dict | None
    after: dict | None
    source: dict
    op: str  # c=create, u=update, d=delete, r=read(snapshot)
    ts_ms: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))


def make_cdc_event(table: str, op: str, before: dict | None = None,
                   after: dict | None = None) -> CDCEvent:
    return CDCEvent(
        before=before, after=after, op=op,
        source={"table": table, "schema": "public", "db": "inventory"},
    )


# ---------------------------------------------------------------------------
# Exercise 1: Deploy and Verify a Debezium Connector
# ---------------------------------------------------------------------------

def exercise1_deploy_verify():
    """Simulate deploying a Debezium connector and verifying CDC events.

    REPLICA IDENTITY FULL is important because:
    - With DEFAULT identity, UPDATE events only include the primary key in `before`.
    - DELETE events only include the primary key in `before`.
    - This means you lose the full old row data needed for SCD Type 2 or auditing.
    - FULL identity captures ALL column values in both `before` and `after`.
    """
    print("\n  Step 1: Docker Compose stack (PostgreSQL, Zookeeper, Kafka, Kafka Connect)")
    print("    docker-compose up -d")

    print("\n  Step 2: Create orders table")
    print("""    CREATE TABLE orders (
        order_id   SERIAL PRIMARY KEY,
        customer   VARCHAR(100),
        amount     NUMERIC(10,2),
        status     VARCHAR(20),
        created_at TIMESTAMP DEFAULT NOW()
    );
    ALTER TABLE orders REPLICA IDENTITY FULL;""")

    print("\n  Step 3: Register Debezium connector")
    connector_config = {
        "name": "inventory-connector",
        "config": {
            "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
            "database.hostname": "postgres",
            "database.port": "5432",
            "database.user": "debezium",
            "database.password": "dbz",
            "database.dbname": "inventory",
            "database.server.name": "dbserver1",
            "table.include.list": "public.orders",
            "plugin.name": "pgoutput",
        },
    }
    print(f"    POST /connectors: {json.dumps(connector_config, indent=4)}")

    print("\n  Step 4 & 5: Perform operations and observe CDC events")

    # Simulate INSERT
    insert_event = make_cdc_event(
        "orders", op="c",
        before=None,
        after={"order_id": 1, "customer": "Alice", "amount": 150.00,
               "status": "pending", "created_at": "2024-11-15T10:00:00"},
    )
    print(f"\n    INSERT:")
    print(f"      op = '{insert_event.op}' (c = create)")
    print(f"      before = {insert_event.before}  (NULL for inserts)")
    print(f"      after  = {insert_event.after}")

    # Simulate UPDATE
    update_event = make_cdc_event(
        "orders", op="u",
        before={"order_id": 1, "customer": "Alice", "amount": 150.00,
                "status": "pending", "created_at": "2024-11-15T10:00:00"},
        after={"order_id": 1, "customer": "Alice", "amount": 150.00,
               "status": "completed", "created_at": "2024-11-15T10:00:00"},
    )
    print(f"\n    UPDATE:")
    print(f"      op = '{update_event.op}' (u = update)")
    print(f"      before = {update_event.before}  (full row with REPLICA IDENTITY FULL)")
    print(f"      after  = {update_event.after}")

    # Simulate DELETE
    delete_event = make_cdc_event(
        "orders", op="d",
        before={"order_id": 1, "customer": "Alice", "amount": 150.00,
                "status": "completed", "created_at": "2024-11-15T10:00:00"},
        after=None,
    )
    print(f"\n    DELETE:")
    print(f"      op = '{delete_event.op}' (d = delete)")
    print(f"      before = {delete_event.before}  (full row with REPLICA IDENTITY FULL)")
    print(f"      after  = {delete_event.after}  (NULL for deletes)")

    print(f"\n  Step 6: Why REPLICA IDENTITY FULL matters:")
    print(f"    Without FULL, UPDATE 'before' only contains the primary key.")
    print(f"    Without FULL, DELETE 'before' only contains the primary key.")
    print(f"    You lose the old column values needed for auditing and SCD Type 2.")

    return [insert_event, update_event, delete_event]


# ---------------------------------------------------------------------------
# Exercise 2: Build a Materialized Cache (Multi-Table Consumer)
# ---------------------------------------------------------------------------

class MaterializedView:
    """In-memory materialized view maintained by CDC events."""
    def __init__(self, table_name: str, primary_key: str):
        self.table_name = table_name
        self.primary_key = primary_key
        self.data: dict[str, dict] = {}
        self.stats = {"inserts": 0, "updates": 0, "deletes": 0}

    def apply_event(self, event: CDCEvent) -> None:
        if event.op in ("c", "r"):  # create or snapshot read
            pk = str(event.after[self.primary_key])
            self.data[pk] = event.after
            self.stats["inserts"] += 1
        elif event.op == "u":
            pk = str(event.after[self.primary_key])
            self.data[pk] = event.after
            self.stats["updates"] += 1
        elif event.op == "d":
            pk = str(event.before[self.primary_key])
            self.data.pop(pk, None)
            self.stats["deletes"] += 1


class MultiTableConsumer:
    """Consumes CDC events from multiple Debezium topics and routes to views."""
    def __init__(self):
        self.views: dict[str, MaterializedView] = {}

    def register_view(self, table_name: str, primary_key: str) -> None:
        self.views[table_name] = MaterializedView(table_name, primary_key)

    def process_event(self, event: CDCEvent) -> None:
        table = event.source.get("table")
        if table in self.views:
            self.views[table].apply_event(event)

    def query_join(self, order_id: str) -> dict | None:
        """Cross-table join: look up order and its customer."""
        orders_view = self.views.get("orders")
        customers_view = self.views.get("customers")
        if not orders_view or not customers_view:
            return None
        order = orders_view.data.get(order_id)
        if not order:
            return None
        customer = customers_view.data.get(str(order.get("customer_id")))
        return {"order": order, "customer": customer}

    def stats(self) -> dict:
        return {table: view.stats for table, view in self.views.items()}


def exercise2_materialized_cache():
    """Build a multi-table materialized view from CDC events."""
    consumer = MultiTableConsumer()
    consumer.register_view("orders", "order_id")
    consumer.register_view("customers", "customer_id")

    # Insert 5 customers
    for i in range(1, 6):
        consumer.process_event(make_cdc_event(
            "customers", "c",
            after={"customer_id": i, "name": f"Customer_{i}", "email": f"c{i}@example.com"},
        ))

    # Insert 10 orders
    for i in range(1, 11):
        consumer.process_event(make_cdc_event(
            "orders", "c",
            after={"order_id": i, "customer_id": (i % 5) + 1,
                   "amount": round(50 + i * 10, 2), "status": "pending"},
        ))

    # Update 3 orders
    for i in [1, 3, 7]:
        old = consumer.views["orders"].data[str(i)]
        new = dict(old)
        new["status"] = "completed"
        consumer.process_event(make_cdc_event("orders", "u", before=old, after=new))

    # Delete 1 customer
    deleted_customer = consumer.views["customers"].data["3"]
    consumer.process_event(make_cdc_event("customers", "d", before=deleted_customer))

    # Display views
    print(f"\n  Orders View ({len(consumer.views['orders'].data)} rows):")
    for pk, row in sorted(consumer.views["orders"].data.items(), key=lambda x: int(x[0])):
        print(f"    order_id={pk}: customer={row['customer_id']}, "
              f"amount=${row['amount']}, status={row['status']}")

    print(f"\n  Customers View ({len(consumer.views['customers'].data)} rows):")
    for pk, row in sorted(consumer.views["customers"].data.items(), key=lambda x: int(x[0])):
        print(f"    customer_id={pk}: {row['name']}, {row['email']}")

    # Cross-table join
    join_result = consumer.query_join("1")
    print(f"\n  query_join('1'): {join_result}")

    # Stats
    print(f"\n  Stats: {consumer.stats()}")
    return consumer


# ---------------------------------------------------------------------------
# Exercise 3: Apply Single Message Transforms (SMTs)
# ---------------------------------------------------------------------------

def exercise3_smts():
    """Simulate three Debezium SMTs chained together.

    ExtractNewRecordState vs full envelope:
    - Use ExtractNewRecordState when consumers only need the current row state
      (e.g., loading into a data warehouse, populating a cache).
    - Keep the full envelope when consumers need both before/after states
      (e.g., SCD Type 2, audit logging, detecting which columns changed).
    """
    print("\n  Debezium SMT Chain Configuration:")
    print("""    {
        "transforms": "route,unwrap,addTimestamp",
        "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
        "transforms.route.regex": "dbserver1\\.public\\.(.*)",
        "transforms.route.replacement": "cdc.$1",
        "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
        "transforms.unwrap.add.fields": "op,source.ts_ms",
        "transforms.addTimestamp.type": "org.apache.kafka.connect.transforms.InsertField$Value",
        "transforms.addTimestamp.timestamp.field": "cdc_processed_at"
    }""")

    # Original event
    original = make_cdc_event(
        "orders", "u",
        before={"order_id": 1, "customer": "Alice", "amount": 150.00, "status": "pending"},
        after={"order_id": 1, "customer": "Alice", "amount": 150.00, "status": "completed"},
    )

    print(f"\n  Original event (topic: dbserver1.public.orders):")
    print(f"    op: {original.op}")
    print(f"    before: {original.before}")
    print(f"    after: {original.after}")

    # SMT 1: RegexRouter -> rename topic
    transformed_topic = "cdc.orders"
    print(f"\n  After RegexRouter:")
    print(f"    topic: dbserver1.public.orders -> {transformed_topic}")

    # SMT 2: ExtractNewRecordState -> flatten to just the after fields
    flattened = dict(original.after)
    flattened["__op"] = original.op
    flattened["__source_ts_ms"] = original.ts_ms
    print(f"\n  After ExtractNewRecordState:")
    print(f"    {flattened}")
    print(f"    (envelope removed, only 'after' fields + __op + __source_ts_ms)")

    # SMT 3: InsertField -> add processing timestamp
    flattened["cdc_processed_at"] = datetime.now().isoformat()
    print(f"\n  After InsertField (cdc_processed_at):")
    print(f"    {flattened}")

    print(f"\n  Final message on topic '{transformed_topic}':")
    print(f"    {json.dumps(flattened, indent=4, default=str)}")

    return flattened


# ---------------------------------------------------------------------------
# Exercise 4: Schema Evolution and Registry Integration
# ---------------------------------------------------------------------------

def exercise4_schema_evolution():
    """Simulate schema evolution with compatibility modes.

    Compatibility modes:
    - BACKWARD: New schema can read old data. Allows: add optional fields,
      delete fields. Blocks: add required fields.
    - FORWARD: Old schema can read new data. Allows: delete optional fields,
      add fields. Blocks: delete required fields.
    - FULL: Both backward AND forward compatible. Most restrictive.
      Allows: add/remove optional fields. Blocks: any change to required fields.
    """
    # Initial schema
    schema_v1 = {
        "name": "OrderValue",
        "type": "record",
        "fields": [
            {"name": "order_id", "type": "int"},
            {"name": "customer", "type": "string"},
            {"name": "amount", "type": "double"},
            {"name": "status", "type": "string"},
        ],
    }

    # After ALTER TABLE ADD COLUMN priority VARCHAR(10)
    schema_v2 = deepcopy(schema_v1)
    schema_v2["fields"].append(
        {"name": "priority", "type": ["null", "string"], "default": None}
    )

    print(f"\n  Schema V1 (original):")
    for f in schema_v1["fields"]:
        print(f"    {f['name']}: {f['type']}")

    print(f"\n  Schema V2 (after ADD COLUMN priority):")
    for f in schema_v2["fields"]:
        print(f"    {f['name']}: {f['type']}")

    # Old records read with new schema: priority = null
    print(f"\n  Reading old records with new schema:")
    print(f"    Old row: {{order_id: 1, customer: 'Alice', amount: 150, status: 'completed'}}")
    print(f"    As V2:   {{order_id: 1, customer: 'Alice', amount: 150, status: 'completed', priority: null}}")

    # Compatibility analysis
    changes = [
        {
            "change": "Add nullable column 'priority'",
            "backward": True,
            "forward": True,
            "full": True,
            "reason": "New field has default=null, so old readers ignore it, new readers default to null for old data",
        },
        {
            "change": "Drop non-nullable column 'status'",
            "backward": False,
            "forward": True,
            "full": False,
            "reason": "BACKWARD fails: new schema can't read old data that has 'status'. "
                      "FORWARD ok: old schema can read new data (ignores missing 'status' if optional in old schema)",
        },
        {
            "change": "Add required (non-nullable) column 'region'",
            "backward": True,
            "forward": False,
            "full": False,
            "reason": "BACKWARD ok: new schema reads old data, 'region' would need a default. "
                      "FORWARD fails: old schema can't provide required 'region' field",
        },
    ]

    print(f"\n  Compatibility Analysis:")
    print(f"  {'Change':<40} {'BACKWARD':>10} {'FORWARD':>10} {'FULL':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8}")
    for c in changes:
        bw = "PASS" if c["backward"] else "BLOCK"
        fw = "PASS" if c["forward"] else "BLOCK"
        full = "PASS" if c["full"] else "BLOCK"
        print(f"  {c['change']:<40} {bw:>10} {fw:>10} {full:>8}")
        print(f"    Reason: {c['reason']}")

    return changes


# ---------------------------------------------------------------------------
# Exercise 5: Implement the Transactional Outbox Pattern
# ---------------------------------------------------------------------------

class OutboxDatabase:
    """Simulates a PostgreSQL database with transactional outbox.

    The outbox pattern solves the dual-write problem:
    - Without outbox: app writes to DB, then publishes to Kafka.
      If Kafka publish fails after DB commit, the event is lost.
      If app crashes between DB and Kafka, inconsistency.
    - With outbox: app writes to DB AND outbox in ONE transaction.
      Debezium reads the outbox table via CDC and publishes to Kafka.
      Atomicity is guaranteed by the database transaction.
    """
    def __init__(self):
        self.orders: list[dict] = []
        self.outbox: list[dict] = []
        self._in_transaction = False
        self._tx_buffer_orders: list[dict] = []
        self._tx_buffer_outbox: list[dict] = []

    def begin(self):
        self._in_transaction = True
        self._tx_buffer_orders = []
        self._tx_buffer_outbox = []

    def commit(self):
        """Atomically apply both order and outbox writes."""
        self.orders.extend(self._tx_buffer_orders)
        self.outbox.extend(self._tx_buffer_outbox)
        self._in_transaction = False
        self._tx_buffer_orders = []
        self._tx_buffer_outbox = []

    def rollback(self):
        """Discard buffered writes (simulates transaction failure)."""
        self._in_transaction = False
        self._tx_buffer_orders = []
        self._tx_buffer_outbox = []

    def insert_order(self, order: dict) -> None:
        if self._in_transaction:
            self._tx_buffer_orders.append(order)
        else:
            self.orders.append(order)

    def insert_outbox(self, event: dict) -> None:
        if self._in_transaction:
            self._tx_buffer_outbox.append(event)
        else:
            self.outbox.append(event)


def create_order(db: OutboxDatabase, customer_id: str, amount: float,
                 simulate_crash: bool = False) -> dict | None:
    """Create an order with an outbox event in a single transaction."""
    order_id = str(uuid.uuid4())[:8]
    order = {
        "order_id": order_id,
        "customer_id": customer_id,
        "amount": amount,
        "status": "created",
        "created_at": datetime.now().isoformat(),
    }

    outbox_event = {
        "id": str(uuid.uuid4()),
        "aggregate_type": "order.events",
        "aggregate_id": order_id,
        "type": "OrderCreated",
        "payload": json.dumps(order),
        "created_at": datetime.now().isoformat(),
    }

    db.begin()
    try:
        db.insert_order(order)
        db.insert_outbox(outbox_event)

        if simulate_crash:
            raise RuntimeError("Application crashed before commit!")

        db.commit()
        return order
    except Exception as e:
        db.rollback()
        print(f"    [ERROR] Transaction rolled back: {e}")
        return None


def exercise5_outbox_pattern():
    """Demonstrate the transactional outbox pattern."""
    db = OutboxDatabase()

    # Successful orders
    print("\n  Creating orders (transactional outbox):")
    for i in range(3):
        order = create_order(db, f"C{i+1:03d}", round(50 + i * 75.50, 2))
        if order:
            print(f"    Order {order['order_id']}: customer={order['customer_id']}, "
                  f"amount=${order['amount']}")

    # Simulate crash
    print(f"\n  Simulating crash before commit:")
    failed = create_order(db, "C004", 200.00, simulate_crash=True)
    print(f"    Order created: {failed is not None}")

    # Verify state
    print(f"\n  Database state:")
    print(f"    Orders: {len(db.orders)} (crash order NOT committed)")
    print(f"    Outbox: {len(db.outbox)} (crash event NOT committed)")

    # Debezium reads outbox
    print(f"\n  Debezium CDC (EventRouter SMT) reads outbox:")
    for event in db.outbox:
        payload = json.loads(event["payload"])
        print(f"    Topic: {event['aggregate_type']} | type={event['type']} | "
              f"order_id={event['aggregate_id']} | amount=${payload['amount']}")

    # Consumer
    print(f"\n  Kafka consumer on 'order.events':")
    for event in db.outbox:
        payload = json.loads(event["payload"])
        print(f"    Received: type={event['type']}, "
              f"aggregate_id={event['aggregate_id']}, "
              f"payload={payload}")

    print(f"\n  Why outbox solves dual-write:")
    print(f"    - DB write + outbox write are in the SAME transaction.")
    print(f"    - If the app crashes before commit, NEITHER is persisted.")
    print(f"    - Debezium reads committed outbox rows via WAL (CDC).")
    print(f"    - Kafka message is produced by Debezium, not the application.")
    print(f"    - Result: exactly-once semantics without distributed transactions.")

    return db


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Deploy and Verify a Debezium Connector")
    print("=" * 70)
    exercise1_deploy_verify()

    print()
    print("=" * 70)
    print("Exercise 2: Materialized Cache (Multi-Table Consumer)")
    print("=" * 70)
    exercise2_materialized_cache()

    print()
    print("=" * 70)
    print("Exercise 3: Single Message Transforms (SMTs)")
    print("=" * 70)
    exercise3_smts()

    print()
    print("=" * 70)
    print("Exercise 4: Schema Evolution and Compatibility")
    print("=" * 70)
    exercise4_schema_evolution()

    print()
    print("=" * 70)
    print("Exercise 5: Transactional Outbox Pattern")
    print("=" * 70)
    exercise5_outbox_pattern()
