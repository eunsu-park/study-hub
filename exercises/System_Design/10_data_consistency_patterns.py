"""
Exercises for Lesson 10: Data Consistency Patterns
Topic: System_Design

Solutions to practice problems from the lesson.
Covers consistency model selection, Saga pattern design,
and Read-Your-Writes client library.
"""

import time
import random
import enum
from collections import defaultdict
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field


# === Exercise 1: Analyze Consistency Requirements ===
# Problem: Choose appropriate consistency model for scenarios.

def exercise_1():
    """Consistency model selection for different services."""
    scenarios = [
        {
            "service": "Bank account balance",
            "model": "Strong Consistency",
            "reason": "Financial accuracy is critical. Users expect to see correct "
                      "balance immediately after transactions. Linearizability required.",
        },
        {
            "service": "Twitter follower count",
            "model": "Eventual Consistency",
            "reason": "Slight differences across replicas are acceptable. "
                      "Users tolerate seeing 10,521 vs 10,523 followers briefly. "
                      "Availability is more important than precision.",
        },
        {
            "service": "Online game ranking",
            "model": "Eventual Consistency (with bounded staleness)",
            "reason": "Rankings can lag a few seconds. Real-time accuracy is nice "
                      "but not critical. Bounded staleness ensures ranks "
                      "converge within a known time window.",
        },
        {
            "service": "Airline seat reservation",
            "model": "Strong Consistency",
            "reason": "Must prevent double booking. Same seat cannot be sold twice. "
                      "Serializable transactions needed for seat allocation.",
        },
        {
            "service": "News article view count",
            "model": "Eventual Consistency",
            "reason": "Approximate view counts are fine. No one notices if count "
                      "is 1,000 vs 1,003. High write throughput matters more.",
        },
    ]

    print("Consistency Model Selection:")
    print("=" * 60)
    for i, s in enumerate(scenarios, 1):
        print(f"\n{i}. {s['service']}")
        print(f"   Model:  {s['model']}")
        print(f"   Reason: {s['reason']}")


# === Exercise 2: Saga Pattern Design ===
# Problem: Design online shopping order process using Saga pattern.

class SagaStepStatus(enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    name: str
    action: Callable
    compensation: Callable
    status: SagaStepStatus = SagaStepStatus.PENDING


class SagaOrchestrator:
    """Orchestration-based Saga pattern implementation."""

    def __init__(self, saga_id: str):
        self.saga_id = saga_id
        self.steps: List[SagaStep] = []
        self.completed_steps: List[SagaStep] = []

    def add_step(self, name, action, compensation):
        self.steps.append(SagaStep(name, action, compensation))

    def execute(self):
        """Execute saga steps in order. Compensate on failure."""
        print(f"  Saga {self.saga_id} starting...")

        for step in self.steps:
            print(f"    Executing: {step.name}")
            try:
                result = step.action()
                if result:
                    step.status = SagaStepStatus.COMPLETED
                    self.completed_steps.append(step)
                    print(f"    -> {step.name} completed")
                else:
                    raise Exception(f"{step.name} returned failure")
            except Exception as e:
                print(f"    -> {step.name} FAILED: {e}")
                step.status = SagaStepStatus.FAILED
                self._compensate()
                return False

        print(f"  Saga {self.saga_id} completed successfully!")
        return True

    def _compensate(self):
        """Execute compensating transactions in reverse order."""
        print(f"    Starting compensation...")
        for step in reversed(self.completed_steps):
            print(f"    Compensating: {step.name}")
            try:
                step.compensation()
                step.status = SagaStepStatus.COMPENSATED
                print(f"    -> {step.name} compensated")
            except Exception as e:
                print(f"    -> COMPENSATION FAILED for {step.name}: {e}")


# Simulated services
class OrderService:
    def __init__(self):
        self.orders = {}

    def create_order(self, order_id, items):
        self.orders[order_id] = {"items": items, "status": "created"}
        return True

    def cancel_order(self, order_id):
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"


class InventoryService:
    def __init__(self):
        self.stock = {"laptop": 10, "mouse": 50, "keyboard": 30}

    def reserve(self, items):
        for item, qty in items.items():
            if self.stock.get(item, 0) < qty:
                return False
            self.stock[item] -= qty
        return True

    def release(self, items):
        for item, qty in items.items():
            self.stock[item] = self.stock.get(item, 0) + qty


class PaymentService:
    def __init__(self, should_fail=False):
        self.payments = {}
        self.should_fail = should_fail

    def charge(self, order_id, amount):
        if self.should_fail:
            return False
        self.payments[order_id] = {"amount": amount, "status": "charged"}
        return True

    def refund(self, order_id):
        if order_id in self.payments:
            self.payments[order_id]["status"] = "refunded"


class ShippingService:
    def __init__(self):
        self.reservations = {}

    def reserve(self, order_id, address):
        self.reservations[order_id] = {"address": address, "status": "reserved"}
        return True

    def cancel(self, order_id):
        if order_id in self.reservations:
            self.reservations[order_id]["status"] = "cancelled"


def exercise_2():
    """Saga pattern for online shopping order process."""
    print("Saga Pattern: Order Process:")
    print("=" * 60)

    # Define services
    order_svc = OrderService()
    inventory_svc = InventoryService()
    payment_svc = PaymentService()
    shipping_svc = ShippingService()

    order_id = "ORD-001"
    items = {"laptop": 1, "mouse": 2}
    total = 1500

    # --- Orchestration Saga (Success case) ---
    print("\n--- Orchestration Saga (Success) ---")
    saga = SagaOrchestrator(order_id)
    saga.add_step(
        "Create Order",
        lambda: order_svc.create_order(order_id, items),
        lambda: order_svc.cancel_order(order_id)
    )
    saga.add_step(
        "Reserve Inventory",
        lambda: inventory_svc.reserve(items),
        lambda: inventory_svc.release(items)
    )
    saga.add_step(
        "Process Payment",
        lambda: payment_svc.charge(order_id, total),
        lambda: payment_svc.refund(order_id)
    )
    saga.add_step(
        "Reserve Shipping",
        lambda: shipping_svc.reserve(order_id, "123 Main St"),
        lambda: shipping_svc.cancel(order_id)
    )
    saga.execute()

    # --- Orchestration Saga (Failure case) ---
    print("\n--- Orchestration Saga (Payment Failure) ---")
    payment_svc_fail = PaymentService(should_fail=True)
    order_svc2 = OrderService()
    inventory_svc2 = InventoryService()

    order_id2 = "ORD-002"
    saga2 = SagaOrchestrator(order_id2)
    saga2.add_step(
        "Create Order",
        lambda: order_svc2.create_order(order_id2, items),
        lambda: order_svc2.cancel_order(order_id2)
    )
    saga2.add_step(
        "Reserve Inventory",
        lambda: inventory_svc2.reserve(items),
        lambda: inventory_svc2.release(items)
    )
    saga2.add_step(
        "Process Payment",
        lambda: payment_svc_fail.charge(order_id2, total),
        lambda: payment_svc_fail.refund(order_id2)
    )
    saga2.execute()

    print(f"\n  After failed saga:")
    print(f"    Order status: {order_svc2.orders.get(order_id2, {}).get('status', 'N/A')}")
    print(f"    Inventory restored: laptop={inventory_svc2.stock['laptop']}, "
          f"mouse={inventory_svc2.stock['mouse']}")

    # --- Choreography version description ---
    print("\n--- Choreography Version ---")
    print("  In choreography, each service publishes events:")
    print("  1. OrderService publishes 'OrderCreated'")
    print("  2. InventoryService listens, reserves, publishes 'InventoryReserved'")
    print("  3. PaymentService listens, charges, publishes 'PaymentProcessed'")
    print("  4. ShippingService listens, reserves, publishes 'ShippingReserved'")
    print("  On failure, reverse events trigger compensations automatically.")


# === Exercise 3: Read-Your-Writes Client Library ===
# Problem: Design a client library guaranteeing Read-Your-Writes.

class ReadYourWritesClient:
    """Client library that guarantees Read-Your-Writes consistency.

    Uses timestamp-based approach with local caching.
    """

    def __init__(self, leader, replicas):
        self.leader = leader
        self.replicas = replicas
        self.last_write_ts = 0
        self.local_cache = {}  # Recent writes cached locally
        self.cache_ttl = 5.0   # Cache entries for 5 seconds

    def write(self, key, value):
        """Write to leader and cache locally."""
        ts = time.time()
        self.leader[key] = (value, ts)
        self.last_write_ts = ts

        # Cache locally for immediate reads
        self.local_cache[key] = (value, ts)

        # Async replication (simulated)
        for replica in self.replicas:
            # Replica may lag
            lag = random.uniform(0, 2.0)
            replica["pending"].append((key, value, ts, lag))

        return ts

    def read(self, key):
        """Read with RYW guarantee."""
        now = time.time()

        # Strategy 1: Check local cache first
        if key in self.local_cache:
            value, ts = self.local_cache[key]
            if now - ts < self.cache_ttl:
                return value, "local_cache"

        # Strategy 2: If recent write, read from leader
        if now - self.last_write_ts < 1.0:
            entry = self.leader.get(key)
            if entry:
                return entry[0], "leader"

        # Strategy 3: Find a replica that's caught up
        for replica in self.replicas:
            # Process pending replications
            still_pending = []
            for k, v, ts, lag in replica["pending"]:
                if now - ts >= lag:  # Replication completed
                    replica["data"][k] = (v, ts)
                    replica["applied_ts"] = max(replica.get("applied_ts", 0), ts)
                else:
                    still_pending.append((k, v, ts, lag))
            replica["pending"] = still_pending

            # Check if replica is caught up to our last write
            if replica.get("applied_ts", 0) >= self.last_write_ts:
                entry = replica["data"].get(key)
                if entry:
                    return entry[0], "replica"

        # Fallback: read from leader
        entry = self.leader.get(key)
        if entry:
            return entry[0], "leader_fallback"

        return None, "not_found"


def exercise_3():
    """Read-Your-Writes client library demonstration."""
    print("Read-Your-Writes Client Library:")
    print("=" * 60)

    random.seed(42)
    leader = {}
    replicas = [
        {"data": {}, "pending": [], "applied_ts": 0},
        {"data": {}, "pending": [], "applied_ts": 0},
        {"data": {}, "pending": [], "applied_ts": 0},
    ]

    client = ReadYourWritesClient(leader, replicas)

    # Write and immediately read
    print("\n--- Write then immediate read ---")
    ts = client.write("balance", "1000")
    value, source = client.read("balance")
    print(f"  Wrote balance=1000 (ts={ts:.3f})")
    print(f"  Read balance={value} from {source}")

    # Update and read again
    ts = client.write("balance", "900")
    value, source = client.read("balance")
    print(f"  Wrote balance=900 (ts={ts:.3f})")
    print(f"  Read balance={value} from {source}")

    # Wait for replication and read from replica
    print("\n--- After replication lag ---")
    time.sleep(0.1)  # Small wait
    value, source = client.read("balance")
    print(f"  Read balance={value} from {source}")

    print("\nImplementation Summary:")
    print("  1. Local cache: Serves reads immediately after writes")
    print("  2. Leader reads: Within 1s of write, go to leader")
    print("  3. Replica reads: After replication catches up")
    print("  4. Fallback: Always fall back to leader if replica is stale")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Consistency Model Selection ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Saga Pattern Design ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Read-Your-Writes Client Library ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
