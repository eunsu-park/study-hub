"""
Exercises for Lesson 24: Event Sourcing and CQRS
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
"""

import time
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


# === Exercise 1: Shopping Cart Aggregate ===
@dataclass
class Event:
    event_type: str = ""
    aggregate_id: str = ""
    version: int = 0
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ShoppingCart:
    """Event-sourced shopping cart aggregate."""
    def __init__(self, cart_id):
        self.cart_id = cart_id
        self.items: Dict[str, dict] = {}
        self.checked_out = False
        self.version = 0
        self.uncommitted = []

    def _emit(self, event_type, data):
        event = Event(event_type=event_type, aggregate_id=self.cart_id, data=data)
        self._apply(event)
        self.uncommitted.append(event)

    def _apply(self, event):
        t = event.event_type
        d = event.data
        if t == "CartCreated":
            self.items = {}
            self.checked_out = False
        elif t == "ItemAdded":
            pid = d["product_id"]
            if pid in self.items:
                self.items[pid]["quantity"] += d["quantity"]
            else:
                self.items[pid] = {"name": d["name"], "price": d["price"],
                                   "quantity": d["quantity"]}
        elif t == "ItemRemoved":
            self.items.pop(d["product_id"], None)
        elif t == "QuantityChanged":
            if d["product_id"] in self.items:
                self.items[d["product_id"]]["quantity"] = d["quantity"]
        elif t == "CartCheckedOut":
            self.checked_out = True
        self.version = event.version

    def create(self):
        if self.items or self.checked_out:
            raise ValueError("Cart already exists")
        self._emit("CartCreated", {})

    def add_item(self, product_id, name, price, quantity=1):
        if self.checked_out:
            raise ValueError("Cart already checked out")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        self._emit("ItemAdded", {"product_id": product_id, "name": name,
                                  "price": price, "quantity": quantity})

    def remove_item(self, product_id):
        if product_id not in self.items:
            raise ValueError("Item not in cart")
        self._emit("ItemRemoved", {"product_id": product_id})

    def change_quantity(self, product_id, quantity):
        if product_id not in self.items:
            raise ValueError("Item not in cart")
        if quantity <= 0:
            raise ValueError("Use remove_item for quantity 0")
        self._emit("QuantityChanged", {"product_id": product_id, "quantity": quantity})

    def checkout(self):
        if self.checked_out:
            raise ValueError("Already checked out")
        if not self.items:
            raise ValueError("Cart is empty")
        self._emit("CartCheckedOut", {"total": self.total()})

    def total(self):
        return sum(i["price"] * i["quantity"] for i in self.items.values())

    def load(self, events):
        for e in events:
            self._apply(e)
            self.version = e.version


def exercise_1():
    print("=== Exercise 1: Shopping Cart Aggregate ===\n")
    cart = ShoppingCart("CART-001")
    cart.create()
    cart.add_item("PROD-1", "Widget", 9.99, 2)
    cart.add_item("PROD-2", "Gadget", 24.99)
    cart.change_quantity("PROD-1", 3)
    cart.checkout()

    print(f"  Items: {cart.items}")
    print(f"  Total: ${cart.total():.2f}")
    print(f"  Events: {len(cart.uncommitted)}")
    for e in cart.uncommitted:
        print(f"    {e.event_type}: {e.data}")


exercise_1()


# === Exercise 2: Multiple Projections ===
def exercise_2():
    print("\n=== Exercise 2: Order Event Projections ===\n")

    events = [
        Event(event_type="OrderPlaced", aggregate_id="O1",
              data={"customer": "alice", "product": "Widget", "amount": 29.99}),
        Event(event_type="OrderPlaced", aggregate_id="O2",
              data={"customer": "bob", "product": "Gadget", "amount": 49.99}),
        Event(event_type="OrderPlaced", aggregate_id="O3",
              data={"customer": "alice", "product": "Widget", "amount": 29.99}),
    ]

    # Projection A: Per-customer order count
    customer_counts = defaultdict(int)
    # Projection B: Daily revenue
    daily_revenue = defaultdict(float)
    # Projection C: Product popularity
    product_counts = defaultdict(int)

    for e in events:
        if e.event_type == "OrderPlaced":
            customer_counts[e.data["customer"]] += 1
            daily_revenue["2026-03-16"] += e.data["amount"]
            product_counts[e.data["product"]] += 1

    print(f"  Customer orders: {dict(customer_counts)}")
    print(f"  Daily revenue: {dict(daily_revenue)}")
    print(f"  Product popularity: {dict(product_counts)}")


exercise_2()


# === Exercise 3: Optimistic Concurrency ===
def exercise_3():
    print("\n=== Exercise 3: Concurrent Withdrawal ===\n")

    class EventStore:
        def __init__(self):
            self.streams = defaultdict(list)
        def append(self, aid, events, expected_version):
            current = len(self.streams[aid])
            if current != expected_version:
                raise Exception(f"Concurrency conflict: expected {expected_version}, got {current}")
            for e in events:
                e.version = current + 1
                self.streams[aid].append(e)
                current += 1

    store = EventStore()
    # Initial balance: $100
    store.append("ACC-1", [
        Event(event_type="Deposited", data={"amount": 100})
    ], 0)

    # User A reads version 1, balance=$100
    # User B reads version 1, balance=$100

    # User A withdraws $80 — succeeds
    store.append("ACC-1", [
        Event(event_type="Withdrawn", data={"amount": 80})
    ], 1)
    print("  User A withdraws $80: SUCCESS (version 1 → 2)")

    # User B tries to withdraw $80 — conflict!
    try:
        store.append("ACC-1", [
            Event(event_type="Withdrawn", data={"amount": 80})
        ], 1)  # Still expects version 1
        print("  User B withdraws $80: SUCCESS (BUG!)")
    except Exception as e:
        print(f"  User B withdraws $80: REJECTED ({e})")
        print("  Overdraft prevented by optimistic concurrency control!")


exercise_3()


# === Exercise 4: Schema Evolution ===
def exercise_4():
    print("\n=== Exercise 4: Schema Evolution ===\n")

    upcasters = {}

    def upcast_v1_to_v2(event):
        """Add currency field."""
        data = dict(event.data)
        data["currency"] = "USD"
        return Event(event_type=event.event_type, data=data)

    def upcast_v2_to_v3(event):
        """Split price into unit_price and total_price."""
        data = dict(event.data)
        price = data.pop("price", 0)
        data["unit_price"] = price
        data["total_price"] = price * data.get("quantity", 1)
        return Event(event_type=event.event_type, data=data)

    def upcast_v3_to_v4(event):
        """Rename product_id to sku."""
        data = dict(event.data)
        data["sku"] = data.pop("product_id", "")
        return Event(event_type=event.event_type, data=data)

    # Chain upcasters
    old_event = Event(
        event_type="OrderPlaced",
        data={"product_id": "PROD-1", "quantity": 3, "price": 9.99}
    )

    print(f"  V1: {old_event.data}")
    v2 = upcast_v1_to_v2(old_event)
    print(f"  V2: {v2.data}")
    v3 = upcast_v2_to_v3(v2)
    print(f"  V3: {v3.data}")
    v4 = upcast_v3_to_v4(v3)
    print(f"  V4: {v4.data}")


exercise_4()


# === Exercise 5: CQRS System ===
def exercise_5():
    print("\n=== Exercise 5: CQRS with Snapshots ===\n")

    class SimpleEventStore:
        def __init__(self):
            self.events = []
        def append(self, event):
            self.events.append(event)
        def all(self):
            return list(self.events)

    store = SimpleEventStore()
    balance_projection = defaultdict(float)
    snapshots = {}

    # Write side: append events
    for i in range(150):
        event = Event(event_type="Deposited",
                     aggregate_id=f"ACC-{i % 5}",
                     data={"amount": 10.0})
        store.append(event)
        # Update projection
        balance_projection[event.aggregate_id] += event.data["amount"]

        # Snapshot every 50 events
        if (i + 1) % 50 == 0:
            snapshots[i + 1] = dict(balance_projection)

    print(f"  Total events: {len(store.all())}")
    print(f"  Snapshots: {list(snapshots.keys())}")
    print(f"  Balances: {dict(balance_projection)}")

    # Rebuild projection from snapshot + remaining events
    snap = snapshots[100]
    remaining = store.all()[100:]
    rebuilt = dict(snap)
    for e in remaining:
        rebuilt[e.aggregate_id] = rebuilt.get(e.aggregate_id, 0) + e.data["amount"]
    print(f"  Rebuilt: {rebuilt}")
    print(f"  Match: {rebuilt == balance_projection}")


exercise_5()


if __name__ == "__main__":
    print("\nAll exercises completed.")
