"""
Saga Pattern

Demonstrates:
- Saga orchestrator for distributed transactions
- Compensating transactions (rollback)
- Forward recovery vs backward recovery
- Saga execution log

Theory:
- In microservices, traditional ACID transactions don't span services.
- A saga is a sequence of local transactions, each with a
  compensating action to undo its effect.
- Orchestrator: a central coordinator drives the saga steps.
- Choreography: each service triggers the next (event-driven).
- On failure, previously completed steps are compensated in
  reverse order (backward recovery).

Adapted from System Design Lesson 14.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any


class StepStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATING = "COMPENSATING"
    COMPENSATED = "COMPENSATED"


class SagaStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPENSATING = "COMPENSATING"
    COMPENSATED = "COMPENSATED"
    FAILED = "FAILED"


# Why: Every saga step must define both an action and its compensating action.
# Unlike database rollback, compensations are not automatic — each service must
# provide explicit undo logic (e.g., refund for charge, release for reserve).
@dataclass
class SagaStep:
    name: str
    action: Callable[[], bool]
    compensation: Callable[[], bool]
    status: StepStatus = StepStatus.PENDING


@dataclass
class SagaLog:
    entries: list[tuple[str, str, str]] = field(default_factory=list)

    def log(self, step: str, action: str, result: str) -> None:
        self.entries.append((step, action, result))

    def print_log(self) -> None:
        print(f"\n    {'Step':<25} {'Action':<15} {'Result'}")
        print(f"    {'-'*25} {'-'*15} {'-'*10}")
        for step, action, result in self.entries:
            print(f"    {step:<25} {action:<15} {result}")


# Why: The orchestrator pattern centralizes saga control in one place, making
# the transaction flow easy to reason about and debug. The alternative
# (choreography) distributes logic across services, which is harder to trace
# but avoids a single point of coordination.
class SagaOrchestrator:
    """Orchestrator-based saga execution."""

    def __init__(self, name: str):
        self.name = name
        self.steps: list[SagaStep] = []
        self.status = SagaStatus.PENDING
        self.log = SagaLog()

    def add_step(self, name: str, action: Callable[[], bool],
                 compensation: Callable[[], bool]) -> None:
        self.steps.append(SagaStep(name, action, compensation))

    def execute(self) -> bool:
        """Execute saga: forward steps, compensate on failure."""
        self.status = SagaStatus.RUNNING
        completed_steps: list[int] = []

        for i, step in enumerate(self.steps):
            step.status = StepStatus.RUNNING
            self.log.log(step.name, "execute", "...")

            try:
                success = step.action()
            except Exception:
                success = False

            if success:
                step.status = StepStatus.COMPLETED
                self.log.entries[-1] = (step.name, "execute", "SUCCESS")
                completed_steps.append(i)
            else:
                step.status = StepStatus.FAILED
                self.log.entries[-1] = (step.name, "execute", "FAILED")
                # Compensate in reverse order
                self._compensate(completed_steps)
                return False

        self.status = SagaStatus.COMPLETED
        return True

    def _compensate(self, completed_indices: list[int]) -> None:
        # Why: Compensating in reverse order mirrors how database rollback works —
        # undo the most recent action first to maintain semantic consistency.
        # E.g., refund payment before releasing inventory, not the other way around.
        self.status = SagaStatus.COMPENSATING
        for i in reversed(completed_indices):
            step = self.steps[i]
            step.status = StepStatus.COMPENSATING
            self.log.log(step.name, "compensate", "...")
            try:
                success = step.compensation()
                if success:
                    step.status = StepStatus.COMPENSATED
                    self.log.entries[-1] = (step.name, "compensate", "SUCCESS")
                else:
                    step.status = StepStatus.FAILED
                    self.log.entries[-1] = (step.name, "compensate", "FAILED")
            except Exception:
                step.status = StepStatus.FAILED
                self.log.entries[-1] = (step.name, "compensate", "ERROR")

        # Why: If any compensation fails, the saga enters a FAILED state requiring
        # manual intervention. In production, this triggers an alert for operators
        # to reconcile the inconsistent state across services.
        all_compensated = all(
            self.steps[i].status == StepStatus.COMPENSATED
            for i in completed_indices
        )
        self.status = (SagaStatus.COMPENSATED if all_compensated
                       else SagaStatus.FAILED)


# ── Simulated Services ────────────────────────────────────────────────

class OrderService:
    def __init__(self):
        self.orders: dict[str, dict] = {}

    def create_order(self, order_id: str, items: list, total: float) -> bool:
        self.orders[order_id] = {
            "items": items, "total": total, "status": "created"
        }
        return True

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            return True
        return False


class PaymentService:
    def __init__(self, balance: float = 1000.0):
        self.balance = balance
        self.transactions: list[dict] = []
        self.should_fail = False

    def charge(self, order_id: str, amount: float) -> bool:
        if self.should_fail:
            return False
        if amount > self.balance:
            return False
        self.balance -= amount
        self.transactions.append({
            "order_id": order_id, "amount": amount, "type": "charge"
        })
        return True

    def refund(self, order_id: str, amount: float) -> bool:
        self.balance += amount
        self.transactions.append({
            "order_id": order_id, "amount": amount, "type": "refund"
        })
        return True


# Why: Inventory uses a "reserve then confirm" pattern — stock is decremented
# immediately but tracked per order. This allows the compensation (release) to
# restore exact quantities without race conditions from concurrent orders.
class InventoryService:
    def __init__(self, stock: dict[str, int]):
        self.stock = dict(stock)
        self.reservations: dict[str, list[tuple[str, int]]] = {}
        self.should_fail = False

    def reserve(self, order_id: str, items: list[tuple[str, int]]) -> bool:
        if self.should_fail:
            return False
        for item, qty in items:
            if self.stock.get(item, 0) < qty:
                return False
        for item, qty in items:
            self.stock[item] -= qty
        self.reservations[order_id] = items
        return True

    def release(self, order_id: str) -> bool:
        if order_id in self.reservations:
            for item, qty in self.reservations[order_id]:
                self.stock[item] = self.stock.get(item, 0) + qty
            del self.reservations[order_id]
            return True
        return False


class ShippingService:
    def __init__(self):
        self.shipments: dict[str, str] = {}
        self.should_fail = False

    def create_shipment(self, order_id: str) -> bool:
        if self.should_fail:
            return False
        self.shipments[order_id] = "scheduled"
        return True

    def cancel_shipment(self, order_id: str) -> bool:
        if order_id in self.shipments:
            self.shipments[order_id] = "cancelled"
            return True
        return False


# ── Demos ──────────────────────────────────────────────────────────────

def demo_successful_saga():
    print("=" * 60)
    print("SAGA: SUCCESSFUL ORDER")
    print("=" * 60)

    order_svc = OrderService()
    payment_svc = PaymentService(balance=500.0)
    inventory_svc = InventoryService({"widget": 10, "gadget": 5})
    shipping_svc = ShippingService()

    order_id = "ORD-001"
    items = [("widget", 2), ("gadget", 1)]
    total = 99.99

    saga = SagaOrchestrator("CreateOrder")
    saga.add_step(
        "1. Create Order",
        lambda: order_svc.create_order(order_id, items, total),
        lambda: order_svc.cancel_order(order_id),
    )
    saga.add_step(
        "2. Reserve Inventory",
        lambda: inventory_svc.reserve(order_id, items),
        lambda: inventory_svc.release(order_id),
    )
    saga.add_step(
        "3. Process Payment",
        lambda: payment_svc.charge(order_id, total),
        lambda: payment_svc.refund(order_id, total),
    )
    saga.add_step(
        "4. Create Shipment",
        lambda: shipping_svc.create_shipment(order_id),
        lambda: shipping_svc.cancel_shipment(order_id),
    )

    print(f"\n  Order: {order_id}, items={items}, total=${total}")
    success = saga.execute()

    print(f"\n  Saga result: {'SUCCESS' if success else 'COMPENSATED'}")
    print(f"  Saga status: {saga.status.value}")
    saga.log.print_log()

    print(f"\n  Final state:")
    print(f"    Order: {order_svc.orders.get(order_id, {}).get('status')}")
    print(f"    Balance: ${payment_svc.balance:.2f}")
    print(f"    Stock: {inventory_svc.stock}")
    print(f"    Shipment: {shipping_svc.shipments.get(order_id)}")


def demo_failed_saga():
    print("\n" + "=" * 60)
    print("SAGA: PAYMENT FAILURE → COMPENSATION")
    print("=" * 60)

    order_svc = OrderService()
    payment_svc = PaymentService(balance=500.0)
    payment_svc.should_fail = True  # Simulate payment failure
    inventory_svc = InventoryService({"widget": 10, "gadget": 5})
    shipping_svc = ShippingService()

    order_id = "ORD-002"
    items = [("widget", 3)]
    total = 150.00

    saga = SagaOrchestrator("CreateOrder")
    saga.add_step(
        "1. Create Order",
        lambda: order_svc.create_order(order_id, items, total),
        lambda: order_svc.cancel_order(order_id),
    )
    saga.add_step(
        "2. Reserve Inventory",
        lambda: inventory_svc.reserve(order_id, items),
        lambda: inventory_svc.release(order_id),
    )
    saga.add_step(
        "3. Process Payment",
        lambda: payment_svc.charge(order_id, total),
        lambda: payment_svc.refund(order_id, total),
    )
    saga.add_step(
        "4. Create Shipment",
        lambda: shipping_svc.create_shipment(order_id),
        lambda: shipping_svc.cancel_shipment(order_id),
    )

    print(f"\n  Order: {order_id}, payment service WILL FAIL")
    print(f"  Initial stock: {inventory_svc.stock}")
    print(f"  Initial balance: ${payment_svc.balance:.2f}")

    success = saga.execute()

    print(f"\n  Saga result: {'SUCCESS' if success else 'COMPENSATED'}")
    print(f"  Saga status: {saga.status.value}")
    saga.log.print_log()

    print(f"\n  After compensation:")
    print(f"    Order: {order_svc.orders.get(order_id, {}).get('status')}")
    print(f"    Balance: ${payment_svc.balance:.2f} (unchanged)")
    print(f"    Stock: {inventory_svc.stock} (restored)")
    print(f"    Shipment: {shipping_svc.shipments.get(order_id, 'none')}")


def demo_late_failure():
    print("\n" + "=" * 60)
    print("SAGA: SHIPPING FAILURE (LATE STAGE)")
    print("=" * 60)

    order_svc = OrderService()
    payment_svc = PaymentService(balance=500.0)
    inventory_svc = InventoryService({"widget": 10})
    shipping_svc = ShippingService()
    shipping_svc.should_fail = True  # Last step fails

    order_id = "ORD-003"
    items = [("widget", 2)]
    total = 49.99

    saga = SagaOrchestrator("CreateOrder")
    saga.add_step(
        "1. Create Order",
        lambda: order_svc.create_order(order_id, items, total),
        lambda: order_svc.cancel_order(order_id),
    )
    saga.add_step(
        "2. Reserve Inventory",
        lambda: inventory_svc.reserve(order_id, items),
        lambda: inventory_svc.release(order_id),
    )
    saga.add_step(
        "3. Process Payment",
        lambda: payment_svc.charge(order_id, total),
        lambda: payment_svc.refund(order_id, total),
    )
    saga.add_step(
        "4. Create Shipment",
        lambda: shipping_svc.create_shipment(order_id),
        lambda: shipping_svc.cancel_shipment(order_id),
    )

    print(f"\n  Order: {order_id}, shipping WILL FAIL")
    print(f"  Balance before: ${payment_svc.balance:.2f}")
    print(f"  Stock before: {inventory_svc.stock}")

    success = saga.execute()

    print(f"\n  Saga result: {'SUCCESS' if success else 'COMPENSATED'}")
    saga.log.print_log()

    print(f"\n  All 3 completed steps were compensated:")
    print(f"    Balance after: ${payment_svc.balance:.2f} (refunded)")
    print(f"    Stock after: {inventory_svc.stock} (released)")
    print(f"    Order: {order_svc.orders[order_id]['status']}")


if __name__ == "__main__":
    demo_successful_saga()
    demo_failed_saga()
    demo_late_failure()
