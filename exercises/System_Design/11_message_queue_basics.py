"""
Exercises for Lesson 11: Message Queue Basics
Topic: System_Design

Solutions to practice problems from the lesson.
Covers sync/async communication selection, delivery guarantees,
and idempotency design.
"""

import time
import uuid
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional


# === Exercise 1: Choose Communication Method ===
# Problem: Select appropriate sync/async method for scenarios.

def exercise_1():
    """Synchronous vs asynchronous communication selection."""
    scenarios = [
        ("User login authentication", "Synchronous",
         "User needs immediate response. Cannot proceed without auth result. "
         "Blocking call is appropriate."),
        ("Send confirmation email after order", "Asynchronous",
         "User doesn't need to wait for email delivery. "
         "Queue the email task and respond to user immediately."),
        ("Payment approval request", "Synchronous",
         "Must know if payment succeeded before confirming order. "
         "Blocking call with timeout is appropriate."),
        ("Log collection", "Asynchronous",
         "Logs can be processed later. High throughput needed. "
         "Message queue buffers during spikes."),
        ("Real-time chat message", "Asynchronous (with push)",
         "Messages queued and delivered via WebSocket push. "
         "Sender doesn't wait for recipient to read."),
    ]

    print("Communication Method Selection:")
    print("=" * 60)
    for i, (scenario, choice, reason) in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario}")
        print(f"   Method: {choice}")
        print(f"   Reason: {reason}")


# === Exercise 2: Choose Delivery Guarantee ===
# Problem: Select delivery guarantee level for use cases.

def exercise_2():
    """Delivery guarantee selection."""
    scenarios = [
        ("IoT sensor temperature data", "At-most-once",
         "Missing one reading is fine. Duplicate readings could skew averages. "
         "High throughput matters more than completeness."),
        ("Bank transfer request", "Exactly-once (or at-least-once + idempotency)",
         "Cannot lose a transfer. Cannot process twice. "
         "Use idempotency key to ensure exactly-once semantics."),
        ("News feed update", "At-most-once or at-least-once",
         "Missing an update is minor. Duplicate updates are visible but tolerable. "
         "Depends on user experience requirements."),
        ("Order creation event", "At-least-once + idempotency",
         "Cannot lose orders. Duplicate processing prevented by order ID check. "
         "Exactly-once semantics through idempotent consumer."),
        ("Game player position update", "At-most-once",
         "Latest position supersedes previous. Missing one update is fine. "
         "Low latency is more important than reliability."),
    ]

    print("Delivery Guarantee Selection:")
    print("=" * 60)
    for i, (scenario, choice, reason) in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario}")
        print(f"   Guarantee: {choice}")
        print(f"   Reason:    {reason}")


# === Exercise 3: Design Idempotency ===
# Problem: Make operations idempotent.

class IdempotentOperationStore:
    """Stores idempotency keys to prevent duplicate processing."""
    def __init__(self):
        self.processed = {}  # idempotency_key -> result

    def execute_once(self, idempotency_key, operation):
        """Execute operation only if not already processed."""
        if idempotency_key in self.processed:
            return self.processed[idempotency_key], True  # (result, was_duplicate)

        result = operation()
        self.processed[idempotency_key] = result
        return result, False


class BankAccount:
    """Bank account with idempotent withdrawal."""
    def __init__(self, account_id, balance):
        self.account_id = account_id
        self.balance = balance
        self.store = IdempotentOperationStore()

    def withdraw(self, request_id, amount):
        """Idempotent withdrawal using request_id as idempotency key."""
        def do_withdraw():
            if self.balance >= amount:
                self.balance -= amount
                return {"status": "success", "new_balance": self.balance}
            return {"status": "insufficient_funds", "balance": self.balance}

        result, was_duplicate = self.store.execute_once(request_id, do_withdraw)
        return result, was_duplicate


class InventoryManager:
    """Inventory with idempotent decrement."""
    def __init__(self):
        self.stock = {"laptop": 100, "mouse": 500}
        self.store = IdempotentOperationStore()

    def decrement(self, order_id, product, quantity):
        """Idempotent inventory decrement using order_id."""
        key = f"{order_id}:{product}"

        def do_decrement():
            if self.stock.get(product, 0) >= quantity:
                self.stock[product] -= quantity
                return {"status": "success", "remaining": self.stock[product]}
            return {"status": "out_of_stock"}

        result, was_duplicate = self.store.execute_once(key, do_decrement)
        return result, was_duplicate


class EmailService:
    """Email service with idempotent sending."""
    def __init__(self):
        self.sent_emails = []
        self.store = IdempotentOperationStore()

    def send(self, message_id, to, subject, body):
        """Idempotent email send using message_id."""
        def do_send():
            self.sent_emails.append({
                "id": message_id, "to": to,
                "subject": subject, "body": body
            })
            return {"status": "sent", "message_id": message_id}

        result, was_duplicate = self.store.execute_once(message_id, do_send)
        return result, was_duplicate


class PointsService:
    """Points service with idempotent point addition."""
    def __init__(self):
        self.points = defaultdict(int)
        self.store = IdempotentOperationStore()

    def add_points(self, transaction_id, user_id, points):
        """Idempotent point addition using transaction_id."""
        def do_add():
            self.points[user_id] += points
            return {"status": "success", "total_points": self.points[user_id]}

        result, was_duplicate = self.store.execute_once(transaction_id, do_add)
        return result, was_duplicate


def exercise_3():
    """Idempotency design for various operations."""
    print("Idempotency Design:")
    print("=" * 60)

    # 1. Bank withdrawal
    print("\n--- 1. Idempotent Bank Withdrawal ---")
    account = BankAccount("ACC-001", 1000)
    request_id = "TXN-001"

    result1, dup1 = account.withdraw(request_id, 100)
    print(f"  First attempt:  {result1}, duplicate={dup1}")
    result2, dup2 = account.withdraw(request_id, 100)  # Retry
    print(f"  Retry attempt:  {result2}, duplicate={dup2}")
    print(f"  Balance: ${account.balance} (correctly deducted only once)")

    # 2. Inventory decrement
    print("\n--- 2. Idempotent Inventory Decrement ---")
    inventory = InventoryManager()
    order_id = "ORD-001"

    result1, dup1 = inventory.decrement(order_id, "laptop", 1)
    print(f"  First attempt:  {result1}, duplicate={dup1}")
    result2, dup2 = inventory.decrement(order_id, "laptop", 1)  # Retry
    print(f"  Retry attempt:  {result2}, duplicate={dup2}")
    print(f"  Laptop stock: {inventory.stock['laptop']} (decremented only once)")

    # 3. Email sending
    print("\n--- 3. Idempotent Email Sending ---")
    email_svc = EmailService()
    msg_id = "MSG-001"

    result1, dup1 = email_svc.send(msg_id, "user@example.com", "Order Confirmed", "...")
    print(f"  First attempt:  {result1}, duplicate={dup1}")
    result2, dup2 = email_svc.send(msg_id, "user@example.com", "Order Confirmed", "...")
    print(f"  Retry attempt:  {result2}, duplicate={dup2}")
    print(f"  Emails sent: {len(email_svc.sent_emails)} (sent only once)")

    # 4. Points addition
    print("\n--- 4. Idempotent Points Addition ---")
    points_svc = PointsService()
    txn_id = "PTS-001"

    result1, dup1 = points_svc.add_points(txn_id, "user_1", 500)
    print(f"  First attempt:  {result1}, duplicate={dup1}")
    result2, dup2 = points_svc.add_points(txn_id, "user_1", 500)  # Retry
    print(f"  Retry attempt:  {result2}, duplicate={dup2}")
    print(f"  User points: {points_svc.points['user_1']} (added only once)")

    print("\n--- Idempotency Key Design Summary ---")
    print("  Withdrawal: request_id (unique per transaction attempt)")
    print("  Inventory:  order_id + product (unique per order-product pair)")
    print("  Email:      message_id (unique per message)")
    print("  Points:     transaction_id (unique per reward event)")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Communication Method Selection ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Delivery Guarantee Selection ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Idempotency Design ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
