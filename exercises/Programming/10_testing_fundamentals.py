"""
Exercises for Lesson 10: Testing Fundamentals
Topic: Programming

Solutions to practice problems from the lesson.
"""
import unittest
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass, field
from typing import List
import random


# === Exercise 1: Write Unit Tests ===
# Problem: Write comprehensive unit tests for BankAccount class.

class BankAccount:
    """The class under test (from the exercise)."""

    def __init__(self, initial_balance=0):
        self.balance = initial_balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount

    def transfer(self, amount, target_account):
        self.withdraw(amount)
        target_account.deposit(amount)


class TestBankAccount(unittest.TestCase):
    """Comprehensive tests covering normal, edge, and error cases."""

    def setUp(self):
        """Create a fresh account before each test."""
        self.account = BankAccount(100)

    # --- Normal operations ---

    def test_initial_balance(self):
        """Account starts with the specified balance."""
        self.assertEqual(self.account.balance, 100)

    def test_default_balance_is_zero(self):
        """Default balance is 0 when not specified."""
        account = BankAccount()
        self.assertEqual(account.balance, 0)

    def test_deposit_increases_balance(self):
        """Deposit adds to balance correctly."""
        self.account.deposit(50)
        self.assertEqual(self.account.balance, 150)

    def test_withdraw_decreases_balance(self):
        """Withdrawal subtracts from balance correctly."""
        self.account.withdraw(30)
        self.assertEqual(self.account.balance, 70)

    def test_multiple_operations(self):
        """Sequence of deposits and withdrawals maintains correct balance."""
        self.account.deposit(50)
        self.account.withdraw(30)
        self.account.deposit(20)
        self.assertEqual(self.account.balance, 140)

    # --- Edge cases ---

    def test_withdraw_exact_balance(self):
        """Can withdraw the exact balance (zero remaining)."""
        self.account.withdraw(100)
        self.assertEqual(self.account.balance, 0)

    def test_deposit_small_amount(self):
        """Can deposit very small positive amounts."""
        self.account.deposit(0.01)
        self.assertAlmostEqual(self.account.balance, 100.01)

    # --- Error cases ---

    def test_deposit_zero_raises(self):
        """Depositing zero raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.deposit(0)

    def test_deposit_negative_raises(self):
        """Depositing a negative amount raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.deposit(-10)

    def test_withdraw_zero_raises(self):
        """Withdrawing zero raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.withdraw(0)

    def test_withdraw_negative_raises(self):
        """Withdrawing a negative amount raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.withdraw(-10)

    def test_withdraw_more_than_balance_raises(self):
        """Withdrawing more than balance raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.withdraw(200)

    def test_insufficient_funds_message(self):
        """Error message mentions insufficient funds."""
        with self.assertRaisesRegex(ValueError, "Insufficient funds"):
            self.account.withdraw(200)

    # --- Transfer ---

    def test_transfer_between_accounts(self):
        """Transfer moves money from one account to another."""
        target = BankAccount(50)
        self.account.transfer(30, target)
        self.assertEqual(self.account.balance, 70)
        self.assertEqual(target.balance, 80)

    def test_transfer_insufficient_funds(self):
        """Transfer fails if source has insufficient funds."""
        target = BankAccount(50)
        with self.assertRaises(ValueError):
            self.account.transfer(200, target)
        # Both accounts unchanged after failed transfer
        self.assertEqual(self.account.balance, 100)
        self.assertEqual(target.balance, 50)


# === Exercise 2: Apply TDD ===
# Problem: TDD for a PriorityQueue class.

class PriorityQueue:
    """Priority queue: dequeue returns the highest-priority item first."""

    def __init__(self):
        self._items = []  # List of (priority, item) tuples

    def enqueue(self, item, priority):
        """Add item with given priority (higher number = higher priority)."""
        self._items.append((priority, item))

    def dequeue(self):
        """Remove and return the highest-priority item."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        # Find the item with the highest priority
        max_index = 0
        for i in range(1, len(self._items)):
            if self._items[i][0] > self._items[max_index][0]:
                max_index = i
        _, item = self._items.pop(max_index)
        return item

    def peek(self):
        """Return highest-priority item without removing it."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return max(self._items, key=lambda x: x[0])[1]

    def is_empty(self):
        """Check if queue has no items."""
        return len(self._items) == 0


class TestPriorityQueue(unittest.TestCase):
    """Tests written FIRST (TDD), then implementation was built to pass them."""

    def test_new_queue_is_empty(self):
        pq = PriorityQueue()
        self.assertTrue(pq.is_empty())

    def test_enqueue_makes_not_empty(self):
        pq = PriorityQueue()
        pq.enqueue("task", 1)
        self.assertFalse(pq.is_empty())

    def test_dequeue_returns_highest_priority(self):
        pq = PriorityQueue()
        pq.enqueue("low", 1)
        pq.enqueue("high", 3)
        pq.enqueue("medium", 2)
        self.assertEqual(pq.dequeue(), "high")

    def test_dequeue_empty_raises(self):
        pq = PriorityQueue()
        with self.assertRaises(IndexError):
            pq.dequeue()

    def test_peek_returns_without_removing(self):
        pq = PriorityQueue()
        pq.enqueue("task", 5)
        self.assertEqual(pq.peek(), "task")
        self.assertFalse(pq.is_empty())

    def test_peek_empty_raises(self):
        pq = PriorityQueue()
        with self.assertRaises(IndexError):
            pq.peek()

    def test_dequeue_order(self):
        """Items dequeue in priority order, not insertion order."""
        pq = PriorityQueue()
        pq.enqueue("C", 1)
        pq.enqueue("A", 3)
        pq.enqueue("B", 2)
        self.assertEqual(pq.dequeue(), "A")
        self.assertEqual(pq.dequeue(), "B")
        self.assertEqual(pq.dequeue(), "C")
        self.assertTrue(pq.is_empty())


# === Exercise 3: Test Doubles ===
# Problem: Test send_order_confirmation using stubs, mocks, and spies.

def exercise_3():
    """Solution: Test with stubs (database), mocks (email), spies (logger)."""

    @dataclass
    class Order:
        id: int
        user_id: int
        total: float

    @dataclass
    class User:
        id: int
        email: str

    def send_order_confirmation(order_id, database, email_service, logger):
        """
        Refactored version: dependencies are injected (not hardcoded).
        This makes it testable with test doubles.
        """
        order = database.get_order(order_id)
        user = database.get_user(order.user_id)
        email_service.send(
            to=user.email,
            subject=f"Order {order_id} confirmed",
            body=f"Your order for ${order.total} has been confirmed",
        )
        logger.log(f"Confirmation sent for order {order_id}")

    # --- Test using mocks ---

    # Stub: returns predefined data (replaces database)
    mock_db = MagicMock()
    mock_db.get_order.return_value = Order(id=42, user_id=1, total=99.99)
    mock_db.get_user.return_value = User(id=1, email="alice@test.com")

    # Mock: verifies interactions (replaces email service)
    mock_email = MagicMock()

    # Spy: records calls for later assertion (replaces logger)
    mock_logger = MagicMock()

    # Execute
    send_order_confirmation(42, mock_db, mock_email, mock_logger)

    # Verify stub was called correctly
    mock_db.get_order.assert_called_once_with(42)
    mock_db.get_user.assert_called_once_with(1)

    # Verify mock (email was sent with correct args)
    mock_email.send.assert_called_once_with(
        to="alice@test.com",
        subject="Order 42 confirmed",
        body="Your order for $99.99 has been confirmed",
    )

    # Verify spy (logger recorded the right message)
    mock_logger.log.assert_called_once_with("Confirmation sent for order 42")

    print("  All test double verifications passed:")
    print("    - Stub (database): returned predefined Order and User")
    print("    - Mock (email): verified send() called with correct args")
    print("    - Spy (logger): verified log() called with correct message")


# === Exercise 4: Integration Test ===
# Problem: Test a REST API CRUD workflow.

def exercise_4():
    """Solution: Integration test for task CRUD API (simulated)."""

    # Simulate a simple in-memory API for testing
    class TaskAPI:
        """Simulated REST API for tasks."""

        def __init__(self):
            self._tasks = {}
            self._next_id = 1

        def post(self, data):
            task_id = self._next_id
            self._next_id += 1
            task = {"id": task_id, **data}
            self._tasks[task_id] = task
            return 201, task

        def get_all(self):
            return 200, list(self._tasks.values())

        def get(self, task_id):
            if task_id in self._tasks:
                return 200, self._tasks[task_id]
            return 404, {"error": "Not found"}

        def put(self, task_id, data):
            if task_id in self._tasks:
                self._tasks[task_id].update(data)
                return 200, self._tasks[task_id]
            return 404, {"error": "Not found"}

        def delete(self, task_id):
            if task_id in self._tasks:
                del self._tasks[task_id]
                return 204, None
            return 404, {"error": "Not found"}

    # Integration test: full CRUD workflow
    api = TaskAPI()
    all_passed = True

    # CREATE
    status, task = api.post({"title": "Write tests", "done": False})
    assert status == 201, f"Create failed: {status}"
    task_id = task["id"]
    print(f"  CREATE: [{status}] {task}")

    # READ (single)
    status, task = api.get(task_id)
    assert status == 200, f"Get failed: {status}"
    assert task["title"] == "Write tests"
    print(f"  READ:   [{status}] {task}")

    # READ (all)
    status, tasks = api.get_all()
    assert status == 200 and len(tasks) == 1
    print(f"  LIST:   [{status}] {len(tasks)} task(s)")

    # UPDATE
    status, task = api.put(task_id, {"done": True})
    assert status == 200 and task["done"] is True
    print(f"  UPDATE: [{status}] {task}")

    # DELETE
    status, _ = api.delete(task_id)
    assert status == 204
    print(f"  DELETE: [{status}]")

    # Verify deleted
    status, _ = api.get(task_id)
    assert status == 404
    print(f"  VERIFY: [{status}] (correctly not found)")

    print("  Integration test passed: full CRUD lifecycle works")


# === Exercise 5: Property-Based Testing ===
# Problem: Property-based tests for merge_sorted_lists.

def exercise_5():
    """Solution: Property-based testing for sorted list merger."""

    def merge_sorted_lists(list1, list2):
        """Merge two sorted lists into a single sorted list."""
        result = []
        i, j = 0, 0
        while i < len(list1) and j < len(list2):
            if list1[i] <= list2[j]:
                result.append(list1[i])
                i += 1
            else:
                result.append(list2[j])
                j += 1
        result.extend(list1[i:])
        result.extend(list2[j:])
        return result

    def is_sorted(lst):
        """Check if a list is sorted in non-decreasing order."""
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    # Property-based testing: generate random inputs and verify properties
    random.seed(42)
    num_tests = 100
    failures = 0

    for _ in range(num_tests):
        # Generate random sorted lists
        list1 = sorted(random.sample(range(-100, 100), random.randint(0, 15)))
        list2 = sorted(random.sample(range(-100, 100), random.randint(0, 15)))

        result = merge_sorted_lists(list1, list2)

        # Property 1: Result length equals sum of input lengths
        if len(result) != len(list1) + len(list2):
            print(f"  FAIL: Length property - {len(result)} != {len(list1)} + {len(list2)}")
            failures += 1

        # Property 2: Result is sorted
        if not is_sorted(result):
            print(f"  FAIL: Sorted property - {result}")
            failures += 1

        # Property 3: All elements from inputs appear in result
        if sorted(result) != sorted(list1 + list2):
            print(f"  FAIL: Elements property")
            failures += 1

    # Property 4: Works with empty lists (explicit test)
    assert merge_sorted_lists([], []) == []
    assert merge_sorted_lists([1, 2], []) == [1, 2]
    assert merge_sorted_lists([], [3, 4]) == [3, 4]

    print(f"  Ran {num_tests} random test cases + 3 empty list tests")
    print(f"  Failures: {failures}")
    print(f"  Properties verified:")
    print(f"    1. Result length = sum of input lengths")
    print(f"    2. Result is always sorted")
    print(f"    3. All input elements appear in result")
    print(f"    4. Works with empty lists")


def exercise_1_run():
    """Run Exercise 1 tests and print results."""
    # Run the test suite programmatically
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestBankAccount))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    print(f"  Ran {result.testsRun} tests, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors")


def exercise_2_run():
    """Run Exercise 2 tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPriorityQueue))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    print(f"  Ran {result.testsRun} tests, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors")


if __name__ == "__main__":
    print("=== Exercise 1: Write Unit Tests ===")
    exercise_1_run()
    print("\n=== Exercise 2: Apply TDD ===")
    exercise_2_run()
    print("\n=== Exercise 3: Test Doubles ===")
    exercise_3()
    print("\n=== Exercise 4: Integration Test ===")
    exercise_4()
    print("\n=== Exercise 5: Property-Based Testing ===")
    exercise_5()
    print("\nAll exercises completed!")
