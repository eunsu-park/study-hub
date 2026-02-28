"""
Exercises for Lesson 11: Debugging and Profiling
Topic: Programming

Solutions to practice problems from the lesson.
"""
import time
import threading


# === Exercise 1: Debug Buggy Code ===
# Problem: Find and fix bugs in transaction processing code.

def exercise_1():
    """Solution: Identify and fix syntax and logic bugs."""

    # Original buggy code:
    # def process_transactions(transactions):
    #     total = 0
    #     for transaction in transactions:
    #         if transaction['type'] = 'debit':    # BUG 1: = instead of ==
    #             total -= transaction['amount']
    #         else:
    #             total += transaction['amount']
    #     return total

    # Bug 1 (Syntax): Single '=' is assignment, not comparison.
    # Should be '==' for equality check.
    # This would cause a SyntaxError in Python (can't assign in if condition).

    # Fixed version:
    def process_transactions(transactions):
        """Calculate net balance from a list of credit/debit transactions."""
        total = 0
        for transaction in transactions:
            if transaction["type"] == "debit":  # FIX: == for comparison
                total -= transaction["amount"]
            else:
                total += transaction["amount"]
        return total

    transactions = [
        {"type": "credit", "amount": 100},
        {"type": "debit", "amount": 50},
        {"type": "credit", "amount": 200},
    ]

    result = process_transactions(transactions)
    # Expected: +100 - 50 + 200 = 250
    print(f"  Fixed result: {result}")
    print(f"  Expected: 250")
    print(f"  Correct: {result == 250}")

    print("\n  Bugs found:")
    print("    1. SYNTAX: 'if transaction['type'] = 'debit'' uses '=' (assignment)")
    print("       instead of '==' (comparison). Python raises SyntaxError.")
    print("    2. No LOGIC error in this case: the else clause correctly handles")
    print("       credits, and debits are subtracted. The algorithm is sound")
    print("       once the syntax is fixed.")


# === Exercise 2: Profile and Optimize ===
# Problem: Profile and optimize find_common_elements.

def exercise_2():
    """Solution: Profile, identify bottleneck, optimize with sets."""

    # Original O(n*m) version with O(n) 'in' check on list
    def find_common_elements_slow(list1, list2):
        """Brute force: O(n * m * k) where k is len(common)."""
        common = []
        for item1 in list1:
            for item2 in list2:
                if item1 == item2 and item1 not in common:  # 'not in' is O(k)
                    common.append(item1)
        return common

    # Optimized O(n + m) version using set intersection
    def find_common_elements_fast(list1, list2):
        """
        Set-based: O(n + m) time.
        Converting to sets gives O(1) membership testing.
        Set intersection is the natural operation for finding common elements.
        """
        return list(set(list1) & set(list2))

    # Profile both versions
    list1 = list(range(5000))
    list2 = list(range(2500, 7500))

    # Time the slow version (with smaller input to avoid long wait)
    small_list1 = list(range(1000))
    small_list2 = list(range(500, 1500))

    start = time.perf_counter()
    result_slow = find_common_elements_slow(small_list1, small_list2)
    time_slow = time.perf_counter() - start

    start = time.perf_counter()
    result_fast = find_common_elements_fast(list1, list2)
    time_fast = time.perf_counter() - start

    print(f"  Slow version (n=1000): {time_slow:.4f}s, found {len(result_slow)} common")
    print(f"  Fast version (n=5000): {time_fast:.4f}s, found {len(result_fast)} common")
    print(f"  Speedup: ~{time_slow / max(time_fast, 0.0001):.0f}x faster (even with 5x more data)")

    print("\n  Bottleneck analysis:")
    print("    - Nested loops: O(n*m) comparisons")
    print("    - 'not in common' on list: O(k) per check, total O(n*m*k)")
    print("    - Fix: set intersection is O(n + m)")


# === Exercise 3: Memory Leak Detection ===
# Problem: Fix memory leak in event subscriber pattern.

def exercise_3():
    """Solution: Fix memory leak by adding unsubscribe method."""

    # The memory leak: subscribers are never removed.
    # When components are destroyed, their callback references persist
    # in the subscribers list, preventing garbage collection.

    class DataStore:
        """Fixed version with proper unsubscribe support."""

        def __init__(self):
            self.data = []
            self._subscribers = []

        def add_data(self, item):
            self.data.append(item)
            self._notify_subscribers()

        def subscribe(self, callback):
            """Register a callback. Returns an unsubscribe function."""
            self._subscribers.append(callback)

            # Return a cleanup function (like React's useEffect cleanup)
            def unsubscribe():
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

            return unsubscribe

        def _notify_subscribers(self):
            for callback in self._subscribers:
                callback(self.data)

        @property
        def subscriber_count(self):
            return len(self._subscribers)

    # Demonstrate the fix
    store = DataStore()

    # Simulate creating and destroying components
    unsubscribers = []
    for i in range(5):
        # Each component subscribes
        unsub = store.subscribe(lambda data, idx=i: None)  # Simulated callback
        unsubscribers.append(unsub)

    print(f"  After creating 5 components: {store.subscriber_count} subscribers")

    # Clean up 3 components
    for unsub in unsubscribers[:3]:
        unsub()

    print(f"  After destroying 3 components: {store.subscriber_count} subscribers")

    # Clean up remaining
    for unsub in unsubscribers[3:]:
        unsub()

    print(f"  After destroying all: {store.subscriber_count} subscribers")

    print("\n  Why it leaked:")
    print("    Callbacks held references to component objects.")
    print("    Without unsubscribe, GC couldn't collect destroyed components.")
    print("  Fix: Return an unsubscribe function from subscribe().")
    print("    Components call unsubscribe() in their cleanup/destructor.")


# === Exercise 4: Fix Race Condition ===
# Problem: Fix concurrent withdrawal race condition.

def exercise_4():
    """Solution: Fix race condition using a lock/mutex."""

    # The race condition: two threads check balance simultaneously,
    # both see sufficient funds, both withdraw, resulting in negative balance.

    # --- Broken version (race condition) ---
    class BrokenAccount:
        def __init__(self, balance):
            self.balance = balance

        def withdraw(self, amount):
            if self.balance >= amount:
                time.sleep(0.01)  # Simulate processing delay
                self.balance -= amount
                return True
            return False

    # --- Fixed version (with lock) ---
    class SafeAccount:
        """Thread-safe account using a mutex lock."""

        def __init__(self, balance):
            self.balance = balance
            self._lock = threading.Lock()

        def withdraw(self, amount):
            """
            Thread-safe withdrawal using a lock.
            The lock ensures only one thread can check-and-modify
            the balance at a time (atomic operation).
            """
            with self._lock:
                if self.balance >= amount:
                    time.sleep(0.01)  # Simulate processing
                    self.balance -= amount
                    return True
                return False

    # Test broken version
    print("  Broken version (race condition):")
    broken = BrokenAccount(100)
    results = [None, None]

    def broken_withdraw(idx, account, amount):
        results[idx] = account.withdraw(amount)

    t1 = threading.Thread(target=broken_withdraw, args=(0, broken, 60))
    t2 = threading.Thread(target=broken_withdraw, args=(1, broken, 60))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"    Withdrawals: {results}")
    print(f"    Final balance: {broken.balance}")
    print(f"    Bug: {'balance went negative!' if broken.balance < 0 else 'got lucky this time'}")

    # Test fixed version
    print("\n  Fixed version (with lock):")
    safe = SafeAccount(100)
    results = [None, None]

    def safe_withdraw(idx, account, amount):
        results[idx] = account.withdraw(amount)

    t1 = threading.Thread(target=safe_withdraw, args=(0, safe, 60))
    t2 = threading.Thread(target=safe_withdraw, args=(1, safe, 60))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"    Withdrawals: {results}")
    print(f"    Final balance: {safe.balance}")
    print(f"    Correct: {safe.balance >= 0} (balance never negative)")

    print("\n  How the lock fixes it:")
    print("    Thread 1 acquires lock -> checks balance (100 >= 60) -> withdraws -> releases")
    print("    Thread 2 acquires lock -> checks balance (40 >= 60?) -> DENIED -> releases")
    print("    The check-and-modify is now atomic (indivisible)")


if __name__ == "__main__":
    print("=== Exercise 1: Debug Buggy Code ===")
    exercise_1()
    print("\n=== Exercise 2: Profile and Optimize ===")
    exercise_2()
    print("\n=== Exercise 3: Memory Leak Detection ===")
    exercise_3()
    print("\n=== Exercise 4: Fix Race Condition ===")
    exercise_4()
    print("\nAll exercises completed!")
