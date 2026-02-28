"""
Exercises for Lesson 12: Concurrency and Parallelism
Topic: Programming

Solutions to practice problems from the lesson.
"""
import threading
import queue
import time
import random
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


# === Exercise 1: Identify Race Conditions ===
# Problem: Find and fix the race condition in concurrent withdrawal.

def exercise_1():
    """Solution: Fix race condition with a threading lock."""

    # The race condition: two threads read balance simultaneously,
    # both see 1000 >= 600, both subtract 600, ending at -200.

    balance = 1000
    lock = threading.Lock()
    results = []

    def withdraw_safe(amount):
        """
        Thread-safe withdrawal using a lock.
        The lock makes the check-and-subtract atomic.
        """
        nonlocal balance
        with lock:  # Only one thread can hold this lock at a time
            if balance >= amount:
                time.sleep(0.01)  # Simulate processing
                balance -= amount
                results.append(("success", amount))
                return True
            results.append(("failed", amount))
            return False

    t1 = threading.Thread(target=lambda: withdraw_safe(600))
    t2 = threading.Thread(target=lambda: withdraw_safe(600))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"  Results: {results}")
    print(f"  Final balance: {balance}")
    print(f"  Correct (>= 0): {balance >= 0}")
    print(f"  One should succeed, one should fail: "
          f"{sum(1 for r, _ in results if r == 'success')} success, "
          f"{sum(1 for r, _ in results if r == 'failed')} failed")


# === Exercise 2: Implement Producer-Consumer ===
# Problem: 3 producers generate numbers, 2 consumers compute sum.

def exercise_2():
    """Solution: Thread-safe producer-consumer with queue."""

    POISON_PILL = None  # Sentinel value to signal consumers to stop
    NUM_PRODUCERS = 3
    NUM_CONSUMERS = 2
    ITEMS_PER_PRODUCER = 10

    work_queue = queue.Queue(maxsize=20)  # Bounded queue prevents memory issues
    total_sum = 0
    sum_lock = threading.Lock()

    def producer(producer_id):
        """Generate random numbers and put them in the queue."""
        for _ in range(ITEMS_PER_PRODUCER):
            num = random.randint(1, 100)
            work_queue.put(num)
        # Signal that this producer is done
        work_queue.put(POISON_PILL)

    def consumer(consumer_id):
        """Take numbers from queue and add to running sum."""
        nonlocal total_sum
        local_sum = 0
        pills_received = 0

        while True:
            item = work_queue.get()
            if item is POISON_PILL:
                pills_received += 1
                if pills_received >= 1:
                    # Put pill back for other consumers, then exit
                    work_queue.put(POISON_PILL)
                    break
            else:
                local_sum += item
            work_queue.task_done()

        # Safely add to global sum
        with sum_lock:
            total_sum += local_sum
            print(f"    Consumer {consumer_id}: local sum = {local_sum}")

    random.seed(42)

    # Start producers
    producers = [threading.Thread(target=producer, args=(i,)) for i in range(NUM_PRODUCERS)]
    for p in producers:
        p.start()

    # Start consumers
    consumers = [threading.Thread(target=consumer, args=(i,)) for i in range(NUM_CONSUMERS)]
    for c in consumers:
        c.start()

    # Wait for all to finish
    for p in producers:
        p.join()
    for c in consumers:
        c.join()

    print(f"  Total sum from {NUM_PRODUCERS} producers x {ITEMS_PER_PRODUCER} items: {total_sum}")


# === Exercise 3: Async/Await for I/O ===
# Problem: Rewrite sequential fetch to use async/await.

def exercise_3():
    """Solution: Concurrent API calls using asyncio."""

    async def fetch_user(user_id):
        """Simulate fetching a user from an API."""
        await asyncio.sleep(0.1)  # Simulate 100ms network delay
        return {"id": user_id, "name": f"User{user_id}"}

    async def fetch_orders(user_id):
        """Simulate fetching orders from an API."""
        await asyncio.sleep(0.1)  # Simulate 100ms network delay
        return [{"id": 1, "total": 100}, {"id": 2, "total": 200}]

    # Sequential version (for comparison)
    async def get_user_with_orders_sequential(user_id):
        """Takes ~200ms: each call waits for the previous one."""
        user = await fetch_user(user_id)
        orders = await fetch_orders(user_id)
        return {"user": user, "orders": orders}

    # Optimized concurrent version
    async def get_user_with_orders_concurrent(user_id):
        """
        Takes ~100ms: both calls run simultaneously.
        asyncio.gather runs multiple coroutines concurrently,
        returning when ALL complete.
        """
        user, orders = await asyncio.gather(
            fetch_user(user_id),
            fetch_orders(user_id),
        )
        return {"user": user, "orders": orders}

    async def compare():
        # Sequential
        start = time.perf_counter()
        result = await get_user_with_orders_sequential(123)
        seq_time = time.perf_counter() - start

        # Concurrent
        start = time.perf_counter()
        result = await get_user_with_orders_concurrent(123)
        conc_time = time.perf_counter() - start

        print(f"  Sequential: {seq_time:.3f}s")
        print(f"  Concurrent: {conc_time:.3f}s")
        print(f"  Speedup: {seq_time / conc_time:.1f}x")
        print(f"  Result: {result}")

    asyncio.run(compare())


# === Exercise 4: Parallel Map-Reduce ===
# Problem: Parallel word count using map-reduce pattern.

def exercise_4():
    """Solution: Parallel word count with ThreadPoolExecutor."""

    # Simulated text files (in-memory strings)
    texts = {
        "file1.txt": "the quick brown fox jumps over the lazy dog",
        "file2.txt": "the fox ate the dog and the cat",
        "file3.txt": "quick fox quick fox lazy lazy dog",
    }

    def count_words_in_text(text):
        """
        MAP phase: count words in a single text.
        Returns a Counter (word frequency dict).
        Each call is independent and can run in parallel.
        """
        words = text.lower().split()
        return Counter(words)

    def merge_counts(count1, count2):
        """REDUCE phase: merge two word count dictionaries."""
        result = Counter(count1)
        result.update(count2)
        return result

    def parallel_word_count(text_dict):
        """
        Map-Reduce word count with parallel map phase.

        Map: Each text is processed independently (parallelizable).
        Reduce: Merge results sequentially (simple aggregation).
        """
        # MAP phase: process each file in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            word_counts = list(executor.map(count_words_in_text, text_dict.values()))

        # REDUCE phase: merge all counts
        total_counts = Counter()
        for counts in word_counts:
            total_counts.update(counts)

        return total_counts

    result = parallel_word_count(texts)

    print("  Word frequencies (parallel map-reduce):")
    for word, count in result.most_common(10):
        print(f"    '{word}': {count}")
    print(f"  Total unique words: {len(result)}")
    print(f"  Total word count: {sum(result.values())}")


# === Exercise 5: Deadlock Scenario ===
# Problem: Fix a deadlock caused by inconsistent lock ordering.

def exercise_5():
    """Solution: Fix deadlock by enforcing consistent lock acquisition order."""

    # The deadlock:
    # Thread 1: lock_a -> lock_b
    # Thread 2: lock_b -> lock_a
    # If Thread 1 gets lock_a and Thread 2 gets lock_b simultaneously,
    # each waits for the other's lock forever.

    # Fix: ALWAYS acquire locks in the same order (e.g., always A before B).
    # This is the "lock ordering" strategy.

    lock_a = threading.Lock()
    lock_b = threading.Lock()
    account_a = {"name": "A", "balance": 1000}
    account_b = {"name": "B", "balance": 1000}

    def transfer(from_account, to_account, amount, lock1, lock2):
        """
        Transfer money between accounts with consistent lock ordering.
        Both transfer directions acquire locks in the SAME order (A then B)
        to prevent deadlock.
        """
        # Always acquire the "lower" lock first (consistent ordering)
        first_lock = lock1 if id(lock1) < id(lock2) else lock2
        second_lock = lock2 if id(lock1) < id(lock2) else lock1

        with first_lock:
            with second_lock:
                if from_account["balance"] >= amount:
                    from_account["balance"] -= amount
                    to_account["balance"] += amount
                    return True
                return False

    # Run concurrent transfers in both directions
    results = []

    def do_transfer(from_acc, to_acc, amount, l1, l2, idx):
        success = transfer(from_acc, to_acc, amount, l1, l2)
        results.append((idx, success))

    threads = []
    for i in range(5):
        # Alternate between A->B and B->A transfers
        if i % 2 == 0:
            t = threading.Thread(target=do_transfer,
                                 args=(account_a, account_b, 100, lock_a, lock_b, i))
        else:
            t = threading.Thread(target=do_transfer,
                                 args=(account_b, account_a, 100, lock_b, lock_a, i))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)  # Timeout to detect deadlock

    all_finished = all(not t.is_alive() for t in threads)
    total = account_a["balance"] + account_b["balance"]

    print(f"  All threads completed: {all_finished}")
    print(f"  Account A: {account_a['balance']}")
    print(f"  Account B: {account_b['balance']}")
    print(f"  Total (should be 2000): {total}")
    print(f"  No deadlock: {all_finished}")
    print(f"  Money conserved: {total == 2000}")

    print("\n  Deadlock prevention strategies:")
    print("    1. Lock ordering: always acquire locks in same global order")
    print("    2. Lock timeout: try_lock with timeout, retry on failure")
    print("    3. Single lock: use one lock for all related operations")


if __name__ == "__main__":
    print("=== Exercise 1: Identify Race Conditions ===")
    exercise_1()
    print("\n=== Exercise 2: Implement Producer-Consumer ===")
    exercise_2()
    print("\n=== Exercise 3: Async/Await for I/O ===")
    exercise_3()
    print("\n=== Exercise 4: Parallel Map-Reduce ===")
    exercise_4()
    print("\n=== Exercise 5: Deadlock Scenario ===")
    exercise_5()
    print("\nAll exercises completed!")
