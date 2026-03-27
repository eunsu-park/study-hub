"""
Producer-Consumer and Dining Philosophers

Classic synchronization problems demonstrating:
- Bounded-buffer producer-consumer with semaphores
- Dining philosophers with deadlock-free solutions

Theory:
- Producer-consumer: producers generate items into a shared buffer,
  consumers remove them. Requires synchronization to prevent overflow,
  underflow, and data corruption.
- Dining philosophers: N philosophers sit around a table with N forks.
  Each needs two forks to eat. Naive solutions can deadlock.

Solutions demonstrated:
- Bounded buffer: mutex + empty/full semaphores
- Philosophers: resource hierarchy (pick lower-numbered fork first)

Adapted from OS Theory Lesson 08.
"""

import threading
import time
import random
from collections import deque


# ── Bounded-Buffer Producer-Consumer ────────────────────────────────────

class BoundedBuffer:
    """Thread-safe bounded buffer using semaphores.

    Synchronization:
    - mutex: protects buffer access
    - empty: counts empty slots (producer waits when 0)
    - full: counts filled slots (consumer waits when 0)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque()
        self.mutex = threading.Lock()
        # Three semaphores enforce distinct invariants: mutex protects the
        # buffer data structure, 'empty' counts available slots (blocks
        # producers when buffer is full), 'full' counts available items
        # (blocks consumers when buffer is empty). Using two counting
        # semaphores instead of one avoids the "lost wakeup" problem.
        self.empty = threading.Semaphore(capacity)  # starts at capacity
        self.full = threading.Semaphore(0)           # starts at 0

    def produce(self, item) -> None:
        # Acquire 'empty' BEFORE mutex — acquiring mutex first would deadlock
        # if the buffer is full (we'd hold mutex while waiting, blocking consumers)
        self.empty.acquire()       # wait for empty slot
        with self.mutex:
            self.buffer.append(item)
        self.full.release()        # signal item available

    def consume(self):
        # Mirror of produce: acquire 'full' BEFORE mutex for the same reason —
        # semaphore ordering must be consistent to prevent deadlock
        self.full.acquire()        # wait for item
        with self.mutex:
            item = self.buffer.popleft()
        self.empty.release()       # signal slot freed
        return item

    def size(self) -> int:
        with self.mutex:
            return len(self.buffer)


def demo_producer_consumer():
    """Demonstrate bounded-buffer producer-consumer."""
    print("=" * 60)
    print("BOUNDED-BUFFER PRODUCER-CONSUMER")
    print("=" * 60)

    buf = BoundedBuffer(capacity=5)
    produced = []
    consumed = []
    produced_lock = threading.Lock()
    consumed_lock = threading.Lock()
    n_items = 15

    def producer(pid: int, count: int) -> None:
        for i in range(count):
            item = f"P{pid}-{i}"
            buf.produce(item)
            with produced_lock:
                produced.append(item)
            print(f"  Producer {pid}: produced {item} (buf size={buf.size()})")
            time.sleep(random.uniform(0.01, 0.05))

    def consumer(cid: int, count: int) -> None:
        for _ in range(count):
            item = buf.consume()
            with consumed_lock:
                consumed.append(item)
            print(f"  Consumer {cid}: consumed {item} (buf size={buf.size()})")
            time.sleep(random.uniform(0.02, 0.06))

    # 3 producers (5 items each), 3 consumers (5 items each)
    threads = []
    for i in range(3):
        threads.append(threading.Thread(target=producer, args=(i, 5)))
    for i in range(3):
        threads.append(threading.Thread(target=consumer, args=(i, 5)))

    print(f"\n  Buffer capacity: 5")
    print(f"  Producers: 3 (5 items each = {n_items} total)")
    print(f"  Consumers: 3 (5 items each = {n_items} total)\n")

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n  Produced: {len(produced)} items")
    print(f"  Consumed: {len(consumed)} items")
    print(f"  All consumed: {sorted(produced) == sorted(consumed)}")
    print(f"  Buffer empty: {buf.size() == 0}")


# ── Dining Philosophers ─────────────────────────────────────────────────

class DiningPhilosophers:
    """Dining philosophers with resource-hierarchy solution.

    Deadlock prevention: each philosopher picks up the lower-numbered
    fork first. This breaks the circular-wait condition.
    """

    def __init__(self, n: int):
        self.n = n
        self.forks = [threading.Lock() for _ in range(n)]
        self.eat_count = [0] * n
        self.log: list[str] = []
        self.log_lock = threading.Lock()

    def _log(self, msg: str) -> None:
        with self.log_lock:
            self.log.append(msg)
            print(f"  {msg}")

    def philosopher(self, pid: int, meals: int) -> None:
        left = pid
        right = (pid + 1) % self.n

        # Resource hierarchy: always pick lower-numbered fork first — this
        # breaks the circular-wait condition because no cycle can form in
        # the resource acquisition graph when all threads order resources
        # by the same global numbering
        first = min(left, right)
        second = max(left, right)

        for _ in range(meals):
            # Think
            self._log(f"Philosopher {pid}: thinking")
            time.sleep(random.uniform(0.01, 0.03))

            # Pick up forks (ordered)
            self.forks[first].acquire()
            self._log(f"Philosopher {pid}: picked fork {first}")
            self.forks[second].acquire()
            self._log(f"Philosopher {pid}: picked fork {second}")

            # Eat
            self._log(f"Philosopher {pid}: eating")
            self.eat_count[pid] += 1
            time.sleep(random.uniform(0.01, 0.03))

            # Put down forks
            self.forks[second].release()
            self.forks[first].release()
            self._log(f"Philosopher {pid}: put down forks")


def demo_dining_philosophers():
    """Demonstrate dining philosophers with deadlock-free solution."""
    print("\n" + "=" * 60)
    print("DINING PHILOSOPHERS (Resource Hierarchy)")
    print("=" * 60)

    n = 5
    meals = 3
    dp = DiningPhilosophers(n)

    print(f"\n  {n} philosophers, {meals} meals each")
    print(f"  Deadlock prevention: resource hierarchy\n")

    threads = [
        threading.Thread(target=dp.philosopher, args=(i, meals))
        for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n  Meals per philosopher: {dp.eat_count}")
    print(f"  Total meals: {sum(dp.eat_count)} (expected {n * meals})")
    print(f"  All ate equally: {all(c == meals for c in dp.eat_count)}")
    print(f"  No deadlock: True (completed successfully)")


# ── Deadlock demonstration (timeout-based) ──────────────────────────────

def demo_deadlock_potential():
    """Show how naive fork ordering CAN deadlock (with timeout escape)."""
    print("\n" + "=" * 60)
    print("DEADLOCK DEMONSTRATION (Naive Fork Ordering)")
    print("=" * 60)

    n = 5
    forks = [threading.Lock() for _ in range(n)]
    deadlocked = threading.Event()

    def naive_philosopher(pid: int) -> None:
        left = pid
        right = (pid + 1) % n

        # Naive: always pick left first, then right → circular wait possible
        acquired_left = forks[left].acquire(timeout=1.0)
        if not acquired_left:
            deadlocked.set()
            return

        # Delay between acquiring left and right forks to widen the window
        # where all philosophers hold one fork — this makes circular wait
        # (and thus deadlock) almost certain
        time.sleep(0.05)

        acquired_right = forks[right].acquire(timeout=1.0)
        if not acquired_right:
            forks[left].release()
            deadlocked.set()
            return

        # Eat (if we get here)
        time.sleep(0.01)
        forks[right].release()
        forks[left].release()

    print(f"\n  {n} philosophers with naive ordering (timeout=1s)")
    print(f"  Each picks left fork first → circular wait possible\n")

    threads = [
        threading.Thread(target=naive_philosopher, args=(i,))
        for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if deadlocked.is_set():
        print("  Result: DEADLOCK detected (timeout triggered)")
        print("  Fix: use resource hierarchy ordering")
    else:
        print("  Result: got lucky, no deadlock this time")
        print("  But deadlock is still possible with naive ordering!")


if __name__ == "__main__":
    demo_producer_consumer()
    demo_dining_philosophers()
    demo_deadlock_potential()
