"""
Synchronization Primitives Demonstration

Illustrates fundamental synchronization concepts:
- The critical section problem
- Peterson's algorithm for two-process mutual exclusion
- Mutex (mutual exclusion lock)
- Semaphore (counting semaphore)
- Barrier synchronization

Theory:
- Critical section: code that accesses shared resources
- Requirements: mutual exclusion, progress, bounded waiting
- Peterson's algorithm: software solution for 2 processes
- Mutex: binary lock (lock/unlock)
- Semaphore: integer counter with atomic wait/signal operations
- Barrier: synchronization point where all threads must arrive

Adapted from OS Theory Lesson 07.
"""

import threading
import time
from typing import Callable


# ── Peterson's Algorithm ─────────────────────────────────────────────────

class PetersonLock:
    """Peterson's algorithm for mutual exclusion between two threads.

    Uses two shared variables:
    - flag[i]: thread i wants to enter critical section
    - turn: whose turn it is when both want to enter
    """

    def __init__(self):
        self.flag = [False, False]
        self.turn = 0

    def acquire(self, tid: int) -> None:
        other = 1 - tid
        # flag[tid] = True announces intent BEFORE yielding turn — both
        # variables are needed: flag alone doesn't break symmetry (both
        # could set flag and enter simultaneously), and turn alone doesn't
        # guarantee mutual exclusion (a fast thread could re-enter while
        # the slow one is still in its critical section)
        self.flag[tid] = True
        self.turn = other
        # Spin while the OTHER thread wants in AND it's the other's turn.
        # If the other resets its flag, we proceed (progress). If both set
        # flags, exactly one will have set turn last, so that thread waits
        # (mutual exclusion).
        while self.flag[other] and self.turn == other:
            pass  # spin

    def release(self, tid: int) -> None:
        self.flag[tid] = False


def demo_peterson():
    """Demonstrate Peterson's algorithm with two threads."""
    print("=" * 60)
    print("PETERSON'S ALGORITHM")
    print("=" * 60)

    lock = PetersonLock()
    shared = [0]
    iterations = 50_000

    def worker(tid: int) -> None:
        for _ in range(iterations):
            lock.acquire(tid)
            # Critical section
            shared[0] += 1
            lock.release(tid)

    t0 = threading.Thread(target=worker, args=(0,))
    t1 = threading.Thread(target=worker, args=(1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    expected = iterations * 2
    print(f"\n  Expected: {expected}")
    print(f"  Actual:   {shared[0]}")
    print(f"  Correct:  {shared[0] == expected}")
    print(f"\n  Note: Peterson's works for exactly 2 threads.")
    print(f"  For N threads, use mutex/semaphore instead.")


# ── Semaphore demonstration ──────────────────────────────────────────────

class CountingSemaphore:
    """Counting semaphore implementation using threading primitives.

    - wait (P/down): decrement, block if zero
    - signal (V/up): increment, wake one blocked thread
    """

    def __init__(self, initial: int = 1):
        self._value = initial
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def wait(self) -> None:
        with self._condition:
            # Must use 'while' (not 'if') because of spurious wakeups — the
            # thread may be woken even when no signal occurred, so we re-check
            # the invariant before proceeding
            while self._value <= 0:
                self._condition.wait()
            self._value -= 1

    def signal(self) -> None:
        with self._condition:
            self._value += 1
            self._condition.notify()

    @property
    def value(self) -> int:
        return self._value


def demo_semaphore():
    """Demonstrate counting semaphore limiting concurrent access."""
    print("\n" + "=" * 60)
    print("COUNTING SEMAPHORE (Resource Pool)")
    print("=" * 60)

    max_concurrent = 3
    sem = CountingSemaphore(max_concurrent)
    active = [0]
    max_active = [0]
    lock = threading.Lock()

    def use_resource(name: str) -> None:
        sem.wait()
        with lock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            current = active[0]
        print(f"  [{name}] acquired (active={current}/{max_concurrent})")
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        sem.signal()
        print(f"  [{name}] released")

    threads = [
        threading.Thread(target=use_resource, args=(f"T{i}",))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n  Max concurrent: {max_active[0]} (limit={max_concurrent})")
    print(f"  Semaphore enforced: {max_active[0] <= max_concurrent}")


# ── Barrier demonstration ────────────────────────────────────────────────

class SimpleBarrier:
    """Reusable barrier for N threads."""

    def __init__(self, n: int):
        self._n = n
        self._count = 0
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._generation = 0

    def wait(self) -> None:
        gen = self._generation
        with self._lock:
            self._count += 1
            if self._count == self._n:
                # Last thread to arrive — reset count and create a NEW event for
                # the next barrier phase. If we reused the same event, threads
                # from the next phase could pass through immediately.
                self._count = 0
                self._generation += 1
                self._event.set()
                self._event = threading.Event()
                return

        # Wait for all threads to arrive — capture the event reference OUTSIDE
        # the lock to avoid holding it during the blocking wait
        event = self._event
        event.wait()


def demo_barrier():
    """Demonstrate barrier synchronization."""
    print("\n" + "=" * 60)
    print("BARRIER SYNCHRONIZATION")
    print("=" * 60)

    n_threads = 4
    barrier = SimpleBarrier(n_threads)
    results: list[str] = []
    results_lock = threading.Lock()

    def phased_worker(tid: int) -> None:
        # Phase 1
        msg = f"  T{tid}: Phase 1 done"
        print(msg)
        with results_lock:
            results.append(f"T{tid}:P1")
        barrier.wait()

        # Phase 2 — starts only after ALL threads finish phase 1
        msg = f"  T{tid}: Phase 2 done"
        print(msg)
        with results_lock:
            results.append(f"T{tid}:P2")

    print(f"\n  {n_threads} threads, 2 phases, barrier between phases:\n")
    threads = [threading.Thread(target=phased_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify: all P1 entries should come before any P2 entry
    p1_entries = [r for r in results if ":P1" in r]
    p2_entries = [r for r in results if ":P2" in r]
    last_p1_idx = max(results.index(e) for e in p1_entries)
    first_p2_idx = min(results.index(e) for e in p2_entries)
    print(f"\n  All Phase 1 before Phase 2: {last_p1_idx < first_p2_idx}")


if __name__ == "__main__":
    demo_peterson()
    demo_semaphore()
    demo_barrier()
