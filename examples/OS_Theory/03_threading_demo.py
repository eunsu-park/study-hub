"""
Threads and Multithreading Demonstration

Illustrates key threading concepts:
- Thread creation and lifecycle
- Race conditions with shared data
- Mutex-based synchronization
- Python GIL effects on CPU-bound vs I/O-bound work

Theory:
- Threads share the same address space within a process
- Concurrent access to shared data causes race conditions
- Mutual exclusion (mutex) prevents data corruption
- Python's GIL limits true parallelism for CPU-bound threads
  but allows concurrency for I/O-bound threads

Adapted from OS Theory Lesson 03.
"""

import threading
import time


# ── Race condition demonstration ─────────────────────────────────────────

shared_counter = 0


def unsafe_increment(n: int) -> None:
    """Increment shared counter without synchronization (race condition)."""
    global shared_counter
    for _ in range(n):
        # Read-modify-write is NOT atomic — we deliberately break it into
        # three separate steps (read, increment, write) to make the race
        # window visible. If another thread reads 'temp' between our read
        # and write, one increment is silently lost.
        temp = shared_counter
        temp += 1
        shared_counter = temp


def safe_increment(n: int, lock: threading.Lock) -> None:
    """Increment shared counter with mutex protection."""
    global shared_counter
    for _ in range(n):
        # The lock serializes the entire read-modify-write into one atomic
        # critical section, guaranteeing no lost updates at the cost of
        # eliminating concurrency on this shared variable
        with lock:
            shared_counter += 1


def demo_race_condition():
    """Show that unsynchronized threads produce incorrect results."""
    global shared_counter
    print("=" * 60)
    print("RACE CONDITION DEMONSTRATION")
    print("=" * 60)

    iterations = 100_000
    expected = iterations * 2

    # Without synchronization
    shared_counter = 0
    t1 = threading.Thread(target=unsafe_increment, args=(iterations,))
    t2 = threading.Thread(target=unsafe_increment, args=(iterations,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"\nWithout mutex:")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {shared_counter}")
    print(f"  Lost:     {expected - shared_counter} updates")

    # With synchronization
    shared_counter = 0
    lock = threading.Lock()
    t1 = threading.Thread(target=safe_increment, args=(iterations, lock))
    t2 = threading.Thread(target=safe_increment, args=(iterations, lock))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"\nWith mutex:")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {shared_counter}")
    print(f"  Correct:  {shared_counter == expected}")


# ── Thread lifecycle demonstration ───────────────────────────────────────

def worker(name: str, delay: float) -> None:
    """Simple worker that simulates work with sleep."""
    tid = threading.current_thread().ident
    print(f"  [{name}] Started (tid={tid})")
    time.sleep(delay)
    print(f"  [{name}] Finished after {delay}s")


def demo_thread_lifecycle():
    """Demonstrate thread creation, start, join."""
    print("\n" + "=" * 60)
    print("THREAD LIFECYCLE")
    print("=" * 60)

    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, args=(f"Worker-{i}", 0.1 * (i + 1)))
        threads.append(t)

    print(f"\nMain thread: {threading.current_thread().name}")
    print(f"Active threads before start: {threading.active_count()}")

    for t in threads:
        t.start()

    print(f"Active threads after start: {threading.active_count()}")

    # join() blocks the caller until the target thread terminates — without it,
    # the main thread could exit (and print "completed") before workers finish
    for t in threads:
        t.join()

    print(f"Active threads after join: {threading.active_count()}")
    print("All threads completed.")


# ── GIL effect demonstration ─────────────────────────────────────────────

def cpu_bound_work(n: int) -> float:
    """CPU-intensive computation."""
    total = 0.0
    for i in range(n):
        total += i * i
    return total


def io_bound_work(delay: float) -> None:
    """I/O-bound work (simulated with sleep)."""
    # time.sleep releases the GIL, just like real I/O syscalls (read, recv)
    # do — this is why I/O-bound threads achieve true concurrency in CPython
    time.sleep(delay)


def demo_gil_effect():
    """Show GIL impact: CPU-bound threads don't speed up, I/O-bound do."""
    print("\n" + "=" * 60)
    print("GIL EFFECT: CPU-BOUND vs I/O-BOUND")
    print("=" * 60)

    n = 2_000_000

    # CPU-bound: sequential
    start = time.perf_counter()
    cpu_bound_work(n)
    cpu_bound_work(n)
    seq_cpu = time.perf_counter() - start

    # CPU-bound: threaded
    start = time.perf_counter()
    t1 = threading.Thread(target=cpu_bound_work, args=(n,))
    t2 = threading.Thread(target=cpu_bound_work, args=(n,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    par_cpu = time.perf_counter() - start

    print(f"\nCPU-bound ({n:,} iterations x2):")
    print(f"  Sequential: {seq_cpu:.3f}s")
    print(f"  Threaded:   {par_cpu:.3f}s")
    print(f"  Speedup:    {seq_cpu / par_cpu:.2f}x (GIL limits ~1.0x)")

    # I/O-bound: sequential
    delay = 0.2
    start = time.perf_counter()
    io_bound_work(delay)
    io_bound_work(delay)
    seq_io = time.perf_counter() - start

    # I/O-bound: threaded
    start = time.perf_counter()
    t1 = threading.Thread(target=io_bound_work, args=(delay,))
    t2 = threading.Thread(target=io_bound_work, args=(delay,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    par_io = time.perf_counter() - start

    print(f"\nI/O-bound ({delay}s sleep x2):")
    print(f"  Sequential: {seq_io:.3f}s")
    print(f"  Threaded:   {par_io:.3f}s")
    print(f"  Speedup:    {seq_io / par_io:.2f}x (I/O releases GIL)")


if __name__ == "__main__":
    demo_race_condition()
    demo_thread_lifecycle()
    demo_gil_effect()
