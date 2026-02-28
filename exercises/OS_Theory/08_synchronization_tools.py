"""
Exercises for Lesson 08: Synchronization Tools
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers mutex vs semaphore, semaphore traces, monitors, dining philosophers,
and readers-writers fairness.
"""

import threading
import queue
import time


# === Exercise 1: Mutex vs Semaphore Semantics ===
# Problem: Choose appropriate synchronization tool for each scenario.

def exercise_1():
    """Choose mutex or semaphore for synchronization scenarios."""
    scenarios = [
        {
            "scenario": "Protecting a shared int counter updated by 4 threads",
            "tool": "Mutex",
            "initial": "N/A (locked/unlocked)",
            "justification": (
                "Mutual exclusion of a single shared variable is the textbook "
                "use case for a mutex. Only one thread should read-modify-write "
                "the counter at a time. A binary semaphore would also work, but "
                "mutex is semantically clearer and enforces ownership (only the "
                "locker can unlock)."
            ),
        },
        {
            "scenario": "Limiting simultaneous database connections to 10",
            "tool": "Counting semaphore",
            "initial": "10",
            "justification": (
                "This is resource counting: up to 10 threads can hold the "
                "semaphore simultaneously. Each connection does P(sem) before "
                "connecting and V(sem) after disconnecting. A mutex only "
                "allows 1 thread -- we need counting semantics."
            ),
        },
        {
            "scenario": "Signaling a worker thread that a new task is in the queue",
            "tool": "Counting semaphore (or condition variable)",
            "initial": "0",
            "justification": (
                "This is signaling/notification, not mutual exclusion. The "
                "producer does V(sem) after adding a task; the worker does "
                "P(sem) to wait for tasks. Initial value 0 means the worker "
                "blocks until the producer signals. A condition variable with "
                "a mutex would also work."
            ),
        },
        {
            "scenario": "Ensuring a config file is read by exactly one thread at a time",
            "tool": "Mutex (or binary semaphore)",
            "initial": "N/A (or 1 for semaphore)",
            "justification": (
                "Single-thread exclusive access is mutual exclusion. Mutex is "
                "the natural choice. If the file is read-only, a readers-writers "
                "lock would allow concurrent reads, but the problem says 'exactly "
                "one thread at a time', so mutex is correct."
            ),
        },
        {
            "scenario": "Coordinating producer-consumer with bounded buffer (size=5)",
            "tool": "Two counting semaphores + mutex",
            "initial": "empty=5, full=0, mutex=1",
            "justification": (
                "Classic bounded buffer requires: 'empty' semaphore (init=5) to "
                "count empty slots, 'full' semaphore (init=0) to count filled "
                "slots, and a mutex to protect buffer access. Producer does "
                "P(empty), P(mutex), produce, V(mutex), V(full). Consumer does "
                "P(full), P(mutex), consume, V(mutex), V(empty)."
            ),
        },
    ]

    print("Mutex vs Semaphore Selection:\n")
    print(f"{'Scenario':<60} {'Tool':<25} {'Init'}")
    print("-" * 100)
    for s in scenarios:
        print(f"{s['scenario']:<60} {s['tool']:<25} {s['initial']}")
    print()
    for s in scenarios:
        print(f"Scenario: {s['scenario']}")
        print(f"  {s['justification']}")
        print()


# === Exercise 2: Semaphore Trace ===
# Problem: Trace semaphore values through producer-consumer operations.

def exercise_2():
    """Trace semaphore values for bounded buffer operations."""
    empty = 3
    full = 0
    mutex = 1

    operations = [
        ("Producer: P(empty)", "empty", -1, "Decrement empty"),
        ("Producer: P(mutex)", "mutex", -1, "Lock buffer"),
        ("Producer: add item", None, 0, "Buffer: [item1]"),
        ("Producer: V(mutex)", "mutex", 1, "Unlock buffer"),
        ("Producer: V(full)", "full", 1, "Signal item available"),
        ("Producer: P(empty)", "empty", -1, "Decrement empty"),
        ("Producer: P(mutex)", "mutex", -1, "Lock buffer"),
        ("Producer: add item", None, 0, "Buffer: [item1, item2]"),
        ("Producer: V(mutex)", "mutex", 1, "Unlock buffer"),
        ("Producer: V(full)", "full", 1, "Signal item available"),
        ("Producer: P(empty)", "empty", -1, "Decrement empty"),
        ("Producer: P(mutex)", "mutex", -1, "Lock buffer"),
        ("Producer: add item", None, 0, "Buffer: [item1, item2, item3]"),
        ("Producer: V(mutex)", "mutex", 1, "Unlock buffer"),
        ("Producer: V(full)", "full", 1, "Signal item available"),
        ("Producer: P(empty)", None, 0, "empty=0 -> BLOCKED! Buffer full"),
    ]

    print("Semaphore Trace (Buffer size=3):\n")
    print(f"{'Step':<5} {'Operation':<30} {'empty':<8} {'full':<8} {'mutex':<8} {'Notes'}")
    print("-" * 85)
    print(f"{'Init':<5} {'--':<30} {empty:<8} {full:<8} {mutex:<8} {'Buffer empty'}")

    for i, (op, sem, delta, notes) in enumerate(operations, 1):
        if sem == "empty":
            empty += delta
        elif sem == "full":
            full += delta
        elif sem == "mutex":
            mutex += delta

        blocked = ""
        if i == 16:
            blocked = " ** BLOCKED **"
            empty = 0  # stays at 0, producer blocks

        print(f"{i:<5} {op:<30} {empty:<8} {full:<8} {mutex:<8} {notes}{blocked}")

    print(f"\nAt step 16, the producer tries P(empty) but empty=0.")
    print(f"The producer BLOCKS until a consumer does V(empty) by consuming an item.")


# === Exercise 3: Monitor and Condition Variable Design ===
# Problem: Design a BoundedQueue monitor.

def exercise_3():
    """Implement a BoundedQueue monitor with condition variables."""
    print("BoundedQueue Monitor Design:\n")
    print("```pseudocode")
    print("monitor BoundedQueue:")
    print("    queue = []")
    print("    capacity = N")
    print("    condition not_full")
    print("    condition not_empty")
    print("")
    print("    procedure enqueue(item):")
    print("        while len(queue) == capacity:   // WHILE, not IF")
    print("            wait(not_full)              // release monitor lock, sleep")
    print("        queue.append(item)")
    print("        signal(not_empty)               // wake one waiting consumer")
    print("")
    print("    procedure dequeue():")
    print("        while len(queue) == 0:          // WHILE, not IF")
    print("            wait(not_empty)             // release monitor lock, sleep")
    print("        item = queue.pop(0)")
    print("        signal(not_full)                // wake one waiting producer")
    print("        return item")
    print("```\n")

    print("Q2: Why 'while' instead of 'if'?")
    print("  Spurious wakeups: A thread can be woken without the condition")
    print("  actually being true (OS implementation detail). Also, between")
    print("  being signaled and re-acquiring the monitor lock, another thread")
    print("  might have consumed the item. The 'while' re-checks the condition")
    print("  after waking, ensuring correctness.\n")

    print("Q3: Signal-and-wait semantics:")
    print("  In Mesa monitors (default): signal() adds the waiter to the ready")
    print("  queue but the signaler continues running. The waiter must re-check.")
    print("  In Hoare monitors (signal-and-wait): the signaler immediately gives")
    print("  up the monitor to the waiter. The waiter runs inside the monitor.")
    print("  The signaler is suspended until the waiter exits the monitor.")
    print("  With Hoare semantics, 'if' would suffice (condition guaranteed true")
    print("  when waiter runs), but 'while' is still recommended for safety.\n")

    # Python implementation
    print("--- Python Implementation ---\n")

    class BoundedQueue:
        def __init__(self, capacity):
            self.queue = []
            self.capacity = capacity
            self.lock = threading.Lock()
            self.not_full = threading.Condition(self.lock)
            self.not_empty = threading.Condition(self.lock)

        def enqueue(self, item):
            with self.not_full:
                while len(self.queue) >= self.capacity:
                    self.not_full.wait()
                self.queue.append(item)
                self.not_empty.notify()

        def dequeue(self):
            with self.not_empty:
                while len(self.queue) == 0:
                    self.not_empty.wait()
                item = self.queue.pop(0)
                self.not_full.notify()
                return item

    bq = BoundedQueue(3)
    results = []

    def producer():
        for i in range(5):
            bq.enqueue(i)
            results.append(f"Produced: {i}")

    def consumer():
        for _ in range(5):
            item = bq.dequeue()
            results.append(f"Consumed: {item}")

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start(); t2.start(); t1.join(); t2.join()

    for r in results:
        print(f"  {r}")
    print(f"\n  All items produced and consumed correctly with capacity=3 buffer.")


# === Exercise 4: Dining Philosophers Deadlock Analysis ===
# Problem: Demonstrate and fix the dining philosophers deadlock.

def exercise_4():
    """Analyze and fix the dining philosophers deadlock."""
    print("Dining Philosophers Deadlock Analysis\n")

    print("Q1: Deadlock scenario (5 philosophers, each picks left then right fork):")
    print("  Philosopher 0 picks up fork[0] (left)")
    print("  Philosopher 1 picks up fork[1] (left)")
    print("  Philosopher 2 picks up fork[2] (left)")
    print("  Philosopher 3 picks up fork[3] (left)")
    print("  Philosopher 4 picks up fork[4] (left)")
    print("  Now:")
    print("  Phil 0 waits for fork[1] (held by Phil 1)")
    print("  Phil 1 waits for fork[2] (held by Phil 2)")
    print("  Phil 2 waits for fork[3] (held by Phil 3)")
    print("  Phil 3 waits for fork[4] (held by Phil 4)")
    print("  Phil 4 waits for fork[0] (held by Phil 0)")
    print("  CIRCULAR WAIT -> DEADLOCK!\n")

    print("Q2: Resource hierarchy fix (always acquire lower-numbered fork first):")
    print("```c")
    print("void philosopher(int i) {")
    print("    int first = min(i, (i+1) % 5);")
    print("    int second = max(i, (i+1) % 5);")
    print("    while (true) {")
    print("        think();")
    print("        lock(fork[first]);    // always lower-numbered first")
    print("        lock(fork[second]);   // then higher-numbered")
    print("        eat();")
    print("        unlock(fork[second]);")
    print("        unlock(fork[first]);")
    print("    }")
    print("}")
    print("```")
    print("  For philosopher 4: forks are {4, 0}. min=0, max=4.")
    print("  So philosopher 4 picks up fork[0] first, then fork[4].")
    print("  This breaks circular wait: no cycle can form because all")
    print("  threads acquire in increasing order. If Phil 4 holds fork[0],")
    print("  Phil 0 cannot hold fork[0] and wait for fork[1].\n")

    print("Q3: Can starvation occur with resource hierarchy?")
    print("  In theory, yes. Consider Phil 1 always beating Phil 0 to fork[1].")
    print("  Phil 0 acquires fork[0] but repeatedly fails to get fork[1].")
    print("  However, this requires extremely unlikely persistent scheduling.")
    print("  With a fair mutex implementation (FIFO queuing), Phil 0 would")
    print("  eventually be granted fork[1]. In practice, starvation is not")
    print("  a real concern with the resource hierarchy solution.")

    # Python simulation
    print("\n--- Python simulation (resource hierarchy, 5 philosophers) ---\n")
    forks = [threading.Lock() for _ in range(5)]
    eat_count = [0] * 5

    def philosopher(pid, iterations=20):
        first = min(pid, (pid + 1) % 5)
        second = max(pid, (pid + 1) % 5)
        for _ in range(iterations):
            forks[first].acquire()
            forks[second].acquire()
            eat_count[pid] += 1
            forks[second].release()
            forks[first].release()

    threads = [threading.Thread(target=philosopher, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"  Eat counts (no deadlock!): {eat_count}")
    print(f"  Total meals: {sum(eat_count)} (expected 100)")


# === Exercise 5: Readers-Writers Problem Fairness ===
# Problem: Analyze starvation in readers-writers solutions.

def exercise_5():
    """Analyze readers-writers fairness and starvation."""
    print("Readers-Writers Fairness Analysis\n")

    print("Q1: Writer starvation under readers' preference:")
    print("  Writer arrives, finds 5 active readers. Under readers' preference,")
    print("  new readers can join even while a writer is waiting. Since readers")
    print("  arrive at 10/sec and each takes 50ms, at any moment ~0.5 readers")
    print("  are active on average. Before the current 5 finish, new readers arrive")
    print("  and start reading (they don't wait for the writer). The writer NEVER")
    print("  gets access as long as there is always at least 1 active reader.\n")

    print("Q2: Writers' preference pseudocode:")
    print("```pseudocode")
    print("semaphore resource = 1        // controls access to resource")
    print("semaphore rmutex = 1          // protects reader_count")
    print("semaphore wmutex = 1          // protects writer_count")
    print("semaphore read_try = 1        // readers must check this before entering")
    print("int reader_count = 0")
    print("int writer_count = 0")
    print("")
    print("WRITER:                        READER:")
    print("  P(wmutex)                      P(read_try)     // blocked if writer waiting")
    print("  writer_count++                 P(rmutex)")
    print("  if writer_count == 1:          reader_count++")
    print("      P(read_try)               if reader_count == 1:")
    print("  V(wmutex)                          P(resource)")
    print("  P(resource)                   V(rmutex)")
    print("  // --- write ---              V(read_try)")
    print("  V(resource)                   // --- read ---")
    print("  P(wmutex)                     P(rmutex)")
    print("  writer_count--                reader_count--")
    print("  if writer_count == 0:         if reader_count == 0:")
    print("      V(read_try)                   V(resource)")
    print("  V(wmutex)                     V(rmutex)")
    print("```\n")

    print("Q3: Writer finishes, 20 readers waiting:")
    print("  Under writers' preference, when the writer releases resource,")
    print("  it also releases read_try (if no more writers waiting).")
    print("  ALL 20 readers can proceed simultaneously! They each increment")
    print("  reader_count; the first one acquires resource, the rest join.")
    print("  Readers run concurrently -- they don't block each other.\n")

    print("Q4: Fair solution (no reader or writer starvation):")
    print("  Use a FIFO queue (service-order fairness). When any thread")
    print("  (reader or writer) arrives, it joins a single queue. Threads")
    print("  are admitted in arrival order. Consecutive readers in the queue")
    print("  can be admitted simultaneously, but a writer in the queue blocks")
    print("  subsequent readers until it finishes. This is sometimes called")
    print("  the 'fair readers-writers lock' or 'FIFO-ordered rw-lock'.")
    print("  Key mechanism: a queue semaphore that enforces FIFO ordering,")
    print("  preventing both reader and writer starvation.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Mutex vs Semaphore Semantics ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Semaphore Trace ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Monitor and Condition Variable Design ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Dining Philosophers Deadlock Analysis ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Readers-Writers Problem Fairness ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
