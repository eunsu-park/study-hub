"""
Exercises for Lesson 03: Threads and Multithreading
Topic: OS_Theory

Solutions to practice problems from the lesson.
"""

import threading
import time


# === Exercise 1: Thread vs Process Resource Sharing ===
# Problem: For a web server creating a thread per request, classify resources.

def exercise_1():
    """Classify thread resources as shared or private."""
    resources = [
        {
            "resource": "Global variable 'int total_requests'",
            "shared_or_private": "Shared",
            "reason": (
                "Global variables live in the Data section of the process's "
                "address space, which all threads share. Any thread can read "
                "or modify total_requests -- which is precisely why it needs "
                "a mutex to avoid race conditions."
            ),
        },
        {
            "resource": "Local variable 'char buf[4096]' inside the handler",
            "shared_or_private": "Private",
            "reason": (
                "Local variables are allocated on the thread's own stack. "
                "Each thread has a private stack, so each gets its own copy "
                "of buf. No synchronization needed."
            ),
        },
        {
            "resource": "Open file descriptor for the access log",
            "shared_or_private": "Shared",
            "reason": (
                "File descriptors (the open file table) are shared among all "
                "threads in a process. Multiple threads can write to the same "
                "log fd, but concurrent writes may interleave -- so buffered "
                "logging or a mutex is recommended."
            ),
        },
        {
            "resource": "errno value",
            "shared_or_private": "Private",
            "reason": (
                "errno is thread-local in POSIX-compliant systems (implemented "
                "as __thread or pthread-specific data). Each thread gets its own "
                "errno so that one thread's system call error doesn't clobber "
                "another thread's errno."
            ),
        },
        {
            "resource": "Heap-allocated malloc()'d request object",
            "shared_or_private": "Shared (accessible), but typically used privately",
            "reason": (
                "The heap is shared among all threads. Any thread with the "
                "pointer can access the object. However, by convention, the "
                "pointer is usually passed only to the owning thread, making "
                "it effectively private. If shared, synchronization is needed."
            ),
        },
        {
            "resource": "Signal disposition (SIGTERM -> graceful shutdown)",
            "shared_or_private": "Shared",
            "reason": (
                "Signal handlers (dispositions) are per-process, not per-thread. "
                "All threads share the same signal handler table. However, each "
                "thread has its own signal mask (which signals are blocked), "
                "and signals can be directed to specific threads."
            ),
        },
    ]

    print("Web Server Thread Resource Classification:\n")
    for r in resources:
        print(f"Resource: {r['resource']}")
        print(f"  Status: {r['shared_or_private']}")
        print(f"  Reason: {r['reason']}")
        print()


# === Exercise 2: Multithreading Model Comparison ===
# Problem: Recommend threading models for given scenarios.

def exercise_2():
    """Recommend threading models for various scenarios."""
    scenarios = [
        {
            "description": (
                "Scientific simulation parallelizing matrix multiplication "
                "across 8 CPU cores on a 12-core machine"
            ),
            "model": "1:1 (One-to-One)",
            "reasoning": (
                "Matrix multiplication is CPU-bound and benefits from true "
                "parallelism. 1:1 maps each user thread to a kernel thread, "
                "allowing the OS to schedule them on separate cores. With 8 "
                "threads on 12 cores, all can run simultaneously. N:1 would "
                "be useless (all threads on one core). M:N adds complexity "
                "with no benefit since we have enough cores."
            ),
        },
        {
            "description": (
                "Legacy embedded system with no kernel thread support that "
                "needs cooperative multitasking"
            ),
            "model": "N:1 (Many-to-One)",
            "reasoning": (
                "Without kernel thread support, N:1 is the only option. A "
                "user-space thread library manages all threads on a single "
                "kernel thread. Cooperative multitasking means threads yield "
                "voluntarily, which is exactly how N:1 works. The downside "
                "(no parallelism, blocking blocks all) is acceptable since "
                "the hardware is likely single-core anyway."
            ),
        },
        {
            "description": (
                "High-concurrency server handling 10,000 simultaneous "
                "connections on a 4-core system"
            ),
            "model": "M:N (Many-to-Many)",
            "reasoning": (
                "10,000 kernel threads would consume enormous memory for stacks "
                "and overwhelm the OS scheduler. M:N maps 10,000 user threads "
                "to ~4-16 kernel threads, keeping scheduling overhead low while "
                "utilizing all 4 cores. Go's goroutine model (M:N) excels here. "
                "Alternatively, an event loop (epoll) avoids threads entirely, "
                "but M:N is the correct answer in a threading context."
            ),
        },
        {
            "description": (
                "Simple desktop GUI app that offloads one background task "
                "to keep the UI responsive"
            ),
            "model": "1:1 (One-to-One)",
            "reasoning": (
                "Only 2 threads needed: UI thread + background worker. The "
                "overhead of 1:1 is negligible for 2 threads. 1:1 ensures "
                "the background thread doesn't block the UI thread (if the "
                "worker does I/O, only its kernel thread blocks). This is "
                "the standard model used by Linux, Windows, and macOS."
            ),
        },
    ]

    print("Threading Model Recommendations:\n")
    for i, s in enumerate(scenarios, 1):
        print(f"Scenario {i}: {s['description']}")
        print(f"  Recommended Model: {s['model']}")
        print(f"  Reasoning: {s['reasoning']}")
        print()


# === Exercise 3: Race Condition Analysis ===
# Problem: Identify race conditions in concurrent code.

def exercise_3():
    """Identify and fix race conditions in concurrent code."""
    print("Race condition analysis of the given code:\n")
    print("```c")
    print("int counter = 0;")
    print("int log_count = 0;")
    print("FILE *log_file;")
    print("void *worker(void *arg) {")
    print("    int id = *(int *)arg;")
    print("    for (int i = 0; i < 1000; i++) {")
    print("        counter++;                          // (A)")
    print('        fprintf(log_file, "thread %d: %d\\n", id, counter);  // (B)')
    print("        log_count++;                        // (C)")
    print("    }")
    print("    return NULL;")
    print("}")
    print("```\n")

    print("Q1: How many race conditions exist? Three:\n")
    races = [
        {
            "location": "(A) counter++",
            "shared_resource": "counter (global int)",
            "interleaving": (
                "Thread1 reads counter=5, Thread2 reads counter=5, "
                "Thread1 writes 6, Thread2 writes 6. One increment lost."
            ),
            "fix": "Protect with mutex_lock/unlock around counter++",
        },
        {
            "location": "(B) fprintf(log_file, ...)",
            "shared_resource": "log_file (shared FILE*) and counter read",
            "interleaving": (
                "Two threads call fprintf simultaneously. fprintf is not "
                "guaranteed atomic -- output from the two threads can "
                "interleave mid-line, producing garbled log entries. "
                "Also, counter is read without synchronization, so the "
                "logged value may be stale or inconsistent."
            ),
            "fix": "Include fprintf inside the same mutex-protected section",
        },
        {
            "location": "(C) log_count++",
            "shared_resource": "log_count (global int)",
            "interleaving": (
                "Same as counter++: read-modify-write is not atomic. "
                "Two threads can both read the same value, increment, "
                "and write back, losing one increment."
            ),
            "fix": "Protect with mutex_lock/unlock around log_count++",
        },
    ]
    for r in races:
        print(f"  Race at {r['location']}:")
        print(f"    Shared resource: {r['shared_resource']}")
        print(f"    Bad interleaving: {r['interleaving']}")
        print(f"    Fix: {r['fix']}")
        print()

    print("Q2: Expected value of counter after both threads finish?")
    print("  Expected (correct): 2000 (1000 increments per thread)")
    print("  Actual range: [1000, 2000]")
    print("  - Minimum 1000: every single increment collides (worst case)")
    print("  - Maximum 2000: no collisions at all (lucky scheduling)")
    print("  - Typical: somewhere between, e.g., 1700-1990\n")

    print("Q3: Fix with pthread_mutex_t:")
    print("```c")
    print("pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;")
    print("void *worker(void *arg) {")
    print("    int id = *(int *)arg;")
    print("    for (int i = 0; i < 1000; i++) {")
    print("        pthread_mutex_lock(&lock);")
    print("        counter++;")
    print('        fprintf(log_file, "thread %d: %d\\n", id, counter);')
    print("        log_count++;")
    print("        pthread_mutex_unlock(&lock);")
    print("    }")
    print("    return NULL;")
    print("}")
    print("```")

    # Demonstrate the race condition in Python
    print("\n--- Python demonstration of race condition ---\n")
    shared_counter = [0]  # mutable container to simulate shared state
    lock = threading.Lock()

    def unsafe_worker():
        for _ in range(100000):
            shared_counter[0] += 1

    def safe_worker():
        for _ in range(100000):
            with lock:
                shared_counter[0] += 1

    # Unsafe version
    shared_counter[0] = 0
    t1 = threading.Thread(target=unsafe_worker)
    t2 = threading.Thread(target=unsafe_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"Unsafe counter (expected 200000): {shared_counter[0]}")

    # Safe version
    shared_counter[0] = 0
    t1 = threading.Thread(target=safe_worker)
    t2 = threading.Thread(target=safe_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"Safe counter (expected 200000):   {shared_counter[0]}")


# === Exercise 4: TCB Fields Under Context Switch ===
# Problem: List TCB fields and consequences if NOT saved.

def exercise_4():
    """Analyze TCB fields required during context switch."""
    fields = [
        {
            "field": "Program Counter (PC)",
            "consequence": (
                "The thread would not resume at the correct instruction. "
                "It would jump to whatever address happened to be in the PC "
                "register after the new thread loaded -- likely causing a "
                "crash (segfault) or executing completely wrong code."
            ),
        },
        {
            "field": "Stack Pointer (SP)",
            "consequence": (
                "The thread would use the wrong stack. It would read garbage "
                "local variables, corrupt another thread's stack data, and "
                "likely crash on the next function return (popping a wrong "
                "return address)."
            ),
        },
        {
            "field": "General-purpose registers",
            "consequence": (
                "Intermediate computation results would be lost. If the thread "
                "was mid-calculation (e.g., accumulating a sum in RAX), that "
                "value would be replaced by another thread's data, producing "
                "silently incorrect results -- the worst kind of bug."
            ),
        },
        {
            "field": "Floating-point registers",
            "consequence": (
                "Any floating-point computation in progress (scientific "
                "calculations, graphics rendering) would produce wrong results. "
                "FP state is particularly large (SSE/AVX registers can be "
                "512 bits each), which is why some OSes use 'lazy FP save' "
                "to avoid saving them unless the thread actually uses FP."
            ),
        },
        {
            "field": "Thread state (Running/Ready/Blocked)",
            "consequence": (
                "The scheduler would not know which threads are eligible "
                "to run. A blocked thread might be scheduled (wasting CPU "
                "on a thread that can't make progress), or a ready thread "
                "might never be scheduled (starvation)."
            ),
        },
    ]

    print("TCB Fields and Consequences of Not Saving:\n")
    print(f"{'TCB Field':<30} {'Consequence if NOT Saved'}")
    print("-" * 80)
    for f in fields:
        print(f"\n{f['field']}:")
        print(f"  {f['consequence']}")


# === Exercise 5: Designing for Concurrency ===
# Problem: Redesign a single-threaded image server for throughput.

def exercise_5():
    """Design a multithreaded image processing server."""
    print("Single-threaded image processing server analysis:\n")

    # Single-threaded analysis
    network_io = 200  # ms
    resize = 50       # ms
    filter_op = 150   # ms
    disk_io = 100     # ms
    total = network_io + resize + filter_op + disk_io

    print(f"Step timings: Network I/O={network_io}ms, Resize={resize}ms, "
          f"Filter={filter_op}ms, Disk I/O={disk_io}ms")
    print(f"\nQ1: Single-threaded performance:")
    print(f"  Total latency: {total}ms per request")
    print(f"  Max throughput: {1000/total:.1f} requests/second")

    print(f"\nQ2: Multithreaded design:")
    print("  Use a thread pool with pipeline parallelism:")
    print("  - N worker threads (e.g., 4-8), each handling a full request")
    print("  - A shared task queue where incoming requests are placed")
    print("  - Each worker: dequeue request -> network recv -> resize -> filter -> disk write")
    print()
    print("  Architecture:")
    print("  [Acceptor Thread] -> [Task Queue] -> [Worker Thread 1]")
    print("                                    -> [Worker Thread 2]")
    print("                                    -> [Worker Thread 3]")
    print("                                    -> [Worker Thread N]")
    print()
    print("  Communication: Shared thread-safe queue (e.g., queue.Queue in Python)")
    print("  Each worker is independent -- no inter-thread communication needed")
    print("  for different requests.")

    print(f"\nQ3: Parallelism analysis:")
    print("  CPU-bound steps (benefit from multi-core parallelism):")
    print(f"    - Resize ({resize}ms): Pure computation, parallelizable")
    print(f"    - Filter ({filter_op}ms): Pure computation, parallelizable")
    print("  I/O-bound steps (limited by I/O, not CPU):")
    print(f"    - Network I/O ({network_io}ms): Waiting for data, thread just blocks")
    print(f"    - Disk I/O ({disk_io}ms): Waiting for disk, thread just blocks")
    print("  With threads, I/O waits don't block other requests' CPU work.")

    cpu_time = resize + filter_op  # 200ms
    io_time = network_io + disk_io  # 300ms
    print(f"\nQ4: Theoretical maximum throughput:")
    print(f"  CPU work per request: {cpu_time}ms")
    print(f"  I/O work per request: {io_time}ms")
    print(f"  Bottleneck: I/O ({io_time}ms > {cpu_time}ms)")
    print(f"  With enough worker threads, CPU and I/O overlap across requests.")
    print(f"  On a multi-core system, multiple CPU steps can run in parallel.")
    print(f"  With 2+ cores: CPU is not the bottleneck ({cpu_time}ms / 2 = {cpu_time//2}ms per core)")
    print(f"  The bottleneck becomes I/O: {io_time}ms per request")
    print(f"  But different requests' I/O can overlap (different sockets/disk blocks)")
    print(f"  Theoretical max: limited by disk bandwidth or network bandwidth,")
    print(f"  potentially processing {1000 / max(cpu_time/2, 1):.0f}+ requests/sec")
    print(f"  Practical estimate with 4 threads: ~{4 * 1000/total:.1f} req/sec")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Thread vs Process Resource Sharing ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Multithreading Model Comparison ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Race Condition Analysis ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: TCB Fields Under Context Switch ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Designing for Concurrency ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
