"""
Exercises for Lesson 04: CPU Scheduling Basics
Topic: OS_Theory

Solutions to practice problems from the lesson.
"""


# === Exercise 1: CPU-Bound vs I/O-Bound Classification ===
# Problem: Classify workloads and predict important scheduling criteria.

def exercise_1():
    """Classify workloads and identify important scheduling criteria."""
    workloads = [
        {
            "workload": "Video transcoder converting 4K footage",
            "classification": "CPU-bound",
            "criterion": "Throughput",
            "reasoning": (
                "Video transcoding is compute-intensive: decoding, filtering, "
                "and re-encoding frames uses nearly 100% CPU. I/O (reading/"
                "writing files) is a small fraction. Throughput matters most "
                "because the goal is to process as many frames/sec as possible."
            ),
        },
        {
            "workload": "Interactive text editor",
            "classification": "I/O-bound",
            "criterion": "Response time",
            "reasoning": (
                "A text editor spends most time waiting for user input (keyboard "
                "I/O). CPU bursts are tiny (rendering a character). Response time "
                "is critical -- users notice keystroke-to-display latency > 50ms."
            ),
        },
        {
            "workload": "Database query with many disk reads",
            "classification": "I/O-bound",
            "criterion": "Response time",
            "reasoning": (
                "Disk I/O dominates: the database reads pages from disk, does "
                "brief CPU work (comparisons, joins), then reads more pages. "
                "Response time matters for interactive queries; throughput for "
                "batch analytics."
            ),
        },
        {
            "workload": "Machine learning model training",
            "classification": "CPU-bound (GPU-bound in practice)",
            "criterion": "Throughput",
            "reasoning": (
                "Training involves matrix multiplications and gradient computations "
                "that saturate CPU/GPU. Data loading (I/O) is pipelined to keep "
                "the compute units busy. Throughput (samples/sec) is the key metric."
            ),
        },
        {
            "workload": "Web browser loading a webpage",
            "classification": "Mixed (I/O-dominant)",
            "criterion": "Response time",
            "reasoning": (
                "Loading involves network I/O (fetching HTML, CSS, JS, images), "
                "CPU work (parsing, rendering, JavaScript execution), and display "
                "I/O. The mix varies by page, but network latency usually dominates. "
                "Response time is crucial for user experience."
            ),
        },
        {
            "workload": "'cp' command copying 10 GB file",
            "classification": "I/O-bound",
            "criterion": "Throughput",
            "reasoning": (
                "File copying is purely I/O: read from source disk, write to "
                "destination. CPU usage is negligible (memcpy of buffers). "
                "Throughput (MB/s transfer rate) is the primary concern, "
                "bounded by disk bandwidth."
            ),
        },
    ]

    print("Workload Classification:\n")
    print(f"{'Workload':<45} {'Class':<20} {'Key Criterion':<15}")
    print("-" * 80)
    for w in workloads:
        print(f"{w['workload']:<45} {w['classification']:<20} {w['criterion']:<15}")
    print()
    for w in workloads:
        print(f"{w['workload']}:")
        print(f"  {w['reasoning']}")
        print()


# === Exercise 2: Scheduling Metrics Calculation ===
# Problem: FCFS scheduling for 5 processes with different arrival times.

def exercise_2():
    """Calculate FCFS scheduling metrics for 5 processes."""
    # Process: (name, arrival_time, burst_time)
    processes = [
        ("P1", 0, 10),
        ("P2", 1, 4),
        ("P3", 2, 7),
        ("P4", 3, 3),
        ("P5", 4, 5),
    ]

    print("FCFS Scheduling Metrics Calculation\n")
    print("Input:")
    print(f"{'Process':<10} {'Arrival':<10} {'CPU Burst':<10}")
    print("-" * 30)
    for name, arrival, burst in processes:
        print(f"{name:<10} {arrival:<10} {burst:<10}")

    # FCFS: processes execute in arrival order
    # Gantt chart construction
    current_time = 0
    gantt = []
    results = []

    for name, arrival, burst in processes:
        start_time = max(current_time, arrival)
        end_time = start_time + burst
        gantt.append((name, start_time, end_time))
        waiting_time = start_time - arrival
        turnaround_time = end_time - arrival
        response_time = start_time - arrival  # same as waiting for FCFS
        results.append({
            "name": name,
            "arrival": arrival,
            "burst": burst,
            "start": start_time,
            "end": end_time,
            "waiting": waiting_time,
            "turnaround": turnaround_time,
            "response": response_time,
        })
        current_time = end_time

    # Q1: Gantt chart
    print("\n1. Gantt Chart:")
    gantt_str = "|"
    time_str = "0"
    for name, start, end in gantt:
        width = max(len(name) + 2, len(str(end)) + 1)
        gantt_str += f" {name:^{width}} |"
        time_str += " " * (width + 1) + str(end)
    print(f"  {gantt_str}")
    print(f"  {time_str}")

    # Q2-4: Per-process metrics
    print("\n2-4. Per-process metrics:")
    print(f"{'Process':<10} {'Waiting':<10} {'Turnaround':<12} {'Response':<10}")
    print("-" * 42)
    for r in results:
        print(f"{r['name']:<10} {r['waiting']:<10} {r['turnaround']:<12} {r['response']:<10}")

    # Q5: Averages
    avg_waiting = sum(r['waiting'] for r in results) / len(results)
    avg_turnaround = sum(r['turnaround'] for r in results) / len(results)
    total_burst = sum(r['burst'] for r in results)
    total_time = results[-1]['end']
    cpu_util = (total_burst / total_time) * 100

    print(f"\n5. System-wide metrics:")
    print(f"  Average waiting time: {avg_waiting:.2f}")
    print(f"  Average turnaround time: {avg_turnaround:.2f}")
    print(f"  CPU utilization: {cpu_util:.1f}% ({total_burst}/{total_time} time units)")


# === Exercise 3: Preemption Decision Points ===
# Problem: Classify OS events as preemptive, non-preemptive, or neither.

def exercise_3():
    """Classify scheduling decision points."""
    events = [
        {
            "event": "Running process calls read() (blocks on I/O)",
            "type": "Non-preemptive",
            "reason": (
                "The process voluntarily gives up the CPU by requesting I/O. "
                "It transitions Running -> Waiting. The scheduler must pick "
                "a new process, but this is not preemption -- the process "
                "chose to block."
            ),
        },
        {
            "event": "Timer interrupt fires (time quantum expired)",
            "type": "Preemptive",
            "reason": (
                "The OS forcibly removes the process from the CPU when its "
                "time slice runs out. Running -> Ready. This is the defining "
                "characteristic of preemptive scheduling."
            ),
        },
        {
            "event": "Higher-priority process moves from Waiting to Ready",
            "type": "Preemptive",
            "reason": (
                "In a preemptive priority scheduler, this triggers re-evaluation. "
                "If the newly-ready process has higher priority than the running "
                "one, the running process is preempted (Running -> Ready) and "
                "the higher-priority process is dispatched."
            ),
        },
        {
            "event": "Running process calls exit()",
            "type": "Non-preemptive",
            "reason": (
                "The process voluntarily terminates. Running -> Terminated. "
                "The scheduler must select a new process, but exit() is a "
                "voluntary action, not forced preemption."
            ),
        },
        {
            "event": "New process is created with fork()",
            "type": "Preemptive (potentially)",
            "reason": (
                "In a preemptive priority scheduler, if the new child process "
                "has higher priority, the current process may be preempted. "
                "In non-preemptive schedulers, the child simply joins the "
                "ready queue. This is a scheduling decision point only in "
                "preemptive systems."
            ),
        },
        {
            "event": "I/O device sends completion interrupt",
            "type": "Preemptive (potentially)",
            "reason": (
                "The I/O completion moves a waiting process to Ready. "
                "In a preemptive scheduler, this triggers re-evaluation: "
                "if the newly-ready process has higher priority or shorter "
                "remaining time, preemption occurs. The interrupt itself "
                "always runs the ISR, but whether it triggers a context "
                "switch depends on the scheduling policy."
            ),
        },
    ]

    print("Scheduling Decision Point Classification:\n")
    print(f"{'Event':<55} {'Type':<20}")
    print("-" * 75)
    for e in events:
        print(f"{e['event']:<55} {e['type']:<20}")
    print()
    for e in events:
        print(f"{e['event']}:")
        print(f"  Type: {e['type']}")
        print(f"  Reason: {e['reason']}")
        print()


# === Exercise 4: Scheduler Type Roles ===
# Problem: Describe what each scheduler decides in a university cluster scenario.

def exercise_4():
    """Analyze scheduler roles in a university computing cluster."""
    print("Scenario: University computing cluster at 2 AM")
    print("  - 50 new batch jobs submitted")
    print("  - 80% memory usage")
    print("  - One simulation swapped out")
    print("  - Interactive student terminals active\n")

    print("1. Long-term Scheduler Decision:")
    print("   Decides: Which of the 50 new batch jobs to admit to the ready queue")
    print("   Criteria:")
    print("   - Memory is at 80%, so it should NOT admit all 50 at once")
    print("   - Admits a mix of CPU-bound (simulations) and I/O-bound jobs")
    print("     to keep both CPU and I/O devices busy")
    print("   - May limit to 10-15 new jobs and queue the rest on disk")
    print("   - Considers current multiprogramming degree vs available memory\n")

    print("2. Short-term Scheduler Decision:")
    print("   Decides: Which ready process gets the CPU next, every few ms")
    print("   Frequency: Runs every timer interrupt (~1-10ms)")
    print("   At this moment:")
    print("   - Must balance interactive terminal responsiveness with batch throughput")
    print("   - Likely uses MLFQ: interactive processes in high-priority queues")
    print("     (short CPU bursts), batch simulations in lower queues")
    print("   - Interactive terminals get fast response; batch jobs get remaining CPU\n")

    print("3. Medium-term Scheduler Decision:")
    print("   Decides: Which processes to swap in/out based on memory pressure")
    print("   In this scenario:")
    print("   - Memory is at 80% and 50 new jobs want to load")
    print("   - The swapped-out simulation should be swapped back in when memory frees")
    print("   - May swap out idle batch jobs to make room for active ones")
    print("   - Needed because admitting all jobs would cause thrashing\n")

    print("4. Most critical for interactive terminal response time:")
    print("   The SHORT-TERM scheduler is most critical.")
    print("   It determines how quickly a terminal process gets the CPU after")
    print("   the student presses a key. If the short-term scheduler gives")
    print("   priority to batch jobs, terminals become unresponsive.")
    print("   The long-term and medium-term schedulers affect which processes")
    print("   are in memory, but the short-term scheduler controls latency.")


# === Exercise 5: Dispatcher Latency Analysis ===
# Problem: Analyze dispatch latency and optimization.

def exercise_5():
    """Analyze dispatcher latency and trade-offs."""
    dispatch_latency_us = 3  # microseconds
    switches_per_sec = 5000

    print("Dispatcher Latency Analysis\n")

    # Q1: Total dispatch time per second
    total_us = dispatch_latency_us * switches_per_sec
    total_ms = total_us / 1000
    print(f"Q1: Total time in dispatcher per second:")
    print(f"  {dispatch_latency_us} us * {switches_per_sec} switches = {total_us} us = {total_ms} ms\n")

    # Q2: Percentage of CPU time
    percentage = (total_us / 1e6) * 100
    print(f"Q2: Percentage of CPU time consumed by dispatching:")
    print(f"  {total_ms} ms / 1000 ms = {percentage:.1f}%\n")

    # Q3: Reducing dispatch latency
    print("Q3: Three operations contributing to dispatch latency and optimizations:\n")
    operations = [
        {
            "operation": "Register save/restore (~1 us)",
            "optimization": (
                "Use hardware-assisted context switch instructions (e.g., x86 "
                "XSAVE/XRSTOR for FP/SSE state). Implement lazy FP save -- "
                "skip saving floating-point registers unless the next process "
                "actually uses FP. Reduces the number of registers to save in "
                "the common case."
            ),
        },
        {
            "operation": "TLB flush / page table switch (~1 us)",
            "optimization": (
                "Use PCID (Process Context Identifiers) on modern CPUs to tag "
                "TLB entries per process. This avoids flushing the entire TLB "
                "on context switch -- entries from the previous process remain "
                "valid if that process runs again soon."
            ),
        },
        {
            "operation": "Scheduling algorithm execution (~1 us)",
            "optimization": (
                "Use O(1) scheduling algorithms like CFS's red-black tree "
                "(O(log n) for pick-next). Pre-compute the next process to run "
                "during idle time. Use per-CPU run queues to avoid lock contention "
                "on multiprocessor systems."
            ),
        },
    ]
    for op in operations:
        print(f"  {op['operation']}:")
        print(f"    Optimization: {op['optimization']}")
        print()

    # Q4: Trade-off analysis
    new_switches = 1000
    new_total_us = dispatch_latency_us * new_switches
    new_percentage = (new_total_us / 1e6) * 100
    print(f"Q4: Reducing switches from {switches_per_sec} to {new_switches}/sec:")
    print(f"  New dispatch overhead: {new_percentage:.2f}% (down from {percentage:.1f}%)")
    print(f"  Response time increases: 10ms -> 25ms\n")

    print("  (a) Batch processing server: WORTHWHILE")
    print("      Batch servers care about throughput, not response time.")
    print("      Saving ~1.35% CPU overhead directly increases throughput.")
    print("      No users are waiting for interactive feedback.\n")

    print("  (b) Interactive desktop OS: NOT WORTHWHILE")
    print("      25ms response time is noticeably sluggish for typing,")
    print("      mouse movement, and UI interactions. Users perceive delays")
    print("      above ~50ms, and 25ms is getting close. The 1.35% CPU")
    print("      savings is not worth the degraded user experience.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: CPU-Bound vs I/O-Bound Classification ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Scheduling Metrics Calculation ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Preemption Decision Points ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Scheduler Type Roles ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Dispatcher Latency Analysis ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
