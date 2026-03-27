"""
Exercises for Lesson 05: Scheduling Algorithms
Topic: OS_Theory

Solutions to practice problems from the lesson.
Implements FCFS, SJF, SRTF, and Round Robin scheduling with Gantt chart output.
"""

from collections import deque


def compute_metrics(processes, schedule):
    """Compute scheduling metrics from a Gantt chart schedule.

    Args:
        processes: list of (name, arrival, burst)
        schedule: list of (name, start, end) Gantt chart entries

    Returns:
        dict with per-process and average metrics
    """
    proc_info = {name: {"arrival": arr, "burst": burst}
                 for name, arr, burst in processes}
    results = {}

    for name in proc_info:
        info = proc_info[name]
        # Find first time this process runs (response time)
        first_start = None
        last_end = 0
        total_run = 0
        for sname, sstart, send in schedule:
            if sname == name:
                if first_start is None:
                    first_start = sstart
                last_end = send
                total_run += (send - sstart)

        completion = last_end
        turnaround = completion - info["arrival"]
        waiting = turnaround - info["burst"]
        response = first_start - info["arrival"] if first_start is not None else 0

        results[name] = {
            "completion": completion,
            "turnaround": turnaround,
            "waiting": waiting,
            "response": response,
        }

    return results


def print_gantt(schedule):
    """Print ASCII Gantt chart."""
    if not schedule:
        return
    bar = "|"
    times = str(schedule[0][1])
    for name, start, end in schedule:
        width = max(len(name) + 2, 4)
        bar += f" {name:^{width}} |"
        times += " " * (width - len(str(end)) + 2) + str(end)
    print(f"  {bar}")
    print(f"  {times}")


def print_results(processes, results):
    """Print per-process metrics table."""
    print(f"\n  {'Process':<10} {'Waiting':<10} {'Turnaround':<12} {'Response':<10}")
    print("  " + "-" * 42)
    for name, _, _ in processes:
        r = results[name]
        print(f"  {name:<10} {r['waiting']:<10} {r['turnaround']:<12} {r['response']:<10}")

    n = len(processes)
    avg_w = sum(results[n]['waiting'] for n, _, _ in processes) / n
    avg_t = sum(results[n]['turnaround'] for n, _, _ in processes) / n
    print(f"\n  Average waiting time: {avg_w:.2f}")
    print(f"  Average turnaround time: {avg_t:.2f}")
    return avg_w


# === Exercise 1: FCFS vs SJF Comparison ===

def fcfs(processes):
    """First-Come, First-Served scheduling."""
    sorted_procs = sorted(processes, key=lambda p: (p[1], p[0]))
    schedule = []
    current_time = 0
    for name, arrival, burst in sorted_procs:
        start = max(current_time, arrival)
        end = start + burst
        schedule.append((name, start, end))
        current_time = end
    return schedule


def sjf_nonpreemptive(processes):
    """Shortest Job First (non-preemptive) scheduling."""
    remaining = list(processes)
    schedule = []
    current_time = 0
    completed = set()

    while len(completed) < len(processes):
        # Find available processes
        available = [(name, arr, burst) for name, arr, burst in remaining
                     if arr <= current_time and name not in completed]
        if not available:
            # Jump to next arrival
            next_arrivals = [arr for name, arr, burst in remaining
                            if name not in completed]
            current_time = min(next_arrivals)
            continue

        # Pick shortest burst
        available.sort(key=lambda p: (p[2], p[1]))
        name, arrival, burst = available[0]
        start = current_time
        end = start + burst
        schedule.append((name, start, end))
        current_time = end
        completed.add(name)

    return schedule


def exercise_1():
    """Compare FCFS and non-preemptive SJF."""
    processes = [
        ("P1", 0, 8),
        ("P2", 1, 3),
        ("P3", 2, 6),
        ("P4", 3, 1),
        ("P5", 4, 4),
    ]

    print("Input processes:")
    print(f"  {'Process':<10} {'Arrival':<10} {'CPU Burst':<10}")
    print("  " + "-" * 30)
    for name, arr, burst in processes:
        print(f"  {name:<10} {arr:<10} {burst:<10}")

    # FCFS
    print("\n--- FCFS ---")
    fcfs_schedule = fcfs(processes)
    print_gantt(fcfs_schedule)
    fcfs_results = compute_metrics(processes, fcfs_schedule)
    fcfs_avg = print_results(processes, fcfs_results)

    # SJF
    print("\n--- SJF (Non-preemptive) ---")
    sjf_schedule = sjf_nonpreemptive(processes)
    print_gantt(sjf_schedule)
    sjf_results = compute_metrics(processes, sjf_schedule)
    sjf_avg = print_results(processes, sjf_results)

    print(f"\nQ4: Convoy effect in FCFS?")
    print(f"  Yes. P1 (burst=8) arrives first and runs for 8 time units.")
    print(f"  P2 (burst=3), P3 (burst=6), P4 (burst=1), and P5 (burst=4)")
    print(f"  all arrive between t=1 and t=4 but must wait for P1 to finish.")
    print(f"  P4 has a burst of only 1 but waits {fcfs_results['P4']['waiting']} time units.")
    print(f"  The cause is P1 -- the longest job that runs first.")
    print(f"\n  SJF avg waiting ({sjf_avg:.2f}) < FCFS avg waiting ({fcfs_avg:.2f})")


# === Exercise 2: SRTF Analysis ===

def srtf(processes):
    """Shortest Remaining Time First (preemptive SJF) scheduling."""
    remaining_time = {name: burst for name, _, burst in processes}
    arrival_map = {name: arr for name, arr, _ in processes}
    schedule = []
    current_time = 0
    completed = set()
    n = len(processes)
    prev_proc = None

    while len(completed) < n:
        # Find available processes with remaining time > 0
        available = [(name, remaining_time[name])
                     for name, arr, _ in processes
                     if arr <= current_time and name not in completed]
        if not available:
            next_arr = min(arr for name, arr, _ in processes
                          if name not in completed)
            current_time = next_arr
            continue

        # Pick shortest remaining time
        available.sort(key=lambda x: (x[1], arrival_map[x[0]]))
        chosen_name = available[0][0]

        # Find next event: either next arrival or completion
        next_arrivals = [arr for name, arr, _ in processes
                        if arr > current_time and name not in completed]
        time_to_complete = remaining_time[chosen_name]
        next_event = current_time + time_to_complete
        if next_arrivals:
            next_event = min(next_event, min(next_arrivals))

        run_time = next_event - current_time

        # Merge consecutive entries for same process
        if schedule and schedule[-1][0] == chosen_name and schedule[-1][2] == current_time:
            schedule[-1] = (chosen_name, schedule[-1][1], current_time + run_time)
        else:
            schedule.append((chosen_name, current_time, current_time + run_time))

        remaining_time[chosen_name] -= run_time
        current_time += run_time

        if remaining_time[chosen_name] == 0:
            completed.add(chosen_name)

    return schedule


def exercise_2():
    """SRTF analysis with the same process set."""
    processes = [
        ("P1", 0, 8),
        ("P2", 1, 3),
        ("P3", 2, 6),
        ("P4", 3, 1),
        ("P5", 4, 4),
    ]

    print("--- SRTF (Shortest Remaining Time First) ---")
    srtf_schedule = srtf(processes)
    print_gantt(srtf_schedule)
    srtf_results = compute_metrics(processes, srtf_schedule)
    srtf_avg = print_results(processes, srtf_results)

    # Also compute SJF for comparison
    sjf_schedule = sjf_nonpreemptive(processes)
    sjf_results = compute_metrics(processes, sjf_schedule)
    sjf_avg = sum(sjf_results[n]['waiting'] for n, _, _ in processes) / len(processes)

    print(f"\nQ2: SRTF avg waiting ({srtf_avg:.2f}) vs SJF avg waiting ({sjf_avg:.2f})")
    if srtf_avg < sjf_avg:
        print(f"  SRTF is lower because preemption allows short arriving jobs to")
        print(f"  interrupt long-running ones, reducing total accumulated waiting.")
    else:
        print(f"  In this case they are close; SRTF's benefit depends on arrival pattern.")

    print(f"\nQ3: At t=3, what is the remaining time of the running process?")
    # Trace through SRTF to find state at t=3
    remaining = {name: burst for name, _, burst in processes}
    current = None
    for name, start, end in srtf_schedule:
        if start <= 3 < end:
            current = name
            elapsed_in_segment = 3 - start
            rem_at_t3 = (end - start) - elapsed_in_segment
            # But we need actual remaining time
            break

    # Manual trace for accuracy
    print(f"  Tracing SRTF schedule entries covering t=3:")
    for name, start, end in srtf_schedule:
        if start <= 3:
            print(f"    {name}: [{start}, {end})")

    print(f"\nQ4: One disadvantage of SRTF not in SJF:")
    print(f"  Context switch overhead. SRTF preempts at every arrival event,")
    print(f"  potentially causing many more context switches than SJF.")
    print(f"  Non-preemptive SJF only switches when a process completes,")
    print(f"  making it cheaper in terms of overhead per decision.")


# === Exercise 3: Round Robin Time Quantum Trade-offs ===

def round_robin(processes, quantum):
    """Round Robin scheduling with given time quantum."""
    queue = deque()
    schedule = []
    remaining = {name: burst for name, _, burst in processes}
    arrival_map = {name: arr for name, arr, _ in processes}
    arrived = set()
    completed = set()
    current_time = 0
    sorted_procs = sorted(processes, key=lambda p: p[1])
    n = len(processes)

    # Add processes arriving at time 0
    for name, arr, burst in sorted_procs:
        if arr <= current_time and name not in arrived:
            queue.append(name)
            arrived.add(name)

    while len(completed) < n:
        if not queue:
            # Jump to next arrival
            next_arr = min(arrival_map[name] for name, _, _ in sorted_procs
                         if name not in completed and name not in arrived)
            current_time = next_arr
            for name, arr, burst in sorted_procs:
                if arr <= current_time and name not in arrived:
                    queue.append(name)
                    arrived.add(name)
            continue

        name = queue.popleft()
        run_time = min(quantum, remaining[name])
        start = current_time
        end = start + run_time
        schedule.append((name, start, end))
        remaining[name] -= run_time
        current_time = end

        # Add newly arrived processes to queue
        for pname, arr, burst in sorted_procs:
            if arr <= current_time and pname not in arrived:
                queue.append(pname)
                arrived.add(pname)

        if remaining[name] == 0:
            completed.add(name)
        else:
            queue.append(name)

    return schedule


def exercise_3():
    """Compare Round Robin with q=2 and q=5."""
    processes = [
        ("P1", 0, 6),
        ("P2", 1, 4),
        ("P3", 2, 8),
        ("P4", 3, 3),
    ]

    print("Input processes:")
    print(f"  {'Process':<10} {'Arrival':<10} {'CPU Burst':<10}")
    print("  " + "-" * 30)
    for name, arr, burst in processes:
        print(f"  {name:<10} {arr:<10} {burst:<10}")

    for q in [2, 5]:
        print(f"\n--- Round Robin (q={q}) ---")
        rr_schedule = round_robin(processes, q)
        print_gantt(rr_schedule)
        rr_results = compute_metrics(processes, rr_schedule)
        avg_w = print_results(processes, rr_results)
        context_switches = len(rr_schedule) - 1
        print(f"  Context switches: {context_switches}")

    print(f"\nQ4: Trade-off analysis:")
    print(f"  {'Quantum':<10} {'Avg Wait':<15} {'Ctx Switches':<18} {'Behaves like'}")
    print(f"  {'-'*55}")

    for q in [2, 5]:
        rr_schedule = round_robin(processes, q)
        rr_results = compute_metrics(processes, rr_schedule)
        avg_w = sum(rr_results[n]['waiting'] for n, _, _ in processes) / len(processes)
        cs = len(rr_schedule) - 1
        behavior = "Fair/responsive (more SRTF-like)" if q == 2 else "Less responsive (more FCFS-like)"
        print(f"  q={q:<7} {avg_w:<15.2f} {cs:<18} {behavior}")

    print(f"\n  Smaller quantum = more context switches but better response time")
    print(f"  Larger quantum = fewer switches but processes wait longer for their turn")


# === Exercise 4: Starvation and Aging ===

def exercise_4():
    """Analyze starvation and aging in priority scheduling."""
    print("Priority Scheduling Starvation Analysis\n")

    print("Initial state at t=0:")
    print("  P1: burst=10, priority=5 (lowest)")
    print("  P2: burst=2, priority=1 (highest)")
    print("  P3: burst=3, priority=2")
    print("  P4: burst=5, priority=3")
    print("  New priority-1 processes arrive every 4 time units\n")

    print("Q1: What happens to P1?")
    print("  Execution order (non-preemptive priority):")
    print("  t=0:  Run P2 (priority 1, burst 2) -> finishes at t=2")
    print("  t=2:  Run P3 (priority 2, burst 3) -> finishes at t=5")
    print("  t=4:  New P5 (priority 1) arrives!")
    print("  t=5:  P3 finishes. Run P5 (priority 1, arrived t=4) -> finishes at t=7")
    print("  t=7:  Run P4 (priority 3). P4 finishes at t=12")
    print("  t=8:  New P6 (priority 1) arrives! But P4 is non-preemptive...")
    print("  t=12: P4 finishes. Run P6 (priority 1) -> finishes at t=14")
    print("  t=12: New P7 (priority 1) arrived at t=12! Run after P6.")
    print("  ...and so on. P1 (priority 5) NEVER gets CPU access.")
    print("  This is indefinite starvation.\n")

    print("Q2: Aging policy: priority improves by 1 every 8 waiting time units")
    print("  P1 starts with priority 5 at t=0")
    t = 0
    priority = 5
    while priority > 1:
        t += 8
        priority -= 1
        print(f"  t={t}: P1 priority becomes {priority}")
    print(f"  At t={t}, P1's priority reaches 1 (highest), guaranteeing execution.\n")

    print("Q3: Real-world starvation scenario:")
    print("  In a hospital's network, critical monitoring data (high priority)")
    print("  might starve administrative traffic (low priority). If the network")
    print("  is always busy with vital signs data, administrative tasks like")
    print("  billing or records transfer never complete. This could cause delays")
    print("  in insurance processing or patient discharge paperwork.")


# === Exercise 5: Algorithm Selection ===

def exercise_5():
    """Select appropriate scheduling algorithms for given systems."""
    systems = [
        {
            "system": "Print server processing documents in submission order",
            "algorithm": "FCFS",
            "justification": (
                "FCFS matches the requirement exactly: first-submitted = first-printed. "
                "It's the simplest to implement (just a FIFO queue). The user specified "
                "that simplicity is valued over optimization. No starvation possible. "
                "The convoy effect is acceptable because print jobs are typically similar "
                "in duration and users expect FIFO behavior from a print queue."
            ),
        },
        {
            "system": "Real-time control system with time-critical tasks",
            "algorithm": "Priority (Preemptive)",
            "justification": (
                "Safety-critical tasks must preempt lower-priority housekeeping. "
                "Preemptive priority scheduling ensures that when a high-priority "
                "task (e.g., emergency stop) becomes ready, it immediately gets the "
                "CPU. RR would waste time on low-priority tasks. SJF doesn't consider "
                "urgency. Rate Monotonic or EDF could also work for periodic tasks."
            ),
        },
        {
            "system": "University cluster with interactive users expecting quick feedback",
            "algorithm": "Round Robin",
            "justification": (
                "RR provides fair time-sharing and bounded response time. With an "
                "appropriate quantum (10-50ms), interactive users get responsive "
                "feedback. No starvation. FCFS would cause poor response time. "
                "SJF/SRTF could starve long-running student jobs."
            ),
        },
        {
            "system": "Batch processing with accurate burst data, throughput is only goal",
            "algorithm": "SJF (or SRTF)",
            "justification": (
                "SJF is provably optimal for average waiting time among non-preemptive "
                "algorithms. Since burst times are accurately known (from historical data), "
                "the main weakness of SJF (prediction difficulty) is eliminated. "
                "Starvation of long jobs is acceptable in a throughput-only environment. "
                "SRTF would be even better if new jobs arrive during execution."
            ),
        },
        {
            "system": "OS kernel balancing interactive and background processes",
            "algorithm": "MLFQ (Multi-Level Feedback Queue)",
            "justification": (
                "MLFQ adapts automatically: interactive processes (short CPU bursts) "
                "stay in high-priority queues with small quanta for fast response. "
                "Background/batch processes (long CPU bursts) sink to lower queues "
                "with larger quanta for throughput. Periodic boost prevents starvation. "
                "This is exactly what Linux CFS and Windows scheduler approximate."
            ),
        },
    ]

    print("Algorithm Selection for Various Systems:\n")
    for i, s in enumerate(systems, 1):
        print(f"{i}. {s['system']}")
        print(f"   Algorithm: {s['algorithm']}")
        print(f"   Justification: {s['justification']}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: FCFS vs SJF Comparison ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: SRTF Analysis ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Round Robin Time Quantum Trade-offs ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Starvation and Aging ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Algorithm Selection ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
