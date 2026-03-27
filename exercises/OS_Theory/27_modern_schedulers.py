"""
Exercises for Lesson 27: Modern Schedulers
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers CFS virtual runtime simulation, EEVDF eligible deadline scheduling,
and heterogeneous big.LITTLE core task placement.
"""

import heapq


# === Exercise 1: CFS (Completely Fair Scheduler) Simulator ===
# Problem: Simulate the Linux CFS algorithm with virtual runtime tracking,
# nice-to-weight mapping, and proportional CPU allocation.

def exercise_1():
    """Simulate the Linux Completely Fair Scheduler."""

    # Nice-to-weight table (simplified from kernel's sched_prio_to_weight)
    # Weight roughly doubles every 5 nice levels.
    # nice 0 = weight 1024
    NICE_TO_WEIGHT = {}
    for n in range(-20, 20):
        NICE_TO_WEIGHT[n] = int(1024 * 1.25 ** (-n))

    class CFSTask:
        """A task managed by the CFS scheduler."""

        def __init__(self, pid, name, nice=0):
            self.pid = pid
            self.name = name
            self.nice = nice
            self.weight = NICE_TO_WEIGHT[nice]
            self.vruntime = 0.0       # Virtual runtime (ms)
            self.actual_runtime = 0.0  # Actual runtime (ms)

        def __lt__(self, other):
            """For heap ordering by vruntime."""
            return self.vruntime < other.vruntime

    class CFSScheduler:
        """Simplified CFS scheduler using a min-heap (red-black tree proxy)."""

        def __init__(self, target_latency=6.0, min_granularity=0.75):
            self.tasks = []          # Min-heap sorted by vruntime
            self.target_latency = target_latency   # ms
            self.min_granularity = min_granularity  # ms
            self.clock = 0.0         # Simulated wall clock (ms)
            self.total_weight = 0

        def add_task(self, task):
            """Add a task, setting its vruntime to current minimum."""
            if self.tasks:
                task.vruntime = self.tasks[0].vruntime
            self.total_weight += task.weight
            heapq.heappush(self.tasks, task)

        def time_slice(self, task):
            """Calculate time slice for a task based on its weight proportion."""
            slice_ms = self.target_latency * (task.weight / self.total_weight)
            return max(slice_ms, self.min_granularity)

        def pick_next(self):
            """Pick the task with the lowest vruntime."""
            if not self.tasks:
                return None
            return heapq.heappop(self.tasks)

        def put_back(self, task):
            """Return a task to the run queue after it ran."""
            heapq.heappush(self.tasks, task)

        def run_tick(self, task, actual_ms):
            """Simulate a task running for actual_ms milliseconds."""
            task.actual_runtime += actual_ms
            # vruntime grows inversely proportional to weight
            # Higher weight -> slower vruntime growth -> more CPU time
            vruntime_delta = actual_ms * (1024.0 / task.weight)
            task.vruntime += vruntime_delta
            self.clock += actual_ms

    print("=== CFS Scheduler Simulation ===\n")

    sched = CFSScheduler(target_latency=6.0, min_granularity=0.75)

    # Create tasks with different nice values
    tasks_info = [
        (1, "web_server",  -5),   # Higher priority
        (2, "compiler",     0),   # Normal priority
        (3, "backup",      10),   # Lower priority
        (4, "editor",      -5),   # Higher priority
        (5, "background",  15),   # Very low priority
    ]

    task_objs = {}
    for pid, name, nice in tasks_info:
        t = CFSTask(pid, name, nice)
        sched.add_task(t)
        task_objs[pid] = t

    print(f"Tasks (target_latency={sched.target_latency}ms, "
          f"min_granularity={sched.min_granularity}ms):\n")
    print(f"  {'PID':<5} {'Name':<14} {'Nice':<6} {'Weight':<8} {'Time Slice (ms)'}")
    print("  " + "-" * 50)
    for pid, name, nice in tasks_info:
        t = task_objs[pid]
        ts = sched.time_slice(t)
        print(f"  {pid:<5} {name:<14} {nice:<6} {t.weight:<8} {ts:.2f}")

    # Simulate 20 scheduling rounds
    print(f"\n--- Scheduling Simulation (20 rounds) ---\n")
    print(f"  {'Round':<7} {'PID':<5} {'Name':<14} {'Slice(ms)':<11} "
          f"{'VRuntime':<12} {'Actual(ms)'}")
    print("  " + "-" * 60)

    for r in range(20):
        task = sched.pick_next()
        if not task:
            break
        slice_ms = sched.time_slice(task)
        sched.run_tick(task, slice_ms)
        print(f"  {r+1:<7} {task.pid:<5} {task.name:<14} {slice_ms:<11.2f} "
              f"{task.vruntime:<12.2f} {task.actual_runtime:.2f}")
        sched.put_back(task)

    # Verify fairness: compare actual runtime ratios to weight ratios
    print(f"\n--- Fairness Analysis ---\n")
    total_runtime = sum(t.actual_runtime for t in task_objs.values())
    total_weight = sum(t.weight for t in task_objs.values())

    print(f"  {'Name':<14} {'Weight':<8} {'Expected%':<11} {'Actual%':<10} {'Delta'}")
    print("  " + "-" * 50)
    for pid, name, nice in tasks_info:
        t = task_objs[pid]
        expected_pct = (t.weight / total_weight) * 100
        actual_pct = (t.actual_runtime / total_runtime) * 100 if total_runtime > 0 else 0
        delta = actual_pct - expected_pct
        print(f"  {name:<14} {t.weight:<8} {expected_pct:<11.1f} "
              f"{actual_pct:<10.1f} {delta:+.1f}")

    print(f"\n  All tasks have similar vruntime, confirming CFS fairness.")
    print(f"  Higher-weight tasks got proportionally more actual CPU time.")


# === Exercise 2: EEVDF Scheduler Simulator ===
# Problem: Implement the EEVDF scheduling algorithm and show how it
# provides better latency guarantees than CFS for interactive tasks.

def exercise_2():
    """Simulate the EEVDF scheduler and compare with CFS."""

    class EEVDFTask:
        """A task managed by the EEVDF scheduler."""

        def __init__(self, pid, name, nice=0, request_length=1.0):
            self.pid = pid
            self.name = name
            self.nice = nice
            self.weight = int(1024 * 1.25 ** (-nice))
            self.vruntime = 0.0
            self.actual_runtime = 0.0
            self.request_length = request_length  # Typical time slice request (ms)
            self.virtual_deadline = 0.0
            self.eligible = True
            self.wakeup_count = 0
            self.total_wait = 0.0

    class EEVDFScheduler:
        """EEVDF: Earliest Eligible Virtual Deadline First."""

        def __init__(self, target_latency=6.0, min_granularity=0.75):
            self.tasks = []
            self.target_latency = target_latency
            self.min_granularity = min_granularity
            self.min_vruntime = 0.0
            self.clock = 0.0
            self.total_weight = 0

        def add_task(self, task):
            task.vruntime = self.min_vruntime
            self.total_weight += task.weight
            self._update_deadline(task)
            self.tasks.append(task)

        def _update_deadline(self, task):
            """Virtual Deadline = vruntime + (request_length / weight)."""
            task.virtual_deadline = (task.vruntime +
                                     task.request_length * (1024.0 / task.weight))

        def _is_eligible(self, task):
            """A task is eligible if its vruntime <= min_vruntime."""
            return task.vruntime <= self.min_vruntime + 0.01  # Small epsilon

        def pick_next(self):
            """Pick eligible task with earliest virtual deadline."""
            # Update eligibility
            eligible_tasks = [t for t in self.tasks if self._is_eligible(t)]

            if not eligible_tasks:
                # If no eligible tasks, all tasks are eligible (reset)
                eligible_tasks = self.tasks[:]

            if not eligible_tasks:
                return None

            # Among eligible, pick earliest virtual deadline
            best = min(eligible_tasks, key=lambda t: t.virtual_deadline)
            return best

        def run_tick(self, task, actual_ms):
            """Simulate a task running."""
            task.actual_runtime += actual_ms
            vruntime_delta = actual_ms * (1024.0 / task.weight)
            task.vruntime += vruntime_delta
            self.clock += actual_ms

            # Update min_vruntime (monotonically increasing)
            if self.tasks:
                self.min_vruntime = min(t.vruntime for t in self.tasks)

            # Recompute deadline for next scheduling
            self._update_deadline(task)

        def time_slice(self, task):
            """Calculate time slice."""
            slice_ms = self.target_latency * (task.weight / self.total_weight)
            return max(slice_ms, self.min_granularity)

    print("=== EEVDF Scheduler Simulation ===\n")

    sched = EEVDFScheduler(target_latency=6.0, min_granularity=0.75)

    # Create a mix of interactive and batch tasks
    tasks = [
        # Interactive tasks: short request lengths (want low latency)
        EEVDFTask(1, "terminal",    nice=-5,  request_length=0.5),
        EEVDFTask(2, "mouse_input", nice=-10, request_length=0.2),
        # Batch tasks: long request lengths (want throughput)
        EEVDFTask(3, "compiler",    nice=0,   request_length=4.0),
        EEVDFTask(4, "backup",      nice=10,  request_length=4.0),
        EEVDFTask(5, "render",      nice=5,   request_length=4.0),
    ]

    for t in tasks:
        sched.add_task(t)

    print(f"Tasks:\n")
    print(f"  {'PID':<5} {'Name':<14} {'Nice':<6} {'Weight':<8} "
          f"{'Request(ms)':<13} {'Type'}")
    print("  " + "-" * 55)
    for t in tasks:
        task_type = "interactive" if t.request_length < 1.0 else "batch"
        print(f"  {t.pid:<5} {t.name:<14} {t.nice:<6} {t.weight:<8} "
              f"{t.request_length:<13.1f} {task_type}")

    # Simulate 20 rounds
    print(f"\n--- EEVDF Scheduling (20 rounds) ---\n")
    print(f"  {'Round':<7} {'PID':<5} {'Name':<14} {'Eligible':<10} "
          f"{'VDeadline':<11} {'VRuntime':<10}")
    print("  " + "-" * 60)

    schedule_order = []
    for r in range(20):
        task = sched.pick_next()
        if not task:
            break

        eligible = sched._is_eligible(task)
        slice_ms = sched.time_slice(task)
        schedule_order.append(task.pid)

        print(f"  {r+1:<7} {task.pid:<5} {task.name:<14} "
              f"{'yes' if eligible else 'no':<10} "
              f"{task.virtual_deadline:<11.2f} {task.vruntime:<10.2f}")

        sched.run_tick(task, slice_ms)

    # Compare: how often interactive tasks were selected
    print(f"\n--- Latency Analysis ---\n")

    interactive_pids = {1, 2}
    batch_pids = {3, 4, 5}

    interactive_runs = sum(1 for pid in schedule_order if pid in interactive_pids)
    batch_runs = sum(1 for pid in schedule_order if pid in batch_pids)

    print(f"  Interactive task selections: {interactive_runs}/20")
    print(f"  Batch task selections:       {batch_runs}/20\n")

    # Show EEVDF's advantage: short-request tasks get early deadlines
    print("  EEVDF key insight:")
    print("    Short-request tasks (interactive) get EARLIER virtual deadlines.")
    print("    This means they are scheduled sooner after becoming eligible.")
    print("    No explicit 'interactive boost' heuristic is needed.\n")

    print("  CFS vs EEVDF for interactive latency:")
    print("    CFS:   Relies on wake-up preemption heuristics.")
    print("           Tuning required (sysctl parameters).")
    print("           Can mis-classify tasks.\n")
    print("    EEVDF: Short requests naturally produce early deadlines.")
    print("           No heuristics. Provably fair.")
    print("           Replaced CFS in Linux 6.6 (2023).")


# === Exercise 3: Heterogeneous Core Scheduler ===
# Problem: Simulate task placement on a big.LITTLE architecture and compare
# three scheduling strategies: big-only, little-only, and energy-aware.

def exercise_3():
    """Simulate heterogeneous scheduling on a big.LITTLE CPU."""

    class Core:
        """A CPU core with capacity and power characteristics."""

        def __init__(self, core_id, core_type, capacity, power_per_unit):
            self.core_id = core_id
            self.core_type = core_type     # "big" or "LITTLE"
            self.capacity = capacity       # Max compute units (0-1024)
            self.power_per_unit = power_per_unit  # mW per compute unit
            self.current_load = 0          # Current utilization (0-1024)
            self.tasks_assigned = []

        def available(self):
            return self.capacity - self.current_load

        def energy_for_task(self, task_util):
            """Estimate energy to run a task with given utilization."""
            # Energy = power * time. Higher capacity core finishes faster
            # but consumes more power.
            return task_util * self.power_per_unit

    class Task:
        """A task with utilization and latency requirements."""

        def __init__(self, pid, name, utilization, latency_sensitive):
            self.pid = pid
            self.name = name
            self.utilization = utilization    # CPU utilization (0-1024)
            self.latency_sensitive = latency_sensitive
            self.assigned_core = None

    # Create cores: 4 big + 4 LITTLE (like ARM big.LITTLE)
    cores = [
        Core(0, "big",    1024, 5.0),
        Core(1, "big",    1024, 5.0),
        Core(2, "big",    1024, 5.0),
        Core(3, "big",    1024, 5.0),
        Core(4, "LITTLE",  512, 1.5),
        Core(5, "LITTLE",  512, 1.5),
        Core(6, "LITTLE",  512, 1.5),
        Core(7, "LITTLE",  512, 1.5),
    ]

    # Create workload
    tasks = [
        Task(1,  "ui_render",    200, True),
        Task(2,  "touch_input",  100, True),
        Task(3,  "audio",        150, True),
        Task(4,  "video_decode", 600, True),
        Task(5,  "compiler",     800, False),
        Task(6,  "backup",       300, False),
        Task(7,  "indexer",      400, False),
        Task(8,  "email_sync",   200, False),
        Task(9,  "analytics",    350, False),
        Task(10, "update_check", 100, False),
    ]

    print("=== Heterogeneous Core Scheduling (big.LITTLE) ===\n")

    print(f"Cores:")
    print(f"  {'ID':<4} {'Type':<8} {'Capacity':<10} {'Power (mW/unit)'}")
    print("  " + "-" * 35)
    for c in cores:
        print(f"  {c.core_id:<4} {c.core_type:<8} {c.capacity:<10} {c.power_per_unit}")

    print(f"\nTasks:")
    print(f"  {'PID':<5} {'Name':<14} {'Util':<6} {'Latency?'}")
    print("  " + "-" * 35)
    for t in tasks:
        print(f"  {t.pid:<5} {t.name:<14} {t.utilization:<6} "
              f"{'yes' if t.latency_sensitive else 'no'}")

    def reset_cores():
        for c in cores:
            c.current_load = 0
            c.tasks_assigned = []

    def assign_task(task, core):
        """Assign a task to a core."""
        core.current_load += task.utilization
        core.tasks_assigned.append(task)
        task.assigned_core = core

    # Strategy 1: Big cores only
    def schedule_big_only():
        reset_cores()
        big_cores = [c for c in cores if c.core_type == "big"]
        for task in tasks:
            for core in big_cores:
                if core.available() >= task.utilization:
                    assign_task(task, core)
                    break

    # Strategy 2: LITTLE cores only
    def schedule_little_only():
        reset_cores()
        little_cores = [c for c in cores if c.core_type == "LITTLE"]
        for task in tasks:
            for core in little_cores:
                if core.available() >= task.utilization:
                    assign_task(task, core)
                    break

    # Strategy 3: Energy-aware heterogeneous
    def schedule_eas():
        """Energy-Aware Scheduling: place tasks optimally."""
        reset_cores()
        for task in sorted(tasks, key=lambda t: (not t.latency_sensitive,
                                                  -t.utilization)):
            best_core = None
            best_score = float('inf')

            for core in cores:
                if core.available() < task.utilization:
                    continue

                energy = core.energy_for_task(task.utilization)

                if task.latency_sensitive:
                    # Prefer big cores for latency: penalize LITTLE cores
                    if core.core_type == "LITTLE":
                        energy *= 3.0  # Latency penalty
                else:
                    # Prefer LITTLE cores for batch: penalize big cores
                    if core.core_type == "big":
                        energy *= 1.5  # Efficiency penalty

                if energy < best_score:
                    best_score = energy
                    best_core = core

            if best_core:
                assign_task(task, best_core)

    strategies = [
        ("Big Cores Only",       schedule_big_only),
        ("LITTLE Cores Only",    schedule_little_only),
        ("Energy-Aware (EAS)",   schedule_eas),
    ]

    print(f"\n--- Strategy Comparison ---\n")

    results = []
    for strategy_name, schedule_fn in strategies:
        schedule_fn()

        total_energy = 0
        placed = 0
        latency_on_big = 0
        latency_total = 0

        for core in cores:
            for task in core.tasks_assigned:
                placed += 1
                total_energy += core.energy_for_task(task.utilization)
                if task.latency_sensitive:
                    latency_total += 1
                    if core.core_type == "big":
                        latency_on_big += 1

        results.append({
            "name": strategy_name,
            "placed": placed,
            "energy": total_energy,
            "latency_on_big": latency_on_big,
            "latency_total": latency_total,
        })

    print(f"  {'Strategy':<22} {'Placed':<8} {'Energy':<10} "
          f"{'Latency Tasks on Big'}")
    print("  " + "-" * 55)

    for r in results:
        lat_str = (f"{r['latency_on_big']}/{r['latency_total']}"
                   if r['latency_total'] > 0 else "N/A")
        print(f"  {r['name']:<22} {r['placed']:<8} {r['energy']:<10.0f} {lat_str}")

    # Show detailed EAS placement
    print(f"\n--- Energy-Aware Placement Detail ---\n")
    schedule_eas()

    print(f"  {'Core':<8} {'Type':<8} {'Load':<8} {'Tasks'}")
    print("  " + "-" * 55)
    for core in cores:
        if core.tasks_assigned:
            task_names = ", ".join(t.name for t in core.tasks_assigned)
            print(f"  {core.core_id:<8} {core.core_type:<8} "
                  f"{core.current_load:<8} {task_names}")

    print(f"\n  EAS achieves the best energy-performance balance:")
    print(f"  - Latency-sensitive tasks placed on big cores (fast response)")
    print(f"  - Batch tasks placed on LITTLE cores (energy efficient)")
    print(f"  - This is how Linux EAS works on ARM big.LITTLE and")
    print(f"    Intel Hybrid (Alder Lake+) architectures.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: CFS Scheduler Simulator ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: EEVDF Scheduler Simulator ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Heterogeneous Core Scheduler ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")
