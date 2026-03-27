"""
Exercises for Lesson 22: Real-Time Operating Systems
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers Rate Monotonic schedulability analysis with response time analysis,
Earliest Deadline First simulation, and priority inversion demonstration.
"""

import math


# === Exercise 1: Rate Monotonic Schedulability Analysis ===
# Problem: Given a set of periodic tasks, determine schedulability using
# both the utilization bound test and exact response time analysis.

def exercise_1():
    """Perform RMS schedulability analysis with utilization and RTA tests."""
    # Task set: (name, period, execution_time)
    # Tasks are sorted by period (shorter period = higher priority in RMS)
    task_sets = [
        {
            "name": "Task Set A (schedulable)",
            "tasks": [
                ("T1", 100, 20),
                ("T2", 150, 30),
                ("T3", 350, 80),
            ],
        },
        {
            "name": "Task Set B (high utilization)",
            "tasks": [
                ("T1", 50, 12),
                ("T2", 100, 20),
                ("T3", 200, 50),
            ],
        },
        {
            "name": "Task Set C (utilization test inconclusive, RTA needed)",
            "tasks": [
                ("T1", 80, 32),
                ("T2", 100, 24),
                ("T3", 200, 44),
            ],
        },
    ]

    def utilization_bound(n):
        """Liu & Layland bound: n * (2^(1/n) - 1)."""
        return n * (2.0 ** (1.0 / n) - 1.0)

    def response_time_analysis(tasks, task_idx):
        """Compute worst-case response time for task at task_idx.

        R_i = C_i + sum over higher priority tasks j of ceil(R_i / T_j) * C_j
        Iterate until convergence.
        """
        ci = tasks[task_idx][2]
        ti = tasks[task_idx][1]
        r = ci

        for _ in range(1000):
            r_new = ci
            for j in range(task_idx):
                tj = tasks[j][1]
                cj = tasks[j][2]
                r_new += math.ceil(r / tj) * cj

            if r_new == r:
                return r
            if r_new > ti:
                return r_new  # Deadline miss
            r = r_new

        return r

    for ts in task_sets:
        print(f"--- {ts['name']} ---\n")
        tasks = ts["tasks"]
        n = len(tasks)

        # Sort by period (RMS priority assignment)
        tasks.sort(key=lambda t: t[1])

        print(f"  {'Task':<6} {'Period (T)':<12} {'WCET (C)':<10} {'Util (C/T)'}")
        print("  " + "-" * 40)

        total_util = 0.0
        for name, period, wcet in tasks:
            util = wcet / period
            total_util += util
            print(f"  {name:<6} {period:<12} {wcet:<10} {util:.4f}")

        bound = utilization_bound(n)
        print(f"\n  Total utilization: {total_util:.4f}")
        print(f"  RM bound (n={n}): {bound:.4f}")

        if total_util <= bound:
            print(f"  Utilization test: SCHEDULABLE (U <= bound)")
        elif total_util <= 1.0:
            print(f"  Utilization test: INCONCLUSIVE (bound < U <= 1.0)")
            print(f"  Need Response Time Analysis for definitive answer.")
        else:
            print(f"  Utilization test: NOT SCHEDULABLE (U > 1.0)")

        # Response Time Analysis (exact test)
        print(f"\n  Response Time Analysis:")
        all_schedulable = True
        for i, (name, period, wcet) in enumerate(tasks):
            wcrt = response_time_analysis(tasks, i)
            status = "OK" if wcrt <= period else "MISS"
            if wcrt > period:
                all_schedulable = False
            print(f"    {name}: WCRT = {wcrt}, Deadline = {period} -> {status}")

        print(f"  RTA result: {'SCHEDULABLE' if all_schedulable else 'NOT SCHEDULABLE'}")
        print()

    # Comparison of bounds
    print("--- RM Utilization Bound vs EDF ---\n")
    print(f"  {'n tasks':<10} {'RM Bound':<12} {'EDF Bound'}")
    print("  " + "-" * 30)
    for n in range(1, 11):
        rm = utilization_bound(n)
        print(f"  {n:<10} {rm:.4f}       1.0000")
    print(f"\n  As n -> infinity, RM bound -> ln(2) = {math.log(2):.4f}")
    print(f"  EDF always allows up to 100% utilization.")
    print(f"  EDF is optimal but harder to implement (dynamic priorities).")


# === Exercise 2: EDF Schedule Simulation ===
# Problem: Simulate Earliest Deadline First scheduling for one hyperperiod
# and show the execution timeline.

def exercise_2():
    """Simulate EDF scheduling and generate a text-based Gantt chart."""
    # Tasks: (name, period, execution_time)
    tasks = [
        ("A", 6, 2),
        ("B", 8, 2),
        ("C", 12, 3),
    ]

    print("Earliest Deadline First (EDF) Simulation\n")
    print(f"  {'Task':<6} {'Period':<8} {'WCET':<8} {'Utilization'}")
    print("  " + "-" * 30)
    total_util = 0.0
    for name, period, wcet in tasks:
        util = wcet / period
        total_util += util
        print(f"  {name:<6} {period:<8} {wcet:<8} {util:.4f}")
    print(f"  Total utilization: {total_util:.4f}")
    print(f"  EDF schedulable: {'Yes' if total_util <= 1.0 else 'No'}\n")

    # Calculate hyperperiod (LCM of all periods)
    def lcm(a, b):
        return a * b // math.gcd(a, b)

    hyperperiod = tasks[0][1]
    for _, p, _ in tasks[1:]:
        hyperperiod = lcm(hyperperiod, p)

    print(f"  Hyperperiod: {hyperperiod} time units\n")

    # Simulate EDF for one hyperperiod
    # Job = (task_name, release_time, deadline, remaining_exec)
    timeline = [None] * hyperperiod  # what runs at each time unit
    jobs = []
    completed = []

    for t in range(hyperperiod):
        # Release new jobs at their period boundaries
        for name, period, wcet in tasks:
            if t % period == 0:
                deadline = t + period
                jobs.append({
                    "name": name,
                    "release": t,
                    "deadline": deadline,
                    "remaining": wcet,
                })

        # Pick job with earliest deadline (EDF policy)
        ready = [j for j in jobs if j["remaining"] > 0]
        if ready:
            ready.sort(key=lambda j: j["deadline"])
            chosen = ready[0]
            timeline[t] = chosen["name"]
            chosen["remaining"] -= 1
            if chosen["remaining"] == 0:
                completed.append({
                    "name": chosen["name"],
                    "release": chosen["release"],
                    "deadline": chosen["deadline"],
                    "finish": t + 1,
                })
                jobs.remove(chosen)

    # Print Gantt chart
    print("  EDF Schedule (Gantt chart):")
    print(f"  Time:  ", end="")
    for t in range(hyperperiod):
        print(f"{t:>3}", end="")
    print()

    for name, _, _ in tasks:
        print(f"  {name}:     ", end="")
        for t in range(hyperperiod):
            if timeline[t] == name:
                print(f"{'[X]':>3}", end="")
            else:
                print(f"{'  .':>3}", end="")
        print()

    # Check for deadline misses
    print(f"\n  Completed jobs:")
    print(f"  {'Task':<6} {'Release':<10} {'Deadline':<10} {'Finish':<10} {'Status'}")
    print("  " + "-" * 45)
    all_met = True
    for job in completed:
        met = job["finish"] <= job["deadline"]
        if not met:
            all_met = False
        print(f"  {job['name']:<6} {job['release']:<10} {job['deadline']:<10} "
              f"{job['finish']:<10} {'OK' if met else 'MISS'}")

    remaining_jobs = [j for j in jobs if j["remaining"] > 0]
    if remaining_jobs:
        for j in remaining_jobs:
            print(f"  {j['name']:<6} {j['release']:<10} {j['deadline']:<10} "
                  f"{'---':<10} INCOMPLETE")
            all_met = False

    print(f"\n  All deadlines met: {'Yes' if all_met else 'No'}")
    print(f"  CPU utilization: {sum(1 for t in timeline if t is not None) / hyperperiod * 100:.1f}%")


# === Exercise 3: Priority Inversion Demonstration ===
# Problem: Simulate the Mars Pathfinder priority inversion scenario
# and show how priority inheritance fixes it.

def exercise_3():
    """Demonstrate priority inversion and priority inheritance."""
    # Three tasks with different priorities
    # High (H): needs mutex, period=10
    # Medium (M): CPU-bound, no mutex, period=15
    # Low (L): holds mutex, period=20

    sim_time = 30

    print("Priority Inversion Demonstration\n")
    print("  Scenario (Mars Pathfinder-inspired):")
    print("  Task H (High priority):   Needs mutex at t=4, runs for 2 units")
    print("  Task M (Medium priority): CPU-bound, runs at t=3 for 5 units")
    print("  Task L (Low priority):    Acquires mutex at t=0, holds for 8 units\n")

    # --- Without Priority Inheritance ---
    print("--- Without Priority Inheritance ---\n")

    timeline_no_pi = []
    mutex_holder = None
    h_blocked = False
    h_done = False
    m_done = False
    l_remaining = 8
    m_remaining = 5
    h_remaining = 2
    l_started = False
    mutex_released = False

    for t in range(sim_time):
        if t == 0:
            mutex_holder = "L"
            l_started = True

        if h_done and m_done and l_remaining <= 0:
            break

        running = None
        event = ""

        # Priority: H > M > L
        # H needs mutex at t=4
        if t >= 4 and not h_done:
            if mutex_holder is None and h_remaining > 0:
                mutex_holder = "H"
                running = "H"
                h_remaining -= 1
                if h_remaining == 0:
                    h_done = True
                    mutex_holder = None
                    event = "H finishes (mutex released)"
                else:
                    event = "H runs with mutex"
            elif mutex_holder == "H":
                running = "H"
                h_remaining -= 1
                if h_remaining == 0:
                    h_done = True
                    mutex_holder = None
                    event = "H finishes (mutex released)"
            elif mutex_holder == "L":
                h_blocked = True
                # H is blocked, M runs (priority inversion!)
                if t >= 3 and m_remaining > 0:
                    running = "M"
                    m_remaining -= 1
                    if m_remaining == 0:
                        m_done = True
                        event = "M finishes (H still blocked!)"
                    else:
                        event = "M runs (priority inversion!)"
                elif l_remaining > 0:
                    running = "L"
                    l_remaining -= 1
                    if l_remaining == 0:
                        mutex_holder = None
                        mutex_released = True
                        event = "L releases mutex"
                    else:
                        event = "L runs (H blocked, M done)"
        elif t >= 3 and not m_done and m_remaining > 0:
            if not (t >= 4 and not h_done and mutex_holder != "L"):
                running = "M"
                m_remaining -= 1
                if m_remaining == 0:
                    m_done = True
                    event = "M finishes"
                else:
                    event = "M runs"
            else:
                running = "L"
                if l_remaining > 0:
                    l_remaining -= 1
                event = "L runs"
        elif l_started and l_remaining > 0:
            running = "L"
            l_remaining -= 1
            if l_remaining == 0:
                mutex_holder = None
                event = "L releases mutex"
            else:
                event = "L runs (holds mutex)"
        else:
            event = "idle"

        if running:
            timeline_no_pi.append((t, running, event))

    print(f"  {'Time':<6} {'Running':<10} {'Event'}")
    print("  " + "-" * 50)
    for t, running, event in timeline_no_pi:
        marker = " ***" if "inversion" in event.lower() else ""
        print(f"  t={t:<3} {running:<10} {event}{marker}")

    h_finish_no_pi = None
    for t, running, event in timeline_no_pi:
        if "H finishes" in event:
            h_finish_no_pi = t + 1
            break

    # --- With Priority Inheritance ---
    print(f"\n--- With Priority Inheritance ---\n")

    timeline_pi = []
    mutex_holder = None
    h_blocked = False
    h_done = False
    m_done = False
    l_remaining_pi = 8
    m_remaining_pi = 5
    h_remaining_pi = 2
    l_boosted = False

    for t in range(sim_time):
        if t == 0:
            mutex_holder = "L"

        if h_done and m_done and l_remaining_pi <= 0:
            break

        running = None
        event = ""

        if t >= 4 and not h_done:
            if mutex_holder is None and h_remaining_pi > 0:
                mutex_holder = "H"
                running = "H"
                h_remaining_pi -= 1
                if h_remaining_pi == 0:
                    h_done = True
                    mutex_holder = None
                    event = "H finishes (mutex released)"
                else:
                    event = "H runs with mutex"
            elif mutex_holder == "H":
                running = "H"
                h_remaining_pi -= 1
                if h_remaining_pi == 0:
                    h_done = True
                    mutex_holder = None
                    event = "H finishes (mutex released)"
            elif mutex_holder == "L":
                # Priority inheritance: L gets H's priority
                l_boosted = True
                running = "L"
                l_remaining_pi -= 1
                if l_remaining_pi == 0:
                    mutex_holder = None
                    l_boosted = False
                    event = "L releases mutex (boosted to H priority)"
                else:
                    event = "L runs at H priority (inheritance)"
        elif not h_done and t < 4:
            # Before H needs mutex
            if t >= 3 and m_remaining_pi > 0 and not l_boosted:
                running = "M"
                m_remaining_pi -= 1
                if m_remaining_pi == 0:
                    m_done = True
                    event = "M finishes"
                else:
                    event = "M runs"
            elif l_remaining_pi > 0:
                running = "L"
                l_remaining_pi -= 1
                if l_remaining_pi == 0:
                    mutex_holder = None
                    event = "L releases mutex"
                else:
                    event = "L runs (holds mutex)"
        elif h_done:
            # H is done, M can run
            if m_remaining_pi > 0:
                running = "M"
                m_remaining_pi -= 1
                if m_remaining_pi == 0:
                    m_done = True
                    event = "M finishes"
                else:
                    event = "M runs"
            elif l_remaining_pi > 0:
                running = "L"
                l_remaining_pi -= 1
                event = "L runs"

        if running:
            timeline_pi.append((t, running, event))

    print(f"  {'Time':<6} {'Running':<10} {'Event'}")
    print("  " + "-" * 50)
    for t, running, event in timeline_pi:
        marker = " ***" if "inheritance" in event.lower() else ""
        print(f"  t={t:<3} {running:<10} {event}{marker}")

    h_finish_pi = None
    for t, running, event in timeline_pi:
        if "H finishes" in event:
            h_finish_pi = t + 1
            break

    print(f"\n--- Comparison ---\n")
    if h_finish_no_pi:
        print(f"  H finish time without PI: t={h_finish_no_pi}")
    else:
        print(f"  H finish time without PI: did not complete in simulation")
    if h_finish_pi:
        print(f"  H finish time with PI:    t={h_finish_pi}")
    else:
        print(f"  H finish time with PI:    did not complete in simulation")

    print(f"\n  Priority Inheritance prevents unbounded blocking:")
    print(f"  - Without PI: M preempts L, causing H to wait for M+L")
    print(f"  - With PI: L inherits H's priority, runs immediately to release mutex")
    print(f"  - H's blocking is bounded by L's remaining critical section time")
    print(f"\n  Priority Ceiling Protocol (alternative):")
    print(f"  - Mutex has a ceiling = max priority of any potential user")
    print(f"  - Task acquiring mutex is immediately boosted to ceiling")
    print(f"  - Prevents deadlock in addition to priority inversion")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Rate Monotonic Schedulability Analysis ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: EDF Schedule Simulation ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Priority Inversion Demonstration ===")
    print("=" * 70)
    exercise_3()

    print("\nAll exercises completed!")
