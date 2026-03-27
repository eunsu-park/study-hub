"""
Exercises for Lesson 06: Advanced Scheduling
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers MLFQ, processor affinity, RMS feasibility, and CFS virtual runtime.
"""

import math


# === Exercise 1: MLFQ Process Placement ===
# Problem: Trace job queue levels in a 3-level MLFQ.

def exercise_1():
    """Trace MLFQ queue levels for different job types."""
    print("MLFQ Configuration:")
    print("  Q0 (highest): TQ=4ms, new processes start here")
    print("  Q1 (medium):  TQ=8ms")
    print("  Q2 (lowest):  FCFS")
    print("  Priority boost every 50ms\n")

    print("Q1: Job A (CPU-intensive, never yields) queue level trace:")
    print("  t=0:  Starts in Q0, runs 4ms, uses full quantum -> demoted to Q1")
    print("  t=5:  In Q1, runs 8ms, uses full quantum -> demoted to Q2")
    print("  t=15: In Q2 (FCFS), runs whenever scheduled")
    print("  t=30: Still in Q2")
    print("  t=50: PRIORITY BOOST! All processes move to Q0")
    print("  t=54: Uses Q0's 4ms quantum again -> demoted to Q1")
    print("  t=60: In Q1 (demoted again after using 8ms quantum)")
    print()
    print("  Summary: t=5ms -> Q1, t=15ms -> Q2, t=30ms -> Q2, t=60ms -> Q1")
    print()

    print("Q2: Job B (I/O-intensive, I/O every 2ms) queue level trace:")
    print("  t=0:  Starts in Q0, runs 2ms, does I/O (yields BEFORE quantum expires)")
    print("  t=5:  Returns from I/O, stays in Q0 (didn't use full quantum)")
    print("  t=15: Still in Q0 (keeps yielding before 4ms quantum)")
    print("  t=30: Still in Q0")
    print()
    print("  Key insight: I/O-bound processes that voluntarily yield before the")
    print("  quantum expires stay in high-priority queues. This is MLFQ's main")
    print("  advantage: it automatically gives interactive/I/O processes priority.")
    print()

    print("Q3: Job C arrives at t=20ms:")
    print("  Starts in Q0 (all new processes start at highest priority)")
    print("  With 6ms bursts, it uses the full Q0 quantum (4ms), gets demoted")
    print("  to Q1, then uses 2ms of Q1's 8ms quantum and yields for I/O")
    print("  Effect on A and B: C competes with B in Q0. A (in Q2) gets even")
    print("  less CPU time because Q0 and Q1 are served before Q2.")
    print()

    print("Q4: Without priority boost, Job A starves!")
    print("  Once demoted to Q2, if B and C keep the CPU busy in Q0/Q1,")
    print("  Q2 never gets served -> A starves indefinitely.")
    print("  The boost every 50ms forces A back to Q0, guaranteeing it")
    print("  gets at least 4ms of CPU time every ~50ms. This prevents")
    print("  starvation while still prioritizing interactive processes.")


# === Exercise 2: Processor Affinity and Load Balancing ===
# Problem: Analyze NUMA-aware scheduling decisions.

def exercise_2():
    """Analyze processor affinity and NUMA scheduling."""
    print("NUMA system: Node 0 (cores 0,1), Node 1 (cores 2,3)")
    print("Intra-node memory: 10ns, Cross-node: 40ns\n")

    print("Q1: Hard affinity rule:")
    print("  The scheduler MUST only schedule a process on the set of CPUs")
    print("  specified in its CPU affinity mask (set via sched_setaffinity).")
    print("  If process X has affinity mask {0,1}, it can ONLY run on cores 0,1.")
    print("  The scheduler cannot override this, even if cores 2,3 are idle.")
    print()

    print("Q2: Soft affinity -- which process for Core 0?")
    print("  Prefer X (needs Node 0 memory)")
    print("  Reasoning: Core 0 is on Node 0. If X runs on Core 0, its memory")
    print("  accesses go to local Node 0 memory (10ns). If Y ran on Core 0,")
    print("  its Node 1 memory accesses would cost 40ns (4x slower).")
    print("  Z has no preference, so X is the best choice.")
    print()

    print("Q3: Pull migration of Y from Core 2 to Core 0:")
    print("  Core 0 (idle, Node 0) could pull work from Core 2 (busy, Node 1)")
    print("  Cost of migrating Y:")
    print("  - Y's cached data on Core 2 is invalidated (cache cold start on Core 0)")
    print("  - Y's memory is on Node 1: every access from Core 0 costs 40ns (vs 10ns)")
    print("  - TLB entries for Y are lost, must be rebuilt on Core 0")
    print("  - Migration cost: ~50-200us (cache refill + TLB rebuild)")
    print("  This is a BAD migration -- Y should stay on Node 1.")
    print()

    print("Q4: When NOT to balance load:")
    print("  - When processes have strong NUMA locality (moving them to another")
    print("    node would quadruple memory access latency)")
    print("  - When cache-hot processes would lose all cached data by migrating")
    print("  - When the imbalance is temporary (e.g., a process about to finish)")
    print("  - In real-time systems where predictable latency matters more")
    print("    than perfect load balance")


# === Exercise 3: Rate Monotonic Scheduling Feasibility ===
# Problem: Check RMS schedulability using utilization bound.

def exercise_3():
    """Check RMS schedulability for two task sets."""

    def rms_bound(n):
        """Calculate RMS utilization bound: n(2^(1/n) - 1)."""
        return n * (2 ** (1.0 / n) - 1)

    # Task Set 1
    print("=== Task Set 1 ===")
    tasks1 = [
        ("T1", 20, 5),   # (name, period, execution)
        ("T2", 50, 10),
        ("T3", 100, 20),
    ]

    total_u1 = 0
    print(f"  {'Task':<6} {'Period':<10} {'Exec':<8} {'Utilization':<12}")
    print("  " + "-" * 36)
    for name, period, exec_time in tasks1:
        u = exec_time / period
        total_u1 += u
        print(f"  {name:<6} {period:<10} {exec_time:<8} {u:.4f}")

    n1 = len(tasks1)
    bound1 = rms_bound(n1)
    print(f"\n  Total utilization: {total_u1:.4f}")
    print(f"  RMS bound for n={n1}: {n1}(2^(1/{n1}) - 1) = {bound1:.4f}")
    print(f"  {total_u1:.4f} {'<=' if total_u1 <= bound1 else '>'} {bound1:.4f}")
    if total_u1 <= bound1:
        print(f"  Result: GUARANTEED SCHEDULABLE under RMS")
    elif total_u1 <= 1.0:
        print(f"  Result: POSSIBLY schedulable (above RMS bound but U <= 1.0)")
    else:
        print(f"  Result: DEFINITELY NOT SCHEDULABLE (U > 1.0)")

    # Task Set 2
    print(f"\n=== Task Set 2 ===")
    tasks2 = [
        ("T1", 10, 4),
        ("T2", 25, 8),
        ("T3", 50, 12),
    ]

    total_u2 = 0
    print(f"  {'Task':<6} {'Period':<10} {'Exec':<8} {'Utilization':<12}")
    print("  " + "-" * 36)
    for name, period, exec_time in tasks2:
        u = exec_time / period
        total_u2 += u
        print(f"  {name:<6} {period:<10} {exec_time:<8} {u:.4f}")

    n2 = len(tasks2)
    bound2 = rms_bound(n2)
    print(f"\n  Total utilization: {total_u2:.4f}")
    print(f"  RMS bound for n={n2}: {bound2:.4f}")
    print(f"  {total_u2:.4f} {'<=' if total_u2 <= bound2 else '>'} {bound2:.4f}")

    if total_u2 <= bound2:
        print(f"  Result: GUARANTEED SCHEDULABLE under RMS")
    elif total_u2 <= 1.0:
        print(f"  Result: POSSIBLY schedulable (above bound but U <= 1.0)")
        print(f"  Need exact analysis (response time calculation) to confirm.")
    else:
        print(f"  Result: DEFINITELY NOT SCHEDULABLE (U > 1.0)")

    print(f"\n  Can EDF schedule Task Set 2?")
    print(f"  EDF schedulability: U <= 1.0 is both necessary and sufficient.")
    print(f"  Total U = {total_u2:.4f} {'<= 1.0' if total_u2 <= 1.0 else '> 1.0'}")
    if total_u2 <= 1.0:
        print(f"  YES -- EDF can schedule it (EDF bound is 100% vs RMS ~{bound2:.1%})")
    print(f"\n  EDF vs RMS under overload:")
    print(f"  RMS: higher-priority (shorter-period) tasks always meet deadlines;")
    print(f"       lower-priority tasks miss deadlines predictably.")
    print(f"  EDF: under overload, ANY task can miss its deadline unpredictably")
    print(f"       (domino effect). This makes RMS preferred for safety-critical systems.")


# === Exercise 4: CFS Virtual Runtime ===
# Problem: Calculate virtual runtime for processes with different weights.

def exercise_4():
    """Analyze CFS virtual runtime calculation."""
    # Nice values and weights
    processes = [
        {"name": "P_nice-10", "nice": -10, "weight": 9548},
        {"name": "P_nice0",   "nice": 0,   "weight": 1024},
        {"name": "P_nice10",  "nice": 10,  "weight": 110},
    ]
    total_weight = sum(p["weight"] for p in processes)
    period = 10  # ms scheduling period

    print("CFS Virtual Runtime Calculation")
    print(f"\nTotal weight: {total_weight}")
    print(f"Scheduling period: {period} ms\n")

    # Q1: Ideal CPU time in first 10ms
    print("Q1: Ideal CPU time per process in first 10ms:")
    for p in processes:
        cpu_share = (p["weight"] / total_weight) * period
        p["cpu_time"] = cpu_share
        print(f"  {p['name']}: ({p['weight']}/{total_weight}) * {period}ms = {cpu_share:.3f} ms")

    # Q2: Virtual runtime after 10ms
    print(f"\nQ2: Virtual runtime after 10ms:")
    print(f"  vruntime = actual_runtime * (NICE_0_WEIGHT / process_weight)")
    print(f"  NICE_0_WEIGHT = 1024\n")
    for p in processes:
        vruntime = p["cpu_time"] * (1024.0 / p["weight"])
        p["vruntime"] = vruntime
        print(f"  {p['name']}: {p['cpu_time']:.3f}ms * (1024/{p['weight']}) = {vruntime:.3f} ms")

    print(f"\n  Key insight: All three processes have approximately EQUAL vruntime!")
    print(f"  This is by design -- CFS aims to equalize vruntime across all processes.")
    print(f"  Higher-weight processes get more actual CPU time but their vruntime")
    print(f"  advances more slowly (the weight division compensates).")

    # Q3: I/O return and CFS compensation
    print(f"\nQ3: If nice=10 process does I/O for 5ms then returns:")
    print(f"  During the 5ms of I/O, nice=10's vruntime does NOT advance")
    print(f"  (sleeping processes accumulate no vruntime).")
    print(f"  When it wakes up, its vruntime is lower than the other processes'.")
    print(f"  CFS prevents 'catch-up burst' by setting the waking process's")
    print(f"  vruntime to: max(current_vruntime, min_vruntime - sched_latency/2)")
    print(f"  This gives the waking process a slight boost (scheduled sooner)")
    print(f"  but NOT an unlimited burst. The sched_latency/2 cap limits how")
    print(f"  far ahead the process can get, preventing it from monopolizing CPU.")

    # Q4: Why vruntime instead of fixed time slices
    print(f"\nQ4: Why virtual runtime instead of fixed time slices?")
    print(f"  Fixed slices: A process with nice -10 gets 9x the slice of nice +10.")
    print(f"  Problem: If only 2 processes exist, fixed slices waste CPU (idle gaps).")
    print(f"  With vruntime:")
    print(f"  - The scheduler always picks the lowest-vruntime process")
    print(f"  - CPU is never idle while runnable processes exist")
    print(f"  - Adding/removing processes automatically adjusts shares")
    print(f"  - No need to recalculate time slices when process count changes")
    print(f"  - Works naturally with any number of processes and any mix of weights")
    print(f"  - O(log n) pick-next using a red-black tree sorted by vruntime")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: MLFQ Process Placement ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Processor Affinity and Load Balancing ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Rate Monotonic Scheduling Feasibility ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: CFS Virtual Runtime ===")
    print("=" * 70)
    exercise_4()

    print("\nAll exercises completed!")
