"""
Exercises for Lesson 09: Deadlock
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers Coffman conditions, resource allocation graphs, Banker's algorithm,
prevention strategies, and detection/recovery.
"""


# === Exercise 1: Coffman Conditions Analysis ===
# Problem: Identify which Coffman conditions are present in each system.

def exercise_1():
    """Analyze Coffman conditions for three systems."""
    systems = [
        {
            "name": "System A: Database with exclusive row locks",
            "description": "Transactions lock rows exclusively, hold all locks until commit, "
                          "locks cannot be forcibly taken, two transactions waiting for each other.",
            "conditions": {
                "Mutual Exclusion": ("PRESENT", "Rows are locked exclusively (no shared reads)."),
                "Hold and Wait": ("PRESENT", "Transactions hold locks while waiting for more."),
                "No Preemption": ("PRESENT", "Locks cannot be forcibly taken away."),
                "Circular Wait": ("PRESENT", "Two transactions each hold a row the other wants."),
            },
            "deadlock": "YES -- all four conditions are present. This is a classic deadlock.",
        },
        {
            "name": "System B: Print spooler with preemption",
            "description": "Any print job can be preempted and cancelled, restarted from scratch.",
            "conditions": {
                "Mutual Exclusion": ("PRESENT", "Printer is exclusive (one job at a time)."),
                "Hold and Wait": ("Partially", "A job uses the printer but doesn't hold other resources."),
                "No Preemption": ("NOT PRESENT", "Jobs CAN be preempted and cancelled."),
                "Circular Wait": ("NOT PRESENT", "Only one resource type (printer), no cycle possible."),
            },
            "deadlock": "NO -- No Preemption is absent. Since jobs can be preempted, the OS "
                       "can always break a potential deadlock by cancelling a job.",
        },
        {
            "name": "System C: Philosopher picks up BOTH forks or waits",
            "description": "Each philosopher either picks up both forks simultaneously or waits.",
            "conditions": {
                "Mutual Exclusion": ("PRESENT", "Forks are exclusive."),
                "Hold and Wait": ("NOT PRESENT", "Philosopher gets both or neither -- never holds one while waiting."),
                "No Preemption": ("PRESENT", "Forks cannot be forcibly taken."),
                "Circular Wait": ("Partially", "Could form if all reach for same fork simultaneously."),
            },
            "deadlock": "NO -- Hold and Wait is broken. Since each philosopher atomically "
                       "acquires both forks or none, no philosopher ever holds one fork "
                       "while waiting for another. Without hold-and-wait, deadlock is impossible.",
        },
    ]

    for sys in systems:
        print(f"\n{sys['name']}")
        print(f"  {sys['description']}\n")
        print(f"  {'Condition':<25} {'Status':<15} {'Explanation'}")
        print("  " + "-" * 75)
        for cond, (status, explanation) in sys["conditions"].items():
            print(f"  {cond:<25} {status:<15} {explanation}")
        print(f"\n  Deadlock possible? {sys['deadlock']}")


# === Exercise 2: Resource Allocation Graph Analysis ===
# Problem: Analyze a resource allocation graph for cycles and deadlock.

def exercise_2():
    """Analyze resource allocation graph for deadlock."""
    print("Resource Allocation Graph:\n")
    print("  Edges:")
    print("  P1 -> R1 (request)    R1 -> P2 (held by)")
    print("  P2 -> R2 (request)    R2 -> P3 (held by)")
    print("  P3 -> R1 (request)    R3 -> P4 (held by)")
    print("  P4 -> R3 (request)")
    print()

    print("Q1: Graph visualization:")
    print("  P1 --request--> R1 --held-by--> P2 --request--> R2 --held-by--> P3")
    print("                   ^                                                |")
    print("                   |________________request_________________________|")
    print()
    print("  P4 --request--> R3 --held-by--> P4 (self-loop -- P4 requests R3 it holds)")
    print()

    print("Q2: Cycles in the graph:")
    print("  Cycle 1: P1 -> R1 -> P2 -> R2 -> P3 -> R1 (back to R1)")
    print("    Processes P1, P2, P3 form a cycle through R1 and R2.")
    print()
    print("  Cycle 2: P4 -> R3 -> P4 (self-loop)")
    print("    P4 holds R3 and requests R3 -- this is a single-process deadlock")
    print("    (P4 is waiting for a resource it already holds, which is a bug).")
    print()

    print("Q3: Deadlock?")
    print("  YES. With single-instance resources, a cycle = deadlock.")
    print("  Deadlocked processes: P1, P2, P3 (in the R1-R2 cycle)")
    print("  P4 is also deadlocked (self-loop on R3).")
    print()

    print("Q4: If P4 releases R3 and P5 gets it:")
    print("  P4's self-loop would be broken if P4 releases R3.")
    print("  However, the P1-P2-P3 cycle through R1-R2 is INDEPENDENT")
    print("  of R3 and P4. Granting R3 to P5 does NOT resolve the")
    print("  P1-P2-P3 deadlock. Those three processes remain deadlocked.")


# === Exercise 3: Banker's Algorithm ===
# Problem: Run the Banker's algorithm on a 5-process, 3-resource system.

def exercise_3():
    """Implement and run the Banker's Algorithm."""
    # Resources: A(10), B(5), C(7)
    total = [10, 5, 7]

    allocation = [
        [0, 1, 0],  # P0
        [2, 0, 0],  # P1
        [3, 0, 2],  # P2
        [2, 1, 1],  # P3
        [0, 0, 2],  # P4
    ]

    max_need = [
        [7, 5, 3],  # P0
        [3, 2, 2],  # P1
        [9, 0, 2],  # P2
        [2, 2, 2],  # P3
        [4, 3, 3],  # P4
    ]

    n = len(allocation)

    # Q1: Calculate Available
    total_alloc = [sum(allocation[i][j] for i in range(n)) for j in range(3)]
    available = [total[j] - total_alloc[j] for j in range(3)]
    print(f"Q1: Available = Total - Allocated")
    print(f"  Total:     {total}")
    print(f"  Allocated: {total_alloc}")
    print(f"  Available: {available}\n")

    # Q2: Calculate Need matrix
    need = [[max_need[i][j] - allocation[i][j] for j in range(3)] for i in range(n)]
    print(f"Q2: Need = Max - Allocation:")
    print(f"  {'Process':<10} {'Allocation':<15} {'Max Need':<15} {'Need':<15}")
    print("  " + "-" * 55)
    for i in range(n):
        print(f"  P{i:<9} {str(allocation[i]):<15} {str(max_need[i]):<15} {str(need[i]):<15}")

    # Q3: Find safe sequence
    print(f"\nQ3: Banker's Algorithm -- finding safe sequence:")
    work = available[:]
    finish = [False] * n
    safe_sequence = []

    step = 0
    while len(safe_sequence) < n:
        found = False
        for i in range(n):
            if not finish[i] and all(need[i][j] <= work[j] for j in range(3)):
                step += 1
                print(f"  Step {step}: P{i} can run (Need {need[i]} <= Work {work})")
                work = [work[j] + allocation[i][j] for j in range(3)]
                print(f"           After P{i} finishes: Work = {work}")
                finish[i] = True
                safe_sequence.append(f"P{i}")
                found = True
                break
        if not found:
            print("  No process can proceed -- UNSAFE STATE!")
            break

    if len(safe_sequence) == n:
        print(f"\n  Safe sequence: {' -> '.join(safe_sequence)}")
        print(f"  System is in a SAFE STATE.")

    # Q4: P1 requests (1, 0, 2)
    print(f"\nQ4: P1 requests (1, 0, 2). Can the system grant this?")
    request = [1, 0, 2]

    # Check: request <= need?
    can_request = all(request[j] <= need[1][j] for j in range(3))
    print(f"  Request {request} <= Need[P1] {need[1]}? {can_request}")

    # Check: request <= available?
    can_allocate = all(request[j] <= available[j] for j in range(3))
    print(f"  Request {request} <= Available {available}? {can_allocate}")

    if can_request and can_allocate:
        # Pretend to allocate
        new_avail = [available[j] - request[j] for j in range(3)]
        new_alloc = [allocation[1][j] + request[j] for j in range(3)]
        new_need = [need[1][j] - request[j] for j in range(3)]

        print(f"  Tentative allocation:")
        print(f"    New Available: {new_avail}")
        print(f"    New Allocation[P1]: {new_alloc}")
        print(f"    New Need[P1]: {new_need}")

        # Run safety check
        t_alloc = [row[:] for row in allocation]
        t_alloc[1] = new_alloc
        t_need = [row[:] for row in need]
        t_need[1] = new_need

        t_work = new_avail[:]
        t_finish = [False] * n
        t_safe = []

        while len(t_safe) < n:
            found = False
            for i in range(n):
                if not t_finish[i] and all(t_need[i][j] <= t_work[j] for j in range(3)):
                    t_work = [t_work[j] + t_alloc[i][j] for j in range(3)]
                    t_finish[i] = True
                    t_safe.append(f"P{i}")
                    found = True
                    break
            if not found:
                break

        if len(t_safe) == n:
            print(f"\n  Safety check: Safe sequence {' -> '.join(t_safe)}")
            print(f"  GRANT the request.")
        else:
            print(f"\n  Safety check: No safe sequence found. DENY the request.")


# === Exercise 4: Prevention Strategy Trade-offs ===
# Problem: Identify practical and impractical uses of prevention strategies.

def exercise_4():
    """Analyze deadlock prevention strategy trade-offs."""
    strategies = [
        {
            "strategy": "Deny mutual exclusion",
            "practical": "Read-only file access: Multiple readers can share a file "
                        "without exclusive locks. Database SELECT queries using shared locks.",
            "impractical": "Printer access: A printer physically cannot serve two jobs "
                          "simultaneously. The resource is inherently exclusive -- mutual "
                          "exclusion cannot be removed.",
        },
        {
            "strategy": "Deny hold-and-wait (request all resources at once)",
            "practical": "Batch job systems: A batch job can declare all needed resources "
                        "upfront (memory, disk, CPU time) before starting. Simple and safe.",
            "impractical": "Interactive database transactions: A transaction doesn't know "
                          "which rows it will need until it processes user input. Locking "
                          "all possible rows upfront would be extremely wasteful.",
        },
        {
            "strategy": "Allow preemption (forcibly take resources)",
            "practical": "CPU scheduling: The OS preempts CPU from processes routinely. "
                        "Memory can be preempted via swapping. The resource state can "
                        "be saved and restored.",
            "impractical": "Printer mid-job: Cannot preempt a half-printed document "
                          "meaningfully. The printed pages are wasted. Some resources "
                          "cannot have their state saved and restored.",
        },
        {
            "strategy": "Impose resource ordering (always acquire in fixed order)",
            "practical": "Lock ordering in multithreaded programs: Developers define a "
                        "global lock ordering (e.g., account locks ordered by account number). "
                        "Linux kernel uses this extensively for nested lock acquisition.",
            "impractical": "Dynamic resource discovery: In a distributed system where "
                          "resources are discovered at runtime (e.g., peer-to-peer nodes), "
                          "it may be impossible to define a global ordering. The set of "
                          "resources changes continuously.",
        },
    ]

    print("Deadlock Prevention Strategy Trade-offs:\n")
    for s in strategies:
        print(f"Strategy: {s['strategy']}")
        print(f"  Practical:   {s['practical']}")
        print(f"  Impractical: {s['impractical']}")
        print()


# === Exercise 5: Detection and Recovery ===
# Problem: Analyze detection frequency and recovery strategies.

def exercise_5():
    """Analyze deadlock detection and recovery strategies."""
    print("Deadlock Detection at t=60min:")
    print("  Deadlocked: P2, P5, P7, P8 holding R1, R2, R3, R4\n")

    print("Q1: Minimum processes to terminate:")
    print("  With 4 processes in a cycle, terminating ANY ONE process breaks")
    print("  the cycle (releases its resource, allowing the chain to continue).")
    print("  Minimum: 1 process.")
    print("  Which one? Since all have equal priority, terminate the one that")
    print("  has done the least work (to minimize wasted computation) or the")
    print("  one holding the most resources (to free the most).\n")

    print("Q2: Resource preemption order:")
    print("  Preempt resources in order of:")
    print("  1. Least work done by the process (minimize wasted computation)")
    print("  2. Most resources held (freeing one process frees many resources)")
    print("  3. Longest remaining execution time (least progress lost)")
    print("  4. Lowest priority process first")
    print("  Roll back the preempted process to a safe checkpoint if available,")
    print("  otherwise restart it from the beginning.\n")

    print("Q3: P2 and P5 immediately deadlock again:")
    print("  This suggests a DESIGN BUG, not a transient issue. The code has")
    print("  a fundamental ordering problem where P2 and P5 always acquire")
    print("  resources in conflicting orders. Solutions:")
    print("  - Fix the code to impose resource ordering (prevention)")
    print("  - Use Banker's algorithm for these processes (avoidance)")
    print("  - Redesign to eliminate the circular dependency")
    print("  Simply detecting and recovering won't help -- the deadlock")
    print("  will recur indefinitely.\n")

    print("Q4: Detection frequency trade-offs (period T):")
    print("  Small T (e.g., 1 second):")
    print("    + Deadlocks detected quickly, minimal wasted time")
    print("    + Less work lost when terminating processes")
    print("    - High overhead: detection algorithm runs frequently")
    print("    - O(n^2) cycle detection on every invocation")
    print()
    print("  Large T (e.g., 30 minutes):")
    print("    + Low overhead: detection runs rarely")
    print("    - Deadlocks persist for up to T minutes before detection")
    print("    - More work wasted (processes may have done significant work")
    print("      since the deadlock started)")
    print("    - Other processes may be blocked waiting for deadlocked resources")
    print()
    print("  Practical approach: trigger detection on symptoms (e.g., CPU")
    print("  utilization drops below threshold) rather than fixed intervals.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Coffman Conditions Analysis ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Resource Allocation Graph Analysis ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Banker's Algorithm ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Prevention Strategy Trade-offs ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Detection and Recovery ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
