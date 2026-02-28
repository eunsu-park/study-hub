"""
Exercises for Lesson 07: Synchronization Basics
Topic: OS_Theory

Solutions to practice problems from the lesson.
Covers race conditions, critical section requirements, Peterson's algorithm,
TAS/CAS semantics, and busy waiting vs blocking.
"""

import threading
import time


# === Exercise 1: Race Condition Identification ===
# Problem: Analyze bank account transfer code for race conditions.

def exercise_1():
    """Identify race conditions in concurrent bank transfer code."""
    print("Bank account transfer code analysis:\n")
    print("```c")
    print("int account_A = 1000; int account_B = 500;")
    print("void transfer(int *from, int *to, int amount) {")
    print("    if (*from >= amount) { *from -= amount; *to += amount; }")
    print("}")
    print("thread1: transfer(&A, &B, 200);")
    print("thread2: transfer(&A, &B, 900);")
    print("```\n")

    print("Q1: Critical section(s):")
    print("  The entire body of transfer() is a critical section because")
    print("  the check-then-act pattern (if balance >= amount then deduct)")
    print("  must be atomic. Steps 1-3 (check, deduct, credit) must execute")
    print("  without interruption from another transfer to the same accounts.\n")

    print("Q2: Interleaving causing negative balance:")
    print("  Initial: A=1000, B=500")
    print("  t1: thread1 checks A >= 200   -> TRUE (A=1000)")
    print("  t2: thread2 checks A >= 900   -> TRUE (A=1000, not yet deducted!)")
    print("  t3: thread1 executes A -= 200  -> A=800")
    print("  t4: thread2 executes A -= 900  -> A=-100  (NEGATIVE!)")
    print("  t5: thread1 executes B += 200  -> B=700")
    print("  t6: thread2 executes B += 900  -> B=1600")
    print("  Final: A=-100, B=1600. Account A is negative!\n")

    print("Q3: Can a race condition change total money?")
    print("  Initial total: 1000 + 500 = 1500")
    print("  In a correct execution, total stays 1500 (conservation of money).")
    print("  With the race condition above: -100 + 1600 = 1500 (total preserved!)")
    print("  However, with non-atomic += and -=, individual operations can lose")
    print("  updates (read-modify-write race), which COULD change the total.")
    print("  Example: both threads read B=500, add their amounts, write back.")
    print("  thread1 writes B=700, thread2 overwrites B=1400. Lost 200!\n")

    print("Q4: Fix with mutex:")
    print("```c")
    print("pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;")
    print("void transfer(int *from, int *to, int amount) {")
    print("    pthread_mutex_lock(&lock);")
    print("    if (*from >= amount) {")
    print("        *from -= amount;")
    print("        *to   += amount;")
    print("    }")
    print("    pthread_mutex_unlock(&lock);")
    print("}")
    print("```")

    # Python demonstration
    print("\n--- Python demonstration ---\n")
    account_a = [1000]
    account_b = [500]
    lock = threading.Lock()

    def unsafe_transfer(from_acc, to_acc, amount, n=10000):
        for _ in range(n):
            if from_acc[0] >= amount:
                from_acc[0] -= amount
                to_acc[0] += amount

    def safe_transfer(from_acc, to_acc, amount, n=10000):
        for _ in range(n):
            with lock:
                if from_acc[0] >= amount:
                    from_acc[0] -= amount
                    to_acc[0] += amount

    # Safe version
    account_a[0], account_b[0] = 1000, 500
    t1 = threading.Thread(target=safe_transfer, args=(account_a, account_b, 1))
    t2 = threading.Thread(target=safe_transfer, args=(account_b, account_a, 1))
    t1.start(); t2.start(); t1.join(); t2.join()
    print(f"Safe: A={account_a[0]}, B={account_b[0]}, Total={account_a[0]+account_b[0]}")


# === Exercise 2: Critical Section Requirements Analysis ===
# Problem: Analyze two proposed solutions for mutual exclusion requirements.

def exercise_2():
    """Analyze critical section solutions for requirement violations."""
    print("=== Solution A: Simple turn variable ===\n")
    print("```c")
    print("int turn = 0;")
    print("// Process Pi: while (turn != i); CS; turn = 1-i;")
    print("```\n")

    print("Q1: Mutual exclusion? YES")
    print("  Only one process can have turn==i at any time.")
    print("  If P0 is in CS, turn=0, so P1's while(turn!=1) spins.\n")

    print("Q2: Progress? NO -- VIOLATED")
    print("  Counterexample: P0 finishes CS, sets turn=1. P0 wants to")
    print("  re-enter CS but must wait for P1 to take its turn first.")
    print("  If P1 is not interested in the CS, P0 is blocked forever.")
    print("  Progress requires that only processes trying to enter CS")
    print("  participate in the decision -- P1's disinterest shouldn't block P0.\n")

    print("Q3: Bounded waiting? YES (trivially)")
    print("  Strict alternation guarantees each process waits at most")
    print("  one turn. But this is a hollow victory since progress fails.\n")

    print("=" * 40)
    print("\n=== Solution B: Flag only, no turn ===\n")
    print("```c")
    print("bool flag[2] = {false, false};")
    print("// Process Pi: flag[i]=true; while(flag[1-i]); CS; flag[i]=false;")
    print("```\n")

    print("Q1: Mutual exclusion? YES")
    print("  If P0 is in CS, flag[0]=true. P1 sets flag[1]=true,")
    print("  then spins on while(flag[0]) -- cannot enter CS.\n")

    print("Q2: Progress? NO -- VIOLATED (deadlock possible)")
    print("  Interleaving: P0 sets flag[0]=true, P1 sets flag[1]=true.")
    print("  Now P0 spins on flag[1]==true, P1 spins on flag[0]==true.")
    print("  Neither can proceed. Both are trying to enter CS but the")
    print("  decision never resolves -- progress is violated.\n")

    print("Q3: Bounded waiting? NO")
    print("  In the deadlock scenario, both processes wait infinitely.")
    print("  Even without deadlock, there's no bound on how many times")
    print("  one process can re-enter before the other gets in.")


# === Exercise 3: Peterson's Algorithm Step-Through ===
# Problem: Trace Peterson's algorithm execution.

def exercise_3():
    """Step through Peterson's algorithm execution."""
    print("Peterson's Algorithm Trace:\n")
    print("```c")
    print("flag[i]=true; turn=1-i;")
    print("while(flag[1-i] && turn==1-i); // busy wait")
    print("// Critical Section")
    print("flag[i]=false;")
    print("```\n")

    steps = [
        ("P0: flag[0]=true", True, False, None, "P0 declares interest"),
        ("P1: flag[1]=true", True, True, None, "P1 declares interest"),
        ("P0: turn=1", True, True, 1, "P0 yields to P1"),
        ("P1: turn=0", True, True, 0, "P1 yields to P0 (OVERWRITES turn)"),
        ("P0: check while", True, True, 0,
         "flag[1]=T && turn=1? -> T && F -> FALSE. P0 ENTERS CS!"),
        ("P1: check while", True, True, 0,
         "flag[0]=T && turn=0? -> T && T -> TRUE. P1 SPINS."),
    ]

    print(f"{'Step':<5} {'Action':<25} {'flag[0]':<10} {'flag[1]':<10} {'turn':<8} {'Result'}")
    print("-" * 85)
    for i, (action, f0, f1, turn, result) in enumerate(steps, 1):
        turn_str = str(turn) if turn is not None else "?"
        print(f"{i:<5} {action:<25} {str(f0):<10} {str(f1):<10} {turn_str:<8} {result}")

    print("\nP0 enters the critical section first.")
    print("\nExplanation of correctness:")
    print("  The key is that 'turn' can only hold ONE value at a time.")
    print("  Both processes set turn to yield to the other:")
    print("    P0 sets turn=1 ('I'll let P1 go'), P1 sets turn=0 ('I'll let P0 go')")
    print("  The LAST writer wins: if P1 writes turn=0 last, then turn=0,")
    print("  and P0's while condition (turn==1) is false, so P0 proceeds.")
    print("  P1's while condition (turn==0) is true, so P1 waits.")
    print("  This guarantees exactly one process enters -- whichever set turn FIRST")
    print("  is the one that 'yielded' and must wait; the other proceeds.")


# === Exercise 4: TAS and CAS Semantics ===
# Problem: Implement spinlock with TAS and lock-free counter with CAS.

def exercise_4():
    """Implement TAS spinlock and CAS lock-free counter."""

    print("Part A: Spinlock using Test-and-Set\n")
    print("```pseudocode")
    print("bool lock = false;")
    print("")
    print("function TestAndSet(bool *target):")
    print("    bool old = *target")
    print("    *target = true")
    print("    return old       // all three steps are atomic")
    print("")
    print("function lock():")
    print("    while TestAndSet(&lock):  // spin until we get false (was unlocked)")
    print("        pass                  // busy wait")
    print("")
    print("function unlock():")
    print("    lock = false              // release the lock")
    print("```\n")

    print("Part B: Lock-free counter using Compare-and-Swap\n")
    print("```pseudocode")
    print("function CompareAndSwap(int *value, int expected, int new_val):")
    print("    int old = *value")
    print("    if old == expected:")
    print("        *value = new_val")
    print("    return old       // all steps are atomic")
    print("")
    print("function atomic_increment(int *counter):")
    print("    while true:")
    print("        int old = *counter")
    print("        if CompareAndSwap(counter, old, old+1) == old:")
    print("            return   // success: counter was 'old', now 'old+1'")
    print("        // else: another thread changed counter; retry with new value")
    print("```\n")

    print("Part C: Bounded waiting violation with TAS spinlock\n")
    print("Scenario with 3 processes (P0, P1, P2):")
    print("  P0 holds lock. P1 and P2 are spinning on TAS.")
    print("  P0 unlocks. P1 and P2 both call TAS simultaneously.")
    print("  P1 wins (gets false, enters CS). P2 continues spinning.")
    print("  P1 finishes, unlocks. P0 arrives and calls TAS.")
    print("  P0 wins. P2 still spinning. P0 finishes, unlocks.")
    print("  P1 wins again. P2 STILL spinning.")
    print("  Pattern repeats: P0 and P1 alternate, P2 starves.\n")

    print("Fix: Ticket lock (guarantees FIFO ordering)")
    print("```pseudocode")
    print("int next_ticket = 0")
    print("int now_serving = 0")
    print("")
    print("function lock():")
    print("    my_ticket = fetch_and_increment(&next_ticket)  // atomic")
    print("    while now_serving != my_ticket:                 // spin")
    print("        pass")
    print("")
    print("function unlock():")
    print("    now_serving++  // serve next ticket")
    print("```")
    print("  Each process gets a unique ticket number. Processes enter CS")
    print("  in ticket order, guaranteeing bounded waiting (max wait: N-1 turns).")


# === Exercise 5: Busy Waiting vs Blocking ===
# Problem: Compare spinlocks and mutexes for 100 threads.

def exercise_5():
    """Compare busy waiting (spinlock) vs blocking (mutex)."""
    threads = 100
    cs_time_ms = 2
    cores = 4

    print(f"System: {threads} threads, {cores} cores, CS takes {cs_time_ms}ms\n")

    print("Q1: CPU time wasted with spinlock (busy waiting):")
    spinning = threads - 1  # 99 waiting, 1 in CS
    print(f"  {spinning} threads spinning at any time, {cores} cores available")
    print(f"  {spinning} spinning threads consume min({spinning},{cores-1})={min(spinning,cores-1)} cores")
    print(f"  (1 core runs the CS holder, remaining {cores-1} cores run spinners)")
    print(f"  Wasted CPU: {min(spinning,cores-1)}/{cores} = {min(spinning,cores-1)/cores*100:.0f}% of total capacity")
    print(f"  In practice, all {cores} cores are at 100% utilization but only")
    print(f"  {1}/{cores} = {1/cores*100:.0f}% is doing useful work. 75% is pure waste.\n")

    print("Q2: With mutex (blocking):")
    print(f"  When a thread cannot acquire the mutex, it is put to SLEEP")
    print(f"  (moved to a wait queue, state: BLOCKED). It consumes NO CPU.")
    print(f"  Only 1 thread is running (in CS) plus a few threads in user code.")
    print(f"  The {cores-1} idle cores are truly idle -- the OS can schedule")
    print(f"  other processes or enter low-power states. No CPU waste.\n")

    print("Q3: When spinlocks are PREFERRED:")
    print("  When the critical section is extremely short (< ~1 microsecond)")
    print("  and the system has multiple cores. Examples:")
    print("  - Kernel interrupt handlers (cannot sleep while holding a lock)")
    print("  - Per-CPU data structures (minimal contention)")
    print("  - Lock held for < context switch time (~5us)")
    print("  Rationale: sleeping and waking a thread (context switch) costs")
    print("  ~5-10us. If the CS takes 100ns, spinning wastes 100ns but sleeping")
    print("  would waste 5000ns. Spinlock wins for ultra-short critical sections.\n")

    print("Q4: Linux's dual-mode mutex_spin_on_owner():")
    print("  Linux's adaptive mutex first SPINS while the lock owner is running")
    print("  on another core (optimistic: owner will release soon). If the owner")
    print("  is not running (e.g., was preempted), the waiter SLEEPS (pessimistic:")
    print("  could be a long wait). This combines both approaches:")
    print("  - Short wait (owner running): spin avoids context switch overhead")
    print("  - Long wait (owner sleeping): block avoids wasting CPU")
    print("  Best of both worlds for the common case in a multi-core system.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Race Condition Identification ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Critical Section Requirements Analysis ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Peterson's Algorithm Step-Through ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: TAS and CAS Semantics ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Busy Waiting vs Blocking ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
